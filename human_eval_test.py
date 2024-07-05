import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


import os
import numpy as np
import pandas as pd
import time

import requests
from fastchat.model import get_conversation_template
from tqdm import tqdm

from param import Config, get_args

# import human_eval
from human_eval.data import write_jsonl, read_problems
from datetime import datetime
import re



choices = ["A", "B", "C", "D"]


def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


# def get_llama_prompt(train_df, subject, train_num, test_df, i):
#     conv = get_conversation_template("llama-2")
#     conv.set_system_message(
#         "You are a helpful, respectful and honest assistant.")
#     conv.append_message(
#         conv.roles[0], f"The following are multiple choice questions (with answers) about {format_subject(subject)}. You need to select the correct answer.\n\n")
#     conv.append_message(conv.roles[1], "Ok.")
#     k = train_df.shape[1] - 2
#     for idx in range(train_num):
#         conv.append_message(conv.roles[0], f"Question: {train_df.iloc[idx, 0]}\n" + "\n".join([f"{choices[j]}. {train_df.iloc[idx, j+1]}" for j in range(k)]) + '\nAnswer: ')
#         conv.append_message(conv.roles[1], f"{train_df.iloc[idx, k+1]}\n\n")
#     k = test_df.shape[1] - 2
#     conv.append_message(conv.roles[0], f"\nQuestion: {test_df.iloc[i, 0]}\n" + "\n".join([f"{choices[j]}. {test_df.iloc[i, j+1]}" for j in range(k)]) + '\nAnswer: ')
#     conv.append_message(conv.roles[1], None)
#     return conv.get_prompt()


headers = {"Content-Type": "application/json"}
error_count = 0
example_count = 0


def evaluate(subject):
    global error_count, example_count
    dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
    test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)
    cors = []

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        if args.llama2_chat_template:
            prompt = get_llama_prompt(dev_df, subject, k, test_df, i)
        else:
            prompt_end = format_example(test_df, i, include_answer=False)
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
        label = test_df.iloc[i, test_df.shape[1]-1]

        pload = {
            "model": args.backbone,
            "prompt": prompt,
            "max_tokens": 1,
            "logprobs": 100
        }
        # response = requests.post(f'http://127.0.0.1:{args.port}/generate', headers=headers, json=pload, stream=False)
        response = requests.post(f'http://127.0.0.1:{args.port}/v1/completions', headers=headers, json=pload, stream=False)
        output_data = json.loads(response.content)
        dist = output_data['choices'][0]['logprobs']['top_logprobs'][0]
        # dist = output_data['text'][0].split("[/INST]")[-1]
        pred = ''
        max_logprobs = float("-inf")
        for a in ['▁A', '▁B', '▁C', '▁D', 'A', 'B', 'C', 'D']:
            if a not in dist:
                continue
            # pred = a
            # break
            if dist[a] > max_logprobs:
                pred = a
                max_logprobs = dist[a]
        example_count += 1
        if pred == '':
            error_count += 1
            print(f'example: {example_count}, error: {error_count}')
        cor = label.lower() in pred.lower()
        cors.append(cor)

    acc = np.mean(cors)
    cors = np.array(cors)

    return cors, acc

def get_llama_prompt(prompt):
    conv = get_conversation_template("llama-2")
    conv.set_system_message(
        "You are a helpful, respectful and honest assistant.")
    conv.append_message(
        conv.roles[0], "Here is a segment of Python code. Please complete the code according to the requirements in the comments, and ensure that the final output is the completed code only.\n\n")
    conv.append_message(conv.roles[1], "Ok.")

    conv.append_message(conv.roles[0], f"\ncodes: {prompt}\n" + '\nAnswer: ')
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

def extraction_codes(text):
    patterns = [r"```python(.*?)```", r"```(.*?)```"]
    codes = ""
    for pattern in patterns:
        pattern = re.compile(pattern, re.DOTALL)
        match = pattern.findall(text)
        if len(match) == 0:
            continue
        else:
            codes = match[0].strip()
    if len(codes) == 0:
        print("提取失败！\n")
        print(text)
    return codes


def generate_one_completion(task_id, prompt, backbone):
    print(datetime.now().strftime("%H:%M:%S"), task_id)
    llama_prompt = get_llama_prompt(prompt)
    pload = {
        "model": backbone,
        "prompt": llama_prompt,
        "max_tokens": 1000,
        "temperature": 0.2
    }
    try:
        response = requests.post(f'http://127.0.0.1:{args.port}/v1/completions', headers=headers, json=pload, stream=False)
        output_data = json.loads(response.content)
        result = output_data["choices"][0]["text"]
        return extraction_codes(result)
    except:
        print(f"Exception occurs, wait 3 seconds then retry...")
        time.sleep(1)
        generate_one_completion(task_id, prompt)

def main(args):
    problems = read_problems()
    num_samples_per_task = 1
    # samples = [
    #     dict(task_id=task_id, completion=generate_one_completion(task_id, problems[task_id]["prompt"], args.backbone))
    #     for task_id in problems
    #     for _ in range(num_samples_per_task)
    # ]
    samples = []
    for task_id in tqdm(problems,total=len(problems.keys())):
        samples.append(dict(task_id=task_id, completion=generate_one_completion(task_id, problems[task_id]["prompt"], args.backbone)))
        # break

    save_path = os.path.join(args.save_dir, "samples1.jsonl")
    write_jsonl(save_path, samples)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--save_dir", "-s", type=str, default="path/to/vllm/general_task/human_eval")
    parser.add_argument("--subject_start", type=int, default=0)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--backbone", type=str, default="path/to/llama2/Llama-2-7b-hf-chat/")
    parser.add_argument('--llama2_chat_template', action='store_true', help='是否使用llama2-chat模板')
    args = parser.parse_args()
    kwargs = vars(args)
    args = Config(**kwargs)


    main(args)

