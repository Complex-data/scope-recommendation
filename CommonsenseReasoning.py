import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import openai
import os
import numpy as np
import pandas as pd
import time

import requests
import torch
from fastchat.model import get_conversation_template
from tqdm import tqdm

from actor_critic import ActorCritic
from param import Config, get_args

huggingface_proxies = {
    'http': '172.31.225.67:12621',
    'https': '172.31.225.67:12621',
    'ftp': '172.31.225.67:12621'
}

openai.api_key = "INSERTYOURKEYHERE"
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


def get_llama_prompt(train_df, subject, train_num, test_df, i):
    conv = get_conversation_template("llama-2")
    conv.set_system_message(
        "You are a helpful, respectful and honest assistant.")
    conv.append_message(
        conv.roles[0], f"The following are multiple choice questions (with answers) about {format_subject(subject)}. You need to select the correct answer.\n\n")
    conv.append_message(conv.roles[1], "Ok.")
    k = train_df.shape[1] - 2
    for idx in range(train_num):
        conv.append_message(conv.roles[0], f"Question: {train_df.iloc[idx, 0]}\n" + "\n".join([f"{choices[j]}. {train_df.iloc[idx, j+1]}" for j in range(k)]) + '\nAnswer: ')
        conv.append_message(conv.roles[1], f"{train_df.iloc[idx, k+1]}\n\n")
    k = test_df.shape[1] - 2
    conv.append_message(conv.roles[0], f"\nQuestion: {test_df.iloc[i, 0]}\n" + "\n".join([f"{choices[j]}. {test_df.iloc[i, j+1]}" for j in range(k)]) + '\nAnswer: ')
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


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

        # pload = {
        #     # "model": args.backbone,
        #     "prompt": prompt,
        #     "n": 1,
        #     'temperature': 0,
        #     'max_tokens': 1,
        # }
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


def main(args):
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])
    subjects = subjects[args.subject_start:]
    # path = args.backbone.replace('/', '_')
    # if not os.path.exists(args.save_dir):
    #     os.mkdir(args.save_dir)
    # if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(path))):
    #     os.mkdir(os.path.join(args.save_dir, "results_{}".format(path)))

    print(subjects)
    print(args)
    print(args.backbone)
    all_cors = []

    # bnb_config = BitsAndBytesConfig(
    #     llm_int8_threshold=6.0,
    #     llm_int8_has_fp16_weight=False,
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    # if args.backbone in ["davinci", "curie", "babbage", "ada"]:
    #     model = args.backbone
    # else:
    #     model = ActorCritic(args, args.gpu)
    #     model.load_parameters(args.SFT_load)
    #     model.eval()

    # else:
    #     model = AutoModelForCausalLM.from_pretrained(args.backbone,
    #                                                  quantization_config=bnb_config,
    #                                                  device_map=args.gpu,
    #                                                  torch_dtype=torch.bfloat16,
    #                                                  proxies=huggingface_proxies if args.proxy else None
    #                                                  )
    #
    #     tokenizer = AutoTokenizer.from_pretrained(args.backbone, proxies=huggingface_proxies if args.proxy else None)
    #     tokenizer.pad_token = tokenizer.unk_token
    #     tokenizer.pad_token_id = tokenizer.unk_token_id
    #     tokenizer.padding_side = 'left'
    #     model = (model, tokenizer)
    results = []
    with torch.no_grad():
        start_time = time.time()
        # pbar = tqdm(total=14042, ncols=50)                            # 23m50s
        # for subject in subjects:
        #     cors, acc = evaluate(subject)
        #     results.append((cors, acc))
        #     pbar.update(len(cors))
        with ProcessPoolExecutor(max_workers=1) as executor:         # 21m59s
            results = list(tqdm(executor.map(evaluate, subjects), total=len(subjects)))
        # with ThreadPoolExecutor(max_workers=57) as executor:          #
        #     results = list(tqdm(executor.map(evaluate, subjects), total=len(subjects)))
        end_time = time.time()
        print(f'time cost: {(end_time-start_time)}s')
        for cors, acc in results:
            print("{:.3f}".format(acc))
            all_cors.append(cors)
            # test_df["{}_correct".format(args.backbone)] = cors
            # for j in range(probs.shape[1]):
            #     choice = choices[j]
            #     test_df["{}_choice{}_probs".format(args.backbone, choice)] = probs[:, j]
            # test_df.to_csv(os.path.join(args.save_dir, "results_{}".format(path), "{}.csv".format(subject)), index=None)

    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="/home/wangshuo/codes/scope-rec/data/dataset/general_task/commonsense")
    parser.add_argument("--save_dir", "-s", type=str, default="/home/wangshuo/codes/scope-rec/vllm/Test/MMLU_results")
    parser.add_argument("--subject_start", type=int, default=0)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--backbone", type=str, default="MMLU.py")
    parser.add_argument('--llama2_chat_template', action='store_true', help='是否使用llama2-chat模板')
    args = parser.parse_args()
    # args = get_args(external_args_func)
    kwargs = vars(args)
    args = Config(**kwargs)
    # torch.manual_seed(args.seed)
    # random.seed(args.seed)
    # np.random.seed(args.seed)

    main(args)

# The following are multiple choice questions (with answers) about abstract algebra.
#
# Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.
# A. 0
# B. 1
# C. 2
# D. 3
# Answer: B
#
# Statement 1 | If aH is an element of a factor group, then |aH| divides |a|. Statement 2 | If H and K are subgroups of G then HK is a subgroup of G.
# A. True, True
# B. False, False
# C. True, False
# D. False, True
# Answer: B
#
# Statement 1 | Every element of a group generates a cyclic subgroup of the group. Statement 2 | The symmetric group S_10 has 10 elements.
# A. True, True
# B. False, False
# C. True, False
# D. False, True
# Answer: C
#
# Statement 1| Every function from a finite set onto itself must be one to one. Statement 2 | Every subgroup of an abelian group is abelian.
# A. True, True
# B. False, False
# C. True, False
# D. False, True
# Answer: A
#
# Find the characteristic of the ring 2Z.
# A. 0
# B. 3
# C. 12
# D. 30
# Answer: A
#
# Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.
# A. 0
# B. 4
# C. 2
# D. 6
# Answer: