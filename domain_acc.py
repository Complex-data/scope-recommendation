import os
import json
import pickle
import random

from Levenshtein import distance
from torch.utils.data import DataLoader
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from openai import OpenAI
import argparse
import pandas as pd
import time

from transformers import StoppingCriteriaList, MaxLengthCriteria

from SFT.SFT_dataloader import Test_task_group_mapping, SFTDataset,Test_Ctrl_task_group_mapping
from metrics import Metrics
from param import Config
from rewrite_by_gpt import GPT
from utils import *
from SFT.SFT_templates import *


random.seed(2024)
headers = {"User-Agent": "Test Client"}
domain_right = "steam"
domain_wrong = "toys"

def quary_vllm(input_text, args):
    for ii in range(args.try_num):
        pload_search = { # 确定性高
            "model": args.model_name,
            "prompt": input_text,
            "n": 1,
            "temperature": 0.0,
            "max_tokens": args.gen_max_length,
            "skip_special_tokens":False,
        }

        pload_sample = {  # 确定性低
            "model": args.model_name,
            "prompt": input_text,
            "n": 1,
            "temperature": 0.7,
            "max_tokens": args.gen_max_length,
            "top_p": 0.2,
            "top_k": 5,
            "skip_special_tokens": False,
        }
        response = requests.post(f'http://127.0.0.1:{args.port}/v1/completions', headers=headers, json=pload_sample if args.sample else pload_search, stream=False)
        output_data = json.loads(response.content)
        output_text = output_data["choices"][0]['text']
        # output_text = output_data["text"][0][len(input_text):]
        return output_text



def quary_api(d, args):
    global wrongtime
    try:
        if f'{args.model_name}_output' not in d:
            input_text = d['input_text']
            if args.model_name in ['snap/Llama-2-7b-hf-chat/', 'snap/gpt-3.5-turbo-1106/']:
                input_text = d['input_text'].split('\n')
                sub_text1 = input_text[1].strip()
                sub_text2 = input_text[4].split('[/INST]')[0].strip()
                sub_text3 = input_text[4].split('[/INST]')[1].split('[INST]')[1].strip()
                if args.SFT_test_task in ['SFTTestSeqRec', 'SFTTestItemCount']:
                    sub_text4 = 'Notice! Do not explain the reason or include any other words.'
                else:
                    sub_text4 = 'Notice! You need to output category information in the brackets after item titles in this template: 1. title (category). Do not explain the reason or include any other words in the front of titles.'
                input_text = f'{sub_text1} {sub_text2} {sub_text3} \n{sub_text4}'
                d['raw_input_text'] = input_text

            # if args.model_name in ['snap/Llama-2-7b-hf-chat/', 'snap/gpt-3.5-turbo-1106/']:
            # if args.model_name in ['snap/gpt-3.5-turbo-1106/']:
            #     d[f'{args.model_name}_output'] = quary_openai(input_text, args)
            # else:
            #     d[f'{args.model_name}_output'] = quary_vllm(input_text, args)

            d[f'{args.model_name}_output'] = quary_vllm(input_text, args)
        assert f'{args.model_name}_output' in d, f'no {args.model_name}_output'
        wrongtime = 0

    except Exception as e:
        print(str(e), flush=True)
        wrongtime += 1
        if wrongtime > 10:
            assert 1 == 0, 'wrong'

    return 1

def rm_idx(s):
    pattern_extract = r"\d+\.\s*(.*?)(?=\s*\d+\.|$)"
    s=s.replace("</s>","")
    matches = re.findall(pattern_extract, s)
    cut_size = min(len(matches),10)
    return matches[:cut_size]
    # return re.sub(r"\d+\.\s", '', s)

def query_server(input_data, args):
     output_text =quary_vllm(input_data[1],args)
     output_text = rm_idx(output_text)
     return (input_data[0], output_text)

def id2title(title_list, metas):
    title_list = list(map(lambda x:metas[x]['title'],title_list))
    return  ",".join(title_list)

def get_test_data(data):
    test_data_list = []
    # 读取 用户交互历史
    category = data['category']
    metas = data['metas']
    sequential = data['sequential']
    # 随机采样200个用户进行测试
    random_users = random.sample(list(sequential), 200)
    random_candidate = random.sample(list(sequential), 200)
    random_intention = random.sample(list(category), 200)

    random_prompt =f'You are an expert recommender engine. Please randomly generate a recommendation list with 10 different items. '
    random_prompt_candidate = 'You are an expert recommender engine. You need to randomly select a recommendation list from candidate items. The candidate items are: {}.Please generate a recommendation list with 10 different items'


    input_field_data = {}
    for index,user in enumerate(random_users):
        history = id2title(sequential[user],metas)
        candidate = id2title(sequential[random_candidate[index]], metas)
        input_field_data.update({
            'history': history,
            'target_category': [],
            'item_count': 10,
            'candidate_titles': candidate,
            'candidate_items': [],
            'synthetic_intention': random_intention[index]
        })
        # 构造prompt
        prompt1 = SeqRec_group['SeqRec-0'].get_input_text(input_field_data)
        test_data_list.append(prompt1)
        prompt2 = SeqRec_group['SeqRec-1'].get_input_text(input_field_data)
        test_data_list.append(prompt2)
        prompt3 = ControlRec_group['ControlRec-0'].get_input_text(input_field_data)
        test_data_list.append(prompt3)
        prompt4 = ControlRec_group['ControlRec-1'].get_input_text(input_field_data)
        test_data_list.append(prompt4)
        # prompt5 = random_prompt
        # test_data_list.append(prompt5)
        prompt6 = random_prompt_candidate.format(candidate)
        test_data_list.append(prompt6)

    return test_data_list

def get_domain_acc_data(data):
    test_data_list = []
    # 读取 用户交互历史
    category = data['category']
    metas = data['metas']
    sequential = data['sequential']
    # domain_right = "steam"
    # domain_wrong = "toys"
    # 随机采样个sample_user_num用户进行测试
    sample_user_num = 500
    # 种类的个数，后面可以进行取余
    intention_size = 200
    random_users = random.sample(list(sequential), sample_user_num)
    random_candidate = random.sample(list(sequential), sample_user_num)
    random_intention = random.sample(list(category), intention_size)

    input_field_data = {}
    for index, user in enumerate(random_users):
        history = id2title(sequential[user], metas)
        candidate = id2title(sequential[random_candidate[index]], metas)
        input_field_data.update({
            'history': history,
            'target_category': [],
            'item_count': 10,
            'candidate_titles': candidate,
            'candidate_items': [],
            'synthetic_intention': random_intention[index%intention_size],
            'domain': domain_right,
            'domain_right' : domain_right,
            'domain_wrong' : domain_wrong

        })
        # 构造prompt
        prompt1 = SeqRec_group['SeqRec-0'].get_input_text(input_field_data)
        test_data_list.append((domain_right, prompt1))
        prompt2 =SeqRec_domain_group['SeqRec-domain-0'].get_input_text(input_field_data)
        test_data_list.append((domain_wrong, prompt2))



    return test_data_list


if __name__ == "__main__":
    def vague_mapping(ts):
        for idx, __ in enumerate(ts):
            if __ in test_data.title2item:
                continue
            for ___ in test_data.title2item:
                if distance(__, ___) <= 3:
                    ts[idx] = ___
                    break

    def process_api_output(d):
        if f'{args.model_name}_output' not in d:
            return d
        if d[f'{args.model_name}_output'] == "":
            d[f'{args.SFT_test_task}_output_title_list'] = []
            return d
        if f'{args.SFT_test_task}_output_title_list' in d:
            return d
        raw_output = d[f'{args.model_name}_output']
        if raw_output[0] == raw_output[-1] == '"' or raw_output[0] == raw_output[-1] == "'":
            raw_output = raw_output[1:-1]

        ts = raw_output.split('\n')
        ts = [rm_idx(_).strip().split('\n')[0].strip() for _ in ts if match_idx(_)]
        if args.SFT_test_task != 'SFTTestSeqRec':
            ts = [re.sub(r' *[(,\[](.*)[),\]]$', '', _) for _ in ts]

        ts = [__[1:-1] if __[0] == __[-1] == "'" or __[0] == __[-1] == "\"" else __ for __ in ts if __ != '']
        ts = [__.strip() for __ in ts]
        ts = ts[:d['input_field_data']['item_count']]

        vague_mapping(ts)
        d[f'{args.SFT_test_task}_output_title_list'] = ts

        return d

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='data/dataset/steam/', help="processed_data path")
    parser.add_argument('--SFT_test_task', type=str, default='SFTTestSeqRec,SFT+TestPersonalControlRec', help='in {SFTTestSeqRec, SFTTestRanking, SFT+TestPersonalControlRec, SFT-TestPersonalControlRec, SFTTestPersonalCategoryRate_xx%, SFTTestItemCount}')
    parser.add_argument("--num_process", type=int, default=40)
    parser.add_argument("--model_name", type=str, default='Llama-2-7b-hf-chat', help="openai model")
    parser.add_argument("--try_num", type=int, default=2, help="The number of attempts to call the API")
    parser.add_argument("--max_item_length", type=int, default=10)
    parser.add_argument("--max_token_length", type=int, default=512, help="The max length of input text to gpt")
    parser.add_argument("--gen_max_length", type=int, default=512)
    parser.add_argument("--candidate_num", type=int, default=10)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--item_index", type=str, default='title')
    parser.add_argument("--backup_ip", type=str, default='0.0.0.0')
    parser.add_argument("--llama2_chat_template", action='store_true')
    parser.add_argument("--idx", action='store_true')
    parser.add_argument("--sample", action='store_true')
    parser.add_argument("--user_control_symbol", action='store_true')
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    args.is_main_process = True
    kwargs = vars(args)
    args = Config(**kwargs)
    print(args)
    # gpt = GPT(model_name=args.model_name)
    data = {
        'category': load_pickle(args.data_path + 'category.pickle'),
        'metas': load_pickle(args.data_path + 'metas.pickle'),
        'sequential': load_pickle(args.data_path + 'sequential.pickle'),
        'preference': load_pickle(args.data_path + 'preference.pickle'),
        'intention': load_pickle(args.data_path + 'intention.pickle'),
        'share_chat_gpt': None,
        'ranking_candidate': load_pickle(args.data_path + 'ranking_candidate.pickle'),
    }
    # test_data_list = get_test_data(data)
    test_data_list = get_domain_acc_data(data)
    with ThreadPoolExecutor(max_workers=args.num_process) as executor:
        results = list(tqdm(executor.map(lambda d: query_server(d, args), test_data_list), total=len(test_data_list)))

    save_pickle(results, "/home/wangshuo/codes/scope-rec/vllm/ctrl-test/domain_100_predict.pickle")
    domain_right_count=0
    domain_wrong_count=0
    for domain, predict_list in results:
        if domain == domain_right:
            if len(predict_list) > 0:
                domain_right_count += 1
        elif domain == domain_wrong:
            if len(predict_list) == 0:
                domain_wrong_count += 1
    print(f"domain right count:{domain_right_count} domain wrong count:{domain_wrong_count}")

    # item_num = 0
    # ctrl_num = 0
    # def extraction_ctrl(text):
    #     pattern_extract = r"<SOI>(.*?)<EOI>"
    #     matches = re.findall(pattern_extract, text)
    #     return matches[0] if len(matches)!=0 else ""
    #     pass
    # save_path = '/home/wangshuo/codes/scope-rec/vllm/ctrl-test/ctrl-acc.pickle'
    # save_pickle(results,save_path)
    # for _, item_list in results:
    #     cut_size = min(len(item_list), 10)
    #     # if cut_size<10:
    #     #     print(item_list)
    #     item_list_ten = item_list[:cut_size]
    #     extraction_list = list(map(lambda x: extraction_ctrl(x),item_list_ten))
    #     item_num += len(item_list_ten)
    #     ctrl_num += len(list(filter(lambda s: s.strip() != '', extraction_list)))

    # print(f"item sum:{item_num}\n ctrl num:{ctrl_num}")





