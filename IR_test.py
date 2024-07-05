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

from SFT.SFT_dataloader import Test_task_group_mapping, SFTDataset
from metrics import Metrics
from param import Config
from rewrite_by_gpt import GPT
from utils import *


IR_task_template = {
    "SFTTestSeqRanking": "Using the user’s historical interactions as input data, predict the next product from the "
                         "following candidates that best matches their preference and query: {candidate_titles}. "
                         "The historical interactions are provided as follows: {history}.",

    "SFTTestSeqRec": "Using the user’s historical interactions as input data, predict the next product that the user is "
                     "most likely to interact with. The historical interactions are provided as follows: {history}.",

    "SFT+TestPersonalControlRec": "As a search engine, you are assisting a user who is searching for the query: "
                                  "{synthetic_intention}. Your task is to recommend products that match the user’s query "
                                  "and also align with their preferences based on their historical interactions, "
                                  "which are reflected in the following: {history}",

    "SFT-TestPersonalControlRec": "As a search engine, you are assisting a user who is searching for the query: "
                                  "{synthetic_intention}. Your task is to recommend products that match the user’s query "
                                  "and also align with their preferences based on their historical interactions, "
                                  "which are reflected in the following: {history}",

    "SFTTestPersonalCategoryRateLP": "As a search engine, you are assisting with user's intention: I like '{target_category}' items, "
                                     "but in the recommendation list, the proportion of '{target_category}' items "
                                     "should be less than {category_proportion}. Your task is to recommend products that "
                                     "match the user’s query and also align with their preferences based on their "
                                     "historical interactions, which are reflected in the following: {history}",

    "SFTTestPersonalCategoryRateMP": "As a search engine, you are assisting with user's intention: In the recommendation list, "
                                     "the proportion of '{target_category}' items should be more than {category_proportion}. "
                                     "Your task is to recommend products that match the user’s query and also align with "
                                     "their preferences based on their historical interactions, "
                                     "which are reflected in the following: {history}"
}


headers = {"User-Agent": "Test Client"}


def quary_IR(input_text, candidate_titles=None):
    for ii in range(args.try_num):
        pload_search = {
            "input_text": input_text,
            "topk": args.topk,
            "max_token_length": args.max_token_length,
            "gen_max_length": args.gen_max_length,
            "model_name": args.model_name
        }
        if candidate_titles is not None:
            pload_search.update({
                'candidate_titles': candidate_titles
            })
        response = requests.post(f'http://127.0.0.1:{args.port}/generate', headers=headers, json=pload_search, stream=False)
        output_data = json.loads(response.content)
        if 'result' not in output_data:
            continue
        output_text = output_data['result']
        return output_text


if __name__ == "__main__":
    def vague_mapping(ts):
        for idx, _ in enumerate(ts):
            if _ in test_data.title2item:
                continue
            for __ in test_data.title2item:
                if distance(_, __) <= 3:
                    ts[idx] = __
                    break

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='data/dataset/sub_movie/', help="processed_data path")
    parser.add_argument('--SFT_test_task', type=str, default='', help='in {SFTTestSeqRec, SFTTestRanking, SFT+TestPersonalControlRec, SFT-TestPersonalControlRec, SFTTestPersonalCategoryRate_xx%, SFTTestItemCount}')
    parser.add_argument("--model_name", type=str, default='Llama-2-7b-hf-chat', help="openai model")
    parser.add_argument("--try_num", type=int, default=2, help="The number of attempts to call the API")
    parser.add_argument("--max_item_length", type=int, default=10)
    parser.add_argument("--max_token_length", type=int, default=512, help="The max length of input text to gpt")
    parser.add_argument("--gen_max_length", type=int, default=64)
    parser.add_argument("--candidate_num", type=int, default=10)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--item_index", type=str, default='title64_t')
    parser.add_argument("--backup_ip", type=str, default='0.0.0.0')
    parser.add_argument("--batch_size", type=int, default=14)
    parser.add_argument("--llama2_chat_template", action='store_true')
    parser.add_argument("--idx", action='store_true')
    parser.add_argument("--candidate_infer", action='store_true')
    parser.add_argument("--port", type=int, default=24680)
    args = parser.parse_args()
    args.is_main_process = True
    kwargs = vars(args)
    args = Config(**kwargs)
    print(args)
    data = {
        'category': load_pickle(args.data_path + 'category1.pickle'),
        'metas': load_pickle(args.data_path + 'meta1.pickle'),
        'sequential': load_pickle(args.data_path + 'sequential.pickle'),
        'preference': load_pickle(args.data_path + 'preference.pickle'),
        'intention': load_pickle(args.data_path + 'intention.pickle'),
        'share_chat_gpt': None,
        'ranking_candidate': load_pickle(args.data_path + 'ranking_candidate.pickle'),
    }
    TestTaskTemplate = {args.SFT_test_task: Test_task_group_mapping[args.SFT_test_task.split('_')[0]]}
    TestTaskNum = {args.SFT_test_task: 1}
    args.output_path = args.model_name+'/'
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    if args.SFT_test_task in ['SFT+TestPersonalControlRec', 'SFT-TestPersonalControlRec'] or args.SFT_test_task.startswith('SFTTestPersonalCategoryRate'):
        TestSeqRec_Result_file = f'{args.output_path}SFTTestSeqRec_Top{args.topk}_Result.pickle'
        data['SFTTestSeqRec_Result'] = load_pickle(TestSeqRec_Result_file)
    test_data = SFTDataset(args, TestTaskTemplate, TestTaskNum, data, None, 'test')
    metrics_dict = Metrics([args.SFT_test_task], args.topk, test_data.category2item, test_data.title2item)
    result_file = f'{args.output_path}{args.SFT_test_task}_Top{args.topk}_Result{"_CI" if args.candidate_infer else ""}.pickle'

    test_data_list = load_pickle(result_file) or [_ for _ in test_data]
    # origin_test_data = [_ for _ in test_data]
    for _ in test_data_list:
        _['input_text'] = IR_task_template[args.SFT_test_task.split('_')[0]].format_map(_['input_field_data'])
    remain_test_data_list = [_ for _ in test_data_list if f'{args.model_name}_output' not in _]
    print(f"Loading Test Task: '{args.SFT_test_task}'. Remain Example Count: {len(remain_test_data_list)}")
    print(test_data_list[1]['input_text'])

    for i in tqdm(range(0, len(remain_test_data_list), args.batch_size)):
        input_text = [_['input_text'] for _ in remain_test_data_list[i:i+args.batch_size]]
        candidate_titles = None
        if args.candidate_infer and 'candidate_items' in remain_test_data_list[0]['input_field_data']:
            candidate_titles = [
                [test_data.get_item_index(__) for __ in _['input_field_data']['candidate_items']]
                for _ in remain_test_data_list[i:i+args.batch_size]
            ]
        output_text: list[str] = quary_IR(input_text, candidate_titles)
        for _, __ in zip(remain_test_data_list[i:i+args.batch_size], output_text):
            _[f'{args.model_name}_output'] = __
            output_title = _[f'{args.model_name}_output']
            output_title_list = [_.strip() for _ in output_title.strip().split('\n')]
            output_title_list = [rm_idx(_) if args.idx else _ for _ in output_title_list]
            vague_mapping(output_title_list)
            _[f'{args.SFT_test_task}_output_title_list'] = output_title_list
            output_label = [_.strip() for _ in _['output_text'].strip().split('\n')]
            output_label = [rm_idx(_) if args.idx else _ for _ in output_label]
            metrics_dict.add_sample(_['task'], _['input_field_data'], output_title_list, output_label, vague_mapping=False)
        metrics_dict.print()

    for _ in test_data_list:
        output_label = [_.strip() for _ in _['output_text'].strip().split('\n')]
        output_label = [rm_idx(_) if args.idx else _ for _ in output_label]
        # if 'SeqRec_Result' in _['input_field_data']:
        #     _['input_field_data']['SeqRec_Result'] = _['input_field_data']['SeqRec_Result'][:5]
        # metrics_dict.add_sample(_['task'], _['input_field_data'], _[f'{args.SFT_test_task}_output_title_list'][:5], output_label, vague_mapping=False)
        metrics_dict.add_sample(_['task'], _['input_field_data'], _[f'{args.SFT_test_task}_output_title_list'], output_label, vague_mapping=False)
    metrics_dict.print()
    if len(remain_test_data_list) > 0:
        save_pickle(test_data_list, result_file)


