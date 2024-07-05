# %%
import os
import json
import pickle

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI, AzureOpenAI
import argparse
import pandas as pd
import time


def load_pickle(filename):
    if filename is None or not os.path.exists(filename):
        return None
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    if filename is None:
        return
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


API_KEY = ""
API_VERSION = ""
AZURE_ENDPOINT = ""


## prompt for sentence to <question, answer> pair.
query_ans_prompt = """下面是来自{}书籍的一段文本。\
你需要基于该段文本生成一个与{}相关的问题及该问题对应的答案，输出json格式，包含<问题>和<答案>两个key。 \
注意，生成的问题确切且要与文本紧密相连但不能明确引用文本；生成的答案尽可能信息丰富且需要参考提供的文本但不能明显地表露对文本的直接依赖。
注意，只需要生成一个问题和答案。
<文本>：{}"""


## prompt for <问题, 答案> to <问题, 答案, 解释> pair.
single_choice_prompt = """下面是一道中国医学考试单项选择题及答案，只有一个选项正确，\
请分析每个选项然后得出所给答案。 
<问题>: {}
<答案>: {}
<解释>: """
multi_choice_prompt = """下面是一道中国医学考试多项选择题及答案，有多个选项正确，\
请分析每个选项然后得出所给答案。 
<问题>: {}
<答案>: {}
<解释>: """
fill_prompt = """下面是一道中国医学考试填空题及答案，\
请进行一步步分析然后得出所给答案。 
<问题>: {}
<答案>: {}
<解释>: """
qa_prompt = """下面是一道中国医学考试问答题及答案，\
请对问题进行一步步分析然后得出所给答案。 
<问题>: {}
<答案>: {}
<解释>: """
prompt_dict = {"单选题": single_choice_prompt,
               "多选题": multi_choice_prompt,
               "填空题": fill_prompt,
               "问答题": qa_prompt}


class GPT:
    def __init__(self, model_name='') -> None:
        self.max_wrong_time = 2
        self.model_name = 'gpt-3.5-turbo-1106' if 'gpt-3.5-turbo-1106' in model_name else model_name
        self.init_client()
        print(f'use model of {self.model_name}')

    def init_client(self):
        # self.client = AzureOpenAI(
        #     api_key=API_KEY,
        #     # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
        #     api_version=API_VERSION,
        #     # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
        #     azure_endpoint=AZURE_ENDPOINT,
        #     max_retries=self.max_wrong_time,
        # )
        self.client = OpenAI(
            api_key='sk-E9oyiDL777ZaNZdRrzRSPzsbvbqvhebRl2xiTheKjh6bE4Jx' if self.model_name == 'gpt-3.5-turbo-1106' else 'EMPTY',
            max_retries=self.max_wrong_time,
            base_url='https://openkey.cloud/v1' if self.model_name == 'gpt-3.5-turbo-1106' else 'http://127.0.0.1:8000/v1'
        )

    def call(self, content):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            temperature=0.3,
            top_p=0.2,
            max_tokens=512,
            model=self.model_name,
        )
        response = chat_completion.choices[0].message.content
        return response

    def test(self):
        try:
            print(self.call('你好'))
        except Exception as e:
            print(e)


# def read_data(data_path):
#     # 用于存储所有数据的列表
#     tmpdata = []
#     # 遍历目录中的所有文件
#     for filename in os.listdir(data_path):
#         if filename.endswith('.jsonl'):
#             file_path = os.path.join(args.data_path, filename)
#             sub_data = []
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 for i, line in enumerate(file):
#                     try:
#                         item = json.loads(line)
#                         sub_data.append(item)
#                     except json.JSONDecodeError:
#                         print(f"Error decoding line {i + 1} in {file_path}. Skipping.")
#             print(f"路径{file_path}下一共{len(sub_data)}条数据")
#             tmpdata.extend(sub_data)
#
#     for ii, text in enumerate(tmpdata):
#         text['id'] = ii
#     data = tmpdata
#
#     return data


# def read_data_csv(data_path):
#     task_explain_path = os.path.join(data_path, 'train_explain.csv')
#     task_predict_path = os.path.join(data_path, 'test.csv')
#     print(data_path)
#     df_predict = pd.read_csv(task_predict_path)
#     df_explain = pd.read_csv(task_explain_path)
#     data = []
#     for index, row in df_predict.iterrows():
#         item = {}
#         item['question'] = row['question']
#         item['label'] = row['response_j']
#         item['task'] = 'predict'
#         data.append(item)
#     for index, row in df_explain.iterrows():
#         item = {}
#         item['question'] = row['question']
#         item['label'] = row['response_j']
#         item['task'] = 'predict'
#         data.append(item)
#
#     print(f'读取到{len(data)}条数据')
#     return data


wrongtime = 0

#
# def write_books_qa(d, gpt, writer, args):
#     global wrongtime
#     try:
#         if 'reply' not in d:
#             for ii in range(args.try_num):
#                 chatgpt_query = query_ans_prompt.format(d['label'], d['label'], d['text'][:args.max_input_length])
#                 output = gpt.call(chatgpt_query)
#                 if output == None:
#                     continue
#                 d['reply'] = output
#                 break
#
#         assert 'reply' in d, 'no reply'
#
#         writer.write(json.dumps(d, ensure_ascii=False) + "\n")
#         wrongtime = 0
#
#     except Exception as e:
#         print(str(e), flush=True)
#         wrongtime += 1
#         if wrongtime > 10:
#             assert 1 == 0, 'wrong'
#
#     return 1
#
#
# def write_qa_qae(d, gpt, writer, args):
#     global wrongtime
#     try:
#         if 'explanation' not in d:
#             for ii in range(args.try_num):
#                 prompt = prompt_dict[d['label']]
#                 chatgpt_query = prompt.format(d['instruction'], d['output'])
#                 explanation = gpt.call(chatgpt_query)
#                 if explanation == "" or len(explanation) >= 512:
#                     continue
#                 d['explanation'] = explanation
#                 break
#
#         assert 'explanation' in d, 'no explanation'
#
#         writer.write(json.dumps(d, ensure_ascii=False) + "\n")
#         wrongtime = 0
#
#     except Exception as e:
#         # print(d)
#         print(str(e), flush=True)
#         wrongtime += 1
#         if wrongtime > 10:
#             assert 1 == 0, 'wrong'
#
#     return 1


def write_qa(d, gpt, writer, args):
    global wrongtime
    try:
        if f'{args.model_name}_output' not in d:
            for ii in range(args.try_num):
                chatgpt_query = d['raw_input_text']
                output = gpt.call(chatgpt_query)

                if output is None:
                    continue
                d[f'{args.model_name}_output'] = output
                break

        assert f'{args.model_name}_output' in d, f'no {args.model_name}_output'

        writer.write(json.dumps(d, ensure_ascii=False) + "\n")
        wrongtime = 0

    except Exception as e:
        print(str(e), flush=True)
        wrongtime += 1
        if wrongtime > 10:
            assert 1 == 0, 'wrong'

    return 1


def main(args):
    data = load_pickle(args.data_path)[args.query_start_idx: args.query_end_idx]
    args.query_end_idx = args.query_start_idx+len(data)-1
    print(f"read data: {args.query_start_idx}-{args.query_start_idx+len(data)-1}")
    gpt = GPT(model_name=args.model_name)
    task_name = f'rewrite_{args.dataset_name}'
    print(task_name)
    args.save_path = f'{args.data_path}_{args.model_name}_Result{args.query_start_idx}-{args.query_start_idx+len(data)-1}.jsonl'
    writer = open(args.save_path, mode="w", encoding="utf-8")
    # if task_name == "rewrite_books":
    #     with ThreadPoolExecutor(max_workers=args.num_process) as executor:
    #         results = list(tqdm(executor.map(lambda x: write_books_qa(x, gpt, writer, args), data), total=len(data),
    #                             desc="Processing samples", unit="sample"))
    # if task_name == "rewrite_exercise":
    #     with ThreadPoolExecutor(max_workers=args.num_process) as executor:
    #         results = list(tqdm(executor.map(lambda x: write_qa_qae(x, gpt, writer, args), data), total=len(data),
    #                             desc="Processing samples", unit="sample"))
    if task_name == "rewrite_steam" or task_name == "rewrite_sub_movie":
        with ThreadPoolExecutor(max_workers=args.num_process) as executor:
            results = list(tqdm(executor.map(lambda x: write_qa(x, gpt, writer, args), data), total=len(data),
                                desc="Processing samples", unit="sample"))
    writer.close()
    print(f'finish_')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_path", type=str, default='data/dataset/sub_movie/GPT_datum_info_test_SFTTestSeqRec_LCT_Top10.pickle', help="processed_data path")
    parser.add_argument("--data_path", type=str, default='data/dataset/sub_movie/GPT_datum_info_test_SFTTestSeqRanking_LCT_Top5.pickle', help="processed_data path")
    parser.add_argument("--dataset_name", type=str, default="sub_movie", help="dataset name")
    parser.add_argument("--num_process", type=int, default=40)
    parser.add_argument("--model_name", type=str, default='gpt-3.5-turbo-1106', help="openai model")
    # parser.add_argument("--model_name", type=str, default='Llama-2-7b-hf-chat', help="openai model")
    parser.add_argument("--try_num", type=int, default=2, help="The number of attempts to call the API")
    parser.add_argument("--max_input_length", type=int, default=512, help="The max length of input text to gpt")
    parser.add_argument("--query_start_idx", type=int, default=5000)
    parser.add_argument("--query_end_idx", type=int, default=7000)
    args = parser.parse_args()
    main(args)
