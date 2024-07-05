import pandas as pd
import math
from tqdm import tqdm
import numpy as np


def calculate_ndcg(label, predict_list, k):
    """
    计算NDCG@K值。

    参数:
    label (str): 标签项目。
    predict_list (list): 推荐列表。
    k (int): 计算NDCG的位置。

    返回:
    float: NDCG@K值。
    """
    # 检查标签项目是否在推荐列表中
    if label in predict_list:
        position = predict_list.index(label) + 1  # 获取标签项目的位置
        # 计算DCG@K
        dcg_at_k = 1 / math.log2(position + 1) if position <= k else 0

        # 计算IDCG@K (在这种情况下，只有一个相关项目，它应该在理想列表的第一位)
        idcg_at_k = 1 / math.log2(1 + 1)

        # 计算NDCG@K
        ndcg_at_k = dcg_at_k / idcg_at_k
        return ndcg_at_k
    else:
        # 如果标签项目不在推荐列表中，NDCG@K为0
        return 0


def calculate_hit_rate(label, predict_list, k):
    """
    计算单个预测记录的命中率。

    参数:
    label (str): 标签项目。
    predict_list (list): 推荐列表。

    返回:
    int: 如果标签项目出现在推荐列表中，则为1，否则为0。
    """
    return 1 if label in predict_list[:k] else 0


def calculate_mrr_single_query(label, predict, k):
    if label in predict[:k]:
        rank = predict.index(label) + 1
        return 1.0 / rank
    return 0


def calculate_repeat_rate(predict , k):
    counts = {}
    array = predict[:k]
    for item in array:
        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 1
    dup_count = 0
    for count in counts.values():
        if count > 1:
            dup_count += count-1

    return dup_count


def out_count(predict, item_list, k):
    count = 0
    for item in predict[:k]:
        item = item.strip()
        if item not in item_list:
            # print(f'erro item:{item}')
            count += 1
    return count

# =================================== 过滤掉pop的item ===================================

def get_pop_by_percent(item_count_dict, test_list=None, percent=0.1):
    sorted_count = sorted(item_count_dict.items(), key=lambda x: x[1])
    frequent_len = int(len(sorted_count) * percent)
    array = sorted_count[-frequent_len:-1]
    print(f"\t临界的次数是 {array[0][1]}")
    frequent_list = list(map(lambda x: x[0], array))
    if test_list != None:
        cal_test_pop(frequent_list, test_list)
    return frequent_list

def get_pop_by_average(item_count_dict,test_list=None):
    counts = np.array(list(item_count_dict.values()))
    mean_count = np.mean(counts)
    sorted_count = sorted(item_count_dict.items(), key=lambda x: x[1])
    frequent_list = []
    for item in sorted_count:
        if item[1] > mean_count:
            frequent_list.append(item[0])
    print(f"\t临界的次数是 {mean_count}")
    if test_list != None:
        cal_test_pop(frequent_list, test_list)
    return frequent_list

def cal_test_pop(frequent_list, test_list):
    frequent_count = 0
    for item in test_list:
        if item in frequent_list:
            frequent_count += 1
    print(f"\t测试集预测的物品为流行物品的个数为：{frequent_count}")

def filter_popular_item(predict_data, items_count_dict, pop_size=0.05, filter_count = 2000, mode='average'):
    if mode == 'average':
        frequent_list = get_pop_by_average(items_count_dict)
    else:
        frequent_list = get_pop_by_percent(items_count_dict, percent=pop_size)
    filter_predict_data = []
    count = 0
    freq_count = 0
    for predict_item in predict_data:
        if predict_item[0][0] in frequent_list:
            freq_count +=1
        if count < filter_count and predict_item[0][0] in frequent_list:
            count += 1
            continue
        filter_predict_data.append(predict_item)

    print(f"frequent count is {freq_count}")
    print(f"过滤后的预测用户数为 {len(filter_predict_data)}")
    return filter_predict_data

item_path_dict = {
    'steam': 'path/to/data/dataset/steam/items.pickle',
    'movies': 'path/to/data/dataset/movies/items.pickle',
    'toys': 'path/to/data/dataset/toys/items.pickle',
}
item_dict_path_dict = {
    'steam': 'path/to/data/dataset/steam/items_count_dict.pickle',
    'movies': 'path/to/data/dataset/movies/items_count_dict.pickle',
    'toys': 'path/to/data/dataset/toys/items_count_dict.pickle',
}

# read  data
dataset_name = 'steam'
# dataset_name = 'steam'
# dataset_name = 'toys'

predict_file_path = 'path/to/Epoch40_SFT_Result_SFTTestSeqRec_Top10.pickle'
# predict_file_path ='path/to/Epoch40_SFT_Result_SFTTestSeqRec_CBS_Top10.pickle'
# predict_file_path = 'path/to/distinguish-acc-finetune.pickle'
# predict_file_path = 'path/to/distinguish-acc.pickle'
# predict_file_path = 'path/to/distinguish-acc-finetune-1.pickle'


item_path = item_path_dict[dataset_name]
items_count_path = item_dict_path_dict[dataset_name]
predict_data = pd.read_pickle(predict_file_path)
item_data = pd.read_pickle(item_path)
item_data = set(item_data)
# items_count_dict = pd.read_pickle(items_count_path)
print(f"训练中的item个数为：{len(item_data)}")

# 去掉一些流行的物品的预测结果
print(f'{len(predict_data)}个用户')
filter_count = 2000
# if dataset_name == 'movies':
#     filter_count = 1000

# predict_data = filter_popular_item(predict_data, items_count_dict, filter_count=filter_count, mode='pop')

# # 重新提取一下对应的item
# for i in range(len(predict_data)):
#     predict = predict_data[i][1]
#     if len(predict) == 1:
#         predict = predict[0].split(' ')
#         predict = list(map(lambda x: x.split('.')[0].strip() if '.' in x else x.strip(), predict))
#         predict_data[i][1] = predict

# cal metric
topk_list = [1,5, 10]

predict_list = []
repeat_rate_dict = {}
NDCG_dict = {}
HR_dict = {}
MRR_dict = {}
out_count_dict = {}
sum_count = 0
for label, predict in tqdm(predict_data, total=(len(predict_data))):
    sum_count += len(predict[:10])
    label = label[0]
    for k in topk_list:
        if k not in NDCG_dict.keys():
            NDCG_dict[k] = []
            HR_dict[k] = []
            MRR_dict[k] = []
            out_count_dict[k] = []
            repeat_rate_dict[k] = []

        num = min(k, len(predict))
        NDCG_dict[k].append(calculate_ndcg(label, predict[:num], k))
        repeat_rate_dict[k].append((calculate_repeat_rate(predict[:num], k)))
        HR_dict[k].append(calculate_hit_rate(label, predict[:num], k))
        MRR_dict[k].append(calculate_mrr_single_query(label, predict[:num], k))
        out_count_dict[k].append(out_count(predict, item_data, k))


print(f'预测了{sum_count}个item')
for k in topk_list:
    NDCG = np.mean(NDCG_dict[k])
    HR = np.mean(HR_dict[k])
    MRR = np.mean(MRR_dict[k])
    out_sum = np.sum(out_count_dict[k])
    repeat_rate = np.sum(repeat_rate_dict[k])
    print(f'NDCG@{k}: {NDCG}')
    print(f'HR@{k}: {HR}')
    print(f'MRR@{k}: {MRR}')
    print(f'out_sum@{k}: {out_sum}')
    print(f'repeat_rate@{k}: {repeat_rate}')
