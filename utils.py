import copy
import os.path
import pickle
import time
import xml.etree.ElementTree as ET
import numpy as np
import openai
import requests
import torch
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from tqdm import tqdm
import re
from prompt import *
import glob
import torch.nn as nn

huggingface_proxies = {
    'http': '',
    'https': '',
    'ftp': ''
}


def rm_idx(s):
    return re.sub(r'^(\d+)\. *', '', s, count=1)


def match_idx(s):
    return re.match(r'^(\d+)\. *', s)


def viz_append(viz, data_dict, step):
    for _, __ in data_dict.items():
        viz.line(
            [__], [step],
            win=_, update='append', opts={'title': _}
        )


def test_metrics(score, k):
    temp = np.array(score)
    sample_num, candidate_num = temp.shape
    ranking = np.argsort(-temp)[:, :k]
    position = np.array([range(1, k + 1)] * sample_num)
    return {
        f"Len": len(score),
        f"Rec@{k}": (ranking == 0).sum(axis=1).mean(),
        f"MRR@{k}": np.where(ranking == 0, 1 / position, 0.0).sum(axis=1).mean(),
        f"NDCG@{k}": np.where(ranking == 0, 1 / np.log2(position + np.ones_like(position)), 0.0).sum(axis=1).mean(),
    }


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


def side_tokenizer(text: list[str] or list[list[str]], padding_side, tokenizer, **kwargs):
    tokenizer.padding_side = padding_side
    tokenizer.truncation_side = padding_side
    tokenizer_res = tokenizer.batch_encode_plus(text, **kwargs)
    if padding_side == 'right': # 说明是处理output data
        input_ids = tokenizer_res['input_ids']
        for i,input_id in enumerate(input_ids):
            if input_id[-1] != 2 and input_id[-1] != -100: #既不是终止符又不是填充符
                input_ids[i][-1] = 2
        tokenizer_res['input_ids'] = input_ids
    return tokenizer_res


def get_item_list(ip, users, sub_sequential, k, candidate_item_list=None, target_category=None, port=2024):
    url = f"http://{ip}:{port}/inference"  # 替换为你要发送 POST 请求的 URL

    # 定义要发送的数据，通常以字典形式
    data = {
        "users": users,
        "item_lengths": [len(_) for _ in sub_sequential],
        "k": k,
        "item_lists": sub_sequential
    }
    if candidate_item_list is not None:
        data['candidate_item_lists'] = candidate_item_list
    if target_category is not None:
        data['target_category'] = target_category
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36 Edg/83.0.478.45",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    }
    # 发送 POST 请求
    response = requests.post(url, json=data, headers=headers)

    if response.status_code != 200:
        print(response)
    # 处理响应
    assert response.status_code == 200
    return response.json()


def get_item_ranking(ip, users, sub_sequential, candidate_item_list=None, port=12621):
    url = f"http://{ip}:{port}/ranking"  # 替换为你要发送 POST 请求的 URL

    # 定义要发送的数据，通常以字典形式
    data = {
        "users": users,
        "item_lengths": [len(_) for _ in sub_sequential],
        "item_lists": sub_sequential
    }
    if candidate_item_list is not None:
        data['candidate_item_lists'] = candidate_item_list
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36 Edg/83.0.478.45",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    }
    # 发送 POST 请求
    response = requests.post(url, json=data, headers=headers)

    # 处理响应
    assert response.status_code == 200
    return response.json()


def get_gpt_sorted_list(intention, history, candidate_item_list, fake=True):
    prompt = copy.deepcopy(ChatGPT_Ranking_item_COT_Prompt)
    information = ""
    if history is not None:
        information = information + f"#User's Historical Watching list#: \n {history}\n"
    if intention is not None:
        information = information + f"#User's Intention#: \n {intention}\n"
    information = information + f"#Candidate Movies#: \n {candidate_item_list}\n"
    prompt[-1]['content'] = prompt[-1]['content'].format_map({'information': information})

    if fake:
        output = generate_fake(prompt, 0.2, 0.3)
    else:
        output = generate_openai(prompt, 0.2, 0.3)
    print(output)


# helper functions
def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def masked_mean(seq, mask, dim=None):
    return (seq * mask).sum(dim=dim) / mask.sum(dim=dim)


def masked_var(values, mask, unbiased=True):
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def whiten(values, masks, shift_mean=True, dim=None):
    if shift_mean:
        mean, var = masked_mean(values, masks, dim=dim), masked_var(values, masks)
        whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    else:
        var = masked_var(values, masks)
        whitened = values * torch.rsqrt(var + 1e-8)
    return whitened


def pad_sequence_fixed(sequences, *args, **kwargs):
    first_el = sequences[0]
    has_no_dimension = first_el.ndim == 0

    # if no dimensions, add a single dimension
    if has_no_dimension:
        sequences = tuple(map(lambda t: t[None], sequences))

    out = pad_sequence(sequences, *args, **kwargs)

    if has_no_dimension:
        out = rearrange(out, '... 1 -> ...')

    return out


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def log_prob(prob, indices):
    assert prob.shape[
           :2] == indices.shape, f'preceding shapes of prob {prob.shape[:2]} and indices {indices.shape} must match'
    return log(prob.gather(-1, indices[..., None])).squeeze(-1)


def shift(t, value=0, shift=1, dim=-1):
    zeros = (0, 0) * (-dim - 1)
    return F.pad(t, (*zeros, shift, -shift), value=value)


def masked_entropy(prob, dim=-1, mask=None):
    entropies = (prob * log(prob)).sum(dim=-1)
    return masked_mean(entropies, mask=mask)


def masked_kl_div(prob1, prob2, action_mask=None, reduce_batch=False):
    """
    need to account for variable sequence lengths, therefore not using the built-in functional version
    """
    kl_divs = (prob1 * (log(prob1) - log(prob2))).sum(dim=-1)
    loss = masked_mean(kl_divs, action_mask)

    if reduce_batch:
        return loss.mean()

    return loss


def clipped_value_loss(values, rewards, old_values, clip):
    value_clipped = old_values + (values - old_values).clamp(-clip, clip)
    value_loss_1 = (value_clipped.flatten() - rewards) ** 2
    value_loss_2 = (values.flatten() - rewards) ** 2
    return torch.mean(torch.max(value_loss_1, value_loss_2))


def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out

    return inner


def get_history_text(output_titles: list[str]):
    history_text = ' → '.join(output_titles)
    return history_text


def get_output_text(output_titles: list[str], eos_token='', idx=False, user_control_symbol=False, use_scope_mask = False):
    if user_control_symbol and not use_scope_mask:
        output_titles = [f'<SOI>{t}<EOI>' for t in output_titles]
    if not idx:
        output_text = '\n '.join(output_titles)
    else:
        output_text = '\n '.join([f'{i+1}. {t}' for i, t in enumerate(output_titles)])
    return output_text + eos_token

def merge_result(pattern):
    files = glob.glob(pattern)
    print(f"检索到{len(files)}分块文件，分别是{files}")
    result_all = []
    for file in files:
        data =load_pickle(file)
        result_all.extend((data))
    print(f"合并的记录条数是{len(result_all)}")
    return result_all

def get_ctrl_item(out_titles):
    pattern = r"<SOI>\s*(.*?)\s*<EOI>"
    item_list = re.findall(pattern, out_titles, re.MULTILINE)
    return item_list

def get_prefix(input_ids, control_symbol):
    ctrl_s, ctrl_e = control_symbol
    # 找到所有ctrl_s和ctrl_e的索引
    
    ctrl_s_indices = (input_ids == ctrl_s).nonzero(as_tuple=True)[0]
    ctrl_e_indices = (input_ids == ctrl_e).nonzero(as_tuple=True)[0]

    # 检查ctrl_s和ctrl_e的数量
    # if len(ctrl_s_indices) <= 2:
    #     return False, []

    # 从后向前找到第一个ctrl_e和ctrl_s的位置
    last_ctrl_e_index = ctrl_e_indices[-1].item() if len(ctrl_e_indices) > 0 else -1
    last_ctrl_s_index = ctrl_s_indices[-1].item() if len(ctrl_s_indices) > 0 else -1

    # 检查索引位置并提取前缀
    if last_ctrl_s_index > last_ctrl_e_index:
        prefix = input_ids[last_ctrl_s_index:]
        return True, prefix.tolist()
    else:
        # return False, torch.tensor([], dtype=torch.int64).to(input_ids.device)
        return False, []


# <s>input_text item1\n item2\n item3\n</s>
def get_complete_text(input_text: str, output_titles: str):
    return input_text + ' ' + output_titles


detach_to_cpu_ = lambda t: rearrange(t.detach().cpu(), '1 ... -> ...')


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = torch.zeros(shape)
        self.S = torch.zeros(shape)
        self.std = torch.sqrt(self.S)

    def update(self, x):
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.clone()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = torch.sqrt(self.S / self.n)


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = torch.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = (x-self.running_ms.mean) / (self.running_ms.std + 1e-8)  # Only divided std
        return x


class RunningMoments:
    def __init__(self):
        """
        Calculates the running mean and standard deviation of a data stream. Reference:
        https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/utils.py#L75
        """
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24

    @torch.no_grad()
    def update(self, xs: torch.Tensor):
        """
        Updates running moments from batch's moments computed across ranks
        """
        xs_count = xs.numel()
        xs_var, xs_mean = torch.var_mean(xs, unbiased=False)
        xs_mean, xs_var = xs_mean.float(), xs_var.float()

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * xs_count / tot_count
        self.var = tot_sum / tot_count
        self.std = (self.var * tot_count / (tot_count - 1)).float().sqrt()
        self.count = tot_count

        return xs_mean.item(), (xs_var * xs_count / (xs_count - 1)).float().sqrt().item()


def GPT_eval(topk):
    start = 530
    seq_result = load_pickle('snap/ICR_SubMovie_title64_t_0_q_llama7b_CR/Epoch39_Result.pickle')
    gic_result = load_pickle('snap/ICR_SubMovie_title64_t_0_q_llama7b_CR/Epoch39_Result_Generate_Intention.pickle')
    intention = load_pickle('data/dataset/sub_movie/intention.pickle')
    meta = load_pickle('data/dataset/sub_movie/meta1.pickle')
    instruction = load_pickle('data/dataset/sub_movie/instruction.pickle')
    title = [meta[_]['title64_t'] for _ in meta]
    length = len(gic_result)

    save_file = f'snap/ICR_SubMovie_title64_t_0_q_llama7b_CR/Epoch39_Result_ChatGPT_eval_{start}.pickle'
    GPT_eval_result = load_pickle(save_file) or []
    metrics = {
        'SEQ_Recall@10': 0.0,
        'SEQ_NonExistRate@10': 0.0,
        'SEQ_RepeatRate@10': 0.0,
        'SEQ_CorrectCount@10': 0.0,

        'GIC_Recall@10': 0.0,
        'GIC_NonExistRate@10': 0.0,
        'GIC_RepeatRate@10': 0.0,
        'GIC_CorrectCount@10': 0.0,

        'GIC_Win': 0,
    }
    for idx in tqdm(range(start, length)):
        # assert seq_result[idx][0][0] == gic_result[idx][0][0]
        seq_o = seq_result[idx][1]
        gic_o = gic_result[idx][1]
        i = intention[idx]
        t = seq_result[idx][0][0]

        metrics[f'SEQ_Recall@{topk}'] += 1 if t in seq_o else 0
        metrics[f'GIC_Recall@{topk}'] += 1 if t in gic_o else 0

        metrics[f'SEQ_NonExistRate@{topk}'] += sum([1 if _ not in title else 0 for _ in seq_o]) / topk
        metrics[f'GIC_NonExistRate@{topk}'] += sum([1 if _ not in title else 0 for _ in gic_o]) / topk

        metrics[f'SEQ_RepeatRate@{topk}'] += sum([1 if _ in seq_o[:idx] else 0 for idx, _ in enumerate(seq_o)]) / topk
        metrics[f'GIC_RepeatRate@{topk}'] += sum([1 if _ in gic_o[:idx] else 0 for idx, _ in enumerate(gic_o)]) / topk

        metrics[f'SEQ_CorrectCount@{topk}'] += 1 if len(seq_o) == topk else 0
        metrics[f'GIC_CorrectCount@{topk}'] += 1 if len(gic_o) == topk else 0

        if idx-start < len(GPT_eval_result):
            metrics[f'GIC_Win'] += GPT_eval_result[idx-start]['gic_win']
            continue

        prompt = copy.deepcopy(ChatGPT_Select_list_COT_Prompt)

        if idx % 2 == 0:
            prompt[-1]['content'] = prompt[-1]['content'].format_map(
                {
                    'user_intention': i,
                    'movie_list_1': '[' + ', '.join(['\'' + _ + '\'' for _ in seq_o]) + ']',
                    'movie_list_2': '[' + ', '.join(['\'' + _ + '\'' for _ in gic_o]) + ']',
                }
            )
        else:
            prompt[-1]['content'] = prompt[-1]['content'].format_map(
                {
                    'user_intention': i,
                    'movie_list_2': '[' + ', '.join(['\'' + _ + '\'' for _ in seq_o]) + ']',
                    'movie_list_1': '[' + ', '.join(['\'' + _ + '\'' for _ in gic_o]) + ']',
                }
            )
        while True:
            output = generate_openai(prompt, 0.2, 0.3)
            res = re.findall(r'<Selected List>.*</Selected List>', output)
            if len(res) > 0:
                if 'List 1' in res[0]:
                    gic_win = (idx % 2)
                    break
                elif 'List 2' in res[0]:
                    gic_win = 1 - (idx % 2)
                    break
                else:
                    gic_win = 0.5
                    break

        metrics[f'GIC_Win'] += gic_win


        GPT_eval_result.append(
            {
                'target_item': t,
                'seq_output': seq_o,
                'gic_output': gic_o,
                'intention': i,
                'intention_key': list(instruction.keys())[idx//32],
                'gic_win': gic_win,
                'GPT_output': output
            }
        )
        print(f'count: {idx+1-start} | ' + " | ".join([f"{_}: {__/(idx+1-start):.4f}" for _, __ in metrics.items()]))
        save_pickle(GPT_eval_result, save_file)
    print('GPT eval done!!!')


class FocalLoss(nn.Module):
    def __init__(self, weight=1.0, gamma=2.0, ignore_index=-100, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, labels):
        """
        logits: 未经softmax的模型输出, 形状为 [B, C, L], 其中 C 是词汇表大小
        labels: 真实标签, 形状为 [B, L]
        """
        # 计算log_softmax，为了数值稳定性
        log_preds = F.log_softmax(logits, dim=1)

        # 将 labels 转换为 one-hot 编码
        # B, C, L = logits.size()
        # labels_one_hot = F.one_hot(labels, num_classes=C).transpose(1, 2)  # 转换为 [B, C, L]

        # 选取对应标签的log预测概率
        log_preds = log_preds.gather(1, labels.unsqueeze(1)).squeeze(1)  # [B, L]

        # 计算Focal Loss
        preds_softmax = log_preds.exp()  # 将 log_softmax 转换回概率值
        focal_factor = (1 - preds_softmax) ** self.gamma  # 调整因子
        focal_loss = -1 * self.weight * focal_factor * log_preds  # 应用权重和调整因子

        # 处理忽略的索引
        if self.ignore_index is not None:
            mask = labels != self.ignore_index
            focal_loss = focal_loss * mask

        if self.reduction == 'mean':
            return focal_loss.masked_select(mask).mean()
        elif self.reduction == 'sum':
            return focal_loss.masked_select(mask).sum()
        else:  # 'none'
            return focal_loss


if __name__ == '__main__':
    GPT_eval(10)
