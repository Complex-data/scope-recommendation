import copy
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from SFT.SFT_templates import *
from utils import get_item_list, load_pickle, side_tokenizer, get_complete_text, get_output_text, get_history_text, \
    save_pickle


class SFTDataset(Dataset):
    def __init__(self, args, task_template, task_num, data, tokenizer, mode='train'):
        self.args = args
        self.task_template = task_template
        self.task_num = task_num
        self.mode = mode
        self.tokenizer = tokenizer
        # 根据不同的数据集需要调整对应sasrec的端口号
        if 'steam' in args.data_path:
            self.teacher_port = 2066
        elif 'toys' in args.data_path:
            self.teacher_port = 2067
        else: # movies
            self.teacher_port = 2068

        self.category2item = data['category']
        self.metas = data['metas']
        self.sequential = data['sequential']
        # self.preference = data['preference']
        # self.intention = data['intention']
        self.share_chat_gpt = data['share_chat_gpt']
        self.ranking_candidate = data['ranking_candidate']
        if args.user_control_symbol:
            self.item_list = data['item_list']
            ctrl_symbols = ['<SOI>', '<EOI>']
            self.ctrl_symbols = list(map(self.tokenizer.convert_tokens_to_ids, ctrl_symbols))
            # 创建前缀树
            self.create_item_prefix_tree()
            self.vocab_size = len(self.tokenizer)
        if self.args.llama2_chat_template:
            self.chat_gpt_conv = get_conversation_template("llama-2")
            self.chat_gpt_conv.set_system_message("You are a helpful, respectful and honest assistant.")
            self.chat_gpt_conv.append_message(self.chat_gpt_conv.roles[0], '')
            self.chat_gpt_conv.append_message(self.chat_gpt_conv.roles[1], None)
        if 'SFTTestSeqRec_Result' in data:
            self.SFTTestSeqRec_Result = {u: data['SFTTestSeqRec_Result'][idx].get('SFTTestSeqRec_output_title_list') or [] for idx, u in enumerate(self.sequential) if idx < len(data['SFTTestSeqRec_Result'])}
        if 'SFTTestSeqRec_Candidate' in data:
            self.SFTTestSeqRec_Candidate = data['SFTTestSeqRec_Candidate']

        self.item2category = {}
        for c in self.category2item:
            for i in self.category2item[c]:
                if self.item2category.get(i) is None:
                    self.item2category[i] = []
                self.item2category[i].append(c)

        self.title2item = {}
        for _ in self.metas:
            if self.title2item.get(self.metas[_][self.args.item_index]) is None:
                self.title2item[self.metas[_][self.args.item_index]] = []
            self.title2item[self.metas[_][self.args.item_index]].append(_)
        if args.is_main_process:
            print(f'compute_{self.mode}_datum_info')
        self.temp_candidate_items = None
        self.datum_info = []
        self.complete_datum_info_path = None
        if self.mode == 'val':
            self.complete_datum_info_path = args.data_path + f'SFT_datum_info_{self.mode}_{self.args.SFT_val_tasks}{"_LCT" if self.args.llama2_chat_template else ""}_Top{self.args.topk}.pickle'
        # if self.mode == 'test':
        #     self.complete_datum_info_path = args.data_path + f'SFT_datum_info_{self.mode}_{self.args.SFT_test_task}{"_LCT" if self.args.llama2_chat_template else ""}_Top{self.args.topk}.pickle'
        self.complete_datum_info = load_pickle(self.complete_datum_info_path) or []

        self.compute_datum_info()

    def create_item_prefix_tree(self):
        item_list = list(map(lambda x: x.strip(), self.item_list)) # list(list)
        item_ids = self.tokenizer.batch_encode_plus(item_list,add_special_tokens=False).data['input_ids']
        input_ids_append =[]
        for item in item_ids:
            new_list = []
            new_list= [self.ctrl_symbols[0]]+item+[self.ctrl_symbols[1]]
            input_ids_append.append(new_list)

        # self.item_prefix_tree = Trie(input_ids_append)
        self.item_prefix_tree = Trie_str(input_ids_append)

    # 根据input ids 获得scope_mask矩阵
    def get_scope_mask(self, labels):
        # return None
        scope_mask = []
        batch_size = labels.size()[0]
        step_size = labels.size()[1]

        scope_mask = torch.zeros(batch_size, step_size, self.vocab_size, dtype=torch.bool)
        # 找出所有控制符号的位置
        start_positions = (labels == self.ctrl_symbols[0]).nonzero()
        end_positions = (labels == self.ctrl_symbols[1]).nonzero()
        # 如果是只训练response部分的情况下就不用考虑这个了
        # separator_positions = (labels == self.separator_symbol).nonzero()
        # 存在一个控制符 self.separator_symbol,找到这个控制符的位置
        # 对start_positions，end_positions 进行过滤，和在self.separator_symbol控制符之前的位置过滤掉

        # 对于每个batch，处理有效区间

        for idx, end_pos in enumerate(end_positions):
            batch_idx, end_idx = end_pos[0], end_pos[1]

            # 检查是否是separator控制符之后的区间
            # separator_pos = separator_positions[batch_idx]
            # separator_batch_idx, separator_idx = separator_pos[0], separator_pos[1]

            # if start_idx < separator_idx:
            #     continue

            # 使用相同的索引来获取start_positions中的对应位置
            start_pos = start_positions[idx]
            start_batch_idx, start_idx = start_pos[0], start_pos[1]

            # 确保结束位置与开始位置在同一个batch中
            # assert batch_idx == separator_batch_idx

            # 确保开始位置与开始位置在同一个batch中
            assert batch_idx == start_batch_idx

            # 检查结束位置是否在开始位置之后
            if end_idx <= start_idx:
                raise ValueError("End index must be greater than start index")

            # 对于有效区间中的每个step，更新scope_mask
            for step_idx in range(start_idx + 1, end_idx):
                scope_list = self.item_prefix_tree.next_tokens(labels[batch_idx, start_idx:step_idx].tolist())
                if len(scope_list) < 2 : # 如果scope 内的token只有1个，这个时候就不做掩码操作了
                    continue
                scope_mask[batch_idx, step_idx, :] = True
                scope_mask[batch_idx, step_idx, scope_list] = False

        scope_mask = scope_mask.to(labels.device)

        return scope_mask

    def get_scope(self, labels):
        pass

    def find_maximum_category(self, item_list, target_item):
        category_count = {c: 0 for c in self.category2item if target_item not in self.category2item[c]}
        # if self.mode == 'test' and 'Genre for Featured Categories' in category_count:
        #     category_count.pop('Genre for Featured Categories')
        # if self.mode == 'test' and 'Studio Specials' in category_count:
        #     category_count.pop('Studio Specials')
        # if self.mode == 'test':
        #     category_count = {c: 0 for c in category_count if all([_ not in c for _ in self.metas[target_item]['category']])}
        for o_i in item_list:
            for c in self.item2category.get(o_i) or []:
                if c in category_count:
                    category_count[c] += 1
        max_count = max(list(category_count.values()))
        category = [c for c in list(category_count.keys()) if category_count[c] == max_count]
        return category

    def compute_datum_info(self):
        val_num = 320
        val_task_num = 0
        for task, num in self.task_num.items():
            if task in ["SFTSeqRec", "SFTPersonalControlRec", "SFTPersonalCategoryRate"]:
                for _ in range(num):
                    self.datum_info += [[task, u] for u in self.sequential]
            elif task in ["SFTControlRec", "SFTControlRec1"]:
                for _ in range(num):
                    self.datum_info += [[task, i] for i in self.metas if self.item2category.get(i)]
            elif task == "SFTCategoryRate":
                for _ in range(num):
                    self.datum_info += [[task, c, 'CategoryRate-LP'] for c in self.category2item]
                    self.datum_info += [[task, c, 'CategoryRate-MP'] for c in self.category2item]
                    self.datum_info += [[task, c, 'CategoryRate-LC'] for c in self.category2item]
                    self.datum_info += [[task, c, 'CategoryRate-MC'] for c in self.category2item]
            elif task == "SFTSeqRec-domain":
                for _ in range(num):
                    # 这里设置样本的数量是，序列推荐的100分之一
                    count = 0
                    for u in self.sequential:
                        self.datum_info += [[task, u]]
                        count += 1
                        if count >= 50:
                            break

            elif task == "GPTTestPersonalControlRec" and self.mode in ['test', 'val']:
                for _ in range(num):
                    self.datum_info += [[task, u, self.intention[idx]] for idx, u in enumerate(self.sequential) if idx < 768]

            elif task in ["SFTTestSeqRec", "SFTTestSeqRec-domain", "SFTTestSeqRanking", "SFT+TestPersonalControlRec", "SFT-TestPersonalControlRec",
                          "SFTTestItemCount"] or task.startswith("SFTTestPersonalCategoryRate"):
                for _ in range(num):
                    if self.mode == 'test':
                        self.datum_info += [[task, u] for u in self.sequential]
                    elif self.mode == 'val':
                        self.datum_info += [[task, u] for u in list(self.sequential.keys())[val_num*val_task_num: val_num*(val_task_num+1)]]
                        val_task_num += 1

            elif task == 'ShareChatGPT' and self.mode == 'train':
                share_chat_gpt_count = int(self.args.share_chat_gpt_ratio * len(self.datum_info))
                self.datum_info += [['ShareChatGPT'] for _ in range(share_chat_gpt_count)]
            else:
                raise NotImplementedError

        if len(self.complete_datum_info) != len(self.datum_info) and self.mode in ['val', 'test']:
            self.complete_datum_info = []
            for idx in tqdm(range(len(self.complete_datum_info), len(self.datum_info)), desc=f'computing {self.mode} datum info'):
                self.complete_datum_info.append(self.getitem(idx))
            if self.mode in ['val']:
                # input_ids = self.tokenizer.batch_encode_plus([_['input_text'] for _ in self.complete_datum_info], truncation=True, max_length=self.args.max_token_length)['input_ids']
                token_length = [len(_['input_text'].split()) for _ in self.complete_datum_info]
                datum_info_index = np.argsort(token_length)
                self.complete_datum_info = [self.complete_datum_info[idx] for idx in datum_info_index][::-1]
                save_pickle(self.complete_datum_info, self.complete_datum_info_path)

        # 在测试的情况下只用测试前一万个用户
        # if self.mode == 'test':
        #     cut_num = min(10000, len(self.datum_info))
        #     self.datum_info = self.datum_info[:cut_num]
        if self.mode == 'train':
            self.shuffle()

    def shuffle(self):
        random.shuffle(self.datum_info)

    def __len__(self):
        return len(self.datum_info)

    def get_item_index(self, item):
        if self.args.item_index == 'title':
            return self.metas[item]['title']
        elif self.args.item_index == 'title32':
            return self.metas[item]['title32']
        elif self.args.item_index == 'title64':
            return self.metas[item]['title64']
        elif self.args.item_index == 'item':
            return item
        else:
            return self.metas[item][self.args.item_index]

    def get_sub_sequential(self, user):
        if self.mode == 'train':
            sequential = self.sequential[user][:-2]
            target_item_index = random.choice(range(1, len(sequential)))
            min_start_item_index = max(0, target_item_index-self.args.max_item_length)
            start_item_index = random.choice(range(min_start_item_index, target_item_index))
            sub_sequential = sequential[start_item_index: target_item_index]
            target_item = sequential[target_item_index]
        elif self.mode == 'val':
            sub_sequential = self.sequential[user][-self.args.max_item_length-2:-2]
            target_item = self.sequential[user][-2]
        elif self.mode == 'test':
            sub_sequential = self.sequential[user][-self.args.max_item_length-1:-1]
            target_item = self.sequential[user][-1]
        else:
            raise NotImplementedError
        return sub_sequential, target_item

    def get_output_item_list(self, task, user=None, sub_sequential=None, target_item=None, target_category=None, direction=None, item_count=None, category_item_count=None, has_candidate=False):
        output_items, candidate_items = [], []
        if task in ['SFTSeqRec','SFTSeqRec-domain' ,'SFTTestSeqRanking']:
            output_items = get_item_list(self.args.backup_ip, [user], [sub_sequential], item_count, port=self.teacher_port)['inference'][0]
            if target_item in output_items:
                output_items.remove(target_item)
            output_items = ([target_item] + output_items)[:item_count]
        elif task in ["SFTControlRec", "SFTControlRec-domain", "SFTControlRec1"]:
            if direction == '+':
                output_items = copy.deepcopy(self.category2item[target_category])
            else:
                output_items = list(set(list(self.metas.keys()))-set(self.category2item[target_category]))
            random.shuffle(output_items)
            if target_item in output_items:
                output_items.remove(target_item)
            output_items = ([target_item] + output_items)[:item_count]
        elif task in ["SFTPersonalControlRec"]:
            output_items = get_item_list(self.args.backup_ip, [user], [sub_sequential], item_count,
                                         target_category=[direction+target_category], port=self.teacher_port)['inference'][0]
            if target_item in output_items:
                output_items.remove(target_item)
                output_items = ([target_item] + output_items)
        elif task in ["SFTCategoryRate"]:
            output_items = random.sample(self.category2item[target_category], category_item_count)
            output_items = output_items + random.sample(list(set(self.metas.keys()) - set(self.category2item[target_category])),
                                                        item_count-category_item_count)
            random.shuffle(output_items)
        elif task in ['SFTPersonalCategoryRate']:
            in_category_items = get_item_list(self.args.backup_ip, [user], [sub_sequential], category_item_count,
                                              target_category=['+'+target_category], port=self.teacher_port)['inference'][0]
            out_category_items = get_item_list(self.args.backup_ip, [user], [sub_sequential], item_count-category_item_count,
                                               target_category=['-'+target_category], port=self.teacher_port)['inference'][0]
            output_items = in_category_items + out_category_items
            random.shuffle(output_items)
        else:
            raise NotImplementedError

        assert len(output_items) == item_count
        if has_candidate:
            candidate_num = random.choice(range(self.args.topk, self.args.candidate_num+1))
            candidate_items = output_items + random.choices(list(set(self.metas.keys())-set(output_items)), k=candidate_num-item_count)
            random.shuffle(candidate_items)
        return output_items, candidate_items

    def getitem(self, idx):
        task = self.datum_info[idx][0]
        template_id = random.choice(list(self.task_template[task].keys()))
        template_selected = self.task_template[task][template_id]
        input_field_data, output_field_data = {}, {}
        if task in ["SFTSeqRec", "SFTSeqRec-domain", "SFTTestSeqRec", "SFTTestSeqRec-domain", "SFTSeqRanking", "SFTTestSeqRanking", "SFTPersonalControlRec",
                    "SFT+TestPersonalControlRec", "SFT-TestPersonalControlRec", "SFTPersonalCategoryRate",
                    "SFTTestItemCount"] or task.startswith("SFTTestPersonalCategoryRate"):
            user = self.datum_info[idx][1]
            sub_sequential, target_item = self.get_sub_sequential(user)
            input_field_data.update({
                'user': user,
                'target_item': target_item,
                'sub_sequential': sub_sequential,
                'history': get_history_text([f"'{self.get_item_index(_)}'" for _ in sub_sequential]),
            })

            if task in ["SFTSeqRec","SFTSeqRec-domain"]:
                item_count = random.choice(range(self.args.topk))+1
                # temp = get_item_list(self.args.backup_ip, [user], [sub_sequential], item_count)['inference'][0]
                # if target_item in temp:
                #     temp.remove(target_item)
                # temp = ([target_item] + temp)[:item_count]
                output_items, candidate_items = self.get_output_item_list(task, user=user, sub_sequential=sub_sequential,
                                                                          target_item=target_item, item_count=item_count,
                                                                          has_candidate='candidate_titles' in template_selected.input_fields)
                input_field_dict = {
                    'target_category': self.item2category.get(target_item)[-1],
                    'item_count': item_count,
                    'candidate_titles': ', '.join([f"'{self.get_item_index(_)}'" for _ in candidate_items]),
                    'candidate_items': candidate_items,
                    'domain':self.args.domain
                }
                if task == "SFTSeqRec-domain": # 如果使用domain，就获取数据集的名称
                    input_field_dict['domain_right'] = self.args.domain
                    input_field_dict['domain_wrong'] = 'toys'

                input_field_data.update(input_field_dict)
                output_field_data.update({
                    'item_list': get_output_text([self.get_item_index(_) for _ in output_items], '\n'+self.tokenizer.eos_token, self.args.idx,self.args.user_control_symbol),
                    'domain_right' : self.args.domain
                })
            elif task in ["SFTTestSeqRec","SFTTestSeqRec-domain"]:
                item_count = self.args.topk
                input_field_dict = {
                    'target_category': self.item2category.get(target_item)[-1],
                    'item_count': item_count
                }
                if task == "SFTTestSeqRec-domain":
                    input_field_dict['domain'] = self.args.domain
                input_field_data.update(input_field_dict)
                output_field_data.update({
                    'item_list': get_output_text([self.get_item_index(target_item)])
                })
            elif task in ['SFTTestSeqRanking']:
                item_count = self.args.topk
                ranking_candidate = self.ranking_candidate[user][:self.args.candidate_num-1]
                insert_idx = idx % self.args.candidate_num
                ranking_candidate.insert(insert_idx, target_item)
                input_field_data.update({
                    'target_category': self.item2category.get(target_item)[-1],
                    'item_count': item_count,
                    'candidate_titles': ', '.join([f"'{self.get_item_index(_)}'" for _ in ranking_candidate]),
                    'candidate_items': ranking_candidate
                })
                output_field_data.update({
                    'item_list': get_output_text([self.get_item_index(target_item)])
                })
            elif task in ['SFTTestItemCount']:
                item_count = self.args.topk + 1 + idx % 5
                input_field_data.update({
                    'target_category': self.item2category.get(target_item)[-1],
                    'item_count': item_count
                })
                output_field_data.update({
                    'item_list': get_output_text([self.get_item_index(target_item)])
                })
            elif task in ["SFTPersonalControlRec"]:
                if random.random() > 0.5:
                    intention_group, d = Intention_plus_group, '+'
                    target_category = random.choice(list(self.category2item.keys()))
                    max_count = min(self.args.topk, len(self.category2item[target_category]))
                else:
                    intention_group, d = Intention_minus_group, '-'
                    SASRec_output = get_item_list(self.args.backup_ip, [user], [sub_sequential], self.args.topk, port=self.teacher_port)['inference'][0]
                    target_category = random.choice(self.find_maximum_category(SASRec_output, target_item))
                    max_count = min(self.args.topk, len(self.metas) - len(self.category2item[target_category]))
                item_count = random.choice(range(max_count))+1 if task == "SFTPersonalControlRec" else 1
                # temp = get_item_list(self.args.backup_ip, [user], [sub_sequential], item_count, target_category=[d+target_category])['inference'][0]
                # if target_item in temp:
                #     temp.remove(target_item)
                #     temp = ([target_item] + temp)
                output_items, candidate_items = self.get_output_item_list(task, user=user, sub_sequential=sub_sequential,
                                                                          target_item=target_item, item_count=item_count,
                                                                          target_category=target_category, direction=d,
                                                                          has_candidate='candidate_titles' in template_selected.input_fields)
                input_field_data.update({'target_category': target_category})
                intention_template_key = random.choice(list(intention_group.keys()))
                intention = intention_group[intention_template_key].get_input_text(input_field_data)
                input_field_data.update({
                    'synthetic_intention': intention,
                    'item_count': item_count,
                    'candidate_titles': ', '.join([f"'{self.get_item_index(_)}'" for _ in candidate_items]),
                    'candidate_items': candidate_items
                })
                output_field_data.update({
                    'item_list': get_output_text([self.get_item_index(_) for _ in output_items], '\n'+self.tokenizer.eos_token, self.args.idx)
                })
            elif task in ["SFT+TestPersonalControlRec", "SFT-TestPersonalControlRec"]:
                if self.mode == 'test':
                    SeqRec_item_list = self.SFTTestSeqRec_Result[user]
                    item_list = [self.title2item[_][0] if _ in self.title2item else 'None' for _ in SeqRec_item_list]
                else:
                    item_list = get_item_list(self.args.backup_ip, [user], [sub_sequential], self.args.topk, port=self.teacher_port)['inference'][0]
                if task == "SFT+TestPersonalControlRec":
                    target_category = self.item2category.get(target_item)[-1]
                    intention_group = Intention_plus_group
                else:
                    target_category = self.find_maximum_category(item_list, target_item)[-1]
                    intention_group = Intention_minus_group
                item_count = self.args.topk
                input_field_data.update({'target_category': target_category})
                intention_template_key = random.choice(list(intention_group.keys()))
                intention = intention_group[intention_template_key].get_input_text(input_field_data)
                input_field_data.update({
                    'synthetic_intention': intention,
                    'item_count': item_count,
                    'SeqRec_Result': item_list
                })
                output_field_data.update({
                    'item_list': get_output_text([self.get_item_index(target_item)])
                })

            elif task == "SFTPersonalCategoryRate":
                template_id = random.choice(list(self.task_template[task].keys()))
                template_selected = self.task_template[task][template_id]
                item_count = 10
                target_category = random.choice(list(self.category2item.keys()))
                category_item_count = min(len(self.category2item[target_category]), item_count)
                target_category_item_count = random.choice(range(category_item_count)) + 1 if template_id[-2] == 'L' else random.choice(range(category_item_count))
                output_category_item_count = target_category_item_count + random.choice([-1 if template_id[-2] == 'L' else 1, 0])
                output_items, candidate_items = self.get_output_item_list(task, user=user, sub_sequential=sub_sequential,
                                                                          item_count=item_count, target_category=target_category,
                                                                          category_item_count=output_category_item_count,
                                                                          has_candidate='candidate_titles' in template_selected.input_fields)
                input_field_data.update({
                    'target_category': target_category,
                    'item_count': item_count,
                    'category_proportion': f"{target_category_item_count}0%",
                    'category_count': target_category_item_count,
                    'candidate_titles': ', '.join([f"'{self.get_item_index(_)}'" for _ in candidate_items]),
                    'candidate_items': candidate_items
                })
                output_field_data.update({
                    'item_list': get_output_text([self.get_item_index(_) for _ in output_items], '\n' + self.tokenizer.eos_token, self.args.idx)
                })

            elif task.startswith("SFTTestPersonalCategoryRate"):
                if self.mode == 'test':
                    SeqRec_item_list = self.SFTTestSeqRec_Result[user]
                    item_list = [self.title2item[_][0] if _ in self.title2item else 'None' for _ in SeqRec_item_list]
                else:
                    item_list = get_item_list(self.args.backup_ip, [user], [sub_sequential], self.args.topk, port=self.teacher_port)['inference'][0]
                if 'LP1' in task or 'LP' in task:
                    target_category = self.find_maximum_category(item_list, target_item)[-1]
                else:
                    target_category = self.item2category.get(target_item)[-1]
                template_selected = self.task_template[task]['PersonalCategoryRate']
                item_count = self.args.topk
                p = int(task.split('_')[-1])
                input_field_data.update({
                    'target_category': target_category,
                    'item_count': item_count,
                    'category_proportion': f'{p}%',
                    'category_count': int(p*item_count/100),
                    'SeqRec_Result': item_list,
                })
                output_field_data.update({
                    'item_list': get_output_text([self.get_item_index(target_item)])
                })
            elif task in ["GPTTestPersonalControlRec"]:
                generate_intention = self.datum_info[idx][2]
                item_count = self.args.topk
                input_field_data.update({
                    'synthetic_intention': generate_intention,
                    'item_count': item_count,
                })
                output_field_data.update({
                    'item_list': ''
                })

        elif task in ["SFTControlRec","SFTControlRec-domain", "SFTControlRec1"]:
            target_item = self.datum_info[idx][1]
            if 'reverse' in template_id:
                input_field_data.update({
                    'item': self.get_item_index(target_item),
                })
                output_field_data.update({
                    'item': self.get_item_index(target_item),
                    'target_category': random.choice(self.item2category[target_item]),
                })
            else:
                # 这里设置成正的 random.random()
                if 1 > 0.5:
                    target_category = random.choice(self.item2category[target_item])
                    max_count = min(self.args.topk, len(self.category2item[target_category]))
                    intention_group, d = Intention_plus_group, '+'
                    # temp = copy.deepcopy(self.category2item[target_category])
                else:
                    categories = [c for c in self.category2item if target_item not in self.category2item[c]]
                    target_category = random.choice(categories)
                    max_count = min(self.args.topk, len(self.metas)-len(self.category2item[target_category]))
                    intention_group, d = Intention_minus_group, '-'
                    # temp = list(set(list(self.metas.keys()))-set(self.category2item[target_category]))
                item_count = random.choice(range(max_count))+1 if task == "SFTControlRec" else 1
                # random.shuffle(temp)
                # if target_item in temp:
                #     temp.remove(target_item)
                # temp = ([target_item] + temp)[:item_count]

                output_items, candidate_items = self.get_output_item_list(task, target_item=target_item, item_count=item_count,
                                                                          target_category=target_category, direction=d,
                                                                          has_candidate='candidate_titles' in template_selected.input_fields)

                input_field_data.update({'target_category': target_category})
                intention_template_key = random.choice(list(intention_group.keys()))
                intention = intention_group[intention_template_key].get_input_text(input_field_data)
                input_field_dict = {
                    'synthetic_intention': intention,
                    'item_count': item_count,
                    'candidate_titles': ', '.join([f"'{self.get_item_index(_)}'" for _ in candidate_items]),
                    'candidate_items': candidate_items
                }
                if task == "SFTControlRec-domain":
                    input_field_dict['domain'] = self.args.domain
                input_field_data.update(input_field_dict)
                output_field_data.update({
                    'item_list': get_output_text([self.get_item_index(_) for _ in output_items], '\n'+self.tokenizer.eos_token, self.args.idx, self.args.user_control_symbol,self.args.use_scope_mask) # method3的情况下只有rec任务有控制符，商品搜索任务没有控制符
                })
        elif task == "SFTCategoryRate":
            target_category = self.datum_info[idx][1]
            template_id = self.datum_info[idx][2]
            template_selected = self.task_template[task][template_id]
            item_count = 10
            category_item_count = min(len(self.category2item[target_category]), item_count)
            target_category_item_count = random.choice(range(category_item_count))+1 if template_id[-2] == 'L' else random.choice(range(category_item_count))
            output_category_item_count = target_category_item_count + random.choice([-1 if template_id[-2] == 'L' else 1, 0])
            # temp = random.sample(self.category2item[target_category], output_category_item_count)
            # temp = temp + random.sample(list(set(self.metas.keys())-set(self.category2item[target_category])), item_count-output_category_item_count)
            # random.shuffle(temp)
            output_items, candidate_items = self.get_output_item_list(task, item_count=item_count, target_category=target_category,
                                                                      category_item_count=output_category_item_count,
                                                                      has_candidate='candidate_titles' in template_selected.input_fields)
            input_field_data.update({
                'target_category': target_category,
                'item_count': item_count,
                'category_proportion': f"{target_category_item_count}0%",
                'category_count': target_category_item_count,
                'candidate_titles': ', '.join([f"'{self.get_item_index(_)}'" for _ in candidate_items]),
                'candidate_items': candidate_items
            })
            output_field_data.update({
                'item_list': get_output_text([self.get_item_index(_) for _ in output_items], '\n'+self.tokenizer.eos_token, self.args.idx)
            })
        elif task == "ShareChatGPT":
            scg_data = random.choice(self.share_chat_gpt)
            if isinstance(scg_data[0], str):
                [input_text, output_text] = scg_data
                if self.args.llama2_chat_template:
                    self.chat_gpt_conv.messages[-2][1] = input_text
                    input_text = self.chat_gpt_conv.get_prompt()
                else:
                    raise NotImplementedError
            else:
                chat_end_idx = random.choice([idx for idx, c in enumerate(scg_data) if c['from'] == 'human'])
                chat_start_idxes = [chat_end_idx]
                chat_start_idx = chat_start_idxes[-1]-2
                while chat_start_idx > 0:
                    chat_start_idx = chat_start_idx-2
                    pre_length = scg_data[chat_start_idx-1]['pre_length'] if chat_start_idx > 0 else 0
                    if scg_data[chat_start_idx]['pre_length'] - pre_length < self.args.max_token_length-64:
                        chat_start_idxes.append(chat_start_idx)
                    else:
                        break
                chat_start_idx_selected = random.choice(chat_start_idxes)
                if self.args.llama2_chat_template:
                    chat_gpt_conv = get_conversation_template("llama-2")
                    chat_gpt_conv.set_system_message("You are a helpful, respectful and honest assistant.")
                    input_data = scg_data[chat_start_idx_selected:chat_end_idx+1]+[{'from': 'gpt', 'value': None}]
                    for idx in range(0, len(input_data), 2):
                        assert input_data[idx]['from'] == 'human'
                        chat_gpt_conv.append_message(chat_gpt_conv.roles[0], input_data[idx]['value'])
                        assert input_data[idx+1]['from'] == 'gpt'
                        chat_gpt_conv.append_message(chat_gpt_conv.roles[1], input_data[idx+1]['value'])
                    input_text = chat_gpt_conv.get_prompt()
                    output_text = scg_data[chat_end_idx+1]['value']
                else:
                    raise NotImplementedError
            out_dict = {
                'input_text': input_text,
                'output_text': output_text+self.tokenizer.eos_token,
                'task': task,
                'input_field_data': input_field_data,
            }
            return out_dict
        else:
            raise NotImplementedError

        input_text = template_selected.get_input_text(input_field_data, llama2_chat_template=self.args.llama2_chat_template).strip()
        output_text = template_selected.get_output_text(output_field_data).strip()


        out_dict = {
            'input_text': input_text,
            'output_text': output_text,
            'task': task,
            'input_field_data': input_field_data,
        }
        return out_dict

    def __getitem__(self, idx):
        if self.mode in ['train']:
            return self.getitem(idx)
        return self.complete_datum_info[idx]

    def collate_fn(self, batch):
        batch_entry = {}

        tasks = []
        input_text = []
        output_text = []
        complete_text = []
        input_field_data = []
        for i, entry in enumerate(batch):
            if 'task' in entry:
                tasks.append(entry['task'])
            if 'input_text' in entry:
                input_text.append(entry['input_text'])
            if 'output_text' in entry:
                output_text.append(entry['output_text'])
            if 'input_field_data' in entry:
                input_field_data.append(entry['input_field_data'])
            complete_text.append(get_complete_text(entry['input_text'], entry['output_text']))
        batch_entry['input_text'] = input_text
        batch_entry['output_text'] = output_text
        batch_entry['complete_text'] = complete_text
        batch_entry['input_field_data'] = input_field_data

        batch_entry['input_data'] = side_tokenizer(batch_entry['input_text'],
                                                   'left', self.tokenizer,
                                                   padding=True, truncation=True,
                                                   max_length=self.args.max_token_length,
                                                   return_tensors='pt').to(self.args.gpu).data
        batch_entry['output_data'] = side_tokenizer(batch_entry['output_text'],
                                                    'right', self.tokenizer,
                                                    padding=True, truncation=True,
                                                    max_length=self.args.gen_max_length,
                                                    return_tensors='pt').to(self.args.gpu).data
        batch_entry['complete_text_data'] = {
            'input_ids':
                torch.cat([batch_entry['input_data']['input_ids'], batch_entry['output_data']['input_ids'][:, 1:]], dim=-1),
            'attention_mask':
                torch.cat([batch_entry['input_data']['attention_mask'], batch_entry['output_data']['attention_mask'][:, 1:]], dim=-1)
        }
        prompt_length = batch_entry['input_data']['input_ids'].shape[-1]
        batch_entry['complete_label_ids'] = copy.deepcopy(batch_entry['complete_text_data']['input_ids'])
        batch_entry['complete_label_ids'][..., :prompt_length] = -100
        batch_entry['complete_label_ids'][batch_entry['complete_label_ids'] == self.tokenizer.pad_token_id] = -100
        batch_entry['task'] = tasks

        return batch_entry


Train_task_group_mapping = {
    "SFTSeqRec": SeqRec_group,
    "SFTSeqRec-domain": SeqRec_domain_group,
    "SFTControlRec": ControlRec_group,
    "SFTControlRec-domain":ControlRec_domain_group,
    "SFTControlRec1": ControlRec1_group,
    "SFTPersonalControlRec": PersonalControlRec_group,
    "SFTCategoryRate": CategoryRate_group,
    "SFTPersonalCategoryRate": PersonalCategoryRate_group,
    "ShareChatGPT": {'ShareChatGPT-1': ''},
}

Val_task_group_mapping = {
    "SFTTestSeqRec": ValSeqRec_group,
    "SFTTestSeqRec-domain": ValSeqRec_domain_group,
    "SFTTestSeqRanking": ValSeqRanking_group,
    # "SFTValControlRec": ValControlRec_group,
    # "SFTValPersonalControlRec": ValPersonalControlRec_group,
    "SFT+TestPersonalControlRec": ValPersonalControlRec_group,
    "SFT-TestPersonalControlRec": ValPersonalControlRec_group,
    "SFTTestPersonalCategoryRateLP1": TestPersonalCategoryRateLP1_group,
    'SFTTestItemCount': ValSeqRec_group,
}
# SFTTestSeqRec,SFTTestSeqRanking,SFT+TestPersonalControlRec,SFT-TestPersonalControlRec,SFTTestPersonalCategoryRate,SFTTestItemCount

Test_task_group_mapping = {
    "SFTTestSeqRec": ValSeqRec_group,
    "SFTTestSeqRec-domain": ValSeqRec_domain_group,
    "SFTTestSeqRanking": ValSeqRanking_group,
    "SFT+TestPersonalControlRec": ValPersonalControlRec_group,
    "SFT-TestPersonalControlRec": ValPersonalControlRec_group,
    "SFTTestPersonalCategoryRateLP": TestPersonalCategoryRateLP_group,
    "SFTTestPersonalCategoryRateLP1": TestPersonalCategoryRateLP1_group,
    "SFTTestPersonalCategoryRateMP": TestPersonalCategoryRateMP_group,
    "SFTTestPersonalCategoryRateEP": TestPersonalCategoryRateEP_group,
    'SFTTestItemCount': ValSeqRec_group,
}
Test_Ctrl_task_group_mapping = {
    "SFTTestSeqRec": ValSeqRec_group,
    "SFTTestSeqRec-domain": ValSeqRec_domain_group,
    "SFTTestSeqRanking": ValSeqRanking_group,
    "SFT+TestPersonalControlRec": ValPersonalControlRec_group,
    "SFT-TestPersonalControlRec": ValPersonalControlRec_group,
    "SFTTestPersonalCategoryRateLP": TestPersonalCategoryRateLP_group,
    "SFTTestPersonalCategoryRateLP1": TestPersonalCategoryRateLP1_group,
    "SFTTestPersonalCategoryRateMP": TestPersonalCategoryRateMP_group,
    "SFTTestPersonalCategoryRateEP": TestPersonalCategoryRateEP_group,
    'SFTTestItemCount': ValSeqRec_group,
}

class Trie:
    def __init__(self,token_ids, no_subsets=False) -> None: 
        self.max_height = max([len(one) for one in token_ids])

        root = {}
        for token_ids in token_ids:
            level = root
            for tidx, token_id in enumerate(token_ids):
                if token_id not in level:
                    level[token_id] = {}

                level = level[token_id]

        if no_subsets and self.has_subsets(root, token_ids):
            raise ValueError(
                "Each list in `token_ids` can't be a complete subset of another list, but is"
                f" {token_ids}."
            )

        self.trie = root

    def has_subsets(self, trie, nested_token_ids):
        leaf_count = self.count_leaves(trie)
        return len(nested_token_ids) != leaf_count
    
    def next_tokens(self, current_seq): # 这个只返回一层对应的token 
        
        if len(current_seq) == 0:
            return list(self.trie.keys())

        start = self.trie

        for current_token in current_seq:
            start = start[current_token]

        next_tokens = list(start.keys())

        return next_tokens

    def reached_leaf(self, current_seq):
        next_tokens = self.next_tokens(current_seq)

        return len(next_tokens) == 0

    def count_leaves(self, root):
        next_nodes = list(root.values())
        if len(next_nodes) == 0:
            return 1
        else:
            return sum([self.count_leaves(nn) for nn in next_nodes])

# 构建一个单层的字典，每个查询都是o(1) 的时间复杂度
class Trie_str:
    def __init__(self, item_list) -> None:
        self.max_height = max([len(one) for one in item_list])
        self.start_id = item_list[0][0] # <SOI> 用于检索第一层
        self.trie = {}
        for tokens in item_list:
            prefix = [str(tokens[0])]
            for i in range(1, len(tokens)):
                prefix_str = ','.join(prefix)
                if prefix_str not in self.trie.keys():
                    self.trie[prefix_str] = set()
                prefix.append(str(tokens[i]))
                self.trie[prefix_str].add(tokens[i])


    def next_tokens(self, current_seq):  # 这个只返回一层对应的token
        if len(current_seq) == 0:
            return self.trie[str(self.start_id)]
        current_seq = list(map(lambda x: str(x),current_seq))
        key_str = ','.join(current_seq)
        next_tokens = self.trie[key_str]

        return list(next_tokens)

    def reached_leaf(self, current_seq):
        next_tokens = self.next_tokens(current_seq)

        return len(next_tokens) == 0

    def count_leaves(self, root):
        next_nodes = list(root.values())
        if len(next_nodes) == 0:
            return 1
        else:
            return sum([self.count_leaves(nn) for nn in next_nodes])