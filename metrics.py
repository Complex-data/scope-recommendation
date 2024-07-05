import copy
import json
import math
from Levenshtein import distance


class Metrics:
    def __init__(self, tasks, topk, category2item, title2item):
        self.tasks = tasks
        self.topk = topk
        self.category2item = category2item
        self.title2item = title2item
        self.metrics_dict = {_: self.task_register(_) for _ in self.tasks}

    def task_register(self, task):
        metrics_dict = {
            f'NonExistRate@{self.topk}': 0.0,
            f'RepeatRate@{self.topk}': 0.0,
            f'CorrectCount@{self.topk}': 0.0,
            'Count': 1e-24,
            f'Recall@{self.topk}': 0.0,
            f'MRR@{self.topk}': 0.0,
            f'NDCG@{self.topk}': 0.0,
            f'TargetCategoryRate@{self.topk}': 0.0,
            f'InHistoryRate@{self.topk}': 0.0,
        }
        if task in ['SFT+TestPersonalControlRec', 'SFT-TestPersonalControlRec', 'RLHF+PersonalControlRec',
                    'RLHF-PersonalControlRec', 'RLHFPersonalCategoryRate'] or task.startswith('SFTTestPersonalCategoryRate'):
            metrics_dict.update({
                f'SRTargetCategoryRate@{self.topk}': 0.0,
            })
        if task in ['RLHFSeqRec', 'RLHF+PersonalControlRec', 'RLHF-PersonalControlRec', 'RLHFSeqRanking', 'RLHFItemCount',
                    'RLHFTotal'] or task.startswith('RLHFPersonalCategoryRate'):
            metrics_dict.update({
                f'RewardSum': 0.0,
            })
        if task.startswith('SFTTestPersonalCategoryRate') or task.startswith('RLHFPersonalCategoryRate'):
            metrics_dict.update({
                f'CategoryRateCorrect@{self.topk}': 0.0,
                f'SRCategoryRateCorrect@{self.topk}': 0.0,
            })
        if task in ['SFTTestSeqRanking', 'RLHFSeqRanking']:
            metrics_dict.update({
                f'NotInCandidateRate@{self.topk}': 0.0,
            })
        return metrics_dict

    # def add_sample(self, task, input_field_data, output_titles, target_title, list_reward=0.0):
    #     CorrectCount = 1 if len(output_titles) == input_field_data['item_count'] else 0
    #     _output_titles = output_titles[:input_field_data['item_count']]
    #     NonExistRate = sum([1 if _ not in self.title2item else 0 for _ in _output_titles])
    #     RepeatRate = sum([1 if _ in _output_titles[:idx] else 0 for idx, _ in enumerate(_output_titles)])
    #     output_items = [self.title2item[_][0] if self.title2item.get(_) else 'None' for _ in _output_titles]
    #
    #     self.metrics_dict[task][f'NonExistRate@{self.topk}'] += NonExistRate
    #     self.metrics_dict[task][f'RepeatRate@{self.topk}'] += RepeatRate
    #     self.metrics_dict[task][f'CorrectCount@{self.topk}'] += CorrectCount
    #     self.metrics_dict[task][f'Count'] += 1
    #
    #     Recall = 1 if target_title[0] in _output_titles else 0
    #     self.metrics_dict[task][f'Recall@{self.topk}'] += Recall
    #     MRR = 1 / (_output_titles.index(target_title[0]) + 1) if target_title[0] in _output_titles else 0
    #     self.metrics_dict[task][f'MRR@{self.topk}'] += MRR
    #     NDCG = 1 / math.log2(_output_titles.index(target_title[0]) + 2) if target_title[0] in _output_titles else 0
    #     self.metrics_dict[task][f'NDCG@{self.topk}'] += NDCG
    #
    #     target_category = input_field_data['target_category']
    #     TargetCategoryRatio = sum([1 if _ in self.category2item[target_category] else 0 for _ in output_items])
    #     self.metrics_dict[task][f'TargetCategoryRatio@{self.topk}'] += TargetCategoryRatio
    #
    #     if task in ['SFT+TestPersonalControlRec', 'SFT-TestPersonalControlRec', 'SFTTestPersonalCategoryRate']:
    #         _SeqRec_output_titles = input_field_data['SFTTestSeqRec_Result'][:input_field_data['item_count']]
    #         SeqRec_item_list = [self.title2item[_][0] if self.title2item.get(_) else 'None' for _ in _SeqRec_output_titles]
    #         SeqRecTargetCategoryRatio = sum([1 if _ in self.category2item[target_category] else 0 for _ in SeqRec_item_list])
    #         self.metrics_dict[task][f'SeqRecTargetCategoryRatio@{self.topk}'] += SeqRecTargetCategoryRatio
    #         if task in ['SFTTestPersonalCategoryRate']:
    #             CategoryRateCorrect = 1 if TargetCategoryRatio <= input_field_data['category_count'] else 0
    #             self.metrics_dict[task][f'CategoryRateCorrect@{self.topk}'] += CategoryRateCorrect
    #
    #     elif task in ['RLHFSeqRec', 'RLHFSeqRanking', 'RLHFItemCount', 'RLHFTotal']:
    #         self.metrics_dict[task]['RewardSum'] += list_reward
    #
    #     elif task in ['RLHF+PersonalControlRec', 'RLHF-PersonalControlRec', 'RLHFPersonalCategoryRate']:
    #         self.metrics_dict[task]['RewardSum'] += list_reward
    #         if 'RLHFSeqRec_Result' in input_field_data:
    #             _SeqRec_output_titles = input_field_data['RLHFSeqRec_Result'][:input_field_data['item_count']]
    #             SeqRec_item_list = [self.title2item[_][0] if self.title2item.get(_) else 'None' for _ in _SeqRec_output_titles]
    #             SeqRecTargetCategoryRatio = sum([1 if _ in self.category2item[target_category] else 0 for _ in SeqRec_item_list])
    #             self.metrics_dict[task][f'SeqRecTargetCategoryRatio@{self.topk}'] += SeqRecTargetCategoryRatio
    #         if task in ['RLHFPersonalCategoryRate']:
    #             CategoryRateCorrect = 1 if TargetCategoryRatio <= input_field_data['category_count'] else 0
    #             self.metrics_dict[task][f'CategoryRateCorrect@{self.topk}'] += CategoryRateCorrect
    #
    #     if 'candidate_items' in input_field_data:
    #         NotInCandidateRatio = sum([1 if _ not in input_field_data['candidate_items'] else 0 for _ in output_items])
    #         self.metrics_dict[task][f'NotInCandidateRatio@{self.topk}'] += NotInCandidateRatio

    def add_sample(self, task, input_field_data, output_titles, target_title, list_reward=0.0, vague_mapping=True):
        CorrectCount = 1 if len(output_titles) == input_field_data['item_count'] else 0
        _output_titles = output_titles[:input_field_data['item_count']]
        if vague_mapping:
            for idx, _ in enumerate(_output_titles):
                if _ not in self.title2item:
                    for __ in self.title2item:
                        if __ not in _output_titles[:idx] and distance(_, __) <= 3:
                            _output_titles[idx] = __
                            break
        NonExistRate = sum([1 if _ not in self.title2item else 0 for _ in _output_titles])
        RepeatRate = sum([1 if _ in _output_titles[:idx] else 0 for idx, _ in enumerate(_output_titles)])
        output_items = [self.title2item[_][0] if self.title2item.get(_) else 'None' for _ in _output_titles]
        InHistoryRate = sum([1 if _ in input_field_data['sub_sequential'] else 0 for idx, _ in enumerate(output_items)])

        self.metrics_dict[task][f'NonExistRate@{self.topk}'] += NonExistRate
        self.metrics_dict[task][f'RepeatRate@{self.topk}'] += RepeatRate
        self.metrics_dict[task][f'InHistoryRate@{self.topk}'] += InHistoryRate
        self.metrics_dict[task][f'CorrectCount@{self.topk}'] += CorrectCount
        self.metrics_dict[task][f'Count'] += 1

        if target_title[0] in _output_titles:
            idx = _output_titles.index(target_title[0])
            self.metrics_dict[task][f'Recall@{self.topk}'] += 1
            self.metrics_dict[task][f'MRR@{self.topk}'] += 1/(idx+1)
            self.metrics_dict[task][f'NDCG@{self.topk}'] += 1/math.log2(idx+2)

        target_category = input_field_data['target_category']
        TargetCategoryRate = sum([1 if _ in self.category2item[target_category] else 0 for _ in output_items])
        self.metrics_dict[task][f'TargetCategoryRate@{self.topk}'] += TargetCategoryRate

        if f'SRTargetCategoryRate@{self.topk}' in self.metrics_dict[task] and 'SeqRec_Result' in input_field_data:
            SeqRec_item_list = input_field_data['SeqRec_Result'][:input_field_data['item_count']]
            # SeqRec_item_list = [self.title2item[_][0] if self.title2item.get(_) else 'None' for _ in _SeqRec_output_titles]
            SRTargetCategoryRate = sum([1 if _ in self.category2item[target_category] else 0 for _ in SeqRec_item_list])
            self.metrics_dict[task][f'SRTargetCategoryRate@{self.topk}'] += SRTargetCategoryRate
            if f'SRCategoryRateCorrect@{self.topk}' in self.metrics_dict[task]:
                if 'MP' in task or 'MC' in task:
                    SRCategoryRateCorrect = 1 if SRTargetCategoryRate >= input_field_data['category_count'] else 0
                elif 'EP' in task or 'EC' in task:
                    SRCategoryRateCorrect = 1 if SRTargetCategoryRate == input_field_data['category_count'] else 0
                elif 'LP' in task or 'LC' in task:
                    SRCategoryRateCorrect = 1 if SRTargetCategoryRate <= input_field_data['category_count'] else 0
                else:
                    raise NotImplementedError
                self.metrics_dict[task][f'SRCategoryRateCorrect@{self.topk}'] += SRCategoryRateCorrect

        if 'RewardSum' in self.metrics_dict[task]:
            self.metrics_dict[task]['RewardSum'] += list_reward

        if f'CategoryRateCorrect@{self.topk}' in self.metrics_dict[task]:
            if 'MP' in task or 'MC' in task:
                CategoryRateCorrect = 1 if TargetCategoryRate >= input_field_data['category_count'] else 0
            elif 'EP' in task or 'EC' in task:
                # CategoryRateCorrect = 1 if TargetCategoryRate == input_field_data['category_count'] else 0
                CategoryRateCorrect = 1 if abs(TargetCategoryRate - input_field_data['category_count']) <= 1 else 0
            elif 'LP' in task or 'LC' in task:
                CategoryRateCorrect = 1 if TargetCategoryRate <= input_field_data['category_count'] else 0
            else:
                raise NotImplementedError
            self.metrics_dict[task][f'CategoryRateCorrect@{self.topk}'] += CategoryRateCorrect

        if 'candidate_items' in input_field_data and f'NotInCandidateRate@{self.topk}' in self.metrics_dict[task]:
            NotInCandidateRate = sum([1 if _ not in input_field_data['candidate_titles'] else 0 for _ in _output_titles])
            self.metrics_dict[task][f'NotInCandidateRate@{self.topk}'] += NotInCandidateRate

    def __getitem__(self, item):
        return self.metrics_dict[item]

    def __iter__(self):
        return iter(self.metrics_dict.keys())

    def print(self):
        temp = copy.deepcopy(self.metrics_dict)
        temp = {_: {__: f'{temp[_][__]/temp[_]["Count"]:.4f}' if __ != 'Count' else temp[_][__] for __ in temp[_]} for _ in temp}
        tasks = [_ for _ in temp if _ != 'RLHFTotal']
        metrics = [f'NonExistRate@{self.topk}',
                   f'RepeatRate@{self.topk}',
                   f'InHistoryRate@{self.topk}',
                   f'CorrectCount@{self.topk}',
                   'Count',
                   f'Recall@{self.topk}',
                   f'MRR@{self.topk}',
                   f'NDCG@{self.topk}',
                   f'TargetCategoryRate@{self.topk}',
                   f'SRTargetCategoryRate@{self.topk}',
                   f'RewardSum',
                   f'CategoryRateCorrect@{self.topk}',
                   f'NotInCandidateRate@{self.topk}',
                   f'SRCategoryRateCorrect@{self.topk}',
                   ]
        table_rows = [f"|{_.center(24)}|{'|'.join([str(temp[__][_]).center(len(__)+4) if _ in temp[__] else '/'.center(len(__)+4) for __ in tasks])}|" for _ in metrics]
        table_rows_str = '\n'.join(table_rows)
        print(f'''
-{'tasks'.center(24, '-')}-{'-'.join([_.center(len(_)+4, '-') for _ in tasks])}-
{table_rows_str}
-{'-' * 24}-{'-'.join(['-' * (len(_)+4) for _ in tasks])}-''')
