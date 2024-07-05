import os.path
import re
import sys
import time

from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from transformers import pipeline, StoppingCriteriaList, MaxLengthCriteria
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from SFT.SFT_dataloader import SFTDataset, Train_task_group_mapping, Val_task_group_mapping, Test_task_group_mapping
from visdom import Visdom
from torch.utils.tensorboard import SummaryWriter
from SFT.SFT_templates import *
from actor_critic import ActorCritic
from metrics import Metrics
from trainer import Trainer
from utils import *
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from accelerate.utils import GradientAccumulationPlugin
from CrossEntropyLoss import CrossEntropyLoss_e


class SFTTrainer(Trainer):
    def __init__(self, args):
        super(SFTTrainer, self).__init__(args)
        self.args = args
        self.accelerator = Accelerator(gradient_accumulation_steps=self.args.gradient_accumulation_steps)
        set_seed(self.args.seed)
        self.args.gpu = self.args.gpu or self.accelerator.device
        self.args.is_main_process = self.accelerator.is_main_process
        self.actor_critic = ActorCritic(args=self.args, device=self.args.gpu)
        if self.accelerator.is_main_process:
            print(args)
            self.actor_critic.print_trainable_parameters()

        self.data = {
            'category': load_pickle(args.data_path + 'category.pickle'),
            # {
            #     'category': [item_ids]
            # }
            'metas': load_pickle(args.data_path + 'metas.pickle'),
            # {
            #     'item_id': {'item_id', 'title'}
            # }
            'sequential': load_pickle(args.data_path + 'sequential.pickle'),
            # {
            #     'user_id': [item_ids]
            # }
            # 'preference': load_pickle(args.data_path + 'preference.pickle'),
            # 'intention': load_pickle(args.data_path + 'intention.pickle'),
            'share_chat_gpt': load_pickle('/home/wangshuo/codes/scope-rec/data/dataset/share_chat_gpt2.pickle'),
            'ranking_candidate': load_pickle(args.data_path + 'ranking_candidate.pickle'),
            'item_list': load_pickle(args.data_path + 'items.pickle'),
        }
        self.sft_loss_fct = CrossEntropyLoss(reduction='none')
        self.sft_loss_e = CrossEntropyLoss_e()
        if self.args.lower:
            for _ in self.data['metas']:
                self.data['metas'][_]['title'] = self.data['metas'][_]['title'].lower().strip()
                self.data['metas'][_][self.args.item_index] = self.data['metas'][_][self.args.item_index].lower().strip()
        self.start_epoch = self.actor_critic.load_parameters(self.args.SFT_load)

    def SFT_Loss(self, logits, labels, input_data=None,task_list=None):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        if self.args.use_scope_mask and task_list[0] == 'SFTSeqRec':
            scope_mask = self.get_scope_mask(input_data['input_ids']) # 查询对应的scope-mask
            neg_inf = float('-inf')
            shift_scope_mask = scope_mask[..., 1:, :].contiguous()
            shift_logits[shift_scope_mask] = neg_inf # 将超出scope-mask 之外的logits设置成负无穷
        loss = self.sft_loss_fct(shift_logits.view(-1, self.actor_critic.model_config.vocab_size), shift_labels.view(-1)) #计算交叉熵损失函数
        loss = loss.view(labels.shape[0], -1)
        loss = loss.sum(dim=1) / (shift_labels != -100).sum(dim=1)  # [bs]
        return loss

    def SFT_loss_edit(self, logits, labels, input_data=None,task_list=None):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        if self.args.use_scope_mask and task_list[0] == 'SFTSeqRec':
            scope_mask = self.get_scope_mask(input_data['input_ids'])  # 查询对应的scope-mask
            neg_inf = float('-inf')
            shift_scope_mask = scope_mask[..., 1:, :].contiguous()
            shift_logits[shift_scope_mask] = neg_inf  # 将超出scope-mask 之外的logits设置成负无穷
            loss = self.sft_loss_fct(shift_logits.view(-1, self.actor_critic.model_config.vocab_size), shift_labels.view(-1))
            loss = self.sft_loss_e(shift_logits.view(-1, self.actor_critic.model_config.vocab_size),
                                   shift_labels.view(-1), shift_scope_mask.view(-1, self.actor_critic.model_config.vocab_size))
            loss = loss.view(-1)
        else:
            loss = self.sft_loss_fct(shift_logits.view(-1, self.actor_critic.model_config.vocab_size), shift_labels.view(-1))  # 计算交叉熵损失函数
            loss = loss.view(labels.shape[0], -1)
            loss = loss.sum(dim=1) / (shift_labels != -100).sum(dim=1)  # [bs]

        return loss


    def SFT_LF_Loss(self, logits, labels, input_data=None,task_list=None):# 这里考虑到task是基于batch的大小是1的情况，如果不是要进行调整
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        start_positions = (shift_labels == 32002).nonzero(as_tuple=True)[0]
        end_positions = (shift_labels == 32000).nonzero(as_tuple=True)[0]

        # 创建全为False的掩码，长度与labels相同
        mask = torch.zeros_like(shift_labels, dtype=torch.bool)

        # 对每一对开始和结束位置，更新掩码以标记为True, 先不包括控制符
        for start, end in zip(start_positions, end_positions):
            mask[start+1:end] = True

        # 使用掩码分割logits为两部分
        shift_logits1 = logits[mask]
        shift_logits2 = logits[~mask]

        # 同理，分割labels
        shift_labels1 = labels[mask]
        shift_labels2 = labels[~mask]

        # 两个损失函数 交叉熵损失函数和 focal loss
        loss1 = CrossEntropyLoss(reduction='none')

        loss = self.sft_loss_fct(shift_logits.view(-1, self.actor_critic.model_config.vocab_size), shift_labels.view(-1))
        loss = loss.view(labels.shape[0], -1)
        loss = loss.sum(dim=1) / (shift_labels != -100).sum(dim=1)  # [bs]
        return loss


    def SFT_train(self):
        TaskTemplate = {_: Train_task_group_mapping[_] for _ in self.args.SFT_train_tasks.split(',')}
        TaskNum = {_: 1 for _ in self.args.SFT_train_tasks.split(',')}
        ValTaskTemplate = {_: Val_task_group_mapping[_.split('_')[0]] for _ in self.args.SFT_val_tasks.split(',')}
        ValTaskNum = {_: 1 for _ in self.args.SFT_val_tasks.split(',')}
        with self.accelerator.main_process_first():
            train_data = SFTDataset(self.args, TaskTemplate, TaskNum, self.data, self.tokenizer, 'train')
            val_data = SFTDataset(self.args, ValTaskTemplate, ValTaskNum, self.data, self.tokenizer, 'val')
            self.get_scope_mask = train_data.get_scope_mask

        train_loader = DataLoader(train_data, batch_size=self.args.batch_size, shuffle=True, collate_fn=train_data.collate_fn)
        val_loader = DataLoader(val_data, batch_size=self.args.val_batch_size, shuffle=False, collate_fn=val_data.collate_fn, drop_last=False)

        SFT_optim = self.get_optimizer(self.actor_critic.actor_parameters)
        batch_per_epoch = len(train_loader)
        step_total = batch_per_epoch * (self.args.epoch - self.start_epoch) // self.args.gradient_accumulation_steps
        warmup_iters = int(step_total * self.args.warmup_ratio)
        # SFT_lr_scheduler = get_linear_schedule_with_warmup(SFT_optim, warmup_iters, step_total)
        SFT_lr_scheduler = get_polynomial_decay_schedule_with_warmup(SFT_optim, warmup_iters, step_total, power=2.0)
        warp_actor_critic, SFT_optim, train_loader, val_loader, SFT_lr_scheduler = self.accelerator.prepare(
            self.actor_critic, SFT_optim, train_loader, val_loader, SFT_lr_scheduler
        )
        # print(SFT_actor_parameters)
        writer = None
        if self.accelerator.is_main_process:
            name = self.args.output.split('snap/')[-1]
            writer = SummaryWriter(log_dir=f'logs/SFT_train/{self.args.SFT_train_tasks}/{name}', flush_secs=30)
        
        if self.args.dry:
            self.SFT_evl_inference(self.start_epoch, val_loader, writer)
        best_val_loss = 1e10
        for epoch in range(self.start_epoch+1, self.args.epoch+1):
            # test save
            # self.actor_critic.save_all_weight(epoch)
            # Train
            task_loss = {_: 0.0 for _ in train_data.task_num}
            task_count = {_: 1e-10 for _ in train_data.task_num}
            # window_loss = {_: [0.0]*16 for _ in train_data.task_num}
            pbar = tqdm(total=len(train_loader), ncols=210, disable=not self.accelerator.is_local_main_process)
            self.train()
            for step_i, batch in enumerate(train_loader):
                with self.accelerator.accumulate(warp_actor_critic):
                    # print(f'parameter {step_i}: ', self.actor_critic.actor_parameters[0].data.abs().max())
                    # self.accelerator.wait_for_everyone()
                    input_data = batch['complete_text_data']
                    if self.accelerator.is_main_process and step_i % 10000 == 0:
                        print(batch['complete_text'][0])
                        print(input_data['input_ids'][0])
                    labels = batch['complete_label_ids']
                    results = warp_actor_critic.forward(scope='actor', **input_data)
                    # loss = self.SFT_Loss(results.logits, labels, input_data, batch['task'])
                    loss = self.SFT_loss_edit(results.logits, labels, input_data, batch['task'])

                    for idx, task in enumerate(batch['task']):
                        task_loss[task] += (float(loss[idx]))
                        # window_loss[task] = window_loss[task][1:] + [float(loss[idx])]
                        task_count[task] += 1

                    self.accelerator.backward(loss.mean())  # auto divide accumulate step, sync grad if arrive accumulate step
                    # print(f'grad {step_i}: ', self.actor_critic.actor_parameters[0].grad.abs().max())
                    # self.accelerator.wait_for_everyone()

                    if self.accelerator.sync_gradients:
                        # print(f'sync grad {step_i}: ', self.actor_critic.actor_parameters[0].grad.abs().max())
                        # self.accelerator.wait_for_everyone()
                        if self.args.clip_grad_norm > 0:
                            total_norm = self.accelerator.clip_grad_norm_(SFT_optim.param_groups[0]['params'], self.args.clip_grad_norm)
                            # writer.add_scalars('training/total_norm', {f'epoch{epoch}': float(total_norm)}, step_i)

                    SFT_optim.step()
                    SFT_lr_scheduler.step()
                    SFT_optim.zero_grad()

                if self.accelerator.sync_gradients:
                    losses = torch.tensor([_ for _ in task_loss.values()], device=self.accelerator.device)
                    counts = torch.tensor([_ for _ in task_count.values()], device=self.accelerator.device)
                    losses = self.accelerator.reduce(losses)  # [task_num]
                    counts = self.accelerator.reduce(counts)  # [task_num]
                    if self.accelerator.is_main_process:
                        for idx, task in enumerate(list(task_loss.keys())):
                            writer.add_scalars(f'training/{task}_Loss', {f'epoch{epoch}': losses[idx] / counts[idx]}, counts[idx])
                        ShareChatGPT_mask = torch.tensor(
                            [1.0 if _ != 'ShareChatGPT' else 0.0 for _ in task_loss.keys()],
                            device=self.accelerator.device
                        )
                        writer.add_scalars('training/All_Loss', {f'epoch{epoch}': float(masked_mean(losses/counts, ShareChatGPT_mask))}, step_i)
                        desc_str = f'E{epoch} | LR {SFT_lr_scheduler.get_lr()[0]:.4f}' \
                                   f' | {" | ".join([f"{task}: {losses[idx] / counts[idx]:.4f}" for idx, task in enumerate(list(task_loss.keys()))])}'
                        pbar.set_description(desc_str, refresh=False)
                        pbar.update(self.args.gradient_accumulation_steps)

                    # GPU_M = torch.cuda.memory_allocated(device=self.args.gpu) + torch.cuda.memory_reserved(
                    #     device=self.args.gpu)
                    # desc_str = f'Epoch {epoch} | LR {lr:.6f}' \
                    #            f' | {" | ".join([f"{task}_Loss: {losses[idx]/counts[idx]:.4f}" for idx, task in enumerate(list(task_loss.keys()))])}'
                    # pbar.set_description(desc_str, refresh=False)
                    # pbar.update(1)
            pbar.close()
            # self.accelerator.print(f'Epoch {epoch} | {" | ".join([f"{task}_Loss: {losses[idx]/counts[idx]:.4f}" for idx, task in enumerate(list(task_loss.keys()))])}'
            #                        f' | SFT_Train_Loss: {float((losses/counts).mean()):.4f}')
            # self.accelerator.print(f'Epoch {epoch} | {" | ".join([f"{_}_Window_Loss: {sum(__)/len(__):.4f}" for _, __ in window_loss.items()])}'
            #                        f' | SFT_Train_Window_Loss: {sum([sum(__)/len(__) for _, __ in window_loss.items()])/len(window_loss):.4f}')
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                # if epoch == 1 or epoch %10 ==0 or epoch == self.args.epoch:
                #     self.actor_critic.save_all_weight(epoch)

                self.actor_critic.save_parameters("Epoch%02d" % epoch)
            if epoch < self.args.val_epoch:
                continue
            val_loss = self.SFT_evl_inference(epoch, val_loader, writer)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.actor_critic.save_parameters("BEST_EVAL_LOSS")
                # self.actor_critic.save_all_weight("BEST_EVAL_LOSS")

    def SFT_evl(self, epoch, writer):
        torch.cuda.empty_cache()
        self.eval()
        ValTaskTemplate = {_: Val_task_group_mapping[_] for _ in self.args.SFT_val_tasks.split(',')}
        ValTaskNum = {_: 1 for _ in self.args.SFT_val_tasks.split(',')}
        with self.accelerator.main_process_first():
            val_data = SFTDataset(self.args, ValTaskTemplate, ValTaskNum, self.data, self.tokenizer, 'val')
        val_loader = DataLoader(val_data, batch_size=self.args.val_batch_size, shuffle=False, collate_fn=val_data.collate_fn, drop_last=False)

        task_loss = {_: 0.0 for _ in val_data.task_num}
        task_count = {_: 1e-10 for _ in val_data.task_num}
        with torch.no_grad():
            pbar = tqdm(total=len(val_loader), ncols=200, disable=not self.accelerator.is_local_main_process)
            for step_i, batch in enumerate(val_loader):
                input_data = batch['complete_text_data']
                if self.accelerator.is_main_process and step_i % 10000 == 0:
                    print(batch['complete_text'][0])
                    print(input_data['input_ids'][0])
                labels = batch['complete_label_ids']
                results = self.actor_critic.forward(scpoe='actor', **input_data)
                loss = self.SFT_Loss(results.logits, labels).detach()
                for idx, task in enumerate(batch['task']):
                    task_loss[task] += (float(loss[idx]))
                    task_count[task] += 1
                # GPU_M = torch.cuda.memory_allocated(device=self.args.gpu) + torch.cuda.memory_reserved(device=self.args.gpu)
                # desc_str = f'Epoch {epoch} | {" | ".join([f"{_}_Loss: {__/task_count[_]:.4f}" for _, __ in task_loss.items()])}'
                # pbar.set_description(desc_str, refresh=False)
                pbar.update(1)
            pbar.close()

        losses = torch.tensor([_ for _ in task_loss.values()], device=self.accelerator.device)
        counts = torch.tensor([_ for _ in task_count.values()], device=self.accelerator.device)
        losses = self.accelerator.reduce(losses)  # [task_num]
        counts = self.accelerator.reduce(counts)  # [task_num]
        val_loss = float((losses/counts).mean())
        if self.accelerator.is_main_process:
            print(f'Epoch {epoch} | {" | ".join([f"Val_{task}_Loss: {losses[idx]/counts[idx]:.4f}" for idx, task in enumerate(list(task_loss.keys()))])}')
            print(f'Epoch {epoch} | SFT_Val_Loss: {val_loss:.4f}\n')
            writer.add_scalars(f'valuating', {f'{task}_Loss': losses[idx]/counts[idx] for idx, task in enumerate(list(task_loss.keys()))}, epoch)
            writer.add_scalars(f'valuating', {'total_Loss': val_loss}, epoch)
        self.train()
        return val_loss

    def SFT_evl_inference(self, epoch, val_loader, writer):
        torch.cuda.empty_cache()
        self.eval()
        stopping_criteria = StoppingCriteriaList(
            [MaxLengthCriteria(max_length=self.args.max_token_length + self.args.gen_max_length)]
        )
        metrics_dict = Metrics(self.args.SFT_val_tasks.split(','), self.args.topk, val_loader.dataset.category2item,
                               val_loader.dataset.title2item)
        pbar = tqdm(total=len(val_loader), ncols=200, disable=not self.accelerator.is_local_main_process)
        for step_i, batch in enumerate(val_loader):
            bs = len(batch['task'])
            input_data = batch['input_data']
            if self.accelerator.is_main_process and step_i % 1000 == 0:
                print(batch['input_text'][0])
                print(input_data['input_ids'][0])
            input_ids_length = input_data['input_ids'].shape[1]

            output_labels = [[__ for __ in _.strip().split('\n')] for _ in batch['output_text']]
            with torch.no_grad():
                if epoch == 0:
                    output_ids = self.actor_critic.base_model.greedy_search(
                        **input_data,
                        stopping_criteria=stopping_criteria,
                    )
                else:
                    output_ids = self.actor_critic.actor_model.greedy_search(
                        **input_data,
                        stopping_criteria=stopping_criteria,
                    )
                output_title = self.tokenizer.batch_decode(output_ids[:, input_ids_length:], skip_special_tokens=True)
                output_title_list = [
                    [__.strip() for __ in _.strip().split('\n')] for _ in output_title
                ]
                if self.args.idx:
                    output_labels = [[rm_idx(__) for __ in _] for _ in output_labels]
                    output_title_list = [[rm_idx(__) for __ in _] for _ in output_title_list]
            for i in range(bs):
                task = batch['task'][i]
                metrics_dict.add_sample(task, batch['input_field_data'][i], output_title_list[i], output_labels[i])
            pbar.update(1)
        pbar.close()

        _ndcg, _non_exist_rate, _repeat_rate, _correct_count = 0.0, 0.0, 0.0, 0.0
        for task in metrics_dict.metrics_dict:
            task_count = metrics_dict[task]['Count']
            recall = metrics_dict[task][f'Recall@{metrics_dict.topk}']
            ndcg = metrics_dict[task][f'NDCG@{metrics_dict.topk}']
            non_exist_rate = metrics_dict[task][f'NonExistRate@{metrics_dict.topk}']
            repeat_rate = metrics_dict[task][f'RepeatRate@{metrics_dict.topk}']
            correct_count = metrics_dict[task][f'CorrectCount@{metrics_dict.topk}']

            if task == 'SFTTestPersonalCategoryRate':
                category_rate_correct = metrics_dict[task][f'CategoryRateCorrect@{metrics_dict.topk}']
                log_d = torch.tensor(
                    [task_count, recall, ndcg, non_exist_rate, repeat_rate, correct_count, category_rate_correct],
                    device=self.accelerator.device)
            else:
                log_d = torch.tensor(
                    [task_count, recall, ndcg, non_exist_rate, repeat_rate, correct_count],
                    device=self.accelerator.device)
            log_d = self.accelerator.reduce(log_d)
            with self.accelerator.main_process_first():
                print(log_d)

            _ndcg += log_d[2] / log_d[0]
            _non_exist_rate += log_d[3] / log_d[0]
            _repeat_rate += log_d[4] / log_d[0]
            _correct_count += log_d[5] / log_d[0]

            if self.accelerator.is_main_process:
                writer.add_scalar(f'valuating/{task}_Recall', log_d[1] / log_d[0], epoch)
                writer.add_scalar(f'valuating/{task}_NDCG', log_d[2] / log_d[0], epoch)
                writer.add_scalar(f'valuating/{task}_NonExist_rate', log_d[3] / log_d[0], epoch)
                writer.add_scalar(f'valuating/{task}_Repeat_rate', log_d[4] / log_d[0], epoch)
                writer.add_scalar(f'valuating/{task}_Correct_count', log_d[5] / log_d[0], epoch)
                if task == 'RLHFPersonalCategoryRate':
                    writer.add_scalar(f'valuating/{task}_Category_rate_correct', log_d[6] / log_d[0], epoch)
        if self.accelerator.is_main_process:
            val_task_num = len(val_loader.dataset.task_num)
            writer.add_scalar(f'valuating/Total_NDCG', _ndcg / val_task_num, epoch)
            writer.add_scalar(f'valuating/Total_NonExist_rate', _non_exist_rate / val_task_num, epoch)
            writer.add_scalar(f'valuating/Total_Repeat_rate', _repeat_rate / val_task_num, epoch)
            writer.add_scalar(f'valuating/Total_Correct_count', _correct_count / val_task_num, epoch)
            print(f'Epoch {epoch} | SFT_Val_NDCG: {_ndcg:.4f}\n')
        self.train()
        return -_ndcg

    def SFT_test(self):
        torch.cuda.empty_cache()
        self.eval()
        TestTaskTemplate = {self.args.SFT_test_task: Test_task_group_mapping[self.args.SFT_test_task.split('_')[0]]}
        TestTaskNum = {self.args.SFT_test_task: 1}
        stopping_criteria = StoppingCriteriaList(
            [MaxLengthCriteria(max_length=self.args.max_token_length + self.args.gen_max_length)]
        )
        if self.args.SFT_test_task in ['SFT+TestPersonalControlRec', 'SFT-TestPersonalControlRec'] or self.args.SFT_test_task.startswith('SFTTestPersonalCategoryRate'):
            TestSeqRec_Result_file = self.args.output + f'Epoch{self.start_epoch:02d}_SFT_Result_SFTTestSeqRec_Top{self.args.topk}.pickle'
            self.data['SFTTestSeqRec_Result'] = load_pickle(TestSeqRec_Result_file)
        # with self.accelerator.main_process_first():
        test_data = SFTDataset(self.args, TestTaskTemplate, TestTaskNum, self.data, self.tokenizer, 'test')
        test_loader = DataLoader(test_data, batch_size=self.args.test_batch_size, shuffle=False,
                                 collate_fn=test_data.collate_fn, drop_last=False)

        # warp_actor_critic, test_loader = self.accelerator.prepare(self.actor_critic, test_loader)
        metrics_dict = Metrics([self.args.SFT_test_task], self.args.topk, test_data.category2item, test_data.title2item)
        # model = warp_actor_critic.module
        result_file = self.args.backbone+f'Result_{self.args.SFT_test_task}{"_CBS" if self.args.use_CBS else ""}_Top{self.args.topk}.pickle'
        if self.args.SFT_load:
            result_file = self.args.SFT_load + \
                          f'_Result_{self.args.SFT_test_task}{"_CBS" if self.args.use_CBS else ""}_Top{self.args.topk}' \
                          f'{("_process_"+str(self.accelerator.local_process_index)) if self.accelerator.num_processes>1 else ""}.pickle'

        def constrain_search_list(batch_id, input_ids):
            # 1、如果识别出来是控制符，我们就返回限制下的token list
            try:
                has_ctrl,prefix_input_ids = get_prefix(input_ids,test_data.ctrl_symbols) #  control_symbol 是一个数组，[s,e]
                if  has_ctrl:
                    next_tokens =test_data.item_prefix_tree.next_tokens(prefix_input_ids)
                    # next_tokens =test_data.item_prefix_tree.next_tokens_matrix(prefix_input_ids,len(tokenizer))
                    if len(next_tokens) != 0:
                        # print("CBS搜索过程,返回从前缀树检索next_tokens")
                        return next_tokens

                # 2、如果没有控制符我们就返回全词典的token list
                # print("CBS搜索过程,batch id {} 返回全词典".format(batch_id))
                return list(test_data.tokenizer.get_vocab().values())
            except Exception as e:
                print("CBS搜索过程中出现问题，batch id 为{}".format(batch_id))
                print("这个step的输入 id 为 {}".format(input_ids))
                print("错误是：{}".format(e))
                return list(test_data.tokenizer.get_vocab().values())

        with torch.no_grad():
            result = load_pickle(result_file) or []
            result = result[:-(len(result) % self.args.batch_size) if (len(result) % self.args.batch_size) != 0 else None]
            pbar = tqdm(total=len(test_loader), ncols=150, disable=not self.accelerator.is_local_main_process)
            for step_i, batch in enumerate(test_loader):
                bs = len(batch['task'])
                input_data = batch['input_data']
                if self.accelerator.is_main_process and step_i % 10000 == 0:
                    print(batch['input_text'][0])
                    print(input_data['input_ids'][0])
                output_labels = [[__.strip() for __ in _.strip().split('\n')] for _ in batch['output_text']]
                if self.args.idx:
                    output_labels = [[rm_idx(__) for __ in _] for _ in output_labels]
                input_ids_length = input_data['input_ids'].shape[1]
                if (step_i+1)*self.args.test_batch_size <= len(result):
                    output_title_list = [_[1] for _ in result[step_i*self.args.test_batch_size: (step_i+1)*self.args.test_batch_size]]
                else:
                    # output_ids = self.actor_critic.actor_model.greedy_search(
                    #     **input_data,
                    #     prefix_allowed_tokens_fn=constrain_search_list if self.args.use_CBS else None,
                    #     stopping_criteria=stopping_criteria,
                    # )
                    # output_ids = model.actor_model.generate(
                    output_ids = self.actor_critic.actor_model.generate(
                        **input_data,
                        prefix_allowed_tokens_fn=constrain_search_list if self.args.use_CBS else None,
                        max_length=self.args.max_token_length + self.args.gen_max_length,
                        num_beams=1,
                        num_return_sequences=1,
                    )
                    # output_title = self.tokenizer.batch_decode(output_ids[:, input_ids_length:], skip_special_tokens= False if self.args.use_CBS else True)
                    output_title = self.tokenizer.batch_decode(output_ids[:, input_ids_length:],
                                                               skip_special_tokens=False)
                    if step_i % 10000 == 0:
                        print(output_title[0])
                    if self.args.use_CBS:
                        output_title_list = [get_ctrl_item(_) for _ in output_title]
                    else:
                        print(output_title)
                        output_title_list = [get_ctrl_item(_) for _ in output_title]
                        # output_title_list = [[__.strip() for __ in _.strip().split('\n')] for _ in output_title]
                        if self.args.idx:
                            output_title_list = [[rm_idx(__) for __ in _] for _ in output_title_list]
                    result += [[_, __] for _, __ in zip(output_labels, output_title_list)]
                    if step_i % 100 == 0 or (step_i + 1) == len(test_loader):
                        save_pickle(result, result_file)

                for i in range(bs):
                    metrics_dict.add_sample(batch['task'][i], batch['input_field_data'][i], output_title_list[i], output_labels[i])
                metrics_dict.print()
                pbar.update(1)
            # self.accelerator.wait_for_everyone()
            # if self.accelerator.num_processes > 1 and self.accelerator.is_main_process:
            #     # 合并不同进程推理的结果
            #     merge_path_patten = self.args.SFT_load + \
            #                   f'_Result_{self.args.SFT_test_task}{"_CBS" if self.args.use_CBS else ""}_Top{self.args.topk}{"_process_*"}.pickle'
            #     print(f"pattern {merge_path_patten}")
            #     save_path = self.args.SFT_load + \
            #                   f'_Result_{self.args.SFT_test_task}{"_CBS" if self.args.use_CBS else ""}_Top{self.args.topk}.pickle'
            #     result = merge_result(merge_path_patten)
            #     save_pickle(result, save_path)
            pbar.close()
        self.train()

    def SFT_test_pipline(self):
        torch.cuda.empty_cache()
        self.eval()
        TestTaskTemplate = {self.args.SFT_test_task: Test_task_group_mapping[self.args.SFT_test_task.split('_')[0]]}
        TestTaskNum = {self.args.SFT_test_task: 1}
        stopping_criteria = StoppingCriteriaList(
            [MaxLengthCriteria(max_length=self.args.max_token_length + self.args.gen_max_length)]
        )
        if self.args.SFT_test_task in ['SFT+TestPersonalControlRec', 'SFT-TestPersonalControlRec'] or self.args.SFT_test_task.startswith('SFTTestPersonalCategoryRate'):
            TestSeqRec_Result_file = self.args.output + f'Epoch{self.start_epoch:02d}_SFT_Result_SFTTestSeqRec_Top{self.args.topk}.pickle'
            self.data['SFTTestSeqRec_Result'] = load_pickle(TestSeqRec_Result_file)
        # with self.accelerator.main_process_first():
        test_data = SFTDataset(self.args, TestTaskTemplate, TestTaskNum, self.data, self.tokenizer, 'test')
        test_loader = DataLoader(test_data, batch_size=self.args.test_batch_size, shuffle=False,
                                 collate_fn=test_data.collate_fn, drop_last=False)

        # warp_actor_critic, test_loader = self.accelerator.prepare(self.actor_critic, test_loader)
        metrics_dict = Metrics([self.args.SFT_test_task], self.args.topk, test_data.category2item, test_data.title2item)
        # model = warp_actor_critic.module
        result_file = self.args.backbone+f'Result_{self.args.SFT_test_task}{"_CBS" if self.args.use_CBS else ""}_Top{self.args.topk}.pickle'
        if self.args.SFT_load:
            result_file = self.args.SFT_load + \
                          f'_Result_{self.args.SFT_test_task}_Pipeline_Top{self.args.topk}' \
                          f'{("_process_"+str(self.accelerator.local_process_index)) if self.accelerator.num_processes>1 else ""}.pickle'

        def constrain_search_list(batch_id, input_ids):
            # 1、如果识别出来是控制符，我们就返回限制下的token list
            try:
                has_ctrl,prefix_input_ids = get_prefix(input_ids,test_data.ctrl_symbols) #  control_symbol 是一个数组，[s,e]
                if  has_ctrl:
                    next_tokens =test_data.item_prefix_tree.next_tokens(prefix_input_ids)
                    # next_tokens =test_data.item_prefix_tree.next_tokens_matrix(prefix_input_ids,len(tokenizer))
                    if len(next_tokens) != 0:
                        # print("CBS搜索过程,返回从前缀树检索next_tokens")
                        return next_tokens

                # 2、如果没有控制符我们就返回全词典的token list
                # print("CBS搜索过程,batch id {} 返回全词典".format(batch_id))
                return list(test_data.tokenizer.get_vocab().values())
            except Exception as e:
                print("CBS搜索过程中出现问题，batch id 为{}".format(batch_id))
                print("这个step的输入 id 为 {}".format(input_ids))
                print("错误是：{}".format(e))
                return list(test_data.tokenizer.get_vocab().values())

        with torch.no_grad():
            result = load_pickle(result_file) or []
            result = result[:-(len(result) % self.args.batch_size) if (len(result) % self.args.batch_size) != 0 else None]
            pbar = tqdm(total=len(test_loader), ncols=150, disable=not self.accelerator.is_local_main_process)
            for step_i, batch in enumerate(test_loader):
                bs = len(batch['task'])
                input_data = batch['input_data']
                if self.accelerator.is_main_process and step_i % 10000 == 0:
                    print(batch['input_text'][0])
                    print(input_data['input_ids'][0])
                output_labels = [[__.strip() for __ in _.strip().split('\n')] for _ in batch['output_text']]
                if self.args.idx:
                    output_labels = [[rm_idx(__) for __ in _] for _ in output_labels]
                input_ids_length = input_data['input_ids'].shape[1]
                # 添加一个判断，只有预测所有的item都是scope内才跳出去
                is_over = False

                scope_title_list =[]
                while not is_over:
                    if (step_i+1)*self.args.test_batch_size <= len(result):
                        output_title_list = [_[1] for _ in result[step_i*self.args.test_batch_size: (step_i+1)*self.args.test_batch_size]]
                    else:
                        output_ids = self.actor_critic.actor_model.generate(
                            **input_data,
                            prefix_allowed_tokens_fn=constrain_search_list if self.args.use_CBS else None,
                            max_length=self.args.max_token_length + self.args.gen_max_length,
                            num_beams=1,
                            num_return_sequences=1,
                        )
                        output_title = self.tokenizer.batch_decode(output_ids[:, input_ids_length:], skip_special_tokens= False if self.args.use_CBS else True)
                        if step_i % 10000 == 0:
                            print(output_title[0])
                        if self.args.use_CBS:
                            output_title_list = [get_ctrl_item(_) for _ in output_title]
                        else:
                            output_title_list = [[__.strip() for __ in _.strip().split('\n')] for _ in output_title]
                            if self.args.idx:
                                output_title_list = [[rm_idx(__) for __ in _] for _ in output_title_list]
                            is_over = True
                            for index, title_list in enumerate(output_title_list):
                                filter_list = list(filter(lambda x: x in self.data['item_list'], title_list))
                                # if len(filter_list)<10:
                                #     is_over = False
                                if index >= len(scope_title_list):
                                    scope_title_list.append(filter_list)
                                elif len(scope_title_list[index]) < 10:
                                    append_number = 10 - len(scope_title_list[index])
                                    append_number = min(append_number, len(filter_list))
                                    scope_title_list[index].extend(filter_list[:append_number])
                                if len(scope_title_list[index]) < 10:
                                    is_over = False

                result += [[_, __] for _, __ in zip(output_labels, scope_title_list)]
                # result += [[_, __] for _, __ in zip(output_labels, output_title_list)]
                if step_i % 100 == 0 or (step_i + 1) == len(test_loader):
                    save_pickle(result, result_file)

                for i in range(bs):
                    metrics_dict.add_sample(batch['task'][i], batch['input_field_data'][i], output_title_list[i], output_labels[i])
                metrics_dict.print()
                pbar.update(1)
            pbar.close()
        self.train()
    # def SFT_test_pipeline(self):
    #     torch.cuda.empty_cache()
    #     self.eval()
    #     TestTaskTemplate = {self.args.SFT_test_task.split('_')[0]: Test_task_group_mapping[self.args.SFT_test_task.split('_')[0]]}
    #     TestTaskNum = {self.args.SFT_test_task.split('_')[0]: 1}
    #
    #     if self.args.SFT_test_task.split('_')[0] in ['SFT+TestPersonalControlRec', 'SFT-TestPersonalControlRec', 'SFTTestPersonalCategoryRate']:
    #         TestSeqRec_Result_file = self.args.output + f'Epoch{self.start_epoch:02d}_SFT_Result_SFTTestSeqRec.pickle'
    #         self.data['SFTTestSeqRec_Result'] = load_pickle(TestSeqRec_Result_file)
    #     test_data = SFTDataset(self.args, TestTaskTemplate, TestTaskNum, self.data, self.tokenizer, 'test')
    #     input_text = [_['input_text'] for _ in test_data]
    #     output_label = [_['output_text'] for _ in test_data]
    #     task = [_['task'] for _ in test_data]
    #     input_field_data = [_['input_field_data'] for _ in test_data]
    #     print(input_text[0])
    #     metrics_dict = Metrics([self.args.SFT_test_task.split('_')[0]], self.args.topk, test_data.category2item, test_data.title2item)
    #     result_file = self.args.backbone+f'Result_{self.args.SFT_test_task}.pickle'
    #     if self.args.SFT_load:
    #         result_file = self.args.SFT_load + f'_Result_{self.args.SFT_test_task}.pickle'
    #     with torch.no_grad():
    #         generator = pipeline("text-generation", model=self.actor_critic.actor_model, tokenizer=self.actor_critic.tokenizer)
    #         result = load_pickle(result_file) or []
    #         result = result[:-(len(result) % self.args.batch_size) if (len(result) % self.args.batch_size) != 0 else None]
    #         start_idx = len(result)
    #         for step_i, output_titles in enumerate(generator(input_text[start_idx:], batch_size=self.args.test_batch_size)):
    #             output_title_list = [[__.strip() for __ in _.strip().split('\n')] for _ in output_titles]
    #             bs = len(output_title_list)
    #             _output_label = output_label[start_idx+step_i: start_idx+step_i+bs]
    #             _task = task[start_idx+step_i: start_idx+step_i+bs]
    #             _input_field_data = input_field_data[start_idx+step_i: start_idx+step_i+bs]
    #             result += [[_, __] for _, __ in zip(_output_label, output_title_list)]
    #             if step_i % 100 == 0:
    #                 save_pickle(result, result_file)
    #
    #             for i in range(bs):
    #                 metrics_dict.add_sample(_task[i], _input_field_data[i], output_title_list[i], _output_label[i])
    #             metrics_dict.print()
    #         save_pickle(result, result_file)
    #     self.train()

    def SFT_test_GPT(self, epoch):
        self.eval()

        TestTaskTemplate = {
            "GPTTestPersonalControlRec": ValPersonalControlRec_group,
        }
        TestTaskNum = {
            "GPTTestPersonalControlRec": 1,
        }
        test_data = SFTDataset(self.args, TestTaskTemplate, TestTaskNum, self.data, self.tokenizer, 'test')
        test_loader = DataLoader(test_data, batch_size=self.args.batch_size, shuffle=False,
                                 collate_fn=test_data.collate_fn, drop_last=False)
        metrics = {
            f'NonExistRate@{self.args.topk}': 0.0,
            f'RepeatRate@{self.args.topk}': 0.0,
            f'CorrectCount@{self.args.topk}': 0.0,
        }
        count = 0
        result_file = self.args.output+f'Epoch{epoch}_Result_Generate_Intention.pickle'

        with torch.no_grad():
            result = load_pickle(result_file) or []
            result = result[:-(len(result) % self.args.batch_size) if (len(result) % self.args.batch_size) != 0 else None]
            pbar = tqdm(total=len(test_loader), ncols=200)
            for step_i, batch in enumerate(test_loader):
                bs = len(batch['task'])
                input_data = batch['input_data']
                output_labels = [[__.strip() for __ in _.strip().split('\n')] for _ in batch['output_text']]
                input_ids_length = input_data['input_ids'].shape[1]
                if step_i*bs+bs <= len(result):
                    output_title_list = [_[1] for _ in result[step_i*bs: step_i*bs+bs]]
                else:
                    output_ids = self.actor_critic.actor_generate(
                        **input_data,
                        max_length=input_ids_length + self.args.gen_max_length,
                        num_beams=1,
                        num_return_sequences=1,
                        early_stopping=True,
                        do_sample=False,
                    )
                    output_title = self.tokenizer.batch_decode(output_ids[:, input_ids_length:], skip_special_tokens=True)
                    output_title_list = [[__.strip() for __ in _.strip().split('\n')] for _ in output_title]
                    result += [[_, __] for _, __ in zip(output_labels, output_title_list)]
                    if step_i % 100 == 0 or (step_i + 1) == len(test_loader):
                        save_pickle(result, result_file)

                CorrectCount = sum([1 if len(_) == self.args.topk else 0 for _ in output_title_list])
                NonExistRate, RepeatRate = 0, 0
                for _ in output_title_list:
                    NonExistRate += sum([1 if __ not in self.SFT_test_loader.dataset.title2item else 0 for __ in _])
                    RepeatRate += sum([1 if __ in _[:idx] else 0 for idx, __ in enumerate(_)])
                metrics[f'NonExistRate@{self.args.topk}'] += NonExistRate/self.args.topk
                metrics[f'RepeatRate@{self.args.topk}'] += RepeatRate/self.args.topk
                metrics[f'CorrectCount@{self.args.topk}'] += CorrectCount
                count += bs

                GPU_M = torch.cuda.memory_allocated(device=self.args.gpu) + torch.cuda.memory_reserved(
                    device=self.args.gpu)
                desc_str = f'GPU {GPU_M / 1024 ** 3:.4f}G | count: {count}' \
                           f' | {" | ".join([f"{_}: {__ / count:.4f}" for _, __ in metrics.items()])}'
                pbar.set_description(desc_str, refresh=False)
                pbar.update(1)
            pbar.close()
        self.train()

    def cli(self):
        self.actor_critic.eval()
        with torch.no_grad():
            while True:
                # inp = "I want to buy some facial mask, please recommend me some product."
                inp = input()
                input_data = side_tokenizer([inp], 'left', self.tokenizer, return_tensors='pt').to(self.args.gpu)
                input_data.data['max_length'] = 512
                input_data.data['num_beams'] = 1
                input_data.data['no_repeat_ngram_size'] = 0
                input_data.data['num_return_sequences'] = 1
                input_data.data['early_stopping'] = True
                input_data.data['do_sample'] = False
                output_ids = self.actor_critic.actor_generate(**input_data.data)
                output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                print('\n'.join(output_text))

    def SFT_adapter_merge(self):
        model = self.actor_critic.lora_model.merge_and_unload(progressbar=True)
        model.save_pretrained(f'{self.args.output}SFT_Epoch{self.start_epoch:02d}', safe_serialization=True)
        self.tokenizer.save_pretrained(f'{self.args.output}SFT_Epoch{self.start_epoch:02d}')


if __name__ == "__main__":
    pass
