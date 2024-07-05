import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from functools import partial
from collections import deque, namedtuple
from typing import Deque
from transformers import get_linear_schedule_with_warmup, LogitsProcessorList, TopKLogitsWarper, \
    TemperatureLogitsWarper, StoppingCriteriaList, MaxLengthCriteria, MinLengthLogitsProcessor, BeamSearchScorer, \
    get_polynomial_decay_schedule_with_warmup
from torch.multiprocessing.queue import Queue
from RLHF.RLHF_dataloader import *
from RLHF.RLHF_template import *
from actor_critic import ActorCritic
from metrics import Metrics
from trainer import Trainer
from utils import *
from visdom import Visdom
import torch.multiprocessing as mp
# data
Memory = namedtuple('Memory', [
    'sequence',
    'action_mask',
    'old_action_value',
    'old_sequence_log_probs_shifted',
    'ref_sequence_log_probs_shifted',
    'whitened_advantages',
    'returns'
])


# rlhf trainer
class RLHFTrainer(Trainer):
    def __init__(self, args):
        super(RLHFTrainer, self).__init__(args)
        self.args = args
        self.actor_critic = ActorCritic(args=self.args, device=self.args.producer_gpu)
        self.consumer_actor_critic = ActorCritic(args=self.args, device=self.args.consumer_gpu)
        print(args)
        self.producer_actor_critic.print_trainable_parameters()
        self.data = {
            'metas': load_pickle(args.data_path + 'meta1.pickle'),
            'sequential': load_pickle(args.data_path + 'sequential.pickle'),
            'preference': load_pickle(args.data_path + 'preference.pickle'),
            # 'intention': load_pickle(args.data_path + 'intention.pickle'),
            'category': load_pickle(args.data_path + 'category1.pickle'),
            'ranking_candidate': load_pickle(args.data_path + 'ranking_candidate.pickle'),
        }
        if self.args.lower:
            for _ in self.data['metas']:
                self.data['metas'][_]['title'] = self.data['metas'][_]['title'].lower().strip()
                self.data['metas'][_][self.args.item_index] = self.data['metas'][_][
                    self.args.item_index].lower().strip()
        self.train_batch = 0
        self.update_step = 0
        self.memories = Queue(maxsize=2)

    def compute_adv(self, old_action_values, rewards, action_mask):
        if self.args.whiten_reward:
            whitened_rewards = whiten(rewards, action_mask, shift_mean=False, dim=None)
        else:
            whitened_rewards = rewards
        last_gae_lam = 0
        advantages_reversed = []
        gen_length = torch.sum(action_mask, dim=1).max().item()
        for time_step in range(1, gen_length + 1):
            next_values = old_action_values[:, -time_step+1] if time_step > 1 else 0.0
            delta = whitened_rewards[:, -time_step] + self.args.gamma * next_values - old_action_values[:, -time_step]
            last_gae_lam = delta + self.args.gamma * self.args.lam * last_gae_lam
            advantages_reversed.append(last_gae_lam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        advantages = F.pad(advantages, (whitened_rewards.size(1) - gen_length, 0), value=0.0)
        returns = (advantages + old_action_values) * action_mask
        whitened_advantages = whiten(advantages, action_mask, dim=None).detach() * action_mask
        return whitened_advantages, returns

    def RLHF_Consumer(self, optim, lr_scheduler, writer):
        self.train()
        ratio_clip_range = 0.2
        value_clip_range = 0.2
        early_stop = False
        train_data = self.memories.get()
        for epoch in range(self.args.epoch):
            for idx, (
                _sequence,
                _action_mask,
                _old_action_values,
                _old_sequence_log_probs_shifted,
                _ref_sequence_log_probs_shifted,
                _whitened_advantages,
                _returns
            ) in enumerate(train_data):
                self.update_step += 1
                sequence_logit = self.consumer_actor_critic.forward('actor', _sequence, attention_mask=_sequence != 0).logits
                sequence_dists = torch.distributions.categorical.Categorical(logits=sequence_logit)
                sequence_dists_shifted = shift(sequence_dists.probs, shift=1, dim=-2).contiguous()
                sequence_log_probs_shifted = log_prob(sequence_dists_shifted, _sequence)
                # entropy loss
                entropy_losses = masked_mean(sequence_dists.entropy(), _action_mask)
                # action loss
                log_ratio = sequence_log_probs_shifted - _old_sequence_log_probs_shifted
                ratio = torch.exp(log_ratio)
                action_losses1 = -_whitened_advantages * ratio
                action_losses2 = -_whitened_advantages * torch.clamp(ratio,
                                                                     min=1.0 - ratio_clip_range,
                                                                     max=1.0 + ratio_clip_range)
                action_losses = masked_mean(torch.max(action_losses1, action_losses2), _action_mask)
                # actor loss
                actor_loss = action_losses - self.args.entropy_weight * entropy_losses
                (actor_loss/self.args.gradient_accumulation_steps).backward()

                # critic loss
                # action_values = warp_actor_critic.get_action_value(_sequence, attention_mask=_sequence != 0)
                action_value = self.consumer_actor_critic.forward('critic', _sequence, attention_mask=_sequence != 0)
                new_values_clipped = torch.clamp(action_value,
                                                 min=_old_action_values - value_clip_range,
                                                 max=_old_action_values + value_clip_range)
                critic_losses1 = torch.square(action_value - _returns)
                critic_losses2 = torch.square(new_values_clipped - _returns)
                critic_losses = 0.5 * masked_mean(torch.max(critic_losses1, critic_losses2), _action_mask)
                (self.args.vf_coef*critic_losses/self.args.gradient_accumulation_steps).backward()

                approx_kl = 0.5 * masked_mean(log_ratio**2, _action_mask)
                policy_kl = masked_mean(-log_ratio, _action_mask)
                critic_clip_frac = masked_mean(torch.gt(critic_losses2, critic_losses1).float(), _action_mask)
                action_clip_frac = masked_mean(torch.gt(action_losses2, action_losses1).float(), _action_mask)
                total_losses = (action_losses + self.args.vf_coef * critic_losses-self.args.entropy_weight*entropy_losses)
                writer.add_scalar('training/actor_loss_mean', float(action_losses.mean()), self.update_step)
                writer.add_scalar('training/critic_loss_mean', float(critic_losses.mean()), self.update_step)
                writer.add_scalar('training/entropy_loss_mean', float(entropy_losses.mean()), self.update_step)
                writer.add_scalar('training/total_loss_mean', float(total_losses.mean()), self.update_step)
                writer.add_scalar('training/approx_kl', float(approx_kl), self.update_step)
                writer.add_scalar('training/policy_kl', float(policy_kl), self.update_step)
                writer.add_scalar('training/action_clip_frac', float(action_clip_frac), self.update_step)
                writer.add_scalar('training/critic_clip_frac', float(critic_clip_frac), self.update_step)

                if (idx + 1) % self.args.gradient_accumulation_steps == 0 or (idx + 1) == len(train_data) or policy_kl > self.args.policy_kl_threshold:
                    if self.args.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(optim.param_groups[0]['params'], self.args.clip_grad_norm)
                    optim.step()
                    lr_scheduler.step()
                    optim.zero_grad()

                RLHF_lr = lr_scheduler.get_lr()[0]
                print(f"Batch: {self.train_batch} | Epoch: {epoch} | Update_step: {self.update_step}"
                      f" | RLHF_lr: {RLHF_lr:.7f} | policy_kl: {policy_kl:.6f}")
                if policy_kl > self.args.policy_kl_threshold:
                    early_stop = True
                    break
            if early_stop:
                break
            np.random.shuffle(train_data)

    def RLHF_Producer(self, train_loader, metrics_dict, writer, running, sample_kwargs):
        sample_num = self.args.sample_num
        for eps in range(self.args.num_episodes):
            for step_i, batch in enumerate(train_loader):
                self.train_batch += 1
                if (step_i+1) % 100 == 0 and self.args.lr > 0:
                    self.actor_critic.save_parameters(f'{step_i+1}step')
                self.eval()
                bs = len(batch['task'])
                input_data = batch['input_data']
                if step_i % 100 == 0:
                    print(batch['input_text'][0])
                    print(input_data['input_ids'][0])
                input_ids_length = input_data['input_ids'].shape[1]
                input_data['input_ids'] = input_data['input_ids'].repeat_interleave(sample_num, dim=0).contiguous()
                input_data['attention_mask'] = input_data['attention_mask'].repeat_interleave(sample_num, dim=0).contiguous()
                output_labels = [[__ for __ in _.strip().split('\n')] for _ in batch['output_text']]
                with torch.no_grad():
                    output_ids = self.actor_critic.actor_model.sample(**input_data, **sample_kwargs)
                    output_title = self.tokenizer.batch_decode(output_ids[:, input_ids_length:], skip_special_tokens=True)
                    output_title_list = [
                        [__.strip() for __ in _.strip().split('\n')]
                        for _ in output_title]
                    if self.args.idx:
                        output_title_list = [[re.sub(r'(\d+). *', '', __, count=1) for __ in _] for _ in output_title_list]

                    complete_text = [
                        [batch['input_text'][i // sample_num], get_output_text(_, '\n'+self.tokenizer.eos_token, idx=self.args.idx)]
                        for i, _ in enumerate(output_title_list)]
                    complete_data = side_tokenizer(complete_text,
                                                   'left', self.tokenizer,
                                                   padding=True, truncation=True,
                                                   max_length=self.args.max_token_length + self.args.gen_max_length,
                                                   return_tensors='pt').to(self.args.gpu).data
                    reward_data = [
                        train_loader.dataset.get_list_reward_hard_encode(
                            batch['task'][i // sample_num],
                            batch['input_field_data'][i // sample_num],
                            output_title_list[i])
                        for i in range(bs * sample_num)]

                    list_rewards = torch.tensor([
                        reward_data[i][0]
                        for i in range(bs * sample_num)], device=self.accelerator.device)
                    for i in range(bs * sample_num):
                        metrics_dict.add_sample('RLHFTotal', batch['input_field_data'][i//sample_num], output_title_list[i],
                                                output_labels[i//sample_num], float(list_rewards[i]))
                    # metrics_dict.print()
                    total_count = metrics_dict.metrics_dict['RLHFTotal']['Count']
                    total_reward_sum = metrics_dict.metrics_dict['RLHFTotal']['RewardSum']
                    total_non_exist_rate = metrics_dict.metrics_dict['RLHFTotal'][f'NonExistRate@{metrics_dict.topk}']
                    total_repeat_rate = metrics_dict.metrics_dict['RLHFTotal'][f'RepeatRate@{metrics_dict.topk}']
                    total_correct_count = metrics_dict.metrics_dict['RLHFTotal'][f'CorrectCount@{metrics_dict.topk}']
                    writer.add_scalar('sampling/total_reward', total_reward_sum/total_count, total_count//sample_num)
                    writer.add_scalar('sampling/total_non_exist_rate', total_non_exist_rate/total_count, total_count//sample_num)
                    writer.add_scalar('sampling/total_repeat_rate', total_repeat_rate/total_count, total_count//sample_num)
                    writer.add_scalar('sampling/total_correct_count', total_correct_count/total_count, total_count//sample_num)
                    if self.args.lr > 0:
                        old_action_values = self.actor_critic.forward(scope='critic', **complete_data)

                        old_sequence_logit = self.actor_critic.forward(scope='actor', **complete_data).logits
                        old_sequence_dists = torch.softmax(old_sequence_logit, dim=-1)
                        old_sequence_dists_shifted = shift(old_sequence_dists, shift=1, dim=-2).contiguous()
                        old_sequence_log_probs_shifted = log_prob(old_sequence_dists_shifted, complete_data['input_ids'])

                        ref_sequence_logit = self.actor_critic.base_model.forward(**complete_data).logits
                        ref_sequence_dists = torch.softmax(ref_sequence_logit, dim=-1)
                        ref_sequence_dists_shifted = shift(ref_sequence_dists, shift=1, dim=-2).contiguous()
                        ref_sequence_log_probs_shifted = log_prob(ref_sequence_dists_shifted, complete_data['input_ids'])

                        # kl penalized, exploration
                        action_mask = torch.zeros_like(complete_data['input_ids'], dtype=torch.bool)
                        action_mask_index = torch.cumsum(torch.eq(complete_data['input_ids'], self.tokenizer.bos_token_id), dim=-1)
                        action_mask[action_mask_index > 1] = True

                        if self.args.reward_scale:
                            running['list'].update(list_rewards)
                            score_scaling_factor = running['list'].std + torch.finfo(list_rewards.dtype).eps
                            list_rewards /= score_scaling_factor
                        total_rewards = torch.zeros_like(complete_data['input_ids'], dtype=torch.float)
                        for _, __ in enumerate(complete_data['input_ids']):
                            total_rewards[_][__ == self.tokenizer.eos_token_id][-1] += list_rewards[_]

                        if self.args.fine_grain_reward:
                            item_rewards = [
                                reward_data[i][1]
                                for i in range(bs * sample_num)]
                            for i in range(bs * sample_num):
                                if self.args.reward_scale:
                                    task = batch['task'][i // sample_num]
                                    running['item'][task].update(item_rewards[i])
                                    score_scaling_factor = running['item'][task].std + torch.finfo(
                                        item_rewards[i].dtype).eps
                                    item_rewards[i] /= score_scaling_factor
                                total_rewards[i][complete_data['input_ids'][i] == 13][4:] += item_rewards[i]

                        kl_penalty = (old_sequence_log_probs_shifted - ref_sequence_log_probs_shifted) * action_mask
                        rewards_penalized = total_rewards - kl_penalty*self.args.kl_coef
                        whitened_advantages, returns = self.compute_adv(old_action_values, rewards_penalized, action_mask)

                        for i in range(bs * sample_num):
                            ignore_index = torch.not_equal(complete_data['input_ids'][i, ...], self.tokenizer.pad_token_id)
                            if ignore_index.sum() > 512:
                                continue
                            (
                                sequence_i,
                                action_mask_i,
                                old_action_values_i,
                                old_sequence_log_probs_shifted_i,
                                ref_sequence_log_probs_shifted_i,
                                whitened_advantages_i,
                                returns_i
                            ) = (
                                rearrange(complete_data['input_ids'][i, ...][ignore_index], '... -> 1 ...').contiguous(),
                                rearrange(action_mask[i, ...][ignore_index], '... -> 1 ...').contiguous(),
                                rearrange(old_action_values[i, ...][ignore_index], '... -> 1 ...').contiguous(),
                                rearrange(old_sequence_log_probs_shifted[i, ...][ignore_index], '... -> 1 ...').contiguous(),
                                rearrange(ref_sequence_log_probs_shifted[i, ...][ignore_index], '... -> 1 ...').contiguous(),
                                rearrange(whitened_advantages[i, ...][ignore_index], '... -> 1 ...').contiguous(),
                                rearrange(returns[i, ...][ignore_index], '... -> 1 ...').contiguous()
                            )

                            # append train sample
                            self.memories.put(
                                Memory(
                                    sequence_i,
                                    action_mask_i,
                                    old_action_values_i,
                                    old_sequence_log_probs_shifted_i,
                                    ref_sequence_log_probs_shifted_i,
                                    whitened_advantages_i,
                                    returns_i
                                )
                            )
                    print(f'rlhf learning in eps_{eps} | step_{step_i} | example: {self.memories.qsize()} | max_length: {complete_data["input_ids"].shape[1]}')

    def RLHF_train(self):
        TaskTemplate = {_: Train_task_group_mapping[_] for _ in self.args.RLHF_train_tasks.split(',')}
        ValTaskTemplate = {_: Val_task_group_mapping[_] for _ in self.args.RLHF_val_tasks.split(',')}
        TaskNum = {_: 1 for _ in self.args.RLHF_train_tasks.split(',')}
        ValTaskNum = {_: 1 for _ in self.args.RLHF_val_tasks.split(',')}
        train_data = RLHFDataset(self.args, TaskTemplate, TaskNum, self.data, self.tokenizer, 'train')
        val_data = RLHFDataset(self.args, ValTaskTemplate, ValTaskNum, self.data, self.tokenizer, 'val')

        train_loader = DataLoader(train_data, batch_size=self.args.batch_size, shuffle=True, collate_fn=train_data.collate_fn)
        val_loader = DataLoader(val_data, batch_size=self.args.val_batch_size, shuffle=False, collate_fn=val_data.collate_fn, drop_last=False)

        metrics_dict = Metrics(['RLHFTotal'], self.args.topk, train_data.category2item, train_data.title2item)
        running = {'item': {_: RunningMoments() for _ in self.args.RLHF_train_tasks.split(',')}, 'list': RunningMoments()}

        RLHF_optim = self.get_optimizer(self.consumer_actor_critic.actor_parameters + self.consumer_actor_critic.critic_parameters)
        schedule_time = self.args.batch_size * self.args.sample_num / self.args.gradient_accumulation_steps * self.args.epoch
        RLHF_lr_scheduler = None
        if self.args.lr > 0:
            RLHF_lr_scheduler = get_polynomial_decay_schedule_with_warmup(RLHF_optim, 500, len(train_loader) * schedule_time, power=2.0)
        writer = SummaryWriter(log_dir=f'logs/RLHF_train/{self.args.model_name}', flush_secs=30)
        sample_kwargs = {
            'logits_processor': LogitsProcessorList([MinLengthLogitsProcessor(15, eos_token_id=self.tokenizer.eos_token_id)]),
            'logits_warper': LogitsProcessorList([TopKLogitsWarper(20), TemperatureLogitsWarper(0.7)]),
            'stopping_criteria': StoppingCriteriaList([MaxLengthCriteria(max_length=self.args.max_token_length + self.args.gen_max_length)]),
        }
        self.consumer_actor_critic.share_memory()
        p_producer = mp.Process(target=self.RLHF_Producer, args=(train_loader, metrics_dict, writer, running, sample_kwargs,))
        p_producer.start()
        p_producer.join()
        p_consumer = mp.Process(target=self.RLHF_Consumer, args=(RLHF_optim, RLHF_lr_scheduler, writer,))
        p_consumer.start()
        p_consumer.join()

    def RLHF_val(self, step, val_loader, writer):
        torch.cuda.empty_cache()
        stopping_criteria = StoppingCriteriaList(
            [MaxLengthCriteria(max_length=self.args.max_token_length + self.args.gen_max_length)]
        )
        metrics_dict = Metrics(self.args.RLHF_val_tasks.split(','), self.args.topk,
                               val_loader.dataset.category2item, val_loader.dataset.title2item)
        self.eval()  # First batch users: ['A254N8K05ZWM4C', 'A2RM9L2MM9EQVE', 'A1D2LB4Z5RMO4M', 'A1D90MKRWB0ZKP']
        pbar = tqdm(total=len(val_loader), ncols=150, disable=not self.accelerator.is_local_main_process)
        for step_i, batch in enumerate(val_loader):
            bs = len(batch['task'])
            input_data = batch['input_data']
            if self.accelerator.is_main_process and step_i % 1000 == 0:
                print(batch['input_text'][0])
                print(input_data['input_ids'][0])
            input_ids_length = input_data['input_ids'].shape[1]
            output_labels = [[__ for __ in _.strip().split('\n')] for _ in batch['output_text']]
            with torch.no_grad():
                output_ids = self.actor_critic.actor_model.greedy_search(
                    **input_data,
                    stopping_criteria=stopping_criteria,
                )
                output_title = self.tokenizer.batch_decode(output_ids[:, input_ids_length:], skip_special_tokens=True)
                output_title_list = [
                    [__.strip() for __ in _.strip().split('\n')]
                    for _ in output_title]
                if self.args.idx:
                    output_title_list = [[re.sub(r'(\d+). *', '', __, count=1) for __ in _] for _ in output_title_list]
                reward_data = [
                    val_loader.dataset.get_list_reward_hard_encode(
                        batch['task'][i],
                        batch['input_field_data'][i],
                        output_title_list[i])
                    for i in range(bs)]

                list_rewards = torch.tensor([
                    reward_data[i][0]
                    for i in range(bs)], device=self.device)
                for i in range(bs):
                    task = batch['task'][i]
                    metrics_dict.add_sample(task, batch['input_field_data'][i], output_title_list[i],
                                            output_labels[i], float(list_rewards[i]))
                # if step_i % 100 == 0 or (step_i + 1) == len(val_loader):
                #     metrics_dict.print()
                pbar.update(1)
        _reward_sum, _non_exist_rate, _repeat_rate, _correct_count = 0.0, 0.0, 0.0, 0.0
        for task in metrics_dict.metrics_dict:
            task_count = metrics_dict.metrics_dict[task]['Count']
            reward_sum = metrics_dict.metrics_dict[task]['RewardSum']
            recall = metrics_dict.metrics_dict[task][f'Recall@{metrics_dict.topk}']
            ndcg = metrics_dict.metrics_dict[task][f'NDCG@{metrics_dict.topk}']
            non_exist_rate = metrics_dict.metrics_dict[task][f'NonExistRate@{metrics_dict.topk}']
            repeat_rate = metrics_dict.metrics_dict[task][f'RepeatRate@{metrics_dict.topk}']
            correct_count = metrics_dict.metrics_dict[task][f'CorrectCount@{metrics_dict.topk}']

            if task == 'RLHFPersonalCategoryRate':
                category_rate_correct = metrics_dict.metrics_dict[task][f'CategoryRateCorrect@{metrics_dict.topk}']
                log_d = torch.tensor([task_count, reward_sum, recall, ndcg, non_exist_rate, repeat_rate, correct_count, category_rate_correct],
                                     device=self.accelerator.device)
            else:
                log_d = torch.tensor([task_count, reward_sum, recall, ndcg, non_exist_rate, repeat_rate, correct_count],
                                     device=self.accelerator.device)
            log_d = self.accelerator.reduce(log_d)

            _reward_sum += log_d[1]/log_d[0]
            _non_exist_rate += log_d[4]/log_d[0]
            _repeat_rate += log_d[5]/log_d[0]
            _correct_count += log_d[6]/log_d[0]

            if self.accelerator.is_main_process:
                writer.add_scalar(f'valuating/{task}_Reward_mean', log_d[1] / log_d[0], step)
                writer.add_scalar(f'valuating/{task}_Recall', log_d[2] / log_d[0], step)
                writer.add_scalar(f'valuating/{task}_NDCG', log_d[3] / log_d[0], step)
                writer.add_scalar(f'valuating/{task}_NonExist_rate', log_d[4] / log_d[0], step)
                writer.add_scalar(f'valuating/{task}_Repeat_rate', log_d[5] / log_d[0], step)
                writer.add_scalar(f'valuating/{task}_Correct_count', log_d[6] / log_d[0], step)
                if task == 'RLHFPersonalCategoryRate':
                    writer.add_scalar(f'valuating/{task}_Category_rate_correct', log_d[7] / log_d[0], step)
        if self.accelerator.is_main_process:
            writer.add_scalar(f'valuating/Total_Reward_mean', _reward_sum/len(metrics_dict.metrics_dict), step)
            writer.add_scalar(f'valuating/Total_NonExist_rate', _non_exist_rate/len(metrics_dict.metrics_dict), step)
            writer.add_scalar(f'valuating/Total_Repeat_rate', _repeat_rate/len(metrics_dict.metrics_dict), step)
            writer.add_scalar(f'valuating/Total_Correct_count', _correct_count/len(metrics_dict.metrics_dict), step)

        # "RLHFValSeqRec": {
        #     "NonExistRate@10": "0.4844",
        #     "RepeatRate@10": "0.4500",
        #     "CorrectCount@10": "1.0000",
        #     "Count": 640,
        #     "Recall@10": "0.1297",
        #     "MRR@10": "0.0934",
        #     "TargetCategoryRatio@10": "0.7766",
        #     "RewardSum": "0.3367"
        # }

    def RLHF_test(self):
        torch.cuda.empty_cache()
        stopping_criteria = StoppingCriteriaList(
            [
                MaxLengthCriteria(max_length=self.args.max_token_length + self.args.gen_max_length)
            ]
        )
        TestTaskTemplate = {self.args.RLHF_test_task: Test_task_group_mapping[self.args.RLHF_test_task]}
        TestTaskNum = {self.args.RLHF_test_task: 1}

        if self.args.RLHF_test_task in ['RLHF+PersonalControlRec', 'RLHF-PersonalControlRec', 'RLHFPersonalCategoryRate']:
            SeqRec_Result_file = f'{self.args.RLHF_load}_Result_RLHFSeqRec.pickle'
            self.data['RLHFSeqRec_Result'] = load_pickle(SeqRec_Result_file)
        test_data = RLHFDataset(self.args, TestTaskTemplate, TestTaskNum, self.data, self.tokenizer, 'test')
        test_loader = DataLoader(test_data, batch_size=self.args.test_batch_size, shuffle=False,
                                 collate_fn=test_data.collate_fn, drop_last=False)
        metrics_dict = Metrics([self.args.RLHF_test_task], self.args.topk, test_data.category2item, test_data.title2item)
        self.eval()

        result_file = f'{self.args.RLHF_load}_Result_{self.args.RLHF_test_task}.pickle'
        result = load_pickle(result_file) or []
        result = result[:-(len(result) % self.args.batch_size) if (len(result) % self.args.batch_size) != 0 else None]
        pbar = tqdm(total=len(test_loader), ncols=150)

        with torch.no_grad():
            for step_i, batch in enumerate(test_loader):
                bs = len(batch['task'])
                input_data = batch['input_data']
                if step_i % 10000 == 0:
                    print(batch['input_text'][0])
                    print(input_data['input_ids'][0])
                output_labels = [[__ for __ in _.strip().split('\n')] for _ in batch['output_text']]
                input_ids_length = input_data['input_ids'].shape[1]
                if (step_i+1)*self.args.test_batch_size <= len(result):
                    output_title_list = [_[1] for _ in result[step_i*self.args.test_batch_size: (step_i+1)*self.args.test_batch_size]]
                else:
                    output_ids = self.actor_critic.actor_model.greedy_search(
                        **input_data,
                        stopping_criteria=stopping_criteria,
                    )
                    output_title = self.tokenizer.batch_decode(output_ids[:, input_ids_length:],
                                                               skip_special_tokens=True)
                    output_title_list = [[__.strip() for __ in _.strip().split('\n')] for _ in output_title]
                    result += [[_, __] for _, __ in zip(output_labels, output_title_list)]
                    if step_i % 100 == 0 or (step_i + 1) == len(test_loader):
                        save_pickle(result, result_file)

                reward_data = [
                    test_data.get_list_reward_hard_encode(
                        batch['task'][i],
                        batch['input_field_data'][i],
                        output_title_list[i])
                    for i in range(bs)
                ]

                list_rewards = torch.tensor([reward_data[i][0] for i in range(bs)], device=self.device)
                for i in range(bs):
                    task = batch['task'][i]
                    metrics_dict.add_sample(task, batch['input_field_data'][i], output_title_list[i],
                                            output_labels[i], float(list_rewards[i]))
                metrics_dict.print()
                pbar.update(1)
            pbar.close()


if __name__ == '__main__':
    pass
