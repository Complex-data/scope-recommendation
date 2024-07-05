from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch import nn
from torch.optim import AdamW, Adam
from actor_critic import ActorCritic
# from RLHF.reward_model import RewardModel


# trainer
class Trainer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def get_optimizer(self, params, filter_by_requires_grad=True, group_wd_params=True):
        def separate_weight_decay_params(p):
            wd_p, no_wd_p = [], []
            for _ in p:
                param_list = no_wd_p if _.ndim < 2 else wd_p
                param_list.append(_)
            return wd_p, no_wd_p

        if filter_by_requires_grad:
            params = list(filter(lambda _: _.requires_grad, params))

        if group_wd_params and self.args.weight_decay > 0:
            wd_params, no_wd_params = separate_weight_decay_params(params)

            params = [
                {'params': wd_params},
                {'params': no_wd_params, 'weight_decay': 0},
            ]

        if self.args.weight_decay == 0:
            return Adam(params,
                        lr=self.args.lr,
                        betas=(self.args.adam_beta1, self.args.adam_beta2),
                        eps=self.args.adam_eps)

        return AdamW(params,
                     lr=self.args.lr,
                     weight_decay=self.args.weight_decay,
                     betas=(self.args.adam_beta1, self.args.adam_beta2),
                     eps=self.args.adam_eps)

    # def create_reward_model(self):
    #     return RewardModel(self.args, self.RLHF_train_loader.dataset, self.tokenizer)

    @property
    def device(self):
        # if hasattr(self, 'accelerator'):
        #     return self.accelerator.device
        return self.args.gpu

    @property
    def tokenizer(self):
        return self.actor_critic.tokenizer


if __name__ == '__main__':
    pass
