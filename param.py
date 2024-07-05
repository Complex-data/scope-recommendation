import argparse
import pprint
import yaml


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


def add_args_SFT(parser):
    parser.add_argument('--share_chat_gpt_ratio', type=float, default=0.0, help='shareGPT语料占比')
    parser.add_argument("--SFT_load", type=str, default=None, help='Load the SFT model (usually the fine-tuned model).')
    parser.add_argument('--SFT_train_tasks', type=str, default='', help='SFTSeqRec, SFTSeqRec-domain, SFTControlRec,SFTPersonalControlRec,SFTPersonalCategoryRate,SFTCategoryRate')
    parser.add_argument('--SFT_val_tasks', type=str, default='', help='SFTTestSeqRec, SFTTestSeqRec-domain, SFTTestSeqRanking,SFT+TestPersonalControlRec,SFT-TestPersonalControlRec,SFTTestPersonalCategoryRate,SFTTestItemCount')
    parser.add_argument('--SFT_test_task', type=str, default='', help='in {SFTTestSeqRec, SFTTestSeqRec-domain,SFTTestRanking, SFT+TestPersonalControlRec, SFT-TestPersonalControlRec, SFTTestPersonalCategoryRate_xx%, SFTTestItemCount}')
    parser.add_argument("--SFT_actor_lora_r", type=int, default=16)
    parser.add_argument("--SFT_actor_lora_a", type=int, default=8)
    parser.add_argument('--user_control_symbol', action='store_true', help='是否添加控制符包裹item，例如，输出 <SOI>item<EOI>\n 2. <SOI>item2<EOI>')
    parser.add_argument('--use_CBS', action='store_true', help='推理的时候是否使用CBS')
    parser.add_argument('--use_scope_mask', action='store_true', help='训练的时候是否使用scope_mask')
    parser.add_argument('--domain', type=str, default='steam', help='确定使用的domain信息')
    return parser


def add_args_RLHF(parser):
    parser.add_argument('--RLHF_load', type=str, default=None, help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--RLHF_train_tasks', type=str, default='', help='RLHFSeqRec,RLHFSeqRanking,RLHF+PersonalControlRec,RLHF-PersonalControlRec,RLHFPersonalCategoryRate')
    parser.add_argument('--RLHF_val_tasks', type=str, default='', help='RLHFSeqRec,RLHFSeqRanking,RLHF+PersonalControlRec,RLHF-PersonalControlRec,RLHFPersonalCategoryRate,RLHFItemCount')
    parser.add_argument('--RLHF_test_task', type=str, default='', help='in {RLHFSeqRec, RLHFSeqRanking, RLHF+PersonalControlRec, RLHF-PersonalControlRec, RLHFPersonalCategoryRate_xx%, RLHFItemCount}')
    parser.add_argument("--RLHF_actor_lora_r", type=int, default=4)
    parser.add_argument("--RLHF_actor_lora_a", type=int, default=2)
    parser.add_argument("--RLHF_critic_lora_r", type=int, default=4)
    parser.add_argument("--RLHF_critic_lora_a", type=int, default=2)

    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--value_clip", type=float, default=0.4)
    parser.add_argument("--beta_s", type=float, default=0.01)
    parser.add_argument("--kl_coef", type=float, default=0.3)
    parser.add_argument("--vf_coef", type=float, default=0.1)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    parser.add_argument("--sample_num", type=int, default=4)
    parser.add_argument("--fine_grain_reward", action='store_true')
    parser.add_argument("--whiten_reward", action='store_true')
    parser.add_argument("--reward_scale", action='store_true')
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--policy_kl_threshold", type=float, default=0.03)
    parser.add_argument("--vague_mapping", action='store_true')
    parser.add_argument("--lr_power", type=float, default=2.0)
    parser.add_argument("--learn_batch", type=int, default=2)
    parser.add_argument("--new_data", action='store_true')
    parser.add_argument("--add_seq", action='store_true')
    parser.add_argument("--model_name", type=str, default=None, help='RLHF训练时自动生成')
    parser.add_argument("--model_name_suffix", type=str, default="")
    return parser


def add_args(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default=None, help='单卡运行时需要设置。accelerate启动则置空，自动配置')
    parser.add_argument('--FA2', action='store_true', help='是否使用flash attention')
    parser.add_argument('--llama2_chat_template', action='store_true', help='是否使用llama2-chat模板')
    parser.add_argument('--idx', action='store_true', help='是否添加输出item序号，例如，输出 1. item1\n 2. item2')

    # Data Splits
    parser.add_argument('--data_path', type=str, default='data/dataset/beauty/', help='data path')
    parser.add_argument('--lower', action='store_true', help='是否将文本全部变成小写')
    parser.add_argument('--candidate_num', type=int, default=10, help='候选集大小')
    parser.add_argument('--max_item_length', type=int, default=10, help='最大历史记录长度')
    parser.add_argument('--max_token_length', type=int, default=512, help='最大输入token长度')

    # Checkpoint
    parser.add_argument('--output', type=str, default='snap/')

    # Model Config
    parser.add_argument('--backbone', type=str, default='google/flan-t5-xl')
    parser.add_argument('--item_index', type=str, default='title', help='in {id, title, title64, title64_t}')

    # Training
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=8)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip_grad_norm', type=float, default=-1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--adam_eps', type=float, default=1e-5)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--val_epoch', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument("--lm_head", action='store_true', help='是否训练lm_head')

    # Inference
    parser.add_argument('--gen_max_length', type=int, default=512, help='训练，及推理生成token的最大长度')
    parser.add_argument('--topk', type=int, default=10)

    # Etc.
    parser.add_argument("--dry", action='store_true')
    parser.add_argument("--train_stage", type=str, default='SFT', help='in {SFT, SFT_Test, SFT_Merge, RLHF, RLHF_Test}')
    parser.add_argument("--log_to_file", action='store_true')
    parser.add_argument("--proxy", action='store_true')
    parser.add_argument("--backup_ip", type=str, default='0.0.0.0', help='服务ip，包括tensorboard服务，SASRec服务')
    parser.add_argument("--quantization", action='store_true', help='是否量化基础模型')

    # PEFT
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    return parser


def get_args(add_external_args_func=None):
    parser = add_args()
    args, remain_args = parser.parse_known_args()
    if args.train_stage in ['SFT', 'SFT_Test', 'SFT_Merge']:
        parser = add_args_SFT(parser)
    elif args.train_stage in ['RLHF', 'RLHF_Test', 'RLHF_Merge']:
        parser = add_args_RLHF(parser)
    if add_external_args_func:
        parser = add_external_args_func(parser)
    args = parser.parse_args(remain_args, args)
    return args


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.load(f)

        return Config(**kwargs)


if __name__ == '__main__':
    pass
