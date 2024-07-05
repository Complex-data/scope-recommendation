import torch
import torch.nn as nn


class CrossEntropyLoss_e(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, ignore_value=-100):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.ignore_value = ignore_value  # 如果labels的值为-100 就忽略此处的loss值

    def forward1(self, logits, labels):
        C = logits.shape[1]
        # logits [B,C,N], labels[B,N], log_prob [B,C,N]

        # 创建掩码，忽略 ignore_index 对应的位置
        mask = (labels != self.ignore_value).float()

        # 将 labels 中的 -100 替换为 0（或任何有效的索引），以避免索引越界
        # 这里我们使用 0 作为一个安全的索引，因为实际损失将通过 mask 被置为 0
        labels = torch.where(labels == self.ignore_value, torch.zeros_like(labels), labels)
        # 未考虑上下溢出计算logSoftmax：
        # s = logits.exp().sum(dim=1,keepdim=True).repeat(1,C,1)
        # log_prob = (logits.exp()/s).log()

        # 考虑上下溢出计算logSoftmax：
        # 沿C维度logits减去最大值 避免logits数值为超大正数，而发生上溢出
        offset_logits = logits - logits.max(dim=1, keepdim=True).values  # 后者C维度长度为1，广播相减
        # s = offset_logits.exp().sum(dim=1,keepdim=True) #sum的C维度长度为1，广播

        # 计算 exp，并检查非 -inf 位置（非 0 值）
        exp_logits = offset_logits.exp()

        # 沿 vocab_size 维度求和，得到 s
        s = exp_logits.sum(dim=1, keepdim=True)

        # 检查每个 step 是否只有一个非 -inf 元素
        # 这可以通过检查 exp_logits 是否只有一个非零元素来确定
        # non_inf_count = (exp_logits != 0).sum(dim=1, keepdim=True)
        #
        # # 创建一个条件矩阵，其中只有一个非 -inf 元素的 step 对应的值为 True
        # condition = non_inf_count == 1
        #
        # # 直接在 s 上根据条件加上 0.1  a b c
        # # s_adjusted = s + 0.1 * condition.float()
        # # 将分母放大1.5倍
        # s_adjusted = s * 1.5 * condition.float() + s * (~condition).float()
        # s_adjusted = s

        # 对于scope范围内只有一个元素的logits这里的求和s需要加上一个常量

        # 分母s过大，分子offset_logits过小时很可能log(0)=NaN，
        # 为避免下溢出，因此把分子分母分别提出来单独log，而分子的log和exp抵消
        log_prob = offset_logits - s.log()

        if self.weight != None:
            weight = self.weight.view(1, C, 1).repeat(logits.shape[0], 1, logits.shape[2])
            log_prob *= weight

            if self.ignore_index in range(0, C):
                # 后面算mean的时候weight要忽略一个值
                weight[:, self.ignore_index] = 0

            # 用标签gather weight w_yn
            w_y = torch.gather(weight, dim=1, index=labels.unsqueeze(1))
            weightSum = w_y.view(-1).sum()

        # labels [B,N] 直接作为index进行gather
        # 或者labels也可以转成one hot向量作为mask进行masked_select（或者点乘），转成one-hot的方法：
        # nn.functional.one_hot() / scatter_ / torch.where(index == target, ones, zeros)

        if self.ignore_index in range(0, C):
            # 直接在log_prob该列全部置0
            log_prob[:, self.ignore_index] = 0

        l = - torch.gather(log_prob, dim=1, index=labels.unsqueeze(1)).squeeze()
        l = l * mask  # 应用掩码
        # reduction默认为在B维度上平均
        if self.weight == None:
            loss = l.sum() / mask.sum()  # 只考虑非忽略位置的平均损失
        else:
            weightSum = (l * mask).sum() / mask.sum()  # 加权损失
            loss = weightSum
        return loss
    def forward(self, logits, labels, scope_mask):
        C = logits.shape[1]

        # 创建掩码，忽略 ignore_index 对应的位置
        mask = (labels != self.ignore_value).float()

        # 将 labels 中的 -100 替换为 0（或任何有效的索引），以避免索引越界
        # 这里我们使用 0 作为一个安全的索引，因为实际损失将通过 mask 被置为 0
        labels = torch.where(labels == self.ignore_value, torch.zeros_like(labels), labels)


        # 未考虑上下溢出计算logSoftmax：
        # s = logits.exp().sum(dim=1,keepdim=True).repeat(1,C,1)
        # log_prob = (logits.exp()/s).log()

        # 考虑上下溢出计算logSoftmax：
        # 沿C维度logits减去最大值 避免logits数值为超大正数，而发生上溢出
        offset_logits = logits - logits.max(dim=1, keepdim=True).values  # 后者C维度长度为1，广播相减
        # s = offset_logits.exp().sum(dim=1,keepdim=True) #sum的C维度长度为1，广播

        # 计算 exp，并检查非 -inf 位置（非 0 值）
        exp_logits = offset_logits.exp()

        # 获取label索引的exp值
        exp_logits_label = torch.gather(exp_logits, dim=1, index=labels.unsqueeze(1))

        # 沿 vocab_size 维度求和，得到 s
        s = exp_logits.sum(dim=1, keepdim=True)

        # 计算p_i
        probability_label = torch.div(exp_logits_label,s)
        # 设置
        gamma = 2
        shrink_coe = torch.pow((1-probability_label), gamma)
        # 只将scope内位置上的添加系数
        scope_positions = scope_mask.any(dim=1)
        shrink_coe[~scope_positions] = 1

        # 分母s过大，分子offset_logits过小时很可能log(0)=NaN，
        # 为避免下溢出，因此把分子分母分别提出来单独log，而分子的log和exp抵消
        log_prob = offset_logits - s.log()

        if self.weight != None:
            weight = self.weight.view(1, C, 1).repeat(logits.shape[0], 1, logits.shape[2])
            log_prob *= weight

            if self.ignore_index in range(0, C):
                # 后面算mean的时候weight要忽略一个值
                weight[:, self.ignore_index] = 0

            # 用标签gather weight w_yn
            w_y = torch.gather(weight, dim=1, index=labels.unsqueeze(1))
            weightSum = w_y.view(-1).sum()

        if self.ignore_index in range(0, C):
            # 直接在log_prob该列全部置0
            log_prob[:, self.ignore_index] = 0

        l = - torch.gather(log_prob, dim=1, index=labels.unsqueeze(1)).squeeze()
        l = l.mul(shrink_coe.view(-1))
        l = l * mask  # 应用掩码
        # reduction默认为在B维度上平均
        if self.weight == None:
            loss = l.sum() / mask.sum()  # 只考虑非忽略位置的平均损失
        else:
            weightSum = (l * mask).sum() / mask.sum()  # 加权损失
            loss = weightSum
        return loss