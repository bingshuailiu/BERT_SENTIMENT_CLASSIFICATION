# -*- coding: utf-8 -*-
# @Time : 2022/7/25 23:56
# @Author : Bingshuai Liu
import torch
from torch import nn


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # 此处的self.smoothing即我们的epsilon平滑参数。

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        _, index = torch.max(true_labels, 1)
        true_dist.scatter_(1, torch.LongTensor(index.unsqueeze(1)), confidence)
    return true_dist


def label_to_one_hot(labels, class_size):
    one_hots = []
    for index, label in enumerate(labels):
        one_hot = torch.zeros(1, class_size, device=labels.device)
        one_hot[index, int(label)] = 1
        one_hots.append(one_hot)
    return torch.cat(one_hots, dim=0)


# one_hots = label_to_one_hot(torch.tensor([1.0]), 3)
# smoothed = smooth_one_hot(one_hots, 3, 0.25)
# print(smoothed)

'''
Loss = CrossEntropyLoss(NonSparse=True, ...)
. . .
data = ...
labels = ...

outputs = model(data)

smooth_label = smooth_one_hot(labels, ...)
loss = (outputs, smooth_label)
...
'''
