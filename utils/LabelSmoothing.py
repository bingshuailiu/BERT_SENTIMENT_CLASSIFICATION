# -*- coding: utf-8 -*-
# @Time : 2022/7/25 23:56
# @Author : Bingshuai Liu
import torch


def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0, device=torch.device('cpu')):
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=device)
        true_dist.fill_(smoothing / (classes - 1))
        _, index = torch.max(true_labels.to(device), 1)
        true_dist.scatter_(1, torch.LongTensor(index.unsqueeze(1)), confidence)
    return true_dist


def label_to_one_hot(labels, class_size, device):
    one_hots = torch.zeros(labels.size(0), class_size, device=device)
    for index, label in enumerate(labels):
        one_hots[index, int(label)] = 1
    return one_hots
