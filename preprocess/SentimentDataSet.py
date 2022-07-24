# -*- coding: utf-8 -*-
# @Time : 2022/7/22 21:27
# @Author : Bingshuai Liu
import torch
from torch.utils.data import Dataset


class SentimentDataSet(Dataset):
    def __init__(self, features, mode, device):
        self.len = len(features)
        self.input_ids = [torch.tensor(feature.input_ids).long().to(device) for feature in features]
        self.input_mask = [torch.tensor(feature.input_mask).float().to(device) for feature in features]
        self.segment_ids = [torch.tensor(feature.segment_ids).float().to(device) for feature in features]
        self.label_id = None
        if mode == 'train' or 'test':
            self.label_id = [torch.tensor(feature.label_id).to(device) for feature in features]

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        data = {'input_ids': self.input_ids[item],
                'input_mask': self.input_mask[item],
                'segment_ids': self.segment_ids[item]}
        if self.label_id is not None:
            data['label_id'] = self.label_id[item]
        return data
