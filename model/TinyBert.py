# -*- coding: utf-8 -*-
# @Time : 2022/7/27 21:10
# @Author : Bingshuai Liu
import torch
from torch import nn
import json
import os
import copy
import logging
from bert_transformer.modeling import BertPreTrainedModel, BertModel


class TinyBert(BertPreTrainedModel):
    def __init__(self, config, num_labels, fit_size=768):
        """
        :param config: BertConfig 用于初始化模型和保存模型参数
        :param num_labels: 分类数
        :param fit_size: 需要转换的大小，bert-chinese的hidden_size是768，这里默认取768
        """
        super(TinyBert, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.fit_dense = nn.Linear(config.hidden_size, fit_size)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, segment_ids, mask, label_id=None, is_student=True):
        bert_output = self.bert(input_ids, segment_ids, mask,
                                output_all_encoded_layers=True, output_att=True)
        sequence_output = bert_output[0]
        attention_output = bert_output[1]
        pooled_out = bert_output[2]
        # 这里使用pooled_out做情感分类的输入
        # [batch_size, hidden_size, len]
        logits = self.classifier(torch.relu(pooled_out))  # 参考华为的做法 加上激活函数

        # 如果是学生模型，需要适应老师模型的大小，使用fit_dense做以下转换
        temp = []
        if is_student:
            for sequence_layer in sequence_output:
                temp.append(self.fit_dense(sequence_layer))
            sequence_output = temp
        return logits, attention_output, sequence_output
