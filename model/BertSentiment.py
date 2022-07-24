# -*- coding: utf-8 -*-
# @Time : 2022/7/22 21:38
# @Author : Bingshuai Liu
from torch import nn
import os
from transformers import BertModel


class BertSentimentClassifier(nn.Module):
    def __init__(self, bertDir):
        super(BertSentimentClassifier, self).__init__()
        self.bert_module = BertModel.from_pretrained(bertDir)
        self.bert_config = self.bert_module.config
        out_dim = self.bert_config.hidden_size
        self.classifier = nn.Linear(out_dim, 3)

    def forward(self, input_ids, segment_ids, input_mask, label_id=None):
        # 经过BERT输出
        bert_output = self.bert_module(input_ids=input_ids.long(),
                                       attention_mask=input_mask.long(),
                                       token_type_ids=segment_ids.long())
        # 对输出结果拆包
        # seq_out 用于NER任务
        seq_out = bert_output[0]
        # pooled_out 用于分类任务
        pooled_out = bert_output[1]

        x = pooled_out.detach()
        # 经过线性层,输出
        out = self.classifier(x)
        return out

class SentimentIntegrationClassifier(nn.Module):
    def __init__(self, bertDir):
        super(SentimentIntegrationClassifier, self).__init__()

        self.bert_module = BertModel.from_pretrained(bertDir)
        self.bert_config = self.bert_module.config
        out_dim = self.bert_config.hidden_size
        self.bert_classifier = nn.Linear(out_dim, 3)

        # self.svm_classifier =

    def forward(self, input_ids, segment_ids, input_mask, label_id=None):
        # 经过BERT输出
        bert_output = self.bert_module(input_ids=input_ids.long(),
                                       attention_mask=input_mask.long(),
                                       token_type_ids=segment_ids.long())
        # 对输出结果拆包
        # seq_out 用于NER任务
        seq_out = bert_output[0]
        # pooled_out 用于分类任务
        pooled_out = bert_output[1]

        x = pooled_out.detach()
        # 经过线性层,输出
        out = self.classifier(x)
        return out
        # def voting(self, ):