# -*- coding: utf-8 -*-
# @Time : 2022/7/21 16:10
# @Author : Bingshuai Liu
import pandas as pd
import os
import logging
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentRawInput(object):
    def __init__(self, text, label):
        self.text = text
        self.label = label


class SentimentFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = int(label_id)
        self.segment_ids = segment_ids


class DataPreprocessor(object):
    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self, data_dir):
        raise NotImplementedError()

    @classmethod
    def _read_csv(cls, input_file):
        data = pd.read_csv(input_file, encoding='gbk')
        return data


class SentimentPreprocessor(DataPreprocessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(data_dir), 'train'
        )

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, 'dev_data.csv')), 'dev'
        )

    def get_labels(self):
        return [-1, 0, 1]

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, 'test_data.csv')), 'test'
        )

    def _create_examples(self, data, set_type):
        raw_inputs = []
        for i in tqdm(range(data.shape[0])):
            row = data.iloc[i, :]
            text = row['review']
            label = row['label']
            raw_inputs.append(SentimentRawInput(text, label))
        return raw_inputs


def convertToFeatures(raw_inputs, labels, max_seq_length, tokenizer):

    label_map = {}
    for (i, label) in enumerate(labels):
        label_map[label] = i

    features = []
    for raw_input in tqdm(raw_inputs):
        tokens = tokenizer.tokenize(raw_input.text)
        encode_dict = tokenizer.encode_plus(text=tokens,
                                            max_seq_length=max_seq_length,
                                            is_pretokenized=True,
                                            return_token_type_ids=True,
                                            return_attention_mask=True)
        input_ids = encode_dict['input_ids']
        input_mask = encode_dict['attention_mask']
        segment_ids = encode_dict['token_type_ids']
        feature = SentimentFeatures(input_ids=input_ids,
                                    input_mask=input_mask,
                                    segment_ids=segment_ids,
                                    label_id=label_map[raw_input.label])
        features.append(feature)
    return features

# from transformers import BertModel, BertTokenizer
# from transformers import logging
#
# logging.set_verbosity_warning()
# DIR = '../data/train.csv'
#
# processor = SentimentPreprocessor()
# raw_inuts = processor.get_train_examples(DIR)
#
# # cleaned_inputs = []
# # for raw_input in raw_inuts:
# #     if raw_input.label is not None:
# #         cleaned_inputs.append(raw_input)
#
# max_seq_len = 512
# module = BertModel.from_pretrained('../bert')
# tokenizer = BertTokenizer.from_pretrained('../bert')
# label_list = processor.get_labels()
#
# convertToFeatures(raw_inuts, label_list, max_seq_len, tokenizer)
