# -*- coding: utf-8 -*-
# @Time : 2022/7/23 22:20
# @Author : Bingshuai Liu
import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
from transformers import logging
from preprocess.preprocess import SentimentPreprocessor, SentimentRawInput, SentimentFeatures, convertToFeatures
from model.BertSentiment import BertSentimentClassifier
from preprocess.SentimentDataSet import SentimentDataSet
from torch.utils.data import DataLoader
import logging


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def train():
    # 路径和超参
    TRAIN_DIR = "./data/train_small.csv"
    BERT_DIR = "./bert"
    bs = 1
    max_seq_len = 128
    epoch = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(BERT_DIR)
    model = BertSentimentClassifier(BERT_DIR).to(device)
    # 准备训练数据
    processor = SentimentPreprocessor()
    train_inputs = processor.get_train_examples(TRAIN_DIR)
    # test_inputs = processor.get_test_examples(TEST_DIR)

    label_list = processor.get_labels()
    train_features = convertToFeatures(train_inputs, label_list, max_seq_len, tokenizer)
    train_dataset = SentimentDataSet(train_features, 'train', device)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)

    model.train()

    loss_fn = nn.CrossEntropyLoss()
    lr = 5e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    train_loss_his = []
    train_acc_his = []
    train_data_len = len(train_dataset)
    step = 0
    for i in range(epoch):
        print(f"---------epoch {i+1}---------")
        total_train_acc = 0.0
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        for step, batch_data in enumerate(bar):
            output = model(**batch_data)
            loss = loss_fn(output, batch_data['label_id'])
            acc = (output.argmax(1) == batch_data['label_id']).sum()
            total_train_acc += acc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            train_loss_his.append(loss)
            bar.set_description("epoch={}\tindex={}\tloss={:.6f}".format(i, step, loss))
        total_train_acc = total_train_acc / train_data_len
        train_acc_his.append(total_train_acc)
        print(f"训练集准确率：{total_train_acc}")
    return


if __name__ == '__main__':
    train()
