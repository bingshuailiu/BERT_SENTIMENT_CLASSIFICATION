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
from utils.LabelSmoothing import smooth_one_hot
from torch.utils.data import DataLoader
import logging
from argparse import ArgumentParser

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def getParser():
    parser = ArgumentParser()
    parser.add_argument('--train_dir', default="../data/train_small.csv", type=str)
    parser.add_argument('--test_dir', default='../data/test_small.csv', type=str)
    parser.add_argument('--bert_dir', default="../ptms/bert-chinese", type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--use_label_smoothing', default=True, type=bool)
    parser.add_argument('--smoothing_rate', default=0.25, type=float)
    parser.add_argument('--class_size', default=3, type=int)
    return parser.parse_args()


def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    model = BertSentimentClassifier(args.bert_dir).to(device)
    # 准备训练数据
    processor = SentimentPreprocessor()
    train_inputs = processor.get_train_examples(args.train_dir)
    # test_inputs = processor.get_test_examples(args.test_dir)

    label_list = processor.get_labels()
    train_features = convertToFeatures(train_inputs, label_list, args.max_seq_len, tokenizer)
    train_dataset = SentimentDataSet(train_features, 'train', device)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    model.train()

    loss_fn = nn.CrossEntropyLoss()
    lr = 5e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    train_loss_his = []
    train_acc_his = []
    train_data_len = len(train_dataset)
    step = 0
    for i in range(args.epoch):
        print(f"---------epoch {i + 1}---------")
        total_train_acc = 0.0
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        for step, batch_data in enumerate(bar):
            output = model(**batch_data)

            if args.use_label_smoothing:
                # smoothed_label = smooth_one_hot(batch_data['label_id'], args.class_size, args.smoothing_rate)
                # loss = loss_fn(output, smoothed_label)
                loss = loss_fn(output, batch_data['label_id'])
            else:
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
    train(getParser())
