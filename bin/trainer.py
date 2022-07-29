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
from utils.LabelSmoothing import smooth_one_hot, label_to_one_hot
from torch.utils.data import DataLoader
import logging
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from model import TinyBert

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def getParser():
    parser = ArgumentParser()
    parser.add_argument('--train_dir', default="../data/train_small.csv", type=str)
    parser.add_argument('--test_dir', default='../data/test_small.csv', type=str)
    parser.add_argument('--bert_dir', default="../ptms/bert-chinese", type=str)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--epoch', default=1, type=int)
    parser.add_argument('--use_label_smoothing', default=True, type=bool)
    parser.add_argument('--smoothing_rate', default=0.25, type=float)
    parser.add_argument('--class_size', default=3, type=int)
    parser.add_argument('--is_detach', default=True, type=bool)
    parser.add_argument('--teacher_model', default="../ptms/bert-chinese", type=str)
    parser.add_argument('--student_model', default="../ptms/tiny-bert", type=str)
    parser.add_argument('--pred_distill', default=True, type=bool)
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
            output = model(**batch_data, is_detach=args.is_detach)

            if args.use_label_smoothing:
                one_hots = label_to_one_hot(batch_data['label_id'], args.class_size, device)
                smoothed_label = smooth_one_hot(one_hots, args.class_size, args.smoothing_rate)
                loss = loss_fn(output, smoothed_label.to(device))
            else:
                loss = loss_fn(output, batch_data['label_id'])

            acc = (output.argmax(1) == batch_data['label_id']).sum()
            total_train_acc += acc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            train_loss_his.append(float(loss))
            bar.set_description("epoch={}\tindex={}\tloss={:.6f}".format(i + 1, step, loss))
        total_train_acc = total_train_acc / train_data_len
        train_acc_his.append(float(total_train_acc))
        print(f"训练集准确率：{total_train_acc}")
    return train_loss_his, train_acc_his


def draw(save_path, data1, data2, label1, label2, title, x, y):
    plt.figure(figsize=(10, 6), dpi=200)
    size = [i for i in range(len(data1))]
    plt.plot(size, data1, color='red', lw=2, label=label1)
    plt.plot(size, data2, color='blue', lw=2, label=label2)
    plt.xlabel(x, fontsize=15)
    plt.ylabel(y, fontsize=15)
    plt.legend([label1, label2], fontsize=12.5)
    plt.title(title)
    plt.savefig(save_path + '.png')


def train_plot():
    args = getParser()
    args.is_detach = False
    simple_loss, simple_acc = train(args)
    args.is_detach = True
    ls_loss, ls_acc = train(args)
    draw(
        save_path='../results/detach_loss',
        data1=simple_loss,
        data2=ls_loss,
        label1="Simple",
        label2="Use Detach",
        title="Loss of use detach and simple",
        x="Step",
        y="Loss"
    )
    draw(
        save_path='../results/detach_acc',
        data1=simple_acc,
        data2=ls_acc,
        label1="Simple",
        label2="Use Detach",
        title="Acc of use detach and simple",
        x="Epoch",
        y="Acc"
    )

    args.use_label_smoothing = False
    simple_loss, simple_acc = train(args)
    args.use_label_smoothing = True
    ls_loss, ls_acc = train(args)

    draw(
        save_path='../results/ls_loss',
        data1=simple_loss,
        data2=ls_loss,
        label1="Simple",
        label2="Label Smoothing",
        title="Loss of label smoothing and simple",
        x="Step",
        y="Loss"
    )
    draw(
        save_path='../results/ls_acc',
        data1=simple_acc,
        data2=ls_acc,
        label1="Simple",
        label2="Label Smoothing",
        title="Acc of label smoothing and simple",
        x="Epoch",
        y="Acc"
    )


def distilling(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(args.student_model)
    # 准备训练数据
    processor = SentimentPreprocessor()
    train_inputs = processor.get_train_examples(args.train_dir)

    label_list = processor.get_labels()
    train_features = convertToFeatures(train_inputs, label_list, args.max_seq_len, tokenizer)
    train_dataset = SentimentDataSet(train_features, 'train', device)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num steps = %d", len(train_features)*args.epoch)

    if not args.pred_distill:
        teacher_model = TinyBert.from_pretrained(args.teacher_model)
    student_model = TinyBert.from_pretrained(args.student_model)

    return


if __name__ == '__main__':
    args = getParser()
    from bert_transformer.modeling import TinyBertForSequenceClassification
    student_model = TinyBertForSequenceClassification.from_pretrained(args.student_model, 3)
    # student_model = TinyBert.from_pretrained(args.student_model)
    student_model.to(torch.device("cuda:0"))
