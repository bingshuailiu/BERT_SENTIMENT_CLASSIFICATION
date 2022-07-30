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
from bert_transformer import BertAdam
from torch.nn import MSELoss, CrossEntropyLoss

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
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--T', default=1.0, type=float)
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

    train_steps = int(
        len(train_features) / args.batch_size * args.epoch
    )
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num steps = %d", len(train_features) * args.epoch)

    if not args.do_eval:
        teacher_model = TinyBertForSequenceClassification.from_pretrained(args.teacher_model)
    student_model = TinyBertForSequenceClassification.from_pretrained(args.student_model)
    if not args.do_eval:
        param_optimizer = list(student_model.named_parameters())
        size = 0
        for n, p in student_model.named_parameters():
            logger.info('n: {}'.format(n))
            size += p.element()
        logger.info('Total parameters: {}'.format(size))
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             schedule='none',
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=train_steps
                             )
        loss_mse = MSELoss()
        global_step = 0
        for i in range(args.epoch):
            print(f"---------epoch {i + 1}---------")
            tr_loss = 0.0
            tr_attention_loss = 0.0
            tr_hidden_state_loss = 0.0
            tr_cls_loss = 0.0
            bar = tqdm(train_dataloader, total=len(train_dataloader))
            for step, batch_data in enumerate(bar):
                attention_loss = 0.0
                hidden_state_loss = 0.0
                cls_loss = 0.0

                # 学生模型的输出
                student_model_output = student_model(**batch_data, is_student=True)
                student_logits = student_model_output[0]
                student_attention_output = student_model_output[1]
                student_sequence_output = student_model_output[2]
                # 老师模型的输出
                with torch.no_grad():
                    teacher_model_output = teacher_model(**batch_data, is_student=False)
                    teacher_logits = teacher_model_output[0]
                    teacher_attention_output = teacher_model_output[1]
                    teacher_sequence_output = teacher_model_output[2]
                # 两种情况计算Loss
                if not args.pred_distill:
                    # 蒸馏transformer部分 需要计算的loss分为两部分
                    # 1. attention部分的loss
                    # 2. hidden_state部分的loss
                    student_layer_num = len(student_attention_output)
                    teacher_layer_num = len(teacher_attention_output)
                    # 这里是映射函数的实现, 映射的方式可以随意定义, 论文里的实现如下:
                    # g(m) = 3 * m 即每三层选取一层
                    assert len(teacher_layer_num) / len(student_layer_num) >= 3
                    new_teacher_attention = []
                    new_teacher_sequence = []
                    for i in range(student_layer_num):
                        new_teacher_attention.append(teacher_attention_output[i*3])
                        new_teacher_sequence.append(teacher_sequence_output[i*3])
                    for student_attention, teacher_attention in zip(student_attention_output, new_teacher_attention):
                        attention_loss += MSELoss(student_attention, teacher_attention)
                    for student_sequence, teacher_sequence in zip(student_sequence_output, new_teacher_sequence):
                        hidden_state_loss += MSELoss(student_sequence, teacher_sequence)
                    loss = hidden_state_loss + attention_loss
                    tr_attention_loss += attention_loss.item()
                    tr_hidden_state_loss += hidden_state_loss.item()
                else:
                    # 蒸馏分类器的部分 使用交叉熵计算loss
                    # cross_entropy(student_logits / T, teacher_logits / T)
                    # 经过论文作者的实验, T最好的值是1.....
                    cls_loss = CrossEntropyLoss(student_logits / args.T, teacher_logits / args.T)
                    loss = cls_loss
                    tr_cls_loss += cls_loss.item()
                loss.backward()
                tr_loss += loss.item()

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if(global_step + 1) % args.eval_step == 0:
                    # 评估模型部分
                    student_model.eval()
                    loss = tr_loss / (step + 1)

                    continue
                # bar.set_description("epoch={}\tindex={}\tloss={:.6f}".format(i + 1, step, loss))

    return


if __name__ == '__main__':
    args = getParser()
    from bert_transformer.modeling import TinyBertForSequenceClassification

    student_model = TinyBertForSequenceClassification.from_pretrained(args.student_model, 3)
    # student_model = TinyBert.from_pretrained(args.student_model)
    student_model.to(torch.device("cuda:0"))
