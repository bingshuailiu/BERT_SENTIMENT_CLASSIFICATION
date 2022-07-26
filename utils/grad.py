# -*- coding: utf-8 -*-
# @Time : 2022/7/26 0:28
# @Author : Bingshuai Liu
import torch
import matplotlib.pyplot as plt


def forward(x):  # y^ = wx
    return x * w  # w是tensor 所以 这个乘法自动转换为tensor数乘 , x被转化成tensor 这里构建了一个计算图


def loss(x, y):  # 计算单个的误差 : 损失
    '''
    每调用一次loss函数,计算图自动构建一次
    :param x:
    :param y:
    :return:
    '''
    y_pred = forward(x)
    return (y_pred - y) ** 2


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = torch.tensor([1.0])  # 假设 w = 1.0的情况
w.requires_grad = True

eli = []
lli = []
print('predict (before training)', 4, forward(4).item())
for epoch in range(100):  # 每轮输出 w的值和损失 loss
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()  # 自动求梯度
        print('\tgrad:', x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data  # 权重的数值更新,纯数值的修改 如果不用.data会新建计算图
        # 如果这里想求平均值 其中的累加操作 要写成sum += l.item()
        w.grad.data.zero_()  # 清空权重里梯度的数据,不然梯度会累加
    eli.append(epoch)
    lli.append(l.item())
    print('progress:', epoch, l.item())
print('Predict (after training)', 4, forward(4).item())
