# -*- coding: utf-8 -*-
# @Time : 2022/7/24 17:04
# @Author : Bingshuai Liu
import numpy as np
import math


def beam_search(nodes, topk=1):
    # log-likelihood可以相加
    paths = {'A': math.log(nodes[0]['A']), 'B': math.log(nodes[0]['B']), 'C': math.log(nodes[0]['C'])}
    calculations = []
    for l in range(1, len(nodes)):
        # 拷贝当前路径
        paths_ = paths.copy()
        paths = {}
        nows = {}
        cur_cal = 0
        for i in nodes[l].keys():
            # 计算到达节点i的所有路径
            for j in paths_.keys():
                nows[j + i] = paths_[j] + math.log(nodes[l][i])
                cur_cal += 1
        calculations.append(cur_cal)
        # 选择topk条路径
        indices = np.argpartition(list(nows.values()), -topk)[-topk:]
        # 保存topk路径
        for k in indices:
            paths[list(nows.keys())[k]] = list(nows.values())[k]

    print(f'calculation number {calculations}')
    return paths


# nodes = [{'A': 0.1, 'B': 0.3, 'C': 0.6}, {'A': 0.2, 'B': 0.4, 'C': 0.4}, {'A': 0.6, 'B': 0.2, 'C': 0.2},
#          {'A': 0.3, 'B': 0.3, 'C': 0.4}]
# print(beam_search(nodes, topk=2))
