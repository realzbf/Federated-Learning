import copy

import torch
from collections import OrderedDict


def average_weights(w):
    """计算参数平均值"""
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def flatten_weight(models):
    w = torch.Tensor()
    try:
        for m in models:
            for param in m.parameters():
                w = torch.cat((w, torch.flatten(param)))
    except TypeError:
        for param in models.parameters():
            w = torch.cat((w, torch.flatten(param)))
    return w
