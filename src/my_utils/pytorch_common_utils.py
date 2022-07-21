# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/10/21 20:52
# Description:
import torch
import random
import numpy as np
def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

import torch.optim as optim
from transformers import AdamW

def get_optimizer(model, lr, weight_decay = 1e-4, fix_bert = False):
    bert_params, task_params = [], []
    size = 0
    for name, params in model.named_parameters():
        if "bert" in name:
            bert_params.append((name, params))
        else:
            task_params.append((name, params))
        size += params.nelement()

    # print("bert parameters")
    # for name, params in bert_params:
    #     print('n: {}, shape: {}'.format(name, params.shape))
    # print('*' * 150)
    print("task parameters")
    for name, params in task_params:
        print('n: {}, shape: {}'.format(name, params.shape))
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    nobert_params = task_params

    if fix_bert:
        params = [
            {"params": nobert_params, "weight_decay": weight_decay,"lr": 100 * lr},
        ]
    else:
        params = [
            {"params": [p for n, p in bert_params if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay, "lr": lr},
            {"params": [p for n, p in bert_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0, "lr": lr},
            {
                "params": [p for n, p in nobert_params],
                "weight_decay": weight_decay,
                "lr": 100*lr
                # "lr": lr
            },
        ]

    optimizer = AdamW(params, correct_bias=False)
    return optimizer