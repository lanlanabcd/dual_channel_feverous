# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/12/14 22:11
# Description:

from base_templates import BasicModule
from transformers import AutoModel,AutoConfig
import torch
import torch.nn as nn

#using the cls of claim
class BertCls(BasicModule):
    def __init__(self, args):
        super(BertCls, self).__init__()
        self.config = AutoConfig.from_pretrained(args.bert_name, num_labels=len(args.id2label))
        hidden_size = self.config.hidden_size
        self.linear1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, len(args.id2label))

        self.init_weights()
        self.args = args

        self.bert = AutoModel.from_pretrained(args.bert_name, config=self.config)
        self.dropout = nn.Dropout(args.dropout)

        self.count_parameters()
        # self.print_modules()

    def forward(self, batch, args, test_mode):
        raw_data, input_ids, input_mask, label = batch

        outputs = self.bert(input_ids, attention_mask=input_mask)
        output = outputs[1]#.view(input_shape[0], -1)
        hg = self.dropout(output)
        hg = self.relu(self.linear1(hg))
        hg = self.linear2(hg)
        pred_logits = torch.log_softmax(hg, dim=-1)
        golds = label

        return pred_logits, golds
