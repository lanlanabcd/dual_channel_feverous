# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2022/1/3 16:00
# Description:
from transformers import TapasConfig, TapasModel
from base_templates import BasicModule
import torch
import torch.nn as nn

class TapasCls(BasicModule):
    def __init__(self, args):
        super(TapasCls, self).__init__()
        self.config = TapasConfig.from_pretrained(args.bert_name, num_labels=len(args.id2label))
        hidden_size = self.config.hidden_size
        self.linear1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, len(args.id2label))

        self.init_weights()
        self.args = args

        self.bert = TapasModel.from_pretrained(args.bert_name, config=self.config)
        self.dropout = nn.Dropout(args.dropout)

        self.count_parameters()


    def forward(self, batch, args, test_mode):
        raw_data, input_ids, attention_mask, token_type_ids, labels = batch

        outputs = self.bert(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        output = outputs[1]#.view(input_shape[0], -1)
        hg = self.dropout(output)
        hg = self.relu(self.linear1(hg))
        hg = self.linear2(hg)

        pred_logits = torch.log_softmax(hg, dim=-1)
        golds = labels

        return pred_logits, golds