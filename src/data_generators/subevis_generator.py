# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/8/30 16:52
# Description:

import json

import config
from data_generators import MultiTaskGenerator
import torch
import numpy as np
import dgl
import random
import os
from tqdm import tqdm
from utils import get_subevis_graph

class SubevisGenerator(MultiTaskGenerator):
    def __init__(self, input_path, tokenizer, root_dir, subdir, data_type, args):
        args.graph_types = "subevis"
        super(SubevisGenerator, self).__init__(input_path, tokenizer, root_dir, subdir, data_type, args)
        self.model_name = str(type(self))
        self.rsm_eye = np.eye(self.args.cla_evi_num, dtype = np.int)
        self.reverse_shuffle_mask = [np.array(list(range(args.cla_evi_num)))] * len(self.labels)
        assert len(self.reverse_shuffle_mask) == len(self.labels)

    def register_graph_func(self):
        self.graph_type2func["subevis"] = self.process_subevis_graphs

    def get_evi_labels(self, data_type):
        self.evi_labels_data = [[0]*self.args.cla_evi_num] * len(self.labels)

    def process_subevis_graphs(self, data, graph_type, graph_func = get_subevis_graph):
        graphs = []
        labels = []
        for entry in tqdm(data, desc="process {} graphs".format(graph_type)):
            graph, graph_words = graph_func(entry, self.args)
            graphs.append(graph)
            labels.append(config.label2idx[entry['label']])

        return graphs, labels

    @classmethod
    def collate_fn(cls):
        return collate_fn

    def get_seq_inputs_from_inputs_lst_shuffle(self, tokenizer, max_len):
        shuffle_idx_lst = super(SubevisGenerator, self).get_seq_inputs_from_inputs_lst_shuffle(tokenizer, max_len)
        self.reverse_shuffle_mask = []
        for shuffle_idx in shuffle_idx_lst:
            self.reverse_shuffle_mask.append(shuffle_idx)
        assert len(self.reverse_shuffle_mask) == len(self.labels)

    def get_seq_inputs_from_inputs_lst_no_shuffle(self, tokenizer, max_len):
        super(SubevisGenerator, self).get_seq_inputs_from_inputs_lst_no_shuffle(tokenizer, max_len)
        self.reverse_shuffle_mask = []
        shuffle_idx_lst = [np.arange(self.args.cla_evi_num)] * len(self.labels)
        for shuffle_idx in shuffle_idx_lst:
            self.reverse_shuffle_mask.append(shuffle_idx)
        assert len(self.reverse_shuffle_mask) == len(self.labels)

    #def get_cls_mask(self):
    #    cls_mask = []
    #    for seq_lens in self.seq_lens:
    #        cls_idx = 0
    #        cm = []
    #        for sl in seq_lens:
    #            cm.append(cls_idx)
    #            cls_idx += sl
    #        cls_mask.append(cm)
    #    return cls_mask

    def get_cls_mask(self):
        cls_mask = []
        for seq_lens in self.seq_lens:
            cls_idx = 0
            cm = []
            cm2 = []
            for sl in seq_lens:
                cm.append(cls_idx)
                cls_idx += sl
                cm2.append(cls_idx-1)
            cls_mask.append([cm, cm2])
        return cls_mask

    def __getitem__(self, idx):
        raw_data, input_ids, input_mask, token_mask, claim_mask, cls_mask, graph, evi_labels, label = super(SubevisGenerator, self).__getitem__(idx)
        reverse_shuffle_mask = torch.tensor(self.rsm_eye[self.reverse_shuffle_mask[idx]], dtype=torch.float32).T.to(self.args.device)
        # (F+L)/2
        cls_mask = (torch.tensor(self.cls_eye[self.cls_mask[idx][0]], dtype=torch.float32).to(self.args.device) + \
                    torch.tensor(self.cls_eye[self.cls_mask[idx][1]], dtype=torch.float32).to(self.args.device)) / 2
        return raw_data,input_ids, input_mask, token_mask, claim_mask, cls_mask, reverse_shuffle_mask, graph, evi_labels, label

def collate_fn(batch):
    raw_data, input_ids, input_mask, token_mask, claim_mask, cls_mask, reverse_shuffle_mask, graph, evi_labels, label = map(list, zip(*batch))
    batched_raw_data = raw_data
    batch_input_ids = torch.stack(input_ids)
    batch_input_mask = torch.stack(input_mask)
    batched_token_mask = token_mask
    batched_claim_mask = claim_mask
    batched_cls_mask = torch.stack(cls_mask)
    batched_reverse_shuffle_mask = torch.stack(reverse_shuffle_mask)
    batched_graph = [dgl.batch(list(g)) for g in zip(*graph)]
    if len(batched_graph) == 1:
        batched_graph = batched_graph[0]
    #batched_graph = dgl.batch(graph)
    batched_evi_labels = torch.stack(evi_labels)
    batched_label = torch.stack(label)

    return batched_raw_data, batch_input_ids, batch_input_mask, batched_token_mask\
        , batched_claim_mask, batched_cls_mask\
        , batched_reverse_shuffle_mask, batched_graph, batched_evi_labels, batched_label