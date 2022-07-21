# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/7/9 20:58
# Description:
import json
from data_generators import NewRobertaGenerator
import torch
import numpy as np
import dgl
import random
import os

class MultiTaskGenerator(NewRobertaGenerator):
    def __init__(self, input_path, tokenizer, root_dir, subdir, data_type, args):
        self.model_name = str(type(self))
        self.args = args
        self.data_type = data_type
        self.evi_labels_data = []
        self.cls_eye = np.eye(512, dtype=np.int)
        super(MultiTaskGenerator, self).__init__(input_path, tokenizer, root_dir, subdir, data_type, args)
        assert len(self.evi_labels_data) == len(self.labels)
        assert len(self.cls_mask) == len(self.labels)

    def get_cls_mask(self):
        cls_mask = []
        for seq_lens in self.seq_lens:
            cls_idx = 0
            cm = []
            for sl in seq_lens:
                cm.append(cls_idx)
                cls_idx += sl
            cls_mask.append(cm[1:])
        return cls_mask

    @classmethod
    def collate_fn(cls):
        return collate_fn

    def get_seq_inputs_from_inputs_lst_shuffle(self, tokenizer, max_len):
        shuffle_idx_lst = super(MultiTaskGenerator, self).get_seq_inputs_from_inputs_lst_shuffle(tokenizer, max_len)
        self.evi_labels = []
        for shuffle_idx, el in zip(shuffle_idx_lst, self.evi_labels_data):
            self.evi_labels.append(el[shuffle_idx])

        self.cls_mask = self.get_cls_mask()
        return shuffle_idx_lst

    def get_seq_inputs_from_inputs_lst_no_shuffle(self, tokenizer, max_len):
        super(MultiTaskGenerator, self).get_seq_inputs_from_inputs_lst_no_shuffle(tokenizer, max_len)
        if not self.evi_labels_data:
            self.get_evi_labels(self.data_type)
            self.evi_labels_data = np.array(self.evi_labels_data, dtype=np.long)
        self.evi_labels = self.evi_labels_data
        self.cls_mask = self.get_cls_mask()

    def get_evi_labels(self, data_type):
        if data_type == "test":
            test_sample_num = 500 if "small_" in self.args.data_dir else 19998
            self.evi_labels_data = [[-100 for _ in range(self.args.cla_evi_num)] for _ in range(test_sample_num)]
            return
        if "small" in self.args.data_dir:
            input_path = os.path.join(self.args.data_dir, "{}_temp.jsonl".format(data_type))
        else:
            input_path = "./data/new_data/{}_temp.jsonl".format(data_type)

        with open(input_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f.readlines()]
        for entry in data:
            if entry["label"] == "NOT ENOUGH INFO":
                self.evi_labels_data.append([-100 for _ in range(self.args.cla_evi_num)])
                continue
            gold_sets = []
            gold_evidences = entry["evidence"]
            for gold_evidence in gold_evidences:
                gold_set = set()
                for g_evi in gold_evidence:
                    gold_set.add((g_evi[2], g_evi[3]))
                gold_sets.append(gold_set)
            pred_evidences = [(evi[0], evi[1]) for evi in entry["predicted_evidence"]]

            evi_labels = [0]
            for pe in pred_evidences:
                evi_label = 0
                for gold_set in gold_sets:
                    if pe in gold_set:
                        evi_label = 1
                        break
                evi_labels.append(evi_label)

            if len(evi_labels) < self.args.cla_evi_num:
                evi_labels.extend([0 for _ in range(self.args.cla_evi_num-len(evi_labels))])

            self.evi_labels_data.append(evi_labels)

    def __getitem__(self, idx):
        raw_data, input_ids, input_mask, token_mask, claim_mask, graph, label = super(MultiTaskGenerator, self).__getitem__(idx)
        try:
            cls_mask = torch.tensor(self.cls_eye[self.cls_mask[idx]], dtype=torch.float32).to(self.args.device)
        except:
            cls_mask = None
        evi_labels = torch.tensor(self.evi_labels[idx][1:], dtype=torch.long).to(self.args.device)
        return raw_data, input_ids, input_mask, token_mask, claim_mask, cls_mask, graph, evi_labels, label

def collate_fn(batch):
    raw_data, input_ids, input_mask, token_mask, claim_mask, cls_mask, graph, evi_labels, label = map(list, zip(*batch))
    batched_raw_data = raw_data
    batch_input_ids = torch.stack(input_ids)
    batch_input_mask = torch.stack(input_mask)
    batched_token_mask = token_mask
    batched_claim_mask = claim_mask
    batched_cls_mask = torch.stack(cls_mask)
    batched_graph = [dgl.batch(list(g)) for g in zip(*graph)]
    if len(batched_graph) == 1:
        batched_graph = batched_graph[0]
    #batched_graph = dgl.batch(graph)
    batched_evi_labels = torch.stack(evi_labels)
    batched_label = torch.stack(label)

    return batch_input_ids, batch_input_mask, batched_token_mask\
        , batched_claim_mask, batched_cls_mask\
        , batched_graph, batched_evi_labels, batched_label
