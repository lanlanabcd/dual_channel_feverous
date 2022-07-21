# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/9/18 22:13
# Description:
# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/8/30 16:52
# Description:

from data_generators import SubevisGenerator
import torch
import dgl
from utils import load_jsonl_data, refine_obj_data
import os


class SubevisSepGenerator(SubevisGenerator):
    def __init__(self, input_path, tokenizer, root_dir, subdir, data_type, args):
        super(SubevisSepGenerator, self).__init__(input_path, tokenizer, root_dir, subdir, data_type, args)
        self.model_name = str(type(self))
        self.get_evi_mask(data_type)
        assert len(self.evidences_mask) == len(self.labels), print(len(self.evidences_mask), len(self.labels))

    @classmethod
    def collate_fn(cls):
        return collate_fn

    def get_evi_mask(self, data_type):
        evi_mask_data = load_jsonl_data(os.path.join(self.args.data_dir, f"{data_type}.jsonl"))
        evi_mask = refine_obj_data(evi_mask_data, "evidences_mask")
        self.evidences_mask = []
        for em in evi_mask:
            em_idxs = []
            em = em["evidences_mask"][:self.args.cla_evi_num-1]
            for idx, m in enumerate(em):
                if m == 1:
                    em_idxs.append(idx)
            self.evidences_mask.append(em_idxs)

    def get_cls_mask(self):
        cls_mask = []
        for seq_lens in self.seq_lens:
            cls_idx = 0
            cm = []
            cm2 = []
            for sl in seq_lens:
                cm.append(cls_idx)
                cls_idx += sl
                cm2.append(cls_idx - 1)
            cls_mask.append([cm, cm2])
        return cls_mask

    def __getitem__(self, idx):
        raw_data, input_ids, input_mask, token_mask, claim_mask, _, reverse_shuffle_mask, graph, evi_labels, label = super(SubevisSepGenerator,
                                                                                           self).__getitem__(idx)
        cls_mask = (torch.tensor(self.cls_eye[self.cls_mask[idx][0]], dtype=torch.float32).to(self.args.device) + \
                    torch.tensor(self.cls_eye[self.cls_mask[idx][1]], dtype=torch.float32).to(self.args.device)) / 2
        evi_num = len([1 for evi_stat, evi_end in zip(self.cls_mask[idx][0], self.cls_mask[idx][1]) if
                       evi_end - evi_stat == 1]) - 1
        evi_mask = torch.tensor(self.evidences_mask[idx], dtype=torch.int64).to(self.args.device)
        return raw_data, input_ids, input_mask, token_mask, claim_mask, cls_mask, reverse_shuffle_mask, evi_num, evi_mask, graph, evi_labels, label


def collate_fn(batch):
    raw_data, input_ids, input_mask, token_mask, claim_mask, cls_mask, reverse_shuffle_mask, evi_num, evi_mask, graph, evi_labels, label = map(
        list, zip(*batch))
    batched_raw_data = raw_data
    batch_input_ids = torch.stack(input_ids)
    batch_input_mask = torch.stack(input_mask)
    batched_token_mask = token_mask
    batched_claim_mask = claim_mask
    batched_cls_mask = torch.stack(cls_mask)
    batched_reverse_shuffle_mask = torch.stack(reverse_shuffle_mask)
    batched_evi_mask = evi_mask
    batch_evi_num = evi_num
    batched_graph = [dgl.batch(list(g)) for g in zip(*graph)]
    if len(batched_graph) == 1:
        batched_graph = batched_graph[0]
    # batched_graph = dgl.batch(graph)
    batched_evi_labels = torch.stack(evi_labels)
    batched_label = torch.stack(label)

    return batched_raw_data, batch_input_ids, batch_input_mask, batched_token_mask \
        , batched_claim_mask, batched_cls_mask \
        , batched_reverse_shuffle_mask, batch_evi_num, batched_evi_mask, batched_graph, batched_evi_labels, batched_label