# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/10/31 20:56
# Description:
from data_generators import BaseGenerator
import torch
from scipy import sparse
import dgl
import numpy as np

def collate_fn(batch):
    raw_data, input_ids, input_mask, claim_mask, label, subevis_fl_mask, graph = map(list, zip(*batch))
    batched_raw_data = raw_data
    batch_input_ids = torch.stack(input_ids)
    batch_input_mask = torch.stack(input_mask)

    batched_claim_mask = claim_mask
    batched_label = torch.stack(label)
    batched_subevis_fl_mask = subevis_fl_mask
    batched_graph = dgl.batch(graph)

    return batched_raw_data, batch_input_ids, batch_input_mask, batched_claim_mask, batched_label\
        , batched_subevis_fl_mask, batched_graph

class SubevisGnnGenerator(BaseGenerator):
    def __init__(self, input_path, tokenizer, cache_root_dir, cache_subdir, data_type, args):
        super(SubevisGnnGenerator, self).__init__(input_path, tokenizer, cache_root_dir, cache_subdir, data_type, args)

    @classmethod
    def collate_fn(cls):
        return collate_fn

    def get_dense_edges(self, entry, max_num):
        evi_num = sum([len(sg) for sg in entry["grouped_subevis"]])
        edges = np.zeros([evi_num, evi_num])
        stat_idx = 0
        for sg in entry["grouped_subevis"]:
            for en in range(stat_idx, stat_idx + len(sg)):
                for en2 in range(en, stat_idx+len(sg)):
                    edges[en][en2] = edges[en2][en] = 1
            stat_idx += len(sg)

        # self-loop
        for i in range(evi_num):
            edges[i][i] = 1

        edges = edges[:max_num, :max_num]
        return edges

    def edges2graph(self, edges):
        sparse_edges = sparse.csr_matrix(edges)
        graph = dgl.from_scipy(sparse_edges)
        return graph

    def get_graph(self, entry, max_num):
        return self.edges2graph(self.get_dense_edges(entry, max_num))

    def __getitem__(self, idx):
        raw_data = self.raw_data[idx]
        grouped_subevis = raw_data["grouped_subevis"]
        flatened_subevis = [e for s_g in grouped_subevis for e in s_g]
        input_ids_lst, mask_dict = self.get_plm_inputs_lst(
            raw_data["claim"], flatened_subevis, self.tokenizer, return_mask=["mask_a", "fl_cls_mask_b"])
        input_ids, input_mask = self.get_plm_inputs_from_input_ids(
            input_ids_lst, pad_idx=self.tokenizer.pad_token_id, end_idx=self.tokenizer.eos_token_id)
        input_ids = torch.tensor(input_ids).to(self.args.device)
        input_mask = torch.tensor(input_mask).to(self.args.device)
        claim_mask = torch.tensor(mask_dict["mask_a"], dtype=torch.float32).to(self.args.device)
        subevis_fl_mask = torch.tensor(mask_dict["fl_cls_mask_b"], dtype=torch.float32).to(self.args.device)
        label = torch.tensor(self.labels[idx]).to(self.args.device)

        graph = self.get_graph(raw_data, len(subevis_fl_mask))

        return raw_data, input_ids, input_mask, claim_mask, label, subevis_fl_mask, graph


class MyArgs():
    def __init__(self):
        self.device = "cpu"

if __name__ == "__main__":
    from tqdm import tqdm
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("../bert_weights/roberta-large")
    input_path = "../data/soft_subevis_data/train.jsonl"

    args = MyArgs()
    generator = SubevisGnnGenerator(input_path, tokenizer, None, None, "train", args)

    data_len = generator.__len__()
    print(generator.__getitem__(0))
    print(data_len)
    for i in tqdm(range(data_len)):
        generator.__getitem__(i)
