# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/11/1 17:03
# Description:

from data_generators import SubevisGnnGenerator
import torch
import dgl
import random
from tqdm import tqdm

def collate_fn(batch):
    raw_data, input_ids, input_mask, claim_mask, label, subevis_fl_mask, graph, evi_labels = map(list, zip(*batch))
    batched_raw_data = raw_data
    batch_input_ids = torch.stack(input_ids)
    batch_input_mask = torch.stack(input_mask)

    batched_claim_mask = claim_mask
    batched_label = torch.stack(label)
    batched_subevis_fl_mask = subevis_fl_mask
    batched_graph = dgl.batch(graph)
    batched_evi_labels = evi_labels

    return batched_raw_data, batch_input_ids, batch_input_mask, batched_claim_mask, batched_label\
        , batched_subevis_fl_mask, batched_graph, batched_evi_labels

class SubevisGnnWLabelGenerator(SubevisGnnGenerator):
    def __init__(self, input_path, tokenizer, cache_root_dir, cache_subdir, data_type, args):
        super(SubevisGnnWLabelGenerator, self).__init__(input_path, tokenizer, cache_root_dir, cache_subdir, data_type, args)

    @classmethod
    def collate_fn(cls):
        return collate_fn

    def shuffle_evis(self):
        for entry in tqdm(self.raw_data, desc="shuffle evidences"):
            try:
                c = list(zip(entry["grouped_subevis"], entry["grouped_subevis_label"]))
                random.shuffle(c)
                entry["grouped_subevis"], entry["grouped_subevis_label"] = zip(*c)
            except:
                entry["grouped_subevis"] = [['']]
                entry["grouped_subevis_label"] = [[1]]
            #random.shuffle(entry["grouped_subevis"])

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

        #返回证据标签，subevis
        flatened_subevis_label = [e for s_g in raw_data["grouped_subevis_label"] for e in s_g]
        evi_labels = torch.tensor([i for i,lb in enumerate(flatened_subevis_label[:len(subevis_fl_mask)]) if lb == 1]).to(self.args.device)

        return raw_data, input_ids, input_mask, claim_mask, label, subevis_fl_mask, graph, evi_labels

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
    generator.shuffle_evis()

    data_len = generator.__len__()
    print(generator.__getitem__(0))
    print(data_len)
    for i in tqdm(range(data_len)):
        generator.__getitem__(i)
