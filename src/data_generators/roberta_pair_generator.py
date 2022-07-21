# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/6/8 15:39
# Description:

from data_generators import BaseGraphGenerator
import torch
import dgl
import os
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info
from utils.get_dgl_graph import get_srl_graph, get_dep_graph
import json
import config
from tqdm import tqdm
import pickle as pkl
from transformers import AutoTokenizer

class RobertaPairGenerator(BaseGraphGenerator):
    def __init__(self, input_path, tokenizer, root_dir, subdir, data_type, args):
        super(RobertaPairGenerator, self).__init__(tokenizer, data_type, args)
        self.bert_tokenizer = AutoTokenizer.from_pretrained("./bert_weights/bert-base-uncased")

        self.input_ids = []
        self.input_mask = []
        self.token_mask_lst = []

        self.labels = []

        # 只包含长度不为0的句子长度，不进行padding
        self.sent_lens = []

        assert 'roberta' in args.bert_name

        self.get_flat_tokens(input_path, root_dir, subdir)

        assert len(self.labels) != []
        assert len(self.input_ids) == len(self.sent_lens), print(len(self.input_ids), len(self.sent_lens))
        assert len(self.input_ids) == len(self.input_mask)
        assert len(self.input_ids) == len(self.labels)
        assert len(self.input_ids) == len(self.token_mask_lst)

        self.graphs, _ = self.process_graphs(input_path, root_dir, subdir)

        assert len(self.graphs) == len(self.labels), print(len(self.graphs), len(self.labels))

        # 在图里节点的索引，所以不用加偏置1
        self.claim_mask = [list(range(sent_lens[0])) for sent_lens in self.sent_lens]

        # 某些模型从bert的embedding中取词，所以需要加上偏置1
        if args.model_type in ["FlatClaimAvg"]:
            self.claim_mask = [[idx + 1 for idx in idxs] for idxs in self.claim_mask]
        #self.print_example()

    def print_example(self):
        example = self.__getitem__(0)
        for item in example:
            print(item)
            print('\n')

    def process_graphs(self, input_path, root_dir, subdir):
        if self.args.main_file == "arg_masker":
            return [dgl.DGLGraph()]*len(self.labels), None

        data_dir = os.path.join(root_dir, subdir)
        if self.args.single_evidence:
            graph_path = os.path.join(data_dir, "single_roberta_srl_graphs.bin")
        elif self.args.cluster_graph_words:
            graph_path = os.path.join(data_dir, "clustered_srl_graphs.bin")
        elif self.args.use_gmn:
            if "dep_data" in self.args.data_dir:
                graph_path = os.path.join(data_dir, "gmn_dep_graphs.bin")
            elif "srl_data" in self.args.data_dir:
                graph_path = os.path.join(data_dir, "gmn_srl_graphs.bin")
            else:
                assert False
        else:
            graph_path = os.path.join(data_dir, "roberta_srl_graphs.bin")

        if self.args.evi_len_file:
            graph_path = "{}_{}.bin".format(graph_path.split('/')[-1].split(".")[0], self.args.evi_len_file.split(".")[0])
            graph_path = os.path.join(data_dir, graph_path)

        print("graph_path:", graph_path)

        if os.path.exists(graph_path) and not self.args.force_generate:
            graphs, label_dict = load_graphs(graph_path)
            labels = label_dict["labels"]
            print(graphs[0])
            return graphs, labels

        graphs = []
        labels = []

        with open(input_path, "r", encoding='utf-8') as f:
            data = f.readlines()
            data = [json.loads(line.strip()) for line in data]
            #data = data[:1000]

        for entry,sent_lens in tqdm(zip(data,self.sent_lens), desc="process graphs"):
            if "srl_data" in self.args.data_dir:
                srl_dicts = [entry["claim_srl"]]
                srl_dicts += entry["evidences_srl"]
                words = [entry["claim_words"]]
                words += entry["evidences_words"]
                graph, graph_words = get_srl_graph(srl_dicts, words, sent_lens)

            elif "dep_data" in self.args.data_dir:
                words = [entry["claim_words"]]
                words += entry["evidences_words"]

                dep_dicts = [entry["claim_dep"]]
                dep_dicts += entry["evidences_dep"]

                if len(dep_dicts) > self.args.cla_evi_num:
                    dep_dicts = dep_dicts[:self.args.cla_evi_num]
                graph, _ = get_dep_graph(dep_dicts, words, sent_lens)
                #graph, _ = get_fc_graph(dep_dicts, words, sent_lens)
            else:
                assert False

            graphs.append(graph)
            labels.append(config.label2idx[entry['label']])

        labels = torch.tensor(labels)
        save_graphs(graph_path, graphs, {'labels': labels})
        print(graphs[0])
        return graphs, labels

    def get_flat_tokens(self, input_path, root_dir, subdir):
        assert not (self.args.single_evidence and self.args.evi_len_file)
        data_dir = os.path.join(root_dir, subdir)

        info_path = os.path.join(data_dir, 'roberta_pair_info.pkl')

        if os.path.exists(info_path) and not self.args.force_generate:
            info_dict = load_info(info_path)
            self.input_ids = info_dict["input_ids"]
            self.input_mask = info_dict["input_mask"]
            self.labels = info_dict["labels"]
            self.token_mask_lst = info_dict["token_mask_lst"]
            self.sent_lens = info_dict["sent_lens"]
            print(self.tokenizer.decode(self.input_ids[0][0]))
            return

        with open(input_path, "r", encoding='utf-8') as f:
            data = f.readlines()
            data = [json.loads(line.strip()) for line in data]

        if self.args.evi_len_file is not None:
            with open(self.args.evi_len_file+"."+self.data_type, "rb") as f:
                evi_len_lst = pkl.load(f)
        else:
            evi_len_lst = [5] * len(data)

        max_seq_len = self.args.max_seq_len
        for evi_len, entry in tqdm(zip(evi_len_lst, data), desc="get_flat_tokens"):
            token_list = []
            cls_idxs = [0]

            claim = ' '.join(self.bert_tokenizer.tokenize(entry["claim"].lower())).replace(' ##', '').replace('[UNK]',
                                                                                                         '<unk>')
            claim_words = self.tokenizer.tokenize(claim)

            token_list.append(claim_words)

            evidences = [evi_t.strip() + " : " + evi for evi, evi_t in
                         zip(entry["evidences"], entry['evidences_title'])]

            for evi in evidences:
                words = self.tokenizer.tokenize(evi.lower())
                token_list.append(words)

            while len(token_list) < self.args.cla_evi_num:
                token_list.append('')

            if len(token_list) > self.args.cla_evi_num:
                token_list = token_list[:self.args.cla_evi_num]

            while sum([len(words) for words in token_list]) + len(token_list) * 2 > max_seq_len:
                max_idx = 1
                max_len = 0
                for i in range(1, len(token_list)):
                    if len(token_list[i]) > max_len:
                        max_len = len(token_list[i])
                        max_idx = i
                token_list[max_idx].pop()

            token_list = token_list[:evi_len+1]

            claim_tokens = token_list[0]
            evis_token_lst = token_list[1:]

            sent_lens = []
            input_ids_lst = []
            input_mask_lst = []
            token_mask_lst = []
            for evi_tokens in evis_token_lst:
                input_ids, input_mask, mask_dict = self.get_plm_inputs(claim_tokens, evi_tokens, self.tokenizer
                                                                       , return_mask=["word_mask_a", "word_mask_b"])
                if not sent_lens:
                    sent_lens.append(mask_dict["sent_len_a"])
                    token_mask_lst.append(mask_dict["word_mask_a"])
                sent_lens.append(mask_dict["sent_len_b"])
                token_mask_lst.append(mask_dict["word_mask_b"])

                input_ids_lst.append(input_ids)
                input_mask_lst.append(input_mask)

            self.input_ids.append(input_ids_lst)
            self.input_mask.append(input_mask_lst)
            self.labels.append(config.label2idx[entry["label"]])
            self.sent_lens.append(sent_lens)
            self.token_mask_lst.append(token_mask_lst)

        print(self.tokenizer.decode(self.input_ids[0][0]))

        save_info(info_path, {"label2idx": config.label2idx,
                              "input_ids": self.input_ids,
                              "input_mask": self.input_mask,
                              "labels": self.labels,
                              "token_mask_lst": self.token_mask_lst,
                              "sent_lens":self.sent_lens,
                              })

    @classmethod
    def collate_fn(cls):
        return collate_fn

    def __len__(self):
        print("use_small_part:", self.args.use_small_part)
        if (self.data_type == "train" or self.data_type == "valid") and self.args.use_small_part:
            return int(len(self.labels) * 0.1)
        return len(self.labels)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.input_ids[idx]).to(self.args.device)
        input_mask = torch.tensor(self.input_mask[idx]).to(self.args.device)
        token_mask = [torch.tensor(tm).to(self.args.device) for tm in self.token_mask_lst[idx]]
        graph = self.graphs[idx]
        label = torch.tensor(self.labels[idx]).to(self.args.device)

        return input_ids, input_mask, token_mask, graph, label

def collate_fn(batch):
    input_ids, input_mask, token_mask, graph, label = map(list, zip(*batch))

    batch_input_ids = torch.stack(input_ids)
    batch_input_mask = torch.stack(input_mask)
    batched_graph = dgl.batch(graph)
    batched_token_mask = token_mask
    batched_label = torch.stack(label)

    return batch_input_ids, batch_input_mask, batched_token_mask, batched_graph, batched_label
