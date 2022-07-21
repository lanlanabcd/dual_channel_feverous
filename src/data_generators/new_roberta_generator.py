# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/6/30 22:33
# Description:

from data_generators import BaseGraphGenerator
import torch
import dgl
import os
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info
import json
import config
from tqdm import tqdm
from transformers import AutoTokenizer
import random
import numpy as np
import pickle as pkl

class NewRobertaGenerator(BaseGraphGenerator):
    def __init__(self, input_path, tokenizer, root_dir, subdir, data_type, args):
        super(NewRobertaGenerator, self).__init__(tokenizer, data_type, args)
        assert 'roberta' in args.bert_name
        self.model_name = str(type(self))

        self.bert_tokenizer = AutoTokenizer.from_pretrained("./bert_weights/bert-base-uncased")

        self.input_ids_lst = []
        self.token_mask_lst = []
        self.labels = []
        self.sent_lens_data = []
        self.seq_lens_data = []

        self.get_flat_tokens(input_path, root_dir, subdir)

        assert len(self.labels) != []
        assert len(self.labels) == len(self.sent_lens_data)
        assert len(self.labels) == len(self.seq_lens_data)
        assert len(self.labels) == len(self.token_mask_lst)
        assert len(self.labels) == len(self.input_ids_lst)

        self.input_ids_lst = np.array(self.input_ids_lst)
        self.seq_lens_data = np.array(self.seq_lens_data, dtype=np.int)
        self.sent_lens_data = np.array(self.sent_lens_data, dtype=np.int)
        #self.token_mask_lst = np.array(self.token_mask_lst)

        self.input_mask = []
        self.input_ids = []
        self.get_seq_inputs_from_inputs_lst(tokenizer, shuffle=False, max_len=self.args.max_seq_len)
        assert len(self.labels) == len(self.sent_lens)
        assert len(self.labels) == len(self.seq_lens)
        assert len(self.labels) == len(self.token_mask)
        assert len(self.labels) == len(self.input_ids)
        assert len(self.labels) == len(self.input_mask)
        self.seq_lens = np.array(self.seq_lens, dtype=np.int)
        self.sent_lens = np.array(self.sent_lens, dtype=np.int)

        print("graph_types:", self.args.graph_types)
        if self.args.graph_types and self.args.graph_types != "\'\'":
            graph_types = self.args.graph_types.split(",")
            self.graphs = self.process_graphs(input_path, root_dir, subdir, graph_types=graph_types)
        else:
            self.graphs = [[dgl.DGLGraph()]]*len(self.labels)

        assert len(self.graphs) == len(self.labels), print(len(self.graphs), len(self.labels))

        if args.model_type in ["FlatClaimAvg"]:
            # 某些模型从bert的embedding中取词，所以需要加上偏置1， 并且长度为序列长度
            #self.claim_mask = [np.eye(seq_lens[0]-2, self.args.max_seq_len, 1) for seq_lens in self.seq_lens]
            self.claim_mask = [(seq_lens[0]-2, self.args.max_seq_len, 1) for seq_lens in self.seq_lens]
        else:
            # 在图里节点的索引，所以不用加偏置1，长度为单词数目
            # self.claim_mask = [np.eye(sent_lens[0], self.args.max_seq_len) for sent_lens in self.sent_lens]
            self.claim_mask = [(sent_lens[0], self.args.max_seq_len, 0) for sent_lens in self.sent_lens]
        #self.print_example()

    def shuffle_evis(self):
        self.input_mask = []
        self.input_ids = []
        self.token_mask = []
        print("shuffle evidences:", self.args.shuffle_evis)
        self.get_seq_inputs_from_inputs_lst(self.tokenizer, shuffle=self.args.shuffle_evis, max_len=self.args.max_seq_len)
        assert len(self.labels) == len(self.token_mask)
        assert len(self.labels) == len(self.input_ids)
        assert len(self.labels) == len(self.input_mask)

    def print_example(self):
        example = self.__getitem__(0)
        for item in example:
            print(item)
            print('\n')

    # 通过claim和evidence生成input_ids和input_mask，evidence可以打乱顺序
    def get_seq_inputs_from_inputs_lst(self, tokenizer, shuffle, max_len):
        if shuffle == True:
            # 不同模型shuffle不同，所以单独写函数调用
            self.get_seq_inputs_from_inputs_lst_shuffle(tokenizer, max_len)
        else:
            self.get_seq_inputs_from_inputs_lst_no_shuffle(tokenizer, max_len)

    def get_seq_inputs_from_inputs_lst_no_shuffle(self, tokenizer, max_len):
        self.input_ids = []
        self.input_mask = []
        self.token_mask = []
        self.sent_lens = []
        self.seq_lens = []
        for tml, idl in zip(self.token_mask_lst, self.input_ids_lst):
            input_ids, input_mask = self.get_plm_inputs_from_input_ids(idl, tokenizer, max_len)
            self.input_ids.append(input_ids)
            self.input_mask.append(input_mask)
            self.token_mask.append(self.token_mask_lst2token_mask(tml, np.arange(len(tml))))
        self.sent_lens = self.sent_lens_data
        self.seq_lens = self.seq_lens_data

    def get_seq_inputs_from_inputs_lst_shuffle(self, tokenizer, max_len):
        self.input_ids = []
        self.input_mask = []
        self.token_mask = []
        self.sent_lens = []
        self.seq_lens = []
        shuffle_idx_lst = []
        for tml, idl, stl, sql in zip(self.token_mask_lst, self.input_ids_lst, self.sent_lens_data, self.seq_lens_data):
            shuffle_idx = np.arange(len(idl))
            random.shuffle(shuffle_idx[1:])
            input_ids, input_mask = self.get_plm_inputs_from_input_ids(idl[shuffle_idx], tokenizer, max_len)
            self.input_ids.append(input_ids)
            self.input_mask.append(input_mask)
            self.token_mask.append(self.token_mask_lst2token_mask(tml, shuffle_idx))
            self.sent_lens.append(stl[shuffle_idx])
            self.seq_lens.append(sql[shuffle_idx])
            shuffle_idx_lst.append(shuffle_idx)
        self.seq_lens = np.array(self.seq_lens, dtype=np.int)
        self.sent_lens = np.array(self.sent_lens, dtype=np.int)
        return shuffle_idx_lst

    def token_mask_lst2token_mask(self, token_mask_lst, shuffle_idx):
        token_masks = self.sparse_json_to_dense(token_mask_lst)
        return self.dense_to_sparse_json([token_masks[idx] for idx in shuffle_idx])

    def get_info_path_prefix(self):
        #prefix = ''
        #if "subevis" in self.args.data_dir:
        #    data_dir = self.args.data_dir.split('/')[-1]
        #    if not data_dir:
        #        data_dir = self.args.data_dir.split('/')[-2]
        #    if data_dir.startswith("small_"):
        #        data_dir = data_dir[6:]
        #    prefix = config.cache_path[data_dir]
        #else:
        #    prefix = config.cache_path.get("mla_data", "new_roberta")

        data_dir = self.args.data_dir.split('/')[-1]
        if not data_dir:
            data_dir = self.args.data_dir.split('/')[-2]
        if data_dir.startswith("small_"):
            data_dir = data_dir[6:]
        prefix = config.cache_path[data_dir]

        return prefix

    def get_info_path(self, input_path, root_dir, subdir):
        data_dir = os.path.join(root_dir, subdir)
        info_path = "info.pkl"

        prefix = self.get_info_path_prefix()
        if prefix:
            info_path = prefix + "_" + info_path

        #if "Gat" in self.args.model_type:
        #    info_path = "gat_" + info_path

        if "small" in self.args.data_dir:
            info_path = "debug_" + info_path

        info_path = os.path.join(data_dir, info_path)

        print("flat_tokens_info_path:", info_path)
        return info_path

    def get_flat_tokens(self, input_path, root_dir, subdir):
        assert not (self.args.single_evidence and self.args.evi_len_file)
        info_path = self.get_info_path(input_path, root_dir, subdir)

        if os.path.exists(info_path) and not self.args.force_generate:
            print("load flat tokens from", info_path)
            info_dict = load_info(info_path)
            self.input_ids_lst = info_dict["input_ids_lst"]
            self.labels = info_dict["labels"]
            self.token_mask_lst = info_dict["token_mask_lst"]
            self.sent_lens_data = info_dict["sent_lens"]
            self.seq_lens_data = info_dict["seq_lens"]
            for input_ids in self.input_ids_lst[0]:
                print(self.tokenizer.decode(input_ids))
            return

        with open(input_path, "r", encoding='utf-8') as f:
            data = f.readlines()
            data = [json.loads(line.strip()) for line in data]

        if self.args.evi_len_file is not None:
            with open(self.args.evi_len_file+"."+self.data_type, "rb") as f:
                evi_len_lst = pkl.load(f)
        else:
            evi_len_lst = [self.args.cla_evi_num-1] * len(data)


        max_seq_len = self.args.max_seq_len
        for evi_len, entry in tqdm(zip(evi_len_lst, data), desc="get_flat_tokens"):
            token_list = []

            claim = ' '.join(self.bert_tokenizer.tokenize(entry["claim"].lower())).replace(' ##', '').replace('[UNK]',
                                                                                                         '<unk>')
            claim_words = self.tokenizer.tokenize(claim)

            token_list.append(claim_words)

            evidences = [evi_t.strip() + " : " + evi for evi, evi_t in
                         zip(entry["evidences"], entry['evidences_title']) if evi_t]

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

            input_ids_lst, mask_dict = self.get_plm_inputs_lst(claim_tokens, evis_token_lst, self.tokenizer
                                                               ,return_mask=["word_mask_a", "word_mask_b"])

            sent_lens = []
            token_mask_lst = []
            seq_lens = []

            sent_lens.append(mask_dict["word_num_a"])
            sent_lens.extend(mask_dict["word_num_b"])

            token_mask_lst.append(mask_dict["word_mask_a"])
            token_mask_lst.extend(mask_dict["word_mask_b"])

            seq_lens.append(mask_dict["seq_len_a"])
            seq_lens.extend(mask_dict["seq_len_b"])

            self.input_ids_lst.append(input_ids_lst)
            self.labels.append(config.label2idx[entry["label"]])
            self.sent_lens_data.append(sent_lens)
            self.seq_lens_data.append(seq_lens)
            self.token_mask_lst.append(self.dense_to_sparse_json(token_mask_lst))

        for input_ids in self.input_ids_lst[0]:
            print(self.tokenizer.decode(input_ids))

        save_info(info_path, {"label2idx": config.label2idx,
                              "input_ids_lst": self.input_ids_lst,
                              "labels": self.labels,
                              "token_mask_lst": self.token_mask_lst,
                              "sent_lens":self.sent_lens_data,
                              "seq_lens": self.seq_lens_data,
                              })

    def __len__(self):
        print("use_small_part:", self.args.use_small_part)
        if (self.data_type == "train" or self.data_type == "valid") and self.args.use_small_part:
            return int(len(self.labels) * 0.1)
        return len(self.labels)

    def __getitem__(self, idx):
        raw_data = self.raw_data[idx]
        input_ids = torch.tensor(self.input_ids[idx]).to(self.args.device)
        input_mask = torch.tensor(self.input_mask[idx]).to(self.args.device)

        token_mask = np.concatenate(self.sparse_json_to_dense(self.token_mask[idx]), axis=1)
        token_mask = np.concatenate([token_mask, np.zeros([token_mask.shape[0],512-token_mask.shape[1]])], axis=1)
        token_mask = torch.tensor(token_mask, dtype = torch.float32).to(self.args.device)

        claim_mask = torch.tensor(np.eye(*(self.claim_mask[idx])), dtype = torch.float32).to(self.args.device)
        graph = self.graphs[idx]
        label = torch.tensor(self.labels[idx]).to(self.args.device)

        return raw_data, input_ids, input_mask, token_mask, claim_mask, graph, label
