# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2020/11/30 13:13
# Description:
import random

from torch.utils.data.dataset import Dataset
import torch
import dgl
import os
from dgl import save_graphs, load_graphs
from scipy.sparse import coo_matrix
import json
import config
from tqdm import tqdm
from utils.get_dgl_graph import get_simple_srl_graph, get_srl_graph, get_dep_graph \
    , get_fully_claim_graph, get_fc_graph, get_srl_evi_graph, get_gmn_graph
import numpy as np
from utils import load_jsonl_data, refine_obj_data


def collate_fn(batch):
    raw_data, input_ids, input_mask, token_mask, claim_mask, graph, label = map(list, zip(*batch))

    batched_raw_data = raw_data
    batch_input_ids = torch.stack(input_ids)
    batch_input_mask = torch.stack(input_mask)
    batched_graph = [dgl.batch(list(g)) for g in zip(*graph)]
    if len(batched_graph) == 1:
        batched_graph = batched_graph[0]
    batched_token_mask = token_mask
    batched_claim_mask = claim_mask
    batched_label = torch.stack(label)

    return batched_raw_data, batch_input_ids, batch_input_mask\
        , batched_token_mask, batched_claim_mask, batched_graph, batched_label


class BaseGraphGenerator(Dataset):
    def __init__(self, tokenizer, data_type, args):
        super(BaseGraphGenerator, self).__init__()
        self.model_name = str(type(self))
        self.args = args
        self.data_type = data_type
        self.tokenizer = tokenizer
        self.graph_type2func = {
            # 基本图， claim全连接， evidence srl结构，之间连接方式通过args.use_gmn决定
            "srl": self.process_srl_graphs,
            "dep": self.process_dep_graphs,
            # 只对claim进行全连接
            "fully_claim": self.process_fully_claim_graphs,
            # 只有evidence到claim的连边
            "gmn": self.process_gmn_graphs,
            # claim nodes 之间没有连接，evidence 根据srl结构连接
            "srl_evis": self.process_srl_evi_graphs,
            "fully_connected": self.process_fully_connect_graphs,
        }
        self.register_graph_func()
        self.raw_data = self.get_raw_data(args.data_dir, data_type)

    def get_raw_data(self, data_dir, data_type, keys = None):
        raw_data = load_jsonl_data(os.path.join(data_dir, data_type)+".jsonl")
        if keys is not None:
            raw_data = refine_obj_data(raw_data, keys)
        return raw_data

    def register_graph_func(self):
        pass

    def get_plm_inputs_from_input_ids(self, input_ids, tokenizer, max_len=512):
        if isinstance(input_ids[0], list):
            flat_input_ids = []
            for ids in input_ids:
                flat_input_ids.extend(ids)
            input_ids = flat_input_ids

        pad_idx = tokenizer.pad_token_id
        input_mask = [1] * len(input_ids)
        input_ids += [pad_idx] * (max_len - len(input_ids))
        input_mask += [0] * (max_len - len(input_mask))
        return input_ids, input_mask

    def encode_one_sent(self, sent, tokenizer, is_first_sent=False):
        if isinstance(sent, str):
            tokens = tokenizer.tokenize(sent)
        else:
            tokens = sent

        if is_first_sent:
            tokens_encode = ["<s>"]
        else:
            tokens_encode = ["</s>"]

        tokens_encode.extend(tokens)
        tokens_encode.append("</s>")
        input_ids = tokenizer.convert_tokens_to_ids(tokens_encode)
        return input_ids, tokens

    def get_plm_inputs_lst(self, sent_a, sent_b_lst, tokenizer, max_len=512, return_mask=None):
        """
        :param sent_b_lst:
        :param sent_a:
        :param tokenizer:
        :param max_len:
        :param return_mask: ["evi_mask_cls", "mask_a", "mask_b", "word_mask_a", "word_mask_b"]
        :return:
        """

        allowed_return_mask = ["word_mask_a", "word_mask_b"]

        input_ids_lst = []
        input_ids, tokens_a = self.encode_one_sent(sent_a, tokenizer, is_first_sent=True)

        input_ids_lst.append(input_ids)
        len_a = len(input_ids)
        len_b_lst = []
        tokens_b_lst = []

        if sent_b_lst is not None:
            for sent_b in sent_b_lst:
                input_ids, tokens_b = self.encode_one_sent(sent_b, tokenizer, is_first_sent=False)
                len_b = len(input_ids)
                len_b_lst.append(len_b)
                input_ids_lst.append(input_ids)
                tokens_b_lst.append(tokens_b)

        assert len(input_ids) <= max_len
        assert sum(len_b_lst) + len_a == sum([len(ids) for ids in input_ids_lst])

        mask_dict = {"seq_len_a": len_a, "seq_len_b": len_b_lst}

        if return_mask is not None:
            for rm in return_mask:
                if rm not in allowed_return_mask:
                    assert False, print(rm)

            if "mask_a" in return_mask:
                mask_dict["mask_a"] = self.get_bpe_mask_roberta(0, len_a, max_len)
            if "mask_b" in return_mask:
                mask_dict["mask_b"] = []
                stat = len_a
                for len_b in len_b_lst:
                    mask_dict["mask_b"].append(self.get_bpe_mask_roberta(stat, len_b, max_len))
                    stat += len_b
            if "evi_mask_cls" in return_mask:
                mask_cls = []
                cls_idx = len_a + 1
                for lb in len_b_lst:
                    mask_cls.append(cls_idx)
                    cls_idx += lb
                mask_dict["evi_mask_cls"] = mask_cls

            word_num_a = self.get_word_num_from_roberta_tokens(tokens_a)
            word_num_b_lst = [self.get_word_num_from_roberta_tokens(tokens_b) for tokens_b in tokens_b_lst]

            if "word_mask_a" in return_mask:
                total_word_num = word_num_a
                if "word_mask_b" in return_mask:
                    total_word_num += sum(word_num_b_lst)
                mask_mat = self.get_word_mask_roberta(tokens_a, 0, word_num_a, len_a, total_word_num)
                mask_dict["word_mask_a"] = mask_mat
                mask_dict["word_num_a"] = word_num_a

            if "word_mask_b" in return_mask:
                total_word_num = word_num_a + sum(word_num_b_lst)
                mask_dict["word_mask_b"] = []
                mask_dict["word_num_b"] = []
                stat = word_num_a
                for tokens_b, len_b, word_num_b in zip(tokens_b_lst, len_b_lst, word_num_b_lst):
                    mask_mat = self.get_word_mask_roberta(tokens_b, stat, word_num_b, len_b, total_word_num)
                    mask_dict["word_mask_b"].append(mask_mat)
                    mask_dict["word_num_b"].append(word_num_b)
                    stat += word_num_b
        return input_ids_lst, mask_dict

    def get_bpe_mask_roberta(self, stat, seq_len, max_len):
        mask_mat = np.zeros([seq_len - 2, max_len], dtype=np.float32)
        mask_mat[:, stat + 1:stat + seq_len - 1] = np.eye(seq_len - 2, dtype=np.float32)
        return mask_mat

    def get_cls_mask_roberta(self, seq_len_lst, max_len):
        sent_num = len(seq_len_lst)
        assert sent_num >= 1
        mask_mat = np.zeros([sent_num, max_len], dtype=np.float32)
        stat = 0
        for idx, sl in enumerate(seq_len_lst):
            if sl is None:
                break
            mask_mat[idx][stat + 1] = 1
            stat += sl
        return mask_mat

    def get_word_num_from_roberta_tokens(self, tokens):
        if tokens:
            word_num = len([token for token in tokens if token.startswith("Ġ")]) + 1
        else:
            word_num = 0
        return word_num

    def get_word_mask_roberta(self, tokens, stat, word_num, seq_len, total_word_num):
        word_len_lst = []
        word_len = 0
        assert seq_len == len(tokens) + 2
        for token in tokens:
            if token.startswith("Ġ"):
                word_len_lst.append(word_len)
                word_len = 1
            else:
                word_len += 1
        if word_len != 0:
            word_len_lst.append(word_len)
        assert len(word_len_lst) == word_num

        #mask_mat = np.zeros([word_num, max_len], dtype=np.float32)
        #ptr = stat + 1
        #for idx, word_len in enumerate(word_len_lst):
        #    for _ in range(word_len):
        #        mask_mat[idx][ptr] = 1.0 / word_len
        #        ptr += 1
        #assert ptr + 1 == stat + seq_len

        mask_mat = np.zeros([total_word_num, len(tokens)+2], dtype=np.float32)
        ptr = 1
        for idx, word_len in enumerate(word_len_lst):
            for _ in range(word_len):
                mask_mat[stat + idx][ptr] = 1.0 / word_len
                ptr += 1
        assert ptr == len(tokens)+1

        return mask_mat

    def process_graphs(self, input_path, root_dir, subdir, graph_types):
        if self.args.main_file == "arg_masker":
            return [dgl.DGLGraph()] * len(self.labels), None

        all_graphs = []
        for graph_type in graph_types:
            graph_path = self.get_graph_path(root_dir, subdir, graph_type)

            graphs = self.load_from_graph_cache(graph_path)
            if graphs is None:
                data = load_jsonl_data(input_path)
                assert len(data) == len(self.sent_lens)
                graphs, labels = self.graph_type2func[graph_type](data, graph_type)
                self.save_graphs(graph_path, graphs, labels)
            all_graphs.append(graphs)

        all_graphs = list(zip(*all_graphs))
        return all_graphs

    def process_srl_graphs(self, data, graph_type, graph_func=get_srl_graph):
        graphs = []
        labels = []

        for entry, sent_lens in tqdm(zip(data, self.sent_lens), desc="process {} graphs".format(graph_type)):
            words = [entry["claim_words"]]
            words += entry["evidences_words"]
            srl_dicts = [entry["claim_srl"]]
            srl_dicts += entry["evidences_srl"]

            graph, graph_words = graph_func(srl_dicts, words, sent_lens, entry, self.args)

            graphs.append(graph)
            labels.append(config.label2idx[entry['label']])

        return graphs, labels

    def process_dep_graphs(self, data, graph_type, graph_func=get_dep_graph):
        graphs = []
        labels = []

        for entry, sent_lens in tqdm(zip(data, self.sent_lens), desc="process {} graphs".format(graph_type)):
            words = [entry["claim_words"]]
            words += entry["evidences_words"]

            dep_dicts = [entry["claim_dep"]]
            dep_dicts += entry["evidences_dep"]

            if len(dep_dicts) > self.args.cla_evi_num:
                dep_dicts = dep_dicts[:self.args.cla_evi_num]
            graph, _ = graph_func(dep_dicts, words, sent_lens, entry, self.args)
            graphs.append(graph)
            labels.append(config.label2idx[entry['label']])
        return graphs, labels

    def process_fully_claim_graphs(self, data, graph_type):
        return self.process_srl_graphs(data, graph_type, graph_func=get_fully_claim_graph)

    def process_fully_connect_graphs(self, data, graph_type):
        return self.process_srl_graphs(data, graph_type, graph_func=get_fc_graph)

    def process_srl_evi_graphs(self, data, graph_type):
        return self.process_srl_graphs(data, graph_type, graph_func=get_srl_evi_graph)

    def process_gmn_graphs(self, data, graph_type):
        return self.process_srl_graphs(data, graph_type, graph_func=get_gmn_graph)

    def get_graph_path_prefix(self, graph_type):
        prefix_lst = []
        if graph_type in ["srl", "dep"] and self.args.use_gmn:
            prefix_lst.append("gmn")

        if graph_type == "subevis":
            data_dir = self.args.data_dir.split('/')[-1]
            if not data_dir:
                data_dir = self.args.data_dir.split('/')[-2]
            if data_dir.startswith("small_"):
                data_dir = data_dir[6:]
            prefix_lst.append(config.graph_path[data_dir])
        return '_'.join(prefix_lst)

    def get_graph_path(self, root_dir, subdir, graph_type):
        data_dir = os.path.join(root_dir, subdir)
        graph_file = f"{graph_type}_graphs.bin"

        prefix = self.get_graph_path_prefix(graph_type)
        if prefix:
            graph_file = prefix + "_" + graph_file

        if "small" in self.args.data_dir:
            graph_file = "debug_" + graph_file

        graph_path = os.path.join(data_dir, graph_file)

        print("graph_path:", graph_path)
        return graph_path

    def load_from_graph_cache(self, graph_path):
        if not self.args.force_generate and os.path.exists(graph_path):
            graphs, label_dict = load_graphs(graph_path)
            labels = label_dict["labels"]
            print("load from graph cache {}".format(graph_path))
            print(graphs[0])
            return graphs
        else:
            return None

    def save_graphs(self, graph_path, graphs, labels):
        labels = torch.tensor(labels)
        save_graphs(graph_path, graphs, {'labels': labels})
        print("save to graph cache {}".format(graph_path))
        print(graphs[0])

    def dense_to_sparse_json(self, dense_mat):
        if isinstance(dense_mat, list):
            sparse_lst = [self.dense_to_sparse_json(dm) for dm in dense_mat]
            return sparse_lst

        sparse_mat = coo_matrix(dense_mat)
        row = sparse_mat.row
        col = sparse_mat.col
        data = sparse_mat.data
        shape = sparse_mat.shape
        sparse_json = ((data, (row, col)), shape)
        return sparse_json

    def sparse_json_to_dense(self, sparse_json):
        if isinstance(sparse_json, list):
            dense_mat_lst = [self.sparse_json_to_dense(sj) for sj in sparse_json]
            return dense_mat_lst
        dense_mat = np.array(coo_matrix(sparse_json[0], shape=sparse_json[1]).todense(), dtype=np.float32)
        return dense_mat

    @classmethod
    def collate_fn(cls):
        return collate_fn

    def __len__(self):
        print("use_small_part:", self.args.use_small_part)
        if (self.data_type == "train" or self.data_type == "valid") and self.args.use_small_part:
            return int(len(self.labels) * 0.1)
        return len(self.labels)

    def __getitem__(self, idx):
        raw_data = self.raw_data[idx]
        input_ids = torch.tensor(self.input_ids[idx]).to(self.args.device)
        input_mask = torch.tensor(self.input_mask[idx]).to(self.args.device)
        token_mask = torch.tensor(np.concatenate(self.sparse_json_to_dense(self.token_mask[idx]), axis=1),
                                  dtype=torch.float32).to(self.args.device)
        claim_mask = torch.tensor(np.eye(*(self.claim_mask[idx])), dtype=torch.float32).to(self.args.device)
        graph = self.graphs[idx]
        label = torch.tensor(self.labels[idx]).to(self.args.device)

        return raw_data, input_ids, input_mask, token_mask, claim_mask, graph, label
