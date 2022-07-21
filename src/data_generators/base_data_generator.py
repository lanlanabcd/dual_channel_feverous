# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/10/28 16:30
# Description:

from torch.utils.data.dataset import Dataset
import torch
import os
from scipy.sparse import coo_matrix
import numpy as np
from utils import load_jsonl_data, refine_obj_data
import config
from utils import clean_wiki_str, remove_bracket_w_nonascii, clean_wiki_title
from tqdm import tqdm
import random


def collate_fn(batch):
    raw_data, input_ids, input_mask, claim_mask, label = map(list, zip(*batch))
    batched_raw_data = raw_data
    batch_input_ids = torch.stack(input_ids)
    batch_input_mask = torch.stack(input_mask)

    batched_claim_mask = claim_mask
    batched_label = torch.stack(label)

    return batched_raw_data, batch_input_ids, batch_input_mask, batched_claim_mask, batched_label


class BaseGenerator(Dataset):
    def __init__(self, input_path, tokenizer, root_dir, subdir, data_type, args):
        super(BaseGenerator, self).__init__()
        self.model_name = str(type(self))
        self.args = args
        self.data_type = data_type
        self.tokenizer = tokenizer

        self.raw_data = self.preprocess_raw_data(self.get_raw_data(input_path, keys=self.get_refine_keys()))
        self.labels = [config.label2idx.get(entry["label"], 0) for entry in self.raw_data]
        assert len(self.labels) != 0

    def print_example(self):
        instance = self.raw_data[0]
        for k, v in instance.items():
            print(k, " : ", v)

    def preprocess_raw_data(self, raw_data, max_evi_len=128):
        for entry in tqdm(raw_data, desc="preprocess_raw_data"):
            entry["claim"] = clean_wiki_str(entry["claim"])
            new_evidences = []
            new_evidences_title = []
            new_evidences_meta = []
            new_evidences_label = []

            '''
            if entry["full_evidences_label"] is None:
                assert entry["evidences_label"] is not None and entry["evidences_mask"] is not None
                entry["full_evidences_label"] = entry["evidences_label"]
                entry["evidences_label"] = None
            '''

            if entry["evidences_label"] is None and entry["evidences_mask"] is not None:
                entry["evidences_label"] = entry["evidences_mask"]
            assert entry["evidences_label"] is not None

            for e, e_t, e_m, e_l in zip(entry["evidences"], entry["evidences_title"], entry["evidences_meta"],
                                        entry["evidences_label"]):
                new_e = remove_bracket_w_nonascii(clean_wiki_str(e))
                tokens = self.tokenizer.tokenize(new_e)[:max_evi_len - 2]
                nel = len(tokens)
                if nel > 3:
                    # 需要再筛查，从训练集和测试集角度
                    new_evidences.append(self.tokenizer.convert_tokens_to_string(tokens))
                    new_evidences_title.append(clean_wiki_title(e_t))
                    new_evidences_meta.append(e_m)
                    new_evidences_label.append(e_l)

            assert len(new_evidences) == len(new_evidences_label)
            entry["evidences"] = new_evidences
            entry["evidences_title"] = new_evidences_title
            entry["evidences_meta"] = new_evidences_meta
            entry["evidences_label"] = new_evidences_label

            # 没有取回证据的情况
            if not entry["evidences"]:
                entry["evidences"] = ['']
                entry["evidences_title"] = ['']
                entry["evidences_meta"] = [[None, None]]
                entry["evidences_label"] = [1]
                entry["full_evidences"] = ['']
                entry["full_evidences_label"] = [1]

            grouped_subevis = []
            grouped_subevis_label = []
            lem = None
            for e, e_t, e_m, e_l in zip(entry["evidences"], entry["evidences_title"], entry["evidences_meta"],
                                        entry["evidences_label"]):
                if e_m == lem:
                    ne = e_t + " : " + e
                    # ne = e
                    grouped_subevis[-1].append(ne)
                    grouped_subevis_label[-1].append(e_l)
                else:
                    ne = e_t + " : " + e
                    grouped_subevis.append([ne])
                    grouped_subevis_label.append([e_l])
                    lem = e_m
            entry["grouped_subevis"] = grouped_subevis
            entry["grouped_subevis_label"] = grouped_subevis_label

            if len(entry["grouped_subevis"]) != len(entry["full_evidences"]):
                subevi_set = set()
                for sem in entry["evidences_meta"]:
                    subevi_set.add(sem[0] + str(sem[1]))
                full_evi_set = set()
                new_full_evidences = []
                new_full_evidences_label = []
                new_full_evidences_meta = []
                for fe, fel, fem in zip(entry["full_evidences"], entry["full_evidences_label"], entry["full_evidences_meta"]):
                    fem_str = fem[0] + str(fem[1])
                    if (fem_str not in full_evi_set) and (fem_str in subevi_set):
                        full_evi_set.add(fem_str)
                        new_full_evidences.append(fe)
                        new_full_evidences_label.append(fel)
                        new_full_evidences_meta.append(fem)
                entry["full_evidences"] = new_full_evidences
                entry["full_evidences_label"] = new_full_evidences_label
                entry["full_evidences_meta"] = new_full_evidences_meta
            assert len(entry["grouped_subevis"]) == len(entry["full_evidences"])

        return raw_data

    def get_refine_keys(self):
        keys = ["id", "claim", "label", "evidences", "evidences_title"
            , "evidences_meta", "evidences_label", "evidences_mask", "full_evidences", "full_evidences_label",
                "full_evidences_meta"]
        return keys

    def get_raw_data(self, input_path, keys=None):
        raw_data = load_jsonl_data(input_path)
        if keys is not None:
            raw_data = refine_obj_data(raw_data, keys)
        return raw_data

    def shuffle_evis(self):
        for entry in tqdm(self.raw_data, desc="shuffle evidences"):
            random.shuffle(entry["grouped_subevis"])

    def get_plm_inputs_from_input_ids(self, input_ids, pad_idx, max_len=512, end_idx=2):
        if isinstance(input_ids[0], list):
            flat_input_ids = []
            for ids in input_ids:
                flat_input_ids.extend(ids)
            input_ids = flat_input_ids

        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len - 1]
            input_ids.append(end_idx)
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

        allowed_return_mask = ["mask_a", "fl_cls_mask_b", "word_mask_a", "word_mask_b"]
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
                cls_idx = len_a
                for lb in len_b_lst:
                    mask_cls.append(cls_idx)
                    cls_idx += lb
                mask_dict["evi_mask_cls"] = mask_cls

            if "fl_cls_mask_b" in return_mask:
                mask_mat = np.zeros([len(len_b_lst), max_len], dtype=np.float32)
                cls_idx = len_a
                for i, lb in enumerate(len_b_lst):
                    if cls_idx >= max_len:
                        mask_mat = mask_mat[:i]
                        break
                    mask_mat[i][cls_idx] = 0.5
                    if cls_idx + lb - 1 >= max_len:
                        mask_mat[i][max_len - 1] = 0.5
                        mask_mat = mask_mat[:i + 1]
                        break
                    mask_mat[i][cls_idx + lb - 1] = 0.5
                    cls_idx += lb
                mask_dict["fl_cls_mask_b"] = mask_mat

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

        mask_mat = np.zeros([total_word_num, len(tokens) + 2], dtype=np.float32)
        ptr = 1
        for idx, word_len in enumerate(word_len_lst):
            for _ in range(word_len):
                mask_mat[stat + idx][ptr] = 1.0 / word_len
                ptr += 1
        assert ptr == len(tokens) + 1

        return mask_mat

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
        return len(self.labels)

    def __getitem__(self, idx):
        raw_data = self.raw_data[idx]
        grouped_subevis = raw_data["grouped_subevis"]
        flatened_subevis = [e for s_g in grouped_subevis for e in s_g]
        input_ids_lst, mask_dict = self.get_plm_inputs_lst(raw_data["claim"], flatened_subevis, self.tokenizer,
                                                           return_mask=["mask_a"])
        input_ids, input_mask = self.get_plm_inputs_from_input_ids(
            input_ids_lst, pad_idx=self.tokenizer.pad_token_id, end_idx=self.tokenizer.eos_token_id)
        input_ids = torch.tensor(input_ids).to(self.args.device)
        input_mask = torch.tensor(input_mask).to(self.args.device)
        claim_mask = torch.tensor(mask_dict["mask_a"], dtype=torch.float32).to(self.args.device)
        label = torch.tensor(self.labels[idx]).to(self.args.device)
        return raw_data, input_ids, input_mask, claim_mask, label


class MyArgs():
    def __init__(self):
        self.device = "cpu"


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("../bert_weights/roberta-large")
    input_path = "../data/mla_subevis_data/train.jsonl"

    args = MyArgs()
    generator = BaseGenerator(input_path, tokenizer, None, None, "train", args)

    data_len = generator.__len__()
    print(generator.__getitem__(0))
    print(data_len)
    for i in tqdm(range(data_len)):
        generator.__getitem__(i)
