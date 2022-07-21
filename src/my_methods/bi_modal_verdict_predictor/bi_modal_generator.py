# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2022/1/3 16:06
# Description:

import os
import re

import pandas as pd
from torch.utils.data import Dataset

from base_templates import BaseGenerator
from tqdm import tqdm
import torch

from utils.annotation_processor import AnnotationProcessor
from utils.prepare_model_input import init_db, prepare_input, remove_bracket_w_nonascii


def collate_fn(batch):
    raw_data, input_ids, attention_mask, token_type_ids, input_ids2, attention_mask2, labels = map(list, zip(*batch))
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    token_type_ids = torch.stack(token_type_ids)
    input_ids2 = torch.stack(input_ids2)
    attention_mask2 = torch.stack(attention_mask2)
    labels = torch.stack(labels)
    return raw_data, input_ids, attention_mask, token_type_ids,  input_ids2, attention_mask2, labels


class BiModalGenerator(BaseGenerator):
    def __init__(self, input_path, tokenizer, cache_dir, data_type, args):
        super(BiModalGenerator, self).__init__(input_path, data_type, args)
        init_db(args.wiki_path)
        self.tokenizer = tokenizer
        self.data_type = data_type
        if data_type in ["train"]:
            claim_evidence_input, evidence_input, annotations = self.get_train_data()
        elif data_type in ["dev_retrieved", "dev", "test"]:
            claim_evidence_input, evidence_input, annotations = self.get_test_data(retrieved = (("retrieved" in data_type) or ("test" in data_type)))
        else:
            assert False, data_type

        self.labels = [self.config.label2idx[anno.get_verdict()] if hasattr(anno, "verdict") else 0 for anno in annotations]
        if self.data_type != "test":
            self.print_label_distribution(self.labels)
        assert self.labels != 0
        self.claims = [anno.claim for anno in annotations]
        self.flatten_tables = process_data(evidence_input)
        self.claim_evidence_input = process_data(claim_evidence_input)

        assert len(self.labels) != 0
        assert len(self.claims) == len(self.labels)
        assert len(self.flatten_tables) == len(self.labels)
        assert len(self.claim_evidence_input) == len(self.labels)
        assert len(self.raw_data) == len(self.labels)

    def print_label_distribution(self, labels_train):
        NEI_num = len([lb for lb in labels_train if lb == 0])
        SUP_num = len([lb for lb in labels_train if lb == 1])
        REF_num = len([lb for lb in labels_train if lb == 2])
        print(f"NEI: {NEI_num * 1.0 / len(labels_train)}, SUPPORTS: {SUP_num * 1.0 / len(labels_train)}, "
              f"REFUTES: {REF_num * 1.0 / len(labels_train)}")

    def get_refine_keys(self):
        keys = ["id", "claim", "evidence", "predicted_evidence", "label"]
        return keys

    def get_train_data(self):
        args = self.args
        all_evidence_input = []
        all_claim_evidence_input = []
        all_annotations_train = []
        self.labels = []
        self.raw_data = []


        args.train_data_path = os.path.join(args.data_dir, 'train.jsonl')
        anno_processor_train = AnnotationProcessor(args.train_data_path, has_content=True)
        annotations_train = [annotation for annotation in anno_processor_train]
        evidence_input = [(prepare_input(anno, 'all2tab', gold=True), anno.get_verdict()) for i, anno in
                                enumerate(tqdm(annotations_train))]
        all_evidence_input.extend(evidence_input)
        claim_evidence_input = [(prepare_input(anno, 'all2text', gold=True), anno.get_verdict()) for i, anno in
                                enumerate(tqdm(annotations_train))]
        all_claim_evidence_input.extend(claim_evidence_input)
        all_annotations_train.extend(annotations_train)
        raw_data = self.preprocess_raw_data(self.get_raw_data(args.train_data_path, keys=self.get_refine_keys()))
        self.raw_data.extend(raw_data)

        args.train_data_path = os.path.join(args.data_dir, 'train.combined.not_precomputed.p5.s5.t3.cells.jsonl')
        anno_processor_train = AnnotationProcessor(args.train_data_path, has_content=True)
        annotations_train = [annotation for annotation in anno_processor_train]
        if args.revise_labels:
            for anno in annotations_train:
                if anno.verdict == "NOT ENOUGH INFO":
                    continue
                predicted_evi = set(anno.predicted_evidence)
                gold_evi_lst = anno.evidence
                sufficient = False
                for gold_evi in gold_evi_lst:
                    if set(gold_evi).issubset(predicted_evi):
                        sufficient = True
                        break
                if not sufficient:
                    # print(anno.verdict, "==> NEI")
                    anno.verdict = "NOT ENOUGH INFO"
        evidence_input = [(prepare_input(anno, 'all2tab', gold=False), anno.get_verdict()) for i, anno in
                                enumerate(tqdm(annotations_train))]
        all_evidence_input.extend(evidence_input)
        claim_evidence_input = [(prepare_input(anno, 'all2text', gold=False), anno.get_verdict()) for i, anno in
                                enumerate(tqdm(annotations_train))]
        all_claim_evidence_input.extend(claim_evidence_input)
        all_annotations_train.extend(annotations_train)
        raw_data = self.preprocess_raw_data(self.get_raw_data(args.train_data_path, keys=self.get_refine_keys()))
        self.raw_data.extend(raw_data)

        return all_claim_evidence_input, all_evidence_input, all_annotations_train

    def get_test_data(self, retrieved = False):
        args = self.args
        if self.data_type == "test":
            file_name = 'test.combined.not_precomputed.p5.s5.t3.cells.jsonl'
        else:
            file_name = "dev.combined.not_precomputed.p5.s5.t3.cells.jsonl" if retrieved else 'dev.jsonl'
        args.dev_data_path = os.path.join(args.data_dir, file_name)
        anno_processor_dev = AnnotationProcessor(args.dev_data_path, has_content=False)
        annotations_dev = [annotation for annotation in anno_processor_dev]
        evidence_input_test = [(prepare_input(anno, 'all2tab', gold=(not retrieved)), anno.get_verdict()) for i, anno in
                                     enumerate(tqdm(annotations_dev))]
        claim_evidence_input_test = [(prepare_input(anno, 'all2text', gold=(not retrieved)), anno.get_verdict()) for
                                     i, anno in
                                     enumerate(tqdm(annotations_dev))]
        return claim_evidence_input_test, evidence_input_test, annotations_dev

    def preprocess_raw_data(self, raw_data):
        if raw_data[0]["claim"]:
            return raw_data
        else:
            return raw_data[1:]

    def __getitem__(self, idx):
        raw_data = self.raw_data[idx]
        encodings = self.tokenizer[0](table=read_text_as_pandas_table(self.flatten_tables[idx]), queries=self.claims[idx]
                                   , padding="max_length", truncation=True)
        input_ids = torch.tensor(encodings["input_ids"]).to(self.args.device)
        attention_mask = torch.tensor(encodings["attention_mask"]).to(self.args.device)
        token_type_ids = torch.tensor(encodings["token_type_ids"]).to(self.args.device)

        encodings = self.tokenizer[1](self.claim_evidence_input[idx], padding="max_length", truncation=True)
        input_ids2 = torch.tensor(encodings["input_ids"]).to(self.args.device)
        attention_mask2 = torch.tensor(encodings["attention_mask"]).to(self.args.device)
        labels = torch.tensor(self.labels[idx]).to(self.args.device)

        return raw_data, input_ids, attention_mask, token_type_ids, input_ids2, attention_mask2, labels

    @classmethod
    def collate_fn(cls):
        return collate_fn

def process_data(claim_verdict_list):
    text = [x[0] for x in claim_verdict_list] # ["I love Pixar.", "I don't care for Pixar."]
    pt = re.compile(r"\[\[.*?\|(.*?)]]")
    text = [re.sub(pt, r"\1", text) for text in text]
    # text_test = [re.sub(pt, r"[ \1 ]", text) for text in text_test]
    text = [remove_bracket_w_nonascii(text) for text in text]

    return text

def read_text_as_pandas_table(table_text: str):
    table = pd.DataFrame([x.split(' | ') for x in table_text.split('\n')][:255], columns=[x for x in table_text.split('\n')[0].split(' | ')]).fillna('')
    table = table.astype(str)
    return table
