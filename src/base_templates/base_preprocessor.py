# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/12/13 20:17
# Description:

import os

class BasePreprocessor(object):
    def __init__(self, args):
        self.args = args
        self.train_data = []
        self.valid_data = []
        self.test_data = []

        self.data_generator = None

    def process(self, input_dir, output_dir, data_generator, tokenizer, dataset = ["train", "dev", "test"]):
        args = self.args
        self.data_generator = data_generator

        if "dev_retrieved" in dataset:
            self.valid_data = data_generator(
                os.path.join(input_dir, "dev.combined.not_precomputed.p5.s5.t3.cells.jsonl")
                , tokenizer, output_dir, "dev_retrieved", args)
        elif "dev" in dataset:
            self.valid_data = data_generator(
                os.path.join(input_dir, "dev.jsonl"), tokenizer, output_dir, "dev", args)
        elif "debug_retrieved" in dataset:
            self.valid_data = data_generator(
                os.path.join(input_dir, "debug.jsonl"), tokenizer, output_dir, "debug_retrieved", args)

        if "train_retrieved" in dataset:
            self.valid_data = data_generator(
                os.path.join(input_dir, "train.combined.not_precomputed.p5.s5.t3.cells.jsonl")
                , tokenizer, output_dir, "train_retrieved", args)
        elif "train" in dataset:
            self.train_data = data_generator(
                os.path.join(input_dir, "train.jsonl"), tokenizer, output_dir, "train", args)

        if "test" in dataset:
            self.test_data = data_generator(
                os.path.join(input_dir, "test.jsonl"), tokenizer, output_dir, "test", args)

        print("train data length:", len(self.train_data))
        print("dev data length:", len(self.valid_data))
        print("test data length:", len(self.test_data))

        return self.train_data, self.valid_data, self.test_data



