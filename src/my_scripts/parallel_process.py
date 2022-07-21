# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/12/17 19:12
# Description:

# PYTHONPATH=../ python parallel_process.py

from my_utils import load_jsonl_data
from my_utils import save_jsonl_data
import os
import random
import argparse

os.chdir("../../")

def main(args):
    split_num = args.split_num

    if args.merge:
        odata = []
        for i in range(split_num):
            split_odata = load_jsonl_data(f"./data/segs/{i}_train.pages.p150.r5.jsonl")
            odata.extend(split_odata)
        save_jsonl_data(odata, "./data/train.pages.p150.r5.jsonl")
        return

    input_path = "./data/train.pages.p150.jsonl"
    ori_data = load_jsonl_data(input_path)
    span = (len(ori_data)//split_num) + 1

    if args.part:
        split_idxs = [int(idx) for idx in args.part.split(",")]
    else:
        split_idxs = list(range(split_num))

    for i in split_idxs:
        save_jsonl_data(ori_data[i*span:(i+1)*span], f"./data/segs/{i}_train.pages.p150.jsonl")
        device = random.randint(0,args.gpu_number-1)
        os.system(f"CUDA_VISIBLE_DEVICES={device} PYTHONPATH=src nohup python -u src/re-ranker/page_reranker.py  "
                  f"--db ./data/feverous_wikiv1.db --device cuda:0 --split {i}_train --max_page 150 --max_rerank 5 "
                  f"--use_precomputed false --data_path data/segs/ "
                  f"--batch_size 4 > logs/log_rerank_train_{i}.txt &")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--split_num", type=int, default=10)
    parser.add_argument("--gpu_number", type=int, default=4)
    parser.add_argument("--part", type=str, default='')
    args = parser.parse_args()
    main(args)
