# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/12/23 15:24
# Description:

from my_utils import load_jsonl_data
from my_utils import save_jsonl_data
import os
import random
import argparse

os.chdir("../../")

def main(args):
    split_num = args.split_num
    segs_dir = "./data/segs_tab"

    if args.merge:
        odata = []
        for i in range(split_num):
            split_odata = load_jsonl_data(f"{segs_dir}/{i}_train.tables.not_precomputed.p5.t3.jsonl")
            odata.extend(split_odata)
        save_jsonl_data(odata, "./data/train.tables.not_precomputed.p5.t3.jsonl")
        return

    input_path = "data/{}.pages.p{}.jsonl".format(args.split, args.count)
    # input_path = "./data/train.pages.p150.jsonl"
    ori_data = load_jsonl_data(input_path)
    span = (len(ori_data)//split_num) + 1

    if args.part:
        split_idxs = [int(idx) for idx in args.part.split(",")]
    else:
        split_idxs = list(range(split_num))

    # PYTHONPATH=src python src/baseline/retriever/table_tfidf_drqa.py --db data/feverous_wikiv1.db --max_page 5 --max_tabs 3 --use_precomputed false --data_path data/ --split dev

    # 0,1,2,3 104
    # 4,5,6,7 102

    for i in split_idxs:
        save_jsonl_data(ori_data[i*span:(i+1)*span], f"{segs_dir}/{i}_{args.split}.pages.p{args.count}.jsonl")
        device = random.randint(0,args.gpu_number-1)
        os.system(f"CUDA_VISIBLE_DEVICES={device} "
                  f"PYTHONPATH=src nohup python -u src/baseline/retriever/table_tfidf_drqa.py --split {i}_{args.split} "
                  f"--db data/feverous_wikiv1.db --max_page 5 --max_tabs 3 --use_precomputed false --data_path {segs_dir} "
                  f" > logs/log_tab_{args.split}_{i}.txt &")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--split_num", type=int, default=10)
    parser.add_argument("--gpu_number", type=int, default=4)
    parser.add_argument("--part", type=str, default='')
    args = parser.parse_args()
    main(args)
