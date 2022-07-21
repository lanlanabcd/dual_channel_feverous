# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/12/23 15:06
# Description:


from my_utils import load_jsonl_data
from my_utils import save_jsonl_data
import os
import random
import argparse

os.chdir("../../")

def main(args):
    split_num = args.split_num
    segs_dir = "./data/segs_sent"

    if args.merge:
        odata = []
        for i in range(split_num):
            split_odata = load_jsonl_data(f"{segs_dir}/{i}_train.sentences.not_precomputed.p5.s10.jsonl")
            odata.extend(split_odata)
        save_jsonl_data(odata, "./data/train.sentences.not_precomputed.p5.s10.jsonl")
        return

    input_path = "data/{}.pages.p{}.jsonl".format(args.split, args.count)
    # input_path = "./data/train.pages.p150.jsonl"
    ori_data = load_jsonl_data(input_path)
    span = (len(ori_data)//split_num) + 1

    if args.part:
        split_idxs = [int(idx) for idx in args.part.split(",")]
    else:
        split_idxs = list(range(split_num))

    # PYTHONPATH=src python src/baseline/retriever/sentence_tfidf_drqa.py --db data/feverous_wikiv1.db --max_page 5 --max_sent 10 --use_precomputed false --data_path data/ --split train

    for i in split_idxs:
        save_jsonl_data(ori_data[i*span:(i+1)*span], f"{segs_dir}/{i}_{args.split}.pages.p{args.count}.jsonl")
        device = random.randint(0,args.gpu_number-1)
        os.system(f"CUDA_VISIBLE_DEVICES={device} "
                  f"PYTHONPATH=src nohup python -u src/baseline/retriever/sentence_tfidf_drqa.py  "
                  f"--db data/feverous_wikiv1.db --max_page 5 --max_sent 10 --use_precomputed false --data_path {segs_dir} "
                  f"--split {i}_{args.split} > logs/log_sent_{args.split}_{i}.txt &")

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
