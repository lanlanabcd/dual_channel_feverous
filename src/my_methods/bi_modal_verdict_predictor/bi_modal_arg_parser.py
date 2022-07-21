# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2022/1/3 17:00
# Description:

import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./data/", type=str,
                        help="The input data dir. ", )
    parser.add_argument("--cache_dir", default="./cached_data/", type=str,
                        help="cached data dir. ", )

    parser.add_argument("--ckpt_root_dir", default="./checkpoints", type=str,
                        help="The checkpoints dir. ", )
    parser.add_argument("--test_ckpt", default=None, type = str,
                        help = "The checkpoint name for test")
    parser.add_argument("--load_model_path", type=str,
                        help="Load model from the path.", )


    parser.add_argument("--bert_name", default="./bert_weights/tapas-large-finetuned-tabfact", type=str,
                        help="name or path of pretrained bert.")
    parser.add_argument("--data_generator", default="BiModal_Generator", type=str,
                        help="Name of preprocessor.", )

    parser.add_argument("--force_generate", action="store_true",
                        help= "if set, generate data regardless of whether there is a cache")
    parser.add_argument("--save_all_ckpt", action="store_true")
    parser.add_argument("--test", action="store_true"
                        , help="whether to generate test results")

    parser.add_argument("--fix_bert", action="store_true",
                        help="whether to train bert parameters")

    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int)

    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--warm_rate", default=0.2, type=float)

    parser.add_argument("--seed", default=1234, type=int,
                        help="random seed")
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float,
                        help='Weight decay (L2 loss on parameters).')

    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--max_epoch", default=3, type=int)
    parser.add_argument("--print_freq", default=100, type=int)
    parser.add_argument("--save_freq", default=1, type=int)

    parser.add_argument('--wiki_path', type=str, default="data/feverous_wikiv1.db", help='/path/to/data')
    parser.add_argument('--revise_labels', action="store_true"
                        , help="set the label of instances without sufficient retrieved evidence to NEI.")

    return parser
