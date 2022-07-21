# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2022/1/3 22:18
# Description:
from my_utils import load_jsonl_data, save_jsonl_data
import os
# os.chdir("{dir}/DCUF_code")
input_path = "data/test.jsonl.bk"
output_path = "data/test.jsonl"
data = load_jsonl_data(input_path)
data[0]["id"] = ""
for idx, entry in enumerate(data[1:]):
    entry["id"] = idx
save_jsonl_data(data, output_path)