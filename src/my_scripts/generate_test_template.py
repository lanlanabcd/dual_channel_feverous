# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2022/1/4 19:46
# Description:
import os
from tqdm import tqdm
# os.chdir("{dir}/DCUF_code")

from my_utils import load_jsonl_data, save_jsonl_data
input_path = "data/test.jsonl"
output_path = "data/test_temp.jsonl"

data = load_jsonl_data(input_path)[1:]
odata = []
for entry in data:
    oentry = {}
    oentry["predicted_label"] = entry.get("predicted_label", "SUPPORTS")
    predicted_evidence = list(set(entry["predicted_evidence"]))
    new_predicted_evidence = [[el.split('_')[0], el.split('_')[1] if 'table_caption' not in el and 'header_cell' not in el else '_'.join(el.split('_')[1:3]), '_'.join(el.split('_')[2:]) if 'table_caption' not in el and 'header_cell' not in el else '_'.join(el.split('_')[3:])] for el in predicted_evidence]
    oentry["predicted_evidence"] = new_predicted_evidence
    odata.append(oentry)
save_jsonl_data(odata, output_path)
