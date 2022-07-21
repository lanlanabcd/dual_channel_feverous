# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/7/1 21:03
# Description:

import json
import pickle as pkl
import os

def average(list):
    return float(sum(list) / len(list))

def generate_data4debug(data_dir, sample_num = 5):
    if ("small_" in data_dir) and os.path.exists(data_dir.replace("small_", '')):
        if os.path.exists(data_dir):
            os.system(f"rm -r {data_dir}")
        print(f"generate debug data {data_dir}")
        os.mkdir(data_dir)
        ori_data_dir = data_dir.replace("small_", '')
        for s in ["dev", "train", "test"]:
            os.system(f"head -n {sample_num} {ori_data_dir}/{s}.jsonl > {data_dir}/{s}.jsonl")
            os.system(f"head -n {sample_num} {ori_data_dir}/{s}.combined.not_precomputed.p5.s5.t3.cells.jsonl > {data_dir}/{s}.combined.not_precomputed.p5.s5.t3.cells.jsonl")
            os.system(
                f"head -n {sample_num} {ori_data_dir}/{s}.combined.not_precomputed.p5.s5.t3.jsonl > {data_dir}/{s}.combined.not_precomputed.p5.s5.t3.jsonl")
            os.system(f"head -n {sample_num-1} {ori_data_dir}/{s}.evi_type.jsonl > {data_dir}/{s}.evi_type.jsonl")
            os.system(f"head -n {sample_num - 1} {ori_data_dir}/{s}.pos_neg.sentences.jsonl > {data_dir}/{s}.pos_neg.sentences.jsonl")


def print_json_format_obj(obj):
    print(json.dumps(obj, sort_keys=True, indent=4, separators=(',', ':')))

def load_jsonl_data(input_path):
    with open(input_path, "r", encoding='utf-8') as f:
        data = f.readlines()
        data = [json.loads(line.strip()) for line in data]
    return data

def save_lines(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for line in data:
            f.write(line.strip()+"\n")

def load_json_data(input_path):
    return json.load(open(input_path, "r", encoding="utf-8"))

def save_json_data(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def refine_jsonl_data(input_path, keys, output_path = None,  max_length = -1, to_json = False):
    data = load_jsonl_data(input_path)
    return refine_obj_data(data, keys, output_path, max_length, to_json)

def load_pkl_data(input_path):
    with open(input_path, "rb") as f:
        data = pkl.load(f)
    return data

def save_pkl_data(data, output_path):
    with open(output_path, "wb") as f:
        pkl.dump(data, f)

def load_jsonl_one_line(input_path):
    return json.loads(open(input_path, "r", encoding='utf-8').readline())

def save_jsonl_data(data, output_path):
    with open(output_path, "w", encoding='utf-8') as fout:
        for entry in data:
            fout.write(json.dumps(entry) + '\n')

def refine_obj_data(data, keys, output_path = None,  max_length = None, to_json = False):
    if isinstance(keys, str):
        keys = [keys]
    if max_length:
        data = data[:max_length]
    odata = []
    for entry in data:
        oentry = {}
        for key in keys:
            oentry[key] = entry.get(key, None)
        odata.append(oentry)
    if output_path is None:
        return odata
    if to_json:
        save_json_data(odata, output_path)
        return
    else:
        save_jsonl_data(odata, output_path)
        return

def merge_obj_data(data1, data2, keys = None):
    if keys is None:
        keys = []
        keys1 = set(data1[0].keys())
        keys2 = set(data2[0].keys())
        for k in keys2:
            if not k in keys1:
                keys.append(k)
    assert isinstance(keys, list)
    for k in keys:
        if not k in data2[0].keys():
            print(k)
            assert False

    odata = []
    map_dict = {}
    for e2 in data2:
        map_dict[e2["id"]] = e2

    for e1 in data1:
        oentry = e1
        e2 = map_dict[e1["id"]]
        for k in keys:
            oentry[k] = e2[k]
        odata.append(oentry)
    return odata

def rename_obj(data, keys, map_keys):
    if isinstance(keys, str):
        keys = [keys]
        map_keys = [map_keys]

    assert len(keys) == len(map_keys)
    for entry in data:
        for k, mk in zip(keys, map_keys):
            entry[mk] = entry[k]
            entry.pop(k)
    return data



