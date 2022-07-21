# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2022/4/14 14:35
# Description:

import argparse
import os
from my_utils import load_jsonl_data, save_jsonl_data

# os.chdir("{dir}/DCUF_code")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--data_path', default="data", type=str)
    args = parser.parse_args()
    split = args.split

    in_path_sent = '{0}/{1}.sentences.roberta.p5.s5.jsonl'.format(args.data_path, split)
    in_path_cells = f"./src/my_methods/graph_cell_selector/cell_results/{split}_predicted_evidence.jsonl"
    in_path_gold = '{0}/{1}.jsonl'.format(args.data_path, split)

    output_path = "{0}/{1}.roberta.graph.p5.s5.t3.cells.jsonl".format(args.data_path, split)
    print("save evidence results to", os.path.abspath(output_path))

    data = load_jsonl_data(in_path_gold)
    sent_data = load_jsonl_data(in_path_sent)
    cell_data = load_jsonl_data(in_path_cells)
    assert len(data) == len(sent_data)
    assert len(sent_data) == len(cell_data)

    odata = [data[0]]
    for idx, (entry, sentry, centry) in enumerate(zip(data, sent_data, cell_data)):
        if idx == 0:
            continue
        assert entry["id"] == sentry["id"]
        assert sentry["id"] == centry["id"]
        oentry = {
            "id": entry["id"],
            "claim": entry["claim"],
            "label": entry["label"],
            "challenge": entry["challenge"],
            "evidence": entry["evidence"]
        }

        predicted_evidence = sentry["predicted_sentences"] + [c for c in centry["predicted_evidence"] if "_cell_" in c]
        oentry["predicted_evidence"] = predicted_evidence
        odata.append(oentry)

    save_jsonl_data(odata, output_path)


