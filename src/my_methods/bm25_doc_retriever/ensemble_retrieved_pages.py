# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/12/13 16:26
# Description:
from my_utils import load_jsonl_data
import argparse
import os
from baseline.drqa.retriever import DocDB
from tqdm import tqdm
import re
from my_utils import average
import nltk

pt = re.compile(r"\[\[(.*?)\|.*?]]")
def retrieve_one_hop_pages(args, doc_db):
    args.input_path = "data/{}.pages.p{}.jsonl".format(args.split, args.count)
    data = load_jsonl_data(args.input_path)

    predicted_pages = []
    res_page_num = []
    ori_res_page_num = []
    for idx, js in tqdm(enumerate(data)):
        cand_pages = [cand_id for cand_id, score in js["predicted_pages"][:5]]
        for cand_id, score in js["predicted_pages"][:5]:
            text = doc_db.get_doc_text(cand_id)
            one_hop_page = pt.findall(text)
            one_hop_page = [ohp.replace("_", ' ') for ohp in one_hop_page]
            cand_pages.extend(one_hop_page)

        cand_pages = list(set(cand_pages))
        ori_predicted_pages = [cand_id for cand_id, score in js["predicted_pages"]]
        res_pages = [cp for cp in cand_pages if cp in ori_predicted_pages]
        oentry = {
            "id": js["id"],
            "predicted_pages": list(zip(res_pages, list(range(len(res_pages)))))
        }
        res_page_num.append(len(res_pages))
        ori_res_page_num.append(len(js["predicted_pages"][:5]))
        predicted_pages.append(oentry)

    print(average(ori_res_page_num))
    print(average(res_page_num))

    from baseline.retriever.eval_doc_retriever import page_coverage_obj
    page_coverage_obj(args, predicted_pages, max_predicted_pages=150)

def main(args, doc_db):
    def polish_cand_pages(cand_pages, claim, doc_db):
        pages_title = [cp[1] for cp in cand_pages]
        tokens = nltk.word_tokenize(claim)
        new_cands = []
        for stat in range(7):
            for wl in range(2, 10):
                cand_id = ' '.join(tokens[stat:stat+wl])
                if (not cand_id in pages_title) and (doc_db.get_doc_text(cand_id) is not None):
                    new_cands.append((0, cand_id))
        cand_pages = new_cands + cand_pages
        return cand_pages

    def remove_unmatch_year(cand_pages, claim, title_index = 0):
        new_cand_pages = []
        for cp in cand_pages:
            wd = cp[title_index].split()[0]
            try:
                if not (wd.isdigit() and (1500 <= int(wd) <= 2100)) or (wd in claim):
                    new_cand_pages.append(cp)
            except:
                if (not wd.isdigit()) or (wd in claim):
                    new_cand_pages.append(cp)
        return new_cand_pages
    # retrieve_one_hop_pages(args, doc_db)

    #ensemble two results
    data1 = load_jsonl_data(f"data/{args.split}.pages.reranker.p5.jsonl")
    data2 = load_jsonl_data(f"data/{args.split}.pages.bm25.p10.jsonl")
    predicted_pages = []
    merge_evi_num = []
    page_num = 5

    for js1, js2 in tqdm(zip(data1, data2)):
        cand_pages1 = [cand_id for cand_id, score in remove_unmatch_year(js1["predicted_pages"], js1["claim"])[:page_num]]
        cand_pages2 = [cand_id for cand_id, score in remove_unmatch_year(js2["predicted_pages"], js1["claim"])[:page_num]]
        
        # rank sum
        score_dict = {}
        for rk, cand_id in enumerate(cand_pages1):
            score_dict[cand_id] = [rk]
        for rk, cand_id in enumerate(cand_pages2):
            if cand_id in score_dict:
                score_dict[cand_id].append(rk)
            else:
                score_dict[cand_id] = [rk]

        cand_pages = []
        for cand_id in score_dict:
            if len(score_dict[cand_id]) == 1:
                score_dict[cand_id].append(page_num)
            cand_pages.append((average(score_dict[cand_id]), cand_id))
        cand_pages = sorted(cand_pages)
        cand_pages = polish_cand_pages(cand_pages, js1["claim"], doc_db)
        merge_evi_num.append(len(cand_pages))
        res_pages = [[cp[1], cp[0]] for cp in cand_pages[:page_num]]

        oentry = {
            "id": js2["id"],
            "claim": js1["claim"],
            "predicted_pages": res_pages
        }
        predicted_pages.append(oentry)

    save_path = f"data/{args.split}.pages.p5.jsonl"
    print(f"save to {save_path}")
    from my_utils import save_jsonl_data
    save_jsonl_data(predicted_pages, save_path)

    print(average(merge_evi_num)) #7.88
    from baseline.retriever.eval_doc_retriever import page_coverage_obj
    page_coverage_obj(args, predicted_pages, max_predicted_pages=page_num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default="dev")
    parser.add_argument('--count', type=int, default=150)
    args = parser.parse_args()
    # os.chdir("{dir}/DCUF_code")
    db_path = "data/feverous-wiki-docs.db"
    doc_db = DocDB(db_path)
    main(args, doc_db)
