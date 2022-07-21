# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/12/10 16:19
# Description:

import argparse
import os
import string

from rank_bm25 import BM25Okapi
from my_utils import load_jsonl_data
from baseline.drqa.retriever import DocDB
from tqdm import tqdm
from cleantext import clean
from urllib.parse import unquote
import unicodedata
import re

pt = re.compile(r"\[\[.*?\|(.*?)]]")

def clean_text(text):
    text = re.sub(pt, r"\1", text)
    text = unquote(text)
    text = unicodedata.normalize('NFD', text)
    text = clean(text.strip(),fix_unicode=True,               # fix various unicode errors
    to_ascii=False,                  # transliterate to closest ASCII representation
    lower=False,                     # lowercase text
    no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
    no_urls=True,                  # replace all URLs with a special token
    no_emails=False,                # replace all email addresses with a special token
    no_phone_numbers=False,         # replace all phone numbers with a special token
    no_numbers=False,               # replace all numbers with a special token
    no_digits=False,                # replace all digits with a special token
    no_currency_symbols=False,      # replace all currency symbols with a special token
    no_punct=False,                 # remove punctuations
    replace_with_url="<URL>",
    replace_with_email="<EMAIL>",
    replace_with_phone_number="<PHONE>",
    replace_with_number="<NUMBER>",
    replace_with_digit="0",
    replace_with_currency_symbol="<CUR>",
    lang="en"                       # set to 'de' for German special handling
    )
    return text

import nltk
ignored_words = set(nltk.corpus.stopwords.words('english'))
punct_set = set(['.', "''", '``', ',', '(', ')'] + list(string.punctuation))
ignored_words = ignored_words.union(punct_set)

import nltk.stem
stemmizer = nltk.stem.SnowballStemmer('english')

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [stemmizer.stem(tk) for tk in tokens if tk.lower() not in ignored_words]
    return tokens

def main(args, doc_db):
    args.input_path = "{}/{}.pages.p{}.jsonl".format(args.data_dir, args.split, args.count)
    ori_data = load_jsonl_data(args.input_path)

    span = 200
    predicted_pages = []

    for stat in tqdm(range(0, len(ori_data), span)):
        end = min(stat + span, len(ori_data))
        data = ori_data[stat:end]
        cand_id_set = set()

        for idx, js in enumerate(data):
            for cand_id, score in js["predicted_pages"]:
                cand_id_set.add(cand_id)

        cand_id_lst = list(cand_id_set)

        corpus = [''.join([cand_id + " " for _ in range(3)])
                  + clean_text(doc_db.get_doc_text(cand_id)) for cand_id in cand_id_lst]
        tokenized_corpus = [tokenize(c)[:64] for c in corpus]
        del corpus

        bm25 = BM25Okapi(tokenized_corpus)

        for idx, js in enumerate(data):
            tokenized_query = tokenize(js["claim"])
            res_pages = bm25.get_top_n(tokenized_query, cand_id_lst, n=10)
            if not "id" in js:
                js["id"] = idx
            oentry = {
                "id": js["id"],
                "predicted_pages": list(zip(res_pages, list(range(len(res_pages)))))
            }
            predicted_pages.append(oentry)

    from my_utils import save_jsonl_data
    save_jsonl_data(predicted_pages, f"{args.data_dir}/{args.split}.pages.bm25.p10.jsonl")

    from baseline.retriever.eval_doc_retriever import page_coverage_obj
    page_coverage_obj(args, predicted_pages, max_predicted_pages=5)

def get_page_tokens():
    from mymodels.bm25_doc_retriever.doc_retrieve_bm25 import tokenize
    from mymodels.bm25_doc_retriever.doc_retrieve_bm25 import clean_text
    from baseline.drqa.retriever import DocDB
    db_path = "data/feverous-wiki-docs.db"
    doc_db = DocDB(db_path)
    text = doc_db.get_doc_text("Anarchism")
    t = tokenize(clean_text(text))
    return t

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default="dev")
    parser.add_argument('--count', type=int, default=150)
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()
    # os.chdir("{dir}/DCUF_code")
    db_path = "data/feverous-wiki-docs.db"
    doc_db = DocDB(db_path)
    main(args, doc_db)








