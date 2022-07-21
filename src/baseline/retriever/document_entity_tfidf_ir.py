import argparse
import json
import os.path

from tqdm import tqdm
from baseline.drqa import retriever
from baseline.drqa.retriever import DocDB
from utils.annotation_processor import AnnotationProcessor
from utils.wiki_processor import WikiDataProcessor
from baseline.drqa.retriever.doc_db import DocDB
from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage


def process(ranker, query, k=1):
    doc_names, doc_scores = ranker.closest_docs(query, k)

    return zip(doc_names, doc_scores)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',type=str)
    parser.add_argument('--count',type=int, default=1)
    parser.add_argument('--db',type=str)
    parser.add_argument('--data_path',type=str)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument("--db_path", type=str, default="data/feverous_wikiv1.db")
    args = parser.parse_args()
    #print(args)
    k = args.count
    split = args.split
    ranker = retriever.get_class('tfidf')(tfidf_path=args.model)
    annotation_processor = AnnotationProcessor("{}/{}.jsonl".format(args.data_path, args.split))
    db = DocDB(args.db)
    document_titles = set(db.get_doc_ids())

    # alias_dict_path = os.path.join(args.data_path, "alias_page_title_dict.json")
    # if os.path.exists(alias_dict_path):
    #     alias_dict = json.load(open(alias_dict_path,"r", encoding="utf-8"))
    # else:
    #     import re
    #     alias_dict = {}
    #     pt = re.compile('\"(.*?)\"')
    #     db = FeverousDB(args.db_path)
    #     for dt in tqdm(document_titles):
    #         page_json = db.get_doc_json(dt)
    #         wiki_page = WikiPage(dt, page_json)
    #         for sent_id in range(3):
    #             rd_sent = wiki_page.page_items.get(f"sentence_{sent_id}", None)
    #             if rd_sent and ("redirect" in rd_sent.content):
    #                 alias_titles = pt.findall(rd_sent.content)
    #                 for at in alias_titles:
    #                     alias_dict[at] = dt
    #                     #print(at, dt)
    #
    # print("alias_page_title_dict length:", len(alias_dict))

    with open("{0}/{1}.pages.p{2}.jsonl".format(args.data_path, args.split, k), "w") as f2:
        annotations = [annotation for annotation in annotation_processor]
        for i, annotation in enumerate(tqdm(annotations)):
            js = {}
            if not args.split == "test":
                js['id'] = annotation.get_id()
            js['claim'] = annotation.get_claim()
            entities = [el[0] for el in annotation.get_claim_entities()]
            # entities = [alias_dict.get(e, e) for e in entities]
            # entities = list(set(entities))
            entities = [ele for ele in entities if ele in document_titles]
            if len(entities) < args.count:
                pages = list(process(ranker, annotation.get_claim(), k=args.count))

            pages = [ele for ele in pages if ele[0] not in entities]
            pages_names = [ele[0] for ele in pages]

            entity_matches = [(el, 2000) if el in pages_names else (el, 500) for el in entities]
            # pages = process(ranker,annotation.get_claim(),k=k)
            js["predicted_pages"] = entity_matches + pages[:(args.count - len(entity_matches))]
            f2.write(json.dumps(js) + "\n")