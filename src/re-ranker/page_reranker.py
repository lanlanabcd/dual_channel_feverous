import argparse
import json
import os
from multiprocessing.pool import ThreadPool
import torch
from baseline.drqa.retriever import utils
from utils.log_helper import LogHelper
from tqdm import tqdm
import numpy as np

from baseline.drqascripts.build_tfidf_lines import OnlineTfidfDocRanker
from baseline.drqa.retriever.doc_db import DocDB
from utils.wiki_page import WikiPage
from utils.util import JSONLineReader
import unicodedata
from transformers import AutoTokenizer, AutoModelForSequenceClassification,TrainingArguments,Trainer
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

# os.chdir("../../")


def tf_idf_sim(claim, lines,freqs=None):
    tfidf = OnlineTfidfDocRanker(args,[line["sentence"] for line in lines],freqs)
    line_ids,scores = tfidf.closest_docs(claim,args.max_sent)
    ret_lines = []
    for idx,line in enumerate(line_ids):
        ret_lines.append(lines[line])
        ret_lines[-1]["score"] = scores[idx]
    return ret_lines



def tf_idf_claim(line):
    if 'predicted_pages' in line:
        sorted_p = list(sorted(line['predicted_pages'], reverse=True, key=lambda elem: elem[1]))

        pages = [p[0] for p in sorted_p[:args.max_page]]
        p_lines = []
        for page in pages:
            page = unicodedata.normalize('NFD', page)
            # lines = db.get_doc_lines(page)
            try:
                lines = json.loads(db.get_doc_json(page))
            except:
                continue
            current_page = WikiPage(page, lines)
            all_sentences = current_page.get_sentences()
            sentences = [str(sent) for sent in all_sentences[:len(all_sentences)]]
            sentence_ids = [sent.get_id() for sent in all_sentences[:len(all_sentences)]]
            # lines = [line.split("\t")[1] if len(line.split("\t")[1]) > 1 else "" for line in
            #          lines.split("\n")]
            # lines = [line.split('\t')[1] for i,line in enumerate(lines.split('[SEP]'))]

            p_lines.extend(zip(sentences, [page] * len(lines), sentence_ids))

        lines = []
        for p_line in p_lines:
            lines.append({
                "sentence": p_line[0],
                "page": p_line[1],
                "line_on_page": p_line[2]
            })

        scores = tf_idf_sim(line["claim"], lines, doc_freqs)

        line["predicted_sentences"] = [(s["page"], s["line_on_page"]) for s in scores]
    return line


def tf_idf_claims_batch(lines):
    with ThreadPool(args.num_workers) as threads:
        results = threads.map(tf_idf_claim, lines)
    return results

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_sentences(page):
    page = unicodedata.normalize('NFD', page)
    try:
        lines = json.loads(db.get_doc_json(page))
    except:
        return 
    current_page = WikiPage(page, lines)
    all_sentences = current_page.get_sentences()   
    sentences = [str(sent) for sent in all_sentences[:len(all_sentences)]]
    return " ".join(sentences)


def preprocess(d):
#[{'id': 7389, 'claim': 'Algebraic logic has five Logical system and Lindenbaum–Tarski algebra which includes Physics algebra and Nodal algebra (provide models of propositional modal logics).', 'predicted_pages': [['Abstract algebraic logic', '4.469703'], ['Modal algebra', '4.3675647'], ['Lindenbaum–Tarski algebra', '3.9394379'], ['Polyadic algebra', '2.8113985'], ['Leibniz operator', '2.199115']]}, {'id': 13969, 'claim': 'Aramais Yepiskoposyan played for FC Ararat Yerevan, an Armenian football club based in Yerevan during 1986 to 1991.', 'predicted_pages': [['Aramais Yepiskoposyan', '6.350273'], ['F.C. Ararat Tehran', '5.806695'], ['List of FC Ararat-Armenia records and statistics', '5.7052217'], ['FC Ararat-Armenia', '5.66559'], ['Hrayr Mkoyan', '5.5898714']]}]
    queries=[]
    passages=[]
    pages=[]
    for dd in tqdm(d):
        query=dd['claim']
        predicted_pages=[x[0] for x in dd['predicted_pages']]
        passage=[get_sentences(page) for page in predicted_pages]
        query=[query]*len(passage)
        queries.extend(query)
        passages.extend(passage)
        pages.extend(predicted_pages)
    return queries,passages,pages



# class FEVEROUSDataset(Dataset):
#     def __init__(self, encodings):
#         self.encodings = encodings
#
#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         return item
#
#     def __len__(self):
#         return len(self.encodings['input_ids'])


class FEVEROUSDataset(Dataset):
    def __init__(self, queries, passages, pages, tokenizer, batch_size):
        self.tokenizer = tokenizer
        self.queries = batchify(queries, batch_size)
        self.passages = batchify(passages, batch_size)
        self.pages = batchify(pages, batch_size)

    def __getitem__(self, idx):
        features = self.tokenizer(self.queries[idx],self.passages[idx], padding=True, truncation=True, return_tensors="pt")
        return features

    def __len__(self):
        return len(self.queries)





def model_trainer_2(args,model, test_dataset):
    #model = RobertaForTokenClassification.from_pretrained(args.model_path, num_labels = 3, return_dict=True)

    training_args = TrainingArguments(
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    # warmup_steps=0,                # number of warmup steps for learning rate scheduler
    logging_dir='./logs',
    output_dir='./model_output'
    )

    trainer = MyTrainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        eval_dataset=test_dataset,
        args=training_args,                  # training arguments, defined above
    )
    return trainer

class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(MyTrainer, self).__init__(*args, **kwargs)
        
    def get_test_dataloader(self):
        pass

@torch.no_grad()
def pred(model, batch, args):
    batch = {k: torch.tensor(v[0], dtype=v.dtype, device=args.device) for k, v in batch.items()}
    scores = list(model(**batch).logits.squeeze().detach().cpu().numpy())
    return scores

def batchify(l,n):
    return [l[i:i+n] for i in range(0,len(l),n)]

if __name__ == "__main__":
    LogHelper.setup()
    LogHelper.get_logger(__name__)

    parser = argparse.ArgumentParser()


    parser.add_argument('--db', type=str, help='/path/to/saved/db.db')
    parser.add_argument('--model', type=str, help='/path/to/saved/db.db')
    parser.add_argument('--in_file', type=str, help='/path/to/saved/db.db')
    parser.add_argument('--max_page',type=int)
    parser.add_argument('--max_rerank',type=int)
    parser.add_argument('--data_path',type=str)
    parser.add_argument('--device', type=str, default="cuda:2")
    parser.add_argument('--use_precomputed', type=str2bool, default=True)
    parser.add_argument('--split', type=str)
    parser.add_argument('--ngram', type=int, default=2,
                        help=('Use up to N-size n-grams '
                              '(e.g. 2 = unigrams + bigrams)'))
    parser.add_argument('--hash-size', type=int, default=int(np.math.pow(2, 24)),
                        help='Number of buckets to use for hashing ngrams')
    parser.add_argument('--tokenizer', type=str, default='simple',
                        help=("String option specifying tokenizer type to use "
                              "(e.g. 'corenlp')"))
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    doc_freqs=None
    if args.use_precomputed:
        _, metadata = utils.load_sparse_csr(args.model)
        doc_freqs = metadata['doc_freqs'].squeeze()

    db = DocDB(args.db)

    # print(db.get_doc_ids())

    jlr = JSONLineReader()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = args.device
    bert_path = './bert_weights/ms-marco-MiniLM-L-12-v2'
    rerank_model = AutoModelForSequenceClassification.from_pretrained(bert_path)
    rerank_model.to(device)
    rerank_tokenizer = AutoTokenizer.from_pretrained(bert_path)
    rerank_model.eval()
    with open("{0}/{1}.pages.p{2}.jsonl".format(args.data_path, args.split, args.max_page),"r") as f, open("{0}/{1}.pages.p{2}.r{3}.jsonl".format(args.data_path, args.split, args.max_page,args.max_rerank), "w+") as out_file:
        lines = jlr.process(f)

        queries, passages, pages = preprocess(lines)
        batch_size = 16
        test_dataset = FEVEROUSDataset(queries, passages, pages, rerank_tokenizer, batch_size)

        test_dataloader = DataLoader(test_dataset, 1, shuffle=False)

        all_scores=[]
        # trainer= model_trainer_2(args,rerank_model, test_dataset)
        for ii, batch in tqdm(enumerate(test_dataloader)):
            scores = pred(rerank_model, batch, args)
            # scores = [s[0] for s in scores]
            all_scores.extend(scores)

        scores=all_scores
        batch_queries=batchify(queries,args.max_page)
        batch_passages=batchify(passages,args.max_page)
        zipped_pages_scores=[[p,scores[i]] for i,p in enumerate(pages)]
        batch_zip_scores=batchify(zipped_pages_scores,args.max_page)
        reranked_pages=[sorted(r,key=lambda x: -x[1])[:args.max_rerank] for r in batch_zip_scores]
        reranked_pages= [[aaa[0],str(aaa[1])] for aa in reranked_pages for aaa in aa]
        reranked_pages=batchify(reranked_pages,args.max_rerank)

        for i,line in tqdm(enumerate(lines)):
            line['predicted_pages']=reranked_pages[i]
            out_file.write(json.dumps(line) + "\n")
