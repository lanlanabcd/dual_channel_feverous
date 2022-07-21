import argparse
import json
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
        # Reverse the predicted pages
        sorted_p = list(sorted(line['predicted_pages'], reverse=True, key=lambda elem: elem[1]))
        # Get list of page title
        pages = [p[0] for p in sorted_p[:args.max_page]]
        p_lines = []
        for page in pages:
            page = unicodedata.normalize('NFD', page) # Unicode the page string to make the db instances read it
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
    return sentences
    # return " ".join(sentences)

def predicting_sentences(claim, lines, args, rerank_tokenizer, rerank_model):
    sentences = [line['sentence'] for line in lines]
    # Get training args
    trainer= model_trainer_2(args,rerank_model)
    # predicting
    features = rerank_tokenizer([claim]*len(sentences), sentences,  padding=True, truncation=True, return_tensors="pt")
    test_dataset = FEVEROUSDataset(features)
    scores = trainer.predict(test_dataset).predictions # Output an arrays of scores

    zipped_sentences_scores=[[p,scores[i]] for i,p in enumerate(lines)] 
    # zipped_sentences_scores=  [[item[0],item[1]] for item in zipped_sentences_scores] 
    zipped_sentences_scores = sorted(zipped_sentences_scores, key=lambda x: x[1][0], reverse=True)
    zipped_sentences_scores = zipped_sentences_scores[0: args.max_sent]

    return zipped_sentences_scores

def preprocess_new(line, rerank_tokenizer, rerank_model, args): 
    if 'predicted_pages' in line:
        # Reverse the predicted pages
        sorted_p = list(sorted(line['predicted_pages'], reverse=True, key=lambda elem: elem[1]))
        # Get list of page title
        pages = [p[0] for p in sorted_p[:args.max_page]]
        p_lines = []
        for page in pages:
            page = unicodedata.normalize('NFD', page) # Unicode the page string allowing db instance read it
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

        claim = line['claim']
        lines = []
        for p_line in p_lines:
            lines.append({
                "sentence": p_line[0],
                "page": p_line[1],
                "line_on_page": p_line[2]
            })

        zipped_sentences_scores = predicting_sentences(claim, lines, args, rerank_tokenizer, rerank_model)
        
        # Their format: List of List of json file and a numpy array of scores(only one score because of the output of huggingface method)
        # [
        #   [
        #       {
        #           'sentence': 'The Lindenbaum–Tarski algebra is considered the origin of the modern [[Algebraic_logic|algebraic logic]].', 
        #           'page': 'Lindenbaum–Tarski algebra', 
        #           'line_on_page': 'sentence_6'
        #       }, 
        #       array([4.6687713], dtype=float32)
        #   ], ... 
        #]
        # line["predicted_sentences"] = [(s[0]['sentence'], str(s[1][0])) for s in zipped_sentences_scores]
        line["predicted_sentences"] = [(s[0]["page"], s[0]["line_on_page"]) for s in zipped_sentences_scores]
    return line

class FEVEROUSDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])




def model_trainer(args, test_dataset):
    #model = RobertaForTokenClassification.from_pretrained(args.model_path, num_labels = 3, return_dict=True)
    rerank_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')

    training_args = TrainingArguments(
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    # warmup_steps=0,                # number of warmup steps for learning rate scheduler
    logging_dir='./logs',
    output_dir='./model_output'
    )

    trainer = Trainer(
    model=rerank_model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    eval_dataset=test_dataset,          # evaluation dataset
    )
    return trainer, rerank_model


def model_trainer_2(args,model):
    #model = RobertaForTokenClassification.from_pretrained(args.model_path, num_labels = 3, return_dict=True)

    training_args = TrainingArguments(
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    # warmup_steps=0,                # number of warmup steps for learning rate scheduler
    logging_dir=args.out_path,
    output_dir=args.out_path,
    disable_tqdm=True
    )

    trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    )
    return trainer



def flatten(t):
    return [item for sublist in t for item in sublist]

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
    parser.add_argument('--max_sent',type=int)
    parser.add_argument('--data_path',type=str)
    parser.add_argument('--out_path',type=str)
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
    args = parser.parse_args()
    doc_freqs=None
    if args.use_precomputed:
        _, metadata = utils.load_sparse_csr(args.model)
        doc_freqs = metadata['doc_freqs'].squeeze()

    db = DocDB(args.db)

    # print(db.get_doc_ids())

    jlr = JSONLineReader()
    rerank_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
    rerank_tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
    with open("{0}/{1}.pages.p{2}.jsonl".format(args.data_path, args.split, args.max_page),"r") as f, open("{0}/{1}.sentences.{4}.p{2}.s{3}.jsonl".format(args.out_path, args.split, args.max_page, args.max_sent,"precomputed" if args.use_precomputed else "not_precomputed"), "w+") as out_file:
        print("STARTING PREPROCESSING 1")
        lines = jlr.process(f)
        #lines = tf_idf_claims_batch(lines)
        print("STARTING PREPROCESSING 2")
        # gen=preprocess_new(lines[:])
        for line in tqdm(lines):
            line = preprocess_new(line, rerank_tokenizer, rerank_model, args)
            out_file.write(json.dumps(line) + "\n")