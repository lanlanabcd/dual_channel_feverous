# DCUF FEVEROUS

This repository contains the code for the paper [Dual-channel Evidence Fusion for Fact Verification over Texts and Tables](https://aclanthology.org/2022.naacl-main.384.pdf) published in NAACL 2022.

You can download the [model outputs](https://drive.google.com/drive/folders/17nerxW9hP5sIFGDjtZUNiDdtsYMZIc4a) of the Page Retriever, Evidence Extraction, and Verdict Prediction steps directly and use them as the inputs to the next steps, which can save a lot of time.

## Shared Task
Visit [http://fever.ai](https://fever.ai/task.html) to find out more about the FEVER Workshop 2021 shared task @EMNLP on FEVEROUS.


## Install Requirements

Create a new Conda environment and install torch: 
```
conda create -n dcuf python=3.7.11
conda activate dcuf
conda install pytorch==1.8.0 -c pytorch

cd dual_channel_feverous
pip install -r requirements.txt

pip install torch-scatter -f https://data.pyg.org/whl/torch-1.8.0+${CUDA}.html

python -m spacy download en_core_web_sm
```

## Download PLM checkpoints
Download checkpoints from 
```
https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2
https://huggingface.co/google/tapas-large-finetuned-tabfact
https://huggingface.co/ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli
```

and put all downloaded files in the dual_channel_feverous/bert_weights directory.

## Prepare Data
Call the following script to download the FEVEROUS data:
```
sh ./scripts/download_data.sh 
```
Or you can download the data from the [FEVEROUS dataset page](https://fever.ai/dataset/feverous.html) directly. Namely:

* Training Data
* Development Data
* Test Data
* Wikipedia Data as a database (sqlite3)

if you download the data yourself, please unpack the given downloaded data to dual_channel_feverous/data/, and rename them to
* train.jsonl
* dev.jsonl
* test.jsonl.bk
* feverous_wikiv1.db

## Running the Code

### prepare the test data
add an id to each test case to make it have the same format as other splits
```
PYTHONPATH=src python ./src/my_scripts/add_id_to_test_set.py
```

### Page Retriever
See src/my_methods/bm25_doc_retriever/readme.md for the Page Retriever step

or you can simply use the page retrieval results we provide in the directory dual_channel_evidence/data, which is named {train, dev, test}.pages.p5.jsonl

### Sentence and Table Evidence Retrieval
you can skip this step and the next step (***Cell Evidence Selection***), and use the evidence extraction results we provide in ***data/{train, dev, test}.combined.not_precomputed.p5.s5.t3.cells.jsonl*** as the input to the ***Verdict Prediction*** step.

The top l sentences and q tables of the selected pages are then scored separately using TF-IDF. We set l=5 and q=3.
```
PYTHONPATH=src python src/baseline/retriever/sentence_tfidf_drqa.py --db data/feverous_wikiv1.db --max_page 5 --max_sent 5 --use_precomputed false --data_path data/ --split dev 
PYTHONPATH=src python src/baseline/retriever/table_tfidf_drqa.py --db data/feverous_wikiv1.db --max_page 5 --max_tabs 3 --use_precomputed false --data_path data/ --split dev
 ```

check the results of table evidence
```
PYTHONPATH=src python src/baseline/retriever/eval_tab_retriever.py --max_page 5 --max_tabs 3 --split {split}
```

The results of table evidence should be
```
dev
# 0.760822510822511
# 0.7559150169000484

train
0.7875517666423647
0.7857859759828386
```

Check the results of sentence evidence
```
PYTHONPATH=src python src/baseline/retriever/eval_sentence_retriever.py --max_page 5 --max_sent 5 --split {split}
```

The results of sentence evidence should be:
```
dev
# 0.6409232365145233
# 0.6254080351537985

train:
0.6865547333114521
0.6794650940283705
```

Combine both retrieved sentences and tables into one file:
 ```
 PYTHONPATH=src python src/baseline/retriever/combine_retrieval.py --data_path data --max_page 5 --max_sent 5 --max_tabs 3 --split {split}
 ```

combined results:
```
PYTHONPATH=src python src/baseline/retriever/eval_combined_retriever.py --max_page 5 --max_sent 5 --max_tabs 3 --split {split}

dev
# 0.6927645944633607
# 0.6816333820813999

train:
0.7285780757380688
0.7231134732218047
```

### Cell Evidence Selection

To train the cell extraction model run (You can also skip this step and download our checkpoint [new_feverous_cell_extractor](https://drive.google.com/file/d/1jlbewWC45_Zf3cQE12Isy95viQKGkYZx/view?usp=sharing), and move the downloaded checkpoint to models/new_feverous_cell_extractor:
```
PYTHONPATH=src python src/baseline/retriever/train_cell_evidence_retriever.py --wiki_path data/feverous_wikiv1.db --model_path models/new_feverous_cell_extractor --input_path data
 ```

To extract relevant cells from extracted tables, run:
 ```
 PYTHONPATH=src python src/baseline/retriever/predict_cells_from_table_multi_turn.py --input_path data/{split}.combined.not_precomputed.p5.s5.t3.jsonl --max_sent 5 --wiki_path data/feverous_wikiv1.db --model_path models/new_feverous_cell_extractor
  ```

test the results of cell retriever:
```
PYTHONPATH=src python src/baseline/retriever/eval_cell_retriever.py --max_page 5 --max_sent 5 --max_tabs 3 --split {split}
```

evaluate the retrieved evidence set
```
PYTHONPATH=src python evaluation/evaluate.py --input_path data/{split}.combined.not_precomputed.p5.s5.t3.cells.jsonl --use_gold_verdict

dev:
evidence precision: 0.15061699270090265
evidence recall: 0.43219264892268694
evidence f1: 0.22338531279897525
```

### Verdict Prediction
To train the verdict prediction model run:
(Or you can download our checkpoint from our checkpoint [dual_channel_checkpoint](https://drive.google.com/drive/folders/17nerxW9hP5sIFGDjtZUNiDdtsYMZIc4a) and place it in the models/prepro_feverous_verdict_predictor directory)
```
PYTHONPATH=src python src/my_methods/bi_modal_verdict_predictor/train_verdict_predictor.py --data_dir data
```

```
PYTHONPATH=src python src/my_methods/bi_modal_verdict_predictor/eval_verdict_predictor.py --data_dir data --test_ckpt dual_channel_checkpoint
```
 

The models are saved every n steps, thus specify the correct path during inference accordingly. 

## Evaluation
To evaluate your generated predictions locally, simply run the file `evaluate.py` as following:
```
PYTHONPATH=src python evaluation/evaluate.py --input_path data/dev.combined.not_precomputed.p5.s5.t3.cells.verdict.jsonl
```

Our final results is in ***submissions*** directory.
