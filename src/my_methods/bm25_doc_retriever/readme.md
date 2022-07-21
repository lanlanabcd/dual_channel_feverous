# Document Retrieval

bm25 top5

entity-matching + TF-IDF+roberta Re-rank top5

with rank sum ensemble method 



### Page Retriever

Following the baseline,  we begin retrieving pages through combination of entity matching and TF-IDF using DrQA. We first extract the top $k$ pages by matching extracted entities from the claim with Wikipedia articles. If less than k pages have been identified this way, the remaining pages are selected by Tf-IDF matching between the introductory sentence of an article and the claim. To use TF-IDF matching we need to build a TF-IDF index. Run:

```
PYTHONPATH=src python src/baseline/retriever/build_db.py --db_path data/feverous_wikiv1.db --save_path data/feverous-wiki-docs.db
PYTHONPATH=src python src/baseline/retriever/build_tfidf.py --db_path data/feverous-wiki-docs.db --out_dir data/index/
```


We can now extract the top k documents:

 ```
PYTHONPATH=src python src/baseline/retriever/document_entity_tfidf_ir.py  --model data/index/feverous-wiki-docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --db data/feverous-wiki-docs.db --count 150 --split dev --data_path data/
PYTHONPATH=src python src/baseline/retriever/document_entity_tfidf_ir.py  --model data/index/feverous-wiki-docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --db data/feverous-wiki-docs.db --count 150 --split train --data_path data/
PYTHONPATH=src python src/baseline/retriever/document_entity_tfidf_ir.py  --model data/index/feverous-wiki-docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --db data/feverous-wiki-docs.db --count 150 --split test --data_path data/
 ```

 We increase the value of $k$ compared to the baseline since we apply a two-stage re-ranking process where, first, a large number of pages to a given query is retrieved from a corpus using a standard mechanism (entity-matching and TF-IDF), and, second, the pages are scored and re-ranked using a more computationally-demanding method.

 ### Page Re-ranker

 We then re-rank the retrieved pages and get the pages with the top $m$ scores.

 ```
PYTHONPATH=src python src/re-ranker/page_reranker.py  --db ./data/feverous_wikiv1.db --split test --max_page 150 --max_rerank 5 --use_precomputed false --data_path data
PYTHONPATH=src python src/re-ranker/page_reranker.py  --db ./data/feverous_wikiv1.db --split dev --max_page 150 --max_rerank 5 --use_precomputed false --data_path data
PYTHONPATH=src python src/re-ranker/page_reranker.py  --db ./data/feverous_wikiv1.db --split train --max_page 150 --max_rerank 5 --use_precomputed false --data_path data
 ```

We rename the output file of the page re-ranker so that we could continue using it with the test of the baseline.

```
mv data/dev.pages.p150.r5.jsonl data/dev.pages.reranker.p5.jsonl
mv data/train.pages.p150.r5.jsonl data/train.pages.reranker.p5.jsonl
mv data/test.pages.p150.r5.jsonl data/test.pages.reranker.p5.jsonl
```

the top5 results should be:
```
dev:
Page coverage...
0.8325780278277229
mean_cover_rate: 0.8256319029512943
coverage_single_evi: 0.7865144789685056
coverage_evi_set: 0.7806083650190114

train:
Page coverage...
0.8269918994433859
mean_cover_rate: 0.8237583945264461
coverage_single_evi: 0.7844588451945267
coverage_evi_set: 0.7751329059769115
```


### BM25 Retriever

get top10 BM25 results from the Page Retriever 150

```
PYTHONPATH=src python src/my_methods/bm25_doc_retriever/doc_retrieve_bm25.py --split dev
PYTHONPATH=src python src/my_methods/bm25_doc_retriever/doc_retrieve_bm25.py --split train
PYTHONPATH=src python src/my_methods/bm25_doc_retriever/doc_retrieve_bm25.py --split test
```

the top5 results shound be:
```
dev:
Page coverage...
0.7884390567703601
mean_cover_rate: 0.7797667330557064
coverage_single_evi: 0.7532234199957726
coverage_evi_set: 0.741318124207858

train:
Page coverage...
0.7795984934688776
mean_cover_rate: 0.7760680930057154
coverage_single_evi: 0.7507867316254458
coverage_evi_set: 0.7359554501970795
```

### Ensemble page retriever results

```
PYTHONPATH=src python src/my_methods/bm25_doc_retriever/ensemble_retrieved_pages.py --split dev
PYTHONPATH=src python src/my_methods/bm25_doc_retriever/ensemble_retrieved_pages.py --split train
PYTHONPATH=src python src/my_methods/bm25_doc_retriever/ensemble_retrieved_pages.py --split test
```

top5 page retriever results should be:

```
PYTHONPATH=src python src/baseline/retriever/eval_doc_retriever.py --split dev --count 5
Page coverage...
0.8594838530892126
mean_cover_rate: 0.8512686341963911
coverage_single_evi: 0.82287042908476
coverage_evi_set: 0.8172370088719899
```



with first several entities in the claim and remove several pages start with unmatched years, top5 results:

```
dev:
Page coverage...
0.8890210029065088
mean_cover_rate: 0.881105981048947
coverage_single_evi: 0.8520397378989643
coverage_evi_set: 0.8481622306717364

train:
Page coverage...
0.8861427395920515
mean_cover_rate: 0.8828027867531452
coverage_single_evi: 0.8538427469171775
coverage_evi_set: 0.8485082268449032
```


