mkdir -p data
mkdir -p models
wget -O data/train.jsonl https://fever.ai/download/feverous/feverous_train_challenges.jsonl
wget -O data/dev.jsonl https://fever.ai/download/feverous/feverous_dev_challenges.jsonl
wget -O data/test.jsonl.bk https://fever.ai/download/feverous/feverous_test_unlabeled.jsonl
# wget -O data/feverous-wiki-pages-db.zip https://s3-eu-west-1.amazonaws.com/fever.public/feverous/feverous-wiki-pages-db.zip
# unzip data/feverous-wiki-pages-db.zip
