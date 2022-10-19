# lmd
Code for paper titled "Language Model Decomposition: Quantifying the Dependency and Correlation of Language Models" (accepted EMNLP 2022). The arxiv version is here: 

## To get started
```
python3 -m venv .venv
source .venv/bin/activate
```

### Git
```
git clone git@github.com:haozhg/lmd.git
cd lmd
pip install -e .
````

### Pip
```
pip install lmd
```

To run cli, `lmd --help` or `python -m lmd.cli --help`

```
$ lmd --help
usage: Language Model Decomposition [-h] [--target TARGET] [--basis BASIS] [--tokenizer-name TOKENIZER_NAME]
                                    [--max-seq-length MAX_SEQ_LENGTH] [--batch-size BATCH_SIZE]
                                    [--dataset-name DATASET_NAME] [--dataset-config-name DATASET_CONFIG_NAME]
                                    [--val-split-percentage VAL_SPLIT_PERCENTAGE]
                                    [--test-split-percentage TEST_SPLIT_PERCENTAGE]
                                    [--max-train-samples MAX_TRAIN_SAMPLES] [--max-val-samples MAX_VAL_SAMPLES]
                                    [--max-test-samples MAX_TEST_SAMPLES]
                                    [--preprocessing-num-workers PREPROCESSING_NUM_WORKERS]
                                    [--overwrite_cache OVERWRITE_CACHE] [--preprocess-dir PREPROCESS_DIR]
                                    [--embedding-dir EMBEDDING_DIR] [--results-dir RESULTS_DIR]
                                    [--models-dir MODELS_DIR] [--alpha ALPHA] [--log-level LOG_LEVEL]
                                    [--try-models TRY_MODELS] [--pre-select-multiplier PRE_SELECT_MULTIPLIER]
                                    [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --target TARGET       target model in LMD (default: bert-base-uncased)
  --basis BASIS         basis model in LMD, separated by comma (default: None)
  --tokenizer-name TOKENIZER_NAME
                        tokenizer used for generating sequences (used by all models as text input) (default: bert-base-
                        uncased)
  --max-seq-length MAX_SEQ_LENGTH
                        max_seq_length (default: 512)
  --batch-size BATCH_SIZE
                        batch size for model inference (default: 32)
  --dataset-name DATASET_NAME
                        The name of the dataset (corpus) to use (via the datasets library). E.g., (bookcorpus,
                        None)/(wikicorpus, raw_en)/(wikitext, wikitext-103-v1) (default: wikicorpus)
  --dataset-config-name DATASET_CONFIG_NAME
                        The configuration name of the dataset (corpus) to use (via the datasets library). E.g.,
                        (bookcorpus, None)/(wikicorpus, raw_en)/(wikitext, wikitext-103-v1) (default: raw_en)
  --val-split-percentage VAL_SPLIT_PERCENTAGE
                        The percentage of the train set used as validation set in case there's no validation split
                        (default: 5)
  --test-split-percentage TEST_SPLIT_PERCENTAGE
                        The percentage of the train set used as test set in case there's no test split (default: 5)
  --max-train-samples MAX_TRAIN_SAMPLES
                        max train samples (default: 128000)
  --max-val-samples MAX_VAL_SAMPLES
                        max validation samples (default: 12800)
  --max-test-samples MAX_TEST_SAMPLES
                        max test samples (default: 12800)
  --preprocessing-num-workers PREPROCESSING_NUM_WORKERS
                        preprocessing_num_workers for datasets.map() (default: None)
  --overwrite_cache OVERWRITE_CACHE
                        if we overwrite cache for datasets.map() (default: False)
  --preprocess-dir PREPROCESS_DIR
                        data dir to save preprocessed datasets (default: data/preprocess)
  --embedding-dir EMBEDDING_DIR
                        data dir to save embedding datasets (default: data/embeddings)
  --results-dir RESULTS_DIR
                        data dir to save LMD results (default: results)
  --models-dir MODELS_DIR
                        data dir to save LMD models (default: models)
  --alpha ALPHA         L2 regularization coefficient (default: 1e-06)
  --log-level LOG_LEVEL
                        logging level (default: INFO)
  --try-models TRY_MODELS
                        try load model and run inference for all models before running main (default: False)
  --pre-select-multiplier PRE_SELECT_MULTIPLIER
                        Filter based on rows in DatasetDict before filtering based on num of seq (default: 1)
  --seed SEED           random seed to ensure reproducibility (default: 42)
```


## Results
To reproduce the results in Appendix B, run `bash scripts/run.sh`

