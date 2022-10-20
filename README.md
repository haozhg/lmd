# lmd
Code for paper titled "Language Model Decomposition: Quantifying the Dependency and Correlation of Language Models" (accepted to EMNLP 2022). The arxiv version is here: https://arxiv.org/abs/2210.10289

## Install
Create virtual env if needed
```
python3 -m venv .venv
source .venv/bin/activate
```

Install from pip
```
pip install nlp.lmd
```

Install from source
```
git clone git@github.com:haozhg/lmd.git
cd lmd
pip install -e .
````

To use lmd cli, run `lmd --help` or `python -m lmd.cli --help`

```
$ lmd --help
usage: Language Model Decomposition [-h] [--target TARGET] [--basis BASIS]
                                    [--tokenizer-name TOKENIZER_NAME]
                                    [--max-seq-length MAX_SEQ_LENGTH]
                                    [--batch-size BATCH_SIZE]
                                    [--dataset-name DATASET_NAME]
                                    [--dataset-config-name DATASET_CONFIG_NAME]
                                    [--val-split-percentage VAL_SPLIT_PERCENTAGE]
                                    [--test-split-percentage TEST_SPLIT_PERCENTAGE]
                                    [--max-train-samples MAX_TRAIN_SAMPLES]
                                    [--max-val-samples MAX_VAL_SAMPLES]
                                    [--max-test-samples MAX_TEST_SAMPLES]
                                    [--preprocessing-num-workers PREPROCESSING_NUM_WORKERS]
                                    [--overwrite_cache OVERWRITE_CACHE]
                                    [--preprocess-dir PREPROCESS_DIR]
                                    [--embedding-dir EMBEDDING_DIR]
                                    [--results-dir RESULTS_DIR]
                                    [--models-dir MODELS_DIR] [--alpha ALPHA]
                                    [--log-level LOG_LEVEL]
                                    [--try-models TRY_MODELS]
                                    [--pre-select-multiplier PRE_SELECT_MULTIPLIER]
                                    [--seed SEED]
```

## Results
To reproduce the results in Appendix B of the paper, run `bash scripts/run.sh`. The results are also stored in [`results/128k`](./results/128k/)

## Citation
If you find this paper/code useful, please cite us:
```
@misc{https://doi.org/10.48550/arxiv.2210.10289,
  doi = {10.48550/ARXIV.2210.10289},
  url = {https://arxiv.org/abs/2210.10289},
  author = {Zhang, Hao},
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences, I.2.7, 68T50 (Primary) 68T30, 68T07 (Secondary)},
  title = {Language Model Decomposition: Quantifying the Dependency and Correlation of Language Models},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```