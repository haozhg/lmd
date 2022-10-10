#!/usr/bin/env bash
set -x

for max_seq_length in 2 4 8 16 32 64 128 256 512
do
    time python -m lmd.cli \
        --batch-size 64 \
        --max-seq-length ${max_seq_length} \
        --dataset-name wikicorpus \
        --dataset-config-name raw_en \
        --max-train-samples 128000 \
        --max-val-samples 12800 \
        --max-test-samples 12800 \
        --pre-select-multiplier 1 \
        --try-models True \
        --preprocess-dir data/preprocess/128k \
        --embedding-dir data/embeddings/128k \
        --models-dir models/128k \
        --results-dir results/128k \
        2>&1 | tee -a logs/lmd.log
done
