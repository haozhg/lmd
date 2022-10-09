#!/usr/bin/env bash
set -x

for max_seq_length in 1 2 4 8 16 32 64 128 256 512
do
    time python lmd/cli.py \
        --batch-size 64 \
        --max-seq-length ${max_seq_length} \
        --dataset-name wikicorpus \
        --dataset-config-name raw_en \
        --try-models True \
        2>&1 | tee -a lmd.log
done