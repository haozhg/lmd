#!/usr/bin/env bash
set -x

python lmd/cli.py --batch-size 64 --max-seq-length 16 --try-models True
