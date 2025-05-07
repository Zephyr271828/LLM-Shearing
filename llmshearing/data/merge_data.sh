#!/bin/bash

export INPUT_DIR='/scratch/yx3038/Research/pruning/LLM-Shearing/llmshearing/data/sample_redpajama'
export OUTPUT_DIR=${INPUT_DIR}

cd data/

python3 merge_data.py \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --output_split eval_merge \
    --split_names arxiv book c4-rp cc github stackexchange \