#!/bin/bash

source ../configs/setup.sh
check_sbash pruning_%j 16 96 1 1 "tandon_h100_1,tandon_a100_1,tandon_a100_2"

export 
MODEL_PATH="${PROJ_DIR}/ckpts/llama2_7b_pruning_scaling_doremi_to2.7b_sl2048/latest-rank0.pt"
OUTPUT_PATH="${PROJ_DIR}/ckpts/Llama-2-2.7b-hf"
MODEL_CLASS=LlamaForCausalLM
HIDDEN_SIZE=2560
NUM_ATTENTION_HEADS=20
NUM_HIDDEN_LAYERS=32
INTERMEDIATE_SIZE=6912
MODEL_NAME=Sheared-Llama-2.7B

python3 -m llmshearing.utils.composer_to_hf save_composer_to_hf $MODEL_PATH $OUTPUT_PATH \
        model_class=${MODEL_CLASS} \
        hidden_size=${HIDDEN_SIZE} \
        num_attention_heads=${NUM_ATTENTION_HEADS} \
        num_hidden_layers=${NUM_HIDDEN_LAYERS} \
        intermediate_size=${INTERMEDIATE_SIZE} \
        num_key_value_heads=${NUM_ATTENTION_HEADS} \
        _name_or_path=${MODEL_NAME}

