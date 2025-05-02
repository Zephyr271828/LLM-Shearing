#!/bin/bash

export 
PROJ_DIR='/scratch/yx3038/Research/pruning/LLM-Shearing'
MODEL_PATH="${PROJ_DIR}/ckpts/Llama-2-7b-composer/state_dict.pt"
OUTPUT_PATH="${PROJ_DIR}/ckpts/Llama-2-7b-hf"
MODEL_CLASS=LlamaForCausalLM
HIDDEN_SIZE=4096
NUM_ATTENTION_HEADS=32
NUM_HIDDEN_LAYERS=32
INTERMEDIATE_SIZE=11008
MODEL_NAME=

python3 -m llmshearing.utils.composer_to_hf save_composer_to_hf $MODEL_PATH $OUTPUT_PATH \
        model_class=${MODEL_CLASS} \
        hidden_size=${HIDDEN_SIZE} \
        num_attention_heads=${NUM_ATTENTION_HEADS} \
        num_hidden_layers=${NUM_HIDDEN_LAYERS} \
        intermediate_size=${INTERMEDIATE_SIZE} \
        num_key_value_heads=${NUM_ATTENTION_HEADS} \
        _name_or_path=${MODEL_NAME}

