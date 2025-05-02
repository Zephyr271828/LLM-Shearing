#!/bin/bash
#SBATCH --job-name=test_eq_%j
#SBATCH --output=logs/test_eq_%j.out
#SBATCH --error=logs/test_eq_%j.err
#SBATCH --tasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

#SBATCH --account=bdhh-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --mem=128GB
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=all
#SBATCH --mail-user=yx3038@nyu.edu
#SBATCH --no-requeue

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh
conda activate llmshearing

export PROJ_DIR='/scratch/yx3038/Research/pruning/LLM-Shearing'
export MODEL_DIR='/scratch/yx3038/model_ckpt'

export HF_MODEL_NAME="${MODEL_DIR}/Llama-2-7b-hf"
export OUTPUT_PATH="${PROJ_DIR}/ckpts/Llama-2-7b-composer/state_dict.pt"
export HF_MODEL_NAME2="${PROJ_DIR}/ckpts/Llama-2-7b-hf"
export MODEL_SIZE=7B

python3 -m llmshearing.utils.test_composer_hf_eq $HF_MODEL_NAME $OUTPUT_PATH $MODEL_SIZE

# python3 -m llmshearing.utils.test_hf2_eq $HF_MODEL_NAME $HF_MODEL_NAME2