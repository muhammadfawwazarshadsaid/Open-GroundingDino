#!/bin/bash
# Save this file as train_lora.sh

# --- ARGUMENTS ---
# $1: Number of GPUs
# $2: Path to the configuration file
# $3: Path to the datasets JSON file
# $4: Directory to save logs and model checkpoints

GPU_NUM=$1
CFG=$2
DATASETS=$3
OUTPUT_DIR=$4

# --- FIXED PARAMETERS ---
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# --- EXECUTION COMMAND ---
python3 -m torch.distributed.launch --nproc_per_node="${GPU_NUM}" main.py \
        --output_dir "${OUTPUT_DIR}" \
        -c "${CFG}" \
        --datasets "${DATASETS}"  \
        --pretrain_model_path groundingdino_swint_ogc.pth \
        \
        --use_lora \
        --lora_r 8 \
        --lora_alpha 16 \
        --lora_dropout 0.1 \
        --options text_encoder_type=bert-base-uncased