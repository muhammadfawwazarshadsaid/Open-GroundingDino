#!/bin/bash
GPU_NUM=$1
CFG=$2
DATASETS=$3
OUTPUT_DIR=$4
LORA_R=${5:-8}
LORA_ALPHA=${6:-16}
LORA_DROPOUT=${7:-0.05}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PRETRAIN_MODEL_PATH=${PRETRAIN_MODEL_PATH:-"groundingdino_swint_ogc.pth"}
TEXT_ENCODER_TYPE=${TEXT_ENCODER_TYPE:-"bert-base-uncased"}

echo "
GPU_NUM = $GPU_NUM
CFG = $CFG
DATASETS = $DATASETS
OUTPUT_DIR = $OUTPUT_DIR
LORA_R = $LORA_R
LORA_ALPHA = $LORA_ALPHA
LORA_DROPOUT = $LORA_DROPOUT
"

python3 -m torch.distributed.launch --nproc_per_node="${GPU_NUM}" main.py \
    --output_dir "${OUTPUT_DIR}" \
    -c "${CFG}" \
    --datasets "${DATASETS}" \
    --pretrain_model_path "${PRETRAIN_MODEL_PATH}" \
    --options text_encoder_type=${TEXT_ENCODER_TYPE} \
              use_lora=True \
              lora_r=${LORA_R} \
              lora_alpha=${LORA_ALPHA} \
              lora_dropout=${LORA_DROPOUT} \
              freeze_keywords="['backbone','bert']" \
              lr_backbone=1e-6 \
              lr_linear_proj_mult=1e-6
