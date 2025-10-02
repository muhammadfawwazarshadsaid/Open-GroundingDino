#!/bin/bash
# ============================================
# LoRA Fine-tuning Launcher for GroundingDINO
# ============================================

GPU_NUM=$1
CFG=$2
DATASETS=$3
OUTPUT_DIR=$4

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PRETRAIN_MODEL_PATH=${PRETRAIN_MODEL_PATH:-"groundingdino_swint_ogc.pth"}
TEXT_ENCODER_TYPE=${TEXT_ENCODER_TYPE:-"bert-base-uncased"}

echo "============================================"
echo " GPU_NUM          = $GPU_NUM"
echo " CFG              = $CFG"
echo " DATASETS         = $DATASETS"
echo " OUTPUT_DIR       = $OUTPUT_DIR"
echo " PRETRAIN_MODEL   = $PRETRAIN_MODEL_PATH"
echo " TEXT_ENCODER     = $TEXT_ENCODER_TYPE"
echo " NNODES           = $NNODES"
echo " NODE_RANK        = $NODE_RANK"
echo " MASTER_ADDR:PORT = $MASTER_ADDR:$PORT"
echo "============================================"

torchrun --standalone --nnodes=${NNODES} --nproc_per_node=${GPU_NUM} main.py \
    --output_dir "${OUTPUT_DIR}" \
    -c "${CFG}" \
    --datasets "${DATASETS}" \
    --pretrain_model_path "${PRETRAIN_MODEL_PATH}" \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target all-linear \
    --options text_encoder_type=${TEXT_ENCODER_TYPE}
