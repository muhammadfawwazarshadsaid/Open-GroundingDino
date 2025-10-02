#!/bin/bash
# ============================================
# Script LoRA fine-tuning GroundingDINO (multi-GPU ready)
# ============================================

GPU_NUM=$1        # jumlah GPU per node (misal 1 / 2 / 4)
CFG=$2            # path ke config file .py
DATASETS=$3       # path ke datasets json
OUTPUT_DIR=$4     # folder output log/checkpoint

# Distributed setup
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PORT=${PORT:-29500}

# Pretrain & Text encoder
PRETRAIN_MODEL_PATH=${PRETRAIN_MODEL_PATH:-"groundingdino_swint_ogc.pth"}
TEXT_ENCODER_TYPE=${TEXT_ENCODER_TYPE:-"bert-base-uncased"}

echo "
============================================
 GPU_NUM          = $GPU_NUM
 CFG              = $CFG
 DATASETS         = $DATASETS
 OUTPUT_DIR       = $OUTPUT_DIR
 PRETRAIN_MODEL   = $PRETRAIN_MODEL_PATH
 TEXT_ENCODER     = $TEXT_ENCODER_TYPE
 NNODES           = $NNODES
 NODE_RANK        = $NODE_RANK
 MASTER_ADDR:PORT = $MASTER_ADDR:$PORT
============================================
"

# Jalankan training dengan torchrun
torchrun --standalone \
  --nnodes=${NNODES} \
  --nproc_per_node=${GPU_NUM} \
  --node_rank=${NODE_RANK} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${PORT} \
  main.py \
    --output_dir "${OUTPUT_DIR}" \
    -c "${CFG}" \
    --datasets "${DATASETS}" \
    --pretrain_model_path "${PRETRAIN_MODEL_PATH}" \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target all-linear \
    --options text_encoder_type=${TEXT_ENCODER_TYPE} \
              freeze_keywords="['backbone','bert']" \
              lr_backbone=1e-6 \
              lr_linear_proj_mult=1e-6
