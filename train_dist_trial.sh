#!/bin/bash
# Usage: bash train_dist.sh <GPU_NUM> <CFG> <DATASETS> <OUTPUT_DIR>

GPU_NUM=$1
CFG=$2
DATASETS=$3
OUTPUT_DIR=$4

# (opsional) multi-node env
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}

# pretrain & text encoder
PRETRAIN_MODEL_PATH=${PRETRAIN_MODEL_PATH:-"groundingdino_swint_ogc.pth"}
TEXT_ENCODER_TYPE=${TEXT_ENCODER_TYPE:-"bert-base-uncased"}

echo "
GPU_NUM = $GPU_NUM
CFG = $CFG
DATASETS = $DATASETS
OUTPUT_DIR = $OUTPUT_DIR
NNODES = $NNODES
NODE_RANK = $NODE_RANK
MASTER_ADDR = $MASTER_ADDR
MASTER_PORT = $MASTER_PORT
PRETRAIN_MODEL_PATH = $PRETRAIN_MODEL_PATH
TEXT_ENCODER_TYPE = $TEXT_ENCODER_TYPE
"

# --- Colab/VM single GPU tips ---
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
# Matikan fabric/IB yang nggak ada di Colab supaya NCCL lebih stabil
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# (opsional) lebih verbose debug NCCL: export NCCL_DEBUG=INFO

# Kurangi thread CPU biar stabil di Colab
export OMP_NUM_THREADS=1

torchrun \
  --nproc_per_node="${GPU_NUM}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  main.py \
    --output_dir "${OUTPUT_DIR}" \
    -c "${CFG}" \
    --datasets "${DATASETS}" \
    --pretrain_model_path "${PRETRAIN_MODEL_PATH}" \
    --options text_encoder_type="${TEXT_ENCODER_TYPE}"
