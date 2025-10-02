#!/bin/bash
GPU_NUM=$1
CFG=$2
DATASETS=$3
OUTPUT_DIR=$4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PRETRAIN_MODEL_PATH=${PRETRAIN_MODEL_PATH:-"/path/to/groundingdino_swint_ogc.pth"}
TEXT_ENCODER_TYPE=${TEXT_ENCODER_TYPE:-"/path/to/bert-base-uncased"}
echo "
GPU_NUM = $GPU_NUM
CFG = $CFG
DATASETS = $DATASETS
OUTPUT_DIR = $OUTPUT_DIR
NNODES = $NNODES
NODE_RANK = $NODE_RANK
PORT = $PORT
MASTER_ADDR = $MASTER_ADDR
PRETRAIN_MODEL_PATH = $PRETRAIN_MODEL_PATH
TEXT_ENCODER_TYPE = $TEXT_ENCODER_TYPE
"

python3 -m torch.distributed.launch  --nproc_per_node="${GPU_NUM}" main.py \
        --output_dir "${OUTPUT_DIR}" \
        -c "${CFG}" \
        --datasets "${DATASETS}"  \
        --pretrain_model_path groundingdino_swint_ogc.pth \
        \
        --use_lora \
        --options text_encoder_type=bert-base-uncased \
        lora_r=8 \
        lora_alpha=16 \
        lora_dropout=0.1