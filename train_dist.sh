#!/bin/bash
# -------------------------------------------
# GroundingDINO Training Script (Colab Safe)
# -------------------------------------------

GPU_NUM=$1
CFG=$2
DATASETS=$3
OUTPUT_DIR=$4
RESUME_CHECKPOINT=$5   # optional fifth arg
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29601}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PRETRAIN_MODEL_PATH=${PRETRAIN_MODEL_PATH:-"groundingdino_swint_ogc.pth"}
TEXT_ENCODER_TYPE=${TEXT_ENCODER_TYPE:-"bert-base-uncased"}

echo "
========================================
GPU_NUM            = $GPU_NUM
CFG                = $CFG
DATASETS           = $DATASETS
OUTPUT_DIR         = $OUTPUT_DIR
RESUME_CHECKPOINT  = ${RESUME_CHECKPOINT:-<none>}
NNODES             = $NNODES
NODE_RANK          = $NODE_RANK
PORT               = $PORT
MASTER_ADDR        = $MASTER_ADDR
PRETRAIN_MODEL_PATH= $PRETRAIN_MODEL_PATH
TEXT_ENCODER_TYPE  = $TEXT_ENCODER_TYPE
========================================
"

# Create output dir if not exist
mkdir -p "${OUTPUT_DIR}"

# Find existing checkpoint if resume not provided
if [ -z "$RESUME_CHECKPOINT" ] && [ -f "${OUTPUT_DIR}/checkpoint_latest.pth" ]; then
    RESUME_CHECKPOINT="${OUTPUT_DIR}/checkpoint_latest.pth"
    echo "🟡 Found existing checkpoint: ${RESUME_CHECKPOINT}"
fi

# Run distributed training (nohup + log)
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "🔁 Resuming training from checkpoint..."
    nohup python3 -m torch.distributed.launch --nproc_per_node="${GPU_NUM}" main.py \
        --output_dir "${OUTPUT_DIR}" \
        -c "${CFG}" \
        --datasets "${DATASETS}" \
        --pretrain_model_path "${PRETRAIN_MODEL_PATH}" \
        --options text_encoder_type="${TEXT_ENCODER_TYPE}" \
        --resume "${RESUME_CHECKPOINT}" \
        > "${OUTPUT_DIR}/train.log" 2>&1 &
else
    echo "🚀 Starting new training run..."
    nohup python3 -m torch.distributed.launch --nproc_per_node="${GPU_NUM}" main.py \
        --output_dir "${OUTPUT_DIR}" \
        -c "${CFG}" \
        --datasets "${DATASETS}" \
        --pretrain_model_path "${PRETRAIN_MODEL_PATH}" \
        --options text_encoder_type="${TEXT_ENCODER_TYPE}" \
        > "${OUTPUT_DIR}/train.log" 2>&1 &
fi

echo "✅ Training started. Logs are being saved to: ${OUTPUT_DIR}/train.log"
echo "Use this to monitor progress:"
echo "    tail -f ${OUTPUT_DIR}/train.log"
