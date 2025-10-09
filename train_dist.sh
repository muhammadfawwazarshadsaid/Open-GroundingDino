#!/bin/bash
GPU_NUM=$1
CFG=$2
DATASETS=$3
OUTPUT_DIR=$4
RESUME_CHECKPOINT=$5

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

mkdir -p "${OUTPUT_DIR}"

if [ "${GPU_NUM}" -le 1 ]; then
  echo "🚀 Single-GPU (no DDP)"
  python3 main.py \
    --output_dir "${OUTPUT_DIR}" \
    -c "${CFG}" \
    --datasets "${DATASETS}" \
    --pretrain_model_path groundingdino_swint_ogc.pth \
    ${RESUME_CHECKPOINT:+--resume "${RESUME_CHECKPOINT}"} \
    --options text_encoder_type=bert-base-uncased \
    > "${OUTPUT_DIR}/train.log" 2>&1
else
  echo "🚀 Multi-GPU (torchrun)"
  PORT=${PORT:-29500}
  MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
  torchrun --nproc_per_node="${GPU_NUM}" --master_addr="${MASTER_ADDR}" --master_port="${PORT}" \
    main.py \
    --output_dir "${OUTPUT_DIR}" \
    -c "${CFG}" \
    --datasets "${DATASETS}" \
    --pretrain_model_path groundingdino_swint_ogc.pth \
    ${RESUME_CHECKPOINT:+--resume "${RESUME_CHECKPOINT}"} \
    --options text_encoder_type=bert-base-uncased \
    > "${OUTPUT_DIR}/train.log" 2>&1
fi
