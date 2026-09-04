#!/bin/bash

export LEAD_PROJECT_ROOT=$(pwd)
# module unload cuda
# module load cuda/12.1.1
source $(pwd)/scripts/main.sh
# export TMPDIR=path_to_your_tmp_folder/lead_tmp # change tmp dir if needed
export CUDA_VISIBLE_DEVICES=0

# export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=""

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1 # Shuts off numpy multithreading, to avoid threads spawning other threads.
export NCCL_P2P_DISABLE=1 # https://github.com/huggingface/accelerate/issues/314
export NCCL_P2P_LEVEL=NVL # https://github.com/huggingface/accelerate/issues/314
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nproc_per_node=$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=name --format=csv,noheader | wc -l) # Get number of GPUs available
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 50000))

export LEAD_TRAINING_CONFIG="logdir=outputs/local_training/posttrain_bridgedrive_k60_4090_32 \
load_file=data/lead_ckpt/tfv6/model_0030_2.pth \
carla_root=data/carla_leaderboard2 \
use_planning_decoder=true \
freeze_backbone=true \
diffusion_speed=false \
plan_anchor_path=anchor_utils/anchor_data/lead_cp_kmeans_60_10.npy \
batch_size=32 \
lr=1e-4"
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    --max_restarts=0 \
    lead/training/train_bridgedrive.py