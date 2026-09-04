#!/bin/bash

# Checkpoints
export CHECKPOINT_DIR=outputs/local_training/bridgedrive_k60
# export ROUTES=data/benchmark_routes/bench2drive_split/bench2drive_111.xml
export ROUTES=3rd_party/Bench2Drive/leaderboard/data/bench2drive220.xml
# Set environment variables
export LEAD_PROJECT_ROOT=$(pwd)
export LD_LIBRARY_PATH=""
source $(pwd)/scripts/main.sh
export BENCHMARK_ROUTE_ID=$(basename $ROUTES .xml) # Last part of the route file name, e.g., 0 for 0.xml
export EVALUATION_OUTPUT_DIR=outputs/local_evaluation/$BENCHMARK_ROUTE_ID/
export PYTHONPATH=3rd_party/Bench2Drive/leaderboard:$PYTHONPATH
export PYTHONPATH=3rd_party/Bench2Drive/scenario_runner:$PYTHONPATH
export SCENARIO_RUNNER_ROOT=3rd_party/Bench2Drive/scenario_runner
export LEADERBOARD_ROOT=3rd_party/Bench2Drive/leaderboard
export IS_BENCH2DRIVE=1
export PLANNER_TYPE=only_traj
export SAVE_PATH=$EVALUATION_OUTPUT_DIR/
export PYTHONUNBUFFERED=1
export LEAD_CLOSED_LOOP_CONFIG="steer_modality=route throttle_modality=target_speed brake_modality=target_speed step_num=20 diffusion_speed=False produce_debug_video=False produce_debug_image=False produce_demo_video=False produce_demo_image=False"
# export LEAD_TRAINING_CONFIG="diffusion_speed=False"
export LEAD_TRAINING_CONFIG="diffusion_speed=False plan_anchor_path=anchor_utils/anchor_data/lead_cp_kmeans_60_10.npy"

set -x
set +e

# Recreate output folders
rm -rf $EVALUATION_OUTPUT_DIR/
mkdir -p $EVALUATION_OUTPUT_DIR

CUDA_VISIBLE_DEVICES=0 python3 3rd_party/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator_v2.py \
    --routes=$ROUTES \
    --track=SENSORS \
    --checkpoint=$EVALUATION_OUTPUT_DIR/checkpoint_endpoint.json \
    --agent=lead/inference/sensor_agent_bridgedrive.py \
    --agent-config=$CHECKPOINT_DIR \
    --debug=0 \
    --record=None \
    --resume=False \
    --port=8848 \
    --traffic-manager-port=8000 \
    --timeout=60 \
    --debug-checkpoint=$EVALUATION_OUTPUT_DIR/debug_checkpoint/debug_checkpoint_endpoint.txt \
    --traffic-manager-seed=0 \
    --repetitions=1 