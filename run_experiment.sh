#!/bin/bash

echo "================================="
echo " SMACv2 MAPPO Experiment Runner "
echo "================================="

# Activate conda (edit path if needed)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate marl-ppo-suite

# Default values
STEPS=10000000
SEED=1

MODE=$1

if [ -z "$MODE" ]; then
    echo "Usage:"
    echo "./run_experiment.sh baseline"
    echo "./run_experiment.sh medivac"
    echo "./run_experiment.sh zerg"
    echo "./run_experiment.sh protoss"
    exit 1
fi

echo "Running mode: $MODE"

# ==============================
# EXPERIMENT SWITCH
# ==============================

if [ "$MODE" = "baseline" ]; then
    echo "Running BASELINE (Terran 5v5)..."
    python light_train.py \
        --env_name smacv2 \
        --map_name terran_5_vs_5 \
        --max_steps $STEPS \
        --seed $SEED

elif [ "$MODE" = "medivac" ]; then
    echo "Running MEDIVAC HEAVY (Terran 5v5)..."
    python light_train.py \
        --env_name smacv2 \
        --map_name terran_5_vs_5 \
        --max_steps $STEPS \
        --seed $SEED \
        --use_medivac_heavy

elif [ "$MODE" = "zerg" ]; then
    echo "Running ZERG TEST (5v5)..."
    python light_train.py \
        --env_name smacv2 \
        --map_name zerg_5_vs_5 \
        --max_steps $STEPS \
        --seed $SEED

elif [ "$MODE" = "protoss" ]; then
    echo "Running PROTOSS TEST (5v5)..."
    python light_train.py \
        --env_name smacv2 \
        --map_name protoss_5_vs_5 \
        --max_steps $STEPS \
        --seed $SEED

else
    echo "Invalid mode"
    exit 1
fi