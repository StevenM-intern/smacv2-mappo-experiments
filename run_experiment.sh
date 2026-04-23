#!/bin/bash

echo "================================="
echo "Running SMACv2 MAPPO Experiment"
echo "================================="

python light_train.py \
  --env_name smacv2 \
  --map_name terran_5_vs_5 \
  --cuda \
  --max_steps 10000000 \
  --seed 1