#!/bin/bash
python train.py \
    --env-id Hopper-v4 \
    --total-timesteps 1000000 \
    --seed 1 \
    --track \
    --wandb-project-name "ppo-mujoco-unified" 