#!/bin/bash

# ==============================================================================
# PPO Training Script - 40 runs distributed across 20 background jobs
# ==============================================================================

echo "Creating conda environment 'ppo'..."
conda create -n ppo python=3.10 -y

# Initialize conda for bash script
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ppo

echo "Installing requirements..."
pip install -r requirements.txt

# Bind all executions to GPU 0 as requested
export CUDA_VISIBLE_DEVICES=0

# The 8 sigma values and 5 seeds (40 combinations total)
sigmas=(0.00 0.05 0.10 0.20 0.30 0.50 0.80 1.00)
seeds=(0 1 2 3 4)

declare -a commands

# Flatten all (sigma, seed) pairs into an array
for sigma in "${sigmas[@]}"; do
    for seed in "${seeds[@]}"; do
        commands+=("python run_training.py --sigmas $sigma --seeds $seed")
    done
done

echo "Starting 20 background terminals (jobs) on GPU 0..."
echo "Each terminal will execute 2 training runs sequentially."

# Run in 20 background jobs, each running 2 commands sequentially
for ((i=0; i<40; i+=2)); do
    (
        echo "Terminal $((i/2 + 1)): Starting ${commands[i]}"
        ${commands[i]}
        
        echo "Terminal $((i/2 + 1)): Starting ${commands[i+1]}"
        ${commands[i+1]}
        
        echo "Terminal $((i/2 + 1)): Finished both runs."
    ) &
done

echo "All 20 background jobs have been dispatched!"
echo "Waiting for all jobs to complete... (Do not close this terminal if you want to wait)"
wait
echo "All 40 training runs completed successfully."
