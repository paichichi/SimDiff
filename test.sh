#!/bin/bash

# Default values for parameters
CUDA_DEVICES=${1:-"0,1,2,3"}  # Default to using GPU 0, 1, 2, 3 if not provided
SEED=${2:-39}                  # Default seed value (39)
LEARNING_RATE=${3:-0.001}      # Default learning rate (0.001)
DEVICE=${4:-"cuda:0"}          # Default device (cuda:0)
NUM_AUG=${5:-2}                # Default num_aug (2)

# Define the experiment configurations (dataset and scale)
experiments=(
    "FB15K_DB15K 0.2"
    "FB15K_DB15K 0.5"
    "FB15K_DB15K 0.8"
    "FB15K_YAGO15K 0.2"
    "FB15K_YAGO15K 0.5"
    "FB15K_YAGO15K 0.8"
)

# Number of repetitions per experiment
num_repeats=1

# Directory for logs
log_dir="logs"

# Create the log directory if it does not exist
mkdir -p $log_dir

# Loop through each experiment configuration (dataset and scale)
for exp in "${experiments[@]}"; do
    # Parse the dataset and scale values
    IFS=" " read -r dataset scale <<< "$exp"

    # Loop to run the experiment multiple times (num_repeats)
    for i in $(seq 1 $num_repeats); do
        echo "Running experiment: $dataset with scale $scale, iteration $i"

        # Define the log file for this specific experiment and repetition
        log_file="$log_dir/${dataset}_${scale}_run_$i.log"

        # Run the experiment and redirect output to the log file
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python src/run_diff.py \
            --file_dir "C:/Users/A/Documents/GitHub/data/mmkg/$dataset" \
            --device $DEVICE \
            --scale $scale \
            --num_aug $NUM_AUG \
            --rate $LEARNING_RATE \
            --lr .0005 \
            --epochs 500 \
            --hidden_units "300,300,300" \
            --check_point 10 \
            --bsize 3500 \
            --il \
            --il_start 0 \
            --semi_learn_step 5 \
            --csls \
            --csls_k 3 \
            --seed $SEED \
            --tau 0.1 \
            --structure_encoder "gat" \
            --img_dim 300 \
            --attr_dim 300 \
            --name_dim 300 \
            --char_dim 300 \
            --w_name \
            --w_char \
            >> $log_file 2>&1

        echo "Experiment: $dataset with scale $scale, iteration $i completed. Logs saved to $log_file"
    done
done
