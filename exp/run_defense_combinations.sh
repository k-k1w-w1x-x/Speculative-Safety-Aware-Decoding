#!/bin/bash

# Array of alpha1 values
alpha1_values=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
# Array of alpha2 values
alpha2_values=(2 3)

# Loop through all combinations
for alpha1 in "${alpha1_values[@]}"; do
    for alpha2 in "${alpha2_values[@]}"; do
        
        if [ "$alpha1" = "0.3" ] && [ "$alpha2" = "2" ]; then
            echo "Skipping alpha1=0.3, alpha2=2"
            continue
        fi
        # Create the suffix for the log file
        suffix="alpha_${alpha1//./dot}_${alpha2}_new"
        
        echo "Starting run with alpha1=$alpha1, alpha2=$alpha2"
        
        # Run the command (removed & to make it run sequentially)
        nohup python defense.py \
            --model_name vicuna \
            --spec_alpha1 $alpha1 \
            --spec_alpha2 $alpha2 \
            --attacker PAIR \
            --defender SpeculativeSafeDecoding \
            --GPT_API sk-7a2a9ad8ad06460ea13b61f31d73101f \
            --additional_save_suffix $suffix \
            > PAIR_vicuna_speculativeSafeDecoding_$suffix.log 2>&1
        
        echo "Completed run with alpha1=$alpha1, alpha2=$alpha2"
    done
done 