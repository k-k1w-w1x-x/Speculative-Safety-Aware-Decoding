#!/bin/bash

# Function to run experiments for a specific alpha2 value
run_experiments() {
    local alpha2=$1
    local suffix=$(echo $alpha2 | sed 's/\./dot/')
    local pids=()
    
    echo "Running experiments for alpha2 = $alpha2"
    
    # Run Harmful-HEx-PHI experiment
    CUDA_VISIBLE_DEVICES=3 nohup python defense.py \
        --model_name llama2 \
        --spec_alpha1 0.3 \
        --spec_alpha2 $alpha2 \
        --attacker Harmful-HEx-PHI_with_prefix20tokens_llama2 \
        --defender SpeculativeSafeDecoding \
        --GPT_API sk-7a2a9ad8ad06460ea13b61f31d73101f \
        --additional_save_suffix 0dot3_${suffix}_ablation \
        > harmful20tokens_llama2_speculativeSafeDecoding_alpha_0dot3_${suffix}_ablation.log 2>&1 &
    pids+=($!)
    
    # Run PAIR experiment
    CUDA_VISIBLE_DEVICES=4 nohup python defense.py \
        --model_name llama2 \
        --spec_alpha1 0.3 \
        --spec_alpha2 $alpha2 \
        --attacker PAIR \
        --defender SpeculativeSafeDecoding \
        --GPT_API sk-7a2a9ad8ad06460ea13b61f31d73101f \
        --additional_save_suffix 0dot3_${suffix}_ablation \
        > PAIR_llama2_speculativeSafeDecoding_alpha_0dot3_${suffix}_ablation.log 2>&1 &
    pids+=($!)
    
    # Run GCG experiment
    CUDA_VISIBLE_DEVICES=5 nohup python defense.py \
        --model_name llama2 \
        --spec_alpha1 0.3 \
        --spec_alpha2 $alpha2 \
        --attacker GCG \
        --defender SpeculativeSafeDecoding \
        --GPT_API sk-7a2a9ad8ad06460ea13b61f31d73101f \
        --additional_save_suffix 0dot3_${suffix}_ablation \
        > GCG_llama2_speculativeSafeDecoding_alpha_0dot3_${suffix}_ablation.log 2>&1 &
    pids+=($!)
    
    echo "Waiting for experiments with alpha2 = $alpha2 to complete..."
    
    # Wait for all processes to complete
    for pid in "${pids[@]}"; do
        while kill -0 $pid 2>/dev/null; do
            sleep 60  # Check every minute
        done
    done
    
    echo "Experiments with alpha2 = $alpha2 completed"
}

# Run experiments for each alpha2 value
run_experiments 0.6
run_experiments 1.5
run_experiments 2.0

echo "All experiments completed!" 