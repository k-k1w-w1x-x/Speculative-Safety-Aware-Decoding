#!/bin/bash

# Function to run experiments for a specific alpha1 value
run_experiments() {
    local alpha1=$1
    local suffix=$(echo $alpha1 | sed 's/\./dot/')
    
    echo "Running experiments for alpha1 = $alpha1"
    
    # Run Harmful-HEx-PHI experiment
    echo "Running Harmful-HEx-PHI experiment..."
    CUDA_VISIBLE_DEVICES=6 nohup python defense.py \
        --model_name llama2 \
        --spec_alpha1 $alpha1 \
        --spec_alpha2 0.8 \
        --attacker Harmful-HEx-PHI_with_prefix20tokens_llama2 \
        --defender SpeculativeSafeDecoding \
        --GPT_API sk-7a2a9ad8ad06460ea13b61f31d73101f \
        --additional_save_suffix ${suffix}down_0dot8_ablation \
        > harmful20tokens_llama2_speculativeSafeDecoding_alpha_${suffix}down_0dot8_ablation.log 2>&1 &
    local pid1=$!
    echo "Harmful-HEx-PHI experiment started with PID: $pid1"
    while kill -0 $pid1 2>/dev/null; do
        sleep 60  # Check every minute
    done
    echo "Harmful-HEx-PHI experiment completed"
    
    # Run PAIR experiment
    echo "Running PAIR experiment..."
    CUDA_VISIBLE_DEVICES=6 nohup python defense.py \
        --model_name llama2 \
        --spec_alpha1 $alpha1 \
        --spec_alpha2 0.8 \
        --attacker PAIR \
        --defender SpeculativeSafeDecoding \
        --GPT_API sk-7a2a9ad8ad06460ea13b61f31d73101f \
        --additional_save_suffix ${suffix}down_0dot8_ablation \
        > PAIR_llama2_speculativeSafeDecoding_alpha_${suffix}down_0dot8_ablation.log 2>&1 &
    local pid2=$!
    echo "PAIR experiment started with PID: $pid2"
    while kill -0 $pid2 2>/dev/null; do
        sleep 60  # Check every minute
    done
    echo "PAIR experiment completed"
    
    # Run GCG experiment
    echo "Running GCG experiment..."
    CUDA_VISIBLE_DEVICES=6 nohup python defense.py \
        --model_name llama2 \
        --spec_alpha1 $alpha1 \
        --spec_alpha2 0.8 \
        --attacker GCG \
        --defender SpeculativeSafeDecoding \
        --GPT_API sk-7a2a9ad8ad06460ea13b61f31d73101f \
        --additional_save_suffix ${suffix}down_0dot8_ablation \
        > GCG_llama2_speculativeSafeDecoding_alpha_${suffix}down_0dot8_ablation.log 2>&1 &
    local pid3=$!
    echo "GCG experiment started with PID: $pid3"
    while kill -0 $pid3 2>/dev/null; do
        sleep 60  # Check every minute
    done
    echo "GCG experiment completed"
    
    echo "Experiments with alpha1 = $alpha1 completed"
}

# Run experiments for each alpha1 value
run_experiments 0.4
run_experiments 0.5
run_experiments 0.6

echo "All experiments completed!" 