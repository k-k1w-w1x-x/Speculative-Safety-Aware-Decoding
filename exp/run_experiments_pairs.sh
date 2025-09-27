#!/bin/bash

# Group 1: vicuna with GCG attacker and different defenses
echo "Starting Group 1..."
nohup python defense.py \
  --model_name vicuna \
  --attacker GCG \
  --defender SpeculativeSafeDecoding \
  --defense_off \
  --GPT_API sk-7a2a9ad8ad06460ea13b61f31d73101f \
  > GCG_vicuna_nodefense.log 2>&1 &
pid1=$!

nohup python defense.py \
  --model_name vicuna \
  --spec_alpha1 0.3 \
  --spec_alpha2 2 \
  --attacker GCG \
  --defender SpeculativeSafeDecoding \
  --GPT_API sk-7a2a9ad8ad06460ea13b61f31d73101f \
  --additional_save_suffix alpha_0dot3_2 \
  > GCG_vicuna_speculativeSafeDecoding_alpha_0dot3_2.log 2>&1 &
pid2=$!

nohup python defense.py \
  --model_name vicuna \
  --attacker GCG \
  --defender SafeDecoding \
  --GPT_API sk-7a2a9ad8ad06460ea13b61f31d73101f \
  > GCG_vicuna_SafeDecoding.log 2>&1 &
pid3=$!

wait $pid1 $pid2 $pid3
echo "Group 1 complete."

# Group 2: vicuna with PAIR attacker and different defenses
echo "Starting Group 2..."
nohup python defense.py \
  --model_name vicuna \
  --attacker PAIR \
  --defender SpeculativeSafeDecoding \
  --defense_off \
  --GPT_API sk-7a2a9ad8ad06460ea13b61f31d73101f \
  > PAIR_vicuna_nodefense.log 2>&1 &
pid4=$!

nohup python defense.py \
  --model_name vicuna \
  --spec_alpha1 0.3 \
  --spec_alpha2 2 \
  --attacker PAIR \
  --defender SpeculativeSafeDecoding \
  --GPT_API sk-7a2a9ad8ad06460ea13b61f31d73101f \
  --additional_save_suffix alpha_0dot3_2 \
  > PAIR_vicuna_speculativeSafeDecoding_alpha_0dot3_2.log 2>&1 &
pid5=$!

nohup python defense.py \
  --model_name vicuna \
  --attacker PAIR \
  --defender SafeDecoding \
  --GPT_API sk-7a2a9ad8ad06460ea13b61f31d73101f \
  > PAIR_vicuna_SafeDecoding.log 2>&1 &
pid6=$!

wait $pid4 $pid5 $pid6
echo "Group 2 complete."

# Group 3: deep_align_llama2 with different attackers
echo "Starting Group 3..."
nohup python defense.py \
  --model_name deep_align_llama2 \
  --attacker GCG \
  --defender SpeculativeSafeDecoding \
  --defense_off \
  --GPT_API sk-7a2a9ad8ad06460ea13b61f31d73101f \
  > GCG_deep_align_llama2_nodefense.log 2>&1 &
pid7=$!

nohup python defense.py \
  --model_name deep_align_llama2 \
  --attacker PAIR \
  --defender SpeculativeSafeDecoding \
  --defense_off \
  --GPT_API sk-7a2a9ad8ad06460ea13b61f31d73101f \
  > PAIR_deep_align_llama2_nodefense.log 2>&1 &
pid8=$!

nohup python defense.py \
  --model_name deep_align_llama2 \
  --attacker Just-Eval \
  --defender SpeculativeSafeDecoding \
  --defense_off \
  --GPT_API sk-7a2a9ad8ad06460ea13b61f31d73101f \
  > justEval_deep_align_llama2_nodefense.log 2>&1 &
pid9=$!

wait $pid7 $pid8 $pid9
echo "Group 3 complete."

echo "All experiments completed!"