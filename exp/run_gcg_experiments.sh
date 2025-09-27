#!/bin/bash

# 批次 1：同时运行两个 13B 任务
echo "Starting batch 1: two 13B jobs"
nohup python defense.py \
  --model_name llama2-13b-chat \
  --attacker GCG \
  --defender SD_SpeculativeSafeDecoding \
  --GPT_API sk-7a2a9ad8ad06460ea13b61f31d73101f \
  > gcg_llama2_13b_chat_SD_SSD.log 2>&1 &
pid1=$!

nohup python defense.py \
  --model_name llama2-13b-chat \
  --attacker GCG \
  --defender SpeculativeSafeDecoding \
  --GPT_API sk-7a2a9ad8ad06460ea13b61f31d73101f \
  --defense_off \
  > gcg_llama2_13b_chat_nodefense.log 2>&1 &
pid2=$!

wait $pid1 $pid2
echo "Batch 1 complete."

# 批次 2：运行 SafeDecoding 任务
echo "Starting batch 2: SafeDecoding job SpeculativeSafeDecoding"
nohup python defense.py \
  --model_name llama2-13b-chat \
  --attacker GCG \
  --defender SafeDecoding \
  --GPT_API sk-7a2a9ad8ad06460ea13b61f31d73101f \
  > gcg_llama2_13b_chat_safeDecoding.log 2>&1 &
pid3=$!


nohup python defense.py \
  --model_name llama2-13b-chat \
  --attacker GCG \
  --defender SpeculativeSafeDecoding \
  --GPT_API sk-7a2a9ad8ad06460ea13b61f31d73101f \
  > GCG_llama2_13b_chat_speculativeSafeDecoding.log 2>&1 &
pid4=$!



wait $pid3 $pid4
echo "Batch 2 complete."