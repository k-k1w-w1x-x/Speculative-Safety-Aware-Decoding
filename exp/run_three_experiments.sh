#!/bin/bash

# Run first experiment with llama2
nohup python defense.py \
  --model_name llama2 \
  --attacker Harmful-HEx-PHI_with_prefix20tokens_llama2 \
  --defender SD_SpeculativeSafeDecoding \
  --GPT_API sk-7a2a9ad8ad06460ea13b61f31d73101f > harmful20tokens_llama_2_7b_chat_SD_SSD.log 2>&1 &

# Wait for the first experiment to complete
wait

# Run second experiment with llama2-13b-chat
nohup python defense.py \
  --model_name llama2-13b-chat \
  --attacker Harmful-HEx-PHI_with_prefix20tokens_llama2 \
  --defender SD_SpeculativeSafeDecoding \
  --GPT_API sk-7a2a9ad8ad06460ea13b61f31d73101f > harmful20tokens_llama_2_13b_chat_SD_SSD.log 2>&1 &

# Wait for the second experiment to complete
wait
