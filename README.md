# Speculative Safety-Aware Decoding

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


The official implementation of our **EMNLP 2025 main conference paper**
**["Speculative Safety-Aware Decoding"](https://arxiv.org/abs/2508.17739)**
by **Xuekang Wang, Shengyu Zhu, and Xueqi Cheng**.

This project is built upon the [SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding](https://github.com/uw-nsl/SafeDecoding).We sincerely appreciate the work done by the original authors.

------

## 🚀 Quick Start

### Environment Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/k-k1w-w1x-x/Speculative-Safety-Aware-Decoding.git
   cd Speculative-Safety-Aware-Decoding
   ```

2. **Create virtual environment**
   ```bash
   conda create -n SSD python=3.10
   conda activate SSD
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install local packages**
   ```bash
   pip install -e ./peft
   pip install -e ./just_eval
   ```

5. **Prepare API Keys**
   
   Before running experiments, you need to prepare API keys for evaluation:
   
   - **Qwen API**: Required for safety evaluation and judgment
     - Get your API key from [Qwen API platform](https://dashscope.aliyun.com/)
     - Pass the API key via `--GPT_API` parameter in command line (see Running Examples below)


### Model Download

1. **Download pre-trained models**
   
   Supported models include:
   - **Vicuna-7B**: `lmsys/vicuna-7b-v1.5`
   - **Llama-2-7B**: `meta-llama/Llama-2-7b-chat-hf`
   - **Llama-2-13B**: `meta-llama/Llama-2-13b-chat-hf`
   - **TinyLlama**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
   - **DeepAlign Models**: Need to be trained using the [shallow-vs-deep-alignment](https://github.com/Unispac/shallow-vs-deep-alignment) repository. Specifically, for the safety-augmented model of Llama3.2-1B-Instruct, you can download it directly from https://huggingface.co/wxk123/llama-3.2-1b-instruct-augmented


2. **Configure model paths**
   
   Model paths are configured directly in `exp/defense.py`. To run experiments, you need to modify at least:
   
   **A. Original Model Paths (lines 81-113):**
   ```python
   # Load model and template
   if args.model_name == "vicuna":
       model_name = "/path/to/your/vicuna_model"  # Update this path
       template_name = 'vicuna'
   elif args.model_name == "llama2":
       model_name = "/path/to/your/llama2_model"  # Update this path
       template_name = 'llama-2'
   # ... other model configurations
   ```
   
   **B. Expert Model Paths (lines 146 or 153):**
   ```python
   # For SpeculativeSafeDecoding defender
   if args.model_name == "llama3.1_8b": # only require when original model is LLama3.1
       expert_model, expert_tokenizer = load_model_and_tokenizer("/path/to/your/llama3.2_1b_augmented_expert_model", ...)
   else: # use this expert model to gain most of the results reported in the paper.
       expert_model, expert_tokenizer = load_model_and_tokenizer("/path/to/your/deepAlign_TinyLlama_expert_model", ...)
   ```
   
   **Required**: Replace the paths with your actual model directories:
   - Original models: `/path/to/your/*_model` → `/home/user/models/your-model-name`
   - Expert models: `/path/to/your/*_expert_model` → `/home/user/expert-models/your-expert-model`
   
   **Note for DeepAlign Models**: 
   The following models require training using the [shallow-vs-deep-alignment](https://github.com/Unispac/shallow-vs-deep-alignment) repository:
   - `deep_align_vicuna`
   - `deep_align_llama2` 
   - `llama3.2_1b_augmented`
   - `deepAlign_llama3.1_8b`
   
   To train these models, follow the instructions in the [shallow-vs-deep-alignment repository](https://github.com/Unispac/shallow-vs-deep-alignment) for data augmentation and fine-tuning with safety constraints.
   


### Running Examples

**Important**: Replace `YOUR_API_KEY` in the commands below with your actual Qwen API key from [Qwen API platform](https://dashscope.aliyun.com/).

1.Run Prefilling Attack with SSD:
for llama2-13b-chat:

```bash
nohup python defense.py \
  --model_name llama2-13b-chat \
  --spec_alpha1 0.3 \
  --spec_alpha2 0.8 \
  --attacker Harmful-HEx-PHI_with_prefix20tokens_llama2 \
  --defender SpeculativeSafeDecoding \
  --GPT_API YOUR_API_KEY \
  --additional_save_suffix 0dot3_0dot8 \
  > harmful20tokens_llama2-13b-chat_speculativeSafeDecoding_alpha_0dot3_0dot8.log 2>&1 &
```

for llama2-7b-chat:

```bash
nohup python defense.py \
  --model_name llama2-7b-chat \
  --spec_alpha1 0.3 \
  --spec_alpha2 0.8 \
  --attacker Harmful-HEx-PHI_with_prefix20tokens_llama2 \
  --defender SpeculativeSafeDecoding \
  --GPT_API YOUR_API_KEY \
  --additional_save_suffix 0dot3_0dot8 \
  > harmful20tokens_llama2-7b-chat_speculativeSafeDecoding_alpha_0dot3_0dot8.log 2>&1 &
```

for vicuna:

```bash
nohup python defense.py \
  --model_name vicuna \
  --spec_alpha1 0.45 \
  --spec_alpha2 2 \
  --attacker Harmful-HEx-PHI_with_prefix20tokens_llama2 \
  --defender SpeculativeSafeDecoding \
  --GPT_API YOUR_API_KEY \
  --additional_save_suffix 0dot45_2 \
  > harmful20tokens_vicuna_speculativeSafeDecoding_alpha_0dot45_2.log 2>&1 &    
```

2. Run GSM8K with SSD:
   for llama2-13b-chat:

```bash
nohup python defense.py \
  --model_name llama2-13b-chat \
  --spec_alpha1 0.3 \
  --spec_alpha2 0.8 \
  --attacker gsm8k \
  --defender SpeculativeSafeDecoding \
  --GPT_API YOUR_API_KEY \
  --additional_save_suffix 0dot3_0dot8 \
  > gsm8k_llama2-13b-chat_speculativeSafeDecoding_alpha_0dot3_0dot8.log 2>&1 &
```

for llama2-7b-chat:

```bash
nohup python defense.py \
  --model_name llama2 \
  --spec_alpha1 0.3 \
  --spec_alpha2 0.8 \
  --attacker gsm8k \
  --defender SpeculativeSafeDecoding \
  --GPT_API YOUR_API_KEY \
  --additional_save_suffix 0dot3_0dot8 \
  > gsm8k_llama2-7b-chat_speculativeSafeDecoding_alpha_0dot3_0dot8.log 2>&1 &
```

for vicuna:

```bash
nohup python defense.py \
  --model_name vicuna \
  --spec_alpha1 0.45 \
  --spec_alpha2 2 \
  --attacker gsm8k \
  --defender SpeculativeSafeDecoding \
  --GPT_API YOUR_API_KEY \
  --additional_save_suffix 0dot45_2 \
  > gsm8k_vicuna_speculativeSafeDecoding_alpha_0dot45_2.log 2>&1 &
```

## More Experiments

We have tested the following experiments:

- The attacker being set to one of the following:GCG,PAIR,Harmful-HEx-PHI_with_prefix{number of tokens}tokens_llama2,AdvBench,DeepInception,HEx-PHI,Just-Eval,gsm8k.
- The defense method being set to one of the following:SpeculativeSafeDecoding,SafeDecoding,Self-Exam,ICD,Paraphase,or just add --defense_off.
- The model being set to one of the following:llama2-13b-chat,llama2(stands for llama2-7b-chat),vicuna,llama3.1_8b.

For the command of other kinds of attacks, defense methods, and evaluation benchmarks, please refer to the [SafeDecoding](https://github.com/uw-nsl/SafeDecoding) repository.We followed their workflow to implement the experiments.

## 📑 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{wang2025speculative,
  title={Speculative Safety-Aware Decoding},
  author={Wang, Xuekang and Zhu, Shengyu and Cheng, Xueqi},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2025}
}
```

