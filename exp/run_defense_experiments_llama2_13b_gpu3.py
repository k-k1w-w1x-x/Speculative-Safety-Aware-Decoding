import subprocess
import time
import os

# List of attackers
attackers = ['AdvBench', 'AutoDAN', 'DeepInception', 'HEx-PHI']

# Base command template for GPU 2
base_cmd = "CUDA_VISIBLE_DEVICES=3 python defense.py --model_name llama2-13b-chat --spec_alpha1 0.3 --spec_alpha2 0.8 --attacker {} --defender SpeculativeSafeDecoding --GPT_API sk-7a2a9ad8ad06460ea13b61f31d73101f --additional_save_suffix 0dot3_0dot8_new"

# Function to run a command and wait for it to complete
def run_command(cmd, log_file):
    print(f"Running command: {cmd}")
    # Open log file for writing
    with open(log_file, 'w') as f:
        # Run the command and redirect both stdout and stderr to the log file
        process = subprocess.Popen(cmd, shell=True, stdout=f, stderr=f)
        # Wait for the process to complete
        process.wait()
        print(f"Command completed with return code: {process.returncode}")
    time.sleep(5)  # Wait 5 seconds between commands

# Run experiments for each attacker
for attacker in attackers:
    print(f"\nStarting experiments for attacker: {attacker}")
    
    # Run with SpeculativeSafeDecoding defense
    log_file = f"{attacker}_llama2-13b-chat_speculativeSafeDecoding_alpha_0dot3_0dot8_new.log"
    cmd = base_cmd.format(attacker)
    run_command(cmd, log_file)

print("\nAll GPU 3 experiments completed!") 