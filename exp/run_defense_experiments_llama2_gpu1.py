import subprocess
import time
import os

# List of attackers
attackers = ['AdvBench', 'AutoDAN', 'DeepInception', 'HEx-PHI']

# Base command template for GPU 1
base_cmd = "CUDA_VISIBLE_DEVICES=1 python defense.py --model_name deep_align_llama2 --attacker {} --defender {} {} --GPT_API sk-7a2a9ad8ad06460ea13b61f31d73101f"

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
    
    # Run with SafeDecoding defense
    # cmd = base_cmd.format(attacker, 'SafeDecoding', '')
    # log_file = f"{attacker}_llama2_SafeDecoding.log"
    # run_command(cmd, log_file)

    # Run with Paraphrase defense
    # cmd = base_cmd.format(attacker, 'Paraphrase', '')
    # log_file = f"{attacker}_llama2_paraphrase.log"
    # run_command(cmd, log_file)
    
    # Run with ICD defense
    # cmd = base_cmd.format(attacker, 'ICD', '')
    # log_file = f"{attacker}_llama2_ICD.log"
    # run_command(cmd, log_file)
    
    # Run with Self-Exam defense
    # cmd = base_cmd.format(attacker, 'Self-Exam', '')
    # log_file = f"{attacker}_llama2_Self-Exam.log"
    # run_command(cmd, log_file)

    # Run with no defense on deep_align_llama2
    cmd = base_cmd.format(attacker, 'SpeculativeSafeDecoding', '--defense_off')
    log_file = f"{attacker}_deep_align_llama2_nodefense.log"
    run_command(cmd, log_file)

    # Run with no defense
    # cmd = base_cmd.format(attacker, 'SpeculativeSafeDecoding', '--defense_off')
    # log_file = f"{attacker}_llama2_nodefense.log"
    # run_command(cmd, log_file)

print("\nAll GPU 1 experiments completed!") 