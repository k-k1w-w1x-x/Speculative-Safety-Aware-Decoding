import subprocess
import time
import os

# List of attackers
attackers = ['HEx-PHI']

# Base command template
base_cmd = "CUDA_VISIBLE_DEVICES=0 python defense.py --model_name {} --attacker {} --defender {} {} --GPT_API sk-7a2a9ad8ad06460ea13b61f31d73101f"

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
    # cmd = base_cmd.format('vicuna', attacker, 'SafeDecoding', '')
    # log_file = f"{attacker}_vicuna_SafeDecoding.log"
    # run_command(cmd, log_file)
    
    #     # Run with Paraphrase defense
    # cmd = base_cmd.format('vicuna', attacker, 'Paraphrase', '')
    # log_file = f"{attacker}_vicuna_paraphrase.log"
    # run_command(cmd, log_file)
    
    # # Run with ICD defense
    # cmd = base_cmd.format('vicuna', attacker, 'ICD', '')
    # log_file = f"{attacker}_vicuna_ICD.log"
    # run_command(cmd, log_file)
    
    # # Run with Self-Exam defense
    # cmd = base_cmd.format('vicuna', attacker, 'Self-Exam', '')
    # log_file = f"{attacker}_vicuna_Self-Exam.log"
    # run_command(cmd, log_file)
    
    # Run with no defense on deep_align_vicuna
    cmd = base_cmd.format('deep_align_vicuna', attacker, 'SpeculativeSafeDecoding', '--defense_off')
    log_file = f"{attacker}_deep_align_vicuna_nodefense.log"
    run_command(cmd, log_file)
    
    # Run with no defense on vicuna
    # cmd = base_cmd.format('vicuna', attacker, 'SpeculativeSafeDecoding', '--defense_off')
    # log_file = f"{attacker}_vicuna_nodefense.log"
    # run_command(cmd, log_file)

print("\nAll experiments completed!") 