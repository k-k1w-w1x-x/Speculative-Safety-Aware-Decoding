import subprocess
import time
import os

# List of attackers
attackers = ['HEx-PHI','AdvBench'] 

# Base command template
base_cmd = "CUDA_VISIBLE_DEVICES=2 python defense.py --model_name llama2-13b-chat --attacker {} --defender {} {} --GPT_API sk-7a2a9ad8ad06460ea13b61f31d73101f"

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
    cmd = base_cmd.format(attacker, 'SafeDecoding', '')
    log_file = f"{attacker}_llama2-13b-chat_SafeDecoding.log"
    run_command(cmd, log_file)
    # Run with Paraphrase defense
    # cmd = base_cmd.format(attacker, 'Paraphrase', '')
    # log_file = f"{attacker}_llama2-13b-chat_paraphrase.log"
    # run_command(cmd, log_file)
    
    # Run with ICD defense
    # cmd = base_cmd.format(attacker, 'ICD', '')
    # log_file = f"{attacker}_llama2-13b-chat_ICD.log"
    # run_command(cmd, log_file)
    
    # # Run with Self-Exam defense
    # cmd = base_cmd.format(attacker, 'Self-Exam', '')
    # log_file = f"{attacker}_llama2-13b-chat_Self-Exam.log"
    # run_command(cmd, log_file)

    # # Run with no defense
    # cmd = base_cmd.format(attacker, 'SpeculativeSafeDecoding', '--defense_off')
    # log_file = f"{attacker}_llama2-13b-chat_nodefense.log"
    # run_command(cmd, log_file)

print("\nAll experiments completed!") 