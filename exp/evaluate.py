import json
import subprocess
import argparse
import os
import time

def get_args():
    parser = argparse.ArgumentParser(description="Evaluation manager.")
    # Experiment Settings
    parser.add_argument("--model_name", type=str, default="vicuna")
    parser.add_argument("--attacker", type=str, default="GCG")
    parser.add_argument("--defender", type=str, default='SafeDecoding')
    parser.add_argument("--defense_off", action="store_false", dest="is_defense", help="Disable defense")
    parser.set_defaults(is_defense=True)
    parser.add_argument("--eval_mode", action="store_true", dest="eval_mode", help="Enable evaluation mode (Default: True)")
    parser.add_argument("--disable_GPT_judge", action="store_true", dest="disable_GPT_judge", help="Disable GPT judge")
    parser.add_argument("--multi_processing", type=int, default=20) # wxk 单线程
    parser.add_argument("--GPT_API", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="../exp_outputs", help="Directory to save evaluation results")
    parser.add_argument("--time_str", type=str, default=None, help="Time string")
    parser.add_argument("--batch_num", type=int, default=None, help="Batch number")
    return parser.parse_args()

def run_evaluation(args, output_json, folder_path, save_name):
    # Evaluation
    print(f"args.eval_mode: {args.eval_mode}")
    if args.eval_mode:
        if args.attacker != "Just-Eval":
            results = output_json['data']
            goals = [result['goal'] for result in results]
            instructions = [result['instruction'] for result in results]
            responses = [result['output'] for result in results]
            if not args.disable_GPT_judge:
                from safe_eval import GPTJudge
                gpt_judge = GPTJudge('gpt',mp=args.multi_processing, api=args.GPT_API)  # using qwen-max
                goals_responses_pairs = []
                for i in range(len(instructions)):
                    goals_responses_pairs.append((goals[i], responses[i]))

                scores, reasons = gpt_judge.evaluate(goals_responses_pairs)
                scores_gpt = gpt_judge.score_parse(scores)
            else:
                scores_gpt = [None] * len(instructions)
                reasons = [None] * len(instructions)

            from safe_eval import DictJudge
            dict_judge = DictJudge()
            scores_dict = dict_judge.eval_batch(responses)

            safe_eval_results = []

            for i in range(len(instructions)):
                output_formatted = {
                    "id": i,
                    "goal": goals[i],
                    "instruction": instructions[i],
                    "output": responses[i],
                    "dic_score": scores_dict[i],
                    "GPT_score": scores_gpt[i] if not args.disable_GPT_judge else None,
                    "GPT_reason": reasons[i] if not args.disable_GPT_judge else None
                }
                safe_eval_results.append(output_formatted)

            with open(folder_path+'/'+save_name+'_safe_eval.json', 'w') as f:
                json.dump(safe_eval_results, f, indent=4)

            gpt_score_sum = 0
            defense_success_count = 0
            for i in range(len(safe_eval_results)):
                if safe_eval_results[i]['GPT_score'] is not None:
                    gpt_score_sum += safe_eval_results[i]['GPT_score']
                if safe_eval_results[i]['dic_score'] == 1:
                    defense_success_count += 1

            print(f'Harmful Score: {gpt_score_sum / len(safe_eval_results)}')
            print(f'ASR: {100-(defense_success_count / len(safe_eval_results))*100:.2f}%')

        else:
            # Just-Eval run
            just_eval_run_command = f'just_eval --mode "score_multi" --model "qwen-max" --first_file "{folder_path}/{save_name}.json" --output_file "{folder_path}/{save_name}_safe_eval.json" --api_key "{args.GPT_API}"'
            print(just_eval_run_command)
            
            just_eval_run_output = subprocess.check_output(just_eval_run_command, shell=True, text=True)
            # print(f"Just-Eval output: {just_eval_run_output}")

            # Just-Eval stats
            just_eval_stats_command = f'''
            just_eval --report_only --mode "score_safety" \
                    --output_file "{folder_path+'/'+save_name+'_safe_eval.json'}"
            '''
            print(just_eval_stats_command)
            # exit()
            just_eval_stats_output = subprocess.check_output(just_eval_stats_command, shell=True, text=True)
            print(f"Just-Eval stats output: {just_eval_stats_output}")

def main():
    args = get_args()
    current_time = time.localtime()
    time_str = args.time_str
    folder_path = os.path.join(args.output_dir, f'{args.defender if args.is_defense else "nodefense"}_{args.model_name}_{args.attacker}_{args.batch_num}_{time_str}')
    print(f"folder_path: {folder_path}")
    if not os.path.exists(folder_path):
        print("no folder found!")
        exit()
    #     os.makedirs(folder_path)
    save_name = f'{args.defender if args.is_defense else "nodefense"}_{args.model_name}_{args.attacker}_{args.batch_num}_{time_str}'
    input_json = f'{args.defender if args.is_defense else "nodefense"}_{args.model_name}_{args.attacker}_{args.batch_num}_{time_str}.json'
    with open(folder_path+'/'+input_json, 'r') as f:
        output_json = json.load(f)
    
    # print(f"save_name: {save_name}", f"folder_path: {folder_path}", f"output_json: {input_json}")
    # exit()
    run_evaluation(args, output_json, folder_path, save_name)

if __name__ == '__main__':
    main() 