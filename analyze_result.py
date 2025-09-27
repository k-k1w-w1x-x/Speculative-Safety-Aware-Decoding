# python analyze_result.py
# this script is used to analyze the result in ./exp_outputs
import json
import os
from pathlib import Path

def analyze_scores(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scores = []
    dict_scores = []
    for item in data:
        if 'GPT_score' in item and item['GPT_score'] is not None:
            scores.append(item['GPT_score'])
            if item['GPT_score'] == None:
                print(f"Warning: GPT_score is None in {json_file.name}")
        if 'dic_score' in item and item['dic_score'] is not None:
            if item['dic_score'] == True:
                dict_scores.append(1)
            else:
                dict_scores.append(0)
    if not scores:
        print(f"Warning: No valid scores found in {json_file.name}")
        return None, None, None
    
    avg_score = sum(scores) / len(scores)
    five_score_rate = scores.count(5) / len(scores) * 100
    false_score_rate = dict_scores.count(0) / len(dict_scores) * 100
    return avg_score, five_score_rate, false_score_rate

def analyze_just_eval_scores(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, dict):
        print(f"Warning: Invalid format in {json_file.name}")
        return None
    
    scores = {
        'helpfulness': data.get('helpfulness_mean'),
        'clarity': data.get('clarity_mean'),
        'factuality': data.get('factuality_mean'),
        'depth': data.get('depth_mean'),
        'engagement': data.get('engagement_mean')
    }
    
    return scores

def main():
    # 获取当前目录下所有的JSON文件
    current_dir = Path(__file__).parent
    json_files = list(current_dir.glob('**/*.json'))
    
    if not json_files:
        print("No JSON files found!")
        return
    
    print("Analysis Results:")
    print("-" * 50)
    
    for json_file in json_files:
        # if not "AdvBench" in json_file.name:
        #     continue
        # if not "llama3" in json_file.name:
        #     continue
        # if not "HEx-PHI_330" in json_file.name:
        #     continue
        # if not "HEx-PHI_330" in json_file.name:
        #     continue
        # if not "HEx-PHI_330" in json_file.name:
        #     continue
        # if "down" in json_file.name:
        #     continue
        # if "ablation" in json_file.name:
        #     continue
        # if not "0dot6down_0dot8" in json_file.name:
        #     continue
        if 'eval_res.json' in json_file.name and "Just-Eval" in json_file.name:
            scores = analyze_just_eval_scores(json_file)
            if scores:
                print(f"File: {json_file.name}")
                print(f"Helpfulness Mean: {scores['helpfulness']:.3f}")
                print(f"Clarity Mean: {scores['clarity']:.3f}")
                print(f"Factuality Mean: {scores['factuality']:.3f}")
                print(f"Depth Mean: {scores['depth']:.3f}")
                print(f"Engagement Mean: {scores['engagement']:.3f}")
                print("-" * 50)
        elif 'safe_eval.json' in json_file.name and 'Just-Eval' not in json_file.name:
            avg_score, five_score_rate, false_score_rate = analyze_scores(json_file)
            if avg_score is not None and "" in json_file.name:
                print(f"File: {json_file.name}")
                print(f"Average Score: {avg_score:.2f}")
                print(f"5-Score Rate: {five_score_rate:.2f}%")
                print("-" * 50)

if __name__ == "__main__":
    main()
