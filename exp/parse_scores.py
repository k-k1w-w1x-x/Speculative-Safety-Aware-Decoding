import sys
import re
import numpy as np

def parse_scores_from_log(log_path):
    # 匹配前面有单引号或双引号的 #thescore: X
    pattern = re.compile(r"[\']#thescore:\s*(\d+)")
    scores = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                scores.append(int(match.group(1)))
    return scores

def print_stats(scores):
    if not scores:
        print("No scores found.")
        return
    arr = np.array(scores)
    mean = arr.mean()
    quantiles = np.percentile(arr, [0, 25, 50, 75, 100])
    count_5 = np.sum(arr == 5)
    ratio_5 = count_5 / len(arr)
    print(f"Count: {len(scores)}")
    print(f"Mean: {mean:.3f}")
    print(f"Quantiles (0%, 25%, 50%, 75%, 100%): {quantiles}")
    print(f"Score == 5: {count_5} ({ratio_5:.2%})")

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <log_file>")
        sys.exit(1)
    log_path = sys.argv[1]
    scores = parse_scores_from_log(log_path)
    print_stats(scores)

if __name__ == "__main__":
    main() 