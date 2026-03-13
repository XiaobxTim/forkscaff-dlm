import json
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])

def load_json(path):
    with open(path) as f:
        return json.load(f)

def find_result(task_dir):
    if not task_dir.exists():
        return None

    # 先找当前目录下的 json
    files = list(task_dir.glob("*.json"))
    if files:
        return load_json(files[0])

    # 当前层没有的话，递归往下找
    files = list(task_dir.rglob("*.json"))
    if files:
        return load_json(files[0])

    return None

def extract_gsm8k(res):
    try:
        return res["results"]["gsm8k_cot"]["exact_match,flexible-extract"]
    except:
        return None

def extract_math(res):
    try:
        return res["results"]["hendrycks_math500"]["exact_match,none"]
    except:
        return None

def extract_code(res, task):
    try:
        if task == 'humaneval':
            return res["results"][task]["pass@1,create_test"]
        else:
            return res["results"][task]["pass_at_1,none"]
    except:
        return None

gsm8k = extract_gsm8k(find_result(result_dir/"gsm8k"))
math = extract_math(find_result(result_dir/"math500"))
humaneval = extract_code(find_result(result_dir/"humaneval"), "humaneval")
mbpp = extract_code(find_result(result_dir/"mbpp"), "mbpp")

print("\n===== FINAL RESULTS =====\n")
print(f"GSM8K      : {gsm8k*100:.2f}" if gsm8k else "GSM8K      : N/A")
print(f"MATH500    : {math*100:.2f}" if math else "MATH500    : N/A")
print(f"HumanEval  : {humaneval*100:.2f}" if humaneval else "HumanEval  : N/A")
print(f"MBPP       : {mbpp*100:.2f}" if mbpp else "MBPP       : N/A")

print("\nPaper Table Format\n")

print("| Method | GSM8K | MATH-500 | HumanEval | MBPP |")
print("|------|------|------|------|------|")

print(f"| Result | {gsm8k*100:.1f} | {math*100:.1f} | {humaneval*100:.1f} | {mbpp*100:.1f} |")