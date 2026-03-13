import json
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
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


def get_first_metric(task_res, candidate_keys):
    if task_res is None:
        return None
    for k in candidate_keys:
        if k in task_res:
            return task_res[k]
    return None


def extract_gsm8k(res):
    if res is None:
        return None, None
    task_res = res.get("results", {}).get("gsm8k_cot", {})
    pass1 = get_first_metric(task_res, [
        "exact_match,flexible-extract",
        "exact_match,strict-match",
        "exact_match,none",
    ])
    pass5 = get_first_metric(task_res, [
        "pass@5,flexible-extract",
        "pass@5,none",
        "pass_at_5,flexible-extract",
        "pass_at_5,none",
    ])
    return pass1, pass5


def extract_math(res):
    if res is None:
        return None, None
    task_res = res.get("results", {}).get("hendrycks_math500", {})
    pass1 = get_first_metric(task_res, [
        "exact_match,none",
        "exact_match",
    ])
    pass5 = get_first_metric(task_res, [
        "pass@5,none",
        "pass@5",
        "pass_at_5,none",
        "pass_at_5",
    ])
    return pass1, pass5


def extract_code(res, task):
    if res is None:
        return None, None
    task_res = res.get("results", {}).get(task, {})

    if task == "humaneval":
        pass1 = get_first_metric(task_res, [
            "pass@1,create_test",
            "pass@1",
            "pass_at_1,create_test",
            "pass_at_1",
        ])
        pass5 = get_first_metric(task_res, [
            "pass@5,create_test",
            "pass@5",
            "pass_at_5,create_test",
            "pass_at_5",
        ])
    else:  # mbpp
        pass1 = get_first_metric(task_res, [
            "pass_at_1,none",
            "pass@1,none",
            "pass_at_1",
            "pass@1",
        ])
        pass5 = get_first_metric(task_res, [
            "pass_at_5,none",
            "pass@5,none",
            "pass_at_5",
            "pass@5",
        ])

    return pass1, pass5


def fmt_percent(x):
    return f"{x * 100:.2f}" if x is not None else "N/A"


def fmt_percent_short(x):
    return f"{x * 100:.1f}" if x is not None else "N/A"


gsm8k_p1, gsm8k_p5 = extract_gsm8k(find_result(result_dir / "gsm8k"))
math_p1, math_p5 = extract_math(find_result(result_dir / "math500"))
humaneval_p1, humaneval_p5 = extract_code(find_result(result_dir / "humaneval"), "humaneval")
mbpp_p1, mbpp_p5 = extract_code(find_result(result_dir / "mbpp"), "mbpp")

print("\n===== FINAL RESULTS =====\n")
print(f"GSM8K pass@1      : {fmt_percent(gsm8k_p1)}")
print(f"GSM8K pass@5      : {fmt_percent(gsm8k_p5)}")
print(f"MATH500 pass@1    : {fmt_percent(math_p1)}")
print(f"MATH500 pass@5    : {fmt_percent(math_p5)}")
print(f"HumanEval pass@1  : {fmt_percent(humaneval_p1)}")
print(f"HumanEval pass@5  : {fmt_percent(humaneval_p5)}")
print(f"MBPP pass@1       : {fmt_percent(mbpp_p1)}")
print(f"MBPP pass@5       : {fmt_percent(mbpp_p5)}")

print("\nPaper Table Format\n")
print("| Method | GSM8K@1 | GSM8K@5 | MATH-500@1 | MATH-500@5 | HumanEval@1 | HumanEval@5 | MBPP@1 | MBPP@5 |")
print("|------|------:|------:|------:|------:|------:|------:|------:|------:|")
print(
    f"| Result | "
    f"{fmt_percent_short(gsm8k_p1)} | "
    f"{fmt_percent_short(gsm8k_p5)} | "
    f"{fmt_percent_short(math_p1)} | "
    f"{fmt_percent_short(math_p5)} | "
    f"{fmt_percent_short(humaneval_p1)} | "
    f"{fmt_percent_short(humaneval_p5)} | "
    f"{fmt_percent_short(mbpp_p1)} | "
    f"{fmt_percent_short(mbpp_p5)} |"
)