import re
import math
from typing import List, Dict, Any


def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    return str(s).strip()


def _extract_last_number(text: str):
    """
    GSM8K 简化版答案抽取：
    取文本中最后一个数字，后续你可替换成你当前任务里已有的 flexible-extract 逻辑。
    """
    if text is None:
        return None
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not matches:
        return None
    return matches[-1]


def _extract_boxed(text: str):
    """
    MATH 简化版：
    优先抽 \\boxed{...}，否则退化为最后一个非空行 / 最后一个数字。
    """
    if text is None:
        return None

    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed[-1].strip()

    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if lines:
        return lines[-1]

    return None


def _math_equal(pred: str, gold: str) -> bool:
    """
    MATH500 简化等价判断。
    第一版先做规范化字符串比较；
    后面你可以替换成仓库已有的 math grader / sympy judge。
    """
    pred = _normalize_text(pred)
    gold = _normalize_text(gold)
    return pred == gold


def _pass_at_k_from_binary(binary_list: List[int], k: int) -> float:
    if not binary_list:
        return 0.0
    k = min(k, len(binary_list))
    return 1.0 if any(binary_list[:k]) else 0.0


def _avg_at_k_from_binary(binary_list: List[int], k: int) -> float:
    if not binary_list:
        return 0.0
    k = min(k, len(binary_list))
    return sum(binary_list[:k]) / float(k)


def pass_at_k_reasoning(items: List[Dict[str, Any]], k=[1, 5], task_type="gsm8k"):
    """
    给一题的多次生成结果做 pass@k。
    约定 items 是“同一道题”的 repeats 个输出。
    每个 item 至少应包含：
      - 'response' 或 'prediction'
      - 'gold'
    返回 dict: {"pass@1": ..., "pass@5": ...}
    """

    binary = []
    for item in items:
        pred_text = item.get("response", item.get("prediction", ""))
        gold = item.get("gold", "")

        if task_type == "gsm8k":
            pred = _extract_last_number(pred_text)
            target = _extract_last_number(str(gold))
            correct = int(pred is not None and target is not None and pred == target)

        elif task_type == "math500":
            pred = _extract_boxed(pred_text)
            target = _extract_boxed(str(gold)) or _normalize_text(gold)
            correct = int(pred is not None and target is not None and _math_equal(pred, target))

        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        binary.append(correct)

    out = {}
    for kk in k:
        out[f"pass@{kk}"] = _pass_at_k_from_binary(binary, kk)
        out[f"avg@{kk}"] = _avg_at_k_from_binary(binary, kk)
    return out


def pass_at_k_code(items: List[Dict[str, Any]], k=[1, 5]):
    """
    代码任务统一接口。
    假设每个 item 已经有 judge 后的字段 correct / passed。
    如果你 humaneval / mbpp 现有流程已经会产生 test pass/fail，可以直接接。
    """
    binary = []
    for item in items:
        correct = item.get("correct", item.get("passed", 0))
        binary.append(int(bool(correct)))

    out = {}
    for kk in k:
        out[f"pass@{kk}"] = _pass_at_k_from_binary(binary, kk)
        out[f"avg@{kk}"] = _avg_at_k_from_binary(binary, kk)
    return out