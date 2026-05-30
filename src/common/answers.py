import re
from typing import Any


def extract_tagged_answer(text: Any) -> str:
    if not isinstance(text, str):
        return str(text)
    match = re.search(r"<answer>(.*?)</answer>", text + "</answer>", flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_prediction_answer(dataset_name: str, pred_item: Any) -> str:
    if isinstance(pred_item, dict):
        if "prediction" in pred_item:
            pred_item = pred_item["prediction"]
        elif "prediction_raw" in pred_item:
            pred_item = pred_item["prediction_raw"]

    if isinstance(pred_item, list):
        return str(pred_item[-1]) if pred_item else ""
    return extract_tagged_answer(pred_item)
