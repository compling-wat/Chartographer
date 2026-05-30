import json
from pathlib import Path
from typing import Any, List

REPO_ROOT = Path(__file__).resolve().parents[2]


def sanitize_model_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def get_output_path(
    dataset: str,
    split: str,
    model: str,
    limit: int = None,
    output_suffix: str = None,
) -> Path:
    model_name = sanitize_model_name(model)
    out_dir = REPO_ROOT / "results" / dataset / "predictions" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    if limit is not None:
        model_name += f"_limit{limit}"
    if output_suffix:
        model_name += f"__{output_suffix}"
    return out_dir / f"{model_name}.json"


def find_prediction_path(dataset: str, split: str, model: str, limit: int = None) -> Path:
    return get_output_path(dataset, split, model, limit=limit)


def find_eval_path(pred_path: Path, model: str) -> Path:
    return pred_path.with_name(f"{pred_path.stem}_eval.json")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def load_existing_predictions(out_path: Path, resume: bool) -> List[Any]:
    if not resume or not out_path.exists():
        return []
    data = load_json(out_path)
    if not isinstance(data, list):
        raise ValueError("Output JSON is not a list")
    return data


def load_eval_labels(eval_path: Path) -> List[int]:
    obj = load_json(eval_path)
    if not isinstance(obj, list):
        raise ValueError(f"Expected eval JSON list: {eval_path}")

    labels: List[int] = []
    for value in obj:
        labels.append(1 if int(value) == 1 else 0)
    return labels
