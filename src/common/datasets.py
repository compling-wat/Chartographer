import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
HF_DATASETS_CACHE = REPO_ROOT / ".cache" / "huggingface" / "datasets"

DATASET_REGISTRY: Dict[str, Dict[str, object]] = {}
DATASET_ALIASES: Dict[str, str] = {}


def _load_user_dataset_config() -> tuple[Dict[str, Dict[str, object]], Dict[str, str]]:
    config_path = os.getenv("CHARTOGRAPHER_DATASETS_FILE")
    if not config_path:
        return {}, {}

    path = Path(config_path).expanduser()
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")

    raw_datasets = payload.get("datasets", payload)
    if not isinstance(raw_datasets, dict):
        raise ValueError(f"{path} field 'datasets' must be a JSON object")

    datasets: Dict[str, Dict[str, object]] = {}
    for name, cfg in raw_datasets.items():
        if name == "aliases":
            continue
        if not isinstance(cfg, dict):
            raise ValueError(f"Dataset config for {name!r} must be a JSON object")
        key = str(name).lower()
        merged = dict(cfg)
        if "hf_id" in merged and isinstance(merged["hf_id"], str):
            merged["hf_id"] = merged["hf_id"].format(repo_root=REPO_ROOT)
        if "local_file_template" in merged and isinstance(merged["local_file_template"], str):
            merged["local_file_template"] = merged["local_file_template"].format(
                repo_root=REPO_ROOT,
                split="{split}",
            )
        merged.setdefault("local_dir", key.split("/")[-1])
        merged.setdefault("question_col", "question")
        merged.setdefault("image_col", "image")
        merged.setdefault("answer_col", "answer")
        datasets[key] = merged

    raw_aliases = payload.get("aliases", {})
    if not isinstance(raw_aliases, dict):
        raise ValueError(f"{path} field 'aliases' must be a JSON object")
    aliases = {str(k).lower(): str(v).lower() for k, v in raw_aliases.items()}
    for name in datasets:
        aliases.setdefault(name, name)
    return datasets, aliases


DATASET_REGISTRY, DATASET_ALIASES = _load_user_dataset_config()


def column_candidates(
    cfg: Dict[str, object],
    primary_key: str,
    fallback_key: str,
    defaults: Sequence[str] = (),
) -> list[str]:
    candidates: list[str] = []
    primary = cfg.get(primary_key)
    if primary:
        candidates.append(str(primary))

    fallback = cfg.get(fallback_key, [])
    if isinstance(fallback, list):
        candidates.extend(str(col) for col in fallback if col)

    candidates.extend(defaults)
    return list(dict.fromkeys(candidates))


def resolve_column(
    cfg: Dict[str, object],
    dataset_obj: Any,
    primary_key: str,
    fallback_key: str,
    defaults: Sequence[str] = (),
) -> str:
    candidates = column_candidates(cfg, primary_key, fallback_key, defaults)
    columns = getattr(dataset_obj, "column_names", None)
    if columns is not None:
        column_set = set(columns)
        for col in candidates:
            if col in column_set:
                return col
    else:
        sample = dataset_obj[0] if len(dataset_obj) else {}
        for col in candidates:
            if col in sample:
                return col

    if candidates:
        return candidates[0]
    raise KeyError(f"No configured column candidates for '{primary_key}'")


def normalize_dataset_name(dataset_name: str) -> str:
    return DATASET_ALIASES.get(dataset_name.lower(), dataset_name)


def get_dataset_config(dataset_name: str) -> Dict[str, object]:
    normalized = normalize_dataset_name(dataset_name)
    cfg = DATASET_REGISTRY.get(normalized)
    if cfg is not None:
        return cfg

    return {
        "hf_id": dataset_name,
        "local_dir": dataset_name.split("/")[-1].lower(),
        "question_col": "question",
        "image_col": "image",
        "answer_col": "answer",
    }


def load_examples(dataset_name: str, split: str):
    HF_DATASETS_CACHE.mkdir(parents=True, exist_ok=True)
    cfg = get_dataset_config(dataset_name)
    local_file_template = cfg.get("local_file_template")
    if local_file_template is not None:
        local_file = Path(local_file_template.format(split=split))
        if not local_file.exists():
            raise FileNotFoundError(f"Missing local dataset file for {dataset_name}: {local_file}")
        ds = load_dataset(
            "json",
            data_files={split: str(local_file)},
            cache_dir=str(HF_DATASETS_CACHE),
        )[split]
    else:
        ds = load_dataset(str(cfg["hf_id"]), cache_dir=str(HF_DATASETS_CACHE))[split]

    row_filter_equals = cfg.get("row_filter_equals")
    if isinstance(row_filter_equals, dict) and row_filter_equals:
        def keep_row(example):
            for key, expected in row_filter_equals.items():
                if example.get(key) != expected:
                    return False
            return True

        ds = ds.filter(keep_row)

    return ds


def resolve_repo_local_path(path_str: str) -> Optional[Path]:
    path = Path(path_str)
    if path.exists():
        return path

    parts = path.parts
    for anchor in ("results", "data"):
        if anchor in parts:
            idx = parts.index(anchor)
            candidate = REPO_ROOT.joinpath(*parts[idx:])
            if candidate.exists():
                return candidate

    return None


def example_image_basename(
    example: Dict[str, Any],
    cfg: Dict[str, object],
    extra_candidates: Sequence[str] = ("image_path", "path", "filename", "file", "img"),
) -> str:
    configured = [
        cfg.get("figure_path_col"),
        cfg.get("original_figure_path_col"),
        cfg.get("image_col"),
    ]
    candidates = [
        c
        for c in [*configured, "image", "figure_path", "original_figure_path", *extra_candidates]
        if isinstance(c, str) and c
    ]

    for col in candidates:
        v = example.get(col)
        if isinstance(v, dict):
            v = v.get("path") or v.get("filename") or ""
        elif not isinstance(v, str):
            fn = getattr(v, "filename", None)
            if isinstance(fn, str) and fn:
                v = fn
        if isinstance(v, str) and v:
            return os.path.basename(v)
    return ""


def load_split_basenames(dataset_name: str, split: str) -> list[str]:
    cfg = get_dataset_config(dataset_name)
    ds = load_examples(dataset_name, split)
    basenames: list[str] = []
    for ex in ds:
        bn = example_image_basename(ex, cfg)
        if bn:
            basenames.append(bn)
    return basenames
