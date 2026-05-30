import argparse
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from common.artifacts import RECONSTRUCTION_VARIANT, chart_data_dir, seed_variant, split_artifact_dir
from common.datasets import get_dataset_config, load_examples, normalize_dataset_name, resolve_column

REPO_ROOT = Path(__file__).resolve().parents[3]
HF_CACHE_DIR = REPO_ROOT / ".cache" / "huggingface"
HF_DATASETS_CACHE_DIR = HF_CACHE_DIR / "datasets"


def _configure_hf_cache() -> None:
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_DATASETS_CACHE_DIR))
    HF_DATASETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    module_dir = str(path.parent.resolve())
    added = False
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
        added = True
    try:
        spec.loader.exec_module(module)
    finally:
        if added and sys.path and sys.path[0] == module_dir:
            sys.path.pop(0)
    return module


def _image_basename(example: Dict[str, Any], ds_cfg: Dict[str, str]) -> str:
    candidates = [
        ds_cfg.get("figure_path_col", ""),
        ds_cfg.get("original_figure_path_col", ""),
        ds_cfg.get("image_col", "image"),
        "image",
        "image_path",
        "path",
        "filename",
        "file",
        "img",
    ]
    for k in candidates:
        if not k:
            continue
        vv = example.get(k)
        if isinstance(vv, dict):
            vv = vv.get("path") or vv.get("filename") or ""
        elif not isinstance(vv, str):
            fn = getattr(vv, "filename", None)
            if isinstance(fn, str) and fn:
                vv = fn
        if isinstance(vv, str) and vv:
            return Path(vv).name
    return ""


def _split_dir(dataset: str, split: str) -> Path:
    return split_artifact_dir(str(get_dataset_config(dataset)["local_dir"]), split)


def _answer_dir(dataset: str, split: str) -> Path:
    return _split_dir(dataset, split) / "generate_answers"


def _adapter_dir(dataset: str, split: str) -> Path:
    return _split_dir(dataset, split) / "question_adapters"


def _find_adapter_file(dataset: str, split: str, qa_key: str) -> Optional[Path]:
    p = _adapter_dir(dataset, split) / f"{qa_key}.py"
    return p if p.exists() else None


def _data_variant(seed: Optional[int]) -> str:
    return RECONSTRUCTION_VARIANT if seed is None else seed_variant(seed)


def _data_dir(dataset: str, split: str, seed: Optional[int]) -> Path:
    return chart_data_dir(_split_dir(dataset, split), _data_variant(seed))


def _resolve_data_json(data_dir: Path, stem: str, seed: Optional[int]) -> Optional[Path]:
    path = data_dir / f"{stem}.json"
    return path if path.exists() else None


def _chart_stems_in_data_dir(data_dir: Path, seed: Optional[int]) -> set[str]:
    stems: set[str] = set()
    for p in data_dir.iterdir():
        if p.is_file() and p.suffix == ".json":
            stems.add(p.stem)
    return stems


def _output_path(dataset: str, split: str, seed: Optional[int]) -> Path:
    filename = "qa_reconstruction.json" if seed is None else f"qa_seed_{seed}.json"
    return _split_dir(dataset, split) / filename


def _find_answer_file(dataset: str, split: str, name: str) -> Optional[Path]:
    if not name.endswith(".py"):
        name += ".py"
    path = _answer_dir(dataset, split) / name
    return path if path.exists() else None


def _qa_key(stem: str, question: str) -> str:
    digest = hashlib.sha1(question.strip().encode("utf-8")).hexdigest()[:12]
    return f"{stem}__{digest}"


def _question_key(question: str) -> str:
    return hashlib.sha1(question.strip().encode("utf-8")).hexdigest()


def _question_uid(stem: str, question: str) -> str:
    # Stable per-question identifier (independent of seed-specific adapted wording).
    digest = hashlib.sha1(question.strip().encode("utf-8")).hexdigest()
    return f"{stem}__{digest}"


def _example_stem(example: Dict[str, Any], ds_cfg: Dict[str, str]) -> str:
    stem = str(example.get("_image_stem") or "").strip()
    if stem:
        return stem
    question_id = str(example.get("question_id") or "").strip()
    if "__" in question_id:
        stem = question_id.split("__", 1)[0].strip()
        if stem:
            return stem
    return Path(_image_basename(example, ds_cfg)).stem


def _qa_key_from_question_id(stem: str, example: Dict[str, Any]) -> Optional[str]:
    question_id = str(example.get("question_id") or "").strip()
    if "__" not in question_id:
        return None
    question_stem, digest = question_id.split("__", 1)
    if question_stem != stem or not digest:
        return None
    return f"{stem}__{digest[:12]}"


def _list_answer_files(dataset: str, split: str) -> List[Path]:
    answer_dir = _answer_dir(dataset, split)
    if not answer_dir.exists():
        return []
    return [p for p in sorted(answer_dir.glob("*.py")) if p.name != "__init__.py"]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    Path(tmp).replace(path)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    fn = getattr(value, "filename", None)
    if isinstance(fn, str) and fn:
        return fn
    return str(value)


def _run_one(
    dataset_key: str,
    split: str,
    example: Dict[str, Any],
    question_col: str,
    answer_col: str,
    ds_cfg: Dict[str, str],
    data_dir: Path,
    seed: Optional[int],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    stem = _example_stem(example, ds_cfg)
    if not stem:
        return None, "missing image stem"

    question_original = str(example.get(question_col, "")).strip()
    qa_key = _qa_key(stem, question_original)
    question_effective = question_original

    answer_file = _find_answer_file(dataset_key, split, qa_key)
    if answer_file is None:
        existing_qa_key = _qa_key_from_question_id(stem, example)
        if existing_qa_key is not None:
            answer_file = _find_answer_file(dataset_key, split, existing_qa_key)
            qa_key = existing_qa_key
    if answer_file is None:
        return None, f"missing answer module: {_answer_dir(dataset_key, split) / (qa_key + '.py')}"

    data_path = _resolve_data_json(data_dir, stem, seed)
    if data_path is None:
        expected = data_dir / f"{stem}.json"
        return None, f"missing data json: {expected}"

    try:
        data = _load_json(data_path)
    except Exception as e:
        return None, f"could not load data json: {type(e).__name__}: {e}"

    # Always apply executable adaptor module when available.
    adapter_file = _find_adapter_file(dataset_key, split, qa_key)
    if adapter_file is not None:
        try:
            ad_mod = load_module(adapter_file)
            if hasattr(ad_mod, "adapt_question") and callable(ad_mod.adapt_question):
                q_eff = str(ad_mod.adapt_question(data)).strip()
                if q_eff:
                    question_effective = q_eff
        except Exception:
            pass

    try:
        mod = load_module(answer_file)
    except Exception as e:
        return None, f"could not import answer module: {type(e).__name__}: {e}"

    if not hasattr(mod, "generate_answer"):
        return None, f"{answer_file} is missing generate_answer()"

    try:
        answer = mod.generate_answer(data)
    except Exception as e:
        detail = "".join(traceback.format_exception_only(type(e), e)).strip()
        return None, f"generate_answer() crashed: {detail}"

    answer_text = str(answer).strip()
    if not answer_text:
        return None, "generate_answer() returned an empty answer"

    out = _json_safe(dict(example))
    out[question_col] = question_effective
    out[answer_col] = answer_text
    out["question_id"] = str(example.get("question_id") or _question_uid(stem, question_original))
    out["_image_stem"] = stem
    return out, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate QA records from generate_answer modules.")
    parser.add_argument("--dataset", required=True, help="Dataset key from config or HF dataset id")
    parser.add_argument("--split", required=True, help="Split name (e.g. dev)")
    parser.add_argument("--chart", default=None, help="Specific chart stem to run (default: all)")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Use chart_data/seed_N as the input data directory. If omitted, use chart_data/reconstruction.",
    )
    args = parser.parse_args()

    dataset_key = normalize_dataset_name(args.dataset)
    ds_cfg = get_dataset_config(dataset_key)
    _configure_hf_cache()

    def load_local_qa_examples(path: Path) -> Tuple[List[Dict[str, Any]], set[str]]:
        loaded = _load_json(path)
        if not isinstance(loaded, list):
            raise ValueError(f"{path} must contain a JSON list")
        local_examples = [ex for ex in loaded if isinstance(ex, dict)]
        for ex in local_examples:
            stem = _example_stem(ex, ds_cfg)
            if stem:
                ex["_image_stem"] = stem
                ex["image"] = f"{stem}.png"
        local_columns: set[str] = set()
        for ex in local_examples:
            local_columns.update(ex.keys())
        return local_examples, local_columns

    ds = load_examples(dataset_key, args.split)
    examples = list(ds)
    question_col = resolve_column(ds_cfg, ds, "question_col", "question_cols")
    answer_col = resolve_column(ds_cfg, ds, "answer_col", "answer_cols")

    answer_dir = _answer_dir(dataset_key, args.split)
    if not answer_dir.exists():
        print(f"Answer module directory not found: {answer_dir}")
        return 1

    data_dir = _data_dir(dataset_key, args.split, args.seed)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return 1

    chart_stems = _chart_stems_in_data_dir(data_dir, args.seed)
    examples = [
        ex
        for ex in examples
        if isinstance(ex, dict) and (stem := _example_stem(ex, ds_cfg)) and stem in chart_stems
    ]

    if args.chart:
        target = args.chart[:-3] if args.chart.endswith(".py") else args.chart
        filtered = [
            ex
            for ex in examples
            if _example_stem(ex, ds_cfg) == target
        ]
        if not filtered:
            print("Chart not found in dataset split.\nAvailable generated answer modules:")
            for p in _list_answer_files(dataset_key, args.split):
                print("  -", p.stem)
            return 1
        examples = filtered

    outputs: List[Dict[str, Any]] = []
    failures: List[Tuple[str, str]] = []
    for ex in examples:
        stem = _example_stem(ex, ds_cfg) or "?"
        out, err = _run_one(
            dataset_key,
            args.split,
            ex,
            question_col,
            answer_col,
            ds_cfg,
            data_dir,
            args.seed,
        )
        if err:
            failures.append((stem, err))
            continue
        outputs.append(out)

    out_path = _output_path(dataset_key, args.split, args.seed)
    _write_json(out_path, outputs)

    if failures:
        for stem, err in failures:
            print(f"[ERROR] {stem}: {err}")

    print(f"Done. errors={len(failures)}")
    print(f"Output: {out_path}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
