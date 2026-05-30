import argparse
import ast
import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from openai import OpenAI
from tqdm import tqdm
from common.artifacts import RECONSTRUCTION_VARIANT, chart_code_dir, chart_data_dir, split_artifact_dir
from common.datasets import example_image_basename, get_dataset_config, load_examples, normalize_dataset_name

REPO_ROOT = Path(__file__).resolve().parents[3]
HF_CACHE_DIR = REPO_ROOT / ".cache" / "huggingface"
HF_DATASETS_CACHE_DIR = HF_CACHE_DIR / "datasets"

PROMPT = r"""
TASK
Write one Python module that defines `adapt_question(data)` for the given question and chart context.

INPUTS
1) Original question
2) Reconstructed chart data JSON
3) Reconstructed chart code

GOAL
- Keep intent and answer format unchanged.
- Default to returning the original question verbatim.
- Rewrite only when needed to keep the question valid/answerable under regenerated same-schema data.
- If rewriting, make the smallest possible change.

OUTPUT RULES
- Return Python code only.
- No markdown, no prose, no backticks.

REQUIRED MODULE STRUCTURE
1) Imports
2) `adapt_question(data)`

RUNTIME CONTRACT
- At runtime, only `data` is provided.
- Chart code and original question are prompt-only context.
- Do not read files, call APIs, or depend on chart code at runtime.

RULES
- Keep original unchanged unless it is clearly invalid/unanswerable for regenerated data
  (e.g., missing entities/categories/axes, impossible condition, stale fixed values).
- If uncertain, do not rewrite.
- When rewriting, keep wording close and make the smallest validity-only edit.
- For any required replacement, use concrete entities/labels/values present in `data`.
- Do not broaden scope, change task type, add constraints, or introduce new entities.
- Do not expose schema/code internals in the question text.
- Preserve specific numbers/labels/entities unless required for validity.
- Return only the final question string (no explanation).

MODULE RULES
- Exactly one top-level function: `adapt_question(data)`.
- `adapt_question` must take only one parameter named `data`.
- Must return one non-empty, concise question string.
"""


def _configure_hf_cache() -> None:
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_DATASETS_CACHE_DIR))
    HF_DATASETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_dataset(dataset_key_or_hf: str) -> Tuple[str, Dict[str, str]]:
    normalized = normalize_dataset_name(dataset_key_or_hf)
    cfg = get_dataset_config(dataset_key_or_hf)
    return (
        str(cfg.get("local_dir") or normalized.split("/")[-1].lower()),
        {
            "question_col": str(cfg.get("question_col", "question")),
            "image_col": str(cfg.get("image_col", "image")),
            "figure_path_col": str(cfg.get("figure_path_col", "")),
            "original_figure_path_col": str(cfg.get("original_figure_path_col", "")),
        },
    )


def _response_text(resp) -> str:
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    out = getattr(resp, "output", None)
    if not out:
        return ""

    parts: List[str] = []
    for item in out:
        content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
        if not content:
            continue
        for block in content:
            btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
            btext = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else None)
            if btype == "output_text" and isinstance(btext, str):
                parts.append(btext)
    return "".join(parts).strip()


def _qa_key(stem: str, question: str) -> str:
    digest = hashlib.sha1(question.strip().encode("utf-8")).hexdigest()[:12]
    return f"{stem}__{digest}"


def _split_dir(repo_key: str, split: str) -> Path:
    return split_artifact_dir(repo_key, split)


def _data_json_dir(repo_key: str, split: str) -> Path:
    return chart_data_dir(_split_dir(repo_key, split), RECONSTRUCTION_VARIANT)


def _code_dir(repo_key: str, split: str) -> Path:
    return chart_code_dir(_split_dir(repo_key, split), RECONSTRUCTION_VARIANT)


def _adapter_dir(repo_key: str, split: str) -> Path:
    return _split_dir(repo_key, split) / "question_adapters"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + f".tmp.{threading.get_ident()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)


def _normalize_module(module_src: str) -> str:
    s = module_src.strip()
    tree = ast.parse(s)
    fns = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(fns) != 1:
        raise ValueError("Output must contain exactly one top-level function.")
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            continue
        raise ValueError("Top-level statements must be imports and adapt_question(data) only.")
    fn = fns[0]
    if fn.name != "adapt_question":
        raise ValueError("Top-level function must be named adapt_question.")
    if len(fn.args.args) != 1 or fn.args.args[0].arg != "data":
        raise ValueError("adapt_question must accept exactly one parameter named data.")
    return s + "\n"


def _run_adapter_module(code: str, data_obj: Dict[str, Any]) -> str:
    scope: Dict[str, Any] = {"__builtins__": __builtins__}
    exec(code, scope, scope)
    fn = scope.get("adapt_question")
    if not callable(fn):
        raise ValueError("Module must define callable adapt_question(data).")
    q = str(fn(data_obj)).strip()
    if not q:
        raise ValueError("adapt_question(data) returned empty question.")
    return q


def _find_example_by_chart(
    examples: List[Dict[str, Any]],
    ds_cfg: Dict[str, str],
    chart: str,
) -> Optional[Dict[str, Any]]:
    target = chart[:-3] if chart.endswith(".py") else chart
    for ex in examples:
        if Path(example_image_basename(ex, ds_cfg)).stem == target:
            return ex
    return None


def _process_one(
    client: OpenAI,
    model_name: str,
    temperature: float,
    max_attempts: int,
    example_idx: int,
    example: Dict[str, Any],
    question_col: str,
    ds_cfg: Dict[str, str],
    data_dir: Path,
    code_dir: Path,
    adapter_dir: Path,
    overwrite: bool,
) -> Tuple[int, str, Optional[str]]:
    try:
        image_name = example_image_basename(example, ds_cfg)
        stem = Path(image_name).stem
        if not stem:
            return example_idx, "", "missing image stem"

        question_original = str(example.get(question_col, "")).strip()
        qa_key = _qa_key(stem, question_original)
        adapter_path = adapter_dir / f"{qa_key}.py"

        if adapter_path.exists() and not overwrite:
            return example_idx, qa_key, None

        data_path = data_dir / f"{stem}.json"
        code_path = code_dir / f"{stem}.py"
        if not data_path.exists():
            return example_idx, qa_key, f"missing data json for {stem}: expected {data_path}"
        if not code_path.exists():
            return example_idx, qa_key, f"missing chart code for {stem}: expected {code_path}"

        data_obj = _load_json(data_path)
        data_src = json.dumps(data_obj, ensure_ascii=False, indent=2)
        code_src = _load_text(code_path)
        last_err = "unknown error"

        for _ in range(max_attempts):
            try:
                resp = client.responses.create(
                    model=model_name,
                    store=False,
                    temperature=temperature,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": PROMPT},
                                {
                                    "type": "input_text",
                                    "text": (
                                        f"\n\n# ORIGINAL_QUESTION\n{question_original}\n\n"
                                        f"# DATA_JSON\n{data_src}\n\n"
                                        f"# CHART_CODE\n{code_src}\n"
                                    ),
                                },
                            ],
                        }
                    ],
                    text={"format": {"type": "text"}},
                )
                raw_code = _response_text(resp)
                adapter_code = _normalize_module(raw_code)
                _run_adapter_module(adapter_code, data_obj)
                _write_text(adapter_path, adapter_code)
                return example_idx, qa_key, None
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"

        return example_idx, qa_key, f"{last_err} (after {max_attempts} attempts)"
    except Exception as e:
        return example_idx, stem if "stem" in locals() else "", f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate GPT question adaptor modules adapt_question(data) per QA key.")
    ap.add_argument("--model-name", required=True, help="OpenAI model name")
    ap.add_argument("--dataset", required=True, help="Dataset key from config or HF dataset id")
    ap.add_argument("--split", required=True, help="Dataset split")
    ap.add_argument("--chart", default=None, help="Specific chart stem or .py filename")
    ap.add_argument("--max-attempts", type=int, default=1, help="Total generation attempts per QA key")
    ap.add_argument("--max-workers", type=int, default=8, help="Worker threads")
    ap.add_argument("--max-items", type=int, default=None, help="Process at most first N items")
    ap.add_argument("--overwrite", action="store_true", help="Regenerate adapters even if output file already exists")
    ap.add_argument("--temperature", type=float, default=0.0, help="OpenAI temperature")
    args = ap.parse_args()

    _configure_hf_cache()
    dataset_key, ds_cfg = _resolve_dataset(args.dataset)
    ds = load_examples(args.dataset, args.split)
    question_col = ds_cfg["question_col"]
    image_col = ds_cfg["image_col"]
    for col in (question_col, image_col):
        if col not in ds.column_names:
            raise ValueError(f"Dataset split missing required column '{col}'. Columns: {ds.column_names}")

    examples = list(ds)
    if args.max_items is not None:
        if args.max_items < 0:
            raise ValueError("--max-items must be >= 0.")
        examples = examples[: args.max_items]

    data_dir = _data_json_dir(dataset_key, args.split).resolve()
    code_dir = _code_dir(dataset_key, args.split).resolve()
    adapter_dir = _adapter_dir(dataset_key, args.split).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Chart data dir not found: {data_dir}")
    if not code_dir.exists():
        raise FileNotFoundError(f"Chart code dir not found: {code_dir}")
    adapter_dir.mkdir(parents=True, exist_ok=True)

    chart_stems = {p.stem for p in data_dir.iterdir() if p.is_file()}
    total_examples = len(examples)
    examples = [
        ex
        for ex in examples
        if (stem := Path(example_image_basename(ex, ds_cfg)).stem) and stem in chart_stems
    ]
    skipped_no_chart = total_examples - len(examples)

    if args.chart:
        matched = _find_example_by_chart(examples, ds_cfg, args.chart)
        if matched is None:
            available = sorted(chart_stems)
            lines = "\n".join(f"  - {name}" for name in available[:80])
            if len(available) > 80:
                lines += f"\n  ... ({len(available) - 80} more)"
            raise FileNotFoundError(f"Chart not found: {args.chart}\nAvailable charts:\n{lines}")
        examples = [matched]
        skipped_no_chart = 0

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    errors = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [
            pool.submit(
                _process_one,
                client,
                args.model_name,
                args.temperature,
                args.max_attempts,
                i,
                ex,
                question_col,
                ds_cfg,
                data_dir,
                code_dir,
                adapter_dir,
                args.overwrite,
            )
            for i, ex in enumerate(examples)
        ]

        with tqdm(total=len(futures), desc=f"q-adapter {dataset_key} [{args.split}]", unit="item", dynamic_ncols=True) as bar:
            for fut in as_completed(futures):
                idx, qa_k, err = fut.result()
                if err:
                    errors += 1
                    tqdm.write(f"[ERROR] idx={idx} qa={qa_k or '?'}: {err}")
                    bar.set_postfix_str(f"errors={errors}")
                bar.update(1)

    if skipped_no_chart:
        print(f"Skipped without matching chart file: {skipped_no_chart}")
    print(f"Done. errors={errors}")
    print(f"Adapters out: {adapter_dir}")


if __name__ == "__main__":
    main()
