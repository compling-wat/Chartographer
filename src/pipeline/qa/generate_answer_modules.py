import argparse
import ast
import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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


def _configure_hf_cache() -> None:
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_DATASETS_CACHE_DIR))
    HF_DATASETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = r"""
TASK
Write one Python module that defines `generate_answer(data)` for the given QA context.

INPUTS
1) Question
2) Chart data JSON
3) Chart code
4) Question-adapter module for this QA pair

OUTPUT RULES
- Return Python code only.
- No markdown, no prose, no backticks.

RUNTIME CONTRACT
- Runtime input is only `data`.
- Question/chart code/adapter are prompt-only context.
- Do not read `data["question"]`, `data["answer"]`, or assume QA text is embedded in `data` unless those keys are explicitly present in the provided JSON schema.

RULES
- Compute exactly the requested operation and scope.
- Use data as value source; use chart code/adapter only for semantic intent recovery.
- Use only fields/entities actually present in the schema/data; do not invent keys or values.
- Do not hard-code the provided answer or example-specific constants.
- Generalize to same-schema regenerated data.

MODULE RULES
- Define exactly one top-level function: `generate_answer(data)`.
- Top-level code may contain imports only.
- `generate_answer(data)` must return only the final answer string.
- Accept exactly one argument named `data`.
- Derive the answer from `data`, not from copied constants.
- Preserve the answer type and format implied by the question.
- Handle reordered elements, changed numeric values, and ties when relevant.
- Never return an empty string for valid same-schema input.
"""

def response_text(resp) -> str:
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


def normalize_generated_module(module_src: str) -> str:
    s = module_src.strip()
    try:
        tree = ast.parse(s)
    except SyntaxError as e:
        raise ValueError(f"Model output is not valid Python: {e}") from e

    top_level_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(top_level_defs) != 1:
        raise ValueError("Model output must contain exactly one top-level function.")

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            continue
        raise ValueError("Top-level statements must be imports and generate_answer(data) only.")

    func = top_level_defs[0]
    if func.name != "generate_answer":
        raise ValueError("Top-level function must be named generate_answer.")
    if len(func.args.args) != 1 or func.args.args[0].arg != "data":
        raise ValueError("generate_answer must accept exactly one parameter named data.")

    return s + "\n"


def _compile_answer_generator(code: str) -> Callable[[Dict[str, Any]], str]:
    sandbox_globals: Dict[str, Any] = {"__builtins__": __builtins__}
    exec(code, sandbox_globals, sandbox_globals)
    fn = sandbox_globals.get("generate_answer")
    if not callable(fn):
        raise ValueError("answer_generator_python must define callable generate_answer(data).")
    return fn


def _run_answer_generator(code: str, data_obj: Dict[str, Any]) -> str:
    fn = _compile_answer_generator(code)
    answer = fn(data_obj)
    if answer is None:
        raise ValueError("generate_answer(data) returned None.")
    answer_text = str(answer).strip()
    if not answer_text:
        raise ValueError("generate_answer(data) returned an empty answer.")
    return answer_text


def _resolve_dataset(dataset_key_or_hf: str) -> Tuple[str, Dict[str, str]]:
    normalized = normalize_dataset_name(dataset_key_or_hf)
    cfg = get_dataset_config(dataset_key_or_hf)
    return (
        str(cfg.get("local_dir") or normalized.split("/")[-1].lower()),
        {
            "question_col": str(cfg.get("question_col", "question")),
            "answer_col": str(cfg.get("answer_col", "answer")),
            "image_col": str(cfg.get("image_col", "image")),
            "figure_path_col": str(cfg.get("figure_path_col", "")),
            "original_figure_path_col": str(cfg.get("original_figure_path_col", "")),
        },
    )


def _split_dir(repo_key: str, split: str) -> Path:
    return split_artifact_dir(repo_key, split)


def _data_json_dir(repo_key: str, split: str) -> Path:
    return chart_data_dir(_split_dir(repo_key, split), RECONSTRUCTION_VARIANT)


def _code_dir(repo_key: str, split: str) -> Path:
    return chart_code_dir(_split_dir(repo_key, split), RECONSTRUCTION_VARIANT)


def _adapter_dir(repo_key: str, split: str) -> Path:
    return _split_dir(repo_key, split) / "question_adapters"


def _output_dir(
    dataset_key: str,
    split: str,
) -> Path:
    return _split_dir(dataset_key, split) / "generate_answers"


def _out_path_for_stem(out_dir: Path, stem: str) -> Path:
    return out_dir / f"{stem}.py"


def _qa_key(stem: str, question: str) -> str:
    q = question.strip()
    digest = hashlib.sha1(q.encode("utf-8")).hexdigest()[:12]
    return f"{stem}__{digest}"


def _out_path_for_qa(out_dir: Path, stem: str, question: str) -> Path:
    return _out_path_for_stem(out_dir, _qa_key(stem, question))


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


def _process_one(
    client: OpenAI,
    model_name: str,
    temperature: float,
    max_attempts: int,
    example_idx: int,
    example: Dict[str, Any],
    question_col: str,
    answer_col: str,
    ds_cfg: Dict[str, str],
    data_dir: Path,
    code_dir: Path,
    adapter_dir: Path,
    out_dir: Path,
    overwrite: bool,
) -> Tuple[int, str, Optional[str]]:
    try:
        image_name = example_image_basename(example, ds_cfg)
        stem = Path(image_name).stem
        if not stem:
            return example_idx, "", "missing image stem"

        original_q = str(example.get(question_col, "")).strip()
        qa_key = _qa_key(stem, original_q)

        out_path = _out_path_for_qa(out_dir, stem, original_q)
        if out_path.exists() and not overwrite:
            return example_idx, qa_key, None

        data_path = data_dir / f"{stem}.json"

        if not data_path.exists():
            return example_idx, qa_key, f"missing data json for {stem}: expected {data_path}"

        code_path = code_dir / f"{stem}.py"
        if not code_path.exists():
            return example_idx, qa_key, f"missing chart code for {stem}: expected {code_path}"
        adapter_path = adapter_dir / f"{qa_key}.py"
        if not adapter_path.exists():
            return example_idx, qa_key, f"missing question adapter for {qa_key}: expected {adapter_path}"

        data_obj = _load_json(data_path)
        data_src = json.dumps(data_obj, ensure_ascii=False, indent=2)
        code_src = _load_text(code_path)
        adapter_src = _load_text(adapter_path)

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
                                        f"\n\n# QA\n"
                                        f"question: {original_q}\n\n"
                                        f"# DATA_JSON\n{data_src}\n\n"
                                        f"# CHART_CODE\n{code_src}\n"
                                        f"\n# QUESTION_ADAPTER_MODULE\n{adapter_src}\n"
                                    ),
                                },
                            ],
                        }
                    ],
                    text={"format": {"type": "text"}},
                )

                raw_code = response_text(resp)
                generator_code = normalize_generated_module(raw_code)
                _run_answer_generator(generator_code, data_obj)
                _write_text(out_path, generator_code)
                return example_idx, stem, None
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"

        return example_idx, qa_key, f"{last_err} (after {max_attempts} attempts)"
    except Exception as e:
        return example_idx, stem if "stem" in locals() else "", f"{type(e).__name__}: {e}"


def main(
    dataset: str,
    split: str,
    model_name: str,
    temperature: float,
    chart: Optional[str],
    max_attempts: int,
    max_workers: int,
    max_items: Optional[int],
    overwrite: bool,
) -> None:
    dataset_key, ds_cfg = _resolve_dataset(dataset)
    _configure_hf_cache()
    ds = load_examples(dataset, split)
    question_col = ds_cfg["question_col"]
    answer_col = ds_cfg["answer_col"]
    image_col = ds_cfg["image_col"]
    for col in (question_col, answer_col, image_col):
        if col not in ds.column_names:
            raise ValueError(f"Dataset split missing required column '{col}'. Columns: {ds.column_names}")

    examples = list(ds)
    if max_items is not None:
        if max_items < 0:
            raise ValueError("--max-items must be >= 0.")
        examples = examples[:max_items]

    data_dir = _data_json_dir(dataset_key, split).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Chart data dir not found: {data_dir}")
    code_dir = _code_dir(dataset_key, split).resolve()
    if not code_dir.exists():
        raise FileNotFoundError(f"Chart code dir not found: {code_dir}")
    adapter_dir = _adapter_dir(dataset_key, split).resolve()
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Question adapter dir not found: {adapter_dir}")

    chart_stems = {p.stem for p in data_dir.iterdir() if p.is_file()}
    total_examples = len(examples)
    examples = [
        ex
        for ex in examples
        if (stem := Path(example_image_basename(ex, ds_cfg)).stem) and stem in chart_stems
    ]
    skipped_no_chart = total_examples - len(examples)

    if chart:
        matched = _find_example_by_chart(examples, ds_cfg, chart)
        if matched is None:
            available = sorted(chart_stems)
            lines = "\n".join(f"  - {name}" for name in available[:80])
            if len(available) > 80:
                lines += f"\n  ... ({len(available) - 80} more)"
            raise FileNotFoundError(f"Chart not found: {chart}\nAvailable charts:\n{lines}")
        examples = [matched]
        skipped_no_chart = 0

    out_dir = _output_dir(dataset_key, split).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    errors = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                _process_one,
                client,
                model_name,
                temperature,
                max_attempts,
                i,
                ex,
                question_col,
                answer_col,
                ds_cfg,
                data_dir,
                code_dir,
                adapter_dir,
                out_dir,
                overwrite,
            )
            for i, ex in enumerate(examples)
        ]

        with tqdm(total=len(futures), desc=f"data2qa {dataset_key} [{split}]", unit="item", dynamic_ncols=True) as bar:
            for fut in as_completed(futures):
                idx, stem, err = fut.result()
                if err:
                    errors += 1
                    tqdm.write(f"[ERROR] idx={idx} chart={stem or '?'}: {err}")
                    bar.set_postfix_str(f"errors={errors}")
                bar.update(1)

    if skipped_no_chart:
        print(f"Skipped without matching chart file: {skipped_no_chart}")
    print(f"Done. errors={errors}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate answer-mapping modules from original QA and data.")
    ap.add_argument("--model-name", required=True, help="OpenAI model name")
    ap.add_argument("--dataset", required=True, help="Dataset key from config or HF dataset id")
    ap.add_argument("--split", required=True, help="Dataset split, e.g. dev/test")
    ap.add_argument("--chart", default=None, help="Specific chart stem or .py filename to process")
    ap.add_argument("--max-attempts", type=int, default=1, help="Total generation attempts per chart")
    ap.add_argument("--max-workers", type=int, default=8, help="Worker threads")
    ap.add_argument("--max-items", type=int, default=None, help="Process at most first N items")
    ap.add_argument("--overwrite", action="store_true", help="Regenerate modules even if output file already exists")
    ap.add_argument("--temperature", type=float, default=0.0, help="OpenAI temperature")
    args = ap.parse_args()

    model_name = args.model_name

    main(
        dataset=args.dataset,
        split=args.split,
        model_name=model_name,
        temperature=args.temperature,
        chart=args.chart,
        max_attempts=args.max_attempts,
        max_workers=args.max_workers,
        max_items=args.max_items,
        overwrite=args.overwrite,
    )
