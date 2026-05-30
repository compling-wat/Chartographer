import argparse
import ast
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from openai import OpenAI
from tqdm import tqdm

from common.artifacts import RECONSTRUCTION_VARIANT, chart_code_dir, chart_data_dir, split_artifact_dir
from common.datasets import get_dataset_config

REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------

PROMPT = r"""
TASK
Given chart code and chart JSON, write one Python module.
The module must define `generate_data(data_template, seed=0)`.

OUTPUT RULES
- Return Python code only.
- No markdown, no prose, no backticks.

REQUIRED MODULE STRUCTURE
1) Imports (allowed: standard library + numpy + pandas + scikit-learn + statsmodels + scipy)
2) `generate_data(data_template, seed=0)`

HARD RULES
- Do not paste/copy the original data literal.
- No extra top-level helper functions/classes; put helper logic inside `generate_data`.
- `generate_data` must return only the generated data dict.
- Output must run as-is.

`generate_data` REQUIREMENTS
- Deterministic for the same seed.
- Different seeds produce qualitatively different datasets (not just tiny noise).
- `data_template` is the schema/value template input supplied by the pipeline.
- Returned dict must match `data_template` schema exactly: keys, nesting, list lengths, and value types.
- Preserve constraints implied by `make_figure` (ordering, positivity, totals, alignment, completeness, etc.).
- Derived values must come from generated base factors, not hardcoded constants.

DIVERSITY REQUIREMENTS (CRITICAL)
- Use seed-driven regimes + latent parameters.
- Add small perturbations within each regime while keeping regime identity.
- Discrete/categorical charts:
  - Use many scenario families with distinct latent parameters/regime shifts.
  - Vary ranking, composition, sparsity, dominance, interactions.
- Continuous/time-series charts:
  - Use parametric generation with effectively unbounded variation.
  - Combine trend/seasonality/shocks/changepoints/noise with seed-varying shape parameters.

ASSUMPTIONS/BACKGROUND CONTEXT
- You may receive an extra section `ASSUMPTIONS_AND_BACKGROUND`.
- Treat it as required context for data semantics and completeness.
- Prioritize those constraints while still satisfying source-module schema/logic.
"""

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

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


def normalize_generated_function(func_src: str) -> str:
    s = func_src.strip()

    try:
        tree = ast.parse(s)
    except SyntaxError as e:
        raise ValueError(f"Model output is not valid Python: {e}") from e

    return s + "\n"


def collect_py_files(root: str) -> List[str]:
    files: List[str] = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(".py"):
                files.append(os.path.join(dp, fn))
    files.sort()
    return files


def find_py_file(root: str, name: str) -> Optional[str]:
    if not name.endswith(".py"):
        name += ".py"
    path = os.path.join(root, name)
    return path if os.path.isfile(path) else None


def dataset_local_dir(dataset_id: str) -> str:
    return str(get_dataset_config(dataset_id)["local_dir"])


def split_dir_for(dataset_id: str, split: str) -> Path:
    return split_artifact_dir(dataset_local_dir(dataset_id), split)


def code_dir_for(dataset_id: str, split: str) -> str:
    return str(chart_code_dir(split_dir_for(dataset_id, split), RECONSTRUCTION_VARIANT))


def gen_dir_for(dataset_id: str, split: str) -> str:
    return str(split_dir_for(dataset_id, split) / "generate_data")


def chart_dir_for(dataset_id: str, split: str) -> str:
    return str(chart_data_dir(split_dir_for(dataset_id, split), RECONSTRUCTION_VARIANT))


def assumptions_dir_for(dataset_id: str, split: str) -> str:
    return str(split_dir_for(dataset_id, split) / "assumptions")


def out_path_for_input(in_path: str, in_root: str, out_root: str) -> str:
    rel = os.path.relpath(in_path, in_root)
    return os.path.join(out_root, rel)


def assumptions_path_for_input(in_path: str, in_root: str, assumptions_root: str) -> str:
    rel = os.path.relpath(in_path, in_root)
    rel_json = os.path.splitext(rel)[0] + ".json"
    return os.path.join(assumptions_root, rel_json)


def chart_json_path_for_input(in_path: str, in_root: str, chart_root: str) -> str:
    rel = os.path.relpath(in_path, in_root)
    rel_json = os.path.splitext(rel)[0] + ".json"
    return os.path.join(chart_root, rel_json)


def _to_lines(items) -> List[str]:
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for x in items:
        s = str(x).strip()
        if s:
            out.append(s)
    return out


def assumptions_prompt_block(assumptions_path: str) -> str:
    with open(assumptions_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(f"Assumptions file must be a JSON object: {assumptions_path}")

    implicit = _to_lines(obj.get("implicit_assumptions", []))
    knowledge = _to_lines(obj.get("background_knowledge", []))
    guidance = _to_lines(obj.get("generation_guidance", []))

    lines: List[str] = ["# ASSUMPTIONS_AND_BACKGROUND"]
    lines.append("Use all relevant items below as hard context when designing generate_data.")

    lines.append("implicit_assumptions:")
    if implicit:
        lines.extend(f"- {x}" for x in implicit)
    else:
        lines.append("- (none)")

    lines.append("background_knowledge:")
    if knowledge:
        lines.extend(f"- {x}" for x in knowledge)
    else:
        lines.append("- (none)")

    lines.append("generation_guidance:")
    if guidance:
        lines.extend(f"- {x}" for x in guidance)
    else:
        lines.append("- (none)")

    return "\n".join(lines).strip()

# ---------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------

def process_file(
    client: OpenAI,
    in_path: str,
    chart_json_path: str,
    out_path: str,
    model_name: str,
    temperature: float,
    assumptions_path: Optional[str],
    strict_assumptions: bool,
) -> None:
    if os.path.exists(out_path):
        return

    with open(in_path, "r", encoding="utf-8") as f:
        src = f.read()
    if not os.path.exists(chart_json_path):
        raise FileNotFoundError(f"Missing chart JSON: {chart_json_path}")
    with open(chart_json_path, "r", encoding="utf-8") as f:
        chart_json = f.read()

    assumptions_block = ""
    if assumptions_path and os.path.exists(assumptions_path):
        assumptions_block = assumptions_prompt_block(assumptions_path)
    elif strict_assumptions:
        raise FileNotFoundError(f"Missing assumptions file: {assumptions_path}")

    extra_context = f"\n\n{assumptions_block}" if assumptions_block else ""

    resp = client.responses.create(
        model=model_name,
        store=False,
        temperature=temperature,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT + extra_context},
                    {"type": "input_text", "text": "\n\n# MODULE SOURCE\n\n" + src},
                    {"type": "input_text", "text": "\n\n# CHART DATA JSON\n\n" + chart_json},
                ],
            }
        ],
        text={"format": {"type": "text"}},
    )

    gen_func = normalize_generated_function(response_text(resp))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + f".tmp.{threading.get_ident()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(gen_func)
    os.replace(tmp, out_path)


def worker(
    client: OpenAI,
    in_path: str,
    chart_json_path: str,
    out_path: str,
    model_name: str,
    temperature: float,
    assumptions_path: Optional[str],
    strict_assumptions: bool,
) -> Tuple[str, Optional[str]]:
    name = os.path.basename(in_path)
    try:
        process_file(
            client,
            in_path,
            chart_json_path,
            out_path,
            model_name,
            temperature,
            assumptions_path=assumptions_path,
            strict_assumptions=strict_assumptions,
        )
        return name, None
    except Exception as e:
        return name, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(
    dataset: str,
    split: str,
    model_name: str,
    client: OpenAI,
    temperature: float,
    chart: Optional[str] = None,
    strict_assumptions: bool = False,
    max_workers: int = 8,
    max_files: Optional[int] = None,
) -> None:
    in_dir = os.path.abspath(code_dir_for(dataset, split))
    chart_dir = os.path.abspath(chart_dir_for(dataset, split))
    out_dir = os.path.abspath(gen_dir_for(dataset, split))
    assumptions_dir = os.path.abspath(assumptions_dir_for(dataset, split))

    if not os.path.isdir(in_dir):
        raise FileNotFoundError(
            f"Input chart-code directory not found: {in_dir}\n"
            f"Expected: {code_dir_for(dataset, split)}"
        )
    if not os.path.isdir(chart_dir):
        raise FileNotFoundError(
            f"Chart JSON directory not found: {chart_dir}\n"
            f"Expected: {chart_dir_for(dataset, split)}"
        )

    if chart:
        chart_file = find_py_file(in_dir, chart)
        if chart_file is None:
            available = "\n".join(f"  - {os.path.splitext(os.path.basename(p))[0]}" for p in collect_py_files(in_dir))
            raise FileNotFoundError(f"Chart file not found.\nAvailable charts:\n{available}")
        py_files = [chart_file]
    else:
        py_files = collect_py_files(in_dir)
        if max_files is not None:
            if max_files < 0:
                raise ValueError("--max-files must be >= 0 or omitted.")
            py_files = py_files[:max_files]

    if not py_files:
        print(f"No .py files found in: {in_dir}")
        return

    tasks: List[Tuple[str, str, str, str]] = []
    for in_path in py_files:
        out_path = out_path_for_input(in_path, in_dir, out_dir)
        if os.path.exists(out_path):
            continue
        chart_json_path = chart_json_path_for_input(in_path, in_dir, chart_dir)
        assump_path = assumptions_path_for_input(in_path, in_dir, assumptions_dir)
        tasks.append((in_path, chart_json_path, out_path, assump_path))

    if not tasks:
        print("Nothing to do (all outputs already exist).")
        print(f"Output folder: {out_dir}")
        return

    errors = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [
            pool.submit(
                worker,
                client,
                in_path,
                chart_json_path,
                out_path,
                model_name,
                temperature,
                assump_path,
                strict_assumptions,
            )
            for (in_path, chart_json_path, out_path, assump_path) in tasks
        ]

        with tqdm(
            total=len(futs),
            desc=f"code2gen {dataset_local_dir(dataset)} [{split}]",
            unit="file",
            smoothing=0.1,
            dynamic_ncols=True,
        ) as bar:
            for fut in as_completed(futs):
                name, err = fut.result()
                if err:
                    errors += 1
                    tqdm.write(f"[ERROR] {name}: {err}")
                    bar.set_postfix_str(f"errors={errors}")
                bar.update(1)

    print(f"Done. errors={errors}")
    print(f"Input : {in_dir}")
    print(f"Chart JSON: {chart_dir}")
    print(f"Assumptions: {assumptions_dir}")
    print(f"Output: {out_dir}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Generate data generation functions for chart code using an OpenAI model. "
    )
    ap.add_argument("--model-name", required=True, help="OpenAI model name")
    ap.add_argument("--dataset", required=True, help="Dataset key from config or HF dataset id")
    ap.add_argument("--split", required=True, help="Split name, e.g. train / validation / test")
    ap.add_argument("--chart", default=None, help="Specific chart file name/stem to process (default: all)")
    ap.add_argument("--max-workers", type=int, default=8, help="Worker threads")
    ap.add_argument("--max-files", type=int, default=None, help="Maximum number of files to process (default: all)")
    ap.add_argument("--temperature", type=float, default=0.0, help="Temperature for OpenAI API")
    ap.add_argument(
        "--strict-assumptions",
        action="store_true",
        help="Require assumptions JSON from assumptions/ for each chart file.",
    )

    args = ap.parse_args()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    main(
        dataset=args.dataset,
        split=args.split,
        model_name=args.model_name,
        client=client,
        temperature=args.temperature,
        chart=args.chart,
        strict_assumptions=args.strict_assumptions,
        max_workers=args.max_workers,
        max_files=args.max_files
    )
