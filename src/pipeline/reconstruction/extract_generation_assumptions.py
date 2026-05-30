import argparse
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

PROMPT = r"""
TASK
Given chart code and matching chart data JSON:
1) Python chart code with `make_figure(data, savepath=None)`
2) The matching chart data JSON

Identify:
1) Implicit assumptions/constraints in data and plotting logic.
2) Background/domain knowledge needed to synthesize realistic data.
3) Concrete generation guidance to avoid under-specified outputs (for example complete code lists, ordered bins, taxonomy trees).

OUTPUT RULES
- Return only valid JSON.
- No markdown, no prose, no extra keys.
- Use this exact schema:
{
  "implicit_assumptions": ["..."],
  "background_knowledge": ["..."],
  "generation_guidance": ["..."]
}

RULES
- Keep each item short and actionable.
- Use both code and chart JSON; focus on data semantics, not styling.
- Capture minimal sufficient constraints for valid realistic data; avoid overfitting one sample unless required by code/domain logic.
- Include completeness requirements when relevant (coverage, ordering, mappings, join keys).
- Exclude chart-instance quirks and superficial visuals (exact positions, one-off labels/years, colors, fonts, line styles).
- Prefer generalizable data rules (schema invariants, domain constraints, aggregation/ranking logic, temporal/categorical structure, dependencies).
- If visuals imply a rule, restate it in data terms.
- Prefer relationships/ranges/distributions over exact values.
- Preserve reasonable variation (multiple plausible compositions, rank orders, magnitudes, temporal patterns).
- If uncertain, provide a cautious best-effort hypothesis and mark it with "Hypothesis:" prefix.
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


def _extract_json_object(s: str) -> dict:
    s = s.strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        start = s.find("{")
        end = s.rfind("}")
        if start < 0 or end < 0 or end <= start:
            raise ValueError("Model output is not valid JSON object.")
        obj = json.loads(s[start : end + 1])

    if not isinstance(obj, dict):
        raise ValueError("Model output JSON must be an object.")

    for k in ("implicit_assumptions", "background_knowledge", "generation_guidance"):
        v = obj.get(k, [])
        if v is None:
            v = []
        if not isinstance(v, list):
            raise ValueError(f"Field `{k}` must be a list.")
        obj[k] = [str(x).strip() for x in v if str(x).strip()]
    return obj


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


def split_dir_for(dataset_id: str, split: str) -> Path:
    dataset_cfg = get_dataset_config(dataset_id)
    return split_artifact_dir(str(dataset_cfg["local_dir"]), split)


def code_dir_for(dataset_id: str, split: str) -> str:
    return str(chart_code_dir(split_dir_for(dataset_id, split), RECONSTRUCTION_VARIANT))


def assumptions_dir_for(dataset_id: str, split: str) -> str:
    return str(split_dir_for(dataset_id, split) / "assumptions")


def chart_dir_for(dataset_id: str, split: str) -> str:
    return str(chart_data_dir(split_dir_for(dataset_id, split), RECONSTRUCTION_VARIANT))


def chart_json_path_for_input(in_path: str, in_root: str, chart_root: str) -> str:
    rel = os.path.relpath(in_path, in_root)
    rel_json = os.path.splitext(rel)[0] + ".json"
    return os.path.join(chart_root, rel_json)


def out_path_for_input(in_path: str, in_root: str, out_root: str) -> str:
    rel = os.path.relpath(in_path, in_root)
    rel_json = os.path.splitext(rel)[0] + ".json"
    return os.path.join(out_root, rel_json)


def process_file(
    client: OpenAI,
    in_path: str,
    chart_json_path: str,
    out_path: str,
    model_name: str,
    temperature: float,
) -> None:
    if os.path.exists(out_path):
        return

    with open(in_path, "r", encoding="utf-8") as f:
        src = f.read()
    with open(chart_json_path, "r", encoding="utf-8") as f:
        chart_json = f.read()

    resp = client.responses.create(
        model=model_name,
        store=False,
        temperature=temperature,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT},
                    {"type": "input_text", "text": "\n\n# MODULE SOURCE\n\n" + src},
                    {"type": "input_text", "text": "\n\n# CHART DATA JSON\n\n" + chart_json},
                ],
            }
        ],
        text={"format": {"type": "text"}},
    )

    parsed = _extract_json_object(response_text(resp))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + f".tmp.{threading.get_ident()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out_path)


def worker(
    client: OpenAI,
    in_path: str,
    chart_json_path: str,
    out_path: str,
    model_name: str,
    temperature: float,
) -> Tuple[str, Optional[str]]:
    name = os.path.basename(in_path)
    try:
        process_file(client, in_path, chart_json_path, out_path, model_name, temperature)
        return name, None
    except Exception as e:
        return name, f"{type(e).__name__}: {e}"


def main(
    dataset: str,
    split: str,
    model_name: str,
    client: OpenAI,
    temperature: float,
    chart: Optional[str] = None,
    max_workers: int = 8,
    max_files: Optional[int] = None,
) -> None:
    in_dir = os.path.abspath(code_dir_for(dataset, split))
    chart_dir = os.path.abspath(chart_dir_for(dataset, split))
    out_dir = os.path.abspath(assumptions_dir_for(dataset, split))

    if not os.path.isdir(in_dir):
        raise FileNotFoundError(f"Input chart-code directory not found: {in_dir}")
    if not os.path.isdir(chart_dir):
        raise FileNotFoundError(f"Input chart-data directory not found: {chart_dir}")

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

    tasks: List[Tuple[str, str, str]] = []
    for in_path in py_files:
        chart_json_path = chart_json_path_for_input(in_path, in_dir, chart_dir)
        if not os.path.isfile(chart_json_path):
            raise FileNotFoundError(f"Missing chart JSON for {os.path.basename(in_path)}: {chart_json_path}")
        out_path = out_path_for_input(in_path, in_dir, out_dir)
        if os.path.exists(out_path):
            continue
        tasks.append((in_path, chart_json_path, out_path))

    if not tasks:
        print("Nothing to do (all outputs already exist).")
        print(f"Output folder: {out_dir}")
        return

    errors = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [
            pool.submit(worker, client, in_path, chart_json_path, out_path, model_name, temperature)
            for (in_path, chart_json_path, out_path) in tasks
        ]

        with tqdm(
            total=len(futs),
            desc=f"assumptions {Path(out_dir).parent.name} [{split}]",
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
    print(f"Code input : {in_dir}")
    print(f"Data input : {chart_dir}")
    print(f"Output     : {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Step 1: extract implicit data assumptions/background knowledge from chart code.")
    ap.add_argument("--model-name", required=True, help="OpenAI model name")
    ap.add_argument("--dataset", required=True, help="Dataset key from config or HF dataset id")
    ap.add_argument("--split", required=True, help="Split name, e.g. train / validation / test")
    ap.add_argument("--chart", default=None, help="Specific chart file name/stem to process (default: all)")
    ap.add_argument("--max-workers", type=int, default=8, help="Worker threads")
    ap.add_argument("--max-files", type=int, default=None, help="Maximum number of files to process (default: all)")
    ap.add_argument("--temperature", type=float, default=0.0, help="Temperature for OpenAI API")
    args = ap.parse_args()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    main(
        dataset=args.dataset,
        split=args.split,
        model_name=args.model_name,
        client=client,
        temperature=args.temperature,
        chart=args.chart,
        max_workers=args.max_workers,
        max_files=args.max_files,
    )
