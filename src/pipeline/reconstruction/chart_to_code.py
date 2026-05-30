import argparse
import base64
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from openai import OpenAI
from tqdm import tqdm

from common.artifacts import RECONSTRUCTION_VARIANT, chart_code_dir, chart_data_dir, split_artifact_dir
from common.datasets import get_dataset_config, load_split_basenames

SRC_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------

PROMPT = r"""
TASK
Given one reference chart image, produce:
1) `chart_data`: underlying semantic data only
2) `chart_code`: runnable Python that recreates a similar chart

OUTPUT RULES
- Return only valid JSON.
- Return one JSON object with exactly two keys: `chart_data`, `chart_code`.
- No markdown, no prose, no extra keys.

ALLOWED LIBRARIES
- Standard library + numpy + pandas + matplotlib + seaborn + cartopy.
- scipy and scikit-learn are allowed for data generation/processing.
- No external data/files/API calls.

`chart_data` RULES
- Keep semantic data only: entities, categories, labels, dates, values, groups, semantic/domain ordering.
- Prefer compact representations (tables, grouped arrays, control points, parameters), not dense traces.
- Do not do pixel/per-vertex tracing from the rendered figure.
- Do not store dense plotting coordinates if code can generate them.
- If exact values are unreadable, infer plausible compact data preserving key trends, ranks, crossings, and boundaries.
- Dense arrays are allowed only when density is intrinsic data (for example raster/scientific fields).
- Do not include visual/layout config (for example color, linestyle, marker, linewidth, axis limits, ticks, title, legend, annotation positions, figsize, dpi, grid/spines, draw/legend/subplot/z-order).

`chart_code` RULES
- Must derive visual encodings from `chart_data`.
- Do not duplicate in `chart_code` any information already present in `chart_data`; read and use it from `data` instead of hardcoding.
- Required structure: imports + `make_figure(data, savepath=None)`.
- `data` is required and must not default to `None`.
- `make_figure` must consume the provided `data` object directly.
- No extra top-level functions/classes. Do not call `plt.show()`.
- If `savepath` is provided, save with DPI >= 300 and `bbox_inches='tight'`, then close the figure.
- Keep style/layout choices in code, not in `chart_data`.
- For maps/geo charts (for example choropleth, projected map, lat/lon geoplots), use cartopy.

QUALITY PRIORITY
- Prioritize semantic fidelity over point-by-point geometric fidelity.
- Match layout/style reasonably; if exact recreation is hard, produce a faithful simpler approximation.

"""

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

VALID_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def resolve_api_key_for_model(model_name: str) -> Optional[str]:
    return os.getenv("OPENAI_API_KEY")


def build_openai_client_for_model(model_name: str) -> OpenAI:
    api_key = resolve_api_key_for_model(model_name)
    if not api_key:
        raise RuntimeError("Missing API key. Set OPENAI_API_KEY.")
    return OpenAI(api_key=api_key)


def encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def dataset_to_local_root(dataset_id: str) -> str:
    """Dataset alias/HF id -> local image folder under REPO_ROOT/data/."""
    cfg = get_dataset_config(dataset_id)
    local_name = str(cfg.get("local_dir", dataset_id.split("/")[-1].lower()))
    return str(REPO_ROOT / "data" / local_name)


def list_local_images(root: str) -> List[str]:
    images: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(VALID_IMAGE_EXTS):
                images.append(os.path.join(dirpath, fn))
    return images


def response_text(resp) -> str:
    """
    Extract plain text from a Responses API response.

    Prefer SDK helper `resp.output_text`. Fall back to scanning output blocks.
    """
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

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


def out_path_for_image(image_path: str, target_folder: str) -> str:
    base = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(target_folder, base + ".py")


def json_out_path_for_image(image_path: str, target_folder: str) -> str:
    base = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(target_folder, base + ".json")


def parse_generated_payload(text: str) -> Dict[str, object]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model output is not valid JSON: {e}") from e

    if not isinstance(obj, dict):
        raise ValueError("Model output must be a JSON object.")

    if set(obj.keys()) != {"chart_data", "chart_code"}:
        raise ValueError("Model output must contain exactly 'chart_data' and 'chart_code'.")

    if not isinstance(obj["chart_data"], dict):
        raise ValueError("'chart_data' must be a JSON object.")

    if not isinstance(obj["chart_code"], str) or not obj["chart_code"].strip():
        raise ValueError("'chart_code' must be a non-empty string.")

    return obj


# ---------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------

def process_image(
    client: OpenAI,
    image_path: str,
    code_target_folder: str,
    data_target_folder: str,
    model_name: str,
    temperature: float,
) -> Optional[str]:
    out_path = out_path_for_image(image_path, code_target_folder)
    json_out_path = json_out_path_for_image(image_path, data_target_folder)
    if os.path.exists(out_path) and os.path.exists(json_out_path):
        return

    b64 = encode_image_b64(image_path)

    raw_text = ""
    needs_fix: Optional[str] = None
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
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{b64}",
                            "detail": "high",
                        },
                    ],
                }
            ],
            text={"format": {"type": "text"}},
        )
        raw_text = response_text(resp)
        payload = parse_generated_payload(raw_text)
    except ValueError as e:
        # Preserve model output for later manual cleanup instead of dropping the case.
        payload = {
            "chart_data": {
                "_parse_error": str(e),
                "_raw_model_output": raw_text,
            },
            "chart_code": (
                "# Manual review needed: model response was not valid JSON with expected keys.\n"
                "# See chart_data['_raw_model_output'] in the matching .json file and fix manually.\n"
            ),
        }
        needs_fix = str(e)
    except Exception as e:
        if isinstance(raw_text, str) and raw_text.strip():
            # If the model produced text but we still failed, keep artifacts for revise.
            payload = {
                "chart_data": {
                    "_parse_error": f"{type(e).__name__}: {e}",
                    "_raw_model_output": raw_text,
                },
                "chart_code": (
                    "# Manual review needed: model response was captured but processing failed.\n"
                    "# See chart_data['_raw_model_output'] in the matching .json file and fix manually.\n"
                ),
            }
            needs_fix = f"{type(e).__name__}: {e}"
        else:
            raise

    content = payload["chart_code"]
    chart_data = payload["chart_data"]

    os.makedirs(code_target_folder, exist_ok=True)
    os.makedirs(data_target_folder, exist_ok=True)
    tmp = out_path + f".tmp.{threading.get_ident()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, out_path)

    tmp_json = json_out_path + f".tmp.{threading.get_ident()}"
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(chart_data, f, indent=2)
        f.write("\n")
    os.replace(tmp_json, json_out_path)
    return needs_fix


def worker(
    client: OpenAI,
    image_path: str,
    code_target_folder: str,
    data_target_folder: str,
    model_name: str,
    temperature: float,
) -> Tuple[str, Optional[str], Optional[str]]:
    name = os.path.basename(image_path)
    try:
        if not os.path.exists(image_path):
            return name, f"File not found: {image_path}", None
        needs_fix = process_image(client, image_path, code_target_folder, data_target_folder, model_name, temperature)
        return name, None, needs_fix
    except Exception as e:
        return name, f"{type(e).__name__}: {e}", None


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(
    dataset: str,
    split: str,
    code_target_folder: str,
    data_target_folder: str,
    model_name: str,
    client: OpenAI,
    temperature: float,
    max_workers: int = 8,
    max_charts: Optional[int] = None,
    chart_name: Optional[str] = None,
) -> None:
    data_root = dataset_to_local_root(dataset)
    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"Local dataset folder not found: {data_root}\n"
            f"Expected rule: ../data/{dataset.split('/')[-1].lower()}"
        )

    os.makedirs(code_target_folder, exist_ok=True)
    os.makedirs(data_target_folder, exist_ok=True)

    split_basenames = set(load_split_basenames(dataset, split))
    local_images = list_local_images(data_root)
    selected = [p for p in local_images if os.path.basename(p) in split_basenames]

    if chart_name:
        requested_stem = Path(chart_name).stem
        selected = [p for p in selected if Path(p).stem == requested_stem]
        if not selected:
            raise ValueError(
                f"Chart '{requested_stem}' not found in dataset='{dataset}', split='{split}', "
                f"under local root '{data_root}'."
            )
        print(f"Filtering to chart: {requested_stem}")

    print(f"Local root: {data_root}")
    print(f"Found {len(selected)} local images matching split '{split}'")

    to_process = [
        p
        for p in selected
        if not (
            os.path.exists(out_path_for_image(p, code_target_folder))
            and os.path.exists(json_out_path_for_image(p, data_target_folder))
        )
    ]

    if not to_process:
        print("Nothing to do (all outputs already exist).")
        return

    if max_charts is not None:
        if max_charts < 0:
            raise ValueError("--max-charts must be >= 0 or omitted.")
        to_process = to_process[:max_charts]
        print(f"Limiting to first {len(to_process)} charts")

    errors = 0
    failed_charts: List[str] = []
    needs_fix_charts: List[str] = []
    needs_fix_records: List[Dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(worker, client, p, code_target_folder, data_target_folder, model_name, temperature)
            for p in to_process
        ]

        with tqdm(
            total=len(to_process),
            desc=f"{dataset} [{split}]",
            unit="img",
            smoothing=0.1,
            dynamic_ncols=True,
        ) as bar:
            for fut in as_completed(futures):
                name, err, needs_fix = fut.result()
                if err:
                    errors += 1
                    failed_charts.append(os.path.splitext(name)[0])
                    tqdm.write(f"[ERROR] {name}: {err}")
                    bar.set_postfix_str(f"errors={errors}")
                elif needs_fix:
                    stem = os.path.splitext(name)[0]
                    needs_fix_charts.append(stem)
                    needs_fix_records.append(
                        {
                            "chart": stem,
                            "reason": needs_fix,
                            "chart_data": os.path.join(data_target_folder, f"{stem}.json"),
                            "chart_code": os.path.join(code_target_folder, f"{stem}.py"),
                        }
                    )
                    tqdm.write(f"[NEEDS_FIX] {name}: {needs_fix}")
                bar.update(1)

    print(f"Done. errors={errors}")
    if failed_charts:
        failed_sorted = sorted(set(failed_charts))
        print(f"Failed charts ({len(failed_sorted)}): {', '.join(failed_sorted)}")
    if needs_fix_charts:
        needs_fix_sorted = sorted(set(needs_fix_charts))
        print(f"Needs-fix charts ({len(needs_fix_sorted)}): {', '.join(needs_fix_sorted)}")
        needs_fix_path = os.path.join(os.path.dirname(os.path.dirname(data_target_folder)), "needs_fix_chart_to_code.jsonl")
        with open(needs_fix_path, "w", encoding="utf-8") as f:
            for rec in sorted(needs_fix_records, key=lambda r: r["chart"]):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Saved needs-fix records to: {needs_fix_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate chart code from reference images using an OpenAI model. "
    )
    parser.add_argument("--model-name", required=True, help="OpenAI model name (must support images)")
    parser.add_argument("--dataset", required=True, help="Dataset key from config or HF dataset id")
    parser.add_argument("--split", required=True, help="Split name, e.g. train / validation / test")
    parser.add_argument("--max-workers", type=int, default=8, help="Worker threads (tune for rate limits)")
    parser.add_argument("--max-charts", type=int, default=None, help="Maximum number of charts to process (default: all)")
    parser.add_argument(
        "--chart",
        dest="chart_name",
        type=str,
        default=None,
        help="Process only one chart by stem or filename (e.g., 1569 or 1569.jpg).",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for OpenAI API")
    args = parser.parse_args()

    dataset_cfg = get_dataset_config(args.dataset)
    base_folder = split_artifact_dir(str(dataset_cfg["local_dir"]), args.split)
    code_target_folder = str(chart_code_dir(base_folder, RECONSTRUCTION_VARIANT))
    data_target_folder = str(chart_data_dir(base_folder, RECONSTRUCTION_VARIANT))
    client = build_openai_client_for_model(args.model_name)

    main(
        dataset=args.dataset,
        split=args.split,
        code_target_folder=code_target_folder,
        data_target_folder=data_target_folder,
        model_name=args.model_name,
        client=client,
        temperature=args.temperature,
        max_workers=args.max_workers,
        max_charts=args.max_charts,
        chart_name=args.chart_name,
    )
