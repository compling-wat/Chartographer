#!/usr/bin/env python3
import argparse
import io
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from PIL import Image
from tqdm import tqdm

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

try:
    from clients.claude_client import ClaudeClient, ClaudeClientConfig
    from clients.huggingface_vlm_client import ClientConfig, ModelClient
    from clients.openai_client import OpenAIClient, OpenAIClientConfig
    from common.datasets import (
        REPO_ROOT,
        get_dataset_config,
        load_examples,
        resolve_column,
        resolve_repo_local_path,
    )
    from common.prediction_io import dump_json as atomic_json_dump
    from common.prediction_io import get_output_path, load_existing_predictions
    from config.task_prompts import CHARTMUSEUM_PROMPTS
except ModuleNotFoundError:
    # Support running as `python -m src.pipeline.prediction.generate_predictions`
    from src.clients.claude_client import ClaudeClient, ClaudeClientConfig
    from src.clients.huggingface_vlm_client import ClientConfig, ModelClient
    from src.clients.openai_client import OpenAIClient, OpenAIClientConfig
    from src.common.datasets import (
        REPO_ROOT,
        get_dataset_config,
        load_examples,
        resolve_column,
        resolve_repo_local_path,
    )
    from src.common.prediction_io import dump_json as atomic_json_dump
    from src.common.prediction_io import get_output_path, load_existing_predictions
    from src.config.task_prompts import CHARTMUSEUM_PROMPTS


# ------------------------ prompt builder ------------------------ #

def build_messages(prompt: str, question: Any, conversation: str = "") -> List[Dict[str, str]]:
    prompt_filled = prompt.replace("[QUESTION]", str(question))
    if "[CONVERSATION]" in prompt_filled:
        prompt_filled = prompt_filled.replace("[CONVERSATION]", conversation)
    return [{"role": "user", "content": prompt_filled}]


def infer_client_backend(model: str) -> str:
    model_lc = (model or "").lower()
    if "gemini" in model_lc:
        return "openai"
    if (
        "claude" in model_lc
        or model_lc.startswith("anthropic/")
        or "/claude" in model_lc
    ):
        return "claude"

    openai_markers = (
        "gpt-",
        "chatgpt",
        "o1",
        "o3",
    )
    if model_lc.startswith(openai_markers) or model_lc in {"o1", "o1-mini", "o3", "o3-mini"}:
        return "openai"
    return "open_source"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_chart_filters(chart_args: Sequence[str] | None, charts_arg: str | None) -> List[str]:
    filters: List[str] = []
    for value in chart_args or []:
        filters.extend(part.strip() for part in str(value).split(",") if part.strip())
    if charts_arg:
        filters.extend(part.strip() for part in str(charts_arg).split(",") if part.strip())
    return list(dict.fromkeys(filters))


def example_matches_chart(example: Any, filters: Sequence[str]) -> bool:
    if not filters:
        return True
    haystack_parts: List[str] = []
    for key in ("image", "chart_id", "question_id", "chart_data_json", "_image_stem"):
        value = example.get(key) if hasattr(example, "get") else None
        if value is not None:
            haystack_parts.append(str(value))
    haystack = "\n".join(haystack_parts)
    return any(chart in haystack for chart in filters)


# ------------------------ main generation ------------------------ #

def generate_predictions(args) -> str:
    ds_cfg = get_dataset_config(args.dataset)
    dataset = load_examples(args.dataset, args.split)

    q_col = resolve_column(ds_cfg, dataset, "question_col", "question_cols")
    img_col = resolve_column(ds_cfg, dataset, "image_col", "image_cols")

    if args.limit is not None:
        dataset = dataset.select(range(min(len(dataset), args.limit)))

    output_suffix = f"seed{args.seed}" if args.seed is not None else None
    out_path = get_output_path(args.dataset, args.split, args.model, args.limit, output_suffix=output_suffix)
    predictions = load_existing_predictions(out_path, args.resume)
    chart_filters = parse_chart_filters(args.chart, args.charts)
    selected_indices = [
        i for i in range(len(dataset))
        if example_matches_chart(dataset[i], chart_filters)
    ]

    if chart_filters:
        if not selected_indices:
            raise ValueError(f"No examples matched chart filter(s): {', '.join(chart_filters)}")
        if len(predictions) != len(dataset):
            raise ValueError(
                "Chart-filtered replacement requires an existing full predictions file with "
                f"{len(dataset)} rows. Found {len(predictions)} rows at {out_path}."
            )
        print(
            "[Chart filter] replacing "
            f"{len(selected_indices)} prediction rows matching: {', '.join(chart_filters)}"
        )
    else:
        start_idx = len(predictions)

        if start_idx > 0:
            print(f"[Resume] {start_idx} predictions already exist")

    client_backend = infer_client_backend(args.model)

    if client_backend == "openai":
        cfg = OpenAIClientConfig(
            model=args.model,
            max_retries=args.api_max_retries,
            retry_base_delay_sec=args.api_retry_base_delay_sec,
            retry_max_delay_sec=args.api_retry_max_delay_sec,
            timeout_sec=args.api_timeout_sec,
        )
        client = OpenAIClient(cfg)
    elif client_backend == "claude":
        cfg = ClaudeClientConfig(
            model=args.model,
            max_tokens=args.max_new_tokens,
            enable_prompt_caching=False,
        )
        client = ClaudeClient(cfg)
    else:
        cfg = ClientConfig(
            model=args.model,
            device="auto",
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            trust_remote_code=True,
        )
        client = ModelClient(cfg)

    use_parallel_api = client_backend in {"openai", "claude"} and args.api_workers > 1

    def process_image(img: Any) -> Any:
        if isinstance(img, str) and not img.startswith("http"):
            normalized = resolve_repo_local_path(img)
            if normalized is not None:
                return str(normalized)
            return str(REPO_ROOT / "data" / args.dataset / img)
        if isinstance(img, (bytes, bytearray)):
            return Image.open(io.BytesIO(img)).convert("RGB")
        if isinstance(img, dict):
            if img.get("bytes") is not None:
                return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            if img.get("path") is not None:
                return img["path"]
        return img

    def predict_single_example(i: int) -> Any:
        if client_backend == "open_source" and args.seed is not None:
            # Keep sampling stochastic but reproducible per example/resume position.
            set_global_seed(args.seed + i)
        ex = dataset[i]

        prompt = CHARTMUSEUM_PROMPTS["QA"]

        img_input = [process_image(ex[img_col])]
        messages = build_messages(prompt, ex[q_col])
        pred = client.generate(messages=messages, images=img_input)
        return pred

    if chart_filters:
        if use_parallel_api:
            with ThreadPoolExecutor(max_workers=args.api_workers) as pool:
                future_to_idx = {
                    pool.submit(predict_single_example, i): i
                    for i in selected_indices
                }
                with tqdm(total=len(future_to_idx), desc="Generating selected") as pbar:
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            predictions[idx] = future.result()
                        except Exception as e:
                            raise RuntimeError(f"Failed at dataset index {idx}: {e}") from e
                        atomic_json_dump(out_path, predictions)
                        pbar.update(1)
        else:
            for i in tqdm(selected_indices, desc="Generating selected", total=len(selected_indices)):
                predictions[i] = predict_single_example(i)
                atomic_json_dump(out_path, predictions)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    elif use_parallel_api:
        next_to_write = start_idx
        pending_by_idx: Dict[int, Any] = {}
        with ThreadPoolExecutor(max_workers=args.api_workers) as pool:
            future_to_idx = {
                pool.submit(predict_single_example, i): i
                for i in range(start_idx, len(dataset))
            }
            with tqdm(total=len(future_to_idx), desc="Generating") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        pending_by_idx[idx] = future.result()
                    except Exception as e:
                        raise RuntimeError(f"Failed at dataset index {idx}: {e}") from e
                    pbar.update(1)

                    while next_to_write in pending_by_idx:
                        pred = pending_by_idx.pop(next_to_write)
                        predictions.append(pred)
                        next_to_write += 1
                        atomic_json_dump(out_path, predictions)
    else:
        for i in tqdm(
            range(start_idx, len(dataset)),
            desc="Generating",
            total=len(dataset),
            initial=start_idx,
        ):
            pred = predict_single_example(i)
            predictions.append(pred)
            atomic_json_dump(out_path, predictions)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return str(out_path)


# ------------------------ entry ------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None, help="Optional base seed for reproducible HF sampling.")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false", help="Ignore existing predictions.")
    ap.add_argument(
        "--chart",
        action="append",
        default=None,
        help="Chart stem/filter to rerun in place. Can be repeated or comma-separated.",
    )
    ap.add_argument(
        "--charts",
        default=None,
        help="Comma-separated chart stems/filters to rerun in place.",
    )
    ap.add_argument(
        "--api_workers",
        type=int,
        default=4,
        help="Worker threads for API-based models (OpenAI/Claude). Ignored for local models.",
    )
    ap.add_argument("--api_timeout_sec", type=float, default=120.0)
    ap.add_argument("--api_max_retries", type=int, default=5)
    ap.add_argument("--api_retry_base_delay_sec", type=float, default=1.0)
    ap.add_argument("--api_retry_max_delay_sec", type=float, default=30.0)
    args = ap.parse_args()
    print(generate_predictions(args))


if __name__ == "__main__":
    main()
