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

from common.artifacts import chart_code_dir, chart_data_dir, image_dir, revision_variant, split_artifact_dir
from common.datasets import get_dataset_config, load_split_basenames

SRC_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------
# Prompt (diagnose)
# ---------------------------------------------------------------------

DIAGNOSE_PROMPT = r"""
TASK
Given current code, current chart JSON, generated image, and original image, compare generated vs original.
Report only fix-worthy differences.

PRIORITY ORDER
1) data faithfulness
2) structure (chart type, mappings, scale/range, ordering, legend semantics)
3) readability/layout (prioritize visual overlap/collision among legend, labels, ticks, annotations, and plotted marks)
4) labels/annotations
5) aesthetics only when misleading

SPARSITY RULES
- Max 6 issues (typically 0-2).
- Ignore cosmetic drift (fonts/theme/antialiasing/minor spacing or shade changes).
- Ignore tiny numeric drift when trend/ranking is preserved.
- Merge duplicate root causes.
- If uncertain, omit.

OUTPUT RULES
- Return only valid JSON.
- No markdown, no prose, no extra keys.
- Use this exact schema:
{
  "issues": [
    {
      "category": "data|structure|readability|labels|aesthetics",
      "targets": "code|data|both",
      "severity": "low|medium|high",
      "summary": "one sentence",
      "details": ["concise visual evidence", "..."],
      "suggested_fix": "short actionable directive"
    }
  ]
}

RULES
- Original is the source of truth.
- For `targets`, use:
  - `data` for JSON values/labels/categories
  - `code` for plotting/layout/style logic
  - `both` for coordinated code and JSON changes
- Keep summaries/fixes specific and directly actionable.
- If no fix-worthy issue, return: {"issues": []}
"""

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def dataset_to_local_root(dataset: str) -> str:
    cfg = get_dataset_config(dataset)
    repo = str(cfg.get("local_dir", dataset.split("/")[-1].lower()))
    return str(REPO_ROOT / "data" / repo)


def list_local_images(data_root: str) -> List[str]:
    images: List[str] = []
    for root, _, files in os.walk(data_root):
        for fn in files:
            if fn.lower().endswith(_IMAGE_EXTS):
                images.append(os.path.join(root, fn))
    return images


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _atomic_write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp.{threading.get_ident()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)


def _normalize_chart_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return name
    return name[:-3] if name.endswith(".py") else name


def _find_code_file(codes_root: str, chart: str) -> Optional[str]:
    stem = _normalize_chart_name(chart)
    if not stem:
        return None
    p = os.path.join(codes_root, stem + ".py")
    return p if os.path.exists(p) else None


def _chart_json_path_for_code(code_path: str, chart_json_root: str) -> str:
    stem = os.path.splitext(os.path.basename(code_path))[0]
    return os.path.join(chart_json_root, stem + ".json")


def _list_code_stems(codes_root: str) -> List[str]:
    if not os.path.isdir(codes_root):
        return []
    out: List[str] = []
    for fn in sorted(os.listdir(codes_root)):
        if fn.endswith(".py") and fn != "__init__.py":
            out.append(os.path.splitext(fn)[0])
    return out


def _extract_json(text: str) -> dict:
    s = (text or "").strip()
    try:
        return json.loads(s)
    except Exception:
        pass

    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j != -1 and j > i:
        return json.loads(s[i : j + 1])

    raise ValueError("Could not parse JSON from model output")


def _response_text(resp) -> str:
    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t.strip():
        return t

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


def responses_call_with_images(
    client: OpenAI,
    model: str,
    prompt_text: str,
    current_code: str,
    current_data_json: str,
    ref_b64: str,
    gen_b64: str,
    temperature: float,
) -> str:
    resp = client.responses.create(
        model=model,
        store=False,
        temperature=temperature,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_text},
                    {"type": "input_text", "text": "\nCURRENT CODE:\n" + current_code},
                    {"type": "input_text", "text": "\nCURRENT CHART DATA JSON:\n" + current_data_json},
                    {"type": "input_text", "text": "\nREFERENCE IMAGE\n"},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{ref_b64}", "detail": "high"},
                    {"type": "input_text", "text": "\nGENERATED IMAGE\n"},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{gen_b64}", "detail": "high"},
                ],
            }
        ],
        text={"format": {"type": "text"}},
    )
    return _response_text(resp)


# ---------------------------------------------------------------------
# Reference index
# ---------------------------------------------------------------------

def build_reference_index(dataset: str, split: str) -> Dict[str, str]:
    root = dataset_to_local_root(dataset)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Missing dataset folder: {root}")

    split_names = set(load_split_basenames(dataset, split))
    local_images = list_local_images(root)

    idx: Dict[str, str] = {}
    for p in local_images:
        bn = os.path.basename(p)
        if bn in split_names:
            stem = os.path.splitext(bn)[0]
            idx.setdefault(stem, p)

    if not idx:
        raise RuntimeError("No reference images matched split")

    return idx


# ---------------------------------------------------------------------
# Diagnose
# ---------------------------------------------------------------------

def diagnose_one(
    client: OpenAI,
    model: str,
    ref_img: str,
    gen_img: str,
    code_path: str,
    chart_json_path: str,
    temperature: float,
) -> dict:
    code = _read_text(code_path)
    data_json = _read_text(chart_json_path)
    ref_b64 = _encode_image(ref_img)
    gen_b64 = _encode_image(gen_img)
    parse_err: Optional[Exception] = None
    prompt_variants = [
        DIAGNOSE_PROMPT,
        DIAGNOSE_PROMPT
        + "\n\nIMPORTANT: Return a single strict JSON object only."
        + " Ensure valid JSON escaping for all strings."
        + " No markdown fences, comments, or trailing text.",
    ]
    for prompt in prompt_variants:
        text = responses_call_with_images(
            client=client,
            model=model,
            prompt_text=prompt,
            current_code=code,
            current_data_json=data_json,
            ref_b64=ref_b64,
            gen_b64=gen_b64,
            temperature=temperature,
        )
        try:
            return _extract_json(text)
        except Exception as e:
            parse_err = e
            continue
    raise ValueError(f"Failed to parse model JSON after retry: {parse_err}")


def worker(
    client: OpenAI,
    model: str,
    code_path: str,
    ref_index: Dict[str, str],
    chart_data_root: str,
    image_root: str,
    issues_root: str,
    temperature: float,
) -> Tuple[str, Optional[str]]:
    stem = os.path.splitext(os.path.basename(code_path))[0]
    issues_path = os.path.join(issues_root, stem + ".issues.json")
    chart_json_path = _chart_json_path_for_code(code_path, chart_data_root)

    try:
        # No overwrite flag: always skip if issues already exist
        if os.path.exists(issues_path):
            return stem, None

        ref = ref_index.get(stem)
        if not ref:
            return stem, "no reference image"

        gen_img = os.path.join(image_root, stem + ".png")
        if not os.path.exists(gen_img):
            return stem, "missing generated image"

        if not os.path.exists(chart_json_path):
            return stem, "missing chart json"

        issues = diagnose_one(
            client=client,
            model=model,
            ref_img=ref,
            gen_img=gen_img,
            code_path=code_path,
            chart_json_path=chart_json_path,
            temperature=temperature,
        )

        _atomic_write(issues_path, json.dumps(issues, ensure_ascii=False, indent=2) + "\n")
        return stem, None

    except Exception as e:
        return stem, str(e)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(
    dataset: str,
    split: str,
    round: int,
    model: str,
    max_workers: int,
    max_items: Optional[int],
    chart: Optional[str],
    temperature: float,
) -> None:
    cfg = get_dataset_config(dataset)
    repo = str(cfg.get("local_dir", dataset.split("/")[-1].lower()))
    base = split_artifact_dir(repo, split)
    variant = revision_variant(round)

    codes_root = str(chart_code_dir(base, variant))
    chart_data_root = str(chart_data_dir(base, variant))
    image_root = str(image_dir(base, variant))
    issues_root = str(base / "issues" / variant)

    if not os.path.isdir(codes_root):
        raise FileNotFoundError(codes_root)
    if not os.path.isdir(chart_data_root):
        raise FileNotFoundError(chart_data_root)
    if not os.path.isdir(image_root):
        raise FileNotFoundError(image_root)

    os.makedirs(issues_root, exist_ok=True)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    ref_index = build_reference_index(dataset, split)

    # Select code files
    if chart:
        code_path = _find_code_file(codes_root, chart)
        if code_path is None:
            print(f"Chart file not found: {chart}")
            stems = _list_code_stems(codes_root)
            if stems:
                print("Available charts:")
                for s in stems:
                    print("  -", s)
            raise SystemExit(1)
        code_files = [code_path]
    else:
        code_files = [os.path.join(codes_root, f) for f in os.listdir(codes_root) if f.endswith(".py")]
        code_files.sort()
        if max_items is not None:
            code_files = code_files[:max_items]

    print(f"Codes: {len(code_files)}")
    print(f"References: {len(ref_index)}")
    print(f"Chart data: {chart_data_root}")
    print(f"Images: {image_root}")
    print(f"Issues out: {issues_root}")
    print(f"Model: {model}")

    errors = 0

    if len(code_files) == 1:
        stem, err = worker(
            client=client,
            model=model,
            code_path=code_files[0],
            ref_index=ref_index,
            chart_data_root=chart_data_root,
            image_root=image_root,
            issues_root=issues_root,
            temperature=temperature,
        )
        if err:
            errors = 1
            print(f"[ERROR] {stem}: {err}")
        print(f"Done. errors={errors}")
        raise SystemExit(0 if errors == 0 else 1)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                worker,
                client,
                model,
                p,
                ref_index,
                chart_data_root,
                image_root,
                issues_root,
                temperature,
            )
            for p in code_files
        ]

        with tqdm(total=len(futures), unit="chart") as bar:
            for fut in as_completed(futures):
                stem, err = fut.result()
                if err:
                    errors += 1
                    tqdm.write(f"[ERROR] {stem}: {err}")
                bar.update(1)

    print(f"Done. errors={errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1: Diagnose issues in generated charts vs reference charts (Responses API). Writes *.issues.json files."
    )

    parser.add_argument("--model-name", required=True, help="OpenAI model name (must support images)")
    parser.add_argument("--dataset", required=True, help="Dataset key from config or HF dataset id")
    parser.add_argument("--split", required=True, help="Split name, e.g. train / validation / test")
    parser.add_argument("--round", type=int, default=0)

    parser.add_argument("--chart", default=None, help="Chart name (stem or .py). If omitted, diagnose ALL charts.")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for OpenAI API")

    args = parser.parse_args()

    main(
        dataset=args.dataset,
        split=args.split,
        round=args.round,
        model=args.model_name,
        max_workers=args.max_workers,
        max_items=args.max_items,
        chart=args.chart,
        temperature=args.temperature,
    )
