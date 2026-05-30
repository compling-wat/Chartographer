import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from openai import OpenAI
from tqdm import tqdm

from common.artifacts import chart_code_dir, chart_data_dir, image_dir, revision_variant, split_artifact_dir
from common.datasets import get_dataset_config

from pipeline.reconstruction.diagnose_chart_issues import (
    _atomic_write,
    _chart_json_path_for_code,
    _encode_image,
    _find_code_file,
    _list_code_stems,
    _read_text,
    build_reference_index,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------
# Prompt (revise)
# ---------------------------------------------------------------------

REVISE_PROMPT = r"""
INPUTS
Given these inputs:

1) The original reference chart image (ground truth).
2) A chart image produced by the current code.
3) The current Python code (included only when revise scope requires code edits).
4) The current chart data JSON used by the current code (included only when revise scope requires data edits).
5) A list of diagnosed issues with the current code.

TASK
Revise the chart artifacts required by scope so the generated chart matches the original reference chart as closely as possible.

ALLOWED LIBRARIES
- Standard library + numpy + pandas + matplotlib + seaborn + cartopy.
- scipy and scikit-learn are allowed for data generation/processing.
- No external data/files/API calls.

RULES
- Address every item in diagnosed issues. Do not add unrelated changes.
- If multiple fixes conflict, prioritize: data > structure > readability > labels > aesthetics.
- Use each issue's "targets" field to decide whether to modify chart_code, chart_data, or both.
- Keep the code compatible with the matching chart JSON schema in chart_data/<variant>/<chart>.json.
- If an issue cannot be addressed due to missing data or code limitations, leave that part unchanged.
- Revised `chart_code`: imports + one top-level `make_figure(data, savepath=None)` only; require `data`, no hardcoded semantic data, no extra top-level defs, no `plt.show()`, and save with DPI>=300 + `bbox_inches='tight'` when `savepath` is provided.
- Keep visual/layout choices in `chart_code`, and keep semantic values in `chart_data`.

OUTPUT RULES
- Return only valid JSON.
- No markdown, no prose, no extra keys.
- Use the scope-specific JSON schema given at the end of the prompt.
"""

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _sanitize_code(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        i = s.find("\n")
        if i != -1:
            s = s[i + 1 :]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip() + "\n"


def _extract_json_object(text: str) -> dict:
    s = (text or "").strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j != -1 and j > i:
        obj = json.loads(s[i : j + 1])
        if isinstance(obj, dict):
            return obj

    raise ValueError("Could not parse JSON object from model output")


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_issue_list(issues_obj: Any) -> list:
    if isinstance(issues_obj, dict):
        issues = issues_obj.get("issues", [])
        return issues if isinstance(issues, list) else []
    return []


def _normalize_issue_target(value: Any) -> str:
    v = str(value or "").strip().lower()
    if v in {"code", "data", "both"}:
        return v
    return "both"


def _revision_scope_from_issues(issue_list: list) -> str:
    has_code = False
    has_data = False
    for issue in issue_list:
        if not isinstance(issue, dict):
            has_code = True
            has_data = True
            continue
        target = _normalize_issue_target(issue.get("targets"))
        if target == "both":
            has_code = True
            has_data = True
        elif target == "code":
            has_code = True
        elif target == "data":
            has_data = True
    if has_code and has_data:
        return "both"
    if has_code:
        return "code"
    if has_data:
        return "data"
    return "both"


def _build_revision_prompt(scope: str) -> str:
    if scope == "code":
        return (
            REVISE_PROMPT
            + "\nOUTPUT SCHEMA\n"
            + '{\n  "chart_code": "full revised python source"\n}'
        )
    if scope == "data":
        return (
            REVISE_PROMPT
            + "\nOUTPUT SCHEMA\n"
            + '{\n  "chart_data": { "your_chart_json_keys_here": "your values here" }\n}'
        )
    return (
        REVISE_PROMPT
        + "\nOUTPUT SCHEMA\n"
        + '{\n  "chart_code": "full revised python source",\n  "chart_data": { "your_chart_json_keys_here": "your values here" }\n}'
    )


def _response_text(resp) -> str:
    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t.strip():
        return t
    return ""


def _responses_call_with_images_scoped(
    client: "OpenAI",
    model: str,
    prompt_text: str,
    ref_b64: str,
    gen_b64: str,
    temperature: float,
    *,
    current_code: Optional[str] = None,
    current_data_json: Optional[str] = None,
) -> str:
    content = [{"type": "input_text", "text": prompt_text}]
    if current_code is not None:
        content.append({"type": "input_text", "text": "\nCURRENT CODE:\n" + current_code})
    if current_data_json is not None:
        content.append({"type": "input_text", "text": "\nCURRENT CHART DATA JSON:\n" + current_data_json})
    content.extend(
        [
            {"type": "input_text", "text": "\nREFERENCE IMAGE\n"},
            {"type": "input_image", "image_url": f"data:image/png;base64,{ref_b64}", "detail": "high"},
            {"type": "input_text", "text": "\nGENERATED IMAGE\n"},
            {"type": "input_image", "image_url": f"data:image/png;base64,{gen_b64}", "detail": "high"},
        ]
    )
    resp = client.responses.create(
        model=model,
        store=False,
        temperature=temperature,
        input=[{"role": "user", "content": content}],
        text={"format": {"type": "text"}},
    )
    return _response_text(resp)


def _extract_revision_payload(text: str, current_code: str, current_data: Any, scope: str) -> Tuple[str, Any]:
    try:
        payload = _extract_json_object(text)
    except Exception:
        if scope in {"code", "both"}:
            return _sanitize_code(text), current_data
        return current_code, current_data

    if scope == "code":
        code = payload.get("chart_code", current_code)
        data = current_data
    elif scope == "data":
        code = current_code
        data = payload.get("chart_data", current_data)
    else:
        code = payload.get("chart_code")
        data = payload.get("chart_data", current_data)

    if not isinstance(code, str) or not code.strip():
        code = current_code
    return _sanitize_code(code), data


def revise_one(
    client: "OpenAI",
    model: str,
    ref_img: str,
    gen_img: str,
    code_path: str,
    chart_json_path: str,
    issues: dict,
    scope: str,
    temperature: float,
) -> Tuple[str, Any]:
    code = _read_text(code_path)
    current_data = _read_json(chart_json_path)
    data_json = json.dumps(current_data, ensure_ascii=False, indent=2)
    ref_b64 = _encode_image(ref_img)
    gen_b64 = _encode_image(gen_img)

    current_code = code if scope in {"code", "both"} else None
    current_data_json = data_json if scope in {"data", "both"} else None
    text = _responses_call_with_images_scoped(
        client=client,
        model=model,
        prompt_text=_build_revision_prompt(scope)
        + "\n\nDIAGNOSED ISSUES (must fix):\n"
        + json.dumps(issues, ensure_ascii=False),
        ref_b64=ref_b64,
        gen_b64=gen_b64,
        temperature=temperature,
        current_code=current_code,
        current_data_json=current_data_json,
    )
    return _extract_revision_payload(text, current_code=code, current_data=current_data, scope=scope)


def worker(
    client: "OpenAI",
    model: str,
    code_path: str,
    ref_index: Dict[str, str],
    chart_data_root: str,
    image_root: str,
    issues_root: str,
    revised_root: str,
    revised_data_root: str,
    temperature: float,
) -> Tuple[str, Optional[str]]:
    stem = os.path.splitext(os.path.basename(code_path))[0]

    out_code_path = os.path.join(revised_root, stem + ".py")
    out_chart_path = os.path.join(revised_data_root, stem + ".json")
    issues_path = os.path.join(issues_root, stem + ".issues.json")
    gen_img = os.path.join(image_root, stem + ".png")
    chart_json_path = _chart_json_path_for_code(code_path, chart_data_root)

    try:
        if stem not in ref_index:
            return stem, "no reference image"

        if not os.path.exists(gen_img):
            return stem, "missing generated image"

        if not os.path.exists(chart_json_path):
            return stem, "missing chart json"

        if not os.path.exists(issues_path):
            return stem, "missing issues json (run diagnose.py first)"

        # Skip only if both revised artifacts already exist.
        if os.path.exists(out_code_path) and os.path.exists(out_chart_path):
            return stem, None

        issues = _read_json(issues_path)
        issue_list = _extract_issue_list(issues)
        scope = _revision_scope_from_issues(issue_list)
        current_code = _sanitize_code(_read_text(code_path))
        current_data = _read_json(chart_json_path)

        # Pre-process no-issue cases locally to avoid unnecessary model calls.
        if not issue_list:
            os.makedirs(revised_root, exist_ok=True)
            os.makedirs(revised_data_root, exist_ok=True)
            _atomic_write(out_code_path, current_code)
            _atomic_write(out_chart_path, json.dumps(current_data, ensure_ascii=False, indent=2) + "\n")
            return stem, None

        revised_code, revised_data = revise_one(
            client=client,
            model=model,
            ref_img=ref_index[stem],
            gen_img=gen_img,
            code_path=code_path,
            chart_json_path=chart_json_path,
            issues=issues,
            scope=scope,
            temperature=temperature,
        )

        # Always emit both artifacts in next round directory.
        # For out-of-scope targets, carry current artifact forward unchanged.
        if scope == "code":
            revised_data = current_data
        elif scope == "data":
            revised_code = current_code

        os.makedirs(revised_root, exist_ok=True)
        os.makedirs(revised_data_root, exist_ok=True)
        _atomic_write(out_code_path, revised_code)
        _atomic_write(out_chart_path, json.dumps(revised_data, ensure_ascii=False, indent=2) + "\n")
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
    next_variant = revision_variant(round + 1)

    codes_root = str(chart_code_dir(base, variant))
    chart_data_root = str(chart_data_dir(base, variant))
    image_root = str(image_dir(base, variant))
    issues_root = str(base / "issues" / variant)
    revised_root = str(chart_code_dir(base, next_variant))
    revised_data_root = str(chart_data_dir(base, next_variant))

    if not os.path.isdir(codes_root):
        raise FileNotFoundError(codes_root)
    if not os.path.isdir(chart_data_root):
        raise FileNotFoundError(chart_data_root)
    if not os.path.isdir(image_root):
        raise FileNotFoundError(image_root)

    os.makedirs(revised_root, exist_ok=True)
    os.makedirs(revised_data_root, exist_ok=True)

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
        code_files = [
            os.path.join(codes_root, f) for f in os.listdir(codes_root) if f.endswith(".py")
        ]
        code_files.sort()
        if max_items is not None:
            code_files = code_files[:max_items]

    # NEW: pre-filter skips so the progress bar reflects real work (and moves predictably)
    to_process = []
    skipped = 0
    for p in code_files:
        stem = os.path.splitext(os.path.basename(p))[0]
        out_code_path = os.path.join(revised_root, stem + ".py")
        out_chart_path = os.path.join(revised_data_root, stem + ".json")
        if os.path.exists(out_code_path) and os.path.exists(out_chart_path):
            skipped += 1
        else:
            to_process.append(p)

    print(f"Codes total: {len(code_files)}")
    print(f"To revise: {len(to_process)}")
    print(f"Skipped (already exists): {skipped}")
    print(f"References: {len(ref_index)}")
    print(f"Chart data in: {chart_data_root}")
    print(f"Images in: {image_root}")
    print(f"Issues in: {issues_root}")
    print(f"Revised code out: {revised_root}")
    print(f"Revised data out: {revised_data_root}")
    print(f"Model: {model}")

    errors = 0
    failed_charts = []

    if len(to_process) == 0:
        print("Done. errors=0")
        raise SystemExit(0)

    if len(to_process) == 1:
        stem, err = worker(
            client,
            model,
            to_process[0],
            ref_index,
            chart_data_root,
            image_root,
            issues_root,
            revised_root,
            revised_data_root,
            temperature,
        )
        if err:
            errors = 1
            failed_charts.append(stem)
            print(f"[ERROR] {stem}: {err}")
        print(f"Done. errors={errors}")
        if failed_charts:
            failed_sorted = sorted(set(failed_charts))
            print(f"Failed charts ({len(failed_sorted)}): {', '.join(failed_sorted)}")
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
                revised_root,
                revised_data_root,
                temperature,
            )
            for p in to_process
        ]

        with tqdm(total=len(futures), unit="chart") as bar:
            for fut in as_completed(futures):
                stem, err = fut.result()
                if err:
                    errors += 1
                    failed_charts.append(stem)
                    tqdm.write(f"[ERROR] {stem}: {err}")
                bar.update(1)

    print(f"Done. errors={errors}")
    if failed_charts:
        failed_sorted = sorted(set(failed_charts))
        print(f"Failed charts ({len(failed_sorted)}): {', '.join(failed_sorted)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 2: Revise chart code using precomputed issues only (no diagnose, no rendering)."
    )

    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--round", type=int, default=0)

    parser.add_argument("--chart", default=None)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature for OpenAI API"
    )

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
