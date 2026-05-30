# run_chart.py
import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import traceback
import warnings
from pathlib import Path
from typing import List, Optional

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

# Ensure Matplotlib has a writable config/cache directory in constrained environments.
if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path(tempfile.gettempdir()) / "matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import matplotlib.pyplot as plt

from common.artifacts import (
    chart_code_dir,
    chart_data_dir,
    image_dir,
    revision_variant,
    seed_variant,
    split_artifact_dir,
)
from common.datasets import get_dataset_config

REPO_ROOT = Path(__file__).resolve().parents[3]


def split_dir(dataset: str, split: str) -> Path:
    dataset_cfg = get_dataset_config(dataset)
    return split_artifact_dir(str(dataset_cfg.get("local_dir", dataset.split("/")[-1].lower())), split)


def load_module(path: Path):
    """Load a Python file as a module."""
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


def find_chart_file(dataset: str, split: str, name: str, round: int = 0) -> Optional[Path]:
    """Find a chart-code module by name. Returns None if not found."""
    if not name.endswith(".py"):
        name += ".py"

    path = chart_code_dir(split_dir(dataset, split), revision_variant(round)) / name
    return path if path.exists() else None


def list_chart_files(dataset: str, split: str, round: int = 0) -> List[Path]:
    """List all runnable chart-code modules for a split/variant."""
    code_dir = chart_code_dir(split_dir(dataset, split), revision_variant(round))
    files = sorted(code_dir.glob("*.py"))
    return [f for f in files if f.name not in {"run_chart.py", "__init__.py"}]


def find_chart_data_file(dataset: str, split: str, name: str, round: int = 0) -> Optional[Path]:
    """Find the chart-data JSON for a chart module."""
    if not name.endswith(".json"):
        name += ".json"
    path = chart_data_dir(split_dir(dataset, split), revision_variant(round)) / name
    return path if path.exists() else None


def _normalize_chart_payload(payload):
    """Accept either full chart JSON or bare chart_data object."""
    if isinstance(payload, dict):
        return payload.get("chart_data", payload)
    return payload


def _call_generate_data(generate_fn, data_template, seed: int):
    """Call the public generate_data(data_template, seed=...) contract."""
    return generate_fn(data_template, seed=seed)


# ---------------------------------------------------------------------
# generate_data lookup
# ---------------------------------------------------------------------

def _generate_data_dir(dataset: str, split: str, round: int) -> Path:
    """Location of seedable data-generation modules."""
    return split_dir(dataset, split) / "generate_data"


def find_generate_file(dataset: str, split: str, name: str, round: int) -> Optional[Path]:
    """Find a generate_data module by chart name. Returns None if not found."""
    if not name.endswith(".py"):
        name += ".py"
    d = _generate_data_dir(dataset, split, round)
    p = d / name
    return p if p.exists() else None


def list_generate_files(dataset: str, split: str, round: int) -> List[Path]:
    """List available generate_data modules."""
    d = _generate_data_dir(dataset, split, round)
    if not d.exists():
        return []
    files = sorted(d.glob("*.py"))
    return [p for p in files if p.name not in {"__init__.py"}]


# ---------------------------------------------------------------------
# Warning + stderr capture (to attribute warnings to chart files)
# ---------------------------------------------------------------------

@contextlib.contextmanager
def capture_warnings_and_stderr():
    """
    Capture:
      - Python warnings (warnings.warn -> e.g., Matplotlib glyph warnings)
      - stderr output (GTK warnings usually go here)

    Yields:
      (warn_records, stderr_buffer)
    """
    with warnings.catch_warnings(record=True) as warn_records:
        warnings.simplefilter("always")  # collect all warnings, even if repeated
        stderr_buf = io.StringIO()
        with contextlib.redirect_stderr(stderr_buf):
            yield warn_records, stderr_buf


def report_captured(
    warn_records,
    stderr_buf,
    label: str,
    file_path: Path,
    *,
    suppress_warnings: bool = False,
):
    """
    Print captured warnings/stderr tagged with the chart/generate file being run.
    """
    stderr_text = stderr_buf.getvalue().strip()
    if suppress_warnings:
        return
    if not warn_records and not stderr_text:
        return

    print(f"\n=== Warnings while running {label} for: {file_path.name} ===")

    # Python warnings
    for w in warn_records:
        cat = w.category.__name__
        msg = str(w.message).strip()
        # w.filename/w.lineno is typically where the warning originated (often inside matplotlib)
        # Header above tells you which chart file was executing when it happened.
        print(f"[PY WARNING] {cat}: {msg}")
        print(f"            issued at: {w.filename}:{w.lineno}")

    # stderr (GTK etc.)
    if stderr_text:
        for line in stderr_text.splitlines():
            line = line.strip()
            if line:
                print(f"[STDERR] {line}")


# ---------------------------------------------------------------------
# Error context helpers
# ---------------------------------------------------------------------

def _read_lines(path: Path) -> List[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return path.read_text(errors="replace").splitlines()
    except Exception:
        return []


def _format_snippet(path: Path, line_no: int, context: int = 3) -> str:
    """
    Return a small code snippet around 1-based line_no.
    """
    lines = _read_lines(path)
    if not lines:
        return f"(Could not read file for context: {path})"

    n = len(lines)
    line_no = max(1, min(line_no, n))
    start = max(1, line_no - context)
    end = min(n, line_no + context)

    width = len(str(end))
    out = [f"  File context: {path} (around line {line_no})"]
    for ln in range(start, end + 1):
        prefix = "-->" if ln == line_no else "   "
        text = lines[ln - 1].rstrip("\n").replace("\t", "    ")
        out.append(f"  {prefix} {ln:>{width}} | {text}")
    return "\n".join(out)


def _extract_relevant_lines(e: BaseException, target: Path, max_hits: int = 3) -> List[int]:
    """
    Find line numbers in traceback that correspond to `target`.
    Returns up to `max_hits` line numbers (most recent first).
    """
    target_resolved = str(target.resolve())

    if isinstance(e, SyntaxError):
        if getattr(e, "filename", None):
            try:
                if str(Path(e.filename).resolve()) == target_resolved and getattr(e, "lineno", None):
                    return [int(e.lineno)]
            except Exception:
                pass
        if getattr(e, "lineno", None):
            return [int(e.lineno)]

    tb = e.__traceback__
    if tb is None:
        return []

    frames = traceback.extract_tb(tb)
    hits: List[int] = []
    for fr in reversed(frames):
        try:
            if str(Path(fr.filename).resolve()) == target_resolved and fr.lineno:
                hits.append(int(fr.lineno))
        except Exception:
            if fr.filename == str(target) and fr.lineno:
                hits.append(int(fr.lineno))
        if len(hits) >= max_hits:
            break

    if not hits and frames:
        last = frames[-1]
        try:
            if last.filename and last.lineno and Path(last.filename).suffix == ".py":
                if str(Path(last.filename).resolve()) == target_resolved:
                    hits.append(int(last.lineno))
        except Exception:
            pass

    return hits


def _format_exception_with_context(e: BaseException, chart_file: Path) -> str:
    """
    Return a readable failure report with traceback + code snippets from chart_file.
    """
    exc_name = type(e).__name__
    msg = str(e).strip()
    header = f"{exc_name}: {msg}" if msg else exc_name

    tb_text = "".join(traceback.format_exception(type(e), e, e.__traceback__)).rstrip()

    line_hits = _extract_relevant_lines(e, chart_file)
    snippets: List[str] = []
    for ln in line_hits:
        snippets.append(_format_snippet(chart_file, ln))

    parts = [header, tb_text]
    if snippets:
        parts.append("\n".join(snippets))
    return "\n\n".join(parts)


def run_one(chart_file: Path, args, outdir: Path) -> bool:
    """
    Run a single chart file. Returns True if succeeded, False otherwise.

    Logging behavior:
      - Only prints the "=== Using: X ===" banner when the chart FAILS/SKIPs.
      - Successful charts produce no per-chart banner output.
      - Warnings (Python + stderr like GTK) are printed and tagged to the file being run.
    """

    def fail(msg: str, e: Optional[BaseException] = None) -> bool:
        print(f"\n=== Using: {chart_file.name} ===")
        if e is None:
            print(msg)
        else:
            print(msg)
            print(_format_exception_with_context(e, chart_file))
        return False

    try:
        mod = load_module(chart_file)
    except Exception as e:
        return fail("[FAIL] Could not import.", e)

    if not hasattr(mod, "make_figure"):
        return fail("[SKIP] Missing make_figure()")

    base_name = chart_file.stem
    generated_data_dir = chart_data_dir(split_dir(args.dataset, args.split), seed_variant(args.seed))
    json_path = generated_data_dir / f"{base_name}.json"

    if not args.regen_data:
        chart_data_file = find_chart_data_file(args.dataset, args.split, base_name, args.round)
        if chart_data_file is None:
            expected = chart_data_dir(split_dir(args.dataset, args.split), revision_variant(args.round)) / f"{base_name}.json"
            return fail(f"[SKIP] Missing chart data JSON.\n  Expected: {expected}")
        try:
            with chart_data_file.open("r", encoding="utf-8") as f:
                data = _normalize_chart_payload(json.load(f))
        except Exception as e:
            return fail(f"[FAIL] Could not load chart data JSON: {chart_data_file}", e)

    # If regen-data: load generate_data from external module
    if args.regen_data:
        chart_data_file = find_chart_data_file(args.dataset, args.split, base_name, args.round)
        if chart_data_file is None:
            expected = chart_data_dir(split_dir(args.dataset, args.split), revision_variant(args.round)) / f"{base_name}.json"
            return fail(f"[SKIP] Missing chart data JSON for template.\n  Expected: {expected}")
        try:
            with chart_data_file.open("r", encoding="utf-8") as f:
                data_template = _normalize_chart_payload(json.load(f))
        except Exception as e:
            return fail(f"[FAIL] Could not load chart data template JSON: {chart_data_file}", e)

        gen_file = find_generate_file(args.dataset, args.split, base_name, args.round)
        if gen_file is None:
            msg_lines = [
                "[SKIP] Missing generate_data module for this chart.",
                f"  Expected: { _generate_data_dir(args.dataset, args.split, args.round) / (base_name + '.py') }",
                "  Available generate_data modules (stems):",
            ]
            avail = list_generate_files(args.dataset, args.split, args.round)
            if avail:
                for p in avail[:80]:
                    msg_lines.append(f"    - {p.stem}")
                if len(avail) > 80:
                    msg_lines.append(f"    ... ({len(avail) - 80} more)")
            else:
                msg_lines.append("    (none)")
            return fail("\n".join(msg_lines))

        try:
            gen_mod = load_module(gen_file)
        except Exception as e:
            return fail(f"[FAIL] Could not import generate_data module: {gen_file}", e)

        if not hasattr(gen_mod, "generate_data"):
            return fail(f"[SKIP] {gen_file} is missing generate_data()")

        try:
            with capture_warnings_and_stderr() as (wrec, serr):
                generated_payload = _call_generate_data(
                    gen_mod.generate_data,
                    data_template,
                    seed=args.seed,
                )
                data = _normalize_chart_payload(generated_payload)
            report_captured(
                wrec,
                serr,
                "generate_data()",
                gen_file,
                suppress_warnings=args.suppress_warnings,
            )

        except Exception as e:
            print(f"\n=== Using: {chart_file.name} ===")
            print("[FAIL] generate_data() crashed.")
            try:
                print(_format_exception_with_context(e, gen_file))
            except Exception:
                print("".join(traceback.format_exception(type(e), e, e.__traceback__)).rstrip())
            return False

        try:
            generated_data_dir.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(generated_payload, f, indent=2)
        except Exception as e:
            return fail("[FAIL] Writing JSON failed.", e)

    # Build figure path
    fig_path = outdir / f"{base_name}"
    fig_path = fig_path.with_suffix(f".{args.format}")

    # Call make_figure
    try:
        with capture_warnings_and_stderr() as (wrec, serr):
            mod.make_figure(
                data=data,
                savepath=str(fig_path),
            )
            plt.close("all")

        report_captured(
            wrec,
            serr,
            "make_figure()",
            chart_file,
            suppress_warnings=args.suppress_warnings,
        )

    except Exception as e:
        return fail("[FAIL] make_figure() crashed.", e)

    return True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True, help="Dataset key from config or HF dataset id.")
    parser.add_argument("--split", required=True, help="Split name (e.g. 'test').")
    parser.add_argument(
        "--chart",
        required=False,
        default=None,
        help="Chart file name. If omitted, run all chart-code modules for the selected variant.",
    )
    parser.add_argument("--round", type=int, default=0, help="Revision round number.")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])

    parser.add_argument(
        "--regen-data",
        action="store_true",
        help="Regenerate data from generate_data/, save it under chart_data/seed_N, and render images/seed_N.",
    )
    parser.add_argument(
        "--suppress-warnings",
        "--supress-warning",
        action="store_true",
        dest="suppress_warnings",
        help="Suppress captured warnings/stderr output and only print failures.",
    )

    args = parser.parse_args()

    render_variant = seed_variant(args.seed) if args.regen_data else revision_variant(args.round)
    outdir = image_dir(split_dir(args.dataset, args.split), render_variant)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.chart:
        chart_file = find_chart_file(args.dataset, args.split, args.chart, args.round)

        if chart_file is None:
            print("Chart file not found.\nAvailable charts:")
            for f in list_chart_files(args.dataset, args.split, args.round):
                print("  -", f.stem)
            raise SystemExit(1)

        ok = run_one(chart_file, args, outdir)
        raise SystemExit(0 if ok else 1)

    chart_files = list_chart_files(args.dataset, args.split, args.round)
    if not chart_files:
        print("No chart-code files found.")
        raise SystemExit(1)

    failed_charts = []
    for f in chart_files:
        ok = run_one(f, args, outdir)
        if not ok:
            failed_charts.append(f.stem)

    if failed_charts:
        print("\nFailed charts:")
        for name in failed_charts:
            print("  -", name)

    raise SystemExit(0 if not failed_charts else 1)


if __name__ == "__main__":
    main()
