import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def _run(cmd: List[str], cwd: Path) -> int:
    started = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    elapsed = time.time() - started
    print(f"  exit={proc.returncode} elapsed={elapsed:.1f}s")
    if proc.returncode != 0:
        if proc.stdout:
            print("  --- stdout ---")
            print(proc.stdout.strip())
        if proc.stderr:
            print("  --- stderr ---")
            print(proc.stderr.strip())
    return proc.returncode


def _build_render_cmd(
    dataset: str,
    split: str,
    seed: int,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "pipeline.reconstruction.render_reconstructed_charts",
        "--dataset",
        dataset,
        "--split",
        split,
        "--seed",
        str(seed),
        "--format",
        "png",
        "--regen-data",
        "--suppress-warnings",
    ]
    return cmd


def _build_qa_cmd(dataset: str, split: str, seed: int) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "pipeline.qa.run_qa_modules",
        "--dataset",
        dataset,
        "--split",
        split,
        "--seed",
        str(seed),
    ]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render and evaluate seeded counterfactual variants over a seed range."
    )
    parser.add_argument("--dataset", required=True, help="Dataset key from config or HF dataset id")
    parser.add_argument("--split", required=True, help="Split (e.g. dev)")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=9)
    args = parser.parse_args()

    if args.seed_end < args.seed_start:
        raise ValueError("--seed-end must be >= --seed-start")

    src_root = Path(__file__).resolve().parents[2]
    render_fail = 0
    qa_fail = 0

    for seed in range(args.seed_start, args.seed_end + 1):
        print(f"=== seed {seed} ===", flush=True)

        render_cmd = _build_render_cmd(
            dataset=args.dataset,
            split=args.split,
            seed=seed,
        )
        print(" render:", " ".join(render_cmd))
        rc = _run(render_cmd, cwd=src_root)
        if rc != 0:
            render_fail += 1
        qa_cmd = _build_qa_cmd(
            dataset=args.dataset,
            split=args.split,
            seed=seed,
        )
        print(" qa    :", " ".join(qa_cmd))
        rc = _run(qa_cmd, cwd=src_root)
        if rc != 0:
            qa_fail += 1

    print("\n=== done ===", flush=True)
    print(f"seeds: {args.seed_start}..{args.seed_end}", flush=True)
    print(f"render_fail_count: {render_fail}", flush=True)
    print(f"qa_fail_count: {qa_fail}", flush=True)


if __name__ == "__main__":
    main()
