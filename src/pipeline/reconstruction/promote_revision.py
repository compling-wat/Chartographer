import argparse
import shutil
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from common.artifacts import RECONSTRUCTION_VARIANT, chart_code_dir, chart_data_dir, image_dir, revision_variant, split_artifact_dir
from common.datasets import get_dataset_config


IMAGE_EXTS = (".png", ".pdf", ".svg", ".jpg", ".jpeg", ".webp")


def split_dir(dataset: str, split: str) -> Path:
    cfg = get_dataset_config(dataset)
    local_dir = str(cfg.get("local_dir", dataset.split("/")[-1].lower()))
    return split_artifact_dir(local_dir, split)


def reset_dir(dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)


def copy_dir_contents(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src}")
    reset_dir(dst)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def copy_chart_file(src: Path, dst: Path, stem: str, suffix: str) -> None:
    source = src / f"{stem}{suffix}"
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dst / source.name)


def copy_chart_images(src: Path, dst: Path, stem: str) -> None:
    matches = [p for p in src.iterdir() if p.is_file() and p.stem == stem and p.suffix.lower() in IMAGE_EXTS] if src.is_dir() else []
    if not matches:
        raise FileNotFoundError(f"No rendered image found for {stem!r} in {src}")
    dst.mkdir(parents=True, exist_ok=True)
    for old in dst.iterdir():
        if old.is_file() and old.stem == stem and old.suffix.lower() in IMAGE_EXTS:
            old.unlink()
    for source in matches:
        shutil.copy2(source, dst / source.name)


def remove_matching_files(directory: Path, patterns: list[str]) -> None:
    if not directory.exists():
        return
    for pattern in patterns:
        for path in directory.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()


def remove_dir_if_empty(directory: Path) -> None:
    if directory.exists() and directory.is_dir() and not any(directory.iterdir()):
        directory.rmdir()


def clean_variant_dirs(root: Path, chart: str | None) -> None:
    if not root.exists():
        return
    for directory in root.glob("revision_*"):
        if not directory.is_dir():
            continue
        if chart is None:
            shutil.rmtree(directory)
        else:
            remove_matching_files(directory, [f"{chart}.*", f"{chart}__*.*"])
            remove_dir_if_empty(directory)


def clean_issues(base: Path, chart: str | None) -> None:
    issues_root = base / "issues"
    if not issues_root.exists():
        return
    issue_dirs = [issues_root / RECONSTRUCTION_VARIANT, *issues_root.glob("revision_*")]
    for directory in issue_dirs:
        if not directory.exists():
            continue
        if chart is None:
            shutil.rmtree(directory)
        else:
            remove_matching_files(directory, [f"{chart}.issues.json", f"{chart}.*"])
            remove_dir_if_empty(directory)
    remove_dir_if_empty(issues_root)


def clean_derived_outputs(base: Path, chart: str | None) -> None:
    derived_dirs = [base / "assumptions", base / "generate_data", base / "question_adapters", base / "generate_answers"]
    if chart is None:
        for directory in derived_dirs:
            if directory.exists():
                shutil.rmtree(directory)
        for path in base.glob("qa_*.json"):
            path.unlink()
        html = base / "visualize_chartographer.html"
        if html.exists():
            html.unlink()
        return

    remove_matching_files(base / "assumptions", [f"{chart}.json"])
    remove_matching_files(base / "generate_data", [f"{chart}.py"])
    remove_matching_files(base / "question_adapters", [f"{chart}__*.py"])
    remove_matching_files(base / "generate_answers", [f"{chart}__*.py"])
    remove_matching_files(base / "generate_data" / "__pycache__", [f"{chart}.*.pyc"])
    remove_matching_files(base / "question_adapters" / "__pycache__", [f"{chart}__*.pyc"])
    remove_matching_files(base / "generate_answers" / "__pycache__", [f"{chart}__*.pyc"])
    for path in base.glob("qa_*.json"):
        path.unlink()
    html = base / "visualize_chartographer.html"
    if html.exists():
        html.unlink()


def clean_seed_outputs(base: Path, chart: str | None) -> None:
    for root in (base / "chart_data", base / "images"):
        if not root.exists():
            continue
        for directory in root.glob("seed_*"):
            if not directory.is_dir():
                continue
            if chart is None:
                shutil.rmtree(directory)
            else:
                remove_matching_files(directory, [f"{chart}.*"])
                remove_dir_if_empty(directory)


def promote_revision(args: argparse.Namespace) -> None:
    if args.round < 1:
        raise ValueError("--round must be >= 1; round 0 is already reconstruction")

    base = split_dir(args.dataset, args.split)
    source_variant = revision_variant(args.round)
    chart = Path(args.chart).stem if args.chart else None

    source_code = chart_code_dir(base, source_variant)
    source_data = chart_data_dir(base, source_variant)
    source_images = image_dir(base, source_variant)
    target_code = chart_code_dir(base, RECONSTRUCTION_VARIANT)
    target_data = chart_data_dir(base, RECONSTRUCTION_VARIANT)
    target_images = image_dir(base, RECONSTRUCTION_VARIANT)

    if chart is None:
        copy_dir_contents(source_code, target_code)
        copy_dir_contents(source_data, target_data)
        copy_dir_contents(source_images, target_images)
    else:
        copy_chart_file(source_code, target_code, chart, ".py")
        copy_chart_file(source_data, target_data, chart, ".json")
        copy_chart_images(source_images, target_images, chart)

    if args.clean_revisions:
        clean_variant_dirs(base / "chart_code", chart)
        clean_variant_dirs(base / "chart_data", chart)
        clean_variant_dirs(base / "images", chart)
        clean_issues(base, chart)

    if args.clean_derived:
        clean_derived_outputs(base, chart)
        clean_seed_outputs(base, chart)

    scope = chart or "all charts"
    print(f"Promoted {source_variant} to {RECONSTRUCTION_VARIANT}: {scope}")
    if args.clean_revisions:
        print("Cleaned revision and issue artifacts.")
    if args.clean_derived:
        print("Cleaned downstream generated artifacts; rerun assumptions/data/QA/seed steps.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a revision_N artifact back into reconstruction.")
    parser.add_argument("--dataset", required=True, help="Dataset key from config or HF dataset id")
    parser.add_argument("--split", required=True, help="Dataset split")
    parser.add_argument("--round", type=int, required=True, help="Revision round to promote, e.g. 1 for revision_1")
    parser.add_argument("--chart", default=None, help="Optional chart stem or filename to promote only one chart")
    parser.add_argument("--clean-revisions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean-derived", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    promote_revision(parse_args())
