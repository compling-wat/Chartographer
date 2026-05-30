import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from common.artifacts import chart_data_dir, image_dir, seed_variant, split_artifact_dir
from common.datasets import (
    example_image_basename,
    get_dataset_config,
    load_examples,
    resolve_column,
    resolve_repo_local_path,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_seeds(value: str) -> List[int]:
    seeds: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            if end < start:
                raise ValueError(f"Invalid seed range: {part}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(part))
    return list(dict.fromkeys(seeds))


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return [dict(row) for row in data]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.link(src, dst)
        except OSError:
            dst.symlink_to(src)


def source_image_path(row: Dict[str, Any], cfg: Dict[str, Any], dataset: str) -> Optional[Path]:
    image_name = example_image_basename(row, cfg)
    candidates: List[Path] = []
    for key in (
        cfg.get("image_col"),
        cfg.get("figure_path_col"),
        cfg.get("original_figure_path_col"),
        "image",
        "figure_path",
        "original_figure_path",
    ):
        if not isinstance(key, str):
            continue
        value = row.get(key)
        if isinstance(value, dict):
            value = value.get("path") or value.get("filename")
        if isinstance(value, str) and value:
            resolved = resolve_repo_local_path(value)
            if resolved is not None:
                return resolved
            candidates.append(Path(value))

    local_root = REPO_ROOT / "data" / dataset
    for value in candidates:
        for candidate in (local_root / value, local_root / "images" / value.name):
            if candidate.exists():
                return candidate
    if image_name:
        direct = local_root / "images" / image_name
        if direct.exists():
            return direct
    return None


def row_stem(row: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    explicit = str(row.get("_image_stem") or "").strip()
    if explicit:
        return explicit
    return Path(example_image_basename(row, cfg)).stem


def chart_ids_for_rows(rows: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, str]:
    stems = sorted({row_stem(row, cfg) for row in rows if row_stem(row, cfg)})
    return {stem: f"c{i:06d}" for i, stem in enumerate(stems)}


def qa_rows_by_question_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        qid = str(row.get("question_id") or "").strip()
        if qid:
            out[qid] = row
    return out


def qa_rows_by_key(rows: List[Dict[str, Any]], cfg: Dict[str, Any], question_col: str) -> Dict[tuple[str, str], Dict[str, Any]]:
    out: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        stem = row_stem(row, cfg)
        question = str(row.get(question_col, "")).strip()
        if stem and question:
            out[(stem, question)] = row
    return out


def family_member_row(
    *,
    base_row: Dict[str, Any],
    member_source: Dict[str, Any],
    variant: str,
    chart_id: str,
    question_id: str,
    question_col: str,
    answer_col: str,
    image_rel: str,
    chart_data_rel: str,
    source_dataset_split: str,
    source_row_index: int,
) -> Dict[str, Any]:
    return {
        "chart_id": chart_id,
        "question_id": question_id,
        "variant": variant,
        "image": image_rel,
        "question": str(member_source.get(question_col, base_row.get(question_col, ""))),
        "answer": str(member_source.get(answer_col, base_row.get(answer_col, ""))),
        "chart_data": chart_data_rel,
        "source_dataset_split": source_dataset_split,
        "source_row_index": source_row_index,
    }


def export_chart_question_families(args: argparse.Namespace) -> Path:
    ds_cfg = get_dataset_config(args.dataset)
    source_rows = [dict(row) for row in load_examples(args.dataset, args.split)]
    question_col = resolve_column(ds_cfg, source_rows, "question_col", "question_cols")
    answer_col = resolve_column(ds_cfg, source_rows, "answer_col", "answer_cols")

    dataset_local_dir = str(ds_cfg["local_dir"])
    split_dir = split_artifact_dir(dataset_local_dir, args.split)
    out_root = REPO_ROOT / "data" / args.output_dataset
    out_split = args.output_split or args.split

    chart_ids = chart_ids_for_rows(source_rows, ds_cfg)
    question_ids = {idx: f"q{idx:06d}" for idx in range(len(source_rows))}
    source_dataset_split = args.source_dataset_split or f"{args.dataset}_{args.split}"

    rows_out: List[Dict[str, Any]] = []
    if args.include_original:
        for idx, row in enumerate(source_rows):
            stem = row_stem(row, ds_cfg)
            if not stem or stem not in chart_ids:
                continue
            src_image = source_image_path(row, ds_cfg, args.dataset)
            image_rel = ""
            if src_image is not None:
                ext = src_image.suffix or ".png"
                image_rel = f"images/original/{chart_ids[stem]}{ext}"
                link_or_copy(src_image, out_root / image_rel, args.copy)
            rows_out.append(
                family_member_row(
                    base_row=row,
                    member_source=row,
                    variant="original",
                    chart_id=chart_ids[stem],
                    question_id=question_ids[idx],
                    question_col=question_col,
                    answer_col=answer_col,
                    image_rel=image_rel,
                    chart_data_rel="",
                    source_dataset_split=source_dataset_split,
                    source_row_index=idx,
                )
            )

    variants = ["reconstruction", *[seed_variant(seed) for seed in parse_seeds(args.seeds)]]
    for variant in variants:
        qa_path = split_dir / f"qa_{variant}.json"
        qa_rows = load_json_list(qa_path)
        by_qid = qa_rows_by_question_id(qa_rows)
        by_key = qa_rows_by_key(qa_rows, ds_cfg, question_col)
        src_image_dir = image_dir(split_dir, variant)
        src_data_dir = chart_data_dir(split_dir, variant)

        for idx, base_row in enumerate(source_rows):
            stem = row_stem(base_row, ds_cfg)
            if not stem or stem not in chart_ids:
                continue
            source_qid = str(base_row.get("question_id") or "").strip()
            qa_row = by_qid.get(source_qid) or by_key.get((stem, str(base_row.get(question_col, "")).strip()))
            if qa_row is None:
                continue

            chart_id = chart_ids[stem]
            src_image = src_image_dir / f"{stem}.png"
            src_data = src_data_dir / f"{stem}.json"
            if not src_image.exists():
                raise FileNotFoundError(f"Missing rendered image for {variant}/{stem}: {src_image}")
            if not src_data.exists():
                raise FileNotFoundError(f"Missing chart data for {variant}/{stem}: {src_data}")

            image_rel = f"images/{variant}/{chart_id}.png"
            chart_data_rel = f"chart_data/{variant}/{chart_id}.json"
            link_or_copy(src_image, out_root / image_rel, args.copy)
            link_or_copy(src_data, out_root / chart_data_rel, args.copy)

            rows_out.append(
                family_member_row(
                    base_row=base_row,
                    member_source=qa_row,
                    variant=variant,
                    chart_id=chart_id,
                    question_id=question_ids[idx],
                    question_col=question_col,
                    answer_col=answer_col,
                    image_rel=image_rel,
                    chart_data_rel=chart_data_rel,
                    source_dataset_split=source_dataset_split,
                    source_row_index=idx,
                )
            )

    split_path = out_root / f"{out_split}.jsonl"
    row_count = write_jsonl(split_path, rows_out)

    config = {
        "datasets": {
            args.output_dataset: {
                "local_file_template": f"{{repo_root}}/data/{args.output_dataset}/{{split}}.jsonl",
                "local_dir": args.output_dataset,
                "question_col": "question",
                "image_col": "image",
                "answer_col": "answer",
                "variant_col": "variant",
                "family_id_col": "question_id",
            }
        }
    }
    config_path = out_root / "datasets.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {row_count} rows: {split_path}")
    print(f"Wrote dataset config: {config_path}")
    print(f"Use this config: export CHARTOGRAPHER_DATASETS_FILE={config_path}")
    return split_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export chart-question families as a local dataset.")
    parser.add_argument("--dataset", required=True, help="Input dataset registered in CHARTOGRAPHER_DATASETS_FILE")
    parser.add_argument("--split", required=True, help="Input split")
    parser.add_argument("--output-dataset", required=True, help="Output family dataset name under data/")
    parser.add_argument("--output-split", default=None, help="Output split filename stem; defaults to the input split")
    parser.add_argument("--source-dataset-split", default=None, help="Value for source_dataset_split; defaults to {dataset}_{split}")
    parser.add_argument("--seeds", default="0-9", help="Counterfactual seeds to include, e.g. '0-9' or '0,1,2'")
    parser.add_argument("--include-original", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of hardlinking/symlinking")
    return parser.parse_args()


if __name__ == "__main__":
    export_chart_question_families(parse_args())
