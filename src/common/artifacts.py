"""Shared artifact layout helpers for Chartographer pipeline outputs."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results" / "chartographer"
RECONSTRUCTION_VARIANT = "reconstruction"


def split_artifact_dir(dataset_local_dir: str, split: str) -> Path:
    """Return the root output directory for one dataset split."""
    return RESULTS_ROOT / f"{dataset_local_dir}_{split}"


def seed_variant(seed: int) -> str:
    """Return the released dataset variant name for a generated seed."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return f"seed_{seed}"


def revision_variant(round: int) -> str:
    """Return reconstruction for round 0, otherwise revision_N."""
    if round < 0:
        raise ValueError("round must be non-negative")
    return RECONSTRUCTION_VARIANT if round == 0 else f"revision_{round}"


def variant_name(seed: int | None = None) -> str:
    """Return reconstruction when seed is absent, otherwise seed_N."""
    return RECONSTRUCTION_VARIANT if seed is None else seed_variant(seed)


def chart_code_dir(split_dir: Path, variant: str = RECONSTRUCTION_VARIANT) -> Path:
    """Directory for runnable chart reconstruction code."""
    return split_dir / "chart_code" / variant


def chart_data_dir(split_dir: Path, variant: str = RECONSTRUCTION_VARIANT) -> Path:
    """Directory for released chart-data JSON artifacts."""
    return split_dir / "chart_data" / variant


def image_dir(split_dir: Path, variant: str = RECONSTRUCTION_VARIANT) -> Path:
    """Directory for rendered chart images."""
    return split_dir / "images" / variant
