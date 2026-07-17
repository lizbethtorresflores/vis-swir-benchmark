"""Complete Edge Alignment 8 px in legacy benchmark quality CSV files."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "projet"))

from registration.evaluation.quality_bench import preprocess_pair_bench  # noqa: E402
from registration.pipeline.metrics import compute_edge_alignment_metrics  # noqa: E402

from benchmark_existing import _dataset_from_method_path, _frame_ids_from_quality  # noqa: E402
from benchmark_subsets import official_subset_frame_ids  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
PAIR_SUFFIXES = ("_vis_aligned", "_swir_aligned", "_visible", "_vis", "_swir")


def _pair_key(path: Path) -> str:
    stem = path.stem
    for suffix in PAIR_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _frame_id(text: str) -> int | None:
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else None


def _official_pairs(split_dir: Path, dataset: str) -> dict[int, tuple[Path, Path]]:
    official_ids = set(official_subset_frame_ids(dataset))
    vis: dict[str, Path] = {}
    swir: dict[str, Path] = {}
    for path in sorted(split_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        stem = path.stem.lower()
        key = _pair_key(path)
        if stem.endswith("_vis") or stem.endswith("_vis_aligned") or stem.endswith("_visible"):
            vis[key] = path
        elif stem.endswith("_swir") or stem.endswith("_swir_aligned"):
            swir[key] = path

    pairs: dict[int, tuple[Path, Path]] = {}
    for key in sorted(set(vis) & set(swir)):
        frame_id = _frame_id(key)
        if frame_id in official_ids:
            pairs[frame_id] = (vis[key], swir[key])
    return pairs


def backfill_method(method_dir: Path, device: str = "cpu") -> int:
    dataset = _dataset_from_method_path(method_dir)
    csv_path = method_dir / "results" / "scores_qualite_recalage.csv"
    split_dir = method_dir / "split_images"
    metrics = pd.read_csv(csv_path)
    pairs = _official_pairs(split_dir, dataset)
    if len(pairs) != len(official_subset_frame_ids(dataset)):
        raise RuntimeError(f"{method_dir.name}: expected 100 official pairs, found {len(pairs)}")

    frame_ids = _frame_ids_from_quality(metrics)
    edge_by_frame: dict[int, dict[str, float]] = {}
    for position, (frame_id, (vis_path, swir_path)) in enumerate(pairs.items(), start=1):
        vis = cv2.imread(str(vis_path), cv2.IMREAD_UNCHANGED)
        swir = cv2.imread(str(swir_path), cv2.IMREAD_UNCHANGED)
        if vis is None or swir is None:
            raise RuntimeError(f"Unable to read pair {frame_id}: {vis_path.name}, {swir_path.name}")
        vis_gray, swir_gray, valid_mask = preprocess_pair_bench(
            vis, swir, np.eye(3, dtype=np.float64), device=device
        )
        edge_by_frame[frame_id] = compute_edge_alignment_metrics(
            vis_gray, swir_gray, valid_mask, tol_px=8
        )
        print(f"[{method_dir.name}] {position:03d}/100 frame {frame_id}", flush=True)

    for column in ["edge_precision_8px", "edge_recall_8px", "edge_f1_8px"]:
        values = frame_ids.map(lambda value: edge_by_frame.get(int(value), {}).get(column, np.nan) if pd.notna(value) else np.nan)
        if column in metrics:
            metrics[column] = metrics[column].where(metrics[column].notna(), values)
        else:
            metrics[column] = values
    metrics.to_csv(csv_path, index=False)
    return len(edge_by_frame)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method_dirs", nargs="+", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    for method_dir in args.method_dirs:
        count = backfill_method(method_dir.resolve(), device=args.device)
        print(f"Completed {method_dir}: {count} official frames", flush=True)


if __name__ == "__main__":
    main()
