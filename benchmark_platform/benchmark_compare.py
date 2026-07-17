"""Charge, aligne et resume les metriques pour comparer les methodes.

La comparaison se fait sur les memes frame IDs quand il y a plusieurs methodes,
afin que les moyennes et rankings ne melangent pas des sous-ensembles differents.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmark_config import LOWER_IS_BETTER, METRIC_COLUMNS
from benchmark_existing import load_existing_bundle
from benchmark_storage import load_json, load_metrics


def _normalize_frame_ids(metrics: pd.DataFrame) -> pd.DataFrame:
    """Assure que la table de metriques possede une colonne numerique frame_id."""
    metrics = metrics.copy()
    if "frame_id" not in metrics and "pair_id" in metrics:
        metrics["frame_id"] = metrics["pair_id"].astype(str).str.extract(r"(\d+)")
    if "frame_id" in metrics:
        metrics["frame_id"] = pd.to_numeric(metrics["frame_id"], errors="coerce").astype("Int64")
    return metrics


def _frame_key_table(metrics: pd.DataFrame) -> pd.Series:
    """Construit une cle dataset/frame commune, independamment de la direction."""
    frame_ids = metrics["frame_id"].astype("Int64").astype(str)
    return metrics["dataset"].astype(str) + "|" + frame_ids


def filter_metrics_to_common_frame_ids(frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Filtre sur les memes frames du dataset, meme si la direction differe."""
    if len(frames) <= 1:
        return frames
    normalized = []
    key_sets = []
    for frame in frames:
        table = _normalize_frame_ids(frame)
        if table.empty or "frame_id" not in table or table["frame_id"].dropna().empty:
            normalized.append(table)
            continue
        table["_frame_key"] = _frame_key_table(table)
        normalized.append(table)
        key_sets.append(set(table["_frame_key"].dropna()))
    if len(key_sets) != len(frames) or not key_sets:
        return [table.drop(columns=["_frame_key"], errors="ignore") for table in normalized]
    common_keys = set.intersection(*key_sets)
    if not common_keys:
        return [table.iloc[0:0].drop(columns=["_frame_key"], errors="ignore") for table in normalized]
    filtered = []
    for table in normalized:
        table = table[table["_frame_key"].isin(common_keys)].copy()
        table = table.sort_values(["dataset", "direction", "frame_id"])
        filtered.append(table.drop(columns=["_frame_key"], errors="ignore"))
    return filtered


def filter_bundles_to_common_frame_ids(bundles: list[dict]) -> list[dict]:
    """Filtre des bundles complets en gardant seulement les frames communs."""
    frames = []
    for bundle in bundles:
        metrics = bundle["metrics"].copy()
        metadata = bundle["metadata"]
        if not metrics.empty:
            metrics["dataset"] = metadata.get("dataset", "")
            metrics["direction"] = metadata.get("direction", "")
        frames.append(metrics)
    filtered_frames = filter_metrics_to_common_frame_ids(frames)
    filtered_bundles = []
    for bundle, metrics in zip(bundles, filtered_frames):
        new_bundle = bundle.copy()
        new_bundle["metrics"] = metrics
        filtered_bundles.append(new_bundle)
    return filtered_bundles


def build_ranking(df: pd.DataFrame, metric_label: str) -> pd.DataFrame:
    """Ordena una tabla por metrica oficial y agrega la columna de ranking."""
    if df.empty:
        return df
    column = METRIC_COLUMNS[metric_label]
    ascending = LOWER_IS_BETTER[metric_label]
    ranked = df.copy()
    ranked[column] = pd.to_numeric(ranked[column], errors="coerce")
    ranked = ranked.sort_values(column, ascending=ascending, na_position="last").reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def load_method_bundle(method_path: str | Path, diagnostic: bool = True) -> dict:
    """Charge les metriques des split_images finaux d'une methode."""
    method_path = Path(method_path)
    if not (method_path / "metadata.json").exists():
        return load_existing_bundle(method_path)
    metadata = load_json(method_path / "metadata.json")
    runtime = load_json(method_path / "runtime.json") if (method_path / "runtime.json").exists() else {}
    diagnostic_path = method_path / "metrics_full_image_diagnostic.csv"
    metrics = pd.read_csv(diagnostic_path) if diagnostic and diagnostic_path.exists() else load_metrics(method_path)
    return {"path": method_path, "metadata": metadata, "runtime": runtime, "metrics": metrics}


def metrics_with_method(
    method_paths: list[str | Path],
    diagnostic: bool = False,
    common_frame_ids: bool = True,
) -> pd.DataFrame:
    """Combine les metriques par frame de plusieurs methodes dans une seule table."""
    frames = []
    for raw_path in method_paths:
        bundle = load_method_bundle(raw_path, diagnostic=diagnostic)
        metrics = bundle["metrics"].copy()
        if metrics.empty:
            continue
        metadata = bundle["metadata"]
        metrics = _normalize_frame_ids(metrics)
        metrics["method_name"] = metadata["method_name"]
        metrics["dataset"] = metadata["dataset"]
        metrics["direction"] = metadata["direction"]
        frames.append(metrics)
    if not frames:
        return pd.DataFrame()
    if common_frame_ids:
        frames = filter_metrics_to_common_frame_ids(frames)
    return pd.concat(frames, ignore_index=True)


def mean_metric_table(
    method_paths: list[str | Path],
    diagnostic: bool = False,
    common_frame_ids: bool = True,
) -> pd.DataFrame:
    """Calcule la moyenne des metriques officielles par methode/dataset/direction."""
    df = metrics_with_method(method_paths, diagnostic=diagnostic, common_frame_ids=common_frame_ids)
    if df.empty:
        return df
    metric_cols = [
        c
        for c in [
            "goa",
            "edge_f1_8px",
            "mind_ssd",
            "reprojection_error",
        ]
        if c in df
    ]
    grouped = df.groupby(["method_name", "dataset", "direction"], as_index=False)
    table = grouped[metric_cols].mean()
    counts = grouped["frame_id"].nunique().rename(columns={"frame_id": "common_frames"})
    return table.merge(counts, on=["method_name", "dataset", "direction"], how="left")
