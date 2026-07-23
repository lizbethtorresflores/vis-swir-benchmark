"""Persistance du benchmark : methodes, metriques, runtime et leaderboard.

Ce module maintient le CSV principal du leaderboard et combine les nouvelles
submissions avec les resultats existants decouverts dans la structure de donnees.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_config import LEADERBOARD_PATH, METHODS_ROOT, METRIC_COLUMNS
from benchmark_existing import (
    discover_existing_methods,
    existing_leaderboard_rows,
    hardware_for_method,
    normalize_method_name,
)


def ensure_storage() -> None:
    """Cree les dossiers et le CSV de leaderboard s'ils n'existent pas encore."""
    METHODS_ROOT.mkdir(parents=True, exist_ok=True)
    LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LEADERBOARD_PATH.exists():
        pd.DataFrame(columns=leaderboard_columns()).to_csv(LEADERBOARD_PATH, index=False)


def leaderboard_columns() -> list[str]:
    """Define el esquema esperado del CSV de leaderboard."""
    return [
        "method_name",
        "dataset",
        "direction",
        "training_pairs_used",
        "goa",
        "edge_f1_8px",
        "mind_ssd",
        "reprojection_error",
        "runtime_seconds",
        "hardware",
        "num_test_frames",
        "num_evaluated_pairs",
        "submission_type",
        "subset_id",
        "validation_status",
        "evaluation_status",
        "evaluation_protocol",
        "submission_version",
        "submission_date",
        "source",
        "method_path",
    ]


def save_json(path: Path, payload: dict) -> None:
    """Sauvegarde un dictionnaire en JSON indente."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    """Charge un JSON depuis le disque."""
    return json.loads(path.read_text(encoding="utf-8"))


def save_method_artifacts(method_path: Path, metadata: dict, H: np.ndarray, runtime: dict) -> None:
    """Cree le dossier d'une version et sauvegarde metadata, runtime et H_global.npy."""
    method_path.mkdir(parents=True, exist_ok=True)
    save_json(method_path / "metadata.json", metadata)
    save_json(method_path / "runtime.json", runtime)
    np.save(method_path / "H_global.npy", H)
    for child in ["split_images", "concatenate_images"]:
        (method_path / child).mkdir(exist_ok=True)
    (method_path / "evaluation_log.txt").touch(exist_ok=True)


def load_metrics(method_path: Path) -> pd.DataFrame:
    """Charge metrics.csv d'une methode ; retourne une table vide s'il n'existe pas."""
    csv_path = method_path / "metrics.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def summarize_metrics(metrics: pd.DataFrame, runtime: dict | None = None) -> dict:
    """Resume les metriques par methode avec les moyennes et donnees de runtime."""
    runtime = runtime or {}
    summary = {}
    for label, column in METRIC_COLUMNS.items():
        if column == "runtime_seconds":
            summary[column] = runtime.get(column, np.nan)
        elif column in metrics:
            summary[column] = pd.to_numeric(metrics[column], errors="coerce").mean()
        else:
            summary[column] = np.nan
    return summary


def upsert_leaderboard(row: dict) -> pd.DataFrame:
    """Insere ou remplace la ligne d'une methode dans le leaderboard."""
    ensure_storage()
    df = pd.read_csv(LEADERBOARD_PATH)
    key = (
        (df["method_name"] == row["method_name"])
        & (df["dataset"] == row["dataset"])
        & (df["direction"] == row["direction"])
    )
    df = df.loc[~key].copy()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(LEADERBOARD_PATH, index=False)
    return df


def load_leaderboard() -> pd.DataFrame:
    """Charge le CSV des submissions sauvegardees."""
    ensure_storage()
    return pd.read_csv(LEADERBOARD_PATH)


def load_benchmark_leaderboard() -> pd.DataFrame:
    """Combine methodes existantes et nouvelles submissions dans une vue unique."""
    submitted = load_leaderboard()
    existing = existing_leaderboard_rows()
    if not submitted.empty:
        submitted = submitted.copy()
        submitted["source"] = submitted.get("source", "submitted")
        if "num_test_frames" not in submitted:
            submitted["num_test_frames"] = np.nan
        if "evaluation_protocol" not in submitted:
            submitted["evaluation_protocol"] = np.nan
        if "submission_version" not in submitted:
            submitted["submission_version"] = np.nan
        if "num_evaluated_pairs" not in submitted:
            submitted["num_evaluated_pairs"] = submitted.get("num_test_frames", np.nan)
        if "submission_type" not in submitted:
            submitted["submission_type"] = np.nan
        if "subset_id" not in submitted:
            submitted["subset_id"] = np.nan
        if "validation_status" not in submitted:
            submitted["validation_status"] = np.nan
        if "evaluation_status" not in submitted:
            submitted["evaluation_status"] = np.nan
    frames = [df for df in [existing, submitted] if not df.empty]
    if not frames:
        return pd.DataFrame(columns=leaderboard_columns())
    df = pd.concat(frames, ignore_index=True, sort=False)
    if "method_name" in df:
        df["method_name"] = df["method_name"].map(normalize_method_name)
    if "hardware" not in df:
        df["hardware"] = np.nan
    df["hardware"] = df["hardware"].astype(object)
    for index, row in df.loc[df["hardware"].isna()].iterrows():
        runtime_path = Path(str(row.get("method_path", ""))) / "runtime.json"
        if not runtime_path.exists():
            continue
        try:
            runtime = load_json(runtime_path)
        except Exception:
            continue
        hardware = runtime.get("hardware", runtime.get("device"))
        if hardware not in (None, ""):
            df.at[index, "hardware"] = hardware
    df["hardware"] = [
        hardware_for_method(method_name, hardware)
        for method_name, hardware in zip(df["method_name"], df["hardware"])
    ]
    return df.drop_duplicates(subset=["method_path"], keep="last")


def _method_selection_key(item: dict) -> tuple[str, str, str]:
    """Cle utilisee pour selectionner la version officielle par methode/dataset/direction."""
    return (
        str(item.get("method_name", "")),
        str(item.get("dataset", "")),
        str(item.get("direction", "")),
    )


def _version_number(item: dict) -> int:
    """Extrait le numero de version d'une submission, en utilisant 0 s'il manque."""
    try:
        return int(item.get("submission_version", 0))
    except (TypeError, ValueError):
        return 0


def _is_preferred_method_item(candidate: dict, current: dict) -> bool:
    """Prefere la salida final no cropped y luego la version mas reciente."""
    candidate_path = str(candidate.get("method_path", ""))
    current_path = str(current.get("method_path", ""))
    candidate_is_crop = "_cropped_like_rasmd" in candidate_path or "official_common_crop" in candidate_path
    current_is_crop = "_cropped_like_rasmd" in current_path or "official_common_crop" in current_path
    if candidate_is_crop != current_is_crop:
        return not candidate_is_crop
    return _version_number(candidate) >= _version_number(current)


def official_method_items(methods: list[dict]) -> list[dict]:
    """Reduit une liste de versions a la version officielle visible par methode."""
    selected: dict[tuple[str, str, str], dict] = {}
    for item in methods:
        key = _method_selection_key(item)
        current = selected.get(key)
        if current is None or _is_preferred_method_item(item, current):
            selected[key] = item
    return sorted(selected.values(), key=lambda item: item["label"])


def discover_methods() -> list[dict]:
    """Decouvre les methodes disponibles pour les selecteurs de comparaison et detail."""
    ensure_storage()
    methods = discover_existing_methods()
    for metadata_path in METHODS_ROOT.glob("*/*/*/**/metadata.json"):
        method_path = metadata_path.parent
        if not (method_path / "metrics.csv").exists():
            continue
        try:
            metadata = load_json(metadata_path)
        except Exception:
            continue
        metadata = metadata.copy()
        metadata["method_name"] = normalize_method_name(
            metadata.get("method_name", method_path.parts[-3])
        )
        methods.append(
            {
                **metadata,
                "method_path": str(method_path),
                "source": "submitted",
                "label": f"{metadata['method_name']} | {metadata.get('dataset')} | {metadata.get('direction')}",
            }
        )
    return official_method_items(methods)
