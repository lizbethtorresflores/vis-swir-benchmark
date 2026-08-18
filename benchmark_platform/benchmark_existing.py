"""Decouverte des methodes deja calculees hors du flux de submission.

Permet aux dossiers existants dans datas/*/paired_images d'apparaitre dans le
leaderboard, les comparaisons et les details sans les renvoyer via l'app.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_config import DATA_ROOT, DATASETS, EVALUATION_PROTOCOL
from benchmark_subsets import official_subset_frame_ids, official_subset_id


EXCLUDED_METHOD_DIRS = {"VISIBLE", "SWIR", "benchmark_comparison_methods"}


def _read_csv(path: Path) -> pd.DataFrame:
    """Lit un CSV s'il existe ; retourne une table vide s'il manque ou est corrompu."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _mean_numeric_column(df: pd.DataFrame, column: str) -> float:
    """Retourne la moyenne numerique d'une colonne optionnelle."""
    if column not in df:
        return np.nan
    return pd.to_numeric(df[column], errors="coerce").mean()


def _frame_ids_from_quality(df: pd.DataFrame) -> pd.Series:
    """Extract frame ids from quality CSV rows."""
    for column in ["pair_id", "vis_file", "swir_file"]:
        if column in df:
            extracted = pd.to_numeric(df[column].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
            if extracted.notna().any():
                return extracted
    if "frame_id" in df:
        return pd.to_numeric(df["frame_id"], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _filter_to_official_subset(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Keep only the official 100 leaderboard frame ids."""
    if df.empty:
        return df
    frame_ids = _frame_ids_from_quality(df)
    official_ids = set(official_subset_frame_ids(dataset))
    return df.loc[frame_ids.isin(official_ids)].copy()


def _complete_edge_metrics_from_cropped_sibling(
    quality_scores: pd.DataFrame,
    method_path: Path,
    dataset: str,
) -> pd.DataFrame:
    """Complete les anciennes metriques diagnostiques avec Edge Alignment si possible."""
    edge_cols = ["edge_precision_8px", "edge_recall_8px", "edge_f1_8px"]
    if quality_scores.empty or all(col in quality_scores for col in edge_cols):
        return quality_scores

    if method_path.name.endswith("_cropped_like_rasmd"):
        return quality_scores

    cropped_path = method_path.with_name(f"{method_path.name}_cropped_like_rasmd")
    cropped_scores = _filter_to_official_subset(
        _read_csv(cropped_path / "results" / "scores_qualite_recalage.csv"),
        dataset,
    )
    available_edge_cols = [col for col in edge_cols if col in cropped_scores]
    if cropped_scores.empty or not available_edge_cols:
        return quality_scores

    current = quality_scores.copy()
    current["_edge_frame_id"] = _frame_ids_from_quality(current)
    edge_values = cropped_scores.copy()
    edge_values["_edge_frame_id"] = _frame_ids_from_quality(edge_values)
    edge_values = edge_values[["_edge_frame_id", *available_edge_cols]].dropna(subset=["_edge_frame_id"])
    edge_values = edge_values.drop_duplicates("_edge_frame_id")

    missing_cols = [col for col in available_edge_cols if col not in current]
    current = current.merge(edge_values[["_edge_frame_id", *missing_cols]], on="_edge_frame_id", how="left")
    return current.drop(columns=["_edge_frame_id"], errors="ignore")


def _dataset_from_method_path(method_path: Path) -> str:
    """Deduce el dataset desde una ruta que contiene sunny_test, cloudy_test, etc."""
    for part in method_path.parts:
        if part.endswith("_test"):
            return part.replace("_test", "")
    return "unknown"


def _direction_from_name(name: str) -> str:
    """Deduit VIS_to_SWIR ou SWIR_to_VIS depuis le nom du dossier de methode."""
    upper = name.upper()
    if "SWIR_TO_VIS" in upper or "SWIR_TO_RGB" in upper:
        return "SWIR_to_VIS"
    if "VIS_TO_SWIR" in upper or "RGB_TO_SWIR" in upper:
        return "VIS_to_SWIR"
    if "RASMD" in upper:
        return "VIS_to_SWIR"
    return "unknown"


def _training_pairs_from_name(name: str) -> int | float:
    """Extrait train150, train500, etc. du nom de methode quand il existe."""
    match = re.search(r"train(\d+)", name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return np.nan


def normalize_method_name(name: str) -> str:
    """Return the canonical short method name used throughout the UI."""
    name = str(name).strip()
    aliases = {
        "Mouad": "RoMa-MIND",
        "RoMa–MIND": "RoMa-MIND",
        "RASMD_aligned_baseline": "RASMD Aligned",
        "RASMD baseline": "RASMD Aligned",
    }
    if name in aliases:
        return aliases[name]
    if name.startswith("opencv_alignment") or name.startswith("OpenCV global H"):
        return "OpenCV"
    if name.startswith("patch_RANSAC_HOMOGRAPHY") or name.startswith("Patch MAGSAC++"):
        return "Patch MAGSAC++"
    if name.startswith("DFM"):
        return "DFM global H"
    return name


def hardware_for_method(method_name: str, recorded_hardware: object = "") -> str:
    """Return the documented hardware used to run a benchmark method."""
    canonical_name = normalize_method_name(method_name)
    if canonical_name in {"Patch MAGSAC++", "OpenCV"}:
        return "Intel i7-7700K / NVIDIA GTX 1080 Ti"
    if canonical_name == "RoMa-MIND":
        return "Intel i9-9980XE / 2× NVIDIA RTX 2080 Ti"
    if canonical_name == "DFM global H":
        return "Intel i7-7700K / NVIDIA GTX 1080 Ti"
    if canonical_name == "RASMD Aligned":
        return "-"
    if recorded_hardware is None or pd.isna(recorded_hardware):
        return "-"
    recorded = str(recorded_hardware).strip()
    return recorded if recorded and recorded.lower() not in {"nan", "none"} else "-"


def _nice_method_name(name: str) -> str:
    """Backward-compatible wrapper for the canonical UI method name."""
    return normalize_method_name(name)


def _parse_runtime_metrics(path: Path) -> dict:
    """Lit runtime_metrics.txt genere par les anciens pipelines."""
    if not path.exists() and path.parent.name.endswith("_cropped_like_rasmd"):
        base_path = path.parent.with_name(path.parent.name.replace("_cropped_like_rasmd", "")) / path.name
        if base_path.exists():
            path = base_path
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip().lower()
        raw_value = raw_value.strip().split()[0] if raw_value.strip() else ""
        if "hardware" in key or key == "device":
            values["hardware"] = raw_value
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if "temps total" in key:
            values["runtime_seconds"] = value
        elif "fps global" in key:
            values["fps"] = value
        elif "train images" in key:
            values["training_pairs_used"] = int(value)
    return values


def _method_dirs() -> list[Path]:
    """Liste les dossiers candidats de methodes existantes dans tous les datasets."""
    dirs = []
    for dataset in DATASETS:
        paired = DATA_ROOT / f"{dataset}_test" / "Sortie_Extraite" / "paired_images"
        if not paired.exists():
            continue
        for child in sorted(p for p in paired.iterdir() if p.is_dir()):
            if child.name in EXCLUDED_METHOD_DIRS or child.name.startswith("."):
                continue
            dirs.append(child)
    return dirs


def _is_common_crop_variant(method_path: Path) -> bool:
    """Conserva solo las salidas finales base; ignora variantes cropped."""
    return not method_path.name.endswith("_cropped_like_rasmd")


def existing_leaderboard_rows() -> pd.DataFrame:
    """Construit les lignes du leaderboard a partir des resultats existants dans datas/."""
    rows = []
    for method_path in _method_dirs():
        if not _is_common_crop_variant(method_path):
            continue
        results_dir = method_path / "results"
        dataset = _dataset_from_method_path(method_path)
        direction = _direction_from_name(method_path.name)
        quality_scores = _filter_to_official_subset(
            _read_csv(results_dir / "scores_qualite_recalage.csv"),
            dataset,
        )
        reproj_framewise = _filter_to_official_subset(
            _read_csv(results_dir / "reprojection_aligned" / "reprojection_aligned_framewise.csv"),
            dataset,
        )
        quality = _read_csv(results_dir / "quality_summary.csv")
        reproj = _read_csv(results_dir / "reprojection_aligned" / "reprojection_aligned_summary.csv")
        reproj_runtime = _read_csv(results_dir / "reprojection_aligned" / "reprojection_aligned_runtime.csv")
        quality_runtime = _read_csv(results_dir / "quality_runtime.csv")

        if quality_scores.empty and reproj_framewise.empty and quality.empty and reproj.empty:
            continue

        method_name = method_path.name
        runtime = _parse_runtime_metrics(method_path / "runtime_metrics.txt")
        row = {
            "method_name": _nice_method_name(method_name),
            "method_id": method_name,
            "dataset": dataset,
            "direction": direction,
            "training_pairs_used": runtime.get("training_pairs_used", _training_pairs_from_name(method_name)),
            "goa": np.nan,
            "edge_f1_8px": np.nan,
            "mind_ssd": np.nan,
            "reprojection_error": np.nan,
            "runtime_seconds": runtime.get("runtime_seconds", np.nan),
            "fps": runtime.get("fps", np.nan),
            "hardware": hardware_for_method(method_name, runtime.get("hardware")),
            "num_test_frames": np.nan,
            "num_evaluated_pairs": np.nan,
            "subset_id": official_subset_id(dataset, direction),
            "submission_type": "existing_method",
            "validation_status": "official_subset",
            "evaluation_status": "passed",
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "source": "existing",
            "method_path": str(method_path),
        }

        if not quality_scores.empty:
            row["goa"] = _mean_numeric_column(quality_scores, "goa")
            row["edge_f1_8px"] = _mean_numeric_column(quality_scores, "edge_f1_8px")
            row["mind_ssd"] = _mean_numeric_column(quality_scores, "mind_ssd")
            row["num_test_frames"] = len(quality_scores)
            row["num_evaluated_pairs"] = len(quality_scores)
        elif not quality.empty:
            q = quality.iloc[0]
            row["goa"] = q.get("mean_goa", np.nan)
            row["edge_f1_8px"] = q.get("mean_edge_f1_8px", np.nan)
            row["mind_ssd"] = q.get("mean_mind_ssd", np.nan)
            row["num_test_frames"] = min(100, q.get("num_frames", q.get("num_images", np.nan)))
            row["num_evaluated_pairs"] = row["num_test_frames"]

        if not reproj_framewise.empty:
            row["reprojection_error"] = pd.to_numeric(
                reproj_framewise.get("mean_error"),
                errors="coerce",
            ).mean()
            row["num_test_frames"] = len(reproj_framewise)
            row["num_evaluated_pairs"] = len(reproj_framewise)
        elif not reproj.empty:
            r = reproj.iloc[0]
            row["reprojection_error"] = r.get("mean_of_mean_error", np.nan)
            row["num_test_frames"] = min(100, r.get("num_frames", row["num_test_frames"]))
            row["num_evaluated_pairs"] = row["num_test_frames"]

        rows.append(row)

    return pd.DataFrame(rows)


def discover_existing_methods() -> list[dict]:
    """Retourne les methodes existantes dans un format utilisable par les selecteurs Streamlit."""
    df = existing_leaderboard_rows()
    if df.empty:
        return []
    methods = []
    for _, row in df.sort_values(["dataset", "direction", "method_name"]).iterrows():
        methods.append(
            {
                "method_name": row["method_name"],
                "method_id": row["method_id"],
                "dataset": row["dataset"],
                "direction": row["direction"],
                "method_path": row["method_path"],
                "source": "existing",
                "label": f"{row['method_name']} | {row['dataset']} | {row['direction']}",
            }
        )
    return methods


def load_existing_bundle(method_path: str | Path) -> dict:
    """Charge une metadata synthetique et les metriques frame-by-frame d'une methode existante."""
    method_path = Path(method_path)
    results_dir = method_path / "results"
    dataset = _dataset_from_method_path(method_path)
    quality_scores = _filter_to_official_subset(
        _read_csv(results_dir / "scores_qualite_recalage.csv"),
        dataset,
    )
    reproj_framewise = _filter_to_official_subset(
        _read_csv(results_dir / "reprojection_aligned" / "reprojection_aligned_framewise.csv"),
        dataset,
    )
    quality_summary = _read_csv(results_dir / "quality_summary.csv")
    runtime = _parse_runtime_metrics(method_path / "runtime_metrics.txt")

    metrics = quality_scores.copy()
    if "frame_id" not in metrics and "pair_id" in metrics:
        metrics["frame_id"] = metrics["pair_id"].astype(str).str.extract(r"(\d+)").astype(float)

    if not reproj_framewise.empty:
        reproj_cols = ["frame_id", "pair_id", "mean_error", "median_error"]
        reproj = reproj_framewise.copy()
        # Les anciens rapports de reprojection utilisent parfois un index local
        # 0..N-1 dans frame_id, alors que vis_file/swir_file portent l'ID reel
        # du dataset (par exemple 500..1499 pour cloudy). L'union avec les
        # metriques qualite doit toujours employer cet ID reel.
        reproj["frame_id"] = _frame_ids_from_quality(reproj)
        if "pair_id" not in reproj:
            reproj["pair_id"] = reproj.get("frame_id", pd.Series(dtype=object)).astype(str)
        reproj = reproj[[c for c in reproj_cols if c in reproj]]
        reproj = reproj.rename(columns={"mean_error": "reprojection_error"})
        reproj = reproj[[c for c in ["frame_id", "pair_id", "reprojection_error", "median_error"] if c in reproj]]
        if metrics.empty:
            metrics = reproj
        elif "frame_id" in metrics and "frame_id" in reproj:
            metrics = metrics.merge(
                reproj.drop(columns=["pair_id"], errors="ignore"),
                on="frame_id",
                how="left",
            )
        else:
            join_cols = [c for c in ["frame_id", "reprojection_error"] if c in reproj]
            metrics = metrics.reset_index(drop=True).join(
                reproj[join_cols].reset_index(drop=True),
                how="left",
            )

    if "frame_id" not in metrics and "pair_id" in metrics:
        metrics["frame_id"] = metrics["pair_id"].astype(str).str.extract(r"(\d+)").astype(float)

    metadata = {
        "method_name": _nice_method_name(method_path.name),
        "method_id": method_path.name,
        "dataset": dataset,
        "direction": _direction_from_name(method_path.name),
        "training_pairs_used": runtime.get("training_pairs_used", _training_pairs_from_name(method_path.name)),
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "source": "existing",
    }
    runtime["hardware"] = hardware_for_method(metadata["method_name"], runtime.get("hardware"))
    if not quality_summary.empty:
        q = quality_summary.iloc[0]
        runtime.setdefault("quality_eval_time_s", q.get("quality_eval_time_s", np.nan))
        runtime.setdefault("fps_quality_eval", q.get("fps_quality_eval", np.nan))
    return {"path": method_path, "metadata": metadata, "runtime": runtime, "metrics": metrics}
