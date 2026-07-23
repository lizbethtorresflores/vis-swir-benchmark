"""Flux de submission : valide, versionne, evalue et publie dans le leaderboard.

Ce module fait le pont entre la page Streamlit et le moteur d'evaluation.
Il recoit metadata + H ou ZIP, cree un dossier versionne et met a jour les resultats.
"""

from __future__ import annotations

from pathlib import Path
import traceback

import pandas as pd

from benchmark_config import DATASETS, DIRECTIONS, EVALUATION_PROTOCOL, method_dir, normalize_direction
from benchmark_artifacts import create_private_aligned_outputs_zip, write_private_artifacts_manifest
from benchmark_eval import (
    evaluate_aligned_image_submission,
    evaluate_homography_submission,
    evaluate_submission,
)
from benchmark_subsets import build_subset_manifest, get_subset_pairs
from benchmark_storage import (
    save_json,
    save_method_artifacts,
    summarize_metrics,
    upsert_leaderboard,
)
from benchmark_validation import (
    discover_uploaded_aligned_pairs,
    save_and_extract_aligned_zip,
    validate_aligned_image_submission,
    validate_homography_submission,
)


def validate_metadata(metadata: dict) -> dict:
    """Normaliza y valida campos comunes enviados desde el formulario."""
    method_name = str(metadata.get("method_name", "")).strip()
    if not method_name:
        raise ValueError("The method name is required.")
    dataset = metadata.get("dataset")
    if dataset not in DATASETS:
        raise ValueError(f"Invalid dataset: {dataset}")
    direction = normalize_direction(metadata.get("direction", ""))
    training_pairs = int(metadata.get("training_pairs_used", 0))
    if training_pairs < 0:
        raise ValueError("training_pairs_used must be >= 0.")
    submission_type = str(metadata.get("submission_type", "")).strip()
    single_global_h_confirmed = (
        True
        if submission_type == "homography_based"
        else bool(metadata.get("single_global_homography_confirmed", False))
    )
    if submission_type == "aligned_images" and not single_global_h_confirmed:
        raise ValueError(
            "Aligned-image submissions require confirmation that the submitted aligned images were generated "
            "using one single global homography for this dataset and direction."
        )
    return {
        "method_name": method_name,
        "dataset": dataset,
        "direction": direction,
        "training_pairs_used": training_pairs,
        "authors": str(metadata.get("authors", "")).strip(),
        "institution": str(metadata.get("institution", "")).strip(),
        "paper_url": str(metadata.get("paper_url", "")).strip(),
        "contact_email": str(metadata.get("contact_email", "")).strip(),
        "submission_date": str(metadata.get("submission_date", "")).strip(),
        "description": str(metadata.get("description", "")).strip(),
        "subset_id": str(metadata.get("subset_id", "")).strip(),
        "submission_type": submission_type,
        "single_global_homography_confirmed": single_global_h_confirmed,
        "estimation_pairs_used": metadata.get("estimation_pairs_used"),
        "estimation_data_description": str(metadata.get("estimation_data_description", "")).strip(),
        "evaluation_protocol": EVALUATION_PROTOCOL,
    }


def next_version_dir(base_dir: Path) -> tuple[Path, int]:
    """Cree le nom du prochain dossier version_XXX pour une methode."""
    base_dir.mkdir(parents=True, exist_ok=True)
    versions = []
    for child in base_dir.iterdir():
        if child.is_dir() and child.name.startswith("version_"):
            try:
                versions.append(int(child.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
    version = max(versions, default=0) + 1
    return base_dir / f"version_{version:03d}", version


def _save_runtime_and_leaderboard(
    method_path: Path,
    metadata: dict,
    runtime: dict,
    metrics: pd.DataFrame,
) -> None:
    """Sauvegarde le runtime final et synchronise la ligne correspondante du leaderboard."""
    summary = summarize_metrics(metrics, runtime)
    num_pairs = int(len(metrics))
    row = {
        **metadata,
        **summary,
        "hardware": runtime.get("hardware", runtime.get("device")),
        "num_test_frames": num_pairs,
        "num_evaluated_pairs": num_pairs,
        "validation_status": "passed",
        "evaluation_status": "passed",
        "method_path": str(method_path),
    }
    save_json(method_path / "runtime.json", runtime)
    upsert_leaderboard(row)


def submit_homography_based(
    metadata: dict,
    H,
    runtime_input: dict,
    device: str = "auto",
    progress_callback=None,
) -> tuple[Path, pd.DataFrame]:
    """Traite une submission basee sur une seule homographie globale."""
    metadata = validate_metadata({**metadata, "submission_type": "homography_based"})
    if not metadata["subset_id"]:
        from benchmark_subsets import official_subset_id

        metadata["subset_id"] = official_subset_id(metadata["dataset"], metadata["direction"])
    manifest = build_subset_manifest(metadata["dataset"], metadata["subset_id"], direction=metadata["direction"])
    subset_pairs = get_subset_pairs(metadata["dataset"], metadata["subset_id"], direction=metadata["direction"])
    metadata, H, validation_report = validate_homography_submission(metadata, H, manifest, subset_pairs)
    base_path = method_dir(metadata["method_name"], metadata["dataset"], metadata["direction"])
    method_path, version = next_version_dir(base_path)
    metadata["submission_version"] = version
    metadata["num_evaluated_pairs"] = len(subset_pairs)
    runtime = {
        "runtime_seconds": runtime_input.get("runtime_seconds"),
        "hardware": runtime_input.get("hardware"),
    }
    save_method_artifacts(method_path, metadata, H, runtime)
    save_json(method_path / "subset_manifest.json", manifest)
    save_json(method_path / "validation_report.json", validation_report)
    try:
        # L'evaluation applique H sur le subset officiel et garde les outputs officiels.
        metrics, eval_runtime = evaluate_homography_submission(
            method_path=method_path,
            metadata=metadata,
            H=H,
            subset_pairs=subset_pairs,
            device=device,
            save_outputs=True,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        failure = {
            **metadata,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        save_json(method_path / "metadata.json", failure)
        raise
    runtime = {
        "runtime_seconds": runtime.get("runtime_seconds"),
        "hardware": runtime.get("hardware"),
        "evaluation_runtime_seconds": eval_runtime["runtime_seconds"],
        "evaluation_fps": eval_runtime["fps"],
        "device": eval_runtime["device"],
        "evaluation_protocol": EVALUATION_PROTOCOL,
    }
    private_zip = create_private_aligned_outputs_zip(method_path)
    write_private_artifacts_manifest(method_path, {"own_aligned_outputs_zip": str(private_zip)})
    _save_runtime_and_leaderboard(method_path, metadata, runtime, metrics)
    return method_path, metrics


def submit_aligned_image_based(
    metadata: dict,
    uploaded_zip,
    runtime_input: dict,
    device: str = "auto",
    progress_callback=None,
) -> tuple[Path, pd.DataFrame]:
    """Traite une submission ou l'utilisateur envoie des images deja alignees."""
    metadata = validate_metadata({**metadata, "submission_type": "aligned_images"})
    if not metadata["subset_id"]:
        from benchmark_subsets import official_subset_id

        metadata["subset_id"] = official_subset_id(metadata["dataset"], metadata["direction"])
    manifest = build_subset_manifest(metadata["dataset"], metadata["subset_id"], direction=metadata["direction"])
    subset_pairs = get_subset_pairs(metadata["dataset"], metadata["subset_id"], direction=metadata["direction"])
    base_path = method_dir(metadata["method_name"], metadata["dataset"], metadata["direction"])
    method_path, version = next_version_dir(base_path)
    metadata["submission_version"] = version
    upload_zip_path = method_path / "uploaded" / "aligned_images.zip"
    extract_dir = method_path / "uploaded" / "extracted"
    # On extrait d'abord le ZIP ; ensuite chaque frame est valide contre le subset officiel.
    save_and_extract_aligned_zip(uploaded_zip, upload_zip_path, extract_dir)
    uploaded_pairs = discover_uploaded_aligned_pairs(extract_dir, metadata["direction"])
    metadata, validation_report = validate_aligned_image_submission(metadata, manifest, uploaded_pairs, subset_pairs)
    metadata["num_evaluated_pairs"] = len(uploaded_pairs)
    runtime = {
        "runtime_seconds": runtime_input.get("runtime_seconds"),
        "hardware": runtime_input.get("hardware"),
    }
    method_path.mkdir(parents=True, exist_ok=True)
    save_json(method_path / "metadata.json", metadata)
    save_json(method_path / "runtime.json", runtime)
    save_json(method_path / "subset_manifest.json", manifest)
    save_json(method_path / "validation_report.json", validation_report)
    try:
        # Les images envoyees sont recadrees avec le common crop avant les metriques officielles.
        metrics, eval_runtime = evaluate_aligned_image_submission(
            method_path=method_path,
            metadata=metadata,
            uploaded_pairs=uploaded_pairs,
            device=device,
            save_outputs=True,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        failure = {
            **metadata,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        save_json(method_path / "metadata.json", failure)
        raise
    runtime = {
        "runtime_seconds": runtime.get("runtime_seconds"),
        "hardware": runtime.get("hardware"),
        "evaluation_runtime_seconds": eval_runtime["runtime_seconds"],
        "evaluation_fps": eval_runtime["fps"],
        "device": eval_runtime["device"],
        "evaluation_protocol": EVALUATION_PROTOCOL,
    }
    _save_runtime_and_leaderboard(method_path, metadata, runtime, metrics)
    return method_path, metrics


def submit_method(metadata: dict, H, runtime_input: dict, device: str = "auto") -> tuple[Path, pd.DataFrame]:
    """Compatibilidad con el flujo antiguo de homografia sin manifest explicito."""
    if metadata.get("submission_type") == "aligned_images":
        raise ValueError("Use submit_aligned_image_based for aligned-image submissions.")
    if metadata.get("subset_id"):
        return submit_homography_based(metadata, H, runtime_input, device=device)

    metadata = validate_metadata(metadata)
    base_path = method_dir(metadata["method_name"], metadata["dataset"], metadata["direction"])
    method_path, version = next_version_dir(base_path)
    metadata["submission_version"] = version
    runtime = {
        "runtime_seconds": runtime_input.get("runtime_seconds"),
        "hardware": runtime_input.get("hardware"),
    }
    save_method_artifacts(method_path, metadata, H, runtime)
    try:
        metrics, eval_runtime = evaluate_submission(
            method_path=method_path,
            dataset=metadata["dataset"],
            direction=metadata["direction"],
            H=H,
            device=device,
        )
    except Exception as exc:
        failure = {
            **metadata,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        save_json(method_path / "metadata.json", failure)
        raise
    runtime = {
        "runtime_seconds": runtime.get("runtime_seconds"),
        "hardware": runtime.get("hardware"),
        "evaluation_runtime_seconds": eval_runtime["runtime_seconds"],
        "evaluation_fps": eval_runtime["fps"],
        "device": eval_runtime["device"],
        "evaluation_protocol": EVALUATION_PROTOCOL,
    }
    save_json(method_path / "runtime.json", runtime)

    summary = summarize_metrics(metrics, runtime)
    row = {**metadata, **summary, "method_path": str(method_path)}
    upsert_leaderboard(row)
    return method_path, metrics
