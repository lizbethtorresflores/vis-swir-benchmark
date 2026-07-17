"""Configuration centrale du benchmark VIS-SWIR.

Definit les routes, datasets, directions acceptees, metriques officielles et le
protocole fixe d'evaluation utilise par toute l'app.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "datas"
RESULTS_ROOT = PROJECT_ROOT / "benchmark_results"
METHODS_ROOT = RESULTS_ROOT / "methods"
LEADERBOARD_PATH = RESULTS_ROOT / "leaderboard.csv"

DATASETS = ["sunny", "cloudy", "rainy", "snowy"]
DIRECTIONS = ["VIS_to_SWIR", "SWIR_to_VIS"]

METRIC_COLUMNS = {
    "GOA": "goa",
    "Edge Alignment": "edge_f1_8px",
    "MIND-SSD": "mind_ssd",
    "Reprojection Error": "reprojection_error",
    "Runtime": "runtime_seconds",
    "FPS": "fps",
}

LOWER_IS_BETTER = {
    "GOA": False,
    "Edge Alignment": False,
    "MIND-SSD": True,
    "Reprojection Error": True,
    "Runtime": True,
    "FPS": False,
}

# A fixed, public protocol. Methods may use any training pairs, but all
# submissions are evaluated on this same deterministic test slice per dataset.
FIXED_TEST_SPLITS = {
    "sunny": {"start": 0, "end": 999},
    "cloudy": {"start": 500, "end": 1499},
    "rainy": {"start": 0, "end": 793},
    "snowy": {"start": 1400, "end": 2399},
}

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
EVALUATION_PROTOCOL = "final_split_images"

def dataset_key(dataset: str) -> str:
    """Convertit un nom court comme 'sunny' en dossier reel 'sunny_test'."""
    return dataset if dataset.endswith("_test") else f"{dataset}_test"


def paired_images_dir(dataset: str) -> Path:
    """Retourne le dossier contenant les paires VIS/SWIR du dataset."""
    return DATA_ROOT / dataset_key(dataset) / "Sortie_Extraite" / "paired_images"


def visible_dir(dataset: str) -> Path:
    """Retourne le dossier des images visibles pour un dataset."""
    return paired_images_dir(dataset) / "VISIBLE"


def swir_dir(dataset: str) -> Path:
    """Retourne le dossier des images SWIR pour un dataset."""
    return paired_images_dir(dataset) / "SWIR"


def normalize_direction(direction: str) -> str:
    """Normalise et valide la direction de recalage acceptee par le benchmark."""
    normalized = direction.replace("VIS_TO_SWIR", "VIS_to_SWIR")
    normalized = normalized.replace("SWIR_TO_VIS", "SWIR_to_VIS")
    if normalized not in DIRECTIONS:
        raise ValueError(f"Invalid direction: {direction}")
    return normalized


def method_dir(method_name: str, dataset: str, direction: str) -> Path:
    """Construit le dossier de base securise ou les versions d'une methode sont gardees."""
    safe_name = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in method_name.strip())
    if not safe_name:
        raise ValueError("The method name cannot be empty.")
    return METHODS_ROOT / safe_name / dataset / normalize_direction(direction)


def method_base_dir(method_name: str, dataset: str, direction: str) -> Path:
    """Alias historique de method_dir pour garder la compatibilite avec l'ancien code."""
    return method_dir(method_name, dataset, direction)
