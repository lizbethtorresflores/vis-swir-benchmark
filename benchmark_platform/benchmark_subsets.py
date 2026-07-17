"""Definition et empaquetage des subsets officiels du benchmark.

Controle que toutes les methodes soient evaluees sur les memes 100 frame IDs
officiels par dataset/direction, et genere aussi les ZIPs RAW a telecharger.
"""

from __future__ import annotations

from io import BytesIO
import json
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from benchmark_config import DATASETS, DIRECTIONS, FIXED_TEST_SPLITS, IMG_EXTENSIONS, RESULTS_ROOT, swir_dir, visible_dir


MAX_SUBSET_PAIRS = 100
OFFICIAL_LEADERBOARD_SUBSET_SIZE = 100
FULL_RAW_DOWNLOADS_ROOT = RESULTS_ROOT / "downloads"
OFFICIAL_LEADERBOARD_STEPS = {
    "sunny": 10,
    "cloudy": 10,
    "snowy": 10,
    "rainy": 8,
}


def list_image_files(folder: Path) -> list[Path]:
    """Liste les images d'un dossier dans l'ordre deterministe du benchmark."""
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)


def official_subset_id(dataset: str, direction: str) -> str:
    """Construit l'identifiant du subset officiel du leaderboard."""
    return f"{dataset}_{direction}_official_leaderboard_100"


def subset_index_from_id(dataset: str, subset_id: str) -> int:
    """Extrait l'index numerique d'un subset public."""
    prefix = f"{dataset}_subset_"
    if not subset_id.startswith(prefix):
        raise ValueError(f"Invalid subset_id for {dataset}: {subset_id}")
    try:
        return int(subset_id.replace(prefix, "", 1))
    except ValueError as exc:
        raise ValueError(f"Invalid subset_id: {subset_id}") from exc


def official_subset_frame_ids(dataset: str) -> list[int]:
    """Retourne les 100 frames officiels echantillonnes sur le full test set."""
    if dataset not in DATASETS:
        raise ValueError(f"Invalid dataset: {dataset}")
    split = FIXED_TEST_SPLITS[dataset]
    step = OFFICIAL_LEADERBOARD_STEPS[dataset]
    frame_ids = list(range(split["start"], split["end"] + 1, step))
    return frame_ids[:OFFICIAL_LEADERBOARD_SUBSET_SIZE]


def available_subsets(dataset: str, direction: str | None = None, max_pairs: int = MAX_SUBSET_PAIRS) -> list[dict]:
    """Retourne uniquement le subset officiel disponible pour le leaderboard."""
    if dataset not in DATASETS:
        raise ValueError(f"Invalid dataset: {dataset}")
    direction = direction or "VIS_to_SWIR"
    split = FIXED_TEST_SPLITS[dataset]
    frame_ids = official_subset_frame_ids(dataset)
    return [
        {
            "dataset": dataset,
            "direction": direction,
            "subset_id": official_subset_id(dataset, direction),
            "full_test_start": split["start"],
            "full_test_end": split["end"],
            "num_full_test_pairs": split["end"] - split["start"] + 1,
            "num_pairs": len(frame_ids),
            "first_frame": frame_ids[0],
            "last_frame": frame_ids[-1],
            "sampling_strategy": f"uniform_stride_{OFFICIAL_LEADERBOARD_STEPS[dataset]}_over_full_official_test_set",
        }
    ]


def build_subset_manifest(
    dataset: str,
    subset_id: str,
    direction: str = "VIS_to_SWIR",
    max_pairs: int = MAX_SUBSET_PAIRS,
) -> dict:
    """Construit le manifest du subset officiel du leaderboard."""
    subset = available_subsets(dataset, direction=direction, max_pairs=max_pairs)[0]
    if subset_id != subset["subset_id"]:
        raise ValueError(f"Unknown official leaderboard subset_id for {dataset}/{direction}: {subset_id}")
    frame_ids = official_subset_frame_ids(dataset)
    split = FIXED_TEST_SPLITS[dataset]
    return {
        "dataset": dataset,
        "direction": direction,
        "subset_id": subset_id,
        "full_test_start": split["start"],
        "full_test_end": split["end"],
        "num_full_test_pairs": split["end"] - split["start"] + 1,
        "official_subset_frame_ids": frame_ids,
        "num_subset_pairs": len(frame_ids),
        "sampling_strategy": subset["sampling_strategy"],
        "evaluation_scope": "official_leaderboard",
        "vis_dir": "VIS",
        "swir_dir": "SWIR",
        "allowed_submission_types": ["homography_based", "aligned_images"],
    }


def official_manifest_path(dataset: str, direction: str) -> Path:
    """Retourne le chemin du manifest officiel materialise sur disque."""
    return RESULTS_ROOT / "subsets" / dataset / direction / "subset_manifest.json"


def write_official_subset_manifest(dataset: str, direction: str) -> Path:
    """Ecrit le subset_manifest.json officiel pour un dataset et une direction."""
    subset_id = official_subset_id(dataset, direction)
    manifest = build_subset_manifest(dataset, subset_id, direction=direction)
    path = official_manifest_path(dataset, direction)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def ensure_official_subset_manifests() -> list[Path]:
    """Cree tous les manifests officiels dataset/direction si necessaire."""
    paths = []
    for dataset in DATASETS:
        for direction in DIRECTIONS:
            paths.append(write_official_subset_manifest(dataset, direction))
    return paths


def build_custom_subset_manifest(dataset: str, direction: str, frame_ids: list[int], subset_id: str) -> dict:
    """Construit un manifest custom non eligible au leaderboard officiel."""
    if dataset not in DATASETS:
        raise ValueError(f"Invalid dataset: {dataset}")
    split = FIXED_TEST_SPLITS[dataset]
    if len(frame_ids) > MAX_SUBSET_PAIRS:
        raise ValueError(f"Custom evaluation is limited to {MAX_SUBSET_PAIRS} pairs.")
    return {
        "dataset": dataset,
        "direction": direction,
        "subset_id": subset_id,
        "full_test_start": split["start"],
        "full_test_end": split["end"],
        "num_full_test_pairs": split["end"] - split["start"] + 1,
        "official_subset_frame_ids": frame_ids,
        "num_subset_pairs": len(frame_ids),
        "sampling_strategy": "custom_not_leaderboard_eligible",
        "evaluation_scope": "custom",
        "vis_dir": "VIS",
        "swir_dir": "SWIR",
        "allowed_submission_types": ["homography_based", "aligned_images"],
    }


def get_subset_pairs(
    dataset: str,
    subset_id: str,
    direction: str = "VIS_to_SWIR",
    max_pairs: int = MAX_SUBSET_PAIRS,
) -> list[tuple[int, Path, Path]]:
    """Retourne les chemins RAW VIS/SWIR pour le subset officiel."""
    manifest = build_subset_manifest(dataset, subset_id, direction=direction, max_pairs=max_pairs)
    vis_files = list_image_files(visible_dir(dataset))
    swir_files = list_image_files(swir_dir(dataset))
    vis_by_frame = {
        int(match.group(1)): path
        for path in vis_files
        if (match := re.search(r"(\d+)$", path.stem)) is not None
    }
    swir_by_frame = {
        int(match.group(1)): path
        for path in swir_files
        if (match := re.search(r"(\d+)$", path.stem)) is not None
    }
    max_len = min(len(vis_files), len(swir_files))
    pairs = []
    for frame_id in manifest["official_subset_frame_ids"]:
        if frame_id in vis_by_frame and frame_id in swir_by_frame:
            pairs.append((frame_id, vis_by_frame[frame_id], swir_by_frame[frame_id]))
        elif frame_id < max_len:
            # Backward-compatible fallback for datasets whose filenames do not
            # contain their original frame index.
            pairs.append((frame_id, vis_files[frame_id], swir_files[frame_id]))
    if not pairs:
        raise FileNotFoundError(f"No RAW VIS/SWIR pairs were found for {dataset} / {subset_id}.")
    return pairs


def get_full_raw_dataset_files(dataset: str) -> tuple[list[Path], list[Path]]:
    """Retourne les fichiers RAW complets VISIBLE et SWIR d'un dataset."""
    if dataset not in DATASETS:
        raise ValueError(f"Invalid dataset: {dataset}")
    vis_files = list_image_files(visible_dir(dataset))
    swir_files = list_image_files(swir_dir(dataset))
    if not vis_files or not swir_files:
        raise FileNotFoundError(f"No RAW VISIBLE/SWIR files were found for {dataset}.")
    return vis_files, swir_files


def build_full_raw_dataset_manifest(dataset: str) -> dict:
    """Construit un manifest informatif pour le full RAW unaligned dataset."""
    if dataset not in DATASETS:
        raise ValueError(f"Invalid dataset: {dataset}")
    split = FIXED_TEST_SPLITS[dataset]
    official_frames = official_subset_frame_ids(dataset)
    vis_files, swir_files = get_full_raw_dataset_files(dataset)
    return {
        "dataset": dataset,
        "directions": DIRECTIONS,
        "download_type": "full_raw_unaligned_dataset",
        "visible_dir": str(visible_dir(dataset)),
        "swir_dir": str(swir_dir(dataset)),
        "num_visible_images": len(vis_files),
        "num_swir_images": len(swir_files),
        "full_test_start": split["start"],
        "full_test_end": split["end"],
        "num_full_test_pairs": split["end"] - split["start"] + 1,
        "official_scoring_subset_ids": {
            direction: official_subset_id(dataset, direction)
            for direction in DIRECTIONS
        },
        "official_scoring_frame_ids": official_frames,
        "num_official_scoring_pairs": len(official_frames),
        "sampling_strategy": f"uniform_stride_{OFFICIAL_LEADERBOARD_STEPS[dataset]}_over_full_official_test_set",
        "note": (
            "Use any number of available pairs from this full RAW unaligned dataset to estimate one "
            "single global homography. The official leaderboard scores only the official_scoring_frame_ids."
        ),
    }


def full_raw_dataset_zip_path(dataset: str) -> Path:
    """Retourne le chemin du ZIP RAW complet pregénéré."""
    if dataset not in DATASETS:
        raise ValueError(f"Invalid dataset: {dataset}")
    return FULL_RAW_DOWNLOADS_ROOT / f"{dataset}_full_raw_unaligned_VIS_SWIR.zip"


def create_raw_full_dataset_zip_file(dataset: str, overwrite: bool = False) -> Path:
    """Cree sur disque le ZIP RAW complet s'il n'existe pas deja."""
    output_path = full_raw_dataset_zip_path(dataset)
    if output_path.exists() and not overwrite:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp")
    manifest = build_full_raw_dataset_manifest(dataset)
    vis_files, swir_files = get_full_raw_dataset_files(dataset)
    root = f"{dataset}_raw_unaligned"
    with ZipFile(tmp_path, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr(f"{root}/full_raw_dataset_manifest.json", json.dumps(manifest, indent=2))
        for vis_path in vis_files:
            zf.write(vis_path, f"{root}/VISIBLE/{vis_path.name}")
        for swir_path in swir_files:
            zf.write(swir_path, f"{root}/SWIR/{swir_path.name}")
    tmp_path.replace(output_path)
    return output_path


def create_raw_full_dataset_zip_bytes(dataset: str) -> bytes:
    """Cree en memoire un ZIP du full RAW unaligned dataset."""
    manifest = build_full_raw_dataset_manifest(dataset)
    vis_files, swir_files = get_full_raw_dataset_files(dataset)
    root = f"{dataset}_raw_unaligned"
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr(f"{root}/full_raw_dataset_manifest.json", json.dumps(manifest, indent=2))
        for vis_path in vis_files:
            zf.write(vis_path, f"{root}/VISIBLE/{vis_path.name}")
        for swir_path in swir_files:
            zf.write(swir_path, f"{root}/SWIR/{swir_path.name}")
    return buffer.getvalue()


def create_raw_subset_zip_bytes(
    dataset: str,
    subset_id: str,
    direction: str = "VIS_to_SWIR",
    max_pairs: int = MAX_SUBSET_PAIRS,
) -> bytes:
    """Cree en memoire un ZIP RAW VIS/SWIR avec son subset_manifest.json."""
    manifest = build_subset_manifest(dataset, subset_id, direction=direction, max_pairs=max_pairs)
    pairs = get_subset_pairs(dataset, subset_id, direction=direction, max_pairs=max_pairs)
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("subset_manifest.json", json.dumps(manifest, indent=2))
        for frame_id, vis_path, swir_path in pairs:
            zf.write(vis_path, f"VIS/frame_{frame_id:06d}_vis{vis_path.suffix.lower()}")
            zf.write(swir_path, f"SWIR/frame_{frame_id:06d}_swir{swir_path.suffix.lower()}")
    return buffer.getvalue()
