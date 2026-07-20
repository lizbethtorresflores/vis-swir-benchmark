"""Validations de securite et de coherence pour submissions du benchmark.

Verifie metadata, matrices H, ZIPs d'images alignees, frame IDs officiels,
lisibilite des images et identite visuelle contre les RAW du benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import cv2
import numpy as np

from benchmark_config import DATASETS, DIRECTIONS, IMG_EXTENSIONS, normalize_direction
from benchmark_io import validate_homography


MAX_ALIGNED_ZIP_MB = 200
MAX_SUBMISSION_PAIRS = 100


@dataclass
class UploadedPair:
    """Paire VIS/SWIR extraite d'un ZIP de aligned-image submission."""

    frame_id: int
    vis_path: Path
    swir_path: Path


def official_manifest_frame_ids(manifest: dict) -> list[int]:
    """Retourne les frame_ids officiels declares dans le manifest."""
    frame_ids = [int(value) for value in manifest.get("official_subset_frame_ids", [])]
    if not frame_ids:
        raise ValueError("subset_manifest.json must contain official_subset_frame_ids.")
    return frame_ids


def validate_official_leaderboard_manifest(manifest: dict) -> None:
    """Verifie que le manifest correspond au subset officiel du leaderboard."""
    frame_ids = official_manifest_frame_ids(manifest)
    if len(frame_ids) > MAX_SUBMISSION_PAIRS:
        raise ValueError(f"Official leaderboard subsets are limited to {MAX_SUBMISSION_PAIRS} pairs.")
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("official_subset_frame_ids contains duplicate frames.")
    if manifest.get("evaluation_scope") != "official_leaderboard":
        raise ValueError("Only official_leaderboard subsets are eligible for the official leaderboard.")
    if int(manifest.get("num_subset_pairs", 0)) != len(frame_ids):
        raise ValueError("num_subset_pairs does not match official_subset_frame_ids.")


def validate_common_metadata(metadata: dict, manifest: dict, submission_type: str) -> dict:
    """Valide les metadonnees communes aux deux types de submission."""
    validate_official_leaderboard_manifest(manifest)
    method_name = str(metadata.get("method_name", "")).strip()
    if not method_name:
        raise ValueError("Method Name is required.")
    dataset = metadata.get("dataset")
    if dataset not in DATASETS:
        raise ValueError(f"Invalid dataset: {dataset}")
    direction = normalize_direction(metadata.get("direction", ""))
    subset_id = str(metadata.get("subset_id", "")).strip()
    if not subset_id:
        raise ValueError("subset_id is required.")
    if manifest.get("dataset") != dataset or manifest.get("subset_id") != subset_id:
        raise ValueError("Submission metadata does not match the selected subset manifest.")
    if direction not in DIRECTIONS:
        raise ValueError(f"Invalid direction: {direction}")
    if manifest.get("direction") != direction:
        raise ValueError("Submission direction does not match the selected subset manifest.")
    if int(manifest.get("num_subset_pairs", 0)) > MAX_SUBMISSION_PAIRS:
        raise ValueError(f"Subsets are limited to {MAX_SUBMISSION_PAIRS} pairs.")
    return {
        **metadata,
        "method_name": method_name,
        "dataset": dataset,
        "direction": direction,
        "subset_id": subset_id,
        "submission_type": submission_type,
    }


def _valid_overlap_ratio(mask_shape: tuple[int, int], H: np.ndarray, output_shape: tuple[int, int]) -> float:
    """Calcule le ratio de pixels sources projetes dans l'image de reference."""
    src_h, src_w = mask_shape
    out_h, out_w = output_shape
    src_mask = np.ones((src_h, src_w), dtype=np.uint8) * 255
    warped = cv2.warpPerspective(
        src_mask,
        H,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return float(np.count_nonzero(warped) / max(1, out_h * out_w))


def validate_homography_geometry(
    H: np.ndarray,
    subset_pairs: list[tuple[int, Path, Path]],
    direction: str,
    min_overlap_ratio: float = 0.30,
    max_checked_pairs: int = 8,
) -> dict:
    """Verifie qu'une homographie produit une region visible suffisante."""
    ratios = []
    for _, vis_path, swir_path in subset_pairs[:max_checked_pairs]:
        vis = cv2.imread(str(vis_path), cv2.IMREAD_COLOR)
        swir = cv2.imread(str(swir_path), cv2.IMREAD_GRAYSCALE)
        if vis is None or swir is None:
            raise ValueError(f"Unable to read RAW subset pair: {vis_path.name} / {swir_path.name}")
        if direction == "VIS_to_SWIR":
            ratios.append(_valid_overlap_ratio(vis.shape[:2], H, swir.shape[:2]))
        elif direction == "SWIR_to_VIS":
            ratios.append(_valid_overlap_ratio(swir.shape[:2], H, vis.shape[:2]))
        else:
            raise ValueError(f"Invalid direction for geometry validation: {direction}")
    median_ratio = float(np.median(ratios)) if ratios else 0.0
    if median_ratio < min_overlap_ratio:
        raise ValueError(
            "Invalid homography geometry: applying H leaves too little valid image area "
            f"(median overlap {median_ratio:.3f}, required >= {min_overlap_ratio:.3f})."
        )
    return {
        "checked_pairs": len(ratios),
        "median_valid_overlap_ratio": median_ratio,
        "min_valid_overlap_ratio": float(np.min(ratios)),
        "max_valid_overlap_ratio": float(np.max(ratios)),
    }


def validate_homography_submission(
    metadata: dict,
    H: np.ndarray,
    manifest: dict,
    subset_pairs: list[tuple[int, Path, Path]],
) -> tuple[dict, np.ndarray, dict]:
    """Valide une Homography-based submission avant evaluation."""
    metadata = validate_common_metadata(metadata, manifest, "homography_based")
    H = validate_homography(H)
    geometry = validate_homography_geometry(H, subset_pairs, metadata["direction"])
    return metadata, H, {
        "validation_status": "passed",
        "submission_type": "homography_based",
        "num_pairs": len(subset_pairs),
        "checks": {
            "metadata": "passed",
            "homography_matrix": "passed",
            "homography_geometry": geometry,
        },
    }


def _safe_zip_member(name: str) -> bool:
    """Evita path traversal y entradas vacias al extraer ZIPs enviados."""
    path = Path(name)
    return not path.is_absolute() and ".." not in path.parts and name and not name.endswith("/")


def _uploaded_size_bytes(uploaded_file) -> int:
    """Obtient la taille d'un fichier Streamlit sans perdre sa position de lecture."""
    if hasattr(uploaded_file, "getbuffer"):
        return len(uploaded_file.getbuffer())
    pos = uploaded_file.tell()
    uploaded_file.seek(0, 2)
    size = uploaded_file.tell()
    uploaded_file.seek(pos)
    return size


def save_and_extract_aligned_zip(uploaded_file, upload_zip_path: Path, extract_dir: Path) -> Path:
    """Sauvegarde et extrait un ZIP aligne en verifiant les chemins."""
    if uploaded_file is None:
        raise ValueError("Upload aligned_images.zip before submitting.")
    size_bytes = _uploaded_size_bytes(uploaded_file)
    if size_bytes > MAX_ALIGNED_ZIP_MB * 1024 * 1024:
        raise ValueError(f"Aligned-image ZIP must be <= {MAX_ALIGNED_ZIP_MB} MB.")
    upload_zip_path.parent.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    data = uploaded_file.read()
    upload_zip_path.write_bytes(data)
    try:
        with ZipFile(upload_zip_path) as zf:
            members = [info for info in zf.infolist() if not info.is_dir()]
            for info in members:
                if not _safe_zip_member(info.filename):
                    raise ValueError(f"Unsafe ZIP member path: {info.filename}")
                if Path(info.filename).suffix.lower() not in IMG_EXTENSIONS and Path(info.filename).name != "submission_manifest.json":
                    raise ValueError(f"Unsupported file in ZIP: {info.filename}")
            zf.extractall(extract_dir)
    except BadZipFile as exc:
        raise ValueError("Invalid ZIP file.") from exc
    return extract_dir


def _frame_id_from_name(path: Path) -> int | None:
    """Extrae frame_id desde nombres esperados tipo frame_000123_vis_aligned."""
    import re

    match = re.search(r"frame_(\d+)", path.stem)
    if not match:
        return None
    return int(match.group(1))


def discover_uploaded_aligned_pairs(extract_dir: Path, direction: str) -> list[UploadedPair]:
    """Decouvre les paires finales VIS/SWIR extraites du ZIP."""
    image_files = sorted(p for p in extract_dir.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS)
    if len(image_files) > MAX_SUBMISSION_PAIRS * 2:
        raise ValueError(f"Aligned-image submissions are limited to {MAX_SUBMISSION_PAIRS * 2} images.")
    vis_by_id: dict[int, Path] = {}
    swir_by_id: dict[int, Path] = {}
    for path in image_files:
        frame_id = _frame_id_from_name(path)
        if frame_id is None:
            raise ValueError(f"Invalid file name, expected frame id: {path.name}")
        name = path.stem.lower()
        if direction == "VIS_to_SWIR":
            if name.endswith("_vis_aligned"):
                if frame_id in vis_by_id:
                    raise ValueError(f"Duplicate VIS image for frame_{frame_id:06d}.")
                vis_by_id[frame_id] = path
            elif name.endswith("_swir_reference") or name.endswith("_swir"):
                if frame_id in swir_by_id:
                    raise ValueError(f"Duplicate SWIR image for frame_{frame_id:06d}.")
                swir_by_id[frame_id] = path
            else:
                raise ValueError(
                    "Invalid VIS_to_SWIR file name: "
                    f"{path.name}. Expected frame_xxxxxx_vis_aligned and frame_xxxxxx_swir."
                )
        else:
            if name.endswith("_vis_reference") or name.endswith("_vis"):
                if frame_id in vis_by_id:
                    raise ValueError(f"Duplicate VIS image for frame_{frame_id:06d}.")
                vis_by_id[frame_id] = path
            elif name.endswith("_swir_aligned"):
                if frame_id in swir_by_id:
                    raise ValueError(f"Duplicate SWIR image for frame_{frame_id:06d}.")
                swir_by_id[frame_id] = path
            else:
                raise ValueError(
                    "Invalid SWIR_to_VIS file name: "
                    f"{path.name}. Expected frame_xxxxxx_swir_aligned and frame_xxxxxx_vis."
                )
    frame_ids = sorted(set(vis_by_id) | set(swir_by_id))
    pairs = []
    for frame_id in frame_ids:
        if frame_id not in vis_by_id or frame_id not in swir_by_id:
            raise ValueError(f"Missing VIS/SWIR pair for frame_{frame_id:06d}.")
        pairs.append(UploadedPair(frame_id, vis_by_id[frame_id], swir_by_id[frame_id]))
    if len(pairs) > MAX_SUBMISSION_PAIRS:
        raise ValueError(f"Aligned-image submissions are limited to {MAX_SUBMISSION_PAIRS} pairs.")
    return pairs


def validate_frame_ids_match_manifest(pairs: list[UploadedPair], manifest: dict) -> None:
    """Exige une correspondance exacte entre ZIP aligne et subset_manifest."""
    uploaded = [pair.frame_id for pair in pairs]
    expected = official_manifest_frame_ids(manifest)
    if uploaded != expected:
        missing = sorted(set(expected) - set(uploaded))
        extra = sorted(set(uploaded) - set(expected))
        raise ValueError(
            "Uploaded frame_ids must match subset_manifest.json exactly. "
            f"Missing frames: {missing[:8]}; extra frames: {extra[:8]}; "
            f"expected {len(expected)} frames, got {len(uploaded)} frames."
        )


def validate_uploaded_images(pairs: list[UploadedPair]) -> dict:
    """Verifie que les images finales sont lisibles et compatibles par paire."""
    for pair in pairs:
        vis = cv2.imread(str(pair.vis_path), cv2.IMREAD_COLOR)
        swir = cv2.imread(str(pair.swir_path), cv2.IMREAD_GRAYSCALE)
        if vis is None or swir is None:
            raise ValueError(f"Unreadable aligned image pair for frame_{pair.frame_id:06d}.")
        if vis.shape[:2] != swir.shape[:2]:
            raise ValueError(
                "The submitted final VIS/SWIR pair must share the same dimensions before metric evaluation "
                f"for frame_{pair.frame_id:06d}: "
                f"VIS {vis.shape[:2]}, SWIR {swir.shape[:2]}."
            )
    return {"checked_pairs": len(pairs), "readable_images": len(pairs) * 2}


def _as_gray(image: np.ndarray) -> np.ndarray:
    """Convertit une image BGR en gris si elle ne l'est pas deja."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _resize_for_identity(gray: np.ndarray, max_side: int = 900) -> np.ndarray:
    """Reduit les grandes images pour accelerer la validation d'identite."""
    height, width = gray.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1.0:
        return gray
    return cv2.resize(gray, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def _feature_identity_score(
    uploaded_image: np.ndarray,
    raw_image: np.ndarray,
    min_good_matches: int = 10,
    min_inliers: int = 6,
    min_inlier_ratio: float = 0.15,
) -> dict:
    """Compare deux images du meme capteur sans supposer le meme crop ou la meme resolution."""
    uploaded_gray = _resize_for_identity(_as_gray(uploaded_image))
    raw_gray = _resize_for_identity(_as_gray(raw_image))
    detector = cv2.ORB_create(nfeatures=2500)
    uploaded_kp, uploaded_des = detector.detectAndCompute(uploaded_gray, None)
    raw_kp, raw_des = detector.detectAndCompute(raw_gray, None)
    uploaded_count = 0 if uploaded_kp is None else len(uploaded_kp)
    raw_count = 0 if raw_kp is None else len(raw_kp)
    if uploaded_des is None or raw_des is None or uploaded_count < min_inliers or raw_count < min_inliers:
        return {
            "passed": False,
            "reason": "not enough local features",
            "uploaded_keypoints": uploaded_count,
            "raw_keypoints": raw_count,
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
        }

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = matcher.knnMatch(uploaded_des, raw_des, k=2)
    good = []
    for match_pair in knn_matches:
        if len(match_pair) != 2:
            continue
        first, second = match_pair
        if first.distance < 0.75 * second.distance:
            good.append(first)
    if len(good) < min_good_matches:
        return {
            "passed": False,
            "reason": "not enough robust feature matches",
            "uploaded_keypoints": uploaded_count,
            "raw_keypoints": raw_count,
            "good_matches": len(good),
            "inliers": 0,
            "inlier_ratio": 0.0,
        }
    if len(good) < 4:
        return {
            "passed": False,
            "reason": "not enough matches for homography",
            "uploaded_keypoints": uploaded_count,
            "raw_keypoints": raw_count,
            "good_matches": len(good),
            "inliers": 0,
            "inlier_ratio": 0.0,
        }

    uploaded_pts = np.float32([uploaded_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    raw_pts = np.float32([raw_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    # RANSAC confirme que les correspondances decrivent une geometrie stable du meme frame.
    H, mask = cv2.findHomography(uploaded_pts, raw_pts, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    inlier_ratio = float(inliers / max(1, len(good)))
    passed = H is not None and inliers >= min_inliers and inlier_ratio >= min_inlier_ratio
    return {
        "passed": passed,
        "reason": "passed" if passed else "RANSAC could not support a stable same-frame mapping",
        "uploaded_keypoints": uploaded_count,
        "raw_keypoints": raw_count,
        "good_matches": len(good),
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
    }


def validate_uploaded_frame_identity(
    pairs: list[UploadedPair],
    subset_pairs: list[tuple[int, Path, Path]],
    manifest: dict,
) -> dict:
    """Verifie que chaque image soumise derive du frame RAW officiel correspondant."""
    raw_by_frame = {int(frame_id): (vis_path, swir_path) for frame_id, vis_path, swir_path in subset_pairs}
    vis_inliers = []
    swir_inliers = []
    vis_ratios = []
    swir_ratios = []
    for pair in pairs:
        if pair.frame_id not in raw_by_frame:
            raise ValueError(f"frame_{pair.frame_id:06d} is not part of the selected official subset.")
        raw_vis_path, raw_swir_path = raw_by_frame[pair.frame_id]
        uploaded_vis = cv2.imread(str(pair.vis_path), cv2.IMREAD_COLOR)
        uploaded_swir = cv2.imread(str(pair.swir_path), cv2.IMREAD_GRAYSCALE)
        raw_vis = cv2.imread(str(raw_vis_path), cv2.IMREAD_COLOR)
        raw_swir = cv2.imread(str(raw_swir_path), cv2.IMREAD_GRAYSCALE)
        if uploaded_vis is None or uploaded_swir is None or raw_vis is None or raw_swir is None:
            raise ValueError(f"Unable to read images for identity validation on frame_{pair.frame_id:06d}.")

        # On valide VIS avec VIS et SWIR avec SWIR ; on ne compare pas entre modalites.
        vis_score = _feature_identity_score(
            uploaded_vis,
            raw_vis,
            min_good_matches=100,
            min_inliers=80,
            min_inlier_ratio=0.65,
        )
        if not vis_score["passed"]:
            raise ValueError(
                "Frame identity validation failed for "
                f"frame_{pair.frame_id:06d} VIS image. The uploaded image does not match "
                "the official RAW VIS image with the same frame ID."
            )
        swir_score = _feature_identity_score(
            uploaded_swir,
            raw_swir,
            min_good_matches=100,
            min_inliers=80,
            min_inlier_ratio=0.65,
        )
        if not swir_score["passed"]:
            raise ValueError(
                "Frame identity validation failed for "
                f"frame_{pair.frame_id:06d} SWIR image. The uploaded image does not match "
                "the official RAW SWIR image with the same frame ID."
            )
        vis_inliers.append(vis_score["inliers"])
        swir_inliers.append(swir_score["inliers"])
        vis_ratios.append(vis_score["inlier_ratio"])
        swir_ratios.append(swir_score["inlier_ratio"])

    return {
        "checked_pairs": len(pairs),
        "method": "ORB + RANSAC same-modality same-frame identity check",
        "min_vis_inliers": int(np.min(vis_inliers)) if vis_inliers else 0,
        "min_swir_inliers": int(np.min(swir_inliers)) if swir_inliers else 0,
        "median_vis_inlier_ratio": float(np.median(vis_ratios)) if vis_ratios else 0.0,
        "median_swir_inlier_ratio": float(np.median(swir_ratios)) if swir_ratios else 0.0,
    }


def validate_aligned_image_submission(
    metadata: dict,
    manifest: dict,
    uploaded_pairs: list[UploadedPair],
    subset_pairs: list[tuple[int, Path, Path]],
) -> tuple[dict, dict]:
    """Valide une Aligned-image submission avant evaluation."""
    metadata = validate_common_metadata(metadata, manifest, "aligned_images")
    validate_frame_ids_match_manifest(uploaded_pairs, manifest)
    image_check = validate_uploaded_images(uploaded_pairs)
    identity_check = validate_uploaded_frame_identity(uploaded_pairs, subset_pairs, manifest)
    return metadata, {
        "validation_status": "passed",
        "submission_type": "aligned_images",
        "num_pairs": len(uploaded_pairs),
        "checks": {
            "metadata": "passed",
            "zip_structure": "passed",
            "frame_ids": "passed",
            "images": image_check,
            "frame_identity": identity_check,
        },
    }
