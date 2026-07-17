"""Moteur d'evaluation officiel du benchmark VIS-SWIR.

Applique des homographies ou consomme des images deja alignees, sauvegarde les
visualisations et calcule GOA, Edge Alignment, MIND-SSD et reprojection directement sur split_images.
"""

from __future__ import annotations

import time
from pathlib import Path
import sys
import re

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJET_ROOT = ROOT / "projet"
for path in [ROOT, PROJET_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_config import (
    EVALUATION_PROTOCOL,
    FIXED_TEST_SPLITS,
    IMG_EXTENSIONS,
    paired_images_dir,
    swir_dir,
    visible_dir,
)
from benchmark_validation import UploadedPair
from crop_patch_auto_like_rasmd import as_bgr, estimate_crop_for_pair, resize, select_reference_indices


def _largest_valid_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Retourne le plus grand rectangle compose uniquement de pixels valides."""
    h, w = mask.shape
    heights = np.zeros(w, dtype=np.int32)
    best_area = 0
    best_rect = (0, 0, w, h)

    for y in range(h):
        heights += mask[y]
        heights[mask[y] == 0] = 0
        stack: list[int] = []
        for x in range(w + 1):
            current = int(heights[x]) if x < w else 0
            while stack and current < heights[stack[-1]]:
                top = stack.pop()
                height = int(heights[top])
                left = stack[-1] + 1 if stack else 0
                area = height * (x - left)
                if area > best_area:
                    best_area = area
                    best_rect = (left, y - height + 1, x, y + 1)
            stack.append(x)

    return best_rect


def _homography_global_valid_crop(
    H: np.ndarray,
    subset_pairs: list[tuple[int, Path, Path]],
    direction: str,
    margin: int = 12,
) -> tuple[int, int, int, int]:
    """Calcule le crop geometrique global sans deduire la validite de la luminosite.

    Comme dans l'application Patch, une image binaire pleine est transformee par
    H. L'intersection des masques de toutes les paires garantit qu'un seul crop
    est utilise pour tout le subset officiel.
    """
    common_mask: np.ndarray | None = None

    for _, vis_path, swir_path in subset_pairs:
        vis = cv2.imread(str(vis_path), cv2.IMREAD_COLOR)
        swir = cv2.imread(str(swir_path), cv2.IMREAD_GRAYSCALE)
        if vis is None or swir is None:
            continue

        if direction == "VIS_to_SWIR":
            out_h, out_w = swir.shape[:2]
            source_shape = vis.shape[:2]
        elif direction == "SWIR_to_VIS":
            out_h, out_w = vis.shape[:2]
            source_shape = swir.shape[:2]
        else:
            raise ValueError(f"Unsupported direction: {direction}")

        source_mask = np.full(source_shape, 255, dtype=np.uint8)
        valid_mask = cv2.warpPerspective(
            source_mask,
            H,
            (out_w, out_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid_mask = (valid_mask > 0).astype(np.uint8)
        if common_mask is None:
            common_mask = valid_mask
        elif common_mask.shape != valid_mask.shape:
            raise ValueError("Official RAW target images do not share one output size.")
        else:
            common_mask &= valid_mask

    if common_mask is None or not np.any(common_mask):
        raise ValueError("The homography leaves no valid common image region.")

    if margin > 0:
        kernel_size = 2 * margin + 1
        eroded = cv2.erode(
            common_mask,
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
            iterations=1,
        )
        if np.any(eroded):
            common_mask = eroded

    crop_rect = _largest_valid_rectangle(common_mask)
    x0, y0, x1, y1 = crop_rect
    if x1 <= x0 or y1 <= y0:
        raise ValueError("No rectangular valid region could be obtained from the homography.")
    return crop_rect


def _import_quality_functions():
    """Importe les metriques lourdes seulement au moment de l'evaluation."""
    from projet.registration.evaluation.quality_bench import (
        compute_edge_alignment_metrics,
        compute_goa,
        compute_mind_map_fast,
        preprocess_pair_bench,
        resolve_device,
        save_single_mind_map,
    )

    return (
        compute_edge_alignment_metrics,
        compute_goa,
        compute_mind_map_fast,
        preprocess_pair_bench,
        resolve_device,
        save_single_mind_map,
    )


def _import_reprojection_aligned_functions():
    """Importe SuperPoint/LightGlue a la demande pour eviter un cout au lancement de l'UI."""
    from reprojection_aligned import (
        balance_matches_by_region,
        compute_direct_distance_metrics,
        detect_and_match_superpoint_lightglue,
        init_superpoint_lightglue,
        remove_sky_matches,
    )

    return (
        balance_matches_by_region,
        compute_direct_distance_metrics,
        detect_and_match_superpoint_lightglue,
        init_superpoint_lightglue,
        remove_sky_matches,
    )


class ReprojectionAlignedEvaluator:
    """Calcule la reprojection avec le meme protocole que reprojection_aligned.py."""

    def __init__(
        self,
        max_keypoints: int = 2048,
        use_ransac: bool = True,
        ransac_thresh: float = 5.0,
    ) -> None:
        import torch

        (
            self.balance_matches_by_region,
            self.compute_direct_distance_metrics,
            self.detect_and_match_superpoint_lightglue,
            self.init_superpoint_lightglue,
            self.remove_sky_matches,
        ) = _import_reprojection_aligned_functions()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extractor, self.matcher = self.init_superpoint_lightglue(self.device, max_keypoints)
        self.use_ransac = use_ransac
        self.ransac_thresh = ransac_thresh

    def compute(self, vis_path: Path, swir_path: Path) -> dict:
        pts_vis, pts_swir, _ = self.detect_and_match_superpoint_lightglue(
            str(vis_path),
            str(swir_path),
            self.extractor,
            self.matcher,
            self.device,
        )
        pts_vis, pts_swir = self.remove_sky_matches(
            str(vis_path),
            pts_vis,
            pts_swir,
            self.device,
        )
        pts_vis, pts_swir = self.balance_matches_by_region(
            pts_vis,
            pts_swir,
            num_rows=2,
            num_cols=3,
            max_matches_per_region=180,
            min_matches_per_region=20,
        )
        return self.compute_direct_distance_metrics(
            pts_vis,
            pts_swir,
            use_ransac=self.use_ransac,
            ransac_thresh=self.ransac_thresh,
        )


def list_image_files(folder: Path) -> list[Path]:
    """Liste les images dans un ordre deterministe."""
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)


def fixed_test_pairs(dataset: str) -> list[tuple[int, Path, Path]]:
    """Retourne toutes les paires du split fixe historique pour un dataset."""
    vis_files = list_image_files(visible_dir(dataset))
    swir_files = list_image_files(swir_dir(dataset))
    max_len = min(len(vis_files), len(swir_files))
    if max_len == 0:
        raise FileNotFoundError(f"No VIS/SWIR pairs were found for {dataset}.")

    split = FIXED_TEST_SPLITS[dataset]
    start = min(split["start"], max_len - 1)
    end = min(split["end"], max_len - 1)
    return [(idx, vis_files[idx], swir_files[idx]) for idx in range(start, end + 1)]


def _save_pair_outputs(
    method_path: Path,
    direction: str,
    frame_id: int,
    vis_bgr: np.ndarray,
    swir_gray: np.ndarray,
    H: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Path, Path]:
    """Applique H, sauvegarde split/overlay et retourne les images utilisees par les metriques."""
    split_dir = method_path / "split_images"
    concat_dir = method_path / "concatenate_images"
    split_dir.mkdir(parents=True, exist_ok=True)
    concat_dir.mkdir(parents=True, exist_ok=True)
    base = f"frame_{frame_id:06d}"

    if direction == "VIS_to_SWIR":
        h, w = swir_gray.shape[:2]
        aligned = cv2.warpPerspective(vis_bgr, H, (w, h), flags=cv2.INTER_LINEAR)
        vis_path = split_dir / f"{base}_vis_aligned.jpg"
        swir_path = split_dir / f"{base}_swir.jpg"
        cv2.imwrite(str(vis_path), aligned)
        cv2.imwrite(str(swir_path), swir_gray)
        swir_bgr = cv2.cvtColor(swir_gray, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(aligned, 0.5, swir_bgr, 0.5, 0)
        metric_vis, metric_swir = aligned, swir_gray
    else:
        h, w = vis_bgr.shape[:2]
        aligned = cv2.warpPerspective(swir_gray, H, (w, h), flags=cv2.INTER_LINEAR)
        vis_path = split_dir / f"{base}_vis.jpg"
        swir_path = split_dir / f"{base}_swir_aligned.jpg"
        cv2.imwrite(str(vis_path), vis_bgr)
        cv2.imwrite(str(swir_path), aligned)
        swir_bgr = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(vis_bgr, 0.5, swir_bgr, 0.5, 0)
        metric_vis, metric_swir = vis_bgr, aligned

    cv2.imwrite(str(concat_dir / f"{base}_overlay.jpg"), overlay)
    return metric_vis, metric_swir, vis_path, swir_path


def _save_official_crop_outputs(
    method_path: Path,
    direction: str,
    frame_id: int,
    metric_vis: np.ndarray,
    metric_swir: np.ndarray,
) -> tuple[Path, Path]:
    """Sauvegarde les images deja recadrees par le protocole Official Common Crop."""
    split_dir = method_path / "split_images_official_common_crop"
    concat_dir = method_path / "concatenate_images_official_common_crop"
    split_dir.mkdir(parents=True, exist_ok=True)
    concat_dir.mkdir(parents=True, exist_ok=True)
    base = f"frame_{frame_id:06d}"

    if direction == "VIS_to_SWIR":
        vis_path = split_dir / f"{base}_vis_aligned.jpg"
        swir_path = split_dir / f"{base}_swir.jpg"
        cv2.imwrite(str(vis_path), metric_vis)
        cv2.imwrite(str(swir_path), metric_swir)
        swir_bgr = cv2.cvtColor(metric_swir, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(metric_vis, 0.5, swir_bgr, 0.5, 0)
    else:
        vis_path = split_dir / f"{base}_vis.jpg"
        swir_path = split_dir / f"{base}_swir_aligned.jpg"
        cv2.imwrite(str(vis_path), metric_vis)
        cv2.imwrite(str(swir_path), metric_swir)
        swir_bgr = cv2.cvtColor(metric_swir, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(metric_vis, 0.5, swir_bgr, 0.5, 0)

    cv2.imwrite(str(concat_dir / f"{base}_overlay.jpg"), overlay)
    return vis_path, swir_path


def _frame_id_from_path(path: Path) -> int | None:
    """Extrae frame_id de nombres tipo frame_000123_*.jpg."""
    match = re.search(r"(\d{6})", path.name)
    return int(match.group(1)) if match else None


def _paths_by_frame_id(folder: Path, pattern: str) -> dict[int, Path]:
    """Indexe les fichiers par frame_id pour croiser VIS, SWIR et references RASMD."""
    paths: dict[int, Path] = {}
    for path in sorted(folder.glob(pattern)):
        frame_id = _frame_id_from_path(path)
        if frame_id is not None:
            paths[frame_id] = path
    return paths


def _rasmd_split_dir(dataset: str) -> Path:
    """Localiza la referencia RASMD usada para definir el crop comun oficial."""
    split_dir = paired_images_dir(dataset) / "RASMD_aligned_baseline" / "split_images"
    if not split_dir.exists():
        raise FileNotFoundError(f"RASMD split_images folder was not found: {split_dir}")
    return split_dir


def apply_official_common_crop_like_rasmd(
    method_path: Path,
    dataset: str,
    direction: str,
    frame_ids: list[int],
    log_lines: list[str] | None = None,
    num_refs: int = 20,
    min_score: float = 0.50,
) -> dict[int, tuple[Path, Path, tuple[int, int, int, int]]]:
    """Applique le crop officiel avec la logique de crop_patch_auto_like_rasmd.py."""
    method_split = method_path / "split_images"
    out_split = method_path / "split_images_official_common_crop"
    out_concat = method_path / "concatenate_images_official_common_crop"
    out_split.mkdir(parents=True, exist_ok=True)
    out_concat.mkdir(parents=True, exist_ok=True)
    # On nettoie les anciennes sorties pour qu'une reevaluation ne melange pas d'anciens frames.
    for old_file in list(out_split.glob("*")) + list(out_concat.glob("*")):
        if old_file.is_file():
            old_file.unlink()

    rasmd_split = _rasmd_split_dir(dataset)
    rasmd_swir = _paths_by_frame_id(rasmd_split, "*_swir.*")

    if direction == "VIS_to_SWIR":
        aligned_paths = _paths_by_frame_id(method_split, "*_vis_aligned.*")
        vis_paths = aligned_paths
        swir_paths = _paths_by_frame_id(method_split, "*_swir.*")
    elif direction == "SWIR_to_VIS":
        aligned_paths = _paths_by_frame_id(method_split, "*_swir_aligned.*")
        vis_paths = _paths_by_frame_id(method_split, "*_vis.*")
        swir_paths = aligned_paths
    else:
        raise ValueError(f"Unsupported direction for RASMD-like crop: {direction}")

    common_ids = sorted(set(frame_ids) & set(aligned_paths) & set(vis_paths) & set(swir_paths) & set(rasmd_swir))
    if not common_ids:
        raise RuntimeError("No common frame IDs between method split_images and RASMD baseline.")

    first_rasmd = cv2.imread(str(rasmd_swir[common_ids[0]]), cv2.IMREAD_UNCHANGED)
    if first_rasmd is None:
        raise RuntimeError(f"Unable to read RASMD reference image: {rasmd_swir[common_ids[0]]}")
    rh, rw = first_rasmd.shape[:2]

    estimates = []
    ref_ids = [common_ids[idx] for idx in select_reference_indices(len(common_ids), num_refs)]
    for frame_id in ref_ids:
        est = estimate_crop_for_pair(aligned_paths[frame_id], rasmd_swir[frame_id], rw, rh)
        if est is not None and est["score"] >= min_score:
            estimates.append(est)
    if not estimates:
        raise RuntimeError(
            "No valid RASMD-like crop estimate was found. "
            "Check the aligned images or lower the crop min_score."
        )

    # On utilise la mediane pour qu'un frame difficile ne deplace pas le crop officiel.
    scale_final = float(np.median([estimate["scale"] for estimate in estimates]))
    x0 = int(round(np.median([estimate["x0"] for estimate in estimates])))
    y0 = int(round(np.median([estimate["y0"] for estimate in estimates])))
    x1 = x0 + rw
    y1 = y0 + rh

    if log_lines is not None:
        log_lines.append(
            "Official Common Crop: crop_patch_auto_like_rasmd "
            f"scale={scale_final:.6f}, x={x0}:{x1}, y={y0}:{y1}, "
            f"valid_refs={len(estimates)}/{len(ref_ids)}"
        )

    crop_info = {
        "protocol": "crop_patch_auto_like_rasmd",
        "dataset": dataset,
        "direction": direction,
        "scale": scale_final,
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "width": rw,
        "height": rh,
        "num_valid_refs": len(estimates),
        "num_refs": len(ref_ids),
        "min_score": min_score,
    }
    (method_path / "official_common_crop_like_rasmd.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in crop_info.items()),
        encoding="utf-8",
    )

    output_paths: dict[int, tuple[Path, Path, tuple[int, int, int, int]]] = {}
    for frame_id in common_ids:
        vis = cv2.imread(str(vis_paths[frame_id]), cv2.IMREAD_UNCHANGED)
        swir = cv2.imread(str(swir_paths[frame_id]), cv2.IMREAD_UNCHANGED)
        if vis is None or swir is None:
            continue
        vis_scaled = resize(vis, scale_final)
        swir_scaled = resize(swir, scale_final)
        sh, sw = swir_scaled.shape[:2]
        if x1 > sw or y1 > sh:
            if log_lines is not None:
                log_lines.append(f"[WARN] RASMD-like crop out of bounds for frame {frame_id}")
            continue

        vis_crop = vis_scaled[y0:y1, x0:x1]
        swir_crop = swir_scaled[y0:y1, x0:x1]
        base = f"frame_{frame_id:06d}"
        if direction == "VIS_to_SWIR":
            out_vis = out_split / f"{base}_vis_aligned.jpg"
            out_swir = out_split / f"{base}_swir.jpg"
        else:
            out_vis = out_split / f"{base}_vis.jpg"
            out_swir = out_split / f"{base}_swir_aligned.jpg"
        cv2.imwrite(str(out_vis), vis_crop)
        cv2.imwrite(str(out_swir), swir_crop)
        overlay = cv2.addWeighted(as_bgr(vis_crop), 0.5, as_bgr(swir_crop), 0.5, 0)
        cv2.imwrite(str(out_concat / f"{base}_overlay.jpg"), overlay)
        output_paths[frame_id] = (out_vis, out_swir, (x0, y0, x1, y1))

    return output_paths


def compute_metric_row(
    frame_id: int,
    metric_vis: np.ndarray,
    metric_swir: np.ndarray,
    reprojection_metrics: dict,
    compute_edge_alignment_metrics,
    compute_goa,
    compute_mind_map_fast,
    preprocess_pair_bench,
    resolved_device: str,
    crop_rect: tuple[int, int, int, int] | None = None,
    mind_maps: list[np.ndarray] | None = None,
) -> dict:
    """Calcule une ligne de metriques officielles pour un frame deja aligne/recadre."""
    vis_gray, swir_warped, valid_mask = preprocess_pair_bench(
        metric_vis,
        metric_swir,
        np.eye(3, dtype=np.float64),
        device=resolved_device,
    )
    goa = compute_goa(vis_gray, swir_warped, valid_mask)
    edge_metrics = compute_edge_alignment_metrics(vis_gray, swir_warped, valid_mask, tol_px=8)
    mind_map = compute_mind_map_fast(vis_gray, swir_warped, valid_mask, resolved_device)
    mind = float(np.nanmean(mind_map)) if np.isfinite(mind_map).any() else float("nan")
    if mind_maps is not None:
        mind_maps.append(mind_map.astype(np.float32))
    row = {
        "frame_id": frame_id,
        "pair_id": f"{frame_id:06d}",
        "goa": goa,
        "edge_f1_8px": edge_metrics["edge_f1_8px"],
        "mind_ssd": mind,
        "reprojection_error": reprojection_metrics.get("mean_error", float("nan")),
        "reprojection_median_error": reprojection_metrics.get("median_error", float("nan")),
        "reprojection_num_matches": reprojection_metrics.get("num_matches", 0),
        "reprojection_num_used": reprojection_metrics.get("num_used", 0),
        "reprojection_inlier_ratio": reprojection_metrics.get("inlier_ratio", float("nan")),
    }
    if crop_rect is not None:
        x0, y0, x1, y1 = crop_rect
        row.update(
            {
                "official_crop_x0": x0,
                "official_crop_y0": y0,
                "official_crop_x1": x1,
                "official_crop_y1": y1,
                "evaluation_protocol": EVALUATION_PROTOCOL,
            }
        )
    return row


def save_mean_mind_map(mind_maps: list[np.ndarray], method_path: Path, save_single_mind_map) -> None:
    """Sauvegarde la carte MIND moyenne officielle avec le format de quality_bench.py."""
    if not mind_maps:
        return
    ref_shape = mind_maps[0].shape
    resized = [
        cv2.resize(
            mind_map.astype(np.float32),
            (ref_shape[1], ref_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        if mind_map.shape != ref_shape
        else mind_map.astype(np.float32)
        for mind_map in mind_maps
    ]
    mean_map = np.nanmean(np.stack(resized, axis=0), axis=0)
    save_single_mind_map(mean_map, method_path / "results")


def _save_aligned_pair_outputs(
    method_path: Path,
    direction: str,
    frame_id: int,
    metric_vis: np.ndarray,
    metric_swir: np.ndarray,
) -> tuple[Path, Path]:
    """Sauvegarde une submission d'images alignees dans le format visuel standard."""
    split_dir = method_path / "split_images"
    concat_dir = method_path / "concatenate_images"
    split_dir.mkdir(parents=True, exist_ok=True)
    concat_dir.mkdir(parents=True, exist_ok=True)
    base = f"frame_{frame_id:06d}"
    swir_bgr = cv2.cvtColor(metric_swir, cv2.COLOR_GRAY2BGR) if metric_swir.ndim == 2 else metric_swir
    overlay = cv2.addWeighted(metric_vis, 0.5, swir_bgr, 0.5, 0)

    if direction == "VIS_to_SWIR":
        vis_path = split_dir / f"{base}_vis_aligned.jpg"
        swir_path = split_dir / f"{base}_swir.jpg"
        cv2.imwrite(str(vis_path), metric_vis)
        cv2.imwrite(str(swir_path), metric_swir)
    else:
        vis_path = split_dir / f"{base}_vis.jpg"
        swir_path = split_dir / f"{base}_swir_aligned.jpg"
        cv2.imwrite(str(vis_path), metric_vis)
        cv2.imwrite(str(swir_path), metric_swir)
    cv2.imwrite(str(concat_dir / f"{base}_overlay.jpg"), overlay)
    return vis_path, swir_path


def evaluate_homography_submission(
    method_path: Path,
    metadata: dict,
    H: np.ndarray,
    subset_pairs: list[tuple[int, Path, Path]],
    device: str = "auto",
    save_outputs: bool = True,
    progress_callback=None,
) -> tuple[pd.DataFrame, dict]:
    """Evalue une submission basee sur H_global sur un subset RAW."""
    (
        compute_edge_alignment_metrics,
        compute_goa,
        compute_mind_map_fast,
        preprocess_pair_bench,
        resolve_device,
        save_single_mind_map,
    ) = _import_quality_functions()
    resolved_device = resolve_device(device)
    rows = []
    diagnostic_rows = []
    official_mind_maps: list[np.ndarray] = []
    direction = metadata["direction"]
    dataset = metadata["dataset"]
    subset_id = metadata.get("subset_id", "")
    crop_rect = _homography_global_valid_crop(H, subset_pairs, direction)
    crop_x0, crop_y0, crop_x1, crop_y1 = crop_rect
    log_lines = [
        f"Dataset: {dataset}",
        f"Direction: {direction}",
        f"Subset: {subset_id}",
        f"Evaluated pairs: {len(subset_pairs)}",
        f"Submission type: {metadata.get('submission_type', 'homography_based')}",
        f"Device: {resolved_device}",
        (
            "Geometric valid crop after homography: "
            f"x={crop_x0}:{crop_x1}, y={crop_y0}:{crop_y1}, "
            f"size={crop_x1 - crop_x0}x{crop_y1 - crop_y0}"
        ),
        "The crop removes only invalid warp borders; it does not remove sky or vehicle content.",
        f'"evaluation_protocol": "{EVALUATION_PROTOCOL}"',
    ]
    reprojection_evaluator = ReprojectionAlignedEvaluator()
    log_lines.append(
        f"Reprojection Error: reprojection_aligned.py protocol "
        f"(SuperPoint + LightGlue, device={reprojection_evaluator.device})"
    )
    t0 = time.time()
    evaluated_frame_ids = []
    progress_total = max(1, len(subset_pairs) * 2)
    for n, (frame_id, vis_path, swir_path) in enumerate(subset_pairs):
        if progress_callback is not None:
            progress_callback(n + 1, progress_total, f"Aligning and cropping frame {n + 1}/{len(subset_pairs)}")
        # Appliquer H aux images RAW, puis le meme crop geometrique a toute la serie.
        vis = cv2.imread(str(vis_path), cv2.IMREAD_COLOR)
        swir = cv2.imread(str(swir_path), cv2.IMREAD_GRAYSCALE)
        if vis is None or swir is None:
            log_lines.append(f"[WARN] lectura fallida frame {frame_id}")
            continue
        evaluated_frame_ids.append(frame_id)
        if direction == "VIS_to_SWIR":
            h, w = swir.shape[:2]
            metric_vis = cv2.warpPerspective(vis, H, (w, h), flags=cv2.INTER_LINEAR)
            metric_swir = swir
        else:
            h, w = vis.shape[:2]
            metric_vis = vis
            metric_swir = cv2.warpPerspective(swir, H, (w, h), flags=cv2.INTER_LINEAR)
        metric_vis = metric_vis[crop_y0:crop_y1, crop_x0:crop_x1]
        metric_swir = metric_swir[crop_y0:crop_y1, crop_x0:crop_x1]
        diagnostic_vis_path, diagnostic_swir_path = _save_aligned_pair_outputs(
            method_path,
            direction,
            frame_id,
            metric_vis,
            metric_swir,
        )
        diagnostic_reprojection = reprojection_evaluator.compute(diagnostic_vis_path, diagnostic_swir_path)
        diagnostic_rows.append(
            compute_metric_row(
                frame_id,
                metric_vis,
                metric_swir,
                diagnostic_reprojection,
                compute_edge_alignment_metrics,
                compute_goa,
                compute_mind_map_fast,
                preprocess_pair_bench,
                resolved_device,
                mind_maps=official_mind_maps,
            )
        )
    rows = diagnostic_rows
    elapsed = time.time() - t0
    fps = len(rows) / elapsed if elapsed > 0 else 0.0
    runtime = {"runtime_seconds": elapsed, "fps": fps, "device": resolved_device}
    metrics = pd.DataFrame(rows)
    metrics.to_csv(method_path / "metrics.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(method_path / "metrics_full_image_diagnostic.csv", index=False)
    save_mean_mind_map(official_mind_maps, method_path, save_single_mind_map)
    (method_path / "evaluation_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    return metrics, runtime


def evaluate_aligned_image_submission(
    method_path: Path,
    metadata: dict,
    uploaded_pairs: list[UploadedPair],
    device: str = "auto",
    save_outputs: bool = True,
    progress_callback=None,
) -> tuple[pd.DataFrame, dict]:
    """Evalue une submission d'images deja alignees sans appliquer d'homographie."""
    (
        compute_edge_alignment_metrics,
        compute_goa,
        compute_mind_map_fast,
        preprocess_pair_bench,
        resolve_device,
        save_single_mind_map,
    ) = _import_quality_functions()
    resolved_device = resolve_device(device)
    rows = []
    official_mind_maps: list[np.ndarray] = []
    direction = metadata["direction"]
    dataset = metadata["dataset"]
    log_lines = [
        f"Dataset: {dataset}",
        f"Direction: {direction}",
        f"Subset: {metadata.get('subset_id', '')}",
        f"Evaluated pairs: {len(uploaded_pairs)}",
        f"Submission type: {metadata.get('submission_type', 'aligned_images')}",
        f"Device: {resolved_device}",
        f'"evaluation_protocol": "{EVALUATION_PROTOCOL}"',
        "Benchmark did not apply a homography for this submission.",
    ]
    reprojection_evaluator = ReprojectionAlignedEvaluator()
    log_lines.append(
        f"Reprojection Error: reprojection_aligned.py protocol "
        f"(SuperPoint + LightGlue, device={reprojection_evaluator.device})"
    )
    t0 = time.time()
    evaluated_frame_ids = []
    progress_total = max(1, len(uploaded_pairs) * 2)
    for n, pair in enumerate(uploaded_pairs):
        if progress_callback is not None:
            progress_callback(n + 1, progress_total, f"Saving aligned frame {n + 1}/{len(uploaded_pairs)}")
        # Les images arrivent deja alignees ; ici on les garde seulement au format standard.
        metric_vis = cv2.imread(str(pair.vis_path), cv2.IMREAD_COLOR)
        metric_swir = cv2.imread(str(pair.swir_path), cv2.IMREAD_GRAYSCALE)
        if metric_vis is None or metric_swir is None:
            log_lines.append(f"[WARN] lectura fallida frame {pair.frame_id}")
            continue
        evaluated_frame_ids.append(pair.frame_id)
        final_vis_path, final_swir_path = _save_aligned_pair_outputs(
            method_path, direction, pair.frame_id, metric_vis, metric_swir
        )
        final_reprojection = reprojection_evaluator.compute(final_vis_path, final_swir_path)
        rows.append(
            compute_metric_row(
                pair.frame_id,
                metric_vis,
                metric_swir,
                final_reprojection,
                compute_edge_alignment_metrics,
                compute_goa,
                compute_mind_map_fast,
                preprocess_pair_bench,
                resolved_device,
                mind_maps=official_mind_maps,
            )
        )
    elapsed = time.time() - t0
    fps = len(rows) / elapsed if elapsed > 0 else 0.0
    runtime = {"runtime_seconds": elapsed, "fps": fps, "device": resolved_device}
    metrics = pd.DataFrame(rows)
    metrics.to_csv(method_path / "metrics.csv", index=False)
    save_mean_mind_map(official_mind_maps, method_path, save_single_mind_map)
    (method_path / "evaluation_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    return metrics, runtime


def evaluate_submission(
    method_path: Path,
    dataset: str,
    direction: str,
    H: np.ndarray,
    device: str = "auto",
    max_examples: int = 12,
) -> tuple[pd.DataFrame, dict]:
    """Ancien flux : evalue H sur le split fixe complet et sauvegarde les diagnostics."""
    (
        compute_edge_alignment_metrics,
        compute_goa,
        compute_mind_map_fast,
        preprocess_pair_bench,
        resolve_device,
        save_single_mind_map,
    ) = _import_quality_functions()
    resolved_device = resolve_device(device)
    rows = []
    diagnostic_rows = []
    official_mind_maps: list[np.ndarray] = []
    pairs = fixed_test_pairs(dataset)
    log_lines = [
        f"Dataset: {dataset}",
        f"Direction: {direction}",
        f"Fixed test pairs: {len(pairs)}",
        f"Device: {resolved_device}",
        f'"evaluation_protocol": "{EVALUATION_PROTOCOL}"',
        "Official metrics: metrics.csv",
        "Full-image diagnostic metrics: metrics_full_image_diagnostic.csv",
    ]
    reprojection_evaluator = ReprojectionAlignedEvaluator()
    log_lines.append(
        f"Reprojection Error: reprojection_aligned.py protocol "
        f"(SuperPoint + LightGlue, device={reprojection_evaluator.device})"
    )

    t0 = time.time()
    evaluated_frame_ids = []
    for n, (frame_id, vis_path, swir_path) in enumerate(pairs):
        vis = cv2.imread(str(vis_path), cv2.IMREAD_COLOR)
        swir = cv2.imread(str(swir_path), cv2.IMREAD_GRAYSCALE)
        if vis is None or swir is None:
            log_lines.append(f"[WARN] lectura fallida frame {frame_id}")
            continue
        evaluated_frame_ids.append(frame_id)

        if direction == "VIS_to_SWIR":
            h, w = swir.shape[:2]
            metric_vis = cv2.warpPerspective(vis, H, (w, h), flags=cv2.INTER_LINEAR)
            metric_swir = swir
        else:
            h, w = vis.shape[:2]
            metric_vis = vis
            metric_swir = cv2.warpPerspective(swir, H, (w, h), flags=cv2.INTER_LINEAR)
        diagnostic_vis_path, diagnostic_swir_path = _save_aligned_pair_outputs(
            method_path,
            direction,
            frame_id,
            metric_vis,
            metric_swir,
        )
        diagnostic_reprojection = reprojection_evaluator.compute(diagnostic_vis_path, diagnostic_swir_path)

        diagnostic_rows.append(
            compute_metric_row(
                frame_id,
                metric_vis,
                metric_swir,
                diagnostic_reprojection,
                compute_edge_alignment_metrics,
                compute_goa,
                compute_mind_map_fast,
                preprocess_pair_bench,
                resolved_device,
                crop_rect=None,
                mind_maps=official_mind_maps,
            )
        )
    rows = diagnostic_rows

    elapsed = time.time() - t0
    fps = len(rows) / elapsed if elapsed > 0 else 0.0
    runtime = {"runtime_seconds": elapsed, "fps": fps, "device": resolved_device}
    metrics = pd.DataFrame(rows)
    metrics.to_csv(method_path / "metrics.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(method_path / "metrics_full_image_diagnostic.csv", index=False)
    save_mean_mind_map(official_mind_maps, method_path, save_single_mind_map)
    (method_path / "evaluation_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    return metrics, runtime
