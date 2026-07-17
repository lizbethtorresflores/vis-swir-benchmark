
import cv2
import argparse
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn.functional as F

from ..pipeline.metrics import (
    _to_gray,
    _build_valid_mask,
    remove_sky_from_mask,
    compute_goa,
    compute_edge_alignment_metrics,
    compute_mind,
    compute_mind_map,
    compute_goa_map,
    MASK_EROSION_PX,
    BLUR_SIGMA_VIS,
    MIND_SIGMA,
    MIND_BORDER_ERODE,
)


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
VIS_KEYWORD    = "vis"
SWIR_KEYWORD   = "swir_aligned"
H_IDENTITY     = np.eye(3, dtype=np.float64)


# ─── I/O ──────────────────────────────────────────────────────────────────────

def load_images_by_keyword(directory: Path, keyword: str) -> dict[str, np.ndarray]:
    """
    Charge toutes les images d'un répertoire dont le nom de fichier contient un mot-clé.

    Args:
        directory: Chemin vers le dossier à parcourir.
        keyword:   Sous-chaîne recherchée dans le stem du fichier.

    Returns:
        Dictionnaire {stem: image_bgr} pour chaque image valide trouvée.
    """
    imgs = {}
    for f in sorted(directory.iterdir()):
        if f.suffix.lower() in IMG_EXTENSIONS and keyword in f.stem:
            img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if img is not None:
                imgs[f.stem] = img
            else:
                print(f"  [WARN] illisible : {f.name}")
    return imgs


def build_pairs(vis_imgs: dict, swir_imgs: dict) -> list[tuple]:
    """
    Prend automatiquement en charge :
      1) *_vis.jpg + *_swir_aligned.jpg      (SWIR_TO_VIS)
      2) *_vis_aligned.jpg + *_swir.jpg      (VIS_TO_SWIR)
    """

    def clean_id(stem: str) -> str:
        for suffix in [
            "_vis_aligned",
            "_swir_aligned",
            "_vis",
            "_swir",
        ]:
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
        return stem

    vis_map = {clean_id(s): img for s, img in vis_imgs.items()}
    swir_map = {clean_id(s): img for s, img in swir_imgs.items()}

    common_ids = sorted(set(vis_map) & set(swir_map))

    if len(common_ids) == 0:
        print("  [WARN] aucune paire VIS/SWIR trouvée")
        return []

    missing_vis = len(set(swir_map) - set(vis_map))
    missing_swir = len(set(vis_map) - set(swir_map))

    if missing_vis > 0:
        print(f"  [WARN] {missing_vis} VIS manquantes")

    if missing_swir > 0:
        print(f"  [WARN] {missing_swir} SWIR manquantes")

    return [(pid, vis_map[pid], swir_map[pid]) for pid in common_ids]

def save_results(rows: list[dict], results_dir: Path) -> None:
    """
    Sauvegarde les scores de qualité dans un fichier CSV.

    Args:
        rows:        Liste de dictionnaires contenant les métriques par paire.
        results_dir: Répertoire de sortie (créé si inexistant).
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "scores_qualite_recalage.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.4f")
    print(f"   → {csv_path}")

def summarize_quality_sequence(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)

    summary = {
        "num_frames": len(df),
    }

    for metric in ["goa", "edge_f1_8px", "mind_ssd"]:
        values = pd.to_numeric(df[metric], errors="coerce").dropna().to_numpy()

        if len(values) == 0:
            summary[f"mean_{metric}"] = np.nan
            summary[f"median_{metric}"] = np.nan
            summary[f"std_{metric}"] = np.nan
            summary[f"q25_{metric}"] = np.nan
            summary[f"q75_{metric}"] = np.nan
            summary[f"p90_{metric}"] = np.nan
            summary[f"p95_{metric}"] = np.nan
        else:
            summary[f"mean_{metric}"] = float(np.mean(values))
            summary[f"median_{metric}"] = float(np.median(values))
            summary[f"std_{metric}"] = float(np.std(values))
            summary[f"q25_{metric}"] = float(np.percentile(values, 25))
            summary[f"q75_{metric}"] = float(np.percentile(values, 75))
            summary[f"p90_{metric}"] = float(np.percentile(values, 90))
            summary[f"p95_{metric}"] = float(np.percentile(values, 95))

    return summary

def save_summary_to_csv(summary: dict, out_csv_path: Path) -> None:
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(out_csv_path, index=False, float_format="%.6f")
    print(f"   → {out_csv_path}")

def save_comparison(all_rows: list[dict], paired_images_dir: Path) -> None:
    """
    Génère un CSV de comparaison inter-méthodes (moyenne, std, médiane) et
    l'affiche sur stdout.

    Args:
        all_rows:           Liste agrégée de toutes les métriques (toutes méthodes).
        paired_images_dir:  Dossier racine ; la sortie est écrite dans
                            <paired_images_dir>/benchmark_comparison/.
    """
    comp_dir = paired_images_dir / "benchmark_comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(comp_dir / "comparison.csv", index=False, float_format="%.4f")
    summary = (
        df.groupby("method")[["goa", "edge_f1_8px", "mind_ssd"]]
        .agg(["mean", "std", "median"])
        .round(4)
    )
    summary.to_csv(comp_dir / "comparison_summary.csv")
    print(f"\n{'='*60}\nSYNTHÈSE COMPARATIVE\n{'='*60}")
    print(summary.to_string())
    print(f"\n→ {comp_dir}")


# ─── Prétraitement bench ──────────────────────────────────────────────────────

def compute_common_bbox(
    vis_gray: np.ndarray, swir_warped: np.ndarray
) -> tuple[int, int, int, int]:
    """
    Calcule la bounding box de l'intersection spatiale des pixels valides
    entre l'image VIS et l'image SWIR warpée.

    Args:
        vis_gray:     Image VIS niveaux de gris.
        swir_warped:  Image SWIR warpée niveaux de gris.

    Returns:
        Tuple (x0, y0, x1, y1) en pixels. Retourne toute l'image si pas
        d'intersection trouvée.
    """
    common = cv2.bitwise_and(
        (vis_gray > 5).astype(np.uint8),
        (swir_warped > 5).astype(np.uint8),
    )
    coords = cv2.findNonZero(common)
    if coords is None:
        h, w = vis_gray.shape[:2]
        return 0, 0, w, h
    x, y, bw, bh = cv2.boundingRect(coords)
    return x, y, x + bw, y + bh


def preprocess_pair_bench(
    vis_bgr: np.ndarray, swir_raw: np.ndarray, H_mat: np.ndarray, device: str = "cpu"
) -> tuple:
    """
    Prépare une paire VIS/SWIR pour le benchmark qualité.
    Contrairement à preprocess_pair (metrics.py), gère les résolutions
    différentes (resize automatique) et recadre les deux images sur leur
    intersection spatiale commune.

    Args:
        vis_bgr:  Image VIS BGR (tout dtype).
        swir_raw: Image SWIR brute (dtype et résolution quelconques).
        H_mat:    Matrice d'homographie 3×3.

    Returns:
        Tuple (vis_gray, swir_warped, valid_mask) recadrés sur la bbox commune,
        tous en uint8.
    """
    h,  w  = vis_bgr.shape[:2]
    hs, ws = swir_raw.shape[:2]

    swir_u8 = _to_gray(swir_raw)
    if (hs, ws) != (h, w):
        swir_u8 = cv2.resize(swir_u8, (w, h), interpolation=cv2.INTER_LINEAR)

    swir_warped = cv2.warpPerspective(swir_u8, H_mat, (w, h), flags=cv2.INTER_LINEAR)

    vis_gray = _to_gray(vis_bgr)
    if BLUR_SIGMA_VIS > 0.0:
        ksize    = 2 * int(np.ceil(BLUR_SIGMA_VIS * 3)) + 1
        vis_gray = cv2.GaussianBlur(vis_gray, (ksize, ksize), BLUR_SIGMA_VIS)

    x0, y0, x1, y1 = compute_common_bbox(vis_gray, swir_warped)
    vis_gray    = vis_gray   [y0:y1, x0:x1]
    swir_warped = swir_warped[y0:y1, x0:x1]
    valid_mask  = _build_valid_mask(swir_warped, MASK_EROSION_PX)
    vis_crop_bgr = vis_bgr[y0:y1, x0:x1]
    valid_mask = remove_sky_from_mask(vis_crop_bgr, valid_mask, device=device)
    return vis_gray, swir_warped, valid_mask


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA demandé mais indisponible; fallback CPU.")
        return "cpu"
    return device_arg


def _torch_gray_tensor(img: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(img.astype(np.float32)).to(device=device)[None, None]


def _torch_mask_tensor(mask: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(mask > 0).to(device=device)


def _gaussian_kernel2d(sigma: float, device: str) -> torch.Tensor:
    ksize = max(2 * int(np.ceil(sigma * 3)) + 1, 3)
    coords = torch.arange(ksize, dtype=torch.float32, device=device) - (ksize // 2)
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d[None, None]


def _shift_torch_no_wrap(img: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    out = torch.zeros_like(img)
    _, _, h, w = img.shape
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    xd0, xd1 = max(0, dx), min(w, w + dx)
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    yd0, yd1 = max(0, dy), min(h, h + dy)
    out[:, :, yd0:yd1, xd0:xd1] = img[:, :, ys0:ys1, xs0:xs1]
    return out


def compute_goa_torch(vis_gray: np.ndarray, swir_gray: np.ndarray, mask: np.ndarray, device: str) -> float:
    vis = _torch_gray_tensor(vis_gray, device)
    swir = _torch_gray_tensor(swir_gray, device)
    roi = _torch_mask_tensor(mask, device)

    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32,
        device=device,
    )[None, None]
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=torch.float32,
        device=device,
    )[None, None]

    vis_pad = F.pad(vis, (1, 1, 1, 1), mode="reflect")
    swir_pad = F.pad(swir, (1, 1, 1, 1), mode="reflect")
    gxv = F.conv2d(vis_pad, sobel_x)[0, 0]
    gyv = F.conv2d(vis_pad, sobel_y)[0, 0]
    gxs = F.conv2d(swir_pad, sobel_x)[0, 0]
    gys = F.conv2d(swir_pad, sobel_y)[0, 0]

    cos2 = torch.cos(torch.atan2(gyv, gxv) - torch.atan2(gys, gxs)) ** 2
    weight = torch.minimum(torch.sqrt(gxv ** 2 + gyv ** 2), torch.sqrt(gxs ** 2 + gys ** 2))
    w_sum = weight[roi].sum()
    if float(w_sum.detach().cpu()) < 1e-8:
        return float("nan")
    return float(((cos2[roi] * weight[roi]).sum() / w_sum).detach().cpu())


def _mind_ssd_torch(
    vis_gray: np.ndarray, swir_gray: np.ndarray, mask: np.ndarray, device: str
) -> tuple[np.ndarray, np.ndarray]:
    kernel = _gaussian_kernel2d(MIND_SIGMA, device)
    pad = kernel.shape[-1] // 2

    def descriptor(img_np: np.ndarray) -> torch.Tensor:
        img = _torch_gray_tensor(img_np, device)
        diffs = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            diff = (img - _shift_torch_no_wrap(img, dx, dy)) ** 2
            diff = F.pad(diff, (pad, pad, pad, pad), mode="reflect")
            diffs.append(F.conv2d(diff, kernel))
        dp = torch.cat(diffs, dim=1)
        v = dp.mean(dim=1, keepdim=True)
        vmean = v.mean() + 1e-8
        v = torch.clamp(v, 1e-3 * vmean, 1e3 * vmean)
        return torch.exp(-dp / v)

    mind_v = descriptor(vis_gray)
    mind_s = descriptor(swir_gray)
    ssd = ((mind_v - mind_s) ** 2).mean(dim=1)[0].detach().cpu().numpy().astype(np.float32)

    if MIND_BORDER_ERODE > 0:
        k = 2 * MIND_BORDER_ERODE + 1
        mask = cv2.erode(mask, np.ones((k, k), np.uint8), iterations=1)
    return ssd, mask


def compute_mind_map_fast(
    vis_gray: np.ndarray, swir_gray: np.ndarray, mask: np.ndarray, device: str
) -> np.ndarray:
    if device == "cuda":
        ssd, eroded_mask = _mind_ssd_torch(vis_gray, swir_gray, mask, device)
        ssd[eroded_mask == 0] = np.nan
        return ssd
    return compute_mind_map(vis_gray, swir_gray, mask).astype(np.float32)


# ─── Visualisation ────────────────────────────────────────────────────────────

def save_mind_map(mind_maps: list[np.ndarray], results_dir: Path) -> None:
    """
    Calcule la carte MIND-SSD moyenne sur toutes les paires, la normalise et
    la sauvegarde en fausse couleur JET.

    Args:
        mind_maps:   Liste de cartes SSD (H, W) float32 avec nan hors masque.
        results_dir: Répertoire de sortie.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    ref_shape = mind_maps[0].shape
    resized = [
        cv2.resize(
            m.astype(np.float32), (ref_shape[1], ref_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        ) if m.shape != ref_shape else m.astype(np.float32)
        for m in mind_maps
    ]
    mean_map = np.nanmean(np.stack(resized, axis=0), axis=0)
    vmin     = np.nanpercentile(mean_map, 2)
    vmax     = np.nanpercentile(mean_map, 98)
    norm     = np.clip((mean_map - vmin) / (vmax - vmin + 1e-8), 0, 1)
    colored  = cv2.applyColorMap(
        (np.nan_to_num(norm, nan=0.0) * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    colored  = np.where(np.isnan(mean_map)[..., np.newaxis], 0, colored).astype(np.uint8)
    out_path = results_dir / "mind_map_mean.png"
    success  = cv2.imwrite(str(out_path), colored)
    print(f"   → {out_path} ({len(mind_maps)} paires) imwrite={success}")


def save_error_map(vis_gray, error_map, valid_mask, out_path, title=""):
    error = error_map.astype(np.float32)
    error[valid_mask == 0] = np.nan

    vmin = np.nanpercentile(error, 2)
    vmax = np.nanpercentile(error, 98)

    norm = np.clip((error - vmin) / (vmax - vmin + 1e-8), 0, 1)
    color = cv2.applyColorMap(
        (np.nan_to_num(norm, nan=0.0) * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    vis_bgr = cv2.cvtColor(vis_gray, cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(vis_bgr, 0.55, color, 0.45, 0)

    overlay[valid_mask == 0] = vis_bgr[valid_mask == 0]

    cv2.putText(
        overlay,
        title,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(out_path), overlay)
    
def run_on_split_dir(split_dir: Path, device: str = "cpu") -> list[dict]:
    """
    Calcule les métriques GOA, Edge Alignment et MIND sur toutes les paires VIS/SWIR
    d'un dossier split_images, sauvegarde la carte MIND moyenne et retourne
    les scores par paire.

    Args:
        split_dir: Dossier contenant les images VIS et SWIR alignées.

    Returns:
        Liste de dictionnaires {method, pair_id, goa, edge_f1_8px, mind_ssd}.
    """
    method_name = split_dir.parent.name
    # --------------------------------------------------
    # Détecter automatiquement la direction.
    # --------------------------------------------------

    vis_imgs = {}
    swir_imgs = {}

    for f in sorted(split_dir.iterdir()):
        if f.suffix.lower() not in IMG_EXTENSIONS:
            continue

        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        stem = f.stem

        if stem.endswith("_vis") or stem.endswith("_vis_aligned"):
            vis_imgs[stem] = img

        elif stem.endswith("_swir") or stem.endswith("_swir_aligned"):
            swir_imgs[stem] = img

    pairs = build_pairs(vis_imgs, swir_imgs)

    if len(pairs) == 0:

        print("  [INFO] tentative VIS_TO_SWIR")

        vis_imgs = load_images_by_keyword(split_dir, "vis_aligned")
        swir_imgs = load_images_by_keyword(split_dir, "swir")

        def pair_id(stem):
            stem = stem.replace("vis_aligned", "")
            stem = stem.replace("swir", "")
            return stem.strip("_-")

        vis_map = {pair_id(s): img for s, img in vis_imgs.items()}
        swir_map = {pair_id(s): img for s, img in swir_imgs.items()}

        pairs = [
            (pid, vis_map[pid], swir_map[pid])
            for pid in sorted(set(vis_map) & set(swir_map))
    ]

    if not pairs:
        print(f"  [WARN] aucune paire trouvée dans {split_dir}")
        return []

    print(f"\n── {method_name} ({len(pairs)} paires)", flush=True)
    print(f"Device qualité : {device}", flush=True)
    rows = []
    case_data = []
    mind_sum = None
    mind_count = None
    ref_shape = None

    t0 = time.time()
    results_dir = split_dir.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for pair_id, vis_img, swir_img in pairs:
        vis_gray, swir_warped, valid_mask = preprocess_pair_bench(
            vis_img, swir_img, H_IDENTITY, device=device
        )
        goa = (
            compute_goa_torch(vis_gray, swir_warped, valid_mask, device)
            if device == "cuda"
            else compute_goa(vis_gray, swir_warped, valid_mask)
        )
        edge_metrics = compute_edge_alignment_metrics(vis_gray, swir_warped, valid_mask, tol_px=8)
        mind_map = compute_mind_map_fast(vis_gray, swir_warped, valid_mask, device)
        mind = float(np.nanmean(mind_map)) if np.isfinite(mind_map).any() else float("nan")
        case_data.append({
            "pair_id": pair_id,
            "goa": goa,
            "edge_f1_8px": edge_metrics["edge_f1_8px"],
            "mind_ssd": mind,
        })

        if ref_shape is None:
            ref_shape = mind_map.shape
            mind_sum = np.zeros(ref_shape, dtype=np.float32)
            mind_count = np.zeros(ref_shape, dtype=np.float32)

        if mind_map.shape != ref_shape:
            mind_map = cv2.resize(
                mind_map,
                (ref_shape[1], ref_shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        valid = ~np.isnan(mind_map)
        mind_sum[valid] += mind_map[valid]
        mind_count[valid] += 1

        del mind_map

        rows.append({
            "method":   method_name,
            "pair_id":  pair_id,
            "goa":      round(goa,  4),
            "edge_f1_8px":        round(edge_metrics["edge_f1_8px"], 4),
            "mind_ssd": round(mind, 4),
        })
        print(
            f"  [{pair_id}] GOA={goa:.3f} EdgeAlignment={edge_metrics['edge_f1_8px']:.3f} "
            f"MIND={mind:.4f}",
            flush=True,
        )
        del vis_gray
        del swir_warped
        del valid_mask
        import gc
        gc.collect()
    mean_map = np.divide(
        mind_sum,
        np.maximum(mind_count, 1),
        out=np.full_like(mind_sum, np.nan, dtype=np.float32),
        where=mind_count > 0
    )

    save_single_mind_map(mean_map, split_dir.parent / "results")
    t1 = time.time()
    elapsed = t1 - t0
    fps = len(rows) / elapsed if elapsed > 0 else 0
    time_per_frame = elapsed / len(rows) if len(rows) > 0 else 0

    summary = summarize_quality_sequence(rows)
    summary["dataset"] = split_dir.parents[2].name
    summary["method"] = method_name
    summary["num_images"] = len(rows)
    summary["quality_eval_time_s"] = elapsed
    summary["fps_quality_eval"] = fps
    summary["time_per_frame_s"] = time_per_frame
    summary["device"] = device
    summary["note"] = "GOA, Edge Alignment and MIND computed on aligned VIS/SWIR pairs after common-area cropping."

    results_dir = split_dir.parent / "results"
    save_summary_to_csv(summary, results_dir / "quality_summary.csv")

    runtime_row = {
        "dataset": split_dir.parents[2].name,
        "method": method_name,
        "num_images": len(rows),
        "quality_eval_time_s": elapsed,
        "fps_quality_eval": fps,
        "time_per_frame_s": time_per_frame,
        "device": device,
        "note": "This runtime corresponds only to GOA/Edge Alignment/MIND metric evaluation."
    }
    save_summary_to_csv(runtime_row, results_dir / "quality_runtime.csv")
    df = pd.DataFrame(rows)

    print("\n===== TOP 3 MEILLEURS / PIRES =====")

    print("\nGOA meilleurs (plus élevé) :")
    print(df.sort_values("goa", ascending=False)[["pair_id", "goa", "edge_f1_8px", "mind_ssd"]].head(3).to_string(index=False))

    print("\nGOA pires (plus bas) :")
    print(df.sort_values("goa", ascending=True)[["pair_id", "goa", "edge_f1_8px", "mind_ssd"]].head(3).to_string(index=False))

    print("\nEdge Alignment meilleurs (plus élevé) :")
    print(df.sort_values("edge_f1_8px", ascending=False)[["pair_id", "goa", "edge_f1_8px", "mind_ssd"]].head(3).to_string(index=False))

    print("\nEdge Alignment pires (plus bas) :")
    print(df.sort_values("edge_f1_8px", ascending=True)[["pair_id", "goa", "edge_f1_8px", "mind_ssd"]].head(3).to_string(index=False))
    panels_dir = results_dir / "quality_panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    def save_top_cases(metric, higher_is_better=True):
        metric_dir = panels_dir / metric
        best_dir = metric_dir / "best_cases"
        worst_dir = metric_dir / "worst_cases"
        best_dir.mkdir(parents=True, exist_ok=True)
        worst_dir.mkdir(parents=True, exist_ok=True)

        sorted_cases = sorted(
            case_data,
            key=lambda x: x[metric],
            reverse=higher_is_better
        )

        best_cases = sorted_cases[:3]
        worst_cases = sorted_cases[-3:][::-1]

        for i, case in enumerate(best_cases, start=1):
            if metric == "goa":
                error_map = 1.0 - compute_goa_map(
                    case["vis_gray"],
                    case["swir_gray"],
                    case["valid_mask"]
                )
            else:
                error_map = compute_mind_map(
                    case["vis_gray"],
                    case["swir_gray"],
                    case["valid_mask"]
                )

            save_error_map(
                case["vis_gray"],
                error_map,
                case["valid_mask"],
                best_dir / f"best_{i:02d}_{case['pair_id']}_{metric}_error.png",
                title=f"BEST {metric.upper()} #{i} | {case['pair_id']}"
            )

        for i, case in enumerate(worst_cases, start=1):
            if metric == "goa":
                error_map = 1.0 - compute_goa_map(
                    case["vis_gray"],
                    case["swir_gray"],
                    case["valid_mask"]
                )
            else:
                error_map = compute_mind_map(
                    case["vis_gray"],
                    case["swir_gray"],
                    case["valid_mask"]
                )

            save_error_map(
                case["vis_gray"],
                error_map,
                case["valid_mask"],
                worst_dir / f"worst_{i:02d}_{case['pair_id']}_{metric}_error.png",
                title=f"WORST {metric.upper()} #{i} | {case['pair_id']}"
            )

    #save_top_cases("goa", higher_is_better=True)
    #save_top_cases("mind_ssd", higher_is_better=False)

    print(f"Quality panels sauvegardés : {panels_dir}")
    return rows


def save_single_mind_map(mean_map: np.ndarray, results_dir: Path, vmin=None, vmax=None, suffix="") -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    np.save(results_dir / f"mind_map_mean_raw{suffix}.npy", mean_map)

    if vmin is None:
        vmin = np.nanpercentile(mean_map, 2)
    if vmax is None:
        vmax = np.nanpercentile(mean_map, 98)

    norm = np.clip((mean_map - vmin) / (vmax - vmin + 1e-8), 0, 1)

    colored = cv2.applyColorMap(
        (np.nan_to_num(norm, nan=0.0) * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )

    colored = np.where(
        np.isnan(mean_map)[..., np.newaxis],
        0,
        colored
    ).astype(np.uint8)

    out_path = results_dir / f"mind_map_mean{suffix}.png"
    cv2.imwrite(str(out_path), colored)

    print(f"   → {out_path}")
# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Point d'entrée CLI. Accepte un ou plusieurs chemins vers des dossiers
    split_images/, calcule les métriques sur chacun et génère un rapport
    comparatif si plusieurs méthodes sont fournies.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark qualité recalage VIS-SWIR — GOA / Edge Alignment / MIND"
    )
    parser.add_argument(
        "inputs", nargs="+", metavar="SPLIT_IMAGES_PATH",
        help="Chemin(s) vers split_images/ relatifs à datas/",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device pour SegFormer et les métriques accélérées. auto utilise CUDA si disponible.",
    )
    args = parser.parse_args()

    projet_root         = Path(__file__).resolve().parents[3]
    all_rows, split_dirs = [], []
    device = resolve_device(args.device)

    for raw_path in args.inputs:
        split_dir = (projet_root / "datas" / raw_path).resolve()
        if not split_dir.exists():
            print(f"[ERREUR] dossier introuvable : {split_dir}")
            continue
        rows = run_on_split_dir(split_dir, device=device)
        if rows:
            save_results(rows, split_dir.parent / "results")
            all_rows.extend(rows)
            split_dirs.append(split_dir)

    if len(split_dirs) > 1 and all_rows:
        save_comparison(all_rows, split_dirs[0].parents[1])


if __name__ == "__main__":
    main()
