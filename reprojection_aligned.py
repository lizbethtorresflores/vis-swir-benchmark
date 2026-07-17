import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from pathlib import Path
import torch.nn.functional as F
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch
from lightglue import SuperPoint, LightGlue
from lightglue.utils import load_image


PROJECT_ROOT = Path("/home/lizbeth/Desktop/intership/recallage-multimodal-zz3")


def build_paths(dataset, method):
    paired_dir = PROJECT_ROOT / "datas" / dataset / "Sortie_Extraite" / "paired_images"

    method_dirs = [
        p for p in paired_dir.iterdir()
        if p.is_dir()
        and p.name == method
    ]

    if len(method_dirs) == 0:
        raise FileNotFoundError(f"Méthode '{method}' introuvable dans {paired_dir}")

    if len(method_dirs) > 1:
        print("⚠️ Plusieurs méthodes trouvées :")
        for p in method_dirs:
            print(" -", p.name)
        print("Utilisation de :", method_dirs[0].name)

    method_dir = method_dirs[0]
    split_dir = method_dir / "split_images"

    if not split_dir.exists():
        raise FileNotFoundError(f"Le dossier split_images n'existe pas : {split_dir}")

    out_dir = method_dir / "results" / "reprojection_aligned"
    out_dir.mkdir(parents=True, exist_ok=True)

    return {
        "method_dir": method_dir,
        "split_dir": split_dir,
        "out_dir": out_dir,
    }


def load_split_pairs(split_dir, num_eval):

    # SWIR -> VIS
    vis_files = sorted([str(p) for p in split_dir.glob("*_vis.jpg")])
    swir_files = sorted([str(p) for p in split_dir.glob("*_swir_aligned.jpg")])

    if len(vis_files) > 0 and len(swir_files) > 0:
        print("[INFO] Direction détectée : SWIR_TO_VIS")

    else:
        # VIS -> SWIR
        vis_files = sorted([str(p) for p in split_dir.glob("*_vis_aligned.jpg")])
        swir_files = sorted([str(p) for p in split_dir.glob("*_swir.jpg")])

        if len(vis_files) > 0 and len(swir_files) > 0:
            print("[INFO] Direction détectée : VIS_TO_SWIR")

    vis_files = vis_files[:num_eval]
    swir_files = swir_files[:num_eval]

    if len(vis_files) != len(swir_files):
        print("⚠️ Nombre différent d'images VIS et SWIR")
        print("VIS :", len(vis_files))
        print("SWIR:", len(swir_files))

    return vis_files, swir_files


def extract_pair_id(path):
    stem = Path(path).stem

    for part in stem.split("_"):
        if part.isdigit():
            return part

    import re
    match = re.search(r"(\d+)", stem)
    return match.group(1) if match else stem


def find_concatenate_image(method_dir, vis_path, swir_path):
    concat_dirs = [
        method_dir / "concatenate_images",
        method_dir / "concatenate",
        method_dir / "cocatenate",
        method_dir / "concatenated",
        method_dir / "concat",
    ]

    pair_ids = [extract_pair_id(vis_path), extract_pair_id(swir_path)]

    for concat_dir in concat_dirs:
        if not concat_dir.exists():
            continue

        for pair_id in pair_ids:
            matches = sorted(
                list(concat_dir.glob(f"*{pair_id}*.jpg"))
                + list(concat_dir.glob(f"*{pair_id}*.png"))
            )

            if matches:
                return str(matches[0])

    return None


def to_grayscale_tensor(img):
    if img.ndim == 3 and img.shape[0] == 3:
        return 0.299 * img[0:1] + 0.587 * img[1:2] + 0.114 * img[2:3]
    return img


def init_superpoint_lightglue(device, max_keypoints):
    extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    return extractor, matcher


def detect_and_match_superpoint_lightglue(
    vis_path,
    swir_path,
    extractor,
    matcher,
    device
):
    img_vis = load_image(vis_path).to(device)
    img_swir = load_image(swir_path).to(device)

    img_swir = to_grayscale_tensor(img_swir)

    with torch.no_grad():
        feats_vis = extractor.extract(img_vis)
        feats_swir = extractor.extract(img_swir)

        matches01 = matcher({
            "image0": feats_vis,
            "image1": feats_swir,
        })

    kpts_vis = feats_vis["keypoints"][0]
    kpts_swir = feats_swir["keypoints"][0]
    matches = matches01["matches"][0]

    if len(matches) == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            matches,
        )

    pts_vis = kpts_vis[matches[:, 0]].detach().cpu().numpy().astype(np.float32)
    pts_swir = kpts_swir[matches[:, 1]].detach().cpu().numpy().astype(np.float32)

    return pts_vis, pts_swir, matches


def get_ransac_inliers(pts_vis, pts_swir, threshold=5.0):
    if len(pts_vis) < 4 or len(pts_swir) < 4:
        return np.zeros(len(pts_vis), dtype=bool)

    _, mask = cv2.findHomography(
        pts_vis,
        pts_swir,
        cv2.RANSAC,
        ransacReprojThreshold=threshold,
    )

    if mask is None:
        return np.ones(len(pts_vis), dtype=bool)

    return mask.ravel().astype(bool)

_segformer_processor = None
_segformer_model = None

def get_segformer_model(device):
    global _segformer_processor, _segformer_model

    if _segformer_processor is None:
        _segformer_processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640"
        )
        _segformer_model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640"
        ).to(device).eval()

    return _segformer_processor, _segformer_model


def remove_sky_matches(vis_path, pts_vis, pts_swir, device):
    img_bgr = cv2.imread(vis_path)

    if img_bgr is None or len(pts_vis) == 0:
        return pts_vis, pts_swir

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    processor, model = get_segformer_model(device)

    inputs = processor(images=img_rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    h, w = img_rgb.shape[:2]

    logits = F.interpolate(
        outputs.logits,
        size=(h, w),
        mode="bilinear",
        align_corners=False
    )

    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    sky_prob = probs[2]
    sky = sky_prob > 0.85

    top = np.zeros_like(sky, dtype=bool)
    top[: int(h * 0.50), :] = True
    sky = sky & top

    x = np.clip(np.round(pts_vis[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(pts_vis[:, 1]).astype(int), 0, h - 1)

    valid = ~sky[y, x]

    return pts_vis[valid], pts_swir[valid]

def balance_matches_by_region(
    pts_vis,
    pts_swir,
    num_rows=2,
    num_cols=3,
    max_matches_per_region=180,
    min_matches_per_region=20,
):
    """
    Équilibre les correspondances par régions spatiales avant de calculer la reprojection.
    Évite qu'une zone avec beaucoup de points domine la métrique globale.
    """

    if len(pts_vis) == 0:
        return pts_vis, pts_swir

    x = pts_vis[:, 0]
    y = pts_vis[:, 1]

    width = np.max(x)
    height = np.max(y)

    x_bins = np.linspace(0, width, num_cols + 1)
    y_bins = np.linspace(0, height, num_rows + 1)

    keep_indices = []

    np.random.seed(42)

    for r in range(num_rows):
        for c in range(num_cols):
            x0, x1 = x_bins[c], x_bins[c + 1]
            y0, y1 = y_bins[r], y_bins[r + 1]

            cell_idx = np.where(
                (x >= x0) & (x < x1) &
                (y >= y0) & (y < y1)
            )[0]

            if len(cell_idx) == 0:
                continue

            if len(cell_idx) < min_matches_per_region:
                keep_indices.extend(cell_idx.tolist())
                continue

            if len(cell_idx) > max_matches_per_region:
                selected = np.random.choice(
                    cell_idx,
                    size=max_matches_per_region,
                    replace=False
                )
                keep_indices.extend(selected.tolist())
            else:
                keep_indices.extend(cell_idx.tolist())

    if len(keep_indices) == 0:
        return pts_vis, pts_swir

    keep_indices = np.array(sorted(keep_indices), dtype=int)

    return pts_vis[keep_indices], pts_swir[keep_indices]

def compute_direct_distance_metrics(pts_vis, pts_swir, use_ransac=False, ransac_thresh=5.0):
    num_matches = len(pts_vis)

    base = {
        "num_matches": int(num_matches),
        "num_used": 0,
        "inlier_ratio": np.nan,
        "mean_error": np.nan,
        "median_error": np.nan,
        "std_error": np.nan,
        "min_error": np.nan,
        "max_error": np.nan,
        "q25_error": np.nan,
        "q75_error": np.nan,
        "p90_error": np.nan,
        "p95_error": np.nan,
    }

    if num_matches == 0:
        return base

    if use_ransac:
        inliers = get_ransac_inliers(pts_vis, pts_swir, ransac_thresh)
    else:
        inliers = np.ones(num_matches, dtype=bool)

    pts_vis_used = pts_vis[inliers]
    pts_swir_used = pts_swir[inliers]

    num_used = len(pts_vis_used)

    if num_used == 0:
        return base

    errors = np.linalg.norm(pts_vis_used - pts_swir_used, axis=1)

    return {
        "num_matches": int(num_matches),
        "num_used": int(num_used),
        "inlier_ratio": float(num_used / num_matches),
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "std_error": float(np.std(errors)),
        "min_error": float(np.min(errors)),
        "max_error": float(np.max(errors)),
        "q25_error": float(np.percentile(errors, 25)),
        "q75_error": float(np.percentile(errors, 75)),
        "p90_error": float(np.percentile(errors, 90)),
        "p95_error": float(np.percentile(errors, 95)),
    }

def save_rows_csv(rows, path):
    if len(rows) == 0:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_summary_csv(summary, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def summarize(rows):
    valid = [r for r in rows if not np.isnan(r["median_error"])]

    if len(valid) == 0:
        return {
            "num_frames": 0,
            "mean_of_mean_error": np.nan,
            "median_of_median_error": np.nan,
            "std_of_mean_error": np.nan,
            "q25_median_error": np.nan,
            "q75_median_error": np.nan,
            "p90_median_error": np.nan,
            "p95_median_error": np.nan,
            "mean_num_matches": np.nan,
            "mean_num_used": np.nan,
            "mean_inlier_ratio": np.nan,
        }

    mean_errors = np.array([r["mean_error"] for r in valid])
    median_errors = np.array([r["median_error"] for r in valid])
    num_matches = np.array([r["num_matches"] for r in valid])
    num_used = np.array([r["num_used"] for r in valid])
    inlier_ratios = np.array([r["inlier_ratio"] for r in valid])
    return {
        "num_frames": int(len(valid)),
        "mean_of_mean_error": float(np.mean(mean_errors)),
        "median_of_median_error": float(np.median(median_errors)),
        "std_of_mean_error": float(np.std(mean_errors)),
        "q25_median_error": float(np.percentile(median_errors, 25)),
        "q75_median_error": float(np.percentile(median_errors, 75)),
        "p90_median_error": float(np.percentile(median_errors, 90)),
        "p95_median_error": float(np.percentile(median_errors, 95)),
        "mean_num_matches": float(np.mean(num_matches)),
        "mean_num_used": float(np.mean(num_used)),
        "mean_inlier_ratio": float(np.nanmean(inlier_ratios)),
    }


def plot_curve(rows, out_path):
    frame_ids = [r["frame_id"] for r in rows]
    mean_errors = [r["mean_error"] for r in rows]
    median_errors = [r["median_error"] for r in rows]

    plt.figure(figsize=(12, 5))
    plt.plot(frame_ids, mean_errors, label="Mean error")
    plt.plot(frame_ids, median_errors, label="Median error")
    plt.xlabel("Frame ID")
    plt.ylabel("Distance VIS-SWIR aligned (px)")
    plt.title("Direct distance between VIS and aligned SWIR keypoints")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_histogram(rows, out_path):
    values = [r["mean_error"] for r in rows if not np.isnan(r["mean_error"])]

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=30, edgecolor="black", alpha=0.8)
    plt.xlabel("Mean distance (px)")
    plt.ylabel("Frequency")
    plt.title("Histogram of VIS-SWIR aligned distance")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_boxplot(rows, out_path):
    values = [r["mean_error"] for r in rows if not np.isnan(r["mean_error"])]

    plt.figure(figsize=(6, 5))
    plt.boxplot(values, vert=True)
    plt.ylabel("Mean distance (px)")
    plt.title("Boxplot of VIS-SWIR aligned distance")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def draw_panel(vis_path, swir_path, concat_path, pts_vis, pts_swir, out_path, title=""):
    img_vis = cv2.imread(vis_path)
    img_swir = cv2.imread(swir_path)
    img_concat = cv2.imread(concat_path) if concat_path is not None else None

    if img_vis is None or img_swir is None:
        return

    img_draw = img_concat.copy() if img_concat is not None else img_swir.copy()

    if img_draw.shape[:2] != img_swir.shape[:2]:
        img_draw = cv2.resize(
            img_draw,
            (img_swir.shape[1], img_swir.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    for p_vis, p_swir in zip(pts_vis, pts_swir):
        x_vis, y_vis = int(round(p_vis[0])), int(round(p_vis[1]))
        x_swir, y_swir = int(round(p_swir[0])), int(round(p_swir[1]))

        cv2.line(img_draw, (x_swir, y_swir), (x_vis, y_vis), (255, 255, 255), 4)
        cv2.line(img_draw, (x_swir, y_swir), (x_vis, y_vis), (255, 80, 0), 2)
        cv2.circle(img_draw, (x_swir, y_swir), 5, (255, 255, 255), -1)
        cv2.circle(img_draw, (x_swir, y_swir), 3, (0, 255, 0), -1)
        cv2.circle(img_draw, (x_vis, y_vis), 5, (255, 255, 255), -1)
        cv2.circle(img_draw, (x_vis, y_vis), 3, (0, 0, 255), -1)

    legend_x, legend_y = 14, 18
    cv2.rectangle(img_draw, (8, 8), (430, 92), (0, 0, 0), -1)
    cv2.circle(img_draw, (legend_x + 10, legend_y + 8), 5, (0, 255, 0), -1)
    cv2.putText(img_draw, "green: SWIR/aligned keypoint", (legend_x + 28, legend_y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.circle(img_draw, (legend_x + 10, legend_y + 34), 5, (0, 0, 255), -1)
    cv2.putText(img_draw, "red: VIS keypoint", (legend_x + 28, legend_y + 39), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.line(img_draw, (legend_x + 4, legend_y + 60), (legend_x + 20, legend_y + 60), (255, 80, 0), 2)
    cv2.putText(img_draw, "blue: distance/error vector", (legend_x + 28, legend_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    axes[0].set_title("VIS original")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(img_swir, cv2.COLOR_BGR2RGB))
    axes[1].set_title("SWIR aligned")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
    axes[2].set_title(title)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_method(dataset, method, num_eval, max_keypoints, use_ransac, ransac_thresh):
    paths = build_paths(dataset, method)
    vis_files, swir_files = load_split_pairs(paths["split_dir"], num_eval)

    print("\n" + "=" * 80)
    print("ALIGNED DISTANCE BENCHMARK - SUPERPOINT + LIGHTGLUE")
    print("=" * 80)
    print("Dataset    :", dataset)
    print("Method     :", method)
    print("Method dir :", paths["method_dir"])
    print("Split dir  :", paths["split_dir"])
    print("VIS files  :", len(vis_files))
    print("SWIR files :", len(swir_files))

    if len(vis_files) > 0:
        print("Primer VIS :", Path(vis_files[0]).name)
        print("Último VIS :", Path(vis_files[-1]).name)

    if len(swir_files) > 0:
        print("Primer SWIR:", Path(swir_files[0]).name)
        print("Último SWIR:", Path(swir_files[-1]).name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device     :", device)

    extractor, matcher = init_superpoint_lightglue(device, max_keypoints)

    rows = []
    start = time.time()

    for frame_id, (vis_path, swir_path) in enumerate(zip(vis_files, swir_files)):
        pts_vis, pts_swir, matches = detect_and_match_superpoint_lightglue(
            vis_path,
            swir_path,
            extractor,
            matcher,
            device,
        )


        # Remove sky region before computing reprojection/aligned distance
        pts_vis, pts_swir = remove_sky_matches(
            vis_path,
            pts_vis,
            pts_swir,
            device,
        )

        pts_vis, pts_swir = balance_matches_by_region(
            pts_vis,
            pts_swir,
            num_rows=2,
            num_cols=3,
            max_matches_per_region=180,
            min_matches_per_region=20,
        )

        metrics = compute_direct_distance_metrics(
            pts_vis,
            pts_swir,
            use_ransac=use_ransac,
            ransac_thresh=ransac_thresh,
        )

        row = {
            "dataset": dataset,
            "method": method,
            "frame_id": frame_id,
            "vis_file": Path(vis_path).name,
            "swir_file": Path(swir_path).name,
            **metrics,
        }

        rows.append(row)

        print(
            f"[{frame_id + 1}/{len(vis_files)}] "
            f"{Path(vis_path).name} ↔ {Path(swir_path).name} | "
            f"matches={metrics['num_matches']} | "
            f"used={metrics['num_used']} | "
            f"median={metrics['median_error']:.3f}px"
        )
    end = time.time()

    out_dir = paths["out_dir"]

    framewise_csv = out_dir / "reprojection_aligned_framewise.csv"
    summary_csv = out_dir / "reprojection_aligned_summary.csv"
    runtime_csv = out_dir / "reprojection_aligned_runtime.csv"

    save_rows_csv(rows, framewise_csv)

    summary = summarize(rows)
    summary["dataset"] = dataset
    summary["method"] = method
    summary["num_eval_requested"] = num_eval
    summary["num_eval_used"] = len(rows)
    summary["split_dir"] = str(paths["split_dir"])
    summary["detector"] = "SuperPoint"
    summary["matcher"] = "LightGlue"
    summary["max_keypoints"] = max_keypoints
    summary["homography_used"] = False
    summary["ransac_filter_used"] = use_ransac
    summary["ransac_thresh"] = ransac_thresh
    summary["note"] = (
        "Distance computed directly between matched keypoints in split_images. "
        "VIS image is the original VIS stored in split_images, and SWIR image is the aligned SWIR. "
        "No homography is loaded or applied. Sky region is removed before distance computation."
    )

    save_summary_csv(summary, summary_csv)

    runtime = {
        "dataset": dataset,
        "method": method,
        "num_images": len(rows),
        "eval_time_s": end - start,
        "fps": len(rows) / (end - start) if end > start else 0,
        "time_per_frame_s": (end - start) / len(rows) if len(rows) > 0 else 0,
    }

    save_summary_csv(runtime, runtime_csv)

    plot_curve(rows, out_dir / "reprojection_aligned_curve.png")
    plot_histogram(rows, out_dir / "reprojection_aligned_histogram.png")
    plot_boxplot(rows, out_dir / "reprojection_aligned_boxplot.png")

    valid_rows = [r for r in rows if not np.isnan(r["median_error"])]
    valid_rows = sorted(valid_rows, key=lambda r: r["median_error"])

    panels_dir = out_dir / "reprojection_aligned_panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    for prefix, selected in [("best", valid_rows[:3]), ("worst", valid_rows[-3:])]:
        for i, row in enumerate(selected):
            idx = row["frame_id"]

            pts_vis, pts_swir, _ = detect_and_match_superpoint_lightglue(
                vis_files[idx],
                swir_files[idx],
                extractor,
                matcher,
                device,
            )

            pts_vis, pts_swir = remove_sky_matches(
                vis_files[idx],
                pts_vis,
                pts_swir,
                device,
            )

            if use_ransac:
                inliers = get_ransac_inliers(pts_vis, pts_swir, ransac_thresh)
                pts_vis = pts_vis[inliers]
                pts_swir = pts_swir[inliers]

            pts_vis, pts_swir = balance_matches_by_region(
                pts_vis,
                pts_swir,
                num_rows=2,
                num_cols=3,
                max_matches_per_region=180,
                min_matches_per_region=20,
            )

            draw_panel(
                vis_files[idx],
                swir_files[idx],
                find_concatenate_image(paths["method_dir"], vis_files[idx], swir_files[idx]),
                pts_vis,
                pts_swir,
                panels_dir / f"{prefix}_{i + 1:02d}_frame_{idx:04d}.png",
                title=f"{prefix.upper()} #{i + 1} | median={row['median_error']:.2f}px",
            )

    print("\nOutputs:")
    print("Framewise CSV:", framewise_csv)
    print("Summary CSV  :", summary_csv)
    print("Runtime CSV  :", runtime_csv)
    print("Plots        :", out_dir)
    print("Panels       :", panels_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", nargs="+", required=True)
    parser.add_argument("--num-eval", type=int, default=1000)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--no-ransac", action="store_true")
    parser.add_argument("--ransac-thresh", type=float, default=5.0)

    args = parser.parse_args()

    use_ransac = not args.no_ransac

    for method in args.method:
        run_method(
            dataset=args.dataset,
            method=method,
            num_eval=args.num_eval,
            max_keypoints=args.max_keypoints,
            use_ransac=use_ransac,
            ransac_thresh=args.ransac_thresh,
        )


if __name__ == "__main__":
    main()
