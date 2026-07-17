"""Analyse content-based fields of view across aligned split_images methods.

The script is read-only with respect to benchmark inputs. It registers VIS images
from several methods in an unrestricted global canvas, projects valid-image masks,
and reports areas, containment and leave-one-out limiting gains.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "benchmark_results" / "fov_analysis" / "cloudy_six_methods"


@dataclass(frozen=True)
class Method:
    name: str
    split_dir: Path


METHODS = [
    Method(
        "mouad_swir_to_vis",
        Path("/home/lizbeth/Desktop/intership/cross-spectral-visible-swir-calibration-for-embedded-image-fusion/datas/cloudy_test_out_generalized_h/split_images"),
    ),
    Method(
        "opencv_swir_to_vis",
        ROOT / "datas/cloudy_test/Sortie_Extraite/paired_images/opencv_alignment_SWIR_TO_VIS_train150_step5_test500_to_1499/split_images",
    ),
    Method(
        "opencv_vis_to_swir",
        ROOT / "datas/cloudy_test/Sortie_Extraite/paired_images/opencv_alignment_VIS_TO_SWIR_train150_step5_test500_to_1499/split_images",
    ),
    Method(
        "patch_swir_to_vis",
        ROOT / "datas/cloudy_test/Sortie_Extraite/paired_images/patch_RANSAC_HOMOGRAPHY_calibration_SWIR_to_VIS_H_from_objectif_12mm_4_boucle/split_images",
    ),
    Method(
        "patch_vis_to_swir",
        ROOT / "datas/cloudy_test/Sortie_Extraite/paired_images/patch_RANSAC_HOMOGRAPHY_calibration_VIS_to_SWIR_H_from_objectif_12mm_4_boucle/split_images",
    ),
    Method(
        "rasmd",
        ROOT / "datas/cloudy_test/Sortie_Extraite/paired_images/RASMD_aligned_baseline/split_images",
    ),
]


VIS_SUFFIXES = ("_vis_aligned", "_visible", "_vis")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#00a9a5"]


def frame_id(path: Path) -> int | None:
    numbers = re.findall(r"\d+", path.stem)
    return int(numbers[-1]) if numbers else None


def discover_vis(method: Method) -> dict[int, Path]:
    images: dict[int, Path] = {}
    for path in sorted(method.split_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        if not path.stem.lower().endswith(VIS_SUFFIXES):
            continue
        fid = frame_id(path)
        if fid is not None:
            images[fid] = path
    return images


def gray_for_features(path: Path, max_side: int = 1400) -> tuple[np.ndarray, float]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot read {path}")
    scale = min(1.0, max_side / max(image.shape))
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    return image, scale


def matched_points(
    source_path: Path,
    anchor_path: Path,
    sift: cv2.SIFT,
) -> tuple[np.ndarray, np.ndarray, int]:
    source, source_scale = gray_for_features(source_path)
    anchor, anchor_scale = gray_for_features(anchor_path)
    kp_s, des_s = sift.detectAndCompute(source, None)
    kp_a, des_a = sift.detectAndCompute(anchor, None)
    if des_s is None or des_a is None:
        return np.empty((0, 2)), np.empty((0, 2)), 0
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(des_s, des_a, k=2)
    reverse = matcher.knnMatch(des_a, des_s, k=2)
    good_f = {m.queryIdx: m for m, n in forward if m.distance < 0.72 * n.distance}
    good_r = {m.queryIdx: m for m, n in reverse if m.distance < 0.72 * n.distance}
    mutual = [m for query, m in good_f.items() if m.trainIdx in good_r and good_r[m.trainIdx].trainIdx == query]
    source_pts = np.float32([kp_s[m.queryIdx].pt for m in mutual]) / source_scale
    anchor_pts = np.float32([kp_a[m.trainIdx].pt for m in mutual]) / anchor_scale
    return source_pts, anchor_pts, len(mutual)


def estimate_transform(
    source: Method,
    anchor: Method,
    image_maps: dict[str, dict[int, Path]],
    selected_frames: list[int],
    transform_model: str,
) -> tuple[np.ndarray, dict]:
    if source.name == anchor.name:
        return np.eye(3, dtype=np.float64), {"matches": 0, "inliers": 0, "inlier_ratio": 1.0}
    sift = cv2.SIFT_create(nfeatures=7000, contrastThreshold=0.02)
    all_source = []
    all_anchor = []
    per_frame = {}
    for fid in selected_frames:
        src, dst, count = matched_points(image_maps[source.name][fid], image_maps[anchor.name][fid], sift)
        per_frame[str(fid)] = count
        if count >= 8:
            all_source.append(src)
            all_anchor.append(dst)
    if not all_source:
        raise RuntimeError(f"No robust VIS matches for {source.name}")
    source_pts = np.concatenate(all_source).reshape(-1, 1, 2)
    anchor_pts = np.concatenate(all_anchor).reshape(-1, 1, 2)
    H, inliers = cv2.findHomography(
        source_pts,
        anchor_pts,
        cv2.USAC_MAGSAC,
        3.0,
        maxIters=20000,
        confidence=0.999,
    )
    if H is None or inliers is None:
        raise RuntimeError(f"Homography estimation failed for {source.name}")
    H = H / H[2, 2]
    inlier_count = int(inliers.sum())
    robust_mask = inliers.ravel().astype(bool)
    if transform_model == "rectilinear":
        # Ajuste restringido que conserva rectangulos horizontales:
        # x_anchor = sx*x_source + tx; y_anchor = sy*y_source + ty.
        src_xy = source_pts.reshape(-1, 2)[robust_mask]
        dst_xy = anchor_pts.reshape(-1, 2)[robust_mask]
        keep = np.ones(len(src_xy), dtype=bool)
        for _ in range(5):
            sx, tx = np.polyfit(src_xy[keep, 0], dst_xy[keep, 0], 1)
            sy, ty = np.polyfit(src_xy[keep, 1], dst_xy[keep, 1], 1)
            predicted = np.column_stack((sx * src_xy[:, 0] + tx, sy * src_xy[:, 1] + ty))
            residual = np.linalg.norm(predicted - dst_xy, axis=1)
            median = float(np.median(residual[keep]))
            mad = float(np.median(np.abs(residual[keep] - median))) + 1e-8
            new_keep = residual <= median + 3.5 * 1.4826 * mad
            if np.array_equal(new_keep, keep):
                break
            keep = new_keep
        H = np.array([[sx, 0.0, tx], [0.0, sy, ty], [0.0, 0.0, 1.0]], dtype=np.float64)
        rectified_residual = residual[keep]
    elif transform_model != "projective":
        raise ValueError(f"Unknown transform model: {transform_model}")
    diagnostics = {
        "matches": int(len(source_pts)),
        "inliers": inlier_count,
        "inlier_ratio": inlier_count / len(source_pts),
        "matches_per_frame": per_frame,
        "transform_model": transform_model,
    }
    if transform_model == "rectilinear":
        diagnostics.update(
            {
                "rectilinear_inliers": int(keep.sum()),
                "rectilinear_rmse_px": float(np.sqrt(np.mean(rectified_residual ** 2))),
                "scale_x": float(H[0, 0]),
                "scale_y": float(H[1, 1]),
                "translation_x": float(H[0, 2]),
                "translation_y": float(H[1, 2]),
            }
        )
    return H, diagnostics


def border_invalid_mask(image_path: Path, black_threshold: int = 10) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Cannot read {image_path}")
    if image.ndim == 3 and image.shape[2] == 4:
        alpha_valid = image[:, :, 3] > 0
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        alpha_valid = np.ones(image.shape[:2], dtype=bool)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    dark = ((gray <= black_threshold) | ~alpha_valid).astype(np.uint8)
    count, labels = cv2.connectedComponents(dark, connectivity=8)
    border_labels = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    invalid = np.isin(labels, border_labels[border_labels != 0]) if count > 1 else np.zeros_like(dark, dtype=bool)
    valid = (~invalid) & alpha_valid
    return valid.astype(np.uint8)


def majority_native_mask(paths: list[Path]) -> np.ndarray:
    masks = [border_invalid_mask(path) for path in paths]
    shape = masks[0].shape
    if any(mask.shape != shape for mask in masks):
        raise RuntimeError("Native dimensions change between selected frames")
    votes = np.sum(masks, axis=0)
    majority = (votes >= int(np.ceil(len(masks) * 0.6))).astype(np.uint8)
    # El campo visual util debe ser la gran region conectada; pequeños islotes
    # JPEG separados del contenido principal no representan un FOV adicional.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(majority, connectivity=8)
    if count <= 1:
        return majority
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_label).astype(np.uint8)


def transformed_corners(H: np.ndarray, width: int, height: int) -> np.ndarray:
    corners = np.float32([[[0, 0], [width, 0], [width, height], [0, height]]])
    return cv2.perspectiveTransform(corners, H)[0]


def global_canvas(
    transforms: dict[str, np.ndarray], native_shapes: dict[str, tuple[int, int]], margin: int = 80
) -> tuple[np.ndarray, tuple[int, int], dict]:
    corners = []
    method_corners = {}
    for name, H in transforms.items():
        height, width = native_shapes[name]
        projected = transformed_corners(H, width, height)
        method_corners[name] = projected.tolist()
        corners.append(projected)
    points = np.concatenate(corners)
    min_x, min_y = np.floor(points.min(axis=0) - margin).astype(int)
    max_x, max_y = np.ceil(points.max(axis=0) + margin).astype(int)
    translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float64)
    return translation, (int(max_x - min_x), int(max_y - min_y)), method_corners


def largest_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    heights = np.zeros(mask.shape[1], dtype=np.int32)
    best_area = 0
    best = (0, 0, 0, 0)
    for y, row in enumerate(mask.astype(bool)):
        heights = np.where(row, heights + 1, 0)
        stack: list[tuple[int, int]] = []
        for x in range(len(heights) + 1):
            height = int(heights[x]) if x < len(heights) else 0
            start = x
            while stack and stack[-1][1] > height:
                index, old_height = stack.pop()
                area = old_height * (x - index)
                if area > best_area:
                    best_area = area
                    best = (index, y - old_height + 1, x - index, old_height)
                start = index
            if not stack or stack[-1][1] < height:
                stack.append((start, height))
    return best


def save_mask(path: Path, mask: np.ndarray) -> None:
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def save_registration_check(
    output: Path,
    method: Method,
    anchor: Method,
    fid: int,
    H_canvas: np.ndarray,
    anchor_canvas: np.ndarray,
    canvas_size: tuple[int, int],
    image_maps: dict[str, dict[int, Path]],
) -> None:
    source = cv2.imread(str(image_maps[method.name][fid]), cv2.IMREAD_COLOR)
    anchor_image = cv2.imread(str(image_maps[anchor.name][fid]), cv2.IMREAD_COLOR)
    warped_source = cv2.warpPerspective(source, H_canvas, canvas_size)
    warped_anchor = cv2.warpPerspective(anchor_image, anchor_canvas, canvas_size)
    overlay = cv2.addWeighted(warped_anchor, 0.5, warped_source, 0.5, 0)
    cv2.imwrite(str(output / f"registration_check_{method.name}_frame_{fid}.jpg"), overlay)


def contour_visualization(output: Path, footprints: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(14, 9), constrained_layout=True)
    for (name, mask), color in zip(footprints.items(), COLORS):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            points = contour[:, 0, :]
            ax.plot(points[:, 0], points[:, 1], color=color, linewidth=2, label=name)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title("Registered visual field contours")
    ax.set_xlabel("Global canvas x")
    ax.set_ylabel("Global canvas y")
    fig.savefig(output / "all_registered_footprint_contours.png", dpi=180)
    plt.close(fig)


def limiting_visualization(
    output: Path,
    common: np.ndarray,
    common_without: dict[str, np.ndarray],
) -> None:
    canvas = np.zeros((*common.shape, 3), dtype=np.uint8)
    canvas[common > 0] = (220, 220, 220)
    for (name, without), color in zip(common_without.items(), COLORS):
        gain = (without > 0) & (common == 0)
        rgb = tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))
        canvas[gain] = rgb
    fig, ax = plt.subplots(figsize=(14, 9), constrained_layout=True)
    ax.imshow(canvas)
    for name, color in zip(common_without, COLORS):
        ax.plot([], [], color=color, linewidth=6, label=name)
    ax.plot([], [], color="#dcdcdc", linewidth=6, label="common to all")
    ax.legend(loc="upper right")
    ax.set_title("Regions gained when each method is removed")
    ax.axis("off")
    fig.savefig(output / "limiting_methods_by_region.png", dpi=180)
    plt.close(fig)


def individual_rectangle_visualization(
    output: Path,
    footprints: dict[str, np.ndarray],
    common: np.ndarray,
) -> None:
    """Muestra cada rectangulo por separado con color estable y zona comun."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax, ((name, footprint), color) in zip(axes.ravel(), zip(footprints.items(), COLORS)):
        rgb = tuple(int(color[index : index + 2], 16) / 255 for index in (1, 3, 5))
        layer = np.zeros((*footprint.shape, 4), dtype=np.float32)
        layer[footprint > 0] = (*rgb, 0.78)
        layer[common > 0] = (0.92, 0.92, 0.92, 1.0)
        ax.imshow(layer)
        contours, _ = cv2.findContours(footprint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)[:, 0, :]
            ax.plot(contour[:, 0], contour[:, 1], color=color, linewidth=2.5)
        area = int(footprint.sum())
        common_area = int(common.sum())
        coverage = common_area / area if area else 0.0
        ax.set_title(f"{name}\ncommon covers {coverage:.2%}; excess {area - common_area:,} px")
        ax.axis("off")
    fig.suptitle("Each registered rectangle (method color) and six-way common region (gray)", fontsize=15)
    fig.savefig(output / "individual_registered_rectangles.png", dpi=180)
    plt.close(fig)


def limiting_gain_chart(output: Path, results: pd.DataFrame) -> None:
    """Grafico inequívoco del aumento de area comun al retirar cada metodo."""
    color_by_name = {method.name: color for method, color in zip(METHODS, COLORS)}
    ordered = results.sort_values("absolute_limiting_gain", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    bars = ax.barh(
        ordered["method"],
        ordered["absolute_limiting_gain"],
        color=[color_by_name[name] for name in ordered["method"]],
    )
    for bar, (_, row) in zip(bars, ordered.iterrows()):
        ax.text(
            bar.get_width() + max(results["absolute_limiting_gain"].max() * 0.012, 20),
            bar.get_y() + bar.get_height() / 2,
            f"{int(row['absolute_limiting_gain']):,} px ({row['relative_limiting_gain']:.2%})",
            va="center",
        )
    ax.set_xlabel("Additional common area when method is removed (pixels)")
    ax.set_title("Which rectangle limits the six-way common region?")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(output / "limiting_gain_by_method.png", dpi=180)
    plt.close(fig)


def run(
    output: Path,
    sample_count: int = 9,
    anchor_name: str = "opencv_swir_to_vis",
    mask_mode: str = "rectangle",
    transform_model: str = "rectilinear",
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    image_maps = {method.name: discover_vis(method) for method in METHODS}
    common_ids = sorted(set.intersection(*(set(images) for images in image_maps.values())))
    if not common_ids:
        raise RuntimeError("No common VIS frame IDs")
    sample_indices = np.linspace(0, len(common_ids) - 1, sample_count, dtype=int)
    selected_frames = [common_ids[index] for index in sample_indices]
    anchor = next(method for method in METHODS if method.name == anchor_name)

    transforms = {}
    registration = {}
    for method in METHODS:
        H, diagnostics = estimate_transform(
            method, anchor, image_maps, selected_frames, transform_model=transform_model
        )
        transforms[method.name] = H
        registration[method.name] = diagnostics

    native_masks = {}
    native_shapes = {}
    inventory_rows = []
    for method in METHODS:
        paths = [image_maps[method.name][fid] for fid in selected_frames]
        sample = cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED)
        if sample is None:
            raise RuntimeError(f"Cannot read {paths[0]}")
        if mask_mode == "rectangle":
            # Cada split_image aporta todo el rectangulo nativo. No se elimina
            # padding ni se infiere una forma irregular desde el contenido.
            mask = np.ones(sample.shape[:2], dtype=np.uint8)
        elif mask_mode == "valid-content":
            mask = majority_native_mask(paths)
        else:
            raise ValueError(f"Unknown mask mode: {mask_mode}")
        native_masks[method.name] = mask
        native_shapes[method.name] = mask.shape
        inventory_rows.append(
            {
                "method": method.name,
                "split_dir": str(method.split_dir),
                "vis_frames": len(image_maps[method.name]),
                "native_width": mask.shape[1],
                "native_height": mask.shape[0],
                "native_valid_pixels": int(mask.sum()),
                "mask_mode": mask_mode,
            }
        )

    translation, canvas_size, anchor_corners = global_canvas(transforms, native_shapes)
    canvas_transforms = {name: translation @ H for name, H in transforms.items()}
    footprints = {
        name: (cv2.warpPerspective(mask, canvas_transforms[name], canvas_size, flags=cv2.INTER_NEAREST) > 0).astype(np.uint8)
        for name, mask in native_masks.items()
    }
    common = np.logical_and.reduce([mask > 0 for mask in footprints.values()]).astype(np.uint8)
    common_area = int(common.sum())
    if common_area == 0:
        raise RuntimeError("Registered footprints have an empty intersection")
    x, y, width, height = cv2.boundingRect(common)
    rectangle = largest_rectangle(common)

    names = [method.name for method in METHODS]
    pairwise = pd.DataFrame(index=names, columns=names, dtype=float)
    common_without = {}
    rows = []
    areas = {name: int(mask.sum()) for name, mask in footprints.items()}
    for name_i in names:
        for name_j in names:
            overlap = int(np.logical_and(footprints[name_i], footprints[name_j]).sum())
            pairwise.loc[name_i, name_j] = overlap / areas[name_i] if areas[name_i] else np.nan
    for name in names:
        others = [footprints[other] > 0 for other in names if other != name]
        without = np.logical_and.reduce(others).astype(np.uint8)
        common_without[name] = without
        without_area = int(without.sum())
        gain = without_area - common_area
        containment_other_5 = float(min(pairwise.loc[name, other] for other in names if other != name))
        rows.append(
            {
                "method": name,
                "native_width": native_shapes[name][1],
                "native_height": native_shapes[name][0],
                "registered_area": areas[name],
                "common_coverage_of_method": common_area / areas[name],
                "excess_area_outside_common": areas[name] - common_area,
                "containment_in_other_5": containment_other_5,
                "common_area_without_method": without_area,
                "absolute_limiting_gain": gain,
                "relative_limiting_gain": gain / common_area,
            }
        )
    results = pd.DataFrame(rows).sort_values("registered_area").reset_index(drop=True)
    results["area_rank"] = np.arange(1, len(results) + 1)
    results["is_smallest_fov_candidate"] = results["containment_in_other_5"] >= 0.95
    max_gain = results["absolute_limiting_gain"].max()
    results["is_main_limiting_method"] = results["absolute_limiting_gain"] == max_gain
    results = results[
        [
            "method", "native_width", "native_height", "registered_area", "area_rank",
            "common_coverage_of_method", "excess_area_outside_common",
            "containment_in_other_5", "common_area_without_method", "absolute_limiting_gain",
            "relative_limiting_gain", "is_smallest_fov_candidate", "is_main_limiting_method",
        ]
    ]

    pd.DataFrame(inventory_rows).to_csv(output / "inventory.csv", index=False)
    results.to_csv(output / "fov_analysis.csv", index=False)
    pairwise.to_csv(output / "pairwise_containment.csv")
    pd.DataFrame(registration).T.to_json(output / "registration_diagnostics.json", indent=2)
    np.savez(output / "transforms_to_temporary_anchor.npz", **transforms)
    for name, mask in footprints.items():
        save_mask(output / f"footprint_{name}.png", mask)
    save_mask(output / "common_intersection.png", common)
    rectangle_mask = np.zeros_like(common)
    rx, ry, rw, rh = rectangle
    rectangle_mask[ry : ry + rh, rx : rx + rw] = 1
    rectangle_overlay = cv2.cvtColor(common * 255, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(rectangle_overlay, (rx, ry), (rx + rw - 1, ry + rh - 1), (0, 0, 255), 4)
    cv2.imwrite(str(output / "largest_common_rectangle.png"), rectangle_overlay)
    contour_visualization(output, footprints)
    limiting_visualization(output, common, common_without)
    individual_rectangle_visualization(output, footprints, common)
    limiting_gain_chart(output, results)
    check_frame = selected_frames[len(selected_frames) // 2]
    for method in METHODS:
        save_registration_check(
            output, method, anchor, check_frame, canvas_transforms[method.name],
            canvas_transforms[anchor.name], canvas_size, image_maps,
        )

    candidates = results.loc[results["is_smallest_fov_candidate"], "method"].tolist()
    summary = {
        "mask_mode": mask_mode,
        "transform_model": transform_model,
        "temporary_anchor": anchor.name,
        "common_frame_count": len(common_ids),
        "common_frame_range": [common_ids[0], common_ids[-1]],
        "registration_frames": selected_frames,
        "canvas_width": canvas_size[0],
        "canvas_height": canvas_size[1],
        "common_area": common_area,
        "common_bbox": {"x": x, "y": y, "width": width, "height": height},
        "largest_common_rectangle": {"x": rx, "y": ry, "width": rw, "height": rh},
        "smallest_registered_fov": results.iloc[0]["method"],
        "most_contained_method": results.sort_values("containment_in_other_5", ascending=False).iloc[0]["method"],
        "main_limiting_method": results.sort_values("absolute_limiting_gain", ascending=False).iloc[0]["method"],
        "unique_small_reference_exists": len(candidates) == 1,
        "small_reference_candidates_over_95_percent": candidates,
        "transformed_native_corners_in_anchor_coordinates": anchor_corners,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-count", type=int, default=9)
    parser.add_argument("--anchor", choices=[method.name for method in METHODS], default="opencv_swir_to_vis")
    parser.add_argument("--mask-mode", choices=["rectangle", "valid-content"], default="rectangle")
    parser.add_argument("--transform-model", choices=["rectilinear", "projective"], default="rectilinear")
    args = parser.parse_args()
    summary = run(
        args.output.resolve(),
        sample_count=args.sample_count,
        anchor_name=args.anchor,
        mask_mode=args.mask_mode,
        transform_model=args.transform_model,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
