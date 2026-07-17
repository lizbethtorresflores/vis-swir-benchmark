# registration/pipeline/estimation.py
import os
import cv2
import numpy as np
from ..io.homography_io import save_homography_and_mask, load_homography_and_mask

def remove_bad_spatial_matches(pts_vis, pts_swir, H, config):
    """
    Filtre les mauvaises correspondances par zones spatiales.
    Divise l'image VIS en une grille et supprime les cellules avec une erreur médiane élevée.
    """

    errors = reprojection_errors_all(H, pts_vis, pts_swir)

    grid_rows = getattr(config, "SPATIAL_FILTER_ROWS", 4)
    grid_cols = getattr(config, "SPATIAL_FILTER_COLS", 4)
    min_matches = getattr(config, "SPATIAL_FILTER_MIN_MATCHES", 10)
    error_factor = getattr(config, "SPATIAL_FILTER_ERROR_FACTOR", 2.0)
    min_threshold = getattr(config, "SPATIAL_FILTER_MIN_THRESHOLD", 5.0)

    x = pts_vis[:, 0]
    y = pts_vis[:, 1]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_bins = np.linspace(x_min, x_max, grid_cols + 1)
    y_bins = np.linspace(y_min, y_max, grid_rows + 1)

    global_med = np.median(errors)
    threshold = max(global_med * error_factor, min_threshold)

    keep = np.ones(len(pts_vis), dtype=bool)

    print("\nAUTO SPATIAL FILTER")
    print(f"Global median error: {global_med:.2f} px")
    print(f"Threshold bad cells: {threshold:.2f} px")

    for r in range(grid_rows):
        for c in range(grid_cols):
            x0, x1 = x_bins[c], x_bins[c + 1]
            y0, y1 = y_bins[r], y_bins[r + 1]

            cell_mask = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)

            if cell_mask.sum() < min_matches:
                continue

            cell_med = np.median(errors[cell_mask])

            if cell_med > threshold:
                keep[cell_mask] = False
                print(
                    f"Bad cell r={r}, c={c} | "
                    f"matches={cell_mask.sum()} | "
                    f"median error={cell_med:.2f} px"
                )

    print(f"Matches kept after spatial filter: {keep.sum()}/{len(keep)}")

    if keep.sum() < 20:
        print("WARNING: Too few matches after spatial filter. Keeping original matches.")
        return pts_vis, pts_swir

    return pts_vis[keep], pts_swir[keep]

def estimate_homography(pts_vis_all: np.ndarray, pts_swir_all: np.ndarray, config) -> tuple:
    """
    Estime la matrice d'homographie à partir de l'ensemble des correspondances
    multi-images, en utilisant MAGSAC ou RANSAC selon config.USE_MAGSAC.
    Vérifie la stabilité numérique via le conditionnement de la matrice.

    Args:
        pts_vis_all:  Points source VIS agrégés (N, 2) float32.
        pts_swir_all: Points destination SWIR agrégés (N, 2) float32.
        config:       Instance Config portant USE_MAGSAC, MAGSAC_THRESHOLD,
                      MAGSAC_CONFIDENCE, MAGSAC_MAX_ITERS (ou variantes RANSAC),
                      et COND_NUMBER_THRESHOLD.

    Returns:
        Tuple (H, inliers) : matrice 3×3 float64 et masque booléen 1D.

    Raises:
        RuntimeError: Si l'homographie n'est pas trouvée ou est numériquement instable.
    """
    direction = getattr(config, "ALIGN_DIRECTION", "VIS_TO_SWIR")

    if direction == "VIS_TO_SWIR":
        src_pts = pts_vis_all.astype(np.float32).reshape(-1, 1, 2)
        dst_pts = pts_swir_all.astype(np.float32).reshape(-1, 1, 2)

    elif direction == "SWIR_TO_VIS":
        src_pts = pts_swir_all.astype(np.float32).reshape(-1, 1, 2)
        dst_pts = pts_vis_all.astype(np.float32).reshape(-1, 1, 2)

    else:
        raise ValueError(f"ALIGN_DIRECTION invalide : {direction}")

    if config.USE_MAGSAC:
        H, mask = cv2.findHomography(
            src_pts, dst_pts,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=config.MAGSAC_THRESHOLD,
            confidence=config.MAGSAC_CONFIDENCE,
            maxIters=config.MAGSAC_MAX_ITERS,
        )
    else:
        H, mask = cv2.findHomography(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=config.RANSAC_THRESHOLD,
            confidence=config.RANSAC_STOP_PROB,
            maxIters=config.RANSAC_MAX_TRIALS,
        )

    if H is None:
        raise RuntimeError("Homographie non trouvée — pas assez de matches valides.")

    det  = np.linalg.det(H)
    cond = np.linalg.cond(H)
    inliers = mask.ravel() == 1
    pts_inliers = pts_vis_all[inliers]

    width = np.max(pts_vis_all[:, 0])
    height = np.max(pts_vis_all[:, 1])

    x = pts_inliers[:, 0]
    y = pts_inliers[:, 1]

    left = np.sum(x < width / 3)
    center = np.sum((x >= width / 3) & (x < 2 * width / 3))
    right = np.sum(x >= 2 * width / 3)

    top = np.sum(y < height / 2)
    bottom = np.sum(y >= height / 2)

    print("\nINLIERS AFTER MAGSAC / RANSAC")
    print(f"LEFT   : {left}")
    print(f"CENTER : {center}")
    print(f"RIGHT  : {right}")
    print(f"TOP    : {top}")
    print(f"BOTTOM : {bottom}")
    print(f"Inliers : {inliers.sum()}/{len(inliers)} ({100*inliers.sum()/len(inliers):.1f}%)")
    print(f"Stabilité : det={det:.3e}, cond={cond:.2e}")

    if cond > config.COND_NUMBER_THRESHOLD:
        raise RuntimeError(f"Homographie instable (cond={cond:.2e} > {config.COND_NUMBER_THRESHOLD:.0e})")

    return H, inliers

def reprojection_errors_all(H, pts_vis, pts_swir):
    pts_vis_h = np.hstack([pts_vis, np.ones((len(pts_vis), 1))])
    pts_proj = (H @ pts_vis_h.T).T
    pts_proj = pts_proj[:, :2] / pts_proj[:, 2:3]
    errors = np.linalg.norm(pts_swir - pts_proj, axis=1)
    return errors


def remove_bad_bottom_matches(pts_vis, pts_swir, H, config):

    errors = reprojection_errors_all(H, pts_vis, pts_swir)
    y = pts_vis[:, 1]

    n_bands = getattr(config, "AUTO_BOTTOM_N_BANDS", 20)
    min_matches = getattr(config, "AUTO_BOTTOM_MIN_MATCHES", 30)
    error_factor = getattr(config, "AUTO_BOTTOM_ERROR_FACTOR", 2.0)

    y_min, y_max = y.min(), y.max()
    bands = np.linspace(y_min, y_max, n_bands + 1)

    global_med = np.median(errors)
    threshold = max(global_med * error_factor, 5.0)

    bad = np.zeros(n_bands, dtype=bool)

    print("\nAUTO BOTTOM FILTER")
    print(f"Global median error: {global_med:.2f} px")

    for b in range(n_bands):
        y0, y1 = bands[b], bands[b + 1]

        m = (y >= y0) & (y < y1)

        if m.sum() < min_matches:
            continue

        med_err = np.median(errors[m])

        if med_err > threshold:
            bad[b] = True

    first_bad_from_bottom = None

    for b in range(n_bands - 1, -1, -1):
        if bad[b]:
            first_bad_from_bottom = b
        else:
            if first_bad_from_bottom is not None:
                break

    if first_bad_from_bottom is None:
        print("No bad bottom region detected.")
        return pts_vis, pts_swir

    crop_y = bands[first_bad_from_bottom]

    keep = y < crop_y

    print(f"Detected bad bottom starts at y ≈ {crop_y:.1f}")

    return pts_vis[keep], pts_swir[keep]

def compute_homography(container, config, paths) -> tuple:

    if config.LOAD_HOMOGRAPHY:
        print("\nChargement homographie existante")

        H, mask = load_homography_and_mask(
            os.path.join(paths['out_dir'], "homography.npy.npz")
        )

        return H, mask.ravel() == 1

    pts_vis_all, pts_swir_all = container.get_all_matches()

    # ==========================================================
    # HOMOGRAPHIE INITIALE
    # ==========================================================

    print("\n[1/2] Homographie initiale")

    H_initial, inliers_initial = estimate_homography(
        pts_vis_all,
        pts_swir_all,
        config
    )

    pts_vis_filtered = pts_vis_all
    pts_swir_filtered = pts_swir_all

    if getattr(config, "AUTO_FILTER_BAD_BOTTOM", False):
        pts_vis_filtered, pts_swir_filtered = remove_bad_bottom_matches(
            pts_vis_filtered,
            pts_swir_filtered,
            H_initial,
            config
        )

    if getattr(config, "AUTO_FILTER_BAD_SPATIAL", False):
        pts_vis_filtered, pts_swir_filtered = remove_bad_spatial_matches(
            pts_vis_filtered,
            pts_swir_filtered,
            H_initial,
            config
        )

    # ==========================================================
    # HOMOGRAPHIE FINALE
    # ==========================================================

    print("\n[2/2] Homographie finale")

    H, inliers = estimate_homography(
        pts_vis_filtered,
        pts_swir_filtered,
        config
    )

    save_homography_and_mask(
        H,
        inliers,
        os.path.join(paths['out_dir'], "homography")
    )

    return H, inliers
