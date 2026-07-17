import cv2
import argparse
import numpy as np
from pathlib import Path
import shutil
import re


def gray(img):
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def resize(img, scale):
    h, w = img.shape[:2]
    return cv2.resize(
        img,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_LINEAR
    )


def estimate_scale_sift(method_ref, rasmd_ref):
    sift = cv2.SIFT_create(nfeatures=8000)

    kp1, des1 = sift.detectAndCompute(method_ref, None)
    kp2, des2 = sift.detectAndCompute(rasmd_ref, None)

    if des1 is None or des2 is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 20:
        return None

    pts_method = np.float32([kp1[m.queryIdx].pt for m in good])
    pts_rasmd = np.float32([kp2[m.trainIdx].pt for m in good])

    A, inliers = cv2.estimateAffinePartial2D(
        pts_method,
        pts_rasmd,
        method=cv2.RANSAC,
        ransacReprojThreshold=4.0
    )

    if A is None:
        return None

    a, b = A[0, 0], A[0, 1]
    scale = np.sqrt(a * a + b * b)

    if scale <= 0:
        return None

    return float(scale)


def estimate_crop_for_pair(method_path, rasmd_path, rw, rh):
    method_ref = gray(cv2.imread(str(method_path), cv2.IMREAD_UNCHANGED))
    rasmd_ref = gray(cv2.imread(str(rasmd_path), cv2.IMREAD_UNCHANGED))

    if method_ref is None or rasmd_ref is None:
        return None

    mh, mw = method_ref.shape[:2]

    scale = estimate_scale_sift(method_ref, rasmd_ref)

    if scale is None or (not np.isfinite(scale)) or scale <= 0.05:
        scale_w = rw / mw
        scale_h = rh / mh
        scale = max(scale_w, scale_h)

    if (not np.isfinite(scale)) or scale <= 0.05:
        return None

    method_scaled = resize(method_ref, scale)
    sh, sw = method_scaled.shape[:2]

    if sw < rw or sh < rh:
        scale = max(rw / mw, rh / mh) * 1.02
        method_scaled = resize(method_ref, scale)
        sh, sw = method_scaled.shape[:2]

    result = cv2.matchTemplate(method_scaled, rasmd_ref, cv2.TM_CCOEFF_NORMED)
    _, score, _, max_loc = cv2.minMaxLoc(result)

    x0, y0 = max_loc
    x1, y1 = x0 + rw, y0 + rh

    if x0 < 0 or y0 < 0 or x1 > sw or y1 > sh:
        return None

    return {
        "scale": float(scale),
        "x0": int(x0),
        "y0": int(y0),
        "score": float(score),
        "method_name": method_path.name,
        "rasmd_name": rasmd_path.name,
    }


def select_reference_indices(n, num_refs):
    if n <= num_refs:
        return list(range(n))
    return np.linspace(0, n - 1, num_refs).astype(int).tolist()


def get_pair_id(path: Path):
    match = re.search(r"(\d{6})", path.name)
    return match.group(1) if match else path.stem


def as_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def overlay_name(method_name, pair_id):
    if "opencv" in method_name.lower():
        return f"opencv_rgb_image_{pair_id}_overlay.jpg"
    return f"rgb_image_{pair_id}_concatenate.jpg"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-split", required=True)
    parser.add_argument("--rasmd-split", required=True)
    parser.add_argument("--out-split", required=True)
    parser.add_argument("--num-refs", type=int, default=20)
    parser.add_argument("--min-score", type=float, default=0.50)
    args = parser.parse_args()

    method_split = Path(args.method_split)
    rasmd_split = Path(args.rasmd_split)
    out_split = Path(args.out_split)
    method_name = out_split.parent.name
    out_concat = out_split.parent / "concatenate_images"

    if out_split.exists():
        shutil.rmtree(out_split)
    out_split.mkdir(parents=True, exist_ok=True)

    if out_concat.exists():
        shutil.rmtree(out_concat)
    out_concat.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Détecter automatiquement la direction.
    # --------------------------------------------------

    method_swir_paths = sorted(method_split.glob("*_swir_aligned.jpg"))
    method_vis_paths  = sorted(method_split.glob("*_vis_aligned.jpg"))

    rasmd_swir_paths = sorted(rasmd_split.glob("*_swir.jpg"))

    if len(method_swir_paths) > 0:
        direction = "SWIR_TO_VIS"
        aligned_paths = method_swir_paths

    elif len(method_vis_paths) > 0:
        direction = "VIS_TO_SWIR"
        aligned_paths = method_vis_paths

    else:
        raise FileNotFoundError(
            f"Aucun fichier *_swir_aligned.jpg ni *_vis_aligned.jpg trouvé dans {method_split}"
        )

    print(f"\nDirection détectée : {direction}")
    if len(rasmd_swir_paths) == 0:
        raise FileNotFoundError(
            f"Aucun fichier *_swir.jpg trouvé dans {rasmd_split}"
        )

    n = min(len(aligned_paths), len(rasmd_swir_paths))

    method_ref0 = gray(cv2.imread(str(aligned_paths[0]), cv2.IMREAD_UNCHANGED))
    rasmd_ref0 = gray(cv2.imread(str(rasmd_swir_paths[0]), cv2.IMREAD_UNCHANGED))

    mh, mw = method_ref0.shape[:2]
    rh, rw = rasmd_ref0.shape[:2]

    print(f"Method: {mw} x {mh}")
    print(f"RASMD : {rw} x {rh}")
    print(f"Refs  : {args.num_refs}")

    ref_indices = select_reference_indices(n, args.num_refs)

    estimates = []

    print("\nEstimation du crop global avec plusieurs références...\n")

    for idx in ref_indices:
        est = estimate_crop_for_pair(
            aligned_paths[idx],
            rasmd_swir_paths[idx],
            rw,
            rh
        )

        if est is None:
            print(f"[WARN] Réf {idx:04d} : échec")
            continue

        print(
            f"[REF {idx:04d}] "
            f"scale={est['scale']:.4f} "
            f"x={est['x0']} "
            f"y={est['y0']} "
            f"score={est['score']:.4f}"
        )

        if est["score"] >= args.min_score:
            estimates.append(est)
        else:
            print(f"         ignorée à cause d'un score faible")

    if len(estimates) == 0:
        raise RuntimeError("Aucune estimation valide. Diminuez --min-score ou vérifiez les images.")

    scales = np.array([e["scale"] for e in estimates])
    xs = np.array([e["x0"] for e in estimates])
    ys = np.array([e["y0"] for e in estimates])
    scores = np.array([e["score"] for e in estimates])

    scale_final = float(np.median(scales))
    x0_final = int(round(np.median(xs)))
    y0_final = int(round(np.median(ys)))

    x1_final = x0_final + rw
    y1_final = y0_final + rh

    print("\n" + "=" * 70)
    print("CROP GLOBAL FINAL")
    print(f"Réfs valides : {len(estimates)} / {len(ref_indices)}")
    print(f"Score moyen  : {scores.mean():.4f}")
    print(f"Score médian : {np.median(scores):.4f}")
    print(f"Scale final  : {scale_final:.4f}")
    print(f"Crop final   : x={x0_final}:{x1_final}, y={y0_final}:{y1_final}")
    print(f"Sortie finale: {rw} x {rh}")
    print("=" * 70)

    processed = 0

    for aligned_path in aligned_paths:

        if direction == "SWIR_TO_VIS":

            swir_path = aligned_path

            vis_path = Path(
                str(swir_path).replace(
                    "_swir_aligned.jpg",
                    "_vis.jpg"
                )
            )

        else:

            vis_path = aligned_path

            swir_path = Path(
                str(vis_path).replace(
                    "_vis_aligned.jpg",
                    "_swir.jpg"
                )
            )

        if not vis_path.exists():
            print(f"[WARN] Image VIS introuvable pour {swir_path.name}")
            continue

        swir = cv2.imread(str(swir_path), cv2.IMREAD_UNCHANGED)
        vis = cv2.imread(str(vis_path), cv2.IMREAD_UNCHANGED)

        swir_scaled = resize(swir, scale_final)
        vis_scaled = resize(vis, scale_final)

        sh, sw = swir_scaled.shape[:2]

        if x1_final > sw or y1_final > sh:
            print(f"[WARN] Crop hors limites pour {swir_path.name}")
            continue

        swir_crop = swir_scaled[y0_final:y1_final, x0_final:x1_final]
        vis_crop = vis_scaled[y0_final:y1_final, x0_final:x1_final]

        if direction == "SWIR_TO_VIS":

            cv2.imwrite(str(out_split / swir_path.name), swir_crop)
            cv2.imwrite(str(out_split / vis_path.name), vis_crop)

        else:

            cv2.imwrite(str(out_split / vis_path.name), vis_crop)
            cv2.imwrite(str(out_split / swir_path.name), swir_crop)

        vis_overlay = as_bgr(vis_crop)
        swir_overlay = as_bgr(swir_crop)
        overlay = cv2.addWeighted(vis_overlay, 0.5, swir_overlay, 0.5, 0)
        pair_id = get_pair_id(aligned_path)
        cv2.imwrite(str(out_concat / overlay_name(method_name, pair_id)), overlay)

        processed += 1

    print("\n✅ Terminé")
    print(f"Images traitées : {processed}")
    print(f"Sortie : {out_split}")
    print(f"Overlays : {out_concat}")


if __name__ == "__main__":
    main()
