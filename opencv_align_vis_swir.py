import cv2
import numpy as np
from pathlib import Path
import argparse
import time
import shutil

parser = argparse.ArgumentParser(description="OpenCV VIS-SWIR global homography benchmark")

parser.add_argument("--dataset", required=True)
parser.add_argument("--direction", choices=["VIS_TO_SWIR", "SWIR_TO_VIS"], default="VIS_TO_SWIR")

parser.add_argument("--train-start", type=int, default=0)
parser.add_argument("--train-step", type=int, default=5)
parser.add_argument("--train-count", type=int, default=50)
parser.add_argument("--test-start", type=int, default=300)
parser.add_argument("--test-end", type=int, default=799)
parser.add_argument("--crop-refs", type=int, default=30)
parser.add_argument("--crop-margin", type=int, default=10)

args = parser.parse_args()

DATASET = args.dataset
DIRECTION = args.direction

BASE = Path(f"/home/lizbeth/Desktop/intership/recallage-multimodal-zz3/datas/{DATASET}/Sortie_Extraite/paired_images")

RGB_DIR = BASE / "VISIBLE"
SWIR_DIR = BASE / "SWIR"

OUT_BASE = BASE / (
    f"opencv_alignment_{DIRECTION}"
    f"_train{args.train_count}"
    f"_step{args.train_step}"
    f"_test{args.test_start}_to_{args.test_end}"
)

OUT_DIR = OUT_BASE / "split_images"
CONCAT_DIR = OUT_BASE / "concatenate_images"
TRAIN_VIS_DIR = OUT_BASE / "train_images" / "VISIBLE"
TRAIN_SWIR_DIR = OUT_BASE / "train_images" / "SWIR"

for d in [OUT_DIR, CONCAT_DIR, TRAIN_VIS_DIR, TRAIN_SWIR_DIR]:
    d.mkdir(parents=True, exist_ok=True)

N_FEATURES = 5000
RANSAC_THRESH = 5.0

rgb_files = sorted(RGB_DIR.glob("*.jpg"))
swir_files = sorted(SWIR_DIR.glob("*.jpg"))

max_len = min(len(rgb_files), len(swir_files))

if max_len == 0:
    raise RuntimeError("Aucune image trouvée dans VISIBLE/SWIR.")

train_indices = [args.train_start + i * args.train_step for i in range(args.train_count)]
test_indices = list(range(args.test_start, args.test_end + 1))

train_indices = [i for i in train_indices if i < max_len]
test_indices = [i for i in test_indices if i < max_len]

rgb_est = [rgb_files[i] for i in train_indices]
swir_est = [swir_files[i] for i in train_indices]

rgb_apply = [rgb_files[i] for i in test_indices]
swir_apply = [swir_files[i] for i in test_indices]

if len(rgb_est) == 0:
    raise RuntimeError("Aucune image train sélectionnée.")

if len(rgb_apply) == 0:
    raise RuntimeError("Aucune image test sélectionnée.")


def largest_valid_rectangle(mask):
    h, w = mask.shape
    heights = np.zeros(w, dtype=np.int32)
    best_area = 0
    best_rect = (0, 0, w, h)

    for y in range(h):
        heights = heights + mask[y]
        heights[mask[y] == 0] = 0

        stack = []

        for x in range(w + 1):
            current = heights[x] if x < w else 0

            while stack and current < heights[stack[-1]]:
                top = stack.pop()
                height = heights[top]
                left = stack[-1] + 1 if stack else 0
                right = x
                area = height * (right - left)

                if area > best_area:
                    best_area = area
                    x1 = left
                    x2 = right
                    y1 = y - height + 1
                    y2 = y + 1
                    best_rect = (x1, y1, x2, y2)

            stack.append(x)

    return best_rect


def valid_crop_rect(valid_mask, margin=10):
    mask = (valid_mask > 0).astype(np.uint8)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)

    if np.count_nonzero(mask) == 0:
        h, w = valid_mask.shape[:2]
        return 0, 0, w, h

    x1, y1, x2, y2 = largest_valid_rectangle(mask)
    h, w = valid_mask.shape[:2]

    x1 = max(x1 + margin, 0)
    y1 = max(y1 + margin, 0)
    x2 = min(x2 - margin, w)
    y2 = min(y2 - margin, h)

    if x2 <= x1 or y2 <= y1:
        return 0, 0, w, h

    return x1, y1, x2, y2


def select_reference_indices(n, num_refs):
    if n <= num_refs:
        return list(range(n))
    return np.linspace(0, n - 1, num_refs).astype(int).tolist()


def valid_mask_for_pair(rgb, swir, H, direction):
    if direction == "VIS_TO_SWIR":
        h, w = swir.shape[:2]
        src_mask = np.ones(rgb.shape[:2], dtype=np.uint8) * 255
    else:
        h, w = rgb.shape[:2]
        src_mask = np.ones(swir.shape[:2], dtype=np.uint8) * 255

    return cv2.warpPerspective(
        src_mask,
        H,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )


def estimate_global_crop_rect(rgb_paths, swir_paths, H, direction, num_refs, margin):
    rects = []
    ref_indices = select_reference_indices(len(rgb_paths), num_refs)

    print("\n" + "=" * 80)
    print("ESTIMATION CROP GLOBAL OPENCV")
    print("=" * 80)
    print(f"Références crop : {len(ref_indices)}")

    for idx in ref_indices:
        rgb = cv2.imread(str(rgb_paths[idx]), cv2.IMREAD_COLOR)
        swir = cv2.imread(str(swir_paths[idx]), cv2.IMREAD_GRAYSCALE)

        if rgb is None or swir is None:
            print(f"[WARN] Crop réf {idx}: lecture image échouée")
            continue

        valid_mask = valid_mask_for_pair(rgb, swir, H, direction)
        rect = valid_crop_rect(valid_mask, margin=margin)
        rects.append(rect)

        x1, y1, x2, y2 = rect
        print(
            f"[CROP REF {idx:04d}] "
            f"x={x1}:{x2} y={y1}:{y2} size={x2-x1}x{y2-y1}"
        )

    if len(rects) == 0:
        raise RuntimeError("Impossible de calculer le crop global OpenCV.")

    x1_strict = max(r[0] for r in rects)
    y1_strict = max(r[1] for r in rects)
    x2_strict = min(r[2] for r in rects)
    y2_strict = min(r[3] for r in rects)

    if x2_strict > x1_strict and y2_strict > y1_strict:
        crop_rect = (x1_strict, y1_strict, x2_strict, y2_strict)
        crop_mode = "strict"
    else:
        crop_rect = (
            int(np.percentile([r[0] for r in rects], 75)),
            int(np.percentile([r[1] for r in rects], 75)),
            int(np.percentile([r[2] for r in rects], 25)),
            int(np.percentile([r[3] for r in rects], 25)),
        )
        crop_mode = "robust"

    x1, y1, x2, y2 = crop_rect

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(
            f"Crop global invalide : x={x1}:{x2}, y={y1}:{y2}"
        )

    print("\nCROP GLOBAL FINAL")
    print(f"Mode       : {crop_mode}")
    print(f"Crop final : x={x1}:{x2}, y={y1}:{y2}")
    print(f"Taille     : {x2-x1} x {y2-y1}")

    return crop_rect, crop_mode, len(rects)

print("\n" + "=" * 80)
print("SÉPARATION TRAIN / TEST")
print("=" * 80)
print(f"Dataset        : {DATASET}")
print(f"Direction      : {DIRECTION}")
print(f"Images totales : {max_len}")
print(f"Train images   : {len(rgb_est)}")
print(f"Test images    : {len(rgb_apply)}")
print(f"Sortie         : {OUT_BASE}")

for rgb_path in rgb_est:
    shutil.copy2(rgb_path, TRAIN_VIS_DIR / rgb_path.name)

for swir_path in swir_est:
    shutil.copy2(swir_path, TRAIN_SWIR_DIR / swir_path.name)

split_file = OUT_BASE / "train_test_split.txt"

with open(split_file, "w") as f:
    f.write("===== TRAIN / TEST SPLIT =====\n\n")
    f.write(f"Dataset       : {DATASET}\n")
    f.write(f"Direction     : {DIRECTION}\n")
    f.write(f"Total images  : {max_len}\n\n")
    f.write(f"Train start   : {args.train_start}\n")
    f.write(f"Train step    : {args.train_step}\n")
    f.write(f"Train count   : {args.train_count}\n")
    f.write(f"Train images  : {len(rgb_est)}\n")
    f.write("Train indices :\n")
    f.write(",".join(map(str, train_indices)))
    f.write("\n\n")
    f.write(f"Test start    : {args.test_start}\n")
    f.write(f"Test end      : {args.test_end}\n")
    f.write(f"Test images   : {len(rgb_apply)}\n")
    f.write("Test indices  :\n")
    f.write(",".join(map(str, test_indices)))
    f.write("\n")

orb = cv2.ORB_create(nfeatures=N_FEATURES)

all_pts_swir = []
all_pts_rgb = []

global_start = time.time()
h_start = time.time()

print("\n" + "=" * 80)
print("ESTIMATION HOMOGRAPHIE GLOBALE OPENCV")
print("=" * 80)

for i, (rgb_path, swir_path) in enumerate(zip(rgb_est, swir_est)):
    print(f"[Estimation {i+1}/{len(rgb_est)}] {rgb_path.name} ↔ {swir_path.name}")

    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    swir = cv2.imread(str(swir_path), cv2.IMREAD_GRAYSCALE)

    if rgb is None or swir is None:
        print("  Lecture image échouée")
        continue

    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    kp_swir, des_swir = orb.detectAndCompute(swir, None)
    kp_rgb, des_rgb = orb.detectAndCompute(rgb_gray, None)

    if des_swir is None or des_rgb is None:
        print("  Descripteurs manquants")
        continue

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_swir, des_rgb)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 10:
        print(f"  Pas assez de matches : {len(matches)}")
        continue

    pts_swir = np.float32([kp_swir[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts_rgb = np.float32([kp_rgb[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    all_pts_swir.append(pts_swir)
    all_pts_rgb.append(pts_rgb)

if len(all_pts_swir) == 0:
    raise RuntimeError("Aucune correspondance trouvée pour estimer l'homographie.")

all_pts_swir = np.vstack(all_pts_swir).reshape(-1, 1, 2)
all_pts_rgb = np.vstack(all_pts_rgb).reshape(-1, 1, 2)

if DIRECTION == "VIS_TO_SWIR":
    H, mask = cv2.findHomography(
        all_pts_rgb,
        all_pts_swir,
        cv2.RANSAC,
        RANSAC_THRESH
    )
else:
    H, mask = cv2.findHomography(
        all_pts_swir,
        all_pts_rgb,
        cv2.RANSAC,
        RANSAC_THRESH
    )

if H is None:
    raise RuntimeError("Impossible d'estimer l'homographie globale.")

h_end = time.time()
h_time = h_end - h_start
fps_h = len(rgb_est) / h_time if h_time > 0 else 0

print("\nHomographie globale estimée :")
print(H)
print("Inliers:", int(mask.sum()), "/", len(mask))
print(f"Temps estimation homographie : {h_time:.4f} s")

np.save(OUT_BASE / "homography_opencv_global.npy", H)

global_crop_rect, global_crop_mode, global_crop_refs = estimate_global_crop_rect(
    rgb_apply,
    swir_apply,
    H,
    DIRECTION,
    args.crop_refs,
    args.crop_margin
)

crop_file = OUT_BASE / "crop_rect_opencv_global.txt"
x1_global, y1_global, x2_global, y2_global = global_crop_rect

with open(crop_file, "w") as f:
    f.write("===== GLOBAL OPENCV CROP =====\n\n")
    f.write(f"Dataset       : {DATASET}\n")
    f.write(f"Direction     : {DIRECTION}\n")
    f.write(f"Mode          : {global_crop_mode}\n")
    f.write(f"References    : {global_crop_refs}\n")
    f.write(f"Margin        : {args.crop_margin}\n")
    f.write(f"x1            : {x1_global}\n")
    f.write(f"y1            : {y1_global}\n")
    f.write(f"x2            : {x2_global}\n")
    f.write(f"y2            : {y2_global}\n")
    f.write(f"Width         : {x2_global - x1_global}\n")
    f.write(f"Height        : {y2_global - y1_global}\n")

apply_start = time.time()

print("\n" + "=" * 80)
print("APPLICATION HOMOGRAPHIE SUR TEST")
print("=" * 80)

for i, (rgb_path, swir_path) in enumerate(zip(rgb_apply, swir_apply)):
    print(f"[Application {i+1}/{len(rgb_apply)}] {rgb_path.name}")

    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    swir = cv2.imread(str(swir_path), cv2.IMREAD_GRAYSCALE)

    if rgb is None or swir is None:
        print("  Lecture image échouée")
        continue

    original_stem = rgb_path.stem
    base_name = f"opencv_{original_stem}"

    if DIRECTION == "VIS_TO_SWIR":
        h, w = swir.shape[:2]

        vis_aligned = cv2.warpPerspective(
            rgb,
            H,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        x1, y1, x2, y2 = global_crop_rect

        if x2 > w or y2 > h:
            print(f"  Crop global hors limites pour {rgb_path.name}")
            continue

        vis_aligned = vis_aligned[y1:y2, x1:x2]
        swir = swir[y1:y2, x1:x2]

        cv2.imwrite(str(OUT_DIR / f"{base_name}_vis_aligned.jpg"), vis_aligned)
        cv2.imwrite(str(OUT_DIR / f"{base_name}_swir.jpg"), swir)
        
        swir_bgr = cv2.cvtColor(swir, cv2.COLOR_GRAY2BGR)

        overlay = cv2.addWeighted(
            vis_aligned,
            0.5,
            swir_bgr,
            0.5,
            0
        )

    else:
        h, w = rgb.shape[:2]

        swir_aligned = cv2.warpPerspective(
            swir,
            H,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        x1, y1, x2, y2 = global_crop_rect

        if x2 > w or y2 > h:
            print(f"  Crop global hors limites pour {rgb_path.name}")
            continue

        rgb = rgb[y1:y2, x1:x2]
        swir_aligned = swir_aligned[y1:y2, x1:x2]

        cv2.imwrite(str(OUT_DIR / f"{base_name}_vis.jpg"), rgb)
        cv2.imwrite(str(OUT_DIR / f"{base_name}_swir_aligned.jpg"), swir_aligned)

        swir_bgr = cv2.cvtColor(swir_aligned, cv2.COLOR_GRAY2BGR)

        overlay = cv2.addWeighted(
            rgb,
            0.5,
            swir_bgr,
            0.5,
            0
        )

    cv2.imwrite(str(CONCAT_DIR / f"{base_name}_overlay.jpg"), overlay)

apply_end = time.time()
apply_time = apply_end - apply_start
fps_apply = len(rgb_apply) / apply_time if apply_time > 0 else 0

global_end = time.time()
total_time = global_end - global_start
fps_total = len(rgb_apply) / total_time if total_time > 0 else 0

runtime_file = OUT_BASE / "runtime_metrics.txt"

with open(runtime_file, "w") as f:
    f.write("===== RUNTIME METRICS OPENCV =====\n\n")
    f.write(f"Dataset              : {DATASET}\n")
    f.write(f"Direction            : {DIRECTION}\n")
    f.write(f"Train images         : {len(rgb_est)}\n")
    f.write(f"Test images          : {len(rgb_apply)}\n\n")
    f.write(f"Train start          : {args.train_start}\n")
    f.write(f"Train step           : {args.train_step}\n")
    f.write(f"Train count          : {args.train_count}\n")
    f.write(f"Test start           : {args.test_start}\n")
    f.write(f"Test end             : {args.test_end}\n\n")
    f.write(f"ORB features         : {N_FEATURES}\n")
    f.write(f"RANSAC threshold     : {RANSAC_THRESH}\n\n")
    f.write(f"Crop mode            : {global_crop_mode}\n")
    f.write(f"Crop references      : {global_crop_refs}\n")
    f.write(f"Crop margin          : {args.crop_margin}\n")
    f.write(f"Crop rect            : x={x1_global}:{x2_global}, y={y1_global}:{y2_global}\n")
    f.write(f"Crop size            : {x2_global - x1_global} x {y2_global - y1_global}\n\n")
    f.write(f"Temps homographie    : {h_time:.4f} s\n")
    f.write(f"FPS homographie      : {fps_h:.4f}\n\n")
    f.write(f"Temps application    : {apply_time:.4f} s\n")
    f.write(f"FPS application      : {fps_apply:.4f}\n\n")
    f.write(f"Temps total          : {total_time:.4f} s\n")
    f.write(f"FPS global           : {fps_total:.4f}\n")

print("\n" + "=" * 80)
print("OPEN CV TERMINÉ")
print("=" * 80)
print(f"Direction           : {DIRECTION}")
print(f"Sortie split_images : {OUT_DIR}")
print(f"Homographie         : {OUT_BASE / 'homography_opencv_global.npy'}")
print(f"Crop global         : {crop_file}")
print(f"Runtime sauvegardé  : {runtime_file}")
print(f"Split sauvegardé    : {split_file}")
