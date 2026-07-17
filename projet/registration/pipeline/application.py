import os
import cv2
import numpy as np
import torch
from pathlib import Path
from ..utils.visualization import overlay_50_alpha

def crop_bottom_using_global_ratio(vis_aligned, img_swir, config):
    ratio = getattr(config, "PRE_H_KEEP_RATIO_VIS", None)

    if ratio is None:
        return vis_aligned, img_swir

    h = vis_aligned.shape[0]
    cut_y = int(h * ratio)

    cut_y = max(cut_y, int(h * 0.50))
    cut_y = min(cut_y, h)

    print(f"Post-H global vehicle crop: y=0:{cut_y}/{h}")

    return vis_aligned[:cut_y, :], img_swir[:cut_y, :]

def largest_valid_rectangle(mask):
    """
    Trouve le plus grand rectangle composé uniquement de pixels valides.
    mask: uint8, 1 = valide, 0 = invalide
    """
    h, w = mask.shape
    heights = np.zeros(w, dtype=int)

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

def compute_common_valid_mask(img_vis, img_swir, H, config):
    """Return the geometric overlap after the exact application preprocessing.

    Validity is propagated with binary masks instead of inferred from image
    intensity.  Dark scene pixels are therefore kept, while black pixels
    introduced by undistortion or perspective warping are rejected.
    """

    direction = getattr(config, "ALIGN_DIRECTION", "VIS_TO_SWIR")

    if direction == "VIS_TO_SWIR":
        h, w = img_swir.shape[:2]

        valid_src = np.ones(img_vis.shape[:2], dtype=np.uint8) * 255
        valid_warp = cv2.warpPerspective(
            valid_src, H, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        mask_warp = valid_warp > 0
        mask_ref = np.ones(img_swir.shape[:2], dtype=bool)

    elif direction == "SWIR_TO_VIS":
        h, w = img_vis.shape[:2]

        valid_src = np.ones(img_swir.shape[:2], dtype=np.uint8) * 255
        valid_warp = cv2.warpPerspective(
            valid_src, H, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        mask_warp = valid_warp > 0
        mask_ref = np.ones(img_vis.shape[:2], dtype=bool)

    else:
        raise ValueError(f"ALIGN_DIRECTION invalide : {direction}")

    return (mask_warp & mask_ref).astype(np.uint8)


def compute_global_valid_crop(Homography, rgb_files, swir_files, config, max_frames=30, margin=12):
    undistort_maps = getattr(config, 'UNDISTORT_MAPS', None)

    if undistort_maps is not None:
        map1_vis, map2_vis = undistort_maps['vis']
        map1_swir, map2_swir = undistort_maps['swir']
    else:
        map1_vis = map2_vis = map1_swir = map2_swir = None

    common_mask = None

    for rgb_path, swir_path in list(zip(rgb_files, swir_files))[:max_frames]:
        img_vis = cv2.imread(rgb_path)
        img_swir = cv2.imread(swir_path)

        if img_vis is None or img_swir is None:
            continue

        valid_vis = np.ones(img_vis.shape[:2], dtype=np.uint8) * 255
        valid_swir = np.ones(img_swir.shape[:2], dtype=np.uint8) * 255

        if map1_vis is not None and map2_vis is not None:
            map1_vis_np = map1_vis.cpu().numpy() if hasattr(map1_vis, "cpu") else map1_vis
            map2_vis_np = map2_vis.cpu().numpy() if hasattr(map2_vis, "cpu") else map2_vis
            img_vis = cv2.remap(
                img_vis,
                map1_vis_np,
                map2_vis_np,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            valid_vis = cv2.remap(
                valid_vis, map1_vis_np, map2_vis_np,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )

        if map1_swir is not None and map2_swir is not None:
            map1_swir_np = map1_swir.cpu().numpy() if hasattr(map1_swir, "cpu") else map1_swir
            map2_swir_np = map2_swir.cpu().numpy() if hasattr(map2_swir, "cpu") else map2_swir
            img_swir = cv2.remap(
                img_swir,
                map1_swir_np,
                map2_swir_np,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            valid_swir = cv2.remap(
                valid_swir, map1_swir_np, map2_swir_np,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
        pre_crop_y_vis = getattr(config, "PRE_H_CROP_Y_VIS", None)
        pre_crop_y_swir = getattr(config, "PRE_H_CROP_Y_SWIR", None)

        if pre_crop_y_vis is not None:
            img_vis = img_vis[:pre_crop_y_vis, :]
            valid_vis = valid_vis[:pre_crop_y_vis, :]

        if pre_crop_y_swir is not None:
            img_swir = img_swir[:pre_crop_y_swir, :]
            valid_swir = valid_swir[:pre_crop_y_swir, :]

        direction = getattr(config, "ALIGN_DIRECTION", "VIS_TO_SWIR")
        if direction == "VIS_TO_SWIR":
            h, w = valid_swir.shape
            moving_valid = cv2.warpPerspective(
                valid_vis, Homography, (w, h), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            pair_mask = (moving_valid > 0) & (valid_swir > 0)
        elif direction == "SWIR_TO_VIS":
            h, w = valid_vis.shape
            moving_valid = cv2.warpPerspective(
                valid_swir, Homography, (w, h), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            pair_mask = (moving_valid > 0) & (valid_vis > 0)
        else:
            raise ValueError(f"ALIGN_DIRECTION invalide : {direction}")

        pair_mask = pair_mask.astype(np.uint8)
        common_mask = pair_mask if common_mask is None else (common_mask & pair_mask)

    if common_mask is None or np.count_nonzero(common_mask) == 0:
        raise RuntimeError("Impossible de calculer le crop global automatique")

    if margin > 0:
        kernel_size = 2 * int(margin) + 1
        common_mask = cv2.erode(
            common_mask,
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
            iterations=1,
        )
    if np.count_nonzero(common_mask) == 0:
        raise RuntimeError("El margen eliminó toda la zona común válida")

    x1, y1, x2, y2 = largest_valid_rectangle(common_mask)
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError("No se encontró un rectángulo común válido")

    print(
        f"GLOBAL common valid rectangle: x={x1}:{x2}, y={y1}:{y2} "
        f"size={x2-x1}x{y2-y1}"
    )

    return x1, y1, x2, y2

def remove_bottom_vehicle_reflection(vis_aligned, img_swir, margin=18):
    cut_y, h = detect_vehicle_cut_y_from_array(vis_aligned, margin=margin)

    if cut_y < h:
        print(f"Post-H vehicle crop: y=0:{cut_y}/{h}")
        return vis_aligned[:cut_y, :], img_swir[:cut_y, :]

    return vis_aligned, img_swir

def detect_vehicle_cut_y_from_array(img, margin=18):
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    h, w = gray.shape

    # Chercher uniquement dans la partie basse de l'image.
    y0 = int(h * 0.55)
    y1 = int(h * 0.95)

    # Gradient vertical : détecte les lignes horizontales fortes.
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    row_score = np.mean(np.abs(sobel_y), axis=1)

    # Lisser.
    row_score = cv2.GaussianBlur(row_score.reshape(-1, 1), (1, 31), 0).flatten()

    search = row_score[y0:y1]
    peak = np.argmax(search) + y0

    cut_y = max(peak - margin, int(h * 0.55))

    print(f"Vehicle horizontal boundary detected: y=0:{cut_y}/{h}")

    return cut_y, h
def apply_homography_to_images(Homography, rgb_files, swir_files, config, paths):
    """
    Applique la matrice d'homographie à chaque paire d'images : undistortion
    optionnelle (via remap maps), warping perspectif du VIS vers le repère SWIR,
    puis sauvegarde des images VIS alignées, SWIR et de l'overlay.

    Args:
        Homography:  Matrice d'homographie 3×3.
        rgb_files:   Liste ordonnée de chemins vers les images VIS.
        swir_files:  Liste ordonnée de chemins vers les images SWIR.
        config:      Instance Config (NUM_IMAGES_TO_WARP, UNDISTORT_MAPS, ...).
        paths:       Dictionnaire avec clé 'out_dir'.
    """
    print(f"\n{'='*80}")
    print(f"APPLICATION HOMOGRAPHIE AUX IMAGES")
    print(f"{'='*80}")


        # ===== RÉCUPÉRER REMAP MAPS =====
    undistort_maps = getattr(config, 'UNDISTORT_MAPS', None)
    
    if undistort_maps is not None:
        map1_vis, map2_vis = undistort_maps['vis']
        map1_swir, map2_swir = undistort_maps['swir']
        print("Undistortion ACTIVÉE pour application")
    
    num_to_warp = config.NUM_IMAGES_TO_WARP if config.NUM_IMAGES_TO_WARP else config.NUM_IMAGES
    global_crop = None

    if getattr(config, "AUTO_CROP_COMMON_REGION", True):
        global_crop = compute_global_valid_crop(
            Homography,
            rgb_files,
            swir_files,
            config,
            max_frames=30,
            margin=12
        )
    for i, (rgb_path, swir_path) in enumerate(zip(rgb_files, swir_files)):
        if i >= num_to_warp:
            break
        
        print(f"Traitement image {i+1}/{num_to_warp}...", end=" ")
        
        img_vis = cv2.imread(rgb_path)
        img_swir = cv2.imread(swir_path)
        
        if img_vis is None or img_swir is None:
            print("Erreur lecture")
            continue
        
        if (undistort_maps is not None) and (map1_vis is not None) and (map2_vis is not None):
            # Convertir maps torch → numpy pour cv2
            map1_vis_np = map1_vis.cpu().numpy()
            map2_vis_np = map2_vis.cpu().numpy()

            
            # Undistort
            img_vis = cv2.remap(img_vis, map1_vis_np, map2_vis_np,
                               interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        if (undistort_maps is not None) and (map1_swir is not None) and (map2_swir is not None):
            map1_swir_np = map1_swir.cpu().numpy()
            map2_swir_np = map2_swir.cpu().numpy()
            img_swir = cv2.remap(img_swir, map1_swir_np, map2_swir_np,
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        pre_crop_y_vis = getattr(config, "PRE_H_CROP_Y_VIS", None)
        pre_crop_y_swir = getattr(config, "PRE_H_CROP_Y_SWIR", None)

        if pre_crop_y_vis is not None:
            img_vis = img_vis[:pre_crop_y_vis, :]

        if pre_crop_y_swir is not None:
            img_swir = img_swir[:pre_crop_y_swir, :]


        direction = getattr(config, "ALIGN_DIRECTION", "VIS_TO_SWIR")

        if direction == "VIS_TO_SWIR":
            h, w = img_swir.shape[:2]
            vis_aligned = cv2.warpPerspective(img_vis, Homography, (w, h))
            ref_img = img_swir

        elif direction == "SWIR_TO_VIS":
            h, w = img_vis.shape[:2]
            swir_aligned = cv2.warpPerspective(img_swir, Homography, (w, h))
            ref_img = img_vis

        else:
            raise ValueError(f"ALIGN_DIRECTION invalide : {direction}")

        if direction == "VIS_TO_SWIR":
            out_vis_img = vis_aligned
            out_swir_img = ref_img
        else:
            out_vis_img = ref_img
            out_swir_img = swir_aligned

        if global_crop is not None:
            x1, y1, x2, y2 = global_crop
            out_vis_img = out_vis_img[y1:y2, x1:x2]
            out_swir_img = out_swir_img[y1:y2, x1:x2]

        if getattr(config, "AUTO_REMOVE_BOTTOM_VEHICLE", True):
            out_vis_img, out_swir_img = crop_bottom_using_global_ratio(
                out_vis_img,
                out_swir_img,
                config
            )

        concat = overlay_50_alpha(out_vis_img, out_swir_img)
        concat = cv2.cvtColor(concat, cv2.COLOR_BGR2GRAY)
                
        # Sauvegarder
        base = os.path.splitext(os.path.basename(rgb_path))[0]
        direction = getattr(config, "ALIGN_DIRECTION", "VIS_TO_SWIR")
        if direction == "VIS_TO_SWIR":
            out_vis = os.path.join(
                paths['out_dir'],
                "split_images",
                f"{base}_vis_aligned.jpg"
            )
            out_swir = os.path.join(
                paths['out_dir'],
                "split_images",
                f"{base}_swir.jpg"
            )

        elif direction == "SWIR_TO_VIS":
            out_vis = os.path.join(
                paths['out_dir'],
                "split_images",
                f"{base}_vis.jpg"
            )
            out_swir = os.path.join(
                paths['out_dir'],
                "split_images",
                f"{base}_swir_aligned.jpg"
            )
        out_con = os.path.join(paths['out_dir'], "concatenate_images", f"{base}_con.jpg")
        
        cv2.imwrite(out_vis, out_vis_img)
        cv2.imwrite(out_swir, out_swir_img)
        cv2.imwrite(out_con, concat)
        
        print("✓")
        
        del img_vis, img_swir, out_vis_img, out_swir_img, concat
    
    print(f"\nTraitement terminé")
