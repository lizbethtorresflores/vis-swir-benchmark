import os
import cv2
import numpy as np
import torch
from ..matching.lightglue_matcher import compute_kpts_and_matches_from_images
# from ..matching.sift_matcher import compute_sift_matches_permissive
from ..data.sequence_matches_container import SequenceMatchesContainer
from ..io.image_loader import get_normalized_images_tensors
from ..preprocessing.undistortion import apply_undistortion
from lightglue import LightGlue, SuperPoint
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch.nn.functional as F

import gc

def get_segformer_model(config):
    if not hasattr(config, "_segformer_processor"):
        config._segformer_processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640"
        )
        config._segformer_model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640"
        ).to(config.DEVICE).eval()

    return config._segformer_processor, config._segformer_model


def build_not_sky_mask_segformer_from_tensor(tensor, config):
    img = tensor_to_uint8_image(tensor)

    if img.ndim == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img

    processor, model = get_segformer_model(config)

    inputs = processor(images=img_rgb, return_tensors="pt")
    inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    h, w = img_rgb.shape[:2]
    logits = F.interpolate(
        logits,
        size=(h, w),
        mode="bilinear",
        align_corners=False
    )

    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    sky_prob = probs[2]  # ADE20K: sky = clase 2
    sky = sky_prob > 0.85

    top = np.zeros_like(sky, dtype=bool)
    top[: int(h * 0.50), :] = True
    sky = sky & top

    return sky == 0

def tensor_to_uint8_image(t):
    img = t.detach().cpu().numpy()

    if img.ndim == 3:
        img = img.transpose(1, 2, 0)
    elif img.ndim == 4:
        img = img[0].transpose(1, 2, 0)

    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def build_not_sky_mask_hsv_from_tensor(tensor):
    img = tensor_to_uint8_image(tensor)

    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        gray = img.squeeze()
        h, w = gray.shape
        edges = cv2.Canny(gray, 50, 150)
        row_density = (edges > 0).mean(axis=1)

        sky = np.zeros((h, w), dtype=bool)
        for y in range(0, int(h * 0.55)):
            if row_density[y] < 0.01:
                sky[y, :] = True
    else:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        h, w = S.shape

        sky = (S < 70) & (V > 120)

        top = np.zeros_like(sky, dtype=bool)
        top[: int(h * 0.55), :] = True
        sky = sky & top

    sky_u8 = sky.astype(np.uint8) * 255
    kernel = np.ones((15, 15), np.uint8)
    sky_u8 = cv2.morphologyEx(sky_u8, cv2.MORPH_OPEN, kernel)
    sky_u8 = cv2.morphologyEx(sky_u8, cv2.MORPH_CLOSE, kernel)

    not_sky = sky_u8 == 0
    return not_sky

def filter_sky_matches(pts_vis, pts_swir, matches_all, vis_tensor, swir_tensor, config):
    vis_not_sky = build_not_sky_mask_segformer_from_tensor(vis_tensor, config)

    swir_not_sky = np.ones(swir_tensor.shape[-2:], dtype=bool)

    h_vis, w_vis = vis_not_sky.shape
    h_swir, w_swir = swir_not_sky.shape

    xv = np.clip(np.round(pts_vis[:, 0]).astype(int), 0, w_vis - 1)
    yv = np.clip(np.round(pts_vis[:, 1]).astype(int), 0, h_vis - 1)

    xs = np.clip(np.round(pts_swir[:, 0]).astype(int), 0, w_swir - 1)
    ys = np.clip(np.round(pts_swir[:, 1]).astype(int), 0, h_swir - 1)

    keep = vis_not_sky[yv, xv] & swir_not_sky[ys, xs]

    print(f"SegFormer sky filtering: {keep.sum()}/{len(keep)} matches kept")

    return pts_vis[keep], pts_swir[keep], matches_all[keep]

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

    threshold = search.mean() + 1.0 * search.std()
    candidates = np.where(search > threshold)[0]

    if len(candidates) > 0:
        # Prendre le premier bord fort du bas vers le haut.
        peak = candidates[-1] + y0
    else:
        peak = np.argmax(search) + y0

    cut_y = max(peak - margin, int(h * 0.55))

    print(f"Vehicle horizontal boundary detected: y=0:{cut_y}/{h}")

    return cut_y, h

def detect_vehicle_cut_y_from_vis_tensor(vis_tensor, margin=18):
    img = vis_tensor.detach().cpu().numpy()

    if img.ndim == 3:
        img_np = img.transpose(1, 2, 0)
    else:
        img_np = img[0].transpose(1, 2, 0)

    return detect_vehicle_cut_y_from_array(img_np, margin=margin)


def estimate_global_vehicle_crop(rgb_train, swir_train, config, max_samples=20):
    cut_vis_list = []
    cut_swir_list = []
    vis_h_list = []
    swir_h_list = []

    n = min(len(rgb_train), max_samples)

    print("\nESTIMATION GLOBAL VEHICLE CROP FROM TRAIN")

    for rgb_path, swir_path in zip(rgb_train[:n], swir_train[:n]):
        vis_tensor, swir_tensor = get_normalized_images_tensors(
            rgb_path, swir_path, config, device=config.DEVICE
        )

        cut_y_vis, vis_h = detect_vehicle_cut_y_from_vis_tensor(
            vis_tensor,
            margin=getattr(config, "VEHICLE_MARGIN", 18)
        )

        cut_y_swir, swir_h = detect_vehicle_cut_y_from_vis_tensor(
            swir_tensor,
            margin=getattr(config, "VEHICLE_MARGIN", 18)
        )

        cut_vis_list.append(cut_y_vis)
        cut_swir_list.append(cut_y_swir)

        vis_h_list.append(vis_h)
        swir_h_list.append(swir_h)

        print(
            f"{os.path.basename(rgb_path)} -> "
            f"VIS cut_y={cut_y_vis}/{vis_h} | "
            f"SWIR cut_y={cut_y_swir}/{swir_h}"
        )

    vis_h_ref = int(np.median(vis_h_list))
    swir_h_ref = int(np.median(swir_h_list))

    global_cut_vis = int(np.median(cut_vis_list))
    global_cut_swir = int(np.median(cut_swir_list))

    extra_margin = getattr(config, "VEHICLE_EXTRA_MARGIN_PX", 120)
    min_keep_ratio = getattr(config, "VEHICLE_MIN_KEEP_RATIO", 0.55)

    global_cut_vis -= extra_margin
    global_cut_swir -= extra_margin

    global_cut_vis = max(global_cut_vis, int(vis_h_ref * min_keep_ratio))
    global_cut_swir = max(global_cut_swir, int(swir_h_ref * min_keep_ratio))

    config.PRE_H_KEEP_RATIO_VIS = global_cut_vis / vis_h_ref
    config.PRE_H_KEEP_RATIO_SWIR = global_cut_swir / swir_h_ref

    print(f"GLOBAL VIS crop : y=0:{global_cut_vis}/{vis_h_ref}")
    print(f"GLOBAL SWIR crop: y=0:{global_cut_swir}/{swir_h_ref}")

    return global_cut_vis, global_cut_swir

def balance_matches_by_region(pts_vis, pts_swir, matches_all, image_width, config):
    """
    Équilibrage 2D des correspondances par cellules spatiales.
    Divise l'image en lignes et colonnes afin d'éviter qu'une zone domine.
    """

    num_rows = getattr(config, "BALANCE_NUM_REGIONS_Y", 2)
    num_cols = getattr(config, "BALANCE_NUM_REGIONS_X", 3)
    max_per_cell = getattr(config, "BALANCE_MAX_MATCHES_PER_REGION", 120)
    min_per_cell = getattr(config, "BALANCE_MIN_MATCHES_PER_REGION", 20)

    x = pts_vis[:, 0]
    y = pts_vis[:, 1]

    image_height = int(np.max(y)) + 1

    x_bins = np.linspace(0, image_width, num_cols + 1)
    y_bins = np.linspace(0, image_height, num_rows + 1)

    keep_indices = []

    print("\nBALANCE MATCHES BY 2D REGION")
    np.random.seed(42)

    for r in range(num_rows):
        for c in range(num_cols):
            x0, x1 = x_bins[c], x_bins[c + 1]
            y0, y1 = y_bins[r], y_bins[r + 1]

            cell_idx = np.where(
                (x >= x0) & (x < x1) &
                (y >= y0) & (y < y1)
            )[0]

            print(
                f"Cell row={r}, col={c}: "
                f"x={x0:.0f}:{x1:.0f}, y={y0:.0f}:{y1:.0f} | "
                f"matches={len(cell_idx)}"
            )

            if len(cell_idx) < min_per_cell:
                keep_indices.extend(cell_idx.tolist())
                continue

            if len(cell_idx) > max_per_cell:
                selected = np.random.choice(
                    cell_idx,
                    size=max_per_cell,
                    replace=False
                )
                keep_indices.extend(selected.tolist())
            else:
                keep_indices.extend(cell_idx.tolist())

    keep_indices = np.array(sorted(keep_indices), dtype=int)

    print(f"Balanced 2D matches kept: {len(keep_indices)}/{len(matches_all)}")

    return (
        pts_vis[keep_indices],
        pts_swir[keep_indices],
        matches_all[keep_indices]
    )

def process_all_images(rgb_files, swir_files, config):
    """
    Itère sur toutes les paires d'images, applique l'undistortion si activée,
    rogne le VIS si nécessaire, puis extrait les correspondances via la grille
    chevauchante. Accumule les résultats dans un SequenceMatchesContainer.

    Args:
        rgb_files:  Liste de chemins vers les images VIS.
        swir_files: Liste de chemins vers les images SWIR.
        config:     Instance Config (NUM_IMAGES, DEVICE, CROP_BOTTOM_PIXEL_VIS, ...).

    Returns:
        SequenceMatchesContainer contenant toutes les paires de correspondances.
    """
    container = SequenceMatchesContainer()
    
    print(f"\n{'='*80}")
    print(f"EXTRACTION MATCHES AVEC BOOST COINS/BORDS")
    print(f"{'='*80}")
    print(f"Images à traiter : {config.NUM_IMAGES}")
    print(f"Points base : {config.NUM_POINT_LG}")
    print(f"Coins : {config.NUM_POINT_LG * config.CORNER_MULTIPLIER} pts | Bords : {config.NUM_POINT_LG * config.EDGE_MULTIPLIER} pts")



    undistort_maps = getattr(config, 'UNDISTORT_MAPS', None)

    if undistort_maps is not None:
        map1_vis, map2_vis = undistort_maps['vis']
        map1_swir, map2_swir = undistort_maps['swir']
        print("Undistortion ACTIVÉE")
    else:
        print("Undistortion DÉSACTIVÉE")

    #extractor = SuperPoint(max_num_keypoints=config.NUM_POINT_LG).eval().to(config.DEVICE)
    #matcher = LightGlue(features='superpoint').eval().to(config.DEVICE)
    
    for i, (rgb_path, swir_path) in enumerate(zip(rgb_files, swir_files)):
        if i >= config.NUM_IMAGES:
            break
        
        print(f"\nImage {i+1}/{config.NUM_IMAGES}")
    
        # Charger images
        vis_tensor, swir_tensor = get_normalized_images_tensors(
            rgb_path, swir_path, config, device=config.DEVICE
        )

        # Keep estimation and final application in the same coordinate system:
        # undistort first, then apply the shared pre-H vehicle cut.
        if undistort_maps is not None:
            vis_tensor = apply_undistortion(vis_tensor, map1_vis, map2_vis)
            swir_tensor = apply_undistortion(swir_tensor, map1_swir, map2_swir)
            print(" Images undistorted")
        
        pre_crop_y_vis = getattr(config, "PRE_H_CROP_Y_VIS", None)
        pre_crop_y_swir = getattr(config, "PRE_H_CROP_Y_SWIR", None)

        
        if pre_crop_y_vis is not None and pre_crop_y_swir is not None:
            if vis_tensor.ndim == 3:
                vis_tensor = vis_tensor[:, :pre_crop_y_vis, :]
            else:
                vis_tensor = vis_tensor[:, :, :pre_crop_y_vis, :]

            if swir_tensor.ndim == 3:
                swir_tensor = swir_tensor[:, :pre_crop_y_swir, :]
            else:
                swir_tensor = swir_tensor[:, :, :pre_crop_y_swir, :]

            print(f"Global pre-H crop applied VIS y=0:{pre_crop_y_vis}, SWIR y=0:{pre_crop_y_swir}")
        
        def save_pre_h_crop_debug(vis_tensor, swir_tensor, rgb_path, swir_path, config):
            base_dir = getattr(config, "PRE_H_DEBUG_DIR", "debug_pre_h_crop")

            vis_dir = os.path.join(base_dir, "VISIBLE")
            swir_dir = os.path.join(base_dir, "SWIR")

            os.makedirs(vis_dir, exist_ok=True)
            os.makedirs(swir_dir, exist_ok=True)

            def tensor_to_img(t):
                img = t.detach().cpu().numpy()
                if img.ndim == 3:
                    img = img.transpose(1, 2, 0)
                elif img.ndim == 4:
                    img = img[0].transpose(1, 2, 0)

                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)

                if img.ndim == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                return img

            cv2.imwrite(os.path.join(vis_dir, os.path.basename(rgb_path)), tensor_to_img(vis_tensor))
            cv2.imwrite(os.path.join(swir_dir, os.path.basename(swir_path)), tensor_to_img(swir_tensor))

        if getattr(config, "AUTO_PRE_CROP_BOTTOM", False):
            crop_y_vis = estimate_bottom_crop_from_image(vis_tensor)

            if vis_tensor.ndim == 3:
                vis_tensor = vis_tensor[:, :crop_y_vis, :]
            elif vis_tensor.ndim == 4:
                vis_tensor = vis_tensor[:, :, :crop_y_vis, :]

            if swir_tensor.ndim == 3:
                swir_tensor = swir_tensor[:, :crop_y_vis, :]
            elif swir_tensor.ndim == 4:
                swir_tensor = swir_tensor[:, :, :crop_y_vis, :]

            print(f"Auto pre-crop bottom: keep y=0:{crop_y_vis}")

        if getattr(config, "SAVE_PRE_H_CROP_DEBUG", True):
            save_pre_h_crop_debug(
                vis_tensor,
                swir_tensor,
                rgb_path,
                swir_path,
                config
            )

        # VIS: [C, H, W] ou [B, C, H, W]
        if (config.CROP_BOTTOM_PIXEL_VIS > 0) and vis_tensor.ndim in [3, 4]:
            vis_tensor = vis_tensor[:, :-(config.CROP_BOTTOM_PIXEL_VIS), :]  # [C, H-100, W]
 


        print(f"VIS original : {vis_tensor.shape}")
        print(f"VIS cropped  : {vis_tensor.shape}")        
        print(f"Shape SWIR : {swir_tensor.shape}")
        
        # ═══ EXTRACTION AVEC BOOST COINS ═══
        vis_kpts, swir_kpts, matches_all = extract_matches_with_overlapping_grid_batched(
                                            vis_tensor, swir_tensor,
                                            device=config.DEVICE, config=config,  
                                           )
        if getattr(config, "FILTER_SKY_MATCHES_PRE_H", True):
            vis_kpts, swir_kpts, matches_all = filter_sky_matches(
                vis_kpts,
                swir_kpts,
                matches_all,
                vis_tensor,
                swir_tensor,
                config
            )
        if getattr(config, "BALANCE_MATCHES_BY_REGION", True):
            vis_width = vis_tensor.shape[-1]

            vis_kpts, swir_kpts, matches_all = balance_matches_by_region(
                vis_kpts,
                swir_kpts,
                matches_all,
                vis_width,
                config
            )
        """
        vis_kpts, swir_kpts, matches_all = extract_matches_with_overlapping_grid_batched(
                                            vis_tensor, swir_tensor,
                                            device=config.DEVICE, config=config,  
                                            batch_size=config.BATCH_SIZE
                                           )
        """
        
        if len(matches_all) > 0:
            container.add_pair(
                matches_idx=matches_all,
                pts_vis=vis_kpts,
                pts_swir=swir_kpts
            )  
        print(f"{len(matches_all)} matches stockés")

        del vis_tensor, swir_tensor, vis_kpts, swir_kpts, matches_all

        if config.DEVICE == "mps":
            torch.mps.empty_cache()
        elif config.DEVICE == "cuda":
            torch.cuda.empty_cache()
    
    gc.collect()
    if config.DEVICE == "mps":
        torch.mps.empty_cache()
    elif config.DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    return container



def build_overlapping_cells(rgb_image, swir_image, config):
    """
    Découpe les images VIS et SWIR en cellules chevauchantes selon la grille
    définie dans config. Calcule les coordonnées SWIR proportionnellement
    à celles du VIS. Catégorise chaque cellule en CORNER, EDGE ou CENTER.

    Args:
        rgb_image:  Tensor VIS [C, H, W].
        swir_image: Tensor SWIR [C, H, W].
        config:     Instance Config (GRID_ROWS, GRID_COLS, GRID_OVERLAP_PERCENT,
                    MIN_CELL_SIZE).

    Returns:
        Liste de dictionnaires avec les clés :
        vis_patch, swir_patch, vis_origin, swir_origin, zone, grid_pos.
    """
    vis_h, vis_w   = rgb_image.shape[-2:]
    swir_h, swir_w = swir_image.shape[-2:]

    grid_rows   = config.GRID_ROWS
    grid_cols   = config.GRID_COLS
    overlap_pct = config.GRID_OVERLAP_PERCENT / 100.0

    vis_cell_h  = int(vis_h  / (grid_rows * (1.0 - overlap_pct)))
    vis_cell_w  = int(vis_w  / (grid_cols * (1.0 - overlap_pct)))
    swir_cell_h = int(swir_h / (grid_rows * (1.0 - overlap_pct)))
    swir_cell_w = int(swir_w / (grid_cols * (1.0 - overlap_pct)))

    vis_step_h  = int(vis_cell_h  * (1.0 - overlap_pct))
    vis_step_w  = int(vis_cell_w  * (1.0 - overlap_pct))
    swir_step_h = int(swir_cell_h * (1.0 - overlap_pct))
    swir_step_w = int(swir_cell_w * (1.0 - overlap_pct))

    print(f"Grid: {grid_rows}x{grid_cols} regions, overlap={config.GRID_OVERLAP_PERCENT}%")

    cell_list = []

    vis_y   = 0
    row_idx = 0
    while vis_y < vis_h:
        vis_x   = 0
        col_idx = 0

        while vis_x < vis_w:
            # type de zone
            is_corner = ((row_idx == 0 or vis_y + vis_cell_h >= vis_h) and
                         (col_idx == 0 or vis_x + vis_cell_w >= vis_w))
            is_edge   = ((row_idx == 0 or vis_y + vis_cell_h >= vis_h or
                          col_idx == 0 or vis_x + vis_cell_w >= vis_w) and not is_corner)
            zone      = "CORNER" if is_corner else "EDGE" if is_edge else "CENTER"

            vis_y_end = min(vis_y + vis_cell_h, vis_h)
            vis_x_end = min(vis_x + vis_cell_w, vis_w)

            swir_y_start = int(vis_y     * swir_h / vis_h)
            swir_y_end   = int(vis_y_end * swir_h / vis_h)
            swir_x_start = int(vis_x     * swir_w / vis_w)
            swir_x_end   = int(vis_x_end * swir_w / vis_w)

            vis_patch  = rgb_image[:, vis_y:vis_y_end,       vis_x:vis_x_end]
            swir_patch = swir_image[:, swir_y_start:swir_y_end, swir_x_start:swir_x_end]

            if (vis_patch.numel() > 0 and swir_patch.numel() > 0 and
                vis_patch.shape[-2] >= config.MIN_CELL_SIZE and
                vis_patch.shape[-1] >= config.MIN_CELL_SIZE and
                swir_patch.shape[-2] >= config.MIN_CELL_SIZE and
                swir_patch.shape[-1] >= config.MIN_CELL_SIZE):

                cell_list.append({
                    "vis_patch":   vis_patch,
                    "swir_patch":  swir_patch,
                    "vis_origin":  (vis_x,       vis_y),
                    "swir_origin": (swir_x_start, swir_y_start),
                    "zone":        zone,
                    "grid_pos":    (row_idx, col_idx),
                })

            vis_x   += vis_step_w
            col_idx += 1

        vis_y   += vis_step_h
        row_idx += 1

    print(f"Valid cells: {len(cell_list)}")
    return cell_list



def match_cell_lightglue(cell, extractor, matcher, device, config):
    """
    Exécute SuperPoint + LightGlue sur un patch VIS/SWIR issu de la grille,
    puis reprojecte les keypoints détectés en coordonnées globales.

    Args:
        cell:      Dictionnaire de cellule issu de build_overlapping_cells.
        extractor: Instance SuperPoint (eval mode).
        matcher:   Instance LightGlue (eval mode).
        device:    Device PyTorch.
        config:    Instance Config.

    Returns:
        Tuple (pts_vis_global, pts_swir_global, matches_idx) en numpy float32/int.
        Retourne (None, None, None) si aucun match.
    """
    vis_patch  = cell["vis_patch"].to(device).float()
    swir_patch = cell["swir_patch"].to(device).float()

    # Prépare grayscale si besoin plus tard
    vis_gray  = _to_grayscale(vis_patch.unsqueeze(0))
    swir_gray = _to_grayscale(swir_patch.unsqueeze(0))

    with torch.no_grad():
        # On garde EXACTEMENT ta logique actuelle : VIS en RGB, SWIR en gris
        feats_vis  = extractor.extract(vis_patch)
        feats_swir = extractor.extract(swir_gray)
        matches    = matcher({"image0": feats_vis, "image1": feats_swir})

    matches_idx = matches["matches"][0]
    if len(matches_idx) == 0:
        return None, None, None

    kpts_vis  = feats_vis["keypoints"][0]
    kpts_swir = feats_swir["keypoints"][0]

    pts_vis  = kpts_vis[matches_idx[:, 0].long()].cpu().numpy().astype(np.float32)
    pts_swir = kpts_swir[matches_idx[:, 1].long()].cpu().numpy().astype(np.float32)

    # Reprojection en coordonnées globales
    vx, vy = cell["vis_origin"]
    sx, sy = cell["swir_origin"]

    pts_vis[:, 0]  += vx
    pts_vis[:, 1]  += vy
    pts_swir[:, 0] += sx
    pts_swir[:, 1] += sy

    return pts_vis, pts_swir, matches_idx


def accumulate_global_matches(
    pts_vis, pts_swir, matches_idx,
    vis_kpts_list, swir_kpts_list, matches_list,
    vis_offset, swir_offset
):
    """
    Ajoute les points et indices de matches d'une cellule dans les listes
    globales en réindexant les indices pour tenir compte des offsets accumulés.

    Args:
        pts_vis:        Points VIS de la cellule (K, 2) float32.
        pts_swir:       Points SWIR de la cellule (K, 2) float32.
        matches_idx:    Indices de matches locaux (K, 2) tensor.
        vis_kpts_list:  Liste cumulée des points VIS.
        swir_kpts_list: Liste cumulée des points SWIR.
        matches_list:   Liste cumulée des indices de matches.
        vis_offset:     Offset courant pour les indices VIS.
        swir_offset:    Offset courant pour les indices SWIR.

    Returns:
        Tuple (vis_offset, swir_offset) mis à jour.
    """
    vis_kpts_list.append(pts_vis)
    swir_kpts_list.append(pts_swir)

    matches_np        = matches_idx.cpu().numpy().astype(np.int32).copy()
    matches_np[:, 0] += vis_offset
    matches_np[:, 1] += swir_offset
    matches_list.append(matches_np)

    vis_offset  += len(pts_vis)
    swir_offset += len(pts_swir)

    return vis_offset, swir_offset


def estimate_bottom_crop_from_image(tensor, min_keep_ratio=0.65, smooth=21):
    """
    Estime automatiquement où commence une zone inférieure problématique.
    Retourne crop_y : conserver l'image de y=0 jusqu'à y=crop_y.
    """

    # Tensor [C,H,W] -> numpy gray
    img = tensor.detach().cpu().numpy()

    if img.ndim == 3:
        img = img.transpose(1, 2, 0)

    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.squeeze()

    h, w = gray.shape[:2]

    # Bords détectés par Canny.
    edges = cv2.Canny(gray, 50, 150)

    # Densité de bords par ligne.
    row_score = edges.mean(axis=1).astype(np.float32)

    # Lissage.
    kernel = np.ones(smooth, dtype=np.float32) / smooth
    row_score_smooth = np.convolve(row_score, kernel, mode="same")

    # Chercher uniquement dans la partie inférieure.
    start_y = int(h * min_keep_ratio)
    bottom_scores = row_score_smooth[start_y:]

    if len(bottom_scores) == 0:
        return h

    # Umbral adaptativo
    threshold = bottom_scores.mean() + 0.5 * bottom_scores.std()

    candidates = np.where(bottom_scores > threshold)[0]

    if len(candidates) == 0:
        return h

    crop_y = start_y + candidates[0]

    # Sécurité : ne pas trop recadrer.
    min_crop_y = int(h * min_keep_ratio)
    crop_y = max(crop_y, min_crop_y)

    return crop_y

def extract_matches_with_overlapping_grid_batched(rgb_image, swir_image, device, config):
    """
    Orchestre l'extraction complète des correspondances sur une paire d'images
    via la grille chevauchante : construit les cellules, appelle
    match_cell_lightglue sur chacune, accumule et concatène les résultats.
    Libère le cache GPU après traitement.

    Args:
        rgb_image:  Tensor VIS [C, H, W].
        swir_image: Tensor SWIR [C, H, W].
        device:     Device PyTorch ("mps", "cuda", "cpu").
        config:     Instance Config (extractor, matcher en cache).

    Returns:
        Tuple (vis_kpts, swir_kpts, matches_all) en numpy arrays float32/int32.
        Chaque tableau est de forme (N, 2).
    """

    # 1. Prépare modèles (cache dans Config)
    extractor = config.extractor
    matcher   = config.matcher

    # 2. Construit les cellules
    cell_list = build_overlapping_cells(rgb_image, swir_image, config)

    vis_kpts_list  = []
    swir_kpts_list = []
    matches_list   = []
    vis_offset     = 0
    swir_offset    = 0
    success_count  = 0
    fail_count     = 0

    for cell in cell_list:
        try:
            pts_vis, pts_swir, matches_idx = match_cell_lightglue(
                cell, extractor, matcher, device, config
            )
            if pts_vis is None:
                fail_count += 1
                continue

            vis_offset, swir_offset = accumulate_global_matches(
                pts_vis, pts_swir, matches_idx,
                vis_kpts_list, swir_kpts_list, matches_list,
                vis_offset, swir_offset
            )
            success_count += 1

        except Exception as e:
            print(f"Cell {cell['grid_pos']}: {type(e).__name__}")
            fail_count += 1
            continue

    print(f"Cells: {success_count} OK, {fail_count} failed")

    if len(vis_kpts_list) == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.int32),
        )

    vis_kpts    = np.vstack(vis_kpts_list).astype(np.float32)
    swir_kpts   = np.vstack(swir_kpts_list).astype(np.float32)
    matches_all = np.vstack(matches_list).astype(np.int32)

    print(f"Total matches: {len(matches_all)}")

    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    return vis_kpts, swir_kpts, matches_all


def _to_grayscale(tensor):
    """
    Convertit un tensor RGB (B, 3, H, W) en niveaux de gris (B, 1, H, W)
    via la pondération ITU-R BT.601. Retourne le tensor inchangé s'il
    n'a pas 3 canaux.

    Args:
        tensor: Tensor PyTorch (B, C, H, W).

    Returns:
        Tensor PyTorch (B, 1, H, W) ou tensor d'entrée si C != 3.
    """    
    if tensor.shape[1] == 3:
        return (0.299 * tensor[:, 0:1] + 
                0.587 * tensor[:, 1:2] + 
                0.114 * tensor[:, 2:3])
    return tensor
