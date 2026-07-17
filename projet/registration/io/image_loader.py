import os
import glob
import cv2
import numpy as np
import torch
from pathlib import Path
from ..preprocessing.contrast import enhance_swir_contrast, normalize_zscore_bgr, to_gray_bgr, normalize_zscore_per_channel



def load_image_paths(paths):
    """
    Charge listes chemins images RGB et SWIR
    
    Args:
        paths: Dict avec 'rgb_dir' et 'swir_dir'
    
    Returns:
        (rgb_files, swir_files) triés
    """
    rgb_files = sorted([
        os.path.join(paths['rgb_dir'], f)
        for f in os.listdir(paths['rgb_dir'])
        if f.lower().endswith(('.tiff', '.tif', '.jpg', '.jpeg', '.png'))
    ])

    swir_files = sorted([
        os.path.join(paths['swir_dir'], f)
        for f in os.listdir(paths['swir_dir'])
        if f.lower().endswith(('.tiff', '.tif', '.jpg', '.jpeg', '.png'))
    ])
    
    return rgb_files, swir_files


def get_normalized_images_tensors(path_vis, path_swir, config, device="mps"):
    """
    Charge et normalise paire d'images VIS/SWIR en tensors PyTorch
    
    Args:
        path_vis: Chemin image VIS (RGB)
        path_swir: Chemin image SWIR
        device: Device PyTorch ("mps", "cuda", "cpu")
    
    Returns:
        (img_vis_tensor, img_swir_tensor) normalisés, shape [1, 3, H, W]
    """
    # Charger en BGR avec OpenCV
    img_vis_bgr = cv2.imread(path_vis)
    img_swir_bgr = cv2.imread(path_swir)
    img_swir_bgr = enhance_swir_contrast(img_swir_bgr)
    
    if img_vis_bgr is None or img_swir_bgr is None:
        raise ValueError(f"Impossible de charger : {path_vis} ou {path_swir}")
    

    # Normalisation (VIS en grayscale, SWIR en BGR normalisé)
    #img_vis_norm = normalize_zscore_bgr(to_gray_bgr(img_vis_bgr))
    #img_swir_norm = normalize_zscore_bgr(img_swir_bgr)

    img_vis_norm = normalize_zscore_per_channel(img_vis_bgr)
    img_swir_norm = normalize_zscore_per_channel(img_swir_bgr)

    
    if img_vis_norm.ndim == 2:
        img_vis_norm = cv2.cvtColor(img_vis_norm, cv2.COLOR_GRAY2BGR)
    if img_swir_norm.ndim == 2:
        img_swir_norm = cv2.cvtColor(img_swir_norm, cv2.COLOR_GRAY2BGR)
    
    img_vis_tensor = torch.from_numpy(img_vis_norm).float().permute(2, 0, 1) / 255.0
    img_vis_tensor = img_vis_tensor.to(device)
    
    img_swir_tensor = torch.from_numpy(img_swir_norm).float().permute(2, 0, 1) / 255.0
    img_swir_tensor = img_swir_tensor.to(device)
    
    return img_vis_tensor, img_swir_tensor

