import os
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional


def apply_undistortion(
    image_tensor: torch.Tensor,
    map1: Optional[torch.Tensor],
    map2: Optional[torch.Tensor]
) -> torch.Tensor:
    """
    Applique une correction de distorsion radiale sur un tensor image
    via cv2.remap. Si map1 ou map2 est None, retourne le tensor inchangé.
    Gère la conversion tensor→numpy→tensor et préserve le device et dtype d'origine.

    Args:
        image_tensor: Tensor PyTorch [C, H, W] float32 dans [0, 1] ou uint8.
        map1:         Première carte de remapping (tensor float ou None).
        map2:         Deuxième carte de remapping (tensor float ou None).

    Returns:
        Tensor PyTorch corrigé, même shape, device et dtype que l'entrée.
    """

    # ===== FIX: Si pas de remap, retourner original =====
    if map1 is None or map2 is None:
        return image_tensor

    original_device = image_tensor.device
    original_dtype = image_tensor.dtype
    was_float = image_tensor.dtype == torch.float32 or image_tensor.dtype == torch.float64
    
    # Conversion tensor → numpy
    img_np = image_tensor.cpu().numpy()
    
    # Gérer format [C, H, W] → [H, W, C] pour OpenCV
    if img_np.ndim == 3:
        img_np = img_np.transpose(1, 2, 0)  # [H, W, C]
    
    # Assurer uint8 pour cv2.remap
    if was_float:
        img_np = (img_np * 255).astype(np.uint8)
    
    # Remap maps sur CPU
    map1_np = map1.cpu().numpy()
    map2_np = map2.cpu().numpy()
    
    # Apply remap
    img_undistorted = cv2.remap(
        img_np, map1_np, map2_np,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    # Reconvertir au format original
    if img_undistorted.ndim == 3:
        img_undistorted = img_undistorted.transpose(2, 0, 1)  # [C, H, W]
    
    # Retour au dtype original
    if was_float:
        img_undistorted = img_undistorted.astype(np.float32) / 255.0
    
    # Retour en tensor sur device original
    result = torch.from_numpy(img_undistorted).to(original_device)
    
    return result.to(original_dtype)




def precompute_remap_maps(calib_path, device='cpu'):
    """
    Charge les paramètres de calibration depuis un fichier .npz et précalcule
    les cartes de remapping (map1, map2) utilisables par cv2.remap.

    Args:
        calib_path: Chemin vers le fichier de calibration .npz
                    (contenant k1, k2, cx, cy, fx, fy, width, height).
        device:     Device PyTorch sur lequel placer les cartes.

    Returns:
        Tuple (map1, map2) en tensors PyTorch sur le device spécifié.
    """    
    calib = np.load(calib_path)
    k1, k2 = calib['k1'], calib['k2']
    cx, cy = calib['cx'], calib['cy']
    w, h = int(calib['width']), int(calib['height'])
    
    print(f"Chargement calibration: {Path(calib_path).name}")
    print(f"  k1={k1:.6f}, k2={k2:.6f}, center=({cx:.1f}, {cy:.1f}), size={w}x{h}")
    
    # ===== FIX: Utiliser focale réaliste =====
    fx = fy = w  # Approximation: focale ≈ largeur
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float32)
    
    # Calculer remap maps (OpenCV)
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist_coeffs, None, K, (w, h), cv2.CV_32FC1
    )
    
    # Convertir en tensors
    map1_tensor = torch.from_numpy(map1).to(device)
    map2_tensor = torch.from_numpy(map2).to(device)
    
    print(f"✓ Remap maps pré-calculées et stockées sur {device}\n")
    
    return map1_tensor, map2_tensor


