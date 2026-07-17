"""
Prétraitement contraste pour images VIS/SWIR
"""

import cv2
import numpy as np
import torch
import kornia


# ============================================================================
# CLAHE CPU (OpenCV) — images numpy BGR
# ============================================================================

def enhance_swir_contrast(img_swir_bgr: np.ndarray) -> np.ndarray:
    """
    Améliore le contraste d'une image SWIR BGR par CLAHE (OpenCV).

    Args:
        img_swir_bgr: Image SWIR BGR uint8.

    Returns:
        Image BGR uint8 avec contraste amélioré.
    """
    gray = cv2.cvtColor(img_swir_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    return cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)


def enhance_vis_contrast(img_vis_bgr: np.ndarray) -> np.ndarray:
    """
    Améliore le contraste d'une image VIS BGR par CLAHE (mêmes paramètres
    que enhance_swir_contrast).

    Args:
        img_vis_bgr: Image VIS BGR uint8.

    Returns:
        Image BGR uint8 avec contraste amélioré.
    """
    gray = cv2.cvtColor(img_vis_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    return cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)


# ============================================================================
# NORMALISATION — images numpy BGR
# ============================================================================

def normalize_zscore_bgr(img: np.ndarray) -> np.ndarray:
    """
    Applique une normalisation Z-score globale sur une image BGR, puis
    rééchelonne sur [0, 255].

    Args:
        img: Image BGR uint8.

    Returns:
        Image normalisée BGR uint8.
    """
    img = img.astype(np.float32)
    mean = img.mean(axis=(0, 1), keepdims=True)
    std  = img.std(axis=(0, 1),  keepdims=True) + 1e-6
    img_n = (img - mean) / std
    img_n = 255 * (img_n - img_n.min()) / (img_n.max() - img_n.min() + 1e-6)
    return img_n.astype(np.uint8)


def to_gray_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convertit une image BGR en niveaux de gris tout en conservant 3 canaux
    (compatible avec les pipelines attendant un format BGR).

    Args:
        img_bgr: Image BGR (H, W, 3).

    Returns:
        Image niveaux de gris sur 3 canaux BGR (H, W, 3).
    """

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ============================================================================
# CLAHE GPU (Kornia) — tensors PyTorch
# ============================================================================

def apply_clahe_kornia(
    tensor: torch.Tensor,
    clip_limit: float = 40.0,
    grid_size: tuple = (6, 6)
) -> torch.Tensor:
    """
    Applique CLAHE sur un tensor image via Kornia (traitement sur GPU).
    Supporte les formats [C, H, W] et [B, C, H, W] avec valeurs dans [0, 1].

    Args:
        tensor:     Tensor PyTorch [C, H, W] ou [B, C, H, W].
        clip_limit: Limite de clipping du CLAHE.
        grid_size:  Taille de la grille tuile (lignes, colonnes).

    Returns:
        Tensor enhanced, même shape et même device que l'entrée.
    """

    was_single = tensor.ndim == 3
    if was_single:
        tensor = tensor.unsqueeze(0)  # [1, C, H, W]

    enhanced = kornia.enhance.equalize_clahe(
        tensor,
        clip_limit=clip_limit,
        grid_size=grid_size
    )
    del tensor

    if was_single:
        enhanced = enhanced.squeeze(0)  # retour [C, H, W]

    return enhanced


def normalize_zscore_per_channel(img: np.ndarray) -> np.ndarray:
    """
    Applique une normalisation Z-score indépendamment sur chaque canal,
    puis rééchelonne chaque canal sur [0, 255].
    Compatible avec les images mono (H, W) et multi-canaux (H, W, C).

    Args:
        img: Image numpy uint8 ou float, shape (H, W) ou (H, W, C).

    Returns:
        Image normalisée, même shape que l'entrée, dtype uint8.

    Raises:
        ValueError: Si le shape de l'image n'est pas 2D ou 3D.
    """
    img = img.astype(np.float32)

    if img.ndim == 2:
        mean = img.mean()
        std = img.std() + 1e-6
        img_n = (img - mean) / std
        img_n = 255.0 * (img_n - img_n.min()) / (img_n.max() - img_n.min() + 1e-6)
        return img_n.astype(np.uint8)

    elif img.ndim == 3:
        channels = []
        for c in range(img.shape[2]):
            channel = img[:, :, c]
            mean = channel.mean()
            std = channel.std() + 1e-6
            channel_n = (channel - mean) / std
            channel_n = 255.0 * (channel_n - channel_n.min()) / (channel_n.max() - channel_n.min() + 1e-6)
            channels.append(channel_n)

        img_n = np.stack(channels, axis=2)
        return img_n.astype(np.uint8)

    else:
        raise ValueError(f"Format d'image non supporté : shape={img.shape}")