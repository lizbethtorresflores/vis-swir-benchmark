"""
Matching IR/VIS avec LightGlue/SuperPoint
"""

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

import cv2
import numpy as np
import torch

from ..data.sequence_matches_container import SequenceMatchesContainer
from ..preprocessing.contrast import enhance_swir_contrast


# ============================================================================
# MATCHING LIGHTGLUE
# ============================================================================

def compute_kpts_and_matches_from_images(
    img_vis_tensor,
    img_swir_tensor,
    extractor,
    matcher,
    device="mps",
    num_points=500,
    filter_threshold=0.1
):
    """
    Extrait les keypoints et les correspondances entre deux tensors images
    via SuperPoint + LightGlue.

    Args:
        img_vis_tensor:  Tensor image VIS [C,H,W], valeurs dans [0,1].
        img_swir_tensor: Tensor image SWIR [C,H,W], valeurs dans [0,1].
        extractor:       Instance SuperPoint (eval mode).
        matcher:         Instance LightGlue (eval mode).
        device:          Device PyTorch ("mps", "cuda", "cpu").
        num_points:      Nombre maximum de keypoints demandés.
        filter_threshold: Seuil de confiance LightGlue (non utilisé directement ici).

    Returns:
        Tuple (pts_vis, pts_swir, matches_idx) sur CPU.
        pts_vis/pts_swir : tensors float (K, 2).
        matches_idx      : ndarray int (K, 2).
        Retourne des tensors/arrays vides si aucun match.
    """
    with torch.no_grad():
        feats_vis  = extractor.extract(img_vis_tensor)
        feats_swir = extractor.extract(img_swir_tensor)

    matches01 = matcher({'image0': feats_vis, 'image1': feats_swir})
    feats_vis, feats_swir, matches01 = [rbd(x) for x in [feats_vis, feats_swir, matches01]]

    kpts_vis_all  = feats_vis['keypoints']   # (N0, 2)
    kpts_swir_all = feats_swir['keypoints']  # (N1, 2)
    matches_idx   = matches01['matches']     # (K, 2)

    if len(matches_idx) == 0:
        return torch.empty(0, 2), torch.empty(0, 2), np.array([]).reshape(0, 2)

    pts_vis  = kpts_vis_all[matches_idx[:, 0]]
    pts_swir = kpts_swir_all[matches_idx[:, 1]]

    return pts_vis.detach().cpu(), pts_swir.detach().cpu(), matches_idx.detach().cpu().numpy()

'''
def compute_kpts_and_matches(path_vis, path_swir, extractor, matcher, device="mps", num_points=500):
    """
    Charge deux images depuis le disque, les normalise en tensors,
    puis délègue à compute_kpts_and_matches_from_images.

    Args:
        path_vis:  Chemin vers l'image VIS.
        path_swir: Chemin vers l'image SWIR.
        extractor: Instance SuperPoint.
        matcher:   Instance LightGlue.
        device:    Device PyTorch.
        num_points: Nombre maximum de keypoints.

    Returns:
        Tuple (pts_vis, pts_swir, matches_idx) — voir compute_kpts_and_matches_from_images.
    """
    from ..io.image_loader import get_normalized_images_tensors
    img_vis_tensor, img_swir_tensor = get_normalized_images_tensors(path_vis, path_swir, device=device)
    return compute_kpts_and_matches_from_images(
        img_vis_tensor, img_swir_tensor,
        extractor, matcher,
        device=device, num_points=num_points
    )
'''

def compute_matches_with_fallback(
    patch_vis_tensor,
    patch_swir_tensor,
    extractor,
    matcher,
    device="mps",
    num_points=500
):
    """
    Tente l'appariement LightGlue avec augmentation progressive du nombre de
    keypoints si le résultat initial est insuffisant (< 20 matches, puis < 10).

    Args:
        patch_vis_tensor:  Tensor patch VIS [C,H,W].
        patch_swir_tensor: Tensor patch SWIR [C,H,W].
        extractor:         Instance SuperPoint.
        matcher:           Instance LightGlue.
        device:            Device PyTorch.
        num_points:        Nombre de keypoints de base.

    Returns:
        Tuple (pts_vis, pts_swir, matches_idx) du meilleur essai.
    """
    kpts_vis, kpts_swir, matches_idx = compute_kpts_and_matches_from_images(
        patch_vis_tensor, patch_swir_tensor,
        extractor, matcher, device=device, num_points=num_points
    )
    if len(matches_idx) >= 20:
        return kpts_vis, kpts_swir, matches_idx

    kpts_vis, kpts_swir, matches_idx = compute_kpts_and_matches_from_images(
        patch_vis_tensor, patch_swir_tensor,
        extractor, matcher, device=device, num_points=int(num_points * 1.2)
    )
    if len(matches_idx) >= 10:
        return kpts_vis, kpts_swir, matches_idx

    return compute_kpts_and_matches_from_images(
        patch_vis_tensor, patch_swir_tensor,
        extractor, matcher, device=device,
        num_points=int(num_points * 2), filter_threshold=0.2
    )


# ============================================================================
# ESTIMATION HOMOGRAPHIE
# ============================================================================

'''
def get_global_homographic_matrix_transform(container: SequenceMatchesContainer):
    """
    Estime une homographie globale depuis tous les matches stockés dans
    container.matches (ancien format) via MAGSAC.

    Args:
        container: SequenceMatchesContainer avec attribut .matches.

    Returns:
        Tuple (H, mask) : matrice 3×3 et masque inliers OpenCV.

    Raises:
        RuntimeError: Si cv2.findHomography ne converge pas.
    """

    src_pts = np.vstack([m['pts_vis']  for m in container.matches]).astype(np.float32).reshape(-1, 1, 2)
    dst_pts = np.vstack([m['pts_swir'] for m in container.matches]).astype(np.float32).reshape(-1, 1, 2)

    print(f"Total matches = {len(src_pts)}")
    H, mask = cv2.findHomography(
        src_pts, dst_pts,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=10,
        confidence=0.99,
        maxIters=5000
    )
    if H is None:
        raise RuntimeError("Homographie non trouvée")
    return H, mask


def get_global_homographic_matrix_transform_patches(container: SequenceMatchesContainer):
    """
    Estime une homographie globale depuis tous les matches stockés dans
    container.pairs (nouveau format) via MAGSAC.

    Args:
        container: SequenceMatchesContainer avec attribut .pairs.

    Returns:
        Tuple (H, mask) : matrice 3×3 et masque inliers OpenCV.

    Raises:
        RuntimeError: Si cv2.findHomography ne converge pas.
    """
    src_pts = np.vstack([m['pts_vis']  for m in container.pairs]).astype(np.float32).reshape(-1, 1, 2)
    dst_pts = np.vstack([m['pts_swir'] for m in container.pairs]).astype(np.float32).reshape(-1, 1, 2)

    print(f"Total matches = {len(src_pts)}")
    H, mask = cv2.findHomography(
        src_pts, dst_pts,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=12,
        confidence=0.9,
        maxIters=5000
    )
    if H is None:
        raise RuntimeError("Homographie non trouvée")
    return H, mask
'''