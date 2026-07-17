import cv2
import numpy as np
import torch    

'''
def compute_sift_matches_permissive(vis_patch, swir_patch, num_points, config):
    """
    Apparie deux patches VIS/SWIR par SIFT avec des paramètres permissifs,
    conçus pour les zones difficiles (coins, bords). Accepte tensors PyTorch
    ou arrays numpy en entrée.

    Args:
        vis_patch:  Patch VIS (tensor [C,H,W] ou ndarray).
        swir_patch: Patch SWIR (tensor [C,H,W] ou ndarray).
        num_points: Nombre de features cibles pour SIFT.
        config:     Instance Config portant les hyperparamètres SIFT
                    (SIFT_N_FEATURES_MULT, SIFT_CONTRAST_THRESHOLD,
                     SIFT_EDGE_THRESHOLD, SIFT_SIGMA, SIFT_LOWE_RATIO).

    Returns:
        Tuple (pts_vis, pts_swir, matches_idx) en tensors PyTorch float/long.
        Retourne des tensors vides si moins de 4 points sont détectés.
    """
    # Convertir tensors en numpy
    if isinstance(vis_patch, torch.Tensor):
        vis_np = (vis_patch.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        swir_np = (swir_patch.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    else:
        vis_np = vis_patch
        swir_np = swir_patch
    
    # Niveaux de gris
    if len(vis_np.shape) == 3:
        vis_gray = cv2.cvtColor(vis_np, cv2.COLOR_RGB2GRAY)
        swir_gray = cv2.cvtColor(swir_np, cv2.COLOR_RGB2GRAY)
    else:
        vis_gray = vis_np
        swir_gray = swir_np
    
    # SIFT permissif
    sift = cv2.SIFT_create(
        nfeatures=num_points * config.SIFT_N_FEATURES_MULT,
        contrastThreshold=config.SIFT_CONTRAST_THRESHOLD,
        edgeThreshold=config.SIFT_EDGE_THRESHOLD,
        sigma=config.SIFT_SIGMA
    )
    
    kp_vis, desc_vis = sift.detectAndCompute(vis_gray, None)
    kp_swir, desc_swir = sift.detectAndCompute(swir_gray, None)
    
    if desc_vis is None or desc_swir is None or len(kp_vis) < 4 or len(kp_swir) < 4:
        return torch.empty(0, 2), torch.empty(0, 2), torch.empty(0, 2, dtype=torch.long)
    
    # Matching
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(desc_vis, desc_swir, k=2)
    
    # Lowe's ratio permissif
    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < config.SIFT_LOWE_RATIO * n.distance:
                good.append(m)
        elif len(pair) == 1:
            good.append(pair[0])
    
    if len(good) < 4:
        return torch.empty(0, 2), torch.empty(0, 2), torch.empty(0, 2, dtype=torch.long)
    
    # Extraire coordonnées
    pts_vis = np.float32([kp_vis[m.queryIdx].pt for m in good])
    pts_swir = np.float32([kp_swir[m.trainIdx].pt for m in good])
    matches_idx = np.column_stack([np.arange(len(good)), np.arange(len(good))])
    
    return (
        torch.from_numpy(pts_vis).float(),
        torch.from_numpy(pts_swir).float(),
        torch.from_numpy(matches_idx).long()
    )

'''