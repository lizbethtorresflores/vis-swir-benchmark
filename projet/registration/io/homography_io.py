import numpy as np




def save_homography_and_mask(H, mask, filepath):
    """
    Sauvegarde homographie 3×3 et masque RANSAC dans .npz
    
    Args:
        H: Matrice 3×3 (float)
        mask: Masque RANSAC (N, 1) ou (N,) de bool ou 0/1
        filepath: Chemin fichier .npz (ex: 'homography.npz')
    """
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"Homographie invalide, shape={H.shape}, attendu (3,3)")
    
    mask = np.asarray(mask)
    mask_flat = mask.ravel().astype(bool)
    
    np.savez(filepath, H=H, mask=mask_flat)
    print(f"Sauvegardé : {filepath}")


def load_homography_and_mask(filepath):
    """
    Charge homographie 3×3 et masque RANSAC depuis .npz
    
    Args:
        filepath: Chemin fichier .npz
    
    Returns:
        (H, mask) - Homographie 3×3 et masque inliers 1D bool
    """
    data = np.load(filepath)
    H = np.asarray(data["H"], dtype=np.float64)
    mask = np.asarray(data["mask"], dtype=bool)
    inliers = mask.ravel() == 1

    
    if H.shape != (3, 3):
        raise ValueError(f"Fichier ne contient pas homographie 3×3, shape={H.shape}")
    
    print(f"Chargé : {filepath}")
    return H, inliers

