"""
Module de diagnostic et visualisation pour le pipeline de recalage VIS-SWIR.

Contient les fonctions d'analyse post-estimation : calcul d'erreurs de reprojection,
statistiques globales, et génération de visualisations diagnostiques.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from registration.pipeline.metrics import compute_all_metrics
import cv2

    
def compute_reprojection_errors(Homography, pts_vis, pts_swir, inliers, config):
    """
    Calcule les erreurs de reprojection euclidiennes pour les points inliers
    en projetant les points VIS vers le repère SWIR via l'homographie.

    Args:
        Homography:  Matrice 3×3.
        pts_vis:     Points source VIS (N, 2).
        pts_swir:    Points destination SWIR (N, 2).
        inliers:     Masque booléen (N,) indiquant les inliers.

    Returns:
        Tuple (errors, pts_vis_inliers, pts_swir_inliers).
        errors           : distances euclidiennes (K,) pour les inliers.
        pts_vis_inliers  : points VIS inliers (K, 2).
        pts_swir_inliers : points SWIR inliers (K, 2).
    """
    direction = getattr(config, "ALIGN_DIRECTION", "VIS_TO_SWIR")

    pts_vis_inliers = pts_vis[inliers]
    pts_swir_inliers = pts_swir[inliers]

    if direction == "VIS_TO_SWIR":
        src = pts_vis_inliers
        dst = pts_swir_inliers
    elif direction == "SWIR_TO_VIS":
        src = pts_swir_inliers
        dst = pts_vis_inliers
    else:
        raise ValueError(f"ALIGN_DIRECTION invalide : {direction}")

    src_homo = np.hstack([src, np.ones((len(src), 1))])
    dst_proj = (Homography @ src_homo.T).T
    dst_proj = dst_proj[:, :2] / dst_proj[:, 2:3]

    errors = np.linalg.norm(dst - dst_proj, axis=1)

    return errors, pts_vis_inliers, pts_swir_inliers


def print_statistics(container, inliers, errors, rgb_files, swir_files, Homography):
    """
    Affiche sur stdout un rapport de recalage : nombre de matches, ratio
    inliers/outliers, statistiques d'erreur de reprojection, évaluation
    qualitative, et métriques image (GOA, Edge Alignment, MIND) sur la première paire.

    Args:
        container:   SequenceMatchesContainer avec tous les matches.
        inliers:     Masque booléen (N,).
        errors:      Erreurs de reprojection (K,) pour les inliers.
        rgb_files:   Liste de chemins vers les images VIS (au moins 1).
        swir_files:  Liste de chemins vers les images SWIR (au moins 1).
        Homography:  Matrice 3×3 estimée.
    """
    pts_vis_all, pts_swir_all = container.get_all_matches()
    
    print("=" * 80)
    print(f"{'DIAGNOSTIC ET MÉTRIQUES':^80}")
    print("=" * 80)
    print("STATISTIQUES GLOBALES")
    print(f"  Images traitées       : {len(container.pairs)}")
    print(f"  Matches totaux        : {len(pts_vis_all)}")
    print(f"  Inliers RANSAC        : {inliers.sum()} ({100 * inliers.sum() / len(inliers):.1f}%)")
    print(f"  Outliers RANSAC       : {len(inliers) - inliers.sum()} ({100 * (1 - inliers.sum() / len(inliers)):.1f}%)")
    print(f"\nERREUR DE REPROJECTION (inliers)")
    print(f"  Moyenne               : {errors.mean():.2f} px")
    print(f"  Médiane               : {np.median(errors):.2f} px")
    print(f"  Max                   : {errors.max():.2f} px")
    print(f"  Std                   : {errors.std():.2f} px")
    
    # Qualité globale
    if errors.mean() < 2:
        quality = "Excellent"
    elif errors.mean() < 5:
        quality = "Bon"
    elif errors.mean() < 10:
        quality = "Moyen"
    else:
        quality = "Mauvais"
    print(f"  Qualité               : {quality}")



    img_vis  = cv2.imread(rgb_files[0])
    img_swir = cv2.imread(swir_files[0])
    m = compute_all_metrics(img_vis, img_swir, Homography)

    print(f"\nMÉTRIQUES IMAGE (paire 0)")
    print(f"  GOA      : {m['goa']:.4f}    (1 = parfait)")
    print(f"  Edge Alignment : {m['edge_f1_8px']:.4f}  (1 = parfait)")
    print(f"  MIND SSD : {m['mind_ssd']:.4f}  (0 = parfait)")


def plot_diagnostics(container, inliers, errors, pts_vis_inliers, pts_swir_inliers, paths):
    """
    Génère et sauvegarde une figure 2×2 comprenant :
    - histogramme des erreurs de reprojection,
    - heatmap de densité des inliers VIS,
    - heatmap de densité des inliers SWIR,
    - scatter plot inliers vs outliers VIS.

    Args:
        container:         SequenceMatchesContainer.
        inliers:           Masque booléen (N,).
        errors:            Erreurs de reprojection (K,).
        pts_vis_inliers:   Points VIS inliers (K, 2).
        pts_swir_inliers:  Points SWIR inliers (K, 2).
        paths:             Dictionnaire avec clé 'out_dir' (chemin de sortie).
    """
    pts_vis_all, pts_swir_all = container.get_all_matches()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogramme erreurs de reprojection
    axes[0, 0].hist(errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(errors.mean(), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {errors.mean():.2f}px')
    axes[0, 0].axvline(np.median(errors), color='orange', linestyle='--', linewidth=2, 
                       label=f'Median: {np.median(errors):.2f}px')
    axes[0, 0].set_xlabel('Erreur de reprojection (pixels)')
    axes[0, 0].set_ylabel('Fréquence')
    axes[0, 0].set_title('Distribution erreur de reprojection')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Heatmap densité inliers VIS
    axes[0, 1].hexbin(pts_vis_inliers[:, 0], pts_vis_inliers[:, 1], 
                      gridsize=30, cmap='hot', mincnt=1)
    axes[0, 1].set_xlabel('X (pixels)')
    axes[0, 1].set_ylabel('Y (pixels)')
    axes[0, 1].set_title('Densité inliers VIS')
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_aspect('equal', adjustable='box')
    
    # 3. Heatmap densité inliers SWIR
    axes[1, 0].hexbin(pts_swir_inliers[:, 0], pts_swir_inliers[:, 1], 
                      gridsize=30, cmap='hot', mincnt=1)
    axes[1, 0].set_xlabel('X (pixels)')
    axes[1, 0].set_ylabel('Y (pixels)')
    axes[1, 0].set_title('Densité inliers SWIR')
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_aspect('equal', adjustable='box')
    
    # 4. Scatter inliers vs outliers VIS
    outlier_count = len(inliers) - inliers.sum()
    axes[1, 1].scatter(pts_vis_all[~inliers, 0], pts_vis_all[~inliers, 1], 
                       c='red', s=2, alpha=0.3, label=f'Outliers: {outlier_count}')
    axes[1, 1].scatter(pts_vis_all[inliers, 0], pts_vis_all[inliers, 1], 
                       c='green', s=2, alpha=0.5, label=f'Inliers: {inliers.sum()}')
    axes[1, 1].set_xlabel('X (pixels)')
    axes[1, 1].set_ylabel('Y (pixels)')
    axes[1, 1].set_title('Inliers vs Outliers VIS')
    axes[1, 1].invert_yaxis()
    axes[1, 1].legend()
    axes[1, 1].set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(os.path.join(paths['out_dir'], 'diagnostic_global.png'), dpi=150)
    plt.close()
    
    print(f"\n✓ Visualisations sauvegardées: {paths['out_dir']}")


def run_diagnostics(container, Homography, inliers, paths, rgb_files=None, swir_files=None):
    """
    Orchestre le diagnostic complet : calcul des erreurs de reprojection,
    affichage des statistiques et génération des visualisations.

    Args:
        container:   SequenceMatchesContainer avec tous les matches.
        Homography:  Matrice 3×3 estimée.
        inliers:     Masque booléen (N,).
        paths:       Dictionnaire avec clé 'out_dir'.
        rgb_files:   Liste de chemins images VIS (optionnel, pour métriques image).
        swir_files:  Liste de chemins images SWIR (optionnel).
    """
    pts_vis_all, pts_swir_all = container.get_all_matches()
    
    # Calcul erreurs de reprojection
    errors, pts_vis_inliers, pts_swir_inliers = compute_reprojection_errors(
        Homography, pts_vis_all, pts_swir_all, inliers, config
    )
        
    # Affichage statistiques
    print_statistics(container, inliers, errors, rgb_files, swir_files, Homography)
    
    # Génération visualisations
    plot_diagnostics(container, inliers, errors, pts_vis_inliers, pts_swir_inliers, paths)
