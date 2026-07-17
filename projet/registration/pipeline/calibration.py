
import os
import cv2  
import numpy as np
import torch
from pathlib import Path
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import List


def get_calibration_mode(save_dir):
    path = str(save_dir).lower()

    if "cloudy" in path:
        return "houghp_score"

    if "snowy" in path or "sunny" in path:
        return "hough_score"

    if "rainy" in path:
        return "hough_score"

    return "hough_score"

def score_frame_for_calibration(frame, min_line_length=50):
    h, w = frame.shape

    edges = cv2.Canny(frame, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=40,
        minLineLength=min_line_length,
        maxLineGap=15
    )

    if lines is None or len(lines) == 0:
        print("lines = 0")
        return 0.0

    print("lines =", len(lines))

    margin_h = h // 4
    margin_w = w // 4

    total_score = 0.0

    for line in lines:
        x1, y1, x2, y2 = line[0]

        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Sécurité supplémentaire : éliminer les lignes courtes.
        if length < min_line_length:
            continue

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        is_corner = (
            (cy < margin_h or cy > h - margin_h) and
            (cx < margin_w or cx > w - margin_w)
        )

        is_edge = (
            (cy < margin_h or cy > h - margin_h) or
            (cx < margin_w or cx > w - margin_w)
        )

        if is_corner:
            spatial_weight = 3.0
        elif is_edge:
            spatial_weight = 1.5
        else:
            spatial_weight = 0.5

        # Ici, la longueur compte vraiment.
        total_score += spatial_weight * (length / min_line_length)

    return total_score

def calibrate_distortion_from_samples(
    sample_tensors: List[torch.Tensor],
    camera_name: str = 'vis',
    save_dir: str = './calibrations',
    method: str = 'hough_entropy'
) -> dict:
    """
    Estime les paramètres de distorsion radiale (k1, k2) d'une caméra à partir
    d'un ensemble de frames. Sélectionne la meilleure frame par score spatial,
    puis optimise les coefficients de distorsion en minimisant l'entropie des
    angles de Hough (lignes les plus droites = entropie plus faible).
    Sauvegarde les paramètres et les visualisations Canny dans save_dir.

    Args:
        sample_tensors: Liste de tensors PyTorch [C, H, W] dans [0, 1].
        camera_name:    Identifiant caméra pour le nommage des fichiers.
        save_dir:       Répertoire de sortie pour les fichiers de calibration.
        method:         Méthode utilisée (seule 'hough_entropy' est implémentée).

    Returns:
        Dictionnaire avec les clés k1, k2, cx, cy, fx, fy, width, height,
        method, entropy_cost.
    """
    calib_mode = get_calibration_mode(save_dir)
    print(f"[{camera_name}] Calibration mode: {calib_mode}")
    from scipy.stats import entropy as scipy_entropy
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    canny_dir = Path(save_dir) / 'canny'
    canny_dir.mkdir(parents=True, exist_ok=True)
    
    numpy_images = []
    for tensor in sample_tensors:
        img_np = tensor.cpu().numpy()
        
        if img_np.ndim == 3:
            if img_np.shape[0] == 3:
                img_np = img_np.transpose(1, 2, 0)
                img_np = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                img_np = img_np[0]
        
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
        
        numpy_images.append(img_np)
    '''
    print(f"[{camera_name}] Sélection meilleure frame (score spatial)...")
    best_frame, best_score, best_idx = None, 0.0, 0
    
    for idx, frame in enumerate(numpy_images):
        score = score_frame_for_calibration(frame, min_line_length=100)
        print(f"  Frame {idx}: score={score:.2f}")
        
        if score > best_score:
            best_score = score
            best_frame = frame
            best_idx = idx
    
    if best_frame is None:
        best_frame = numpy_images[0]
        best_idx = 0'''

    print(f"[{camera_name}] Sélection meilleure frame (score spatial)...")
    best_frame, best_score, best_idx = None, 0.0, 0

    for idx, frame in enumerate(numpy_images):
        score = score_frame_for_calibration(frame)
        print(f"  Frame {idx}: score={score:.2f}")

        if score > best_score:
            best_score = score
            best_frame = frame
            best_idx = idx

    if best_frame is None:
        best_frame = numpy_images[0]
        best_idx = 0
            
    h, w = best_frame.shape
    print(f"[{camera_name}] Frame sélectionnée (score={best_score:.2f}): {w}x{h}")
    
    edges_original = cv2.Canny(best_frame, 50, 150)
    cv2.imwrite(str(canny_dir / f'{camera_name}_canny_original_frame_{best_idx}.png'), edges_original)
    print(f"  Canny original sauvegardé")
    
    print(f"[{camera_name}] Calibration via Hough Entropy...")
    
    fx = fy = w
    
    def hough_entropy_cost(params):
        k1, k2, cx, cy = params
        
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float32)
        
        img_ud = cv2.undistort(best_frame, K, dist_coeffs)
        edges = cv2.Canny(img_ud, 100, 200)
        if calib_mode == "hough_score":

            lines = cv2.HoughLines(
                edges,
                1,
                np.pi/180,
                threshold=50
            )

            if lines is None or len(lines) < 10:
                return 1e6

            angles = lines[:, 0, 1] * 180 / np.pi

        else:

            lines = cv2.HoughLinesP(
                edges,
                1,
                np.pi/180,
                threshold=40,
                minLineLength=100,
                maxLineGap=15
            )

            if lines is None or len(lines) < 10:
                return 1e6

            angles = []

            for line in lines:
                x1, y1, x2, y2 = line[0]

                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                if length < 100:
                    continue

                angle = np.degrees(
                    np.arctan2(y2 - y1, x2 - x1)
                ) % 180

                angles.append(angle)

            if len(angles) < 10:
                return 1e6

        hist, _ = np.histogram(angles, bins=180, range=(0, 180))

        return scipy_entropy(hist + 1)

    
    print(f"  Optimisation multi-start...")
    
    initial_guesses = [
        [0.0, 0.0, w/2, h/2],
        [-0.1, 0.0, w/2, h/2],
        [-0.2, 0.05, w/2, h/2],
        [0.1, 0.0, w/2, h/2],
        [-0.15, 0.0, w*0.48, h*0.48],
    ]
    
    bounds = [(-0.5, 0.5), (-0.2, 0.2), (w*0.3, w*0.7), (h*0.3, h*0.7)]
    
    best_result = None
    best_cost = float('inf')
    
    for idx, initial in enumerate(initial_guesses):
        result = minimize(
            hough_entropy_cost,
            initial,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 30, 'disp': False}
        )
        
        if result.fun < best_cost:
            best_cost = result.fun
            best_result = result
            print(f"    Start {idx+1}/5: cost={result.fun:.6f} BEST")
        else:
            print(f"    Start {idx+1}/5: cost={result.fun:.6f}")
    
    k1, k2, cx, cy = best_result.x
    
    print(f"  Convergence: k1={k1:.6f}, k2={k2:.6f}, center=({cx:.1f}, {cy:.1f})")
    print(f"    Entropy: {best_cost:.6f}")
    original_cost = hough_entropy_cost([0.0, 0.0, w/2, h/2])

    print(f"    Original entropy : {original_cost:.6f}")
    print(f"    Best entropy     : {best_cost:.6f}")
    print(f"    Gain             : {original_cost - best_cost:.6f}")
    if abs(k1) >= 0.01 or abs(k2) >= 0.01:
        K_undist = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float32)
        img_undistorted = cv2.undistort(best_frame, K_undist, dist_coeffs)
        
        edges_undistorted = cv2.Canny(img_undistorted, 100, 200)
        cv2.imwrite(str(canny_dir / f'{camera_name}_canny_undistorted_frame_{best_idx}.png'), edges_undistorted)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].imshow(edges_original, cmap='gray')
        axes[0].set_title(f'{camera_name.upper()} Canny Original')
        axes[0].axis('off')
        
        axes[1].imshow(edges_undistorted, cmap='gray')
        axes[1].set_title(f'{camera_name.upper()} Canny Undistorted (k1={k1:.4f})')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(canny_dir / f'{camera_name}_canny_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Comparaison Canny sauvegardée")
    
    debug_path = Path(save_dir) / f'{camera_name}_calib_debug.png'
    try:
        if abs(k1) >= 0.01 or abs(k2) >= 0.01:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            axes[0].imshow(best_frame, cmap='gray')
            axes[0].set_title(f'{camera_name.upper()} Original')
            axes[0].grid(True, color='red', linewidth=0.5, alpha=0.5)
            
            axes[1].imshow(img_undistorted, cmap='gray')
            axes[1].set_title(f'{camera_name.upper()} Undistorted (k1={k1:.4f}, k2={k2:.4f})')
            axes[1].grid(True, color='red', linewidth=0.5, alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(debug_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Debug image sauvegardé")
        else:
            print(f"  Pas de visualisation debug")
            
    except Exception as e:
        print(f"  Visualisation debug failed: {e}")
    
    calib_params = {
        'k1': float(k1),
        'k2': float(k2),
        'cx': float(cx),
        'cy': float(cy),
        'fx': float(fx),
        'fy': float(fy),
        'width': int(w),
        'height': int(h),
        'method': 'hough_entropy',
        'entropy_cost': float(best_cost)
    }
    
    save_path = Path(save_dir) / f'{camera_name}_calib.npz'
    np.savez(save_path, **calib_params)
    print(f"Calibration sauvegardée\n")
    
    return calib_params
