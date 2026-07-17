"""
3. Fichier d'exécution principal - Pipeline IR/VIS
Nettoyé de dépendances ImageHandler
"""
from torch.profiler import profile, record_function, ProfilerActivity
import os
import torch
from lightglue.utils import load_image
import cv2 as cv
import numpy as np
import argparse
import time
import shutil
from pathlib import Path

from config import Config



from registration.preprocessing import precompute_remap_maps, apply_undistortion, apply_clahe_kornia
from registration.io.image_loader       import load_image_paths
from registration.io.homography_io      import load_homography_and_mask
from registration.data.sequence_matches_container import SequenceMatchesContainer
from registration.pipeline.calibration       import calibrate_distortion_from_samples
from registration.preprocessing.undistortion import precompute_remap_maps
from registration.pipeline.extraction import process_all_images, estimate_global_vehicle_crop
from registration.pipeline.estimation   import compute_homography
from registration.pipeline.application  import apply_homography_to_images
from registration.pipeline.diagnostics  import run_diagnostics
from registration.preprocessing.contrast     import apply_clahe_kornia
from registration.preprocessing.undistortion import apply_undistortion




def print_system_info(config):
    """Affiche informations système et device"""
    print(f"\n{'='*80}")
    print(f"RECALAGE MULTIMODAL VIS-SWIR - HOMOGRAPHIE CLASSIQUE")
    print(f"{'='*80}")
    
    print(f"\nCONFIGURATION SYSTÈME")
    print(f"PyTorch version      : {torch.__version__}")
    
    if config.DEVICE == "mps":
        print(f"MPS built            : {torch.backends.mps.is_built()}")
        print(f"MPS available        : {torch.backends.mps.is_available()}")
    elif config.DEVICE == "cuda":
        print(f"CUDA available       : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device          : {torch.cuda.get_device_name(0)}")
    
    print(f"Device utilisé       : {config.DEVICE}")


def print_data_info(paths, rgb_files, swir_files, config):
    """Affiche informations sur les données"""
    print(f"\nDONNÉES")
    print(f"RGB dir              : {paths['rgb_dir']}")
    print(f"SWIR dir             : {paths['swir_dir']}")
    print(f"Images RGB trouvées  : {len(rgb_files)}")
    print(f"Images SWIR trouvées : {len(swir_files)}")
    print(f"Images à traiter     : {config.NUM_IMAGES}")
    print(f"Dossier sortie       : {paths['out_dir']}")


def print_extraction_params(config):
    """Affiche paramètres extraction"""
    print(f"\nPARAMÈTRES EXTRACTION")
    print(f"Grille               : {config.GRID_ROWS}×{config.GRID_COLS}")
    print(f"Overlap              : {config.GRID_OVERLAP_PERCENT}%")
    print(f"Points base          : {config.NUM_POINT_LG}")
    print(f"Coins x{config.CORNER_MULTIPLIER} | Bords x{config.EDGE_MULTIPLIER} | Centre x{config.CENTER_MULTIPLIER}")


def print_ransac_params(config):
    """Affiche paramètres RANSAC"""
    print(f"\nPARAMÈTRES ESTIMATION")
    if config.USE_MAGSAC:
        print(f"Algorithme           : MAGSAC++")
        print(f"Threshold            : {config.MAGSAC_THRESHOLD} px")
        print(f"Confidence           : {config.MAGSAC_CONFIDENCE}")
        print(f"Max iterations       : {config.MAGSAC_MAX_ITERS}")
    else:
        print(f"Algorithme           : RANSAC")
        print(f"Min samples          : {config.RANSAC_MIN_SAMPLES}")
        print(f"Threshold            : {config.RANSAC_THRESHOLD} px")
        print(f"Max trials           : {config.RANSAC_MAX_TRIALS}")
        print(f"Stop probability     : {config.RANSAC_STOP_PROB}")





# def step_0_calibration_estimation(rgb_files, swir_files, config, paths):
#     # Calibration des caméras
#     img_vis_tensor = [] 
#     img_swir_tensor = []
#     distortion_threshold = 0.01
#     map1_vis = None
#     map2_vis = None
#     map1_swir = None
#     map2_swir = None

#     NUM_CALIB_SAMPLES = 20

#     if config.APPLY_DISTORTION_VIS :
#         for i, (rgb_path, swir_path) in enumerate(zip(rgb_files, swir_files)):
#             if i < NUM_CALIB_SAMPLES:
#                 if (config.CLAHE_RADIAL_ESTIMATION_VIS):
#                     img_vis_tensor.append(apply_clahe_kornia(load_image(rgb_path), 15.0).to(config.DEVICE))
#                 else:
#                     img_vis_tensor.append(load_image(rgb_path).to(config.DEVICE))

#         calib_vis = calibrate_distortion_from_samples(img_vis_tensor, save_dir=paths['out_dir']+'/calibrations/',camera_name='vis')
#         del img_vis_tensor
#         if (abs(calib_vis['k1']) >= distortion_threshold):
#             map1_vis, map2_vis = precompute_remap_maps(paths['out_dir']+'/calibrations/vis_calib.npz', config.DEVICE)


#     if config.APPLY_DISTORTION_SWIR:
#         for i, (rgb_path, swir_path) in enumerate(zip(rgb_files, swir_files)):
#             if i < NUM_CALIB_SAMPLES:
#                 if (config.CLAHE_RADIAL_ESTIMATION_SWIR):
#                     img_swir_tensor.append(apply_clahe_kornia(load_image(swir_path), 15.0).to(config.DEVICE))
#                 else:
#                     img_swir_tensor.append(load_image(swir_path).to(config.DEVICE))

                
#         calib_swir = calibrate_distortion_from_samples(img_swir_tensor, save_dir=paths['out_dir']+'/calibrations/',camera_name='swir')
#         del img_swir_tensor
#         if (abs(calib_swir['k1']) >= distortion_threshold):
#             map1_swir, map2_swir = precompute_remap_maps(paths['out_dir']+'/calibrations/swir_calib.npz', config.DEVICE)

#     return map1_vis, map2_vis, map1_swir, map2_swir


#     # ========== CHECK SI DISTORSION NÉGLIGEABLE ==========



#     # if (abs(calib_vis['k1']) < distortion_threshold and 
#     #     abs(calib_swir['k1']) < distortion_threshold):
        
#     #     print("\n" + "="*80)
#     #     print("DISTORSION NÉGLIGEABLE CONFIRMÉE")
#     #     print("="*80)
#     #     print("k1_vis = {:.6f}, k1_swir = {:.6f}".format(calib_vis['k1'], calib_swir['k1']))
#     #     print("→ Pas besoin d'undistortion, images déjà sans distorsion")
#     #     print("→ Remap désactivé (identité)")
#     #     print("="*80 + "\n")
        
#     #     # Retourner None pour signaler "pas de remap nécessaire"
#     #     return None, None, None, None
    
#     # else:
#     #     print("\n" + "="*80)
#     #     print("DISTORSION DÉTECTÉE")
#     #     print("="*80)
#     #     print("k1_vis = {:.6f}, k1_swir = {:.6f}".format(calib_vis['k1'], calib_swir['k1']))
#     #     print("→ Calcul remap maps...")
#     #     print("="*80 + "\n")
        
#         # map1_vis, map2_vis = precompute_remap_maps(paths['out_dir']+'/calibrations/vis_calib.npz', config.DEVICE)
#         # map1_swir, map2_swir = precompute_remap_maps(paths['out_dir']+'/calibrations/swir_calib.npz', config.DEVICE)
        


def step_0_calibration_estimation(rgb_files, swir_files, config, paths, num_calib_samples=50):
    distortion_threshold = 0.01
    NUM_CALIB_SAMPLES = num_calib_samples

    map1_vis = None
    map2_vis = None
    map1_swir = None
    map2_swir = None

    if config.APPLY_DISTORTION_VIS:
        img_vis_tensor = []
        for i, (rgb_path, _) in enumerate(zip(rgb_files, swir_files)):
            if i >= NUM_CALIB_SAMPLES:
                break

            img = load_image(rgb_path).to(config.DEVICE)
            if config.CLAHE_RADIAL_ESTIMATION_VIS:
                img = apply_clahe_kornia(img, 2.0)
            img_vis_tensor.append(img)

        calib_vis = calibrate_distortion_from_samples(
            img_vis_tensor,
            save_dir=paths['out_dir'] + '/calibrations/',
            camera_name='vis'
        )
        del img_vis_tensor

        if abs(calib_vis['k1']) >= distortion_threshold or abs(calib_vis['k2']) >= distortion_threshold:
            print(f"[VIS] Distorsion détectée (k1={calib_vis['k1']:.6f}, k2={calib_vis['k2']:.6f}) → calcul remap")
            map1_vis, map2_vis = precompute_remap_maps(
                paths['out_dir'] + '/calibrations/vis_calib.npz',
                config.DEVICE
            )
        else:
            print(f"[VIS] Pas de distorsion → remap=None")

    if config.APPLY_DISTORTION_SWIR:
        img_swir_tensor = []
        for i, (_, swir_path) in enumerate(zip(rgb_files, swir_files)):
            if i >= NUM_CALIB_SAMPLES:
                break

            img = load_image(swir_path).to(config.DEVICE)
            if config.CLAHE_RADIAL_ESTIMATION_SWIR:
                img = apply_clahe_kornia(img, 2.0)
            img_swir_tensor.append(img)

        calib_swir = calibrate_distortion_from_samples(
            img_swir_tensor,
            save_dir=paths['out_dir'] + '/calibrations/',
            camera_name='swir'
        )
        del img_swir_tensor

        if abs(calib_swir['k1']) >= distortion_threshold or abs(calib_swir['k2']) >= distortion_threshold:
            print(f"[SWIR] Distorsion détectée (k1={calib_swir['k1']:.6f}, k2={calib_swir['k2']:.6f}) → calcul remap")
            map1_swir, map2_swir = precompute_remap_maps(
                paths['out_dir'] + '/calibrations/swir_calib.npz',
                config.DEVICE
            )
        else:
            print(f"[SWIR] Pas de distorsion → remap=None")

    return map1_vis, map2_vis, map1_swir, map2_swir

def step_1_extraction(rgb_files, swir_files, config):
    """
    ÉTAPE 1 : Extraction matches avec grille chevauchante
    
    Args:
        rgb_files: Liste chemins images RGB
        swir_files: Liste chemins images SWIR
        config: Instance Config
    
    Returns:
        SequenceMatchesContainer avec tous les matches
    """
    print(f"\n{'='*80}")
    print(f"ÉTAPE 1/3 : EXTRACTION MATCHES")
    print(f"{'='*80}\n")
    
    container = process_all_images(rgb_files, swir_files, config)
    
    if len(container.pairs) == 0:
        raise RuntimeError(
            "ERREUR : Aucune paire d'images traitée avec succès\n"
            "Vérifiez les chemins et la qualité des images"
        )
    
    print(f"\n Extraction terminée : {len(container.pairs)} paires traitées")
    return container


def step_2_estimation(container, config, paths):
    """
    ÉTAPE 2 : Estimation homographie robuste
    
    Args:
        container: SequenceMatchesContainer avec matches
        config: Instance Config
        paths: Dict chemins
    
    Returns:
        (Homography matrix 3×3, inliers mask)
    """
    print(f"\n{'='*80}")
    print(f"ÉTAPE 2/3 : ESTIMATION HOMOGRAPHIE")
    print(f"{'='*80}\n")

    if (config.LOAD_HOMOGRAPHY):
        Homography, inliers = load_homography_and_mask(paths['homography_path'])
    else:
        Homography, inliers = compute_homography(container, config, paths)
    
    print(f"\n Homographie estimée")
    print(f"Matrice sauvegardée  : {os.path.join(paths['out_dir'], 'homography.npz')}")
    
    return Homography, inliers


def step_3_application(Homography, rgb_files, swir_files, config, paths):
    """
    ÉTAPE 3 : Application homographie à toutes les images
    
    Args:
        Homography: Matrice homographie 3×3
        rgb_files: Liste chemins images RGB
        swir_files: Liste chemins images SWIR
        config: Instance Config
        paths: Dict chemins
    """
    print(f"\n{'='*80}")
    print(f"ÉTAPE 3/3 : APPLICATION HOMOGRAPHIE")
    print(f"{'='*80}\n")
    
    apply_homography_to_images(Homography, rgb_files, swir_files, config, paths)
    
    print(f"\n Application terminée")


def step_4_diagnostic(container, Homography, inliers, paths, config, rgb_files=None, swir_files=None):
    """
    ÉTAPE 4 (optionnel) : Diagnostic et visualisations
    
    Args:
        container: SequenceMatchesContainer
        Homography: Matrice homographie 3×3
        inliers: Masque inliers
        paths: Dict chemins
        config: Instance Config
        rgb_files: Liste chemins images RGB (optionnel, pour diagnostics)
        swir_files: Liste chemins images SWIR (optionnel, pour diagnostics)
    """
    if not config.DEBUG:
        return
    
    print(f"\n{'='*80}")
    print(f"ÉTAPE 4 : DIAGNOSTIC")
    print(f"{'='*80}\n")
    
    run_diagnostics(container, Homography, inliers, paths, rgb_files=rgb_files, swir_files=swir_files)
    
    print(f"\n Diagnostic généré")


def print_results_summary(paths, config):
    """Affiche résumé des résultats"""
    print(f"\n{'='*80}")
    print(f"PIPELINE TERMINÉ AVEC SUCCÈS")
    print(f"{'='*80}")
    print(f"\nRésultats disponibles dans : {paths['out_dir']}")
    print(f"\nFichiers générés :")
    print(f"  split_images/         : Images VIS alignées et SWIR originales")
    print(f"  concatenate_images/   : Overlays 50% alpha (fusion)")
    print(f"  homography.npz        : Matrice homographie estimée")
    
    if config.DEBUG:
        print(f"  diagnostic_global.png : Visualisations statistiques")
    
    print()


def main():
    """
    Pipeline principal de recalage IR/VIS
    
    Étapes :
    1. Extraction matches avec grille chevauchante et boost coins/bords
    2. Estimation homographie robuste (MAGSAC++ ou RANSAC)
    3. Application homographie à toutes les images
    4. Diagnostic et visualisations (optionnel)
    """
    
    # ═══════════════════════════════════════════════════════════════
    # INITIALISATION
    # ═══════════════════════════════════════════════════════════════
    


    parser = argparse.ArgumentParser(description="Pipeline de recalage VIS-SWIR")

    parser.add_argument("--bag", default=None,
                        help="Chemin relatif depuis datas/ vers le dossier paired_images")

    parser.add_argument("--train-start", type=int, default=0,
                        help="Indice de départ du train")

    parser.add_argument("--train-step", type=int, default=10,
                        help="Pas entre les images du train")

    parser.add_argument("--train-count", type=int, default=10,
                        help="Nombre d’images utilisées pour le train")

    parser.add_argument("--test-start", type=int, default=100,
                        help="Indice de départ du test")

    parser.add_argument("--test-end", type=int, default=500,
                        help="Indice final du test")

    parser.add_argument("--calib-samples", type=int, default=50,
                        help="Nombre d'images utilisées pour la calibration")

    args = parser.parse_args()

    config    = Config()
    script_dir = os.path.dirname(os.path.abspath(__file__))


    if args.bag:
        projet_root = Path(__file__).resolve().parent.parent
        bag_path    = str(projet_root / "datas" / args.bag / "Sortie_Extraite" / "paired_images")
    else:
        bag_path = None

    paths = config.setup_paths(script_dir, bag_path=bag_path)
    config.PRE_H_DEBUG_DIR = os.path.join(paths["out_dir"], "pre_h_crop_images")

    # Affichage informations
    print_system_info(config)
    
    # Charger les chemins des images
    rgb_files, swir_files = load_image_paths(paths)

    # Afficher les informations générales
    print_data_info(paths, rgb_files, swir_files, config)
    print_extraction_params(config)
    print_ransac_params(config)

    # ============================================================
    # SÉPARATION TRAIN / TEST
    # ============================================================

    train_indices = [
        args.train_start + i * args.train_step
        for i in range(args.train_count)
    ]

    test_indices = list(range(args.test_start, args.test_end + 1))

    # Éviter toute intersection entre train et test
    # train_set = set(train_indices)
    # test_indices = [i for i in test_indices if i not in train_set]

    # Éviter les indices hors limites du dataset
    max_len = min(len(rgb_files), len(swir_files))

    train_indices = [i for i in train_indices if i < max_len]
    test_indices = [i for i in test_indices if i < max_len]

    # Construire les sous-ensembles train et test
    rgb_train = [rgb_files[i] for i in train_indices]
    swir_train = [swir_files[i] for i in train_indices]

    rgb_test = [rgb_files[i] for i in test_indices]
    swir_test = [swir_files[i] for i in test_indices]
    # ============================================================
    # ESTIMATION GLOBAL VEHICLE CROP
    # ============================================================

    config.PRE_H_CROP_Y_VIS, config.PRE_H_CROP_Y_SWIR = estimate_global_vehicle_crop(
        rgb_train,
        swir_train,
        config,
        max_samples=20
    )

    print(f"Global pre-H crop VIS : y=0:{config.PRE_H_CROP_Y_VIS}")
    print(f"Global pre-H crop SWIR: y=0:{config.PRE_H_CROP_Y_SWIR}")
    # Affichage du split utilisé
    print("\nSÉPARATION TRAIN / TEST")
    print(f"Indices train : {train_indices}")

    if len(test_indices) > 0:
        print(f"Test début/fin : {test_indices[0]} -> {test_indices[-1]}")
    else:
        print("Aucune image de test sélectionnée")

    print(f"Images train : {len(rgb_train)}")
    print(f"Images test : {len(rgb_test)}")
    # ============================================================
    # SAUVEGARDE DES IMAGES TRAIN
    # ============================================================

    train_vis_dir = os.path.join(paths['out_dir'], "train_images", "VISIBLE")
    train_swir_dir = os.path.join(paths['out_dir'], "train_images", "SWIR")

    os.makedirs(train_vis_dir, exist_ok=True)
    os.makedirs(train_swir_dir, exist_ok=True)

    for rgb_path in rgb_train:
        shutil.copy2(
            rgb_path,
            os.path.join(train_vis_dir, os.path.basename(rgb_path))
        )

    for swir_path in swir_train:
        shutil.copy2(
            swir_path,
            os.path.join(train_swir_dir, os.path.basename(swir_path))
        )

    print("\nImages train sauvegardées")
    print(f"VIS  : {train_vis_dir}")
    print(f"SWIR : {train_swir_dir}")


    # ═══════════════════════════════════════════════════════════════
    # PIPELINE
    # ═══════════════════════════════════════════════════════════════
    global_start = time.time()

    try:

        print("debut estimation calibration")
        map1_vis, map2_vis, map1_swir, map2_swir = step_0_calibration_estimation(
            rgb_train,
            swir_train,
            config,
            paths,
            num_calib_samples=args.calib_samples
        )
        print("fin estimation calibration")


        config.UNDISTORT_MAPS = {
                    'vis': (map1_vis, map2_vis),
                    'swir': (map1_swir, map2_swir)
                }


        if (config.LOAD_HOMOGRAPHY == False):
            # Étape 1 : Extraction matches
            config.NUM_IMAGES = len(rgb_train)
            extract_start = time.time()
            container = step_1_extraction(rgb_train, swir_train, config)
            extract_end = time.time()
            extract_time = extract_end - extract_start
            fps_extract = len(rgb_train) / extract_time if extract_time > 0 else 0
            print(f"\nTemps extraction : {extract_time:.2f} s")
            print(f"FPS extraction : {fps_extract:.2f}")


        else : 
            container = SequenceMatchesContainer()

        # Étape 2 : Estimation homographie
        h_start = time.time()
        Homography, inliers = step_2_estimation(container, config, paths)
        h_end = time.time()
        h_time = h_end - h_start

        fps_h = len(rgb_train) / h_time if h_time > 0 else 0

        print(f"\nTemps estimation homographie : {h_time:.2f} s")
        print(f"FPS homographie : {fps_h:.2f}")
                        
        # Étape 3 : Application homographie
        config.NUM_IMAGES_TO_WARP = len(rgb_test)
        apply_start = time.time()
        step_3_application(Homography, rgb_test, swir_test, config, paths)
        apply_end = time.time()
        apply_time = apply_end - apply_start
        fps_apply = len(rgb_test) / apply_time if apply_time > 0 else 0

        print(f"\nTemps application : {apply_time:.2f} s")
        print(f"FPS application : {fps_apply:.2f}")
        
        if (config.LOAD_HOMOGRAPHY == False):
            # Étape 4 : Diagnostic (optionnel)
            # Juste ajouter rgb_files/swir_files en paramètre
            step_4_diagnostic(container, Homography, inliers, paths, config,
                            rgb_files=rgb_test,swir_files=swir_test)

            #step_4_diagnostic(container, Homography, inliers, paths, config)
            
        # Afficher résumé
        print_results_summary(paths, config)
        global_end = time.time()

        total_time = global_end - global_start

        fps_total = len(rgb_test) / total_time if total_time > 0 else 0

        print(f"\nTemps total pipeline : {total_time:.2f} s")
        print(f"FPS global : {fps_total:.2f}")

        runtime_file = os.path.join(paths['out_dir'], "runtime_metrics.txt")
        with open(runtime_file, "w") as f:
            f.write("===== RUNTIME METRICS =====\n\n")
            f.write(f"Train images        : {len(rgb_train)}\n")
            f.write(f"Test images         : {len(rgb_test)}\n\n")
            f.write(f"Temps extraction    : {extract_time:.4f} s\n")
            f.write(f"FPS extraction      : {fps_extract:.4f}\n\n")
            f.write(f"Temps homographie   : {h_time:.4f} s\n")
            f.write(f"FPS homographie     : {fps_h:.4f}\n\n")
            f.write(f"Temps application   : {apply_time:.4f} s\n")
            f.write(f"FPS application     : {fps_apply:.4f}\n\n")
            f.write(f"Temps total         : {total_time:.4f} s\n")
            f.write(f"FPS global          : {fps_total:.4f}\n")
        print(f"Runtime sauvegardé : {runtime_file}")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERREUR PIPELINE")
        print(f"{'='*80}")
        print(f"Message : {str(e)}")
        print(f"\nVérifiez :")
        print(f"  1. Les chemins images (rgb_dir, swir_dir)")
        print(f"  2. La qualité des images")
        print(f"  3. Les paramètres de configuration")
        print()
        raise


if __name__ == "__main__":
    main()



"""
# Étape par étape
make extract      BAG=transfer_11680406_files_36b18a31/objectif_12mm_5_arret
make pair         BAG=transfer_11680406_files_36b18a31/objectif_12mm_5_arret/Sortie_Extraite
make run          BAG=transfer_11680406_files_36b18a31/objectif_12mm_5_arret
make fuse         BAG=transfer_11680406_files_36b18a31/objectif_12mm_5_arret
make bench        BAG=transfer_11680406_files_36b18a31/objectif_12mm_5_arret
make plot         BAG=transfer_11680406_files_36b18a31/objectif_12mm_5_arret

# Évaluation seule (fuse + bench + plot)
make evaluate     BAG=transfer_11680406_files_36b18a31/objectif_12mm_5_arret

# Tout d'un coup
make all          BAG=transfer_11680406_files_36b18a31/objectif_12mm_5_arret
"""
