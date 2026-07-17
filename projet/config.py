import os
import re

class Config:
    """Configuration patch-based avec grille chevauchante"""
    
    # ══════════════════════════════════════════════════════════════
    # HARDWARE
    # ══════════════════════════════════════════════════════════════
    DEVICE = "cuda"  # "mps", "cuda", ou "cpu"
    AUTO_PRE_CROP_BOTTOM = False
    AUTO_FILTER_BAD_BOTTOM = False
    AUTO_BOTTOM_N_BANDS = 20
    AUTO_BOTTOM_MIN_MATCHES = 10
    AUTO_BOTTOM_ERROR_FACTOR = 2.0
    AUTO_CROP_COMMON_REGION = True
    AUTO_REMOVE_BOTTOM_VEHICLE = False
    AUTO_REMOVE_BOTTOM_VEHICLE_PRE_H = True
    VEHICLE_MARGIN = 18
    VEHICLE_SMOOTH_KERNEL = 31
    VEHICLE_GRADIENT_THRESH_FACTOR = 2.5
    SAVE_PRE_H_CROP_DEBUG = True
    PRE_H_DEBUG_DIR = "debug_pre_h_crop"
    VEHICLE_EXTRA_MARGIN_PX = 80
    VEHICLE_MIN_KEEP_RATIO = 0.55
    FILTER_SKY_MATCHES_PRE_H = True
    AUTO_FILTER_BAD_SPATIAL = False
    SPATIAL_FILTER_ROWS = 4
    SPATIAL_FILTER_COLS = 4
    SPATIAL_FILTER_MIN_MATCHES = 8
    SPATIAL_FILTER_ERROR_FACTOR = 2.0
    SPATIAL_FILTER_MIN_THRESHOLD = 5.0
    BALANCE_MATCHES_BY_REGION = True
    BALANCE_NUM_REGIONS_X = 3
    BALANCE_NUM_REGIONS_Y = 2
    BALANCE_MAX_MATCHES_PER_REGION = 180
    BALANCE_MIN_MATCHES_PER_REGION = 20

    ALIGN_DIRECTION = "SWIR_TO_VIS"
    # ══════════════════════════════════════════════════════════════
    # TRAITEMENT IMAGES
    # ══════════════════════════════════════════════════════════════

    CLAHE_RADIAL_ESTIMATION_VIS = False
    CLAHE_RADIAL_ESTIMATION_SWIR = False

    APPLY_DISTORTION_VIS = True
    APPLY_DISTORTION_SWIR = True
    COMPUTE_CANNY = False

    NUM_IMAGES = 20 # Nombre d'images à traiter
    NUM_IMAGES_TO_WARP = 500
    CROP_BOTTOM_PIXEL_VIS = 0
    
    # ══════════════════════════════════════════════════════════════
    # EXTRACTION FEATURES - GRILLE CHEVAUCHANTE
    # ══════════════════════════════════════════════════════════════
    
    # Grille overlap (méthode moderne)
    GRID_ROWS = 2  # Nombre de lignes dans la grille
    GRID_COLS = 3  # Nombre de colonnes
    BATCH_SIZE = 36
    GRID_OVERLAP_PERCENT = 30  # Chevauchement entre cellules (%)
    
    # Points par zone
    NUM_POINT_LG = 300 # Points de base pour cellules centrales
    CORNER_MULTIPLIER = 1  # Multiplicateur pour les coins
    EDGE_MULTIPLIER = 1  # Multiplicateur pour les bords
    CENTER_MULTIPLIER = 1  # Multiplicateur pour le centre
    

    # ══════════════════════════════════════════════════════════════
    # MAGSAC++ - ESTIMATION HOMOGRAPHIE ROBUSTE
    # ══════════════════════════════════════════════════════════════
    USE_MAGSAC = True  # True = MAGSAC++, False = RANSAC classique
    
    MAGSAC_CONFIDENCE = 0.90  # Confiance (équivalent RANSAC_STOP_PROB)
    MAGSAC_MAX_ITERS = 10000  # Itérations max (plus élevé que RANSAC)
    MAGSAC_THRESHOLD = 6.0  # Seuil initial (pixels) - sera adapté automatiquement
    MAGSAC_SIGMA = 1  # Écart-type du bruit (pixels) - pour adaptation

    # ══════════════════════════════════════════════════════════════
    # RANSAC - ESTIMATION HOMOGRAPHIE
    # ══════════════════════════════════════════════════════════════
    RANSAC_MIN_SAMPLES = 4  # Minimum de points pour estimer homographie
    RANSAC_THRESHOLD = 8.0  # Seuil résiduel (pixels) - tolérant pour multimodal
    RANSAC_MAX_TRIALS = 3000  # Nombre max d'itérations
    RANSAC_STOP_PROB = 0.9999  # Probabilité d'arrêt
    
    # ══════════════════════════════════════════════════════════════
    # VALIDATION MODÈLE
    # ══════════════════════════════════════════════════════════════
    COND_NUMBER_THRESHOLD = 1e10  # Seuil condition number
    MIN_CELL_SIZE = 50  # Taille minimale patch (px)
    
    # ══════════════════════════════════════════════════════════════
    # SIFT FALLBACK (zones difficiles)
    # ══════════════════════════════════════════════════════════════
    SIFT_N_FEATURES_MULT = 2  # Multiplicateur features SIFT
    SIFT_CONTRAST_THRESHOLD = 0.01  # Seuil contraste SIFT
    SIFT_EDGE_THRESHOLD = 20  # Seuil edge SIFT
    SIFT_SIGMA = 1.2  # Sigma SIFT
    SIFT_LOWE_RATIO = 0.85  # Ratio de Lowe (permissif)
    
    # ══════════════════════════════════════════════════════════════
    # MODE DEBUG ET I/O
    # ══════════════════════════════════════════════════════════════
    DEBUG = False  # Active les diagnostics et visualisations
    LOAD_HOMOGRAPHY = False
      # Charger homographie existante

    # ══════════════════════════════════════════════════════════════
    # CHEMINS (seront overridés par setup_paths)
    # ══════════════════════════════════════════════════════════════
    '''
    @staticmethod
    def setup_paths(script_dir=None, bag_path=None):
        """Configure les chemins d'entrée/sortie"""
        if script_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))

        project_root = os.path.dirname(script_dir)

        if bag_path is not None:
            base_data_path = bag_path
        else:
            base_data_path = os.path.join(
                project_root,
                "datas/transfer_11680406_files_36b18a31/objectif_12mm_3_arret/Sortie_Extraite/paired_images/"
            )

        print(base_data_path)

        paths = {
            'out_dir':  os.path.join(base_data_path, "patch_RANSAC_HOMOGRAPHY_calibration/"),
            'rgb_dir':  os.path.join(base_data_path, "VISIBLE"),
            'swir_dir': os.path.join(base_data_path, "SWIR"),
            'homography_path': os.path.join(project_root,
                                           "datas/transfer_11680406_files_36b18a31/objectif_12mm_4_boucle/Sortie_Extraite/paired_images/patch_RANSAC_HOMOGRAPHY_calibration/homography.npz")
        }

        os.makedirs(paths['out_dir'], exist_ok=True)
        os.makedirs(os.path.join(paths['out_dir'], "split_images"), exist_ok=True)
        os.makedirs(os.path.join(paths['out_dir'], "concatenate_images"), exist_ok=True)

        return paths
    '''



    @staticmethod
    def _extract_seq_name(path: str) -> str | None:
        """Extrait le nom de séquence (dossier avant /Sortie_Extraite/) depuis un chemin."""
        m = re.search(r'[\\/]([^\\/]+)[\\/]Sortie_Extraite', path)
        return m.group(1) if m else None

    @staticmethod
    def setup_paths(script_dir=None, bag_path=None, homography_path=None):
        if script_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))

        project_root = os.path.dirname(script_dir)

        if bag_path is not None:
            base_data_path = bag_path
        else:
            base_data_path = os.path.join(
                project_root,
                "datas/transfer_11680406_files_36b18a31/objectif_12mm_4_arret/Sortie_Extraite/paired_images/"
            )

        if homography_path is None:
            homography_path = os.path.join(
                project_root,
                "datas/transfer_11680406_files_36b18a31/objectif_12mm_4_boucle/Sortie_Extraite/paired_images/"
                "patch_RANSAC_HOMOGRAPHY_calibration/homography.npz"
            )

        # ── Détection de séquence croisée ─────────────────────────────────
        seq_data = Config._extract_seq_name(base_data_path)
        seq_homo = Config._extract_seq_name(homography_path)

        direction = getattr(Config, "ALIGN_DIRECTION", "VIS_TO_SWIR")

        if direction == "VIS_TO_SWIR":
            direction_tag = "VIS_to_SWIR"
        elif direction == "SWIR_TO_VIS":
            direction_tag = "SWIR_to_VIS"
        else:
            raise ValueError(f"ALIGN_DIRECTION invalide : {direction}")

        out_dir_name = f"patch_RANSAC_HOMOGRAPHY_calibration_{direction_tag}"

        if seq_homo and seq_data and seq_homo != seq_data:
            out_dir_name += f"_H_from_{seq_homo}"

        # ── Construction des chemins ───────────────────────────────────────
        paths = {
            'out_dir':          os.path.join(base_data_path, out_dir_name + "/"),
            'rgb_dir':          os.path.join(base_data_path, "VISIBLE"),
            'swir_dir':         os.path.join(base_data_path, "SWIR"),
            'homography_path':  homography_path,
        }

        for sub in ("", "split_images", "concatenate_images"):
            os.makedirs(os.path.join(paths['out_dir'], sub), exist_ok=True)

        print(f"[paths] data_seq   : {seq_data}")
        print(f"[paths] homo_seq   : {seq_homo}")
        print(f"[paths] out_dir    : {paths['out_dir']}")

        return paths


    def __init__(self):
        self._extractor = None
        self._matcher   = None

    @property
    def extractor(self):
        if self._extractor is None:
            from lightglue import SuperPoint
            print(f"[Config] Instanciation SuperPoint (device={self.DEVICE})")
            self._extractor = SuperPoint(max_num_keypoints=self.NUM_POINT_LG).eval().to(self.DEVICE)
        return self._extractor

    @property
    def matcher(self):
        if self._matcher is None:
            from lightglue import LightGlue
            print(f"[Config] Instanciation LightGlue (device={self.DEVICE})")
            self._matcher = LightGlue(features='superpoint').eval().to(self.DEVICE)
        return self._matcher

    def release_models(self):
        """Libère les modèles si nécessaire (fin de pipeline)."""
        self._extractor = None
        self._matcher   = None
