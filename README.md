# Images Registration Project

> Pipeline de recalage multimodal **VIS / SWIR** basé sur SuperPoint, LightGlue et MAGSAC.

---

## Table des matières

- [Images Registration Project](#images-registration-project)
  - [Table des matières](#table-des-matières)
  - [Vue d'ensemble](#vue-densemble)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Utilisation](#utilisation)
    - [Via le Makefile](#via-le-makefile)
    - [Directement en Python](#directement-en-python)
  - [Structure du projet](#structure-du-projet)
  - [Documentation API](#documentation-api)
    - [Générer la doc localement](#générer-la-doc-localement)
  - [Pipeline — logique générale](#pipeline--logique-générale)
  - [Dossier `datas/`](#dossier-datas)

---

## Vue d'ensemble

Ce projet implémente un pipeline complet de recalage d'images multimodales VIS (visible) et SWIR (short-wave infrared).  
Il couvre l'ensemble de la chaîne : prétraitement, extraction de features, matching, estimation robuste de l'homographie, application et évaluation quantitative.

---

## Installation

```bash
git clone <url-du-repo>
cd Images_Registration_Project

python -m venv venv
source venv/bin/activate       # Windows : venv\Scripts\activate

pip install -r requirements.txt
```

> **Prérequis :** Python ≥ 3.11, PyTorch avec support MPS (Apple Silicon) ou CUDA.

---

## Configuration

Tous les hyperparamètres sont centralisés dans `projet/config.py` :

| Paramètre | Description |
|---|---|
| `GRID_ROWS / GRID_COLS` | Découpage en patchs |
| `GRID_OVERLAP_PERCENT` | Recouvrement entre patchs |
| `NUM_POINT_LG` | Nombre de points LightGlue |
| `RANSAC_THRESHOLD` | Seuil de reprojection MAGSAC |
| `APPLY_DISTORTION_VIS/SWIR` | Activation de la correction de distorsion |
| `DEVICE` | Device PyTorch (`mps`, `cuda`, `cpu`) |

Modifier `config.py` suffit à changer le comportement du pipeline sans toucher au code des modules.

---

## Utilisation

### Via le Makefile
Toutes les commandes prennent un argument BAG= correspondant au chemin relatif depuis datas/.

Workflow complet sur une nouvelle séquence

```bash
# 1. Extraire les images depuis un bag ROS2 (.zip ou dossier) depuis son chemin relatif à datas
make extract BAG=transfer_11680406_files_36b18a31/objectif_12mm_1_arret.zip

# 2. Apparier temporellement les images SWIR / VISIBLE
make pair BAG=transfer_11680406_files_36b18a31/objectif_12mm_1_arret/Sortie_Extraite

# 3. Lancer le pipeline de recalage (extraction + estimation homographie + application)
make run BAG=transfer_11680406_files_36b18a31/objectif_12mm_2_vg

# 4. Évaluer la qualité du recalage (GOA / Edge Alignment / MIND-SSD) équivalent à un fuse/bench/plot
make evaluate BAG=transfer_11680406_files_36b18a31/objectif_12mm_2_vg/Sortie_Extraite/paired_images/patch_RANSAC_HOMOGRAPHY_calibration
```

### Directement en Python

```bash
# Pipeline complet
cd projet/
python main.py

# Benchmark qualité sur un dossier split_images/
python -m registration.evaluation.quality_bench \
    transfer_11680406_files_36b18a31/objectif_12mm_2_vg/Sortie_Extraite/paired_images/patch_RANSAC_HOMOGRAPHY_calibration

# Appariement temporel SWIR / VISIBLE depuis un bag ROS2
python -m registration.temporal.recalage_temporel \
    transfer_11680406_files_36b18a31/objectif_12mm_1_arret/Sortie_Extraite

# Fusion HSV
python -m registration.fusion.fusion_HSV \
    objectif_12mm_1_arret/Sortie_Extraite/paired_images/patch_RANSAC/split_images
```

> Tous les chemins sont relatifs à `datas/`. La racine du projet est détectée automatiquement.

---

## Structure du projet

```text
Images_Registration_Project/
├── projet/
│   ├── main.py                  # Point d'entrée principal
│   ├── config.py                # Configuration centralisée
│   ├── Makefile                 # Commandes de lancement
│   ├── requirements.txt
│   └── registration/
│       ├── pipeline/
│       │   ├── extraction.py    # Extraction des correspondances (patches + LightGlue)
│       │   ├── estimation.py    # Estimation homographie (MAGSAC)
│       │   ├── application.py   # Application de la transformation
│       │   ├── diagnostics.py   # Erreurs de reprojection, visualisations
│       │   ├── calibration.py   # Calibration distorsion
│       │   └── metrics.py       # Métriques qualité (GOA, Edge Alignment, MIND-SSD)
│       ├── preprocessing/
│       │   ├── contrast.py      # CLAHE, normalisation, Z-score
│       │   └── undistortion.py  # Correction de distorsion, remap
│       ├── io/
│       │   ├── image_loader.py  # Chargement et mise en tensor des images
│       │   └── homography_io.py # Sérialisation / désérialisation .npz
│       ├── data/
│       │   └── sequence_matches_container.py  # Conteneur de correspondances
│       ├── matching/
│       │   ├── lightglue_matcher.py  # Matching SuperPoint + LightGlue
│       │   └── sift_matcher.py       # Fallback SIFT permissif
│       ├── evaluation/
│       │   ├── quality_bench.py      # Benchmark GOA / Edge Alignment / MIND
│       │   ├── plot_quality.py       # Graphiques métriques
│       │   └── num_images_benchmark.py  # Benchmark temporel
│       ├── fusion/
│       │   └── fusion_HSV.py    # Fusion HSV VIS + SWIR
│       ├── temporal/
│       │   └── recalage_temporel.py  # Appariement temporel SWIR / VISIBLE
│       └── utils/
│           ├── visualization.py     # Overlays, figures de debug
│           └── profiling_utils.py   # Profilage des étapes du pipeline
├── tools/
│   ├── runner.py                # Exécution en série + export CSV
│   └── ros2_bag_extractor.py    # Extraction d'images depuis bags ROS2
└── datas/                       # Données et résultats (non versionné)
```

---

## Documentation API

La documentation complète des modules est générée avec [pdoc](https://pdoc.dev).

### Générer la doc localement

```bash
pip install pdoc

# Depuis le dossier projet
pdoc registration --output-dir docs/

open docs/index.html    # macOS
xdg-open docs/index.html  # Linux
```

La doc couvre l'ensemble du package `registration/` :
- signatures typées et docstrings de chaque fonction
- navigation entre modules
- liens vers le code source

---

## Pipeline — logique générale

```
Images VIS + SWIR
       │
       ▼
  Prétraitement (contraste, distorsion, normalisation)
       │
       ▼
  Découpage en patchs (grille avec recouvrement)
       │
       ▼
  Extraction features — SuperPoint + LightGlue
       │
       ▼
  Accumulation des correspondances (SequenceMatchesContainer)
       │
       ▼
  Estimation homographie — cv2.USAC_MAGSAC
       │
       ▼
  Application à la séquence complète
       │
       ▼
  Métriques & diagnostics (GOA / Edge Alignment / MIND-SSD)
```

---

## Dossier `datas/`

Le dossier `datas/` est ignoré par Git (`.gitignore`) mais sa présence est conservée via `datas/.gitkeep`.

Structure attendue après extraction :

```text
datas/
└── <nom_sequence>/
    └── Sortie_Extraite/
        ├── camera_AV/          # Images VIS (.tiff)
        ├── camera_SWIR/        # Images SWIR (.tiff)
        └── paired_images/
            └── <méthode>/
                ├── split_images/    # Paires VIS + SWIR alignées
                ├── homography.npz   # Homographie sauvegardée
                ├── fusion_HSV/      # Images fusionnées
                └── results/         # Scores CSV + MIND maps
```

Pour reconstruire cette structure depuis un bag ROS2 :

```bash
make extract BAG=<chemin_vers_le_bag>.zip
make pair    BAG=<chemin_vers_le_bag>/Sortie_Extraite
```
