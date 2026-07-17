import cv2
import argparse
import numpy as np
from pathlib import Path
from ..pipeline.metrics import _to_uint8


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
VIS_KEYWORD    = "vis"
SWIR_KEYWORD   = "swir_aligned"

H_CANDIDATES   = ["homography.npz", "H_general.npy", "H_general.npz"]


def load_images_by_keyword(directory: Path, keyword: str) -> dict[str, np.ndarray]:
    """
    Charge toutes les images d'un répertoire dont le nom de fichier contient un mot-clé.

    Args:
        directory: Chemin vers le dossier à parcourir.
        keyword:   Sous-chaîne recherchée dans le stem du fichier.

    Returns:
        Dictionnaire {stem: image_bgr} pour chaque image valide trouvée.
    """
    imgs = {}
    for f in sorted(directory.iterdir()):
        if f.suffix.lower() in IMG_EXTENSIONS and keyword in f.stem:
            img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if img is not None:
                imgs[f.stem] = img
            else:
                print(f"  ⚠ Impossible de lire : {f.name}")
    return imgs


def extract_pair_id(stem: str, keyword: str) -> str:
    """
    Extrait l'identifiant de paire en supprimant le mot-clé et les séparateurs
    du nom de fichier (stem).

    Args:
        stem:    Nom de fichier sans extension.
        keyword: Sous-chaîne à retirer.

    Returns:
        Identifiant de paire nettoyé (str).
    """
    return stem.replace(keyword, "").strip("_- ")


def build_pairs(vis_imgs, swir_imgs):
    """
    Construit les paires VIS/SWIR en faisant correspondre les identifiants de paire.
    Signale les images sans correspondance.

    Args:
        vis_imgs:  Dictionnaire {stem: image} pour les images VIS.
        swir_imgs: Dictionnaire {stem: image} pour les images SWIR.

    Returns:
        Liste de tuples (pair_id, (vis_stem, vis_img), (swir_stem, swir_img)).
    """
    vis_map  = {extract_pair_id(s, VIS_KEYWORD):  (s, img) for s, img in vis_imgs.items()}
    swir_map = {extract_pair_id(s, SWIR_KEYWORD): (s, img) for s, img in swir_imgs.items()}

    common = sorted(set(vis_map) & set(swir_map))
    pairs  = [(pid, vis_map[pid], swir_map[pid]) for pid in common]

    missing_vis  = set(swir_map) - set(vis_map)
    missing_swir = set(vis_map)  - set(swir_map)
    if missing_vis:
        print(f"  ⚠ Pas de VIS pour : {missing_vis}")
    if missing_swir:
        print(f"  ⚠ Pas de SWIR pour : {missing_swir}")
    return pairs


def to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Convertit une image en uint8 par normalisation min-max sur [0, 255].
    Si l'image est déjà uint8, elle est retournée sans modification.

    Args:
        img: Image numpy de dtype quelconque.

    Returns:
        Image numpy uint8.
    """
    if img.dtype == np.uint8:
        return img
    img_f = img.astype(np.float32)
    img_f = (img_f - img_f.min()) / (img_f.max() - img_f.min() + 1e-8) * 255.0
    return img_f.astype(np.uint8)


def build_vis_mask(vis_bgr: np.ndarray, erode_px: int = 8) -> np.ndarray:
    """
    Génère un masque binaire des pixels valides de l'image VIS (non noirs),
    puis l'érode pour écarter les bords.

    Args:
        vis_bgr:  Image VIS en BGR (uint8).
        erode_px: Taille du noyau d'érosion en pixels.

    Returns:
        Masque uint8 (255 = valide, 0 = invalide).
    """
    gray = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    kernel  = np.ones((erode_px, erode_px), np.uint8)
    return cv2.erode(mask, kernel, iterations=1)


def fuse_hsv(vis_bgr: np.ndarray, swir_raw: np.ndarray, H_mat: np.ndarray) -> np.ndarray:
    """
    Fusionne une image VIS et une image SWIR dans l'espace HSV :
    teinte (H) et saturation (S) issues du VIS, luminance (V) issue du SWIR aligné.

    Args:
        vis_bgr:  Image VIS BGR uint8.
        swir_raw: Image SWIR brute (tout dtype, mono ou BGR).
        H_mat:    Matrice d'homographie 3×3 pour aligner le SWIR sur le VIS.

    Returns:
        Image fusionnée BGR uint8.
    """
    h, w = vis_bgr.shape[:2]

    swir_u8 = _to_uint8(swir_raw)
    if swir_u8.ndim == 3:
        swir_u8 = cv2.cvtColor(swir_u8, cv2.COLOR_BGR2GRAY)
    swir_warped = cv2.warpPerspective(swir_u8, H_mat, (w, h), flags=cv2.INTER_LINEAR)

    vis_hsv       = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2HSV)
    H_ch, S_ch, _ = cv2.split(vis_hsv)
    vis_mask      = build_vis_mask(vis_bgr, erode_px=8)
    S_ch_masked   = cv2.bitwise_and(S_ch, S_ch, mask=vis_mask)

    fused_hsv = cv2.merge([H_ch, S_ch_masked, swir_warped])
    return cv2.cvtColor(fused_hsv, cv2.COLOR_HSV2BGR)


def load_homography(calib_dir: Path) -> np.ndarray:
    """
    Recherche et charge automatiquement une matrice d'homographie depuis
    un répertoire de calibration (.npz ou .npy). Retourne l'identité si
    aucun fichier n'est trouvé.

    Args:
        calib_dir: Dossier contenant le fichier d'homographie.

    Returns:
        Matrice d'homographie 3×3 float64.
    """
    for name in H_CANDIDATES:
        h_path = calib_dir / name
        if not h_path.exists():
            continue
        data = np.load(str(h_path))
        if isinstance(data, np.ndarray):          # .npy
            return data.astype(np.float64)
        for key in ("H", "homography", list(data.keys())[0]):  # .npz
            if key in data:
                return data[key].astype(np.float64)

    # Aucun fichier trouvé → identité + avertissement
    print(f"  ⚠ Aucun fichier H trouvé dans {calib_dir} — utilisation de l'identité")
    return np.eye(3, dtype=np.float64)


def main() -> None:
    """
    Point d'entrée CLI. Lit un dossier d'images VIS+SWIR alignées,
    charge ou détermine l'homographie, effectue la fusion HSV sur chaque
    paire et écrit les résultats dans le dossier de sortie.
    """
    parser = argparse.ArgumentParser(description="Fusion HSV VIS+SWIR")
    parser.add_argument(
        "input",
        help="Chemin relatif depuis datas/ vers le dossier split_images",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Dossier de sortie (défaut: <dossier_calibration>/fusion_HSV)",
    )
    args = parser.parse_args()

    projet_root = Path(__file__).resolve().parents[3]
    input_dir   = (projet_root / "datas" / args.input).resolve()
    calib_dir   = input_dir.parent          # dossier de calibration contenant H + split_images
    output_dir  = Path(args.output).resolve() if args.output else calib_dir / "fusion_HSV"

    if not input_dir.exists():
        print(f"[ERREUR] Dossier introuvable : {input_dir}")
        raise SystemExit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    H_mat = load_homography(calib_dir)

    print(f"📂 Images      : {input_dir}")
    print(f"📂 Calibration : {calib_dir}")
    print(f"📂 Sortie      : {output_dir}")

    vis_imgs  = load_images_by_keyword(input_dir, VIS_KEYWORD)
    swir_imgs = load_images_by_keyword(input_dir, SWIR_KEYWORD)
    print(f"  → {len(vis_imgs)} image(s) VIS, {len(swir_imgs)} image(s) SWIR")

    pairs = build_pairs(vis_imgs, swir_imgs)
    if not pairs:
        print("❌ Aucune paire trouvée. Vérifie VIS_KEYWORD / SWIR_KEYWORD.")
        return

    print(f"\n✓ {len(pairs)} paire(s) — fusion en cours :")
    for pair_id, (vis_stem, vis_img), (swir_stem, swir_img) in pairs:
        fused    = fuse_hsv(vis_img, swir_img, H_mat)
        out_name = f"fusion_{pair_id}.png"
        cv2.imwrite(str(output_dir / out_name), fused)
        print(f"  [{pair_id}]  {vis_stem} + {swir_stem}  →  {out_name}")

    print(f"\n✅ Résultats dans : {output_dir}")


if __name__ == "__main__":
    main()
