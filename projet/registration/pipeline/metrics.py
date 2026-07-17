'''
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


# ─── Paramètres ──────────────────────────────────────────────────────────────
MASK_EROSION_PX  = 8
BLUR_SIGMA_VIS   = 1.0
GOA_SOBEL_KSIZE  = 3
EDGE_CANNY_LOW   = 50
EDGE_CANNY_HIGH  = 150
MIND_SIGMA       = 1.5
MIND_BORDER_ERODE = 3


# ─── Utilitaires image ────────────────────────────────────────────────────────
def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    f = img.astype(np.float32)
    f = (f - f.min()) / (f.max() - f.min() + 1e-8) * 255.0
    return f.astype(np.uint8)


def _to_gray(img: np.ndarray) -> np.ndarray:
    """
    Convertit une image (tout dtype, mono ou BGR) en niveaux de gris uint8.

    Args:
        img: Image numpy.

    Returns:
        Image niveaux de gris uint8 (H, W).
    """

    u8 = _to_uint8(img)
    if u8.ndim == 2:
        return u8
    return cv2.cvtColor(u8, cv2.COLOR_BGR2GRAY)


def _build_valid_mask(swir_warped_gray: np.ndarray, erode_px: int) -> np.ndarray:
    """
    Construit un masque des pixels valides (non noirs) de l'image SWIR
    après warping, puis l'érode pour réduire les artefacts de bord.

    Args:
        swir_warped_gray: Image SWIR warpée en niveaux de gris.
        erode_px:         Taille du noyau d'érosion.

    Returns:
        Masque binaire uint8 (255 = valide).
    """
    _, mask = cv2.threshold(swir_warped_gray, 5, 255, cv2.THRESH_BINARY)
    if erode_px > 0:
        k = np.ones((erode_px, erode_px), np.uint8)
        mask = cv2.erode(mask, k, iterations=1)
    return mask


# ─── Prétraitement ────────────────────────────────────────────────────────────
def preprocess_pair(vis_bgr, swir_raw, H):
    """
    Prépare la paire VIS/SWIR pour le calcul des métriques : conversion
    niveaux de gris, warping du SWIR, flou gaussien optionnel sur le VIS,
    et calcul du masque de validité.

    Args:
        vis_bgr:  Image VIS BGR (tout dtype).
        swir_raw: Image SWIR brute (tout dtype).
        H:        Matrice d'homographie 3×3.

    Returns:
        Tuple (vis_gray, swir_warped, valid_mask) en uint8.
    """

    h, w = vis_bgr.shape[:2]
    swir_u8     = _to_gray(swir_raw)
    swir_warped = cv2.warpPerspective(swir_u8, H, (w, h), flags=cv2.INTER_LINEAR)
    valid_mask  = _build_valid_mask(swir_warped, MASK_EROSION_PX)

    vis_gray = _to_gray(vis_bgr)
    if BLUR_SIGMA_VIS > 0.0:
        ksize   = 2 * int(np.ceil(BLUR_SIGMA_VIS * 3)) + 1
        vis_gray = cv2.GaussianBlur(vis_gray, (ksize, ksize), BLUR_SIGMA_VIS)

    return vis_gray, swir_warped, valid_mask


def _shift_no_wrap(img, dx, dy):
    """
    Décale une image de (dx, dy) pixels sans repliement (remplissage par zéros).

    Args:
        img: Image numpy 2D ou 3D.
        dx:  Décalage horizontal (positif = droite).
        dy:  Décalage vertical (positif = bas).

    Returns:
        Image décalée, même shape que l'entrée.
    """
    out = np.zeros_like(img)
    h, w = img.shape[:2]
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    xd0, xd1 = max(0, dx),  min(w, w + dx)
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    yd0, yd1 = max(0, dy),  min(h, h + dy)
    out[yd0:yd1, xd0:xd1] = img[ys0:ys1, xs0:xs1]
    return out



# ─── Calcul global (une seule image) ─────────────────────────────────────────
def compute_all_metrics(vis_bgr, swir_raw, H) -> dict:
    """
    Point d'entrée unique pour évaluer la qualité du recalage sur une paire.
    Enchaîne le prétraitement et le calcul des trois métriques.

    Args:
        vis_bgr:  Image VIS BGR.
        swir_raw: Image SWIR brute.
        H:        Matrice d'homographie 3×3.

    Returns:
        Dictionnaire {"goa": float, "edge_f1_8px": float, "mind_ssd": float}.
    """
    vis_gray, swir_warped, valid_mask = preprocess_pair(vis_bgr, swir_raw, H)
    return {
        "goa":      compute_goa(vis_gray, swir_warped, valid_mask),
        "edge_f1_8px": compute_edge_alignment_metrics(vis_gray, swir_warped, valid_mask)["edge_f1_8px"],
        "mind_ssd": compute_mind(vis_gray, swir_warped, valid_mask),
    }


# ─── Métriques BENCH ────────────────────────────────────────────────────────────────

def compute_goa(vis_gray, swir_gray, mask) -> float:
    gxv = cv2.Sobel(vis_gray,  cv2.CV_64F, 1, 0, ksize=GOA_SOBEL_KSIZE)
    gyv = cv2.Sobel(vis_gray,  cv2.CV_64F, 0, 1, ksize=GOA_SOBEL_KSIZE)
    gxs = cv2.Sobel(swir_gray, cv2.CV_64F, 1, 0, ksize=GOA_SOBEL_KSIZE)
    gys = cv2.Sobel(swir_gray, cv2.CV_64F, 0, 1, ksize=GOA_SOBEL_KSIZE)
    cos2   = np.cos(np.arctan2(gyv, gxv) - np.arctan2(gys, gxs)) ** 2
    weight = np.minimum(np.sqrt(gxv**2 + gyv**2), np.sqrt(gxs**2 + gys**2))
    roi    = mask > 0
    w_sum  = weight[roi].sum()
    return float("nan") if w_sum < 1e-8 else float((cos2[roi] * weight[roi]).sum() / w_sum)


def compute_edge_distance_legacy(vis_gray, swir_gray, mask) -> float:
    mask_u8 = (mask > 0).astype(np.uint8)
    edges_v = cv2.bitwise_and(cv2.Canny(vis_gray,  EDGE_CANNY_LOW, EDGE_CANNY_HIGH),
                               cv2.Canny(vis_gray,  EDGE_CANNY_LOW, EDGE_CANNY_HIGH), mask=mask_u8)
    edges_s = cv2.bitwise_and(cv2.Canny(swir_gray, EDGE_CANNY_LOW, EDGE_CANNY_HIGH),
                               cv2.Canny(swir_gray, EDGE_CANNY_LOW, EDGE_CANNY_HIGH), mask=mask_u8)
    ev, es = edges_v > 0, edges_s > 0
    if ev.sum() == 0 or es.sum() == 0:
        return float("nan")
    dt_v = distance_transform_edt(~ev)
    dt_s = distance_transform_edt(~es)
    return float(dt_v[es].mean() + dt_s[ev].mean()) / 2.0


def _shift_nowrap(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = np.zeros_like(img)
    h, w = img.shape[:2]
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    xd0, xd1 = max(0,  dx), min(w, w + dx)
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    yd0, yd1 = max(0,  dy), min(h, h + dy)
    out[yd0:yd1, xd0:xd1] = img[ys0:ys1, xs0:xs1]
    return out


def _mind_descriptor(img_gray: np.ndarray, sigma: float) -> np.ndarray:
    imgf  = img_gray.astype(np.float32)
    ksize = max(2 * int(np.ceil(sigma * 3)) + 1, 3)
    Dp    = np.stack(
        [cv2.GaussianBlur((imgf - _shift_nowrap(imgf, dx, dy)) ** 2, (ksize, ksize), sigma)
         for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]],
        axis=-1,
    ).astype(np.float32)
    V     = Dp.mean(axis=-1)
    vmean = float(V.mean()) + 1e-8
    V     = np.clip(V, 1e-3 * vmean, 1e3 * vmean)[..., np.newaxis]
    mind  = np.exp(-Dp / V)
    return mind #/ np.maximum(mind.max(axis=-1, keepdims=True), 1e-8)


def _mind_ssd(vis_gray, swir_gray, mask) -> tuple[np.ndarray, np.ndarray]:
    mind_v = _mind_descriptor(vis_gray,  MIND_SIGMA)
    mind_s = _mind_descriptor(swir_gray, MIND_SIGMA)
    ssd    = ((mind_v - mind_s) ** 2).mean(axis=-1)
    if MIND_BORDER_ERODE > 0:
        k    = 2 * MIND_BORDER_ERODE + 1
        mask = cv2.erode(mask, np.ones((k, k), np.uint8), iterations=1)
    return ssd, mask


def compute_mind(vis_gray, swir_gray, mask) -> float:
    ssd, mask = _mind_ssd(vis_gray, swir_gray, mask)
    roi       = mask > 0
    return float("nan") if roi.sum() == 0 else float(ssd[roi].mean())


def compute_mind_map(vis_gray, swir_gray, mask) -> np.ndarray:
    ssd, mask      = _mind_ssd(vis_gray, swir_gray, mask)
    ssd[mask == 0] = np.nan
    return ssd
'''


import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


# ─── Paramètres ──────────────────────────────────────────────────────────────
MASK_EROSION_PX   = 8
BLUR_SIGMA_VIS    = 1.0
GOA_SOBEL_KSIZE   = 3
EDGE_CANNY_LOW    = 20
EDGE_CANNY_HIGH   = 80
EDGE_MATCH_TOL_PX = 8
MIND_SIGMA        = 1.5
MIND_BORDER_ERODE = 3


# ─── Utilitaires image ────────────────────────────────────────────────────────

def _to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Convertit une image en uint8 par normalisation min-max sur [0, 255].
    Passthrough si l'image est déjà uint8.

    Args:
        img: Image numpy de dtype quelconque.

    Returns:
        Image uint8.
    """
    if img.dtype == np.uint8:
        return img
    f = img.astype(np.float32)
    f = (f - f.min()) / (f.max() - f.min() + 1e-8) * 255.0
    return f.astype(np.uint8)


def _to_gray(img: np.ndarray) -> np.ndarray:
    """
    Convertit une image (tout dtype, mono ou BGR) en niveaux de gris uint8.

    Args:
        img: Image numpy.

    Returns:
        Image niveaux de gris uint8 (H, W).
    """
    u8 = _to_uint8(img)
    if u8.ndim == 2:
        return u8
    return cv2.cvtColor(u8, cv2.COLOR_BGR2GRAY)


def _build_valid_mask(swir_warped_gray: np.ndarray, erode_px: int) -> np.ndarray:
    """
    Construit un masque des pixels valides (non noirs) de l'image SWIR
    après warping, puis l'érode pour réduire les artefacts de bord.

    Args:
        swir_warped_gray: Image SWIR warpée en niveaux de gris.
        erode_px:         Taille du noyau d'érosion en pixels.

    Returns:
        Masque binaire uint8 (255 = valide, 0 = invalide).
    """
    _, mask = cv2.threshold(swir_warped_gray, 5, 255, cv2.THRESH_BINARY)
    if erode_px > 0:
        k    = np.ones((erode_px, erode_px), np.uint8)
        mask = cv2.erode(mask, k, iterations=1)
    return mask

import torch
import torch.nn.functional as F
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

_segformer_processor = None
_segformer_model = None

def get_segformer_quality_model(device="cuda"):
    global _segformer_processor, _segformer_model

    if _segformer_processor is None:
        _segformer_processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640"
        )
        _segformer_model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640"
        ).to(device).eval()

    return _segformer_processor, _segformer_model


def remove_sky_from_mask(vis_bgr, valid_mask, device="cuda"):
    img_rgb = cv2.cvtColor(_to_uint8(vis_bgr), cv2.COLOR_BGR2RGB)

    processor, model = get_segformer_quality_model(device)

    inputs = processor(images=img_rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    h, w = img_rgb.shape[:2]

    logits = F.interpolate(
        outputs.logits,
        size=(h, w),
        mode="bilinear",
        align_corners=False
    )

    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    sky_prob = probs[2]
    sky = sky_prob > 0.85

    top = np.zeros_like(sky, dtype=bool)
    top[: int(h * 0.50), :] = True
    sky = sky & top

    new_mask = valid_mask.copy()
    new_mask[sky] = 0

    return new_mask

def _shift_nowrap(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """
    Décale une image de (dx, dy) pixels sans repliement (remplissage par zéros).

    Args:
        img: Image numpy 2D ou 3D.
        dx:  Décalage horizontal (positif = droite).
        dy:  Décalage vertical (positif = bas).

    Returns:
        Image décalée, même shape que l'entrée.
    """
    out = np.zeros_like(img)
    h, w = img.shape[:2]
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    xd0, xd1 = max(0, dx),  min(w, w + dx)
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    yd0, yd1 = max(0, dy),  min(h, h + dy)
    out[yd0:yd1, xd0:xd1] = img[ys0:ys1, xs0:xs1]
    return out


# ─── Prétraitement ────────────────────────────────────────────────────────────

def preprocess_pair(vis_bgr: np.ndarray, swir_raw: np.ndarray, H: np.ndarray) -> tuple:
    """
    Prépare la paire VIS/SWIR pour le calcul des métriques : conversion
    niveaux de gris, warping du SWIR, flou gaussien optionnel sur le VIS,
    et calcul du masque de validité.

    Args:
        vis_bgr:  Image VIS BGR (tout dtype).
        swir_raw: Image SWIR brute (tout dtype), même résolution que vis_bgr.
        H:        Matrice d'homographie 3×3.

    Returns:
        Tuple (vis_gray, swir_warped, valid_mask) en uint8.
    """
    h, w        = vis_bgr.shape[:2]
    swir_u8     = _to_gray(swir_raw)
    swir_warped = cv2.warpPerspective(swir_u8, H, (w, h), flags=cv2.INTER_LINEAR)
    valid_mask  = _build_valid_mask(swir_warped, MASK_EROSION_PX)

    vis_gray = _to_gray(vis_bgr)
    if BLUR_SIGMA_VIS > 0.0:
        ksize    = 2 * int(np.ceil(BLUR_SIGMA_VIS * 3)) + 1
        vis_gray = cv2.GaussianBlur(vis_gray, (ksize, ksize), BLUR_SIGMA_VIS)
    
    valid_mask = remove_sky_from_mask(vis_bgr, valid_mask)
    return vis_gray, swir_warped, valid_mask


# ─── Métriques ────────────────────────────────────────────────────────────────

def compute_goa(vis_gray: np.ndarray, swir_gray: np.ndarray, mask: np.ndarray) -> float:
    """
    Calcule le Gradient Orientation Alignment (GOA) entre les deux images.
    Mesure l'alignement des orientations de gradient, pondéré par la magnitude.

    Args:
        vis_gray:  Image VIS niveaux de gris uint8.
        swir_gray: Image SWIR niveaux de gris uint8 (déjà warpée).
        mask:      Masque de région valide uint8.

    Returns:
        Score GOA dans [0, 1] (1 = alignement parfait), ou nan si le masque est vide.
    """
    gxv    = cv2.Sobel(vis_gray,  cv2.CV_64F, 1, 0, ksize=GOA_SOBEL_KSIZE)
    gyv    = cv2.Sobel(vis_gray,  cv2.CV_64F, 0, 1, ksize=GOA_SOBEL_KSIZE)
    gxs    = cv2.Sobel(swir_gray, cv2.CV_64F, 1, 0, ksize=GOA_SOBEL_KSIZE)
    gys    = cv2.Sobel(swir_gray, cv2.CV_64F, 0, 1, ksize=GOA_SOBEL_KSIZE)
    cos2   = np.cos(np.arctan2(gyv, gxv) - np.arctan2(gys, gxs)) ** 2
    weight = np.minimum(np.sqrt(gxv**2 + gyv**2), np.sqrt(gxs**2 + gys**2))
    roi    = mask > 0
    w_sum  = weight[roi].sum()
    if w_sum < 1e-8:
        return float("nan")
    return float((cos2[roi] * weight[roi]).sum() / w_sum)

def compute_goa_map(vis_gray, swir_gray, mask):

    gxv = cv2.Sobel(vis_gray, cv2.CV_64F, 1, 0, ksize=GOA_SOBEL_KSIZE)
    gyv = cv2.Sobel(vis_gray, cv2.CV_64F, 0, 1, ksize=GOA_SOBEL_KSIZE)

    gxs = cv2.Sobel(swir_gray, cv2.CV_64F, 1, 0, ksize=GOA_SOBEL_KSIZE)
    gys = cv2.Sobel(swir_gray, cv2.CV_64F, 0, 1, ksize=GOA_SOBEL_KSIZE)

    cos2 = np.cos(
        np.arctan2(gyv, gxv) -
        np.arctan2(gys, gxs)
    ) ** 2

    cos2[mask == 0] = np.nan

    return cos2

def adaptive_canny(img, mask, low_pct=90, high_pct=97):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx**2 + gy**2)
    roi = mag[mask > 0]

    if roi.size == 0:
        return np.zeros_like(img, dtype=np.uint8)

    low = np.percentile(roi, low_pct)
    high = np.percentile(roi, high_pct)

    if high <= low:
        high = low + 1

    edges = cv2.Canny(img, int(low), int(high))
    return cv2.bitwise_and(edges, edges, mask=(mask > 0).astype(np.uint8))

def compute_edge_alignment_metrics(
    vis_gray: np.ndarray,
    swir_gray: np.ndarray,
    mask: np.ndarray,
    tol_px: int = EDGE_MATCH_TOL_PX,
) -> dict:
    """
    Calcule des métriques de coïncidence de contours dans une tolérance en pixels.

    Contrairement à une simple distance au contour le plus proche, ces scores comptent la proportion de pixels de contour
    qui trouvent un contour correspondant dans l'autre modalité à moins de tol_px.

    Args:
        vis_gray:  Image VIS niveaux de gris uint8.
        swir_gray: Image SWIR niveaux de gris uint8.
        mask:      Masque de région valide uint8.
        tol_px:    Tolérance de correspondance en pixels.

    Returns:
        Dictionnaire avec precision, recall et F1 des contours.
    """
    edges_v = adaptive_canny(vis_gray, mask, low_pct=90, high_pct=97)
    edges_s = adaptive_canny(swir_gray, mask, low_pct=90, high_pct=97)

    ev, es = edges_v > 0, edges_s > 0
    num_v = int(ev.sum())
    num_s = int(es.sum())
    if num_v == 0 or num_s == 0:
        return {
            f"edge_precision_{tol_px}px": float("nan"),
            f"edge_recall_{tol_px}px": float("nan"),
            f"edge_f1_{tol_px}px": float("nan"),
        }

    dt_v = distance_transform_edt(~ev)
    dt_s = distance_transform_edt(~es)
    precision = float((dt_v[es] <= tol_px).sum() / num_s)
    recall = float((dt_s[ev] <= tol_px).sum() / num_v)
    denom = precision + recall
    f1 = float(2.0 * precision * recall / denom) if denom > 0 else 0.0
    return {
        f"edge_precision_{tol_px}px": precision,
        f"edge_recall_{tol_px}px": recall,
        f"edge_f1_{tol_px}px": f1,
    }

def _mind_descriptor(img_gray: np.ndarray, sigma: float) -> np.ndarray:
    """
    Calcule le descripteur MIND (Modality Independent Neighbourhood Descriptor).

    Args:
        img_gray: Image niveaux de gris uint8 ou float.
        sigma:    Ecart-type du lissage gaussien appliqué aux différences de voisinage.

    Returns:
        Descripteur MIND de shape (H, W, 4), valeurs dans (0, 1].
    """
    imgf  = img_gray.astype(np.float32)
    ksize = max(2 * int(np.ceil(sigma * 3)) + 1, 3)
    Dp    = np.stack(
        [cv2.GaussianBlur((imgf - _shift_nowrap(imgf, dx, dy)) ** 2,
                          (ksize, ksize), sigma)
         for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]],
        axis=-1,
    ).astype(np.float32)
    V     = Dp.mean(axis=-1)
    vmean = float(V.mean()) + 1e-8
    V     = np.clip(V, 1e-3 * vmean, 1e3 * vmean)[..., np.newaxis]
    return np.exp(-Dp / V)


def _mind_ssd(vis_gray: np.ndarray, swir_gray: np.ndarray, mask: np.ndarray) -> tuple:
    """
    Calcule la SSD entre les descripteurs MIND des deux images et érode le masque
    pour exclure les bords.

    Args:
        vis_gray:  Image VIS niveaux de gris.
        swir_gray: Image SWIR niveaux de gris.
        mask:      Masque de région valide uint8.

    Returns:
        Tuple (ssd, mask_erode) : carte SSD (H, W) float32 et masque érodé.
    """
    mind_v = _mind_descriptor(vis_gray,  MIND_SIGMA)
    mind_s = _mind_descriptor(swir_gray, MIND_SIGMA)
    ssd    = ((mind_v - mind_s) ** 2).mean(axis=-1)
    if MIND_BORDER_ERODE > 0:
        k    = 2 * MIND_BORDER_ERODE + 1
        mask = cv2.erode(mask, np.ones((k, k), np.uint8), iterations=1)
    return ssd, mask


def compute_mind(vis_gray: np.ndarray, swir_gray: np.ndarray, mask: np.ndarray) -> float:
    """
    Calcule la MIND-SSD scalaire entre les deux images.

    Args:
        vis_gray:  Image VIS niveaux de gris.
        swir_gray: Image SWIR niveaux de gris.
        mask:      Masque de région valide uint8.

    Returns:
        MIND-SSD dans [0, 1] (0 = alignement parfait), ou nan si le masque est vide.
    """
    ssd, mask = _mind_ssd(vis_gray, swir_gray, mask)
    roi = mask > 0
    return float("nan") if roi.sum() == 0 else float(ssd[roi].mean())


def compute_mind_map(vis_gray: np.ndarray, swir_gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Calcule la carte spatiale MIND-SSD pixel par pixel.
    Les pixels hors masque sont mis à nan.

    Args:
        vis_gray:  Image VIS niveaux de gris.
        swir_gray: Image SWIR niveaux de gris.
        mask:      Masque de région valide uint8.

    Returns:
        Carte SSD (H, W) float32, nan hors masque.
    """
    ssd, mask = _mind_ssd(vis_gray, swir_gray, mask)
    ssd[mask == 0] = np.nan
    return ssd


# ─── Point d'entrée pipeline ─────────────────────────────────────────────────

def compute_all_metrics(vis_bgr: np.ndarray, swir_raw: np.ndarray, H: np.ndarray) -> dict:
    """
    Point d'entrée unique pour évaluer la qualité du recalage sur une paire.
    Enchaîne le prétraitement et le calcul des trois métriques.

    Args:
        vis_bgr:  Image VIS BGR.
        swir_raw: Image SWIR brute (même résolution que vis_bgr).
        H:        Matrice d'homographie 3×3.

    Returns:
        Dictionnaire {"goa": float, "edge_f1_8px": float, "mind_ssd": float}.
    """
    vis_gray, swir_warped, valid_mask = preprocess_pair(vis_bgr, swir_raw, H)
    return {
        "goa":      compute_goa(vis_gray,  swir_warped, valid_mask),
        "edge_f1_8px": compute_edge_alignment_metrics(vis_gray, swir_warped, valid_mask)["edge_f1_8px"],
        "mind_ssd": compute_mind(vis_gray, swir_warped, valid_mask),
    }
