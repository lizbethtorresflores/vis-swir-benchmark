# registration/utils/visualization.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from lightglue.utils import load_image, rbd


def overlay_50_alpha(rgb_img, swir_img):
    h, w = rgb_img.shape[:2]
    if len(swir_img.shape) == 3:
        swir_gray = cv2.cvtColor(swir_img, cv2.COLOR_BGR2GRAY)
    else:
        swir_gray = swir_img
    swir_resized = cv2.resize(swir_gray, (w, h))
    swir_bgr     = cv2.cvtColor(swir_resized, cv2.COLOR_GRAY2BGR)
    alpha   = 0.5
    overlay = (alpha * rgb_img + (1 - alpha) * swir_bgr).astype(np.uint8)
    return overlay


def affichage_side(swir_bgr, vis_bgr, left_label: str, right_label: str):
    swir_rgb = cv2.cvtColor(swir_bgr, cv2.COLOR_BGR2RGB)
    vis_rgb  = cv2.cvtColor(vis_bgr,  cv2.COLOR_BGR2RGB)
    concat   = np.concatenate([swir_rgb, vis_rgb], axis=1)
    plt.figure(figsize=(14, 6))
    plt.title(f"{left_label} | {right_label}")
    plt.imshow(concat)
    plt.axis('off')
    plt.show()


def match_pair_paths(path_vis, path_swir, extractor, matcher, device='mps'):
    img_vis_bgr  = cv2.imread(path_vis)
    img_swir_bgr = cv2.imread(path_swir)
    from ..preprocessing.contrast import enhance_swir_contrast
    img_swir_bgr = enhance_swir_contrast(img_swir_bgr)
    cv2.imwrite('tmp_img0.jpg', img_vis_bgr)
    cv2.imwrite('tmp_img1.jpg', img_swir_bgr)
    img_vis  = load_image('tmp_img0.jpg').to(device)
    img_swir = load_image('tmp_img1.jpg').to(device)
    feats0   = extractor.extract(img_vis)
    feats1   = extractor.extract(img_swir)
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in (feats0, feats1, matches01)]
    matches = matches01['matches']
    pts0 = feats0['keypoints'][matches[:, 0]]
    pts1 = feats1['keypoints'][matches[:, 1]]
    return pts0, pts1, matches01
