import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    image_id = str(args.image_id).zfill(6)

    root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    paired = os.path.join(root, "datas", args.dataset, "Sortie_Extraite", "paired_images")

    img_path = os.path.join(paired, "VISIBLE", f"rgb_image_{image_id}.jpg")

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Fichier introuvable : {img_path}")

    out_dir = os.path.join(paired, args.method, "debug_segformer")
    os.makedirs(out_dir, exist_ok=True)

    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640"
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640"
    ).to(args.device).eval()

    inputs = processor(images=img_rgb, return_tensors="pt")
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    h, w = img_rgb.shape[:2]

    logits = F.interpolate(
        outputs.logits,
        size=(h, w),
        mode="bilinear",
        align_corners=False
    )

    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    sky_prob = probs[2]   # clase sky en ADE20K
    sky = sky_prob > 0.85

    # Partie supérieure uniquement, mais moins restrictive.
    top = np.zeros_like(sky, dtype=bool)
    top[: int(h * 0.50), :] = True
    sky = sky & top

    sky_u8 = sky.astype(np.uint8) * 255

    # Fermer uniquement les petits trous, sans éroder les bords.
    kernel = np.ones((5, 5), np.uint8)
    sky_u8 = cv2.morphologyEx(sky_u8, cv2.MORPH_CLOSE, kernel)

    sky = sky_u8 > 0
    overlay = img_rgb.copy()
    overlay[sky] = [255, 0, 0]

    debug = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)

    out_overlay = os.path.join(out_dir, f"rgb_image_{image_id}_sky_overlay.jpg")
    out_mask = os.path.join(out_dir, f"rgb_image_{image_id}_sky_mask.jpg")

    cv2.imwrite(out_overlay, cv2.cvtColor(debug, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_mask, sky.astype(np.uint8) * 255)

    print("Enregistré :")
    print(out_overlay)
    print(out_mask)


if __name__ == "__main__":
    main()
