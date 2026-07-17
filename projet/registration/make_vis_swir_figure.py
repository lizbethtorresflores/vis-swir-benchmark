import cv2
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path("/home/lizbeth/Desktop/intership/recallage-multimodal-zz3")

datasets = {
    "Sunny": "sunny_test",
    "Cloudy": "cloudy_test",
    "Rainy": "rainy_test",
    "Snowy": "snowy_test",
}

image_idx = 680

def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Impossible de charger : {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(
    nrows=2,
    ncols=4,
    figsize=(16, 7)
)

for col, (condition, dataset) in enumerate(datasets.items()):

    paired_dir = (
        PROJECT_ROOT
        / "datas"
        / dataset
        / "Sortie_Extraite"
        / "paired_images"
    )

    vis_path = paired_dir / "VISIBLE" / f"rgb_image_{image_idx:06d}.jpg"
    swir_path = paired_dir / "SWIR" / f"swir_image_{image_idx:06d}.jpg"

    vis_img = load_rgb(vis_path)
    swir_img = load_rgb(swir_path)

    # ======================
    # LIGNE 1 -> VIS
    # ======================

    axes[0, col].imshow(vis_img)
    axes[0, col].set_title(f"{condition} VIS", fontsize=14)
    axes[0, col].axis("off")

    # ======================
    # LIGNE 2 -> SWIR
    # ======================

    axes[1, col].imshow(swir_img, cmap="gray")
    axes[1, col].set_title(f"{condition} SWIR", fontsize=14)
    axes[1, col].axis("off")

plt.tight_layout()

out_path = PROJECT_ROOT / "vis_swir_conditions_poster.png"

plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Figure enregistrée dans : {out_path}")
