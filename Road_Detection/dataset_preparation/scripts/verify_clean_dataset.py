from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "input" / "Segformer_Dataset"

IMAGE_DIR = DATASET_DIR / "images"
MASK_DIR = DATASET_DIR / "masks"

images = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(".jpg")
])

masks = sorted([
    f for f in os.listdir(MASK_DIR)
    if f.lower().endswith(".png")
])

print("Clean images:", len(images))
print("Clean masks :", len(masks))

image_names = {os.path.splitext(f)[0] for f in images}
mask_names = {os.path.splitext(f)[0] for f in masks}

print("Images without masks:", len(image_names - mask_names))
print("Masks without images:", len(mask_names - image_names))

if (
    len(images) == 6311
    and len(masks) == 6311
    and image_names == mask_names
):
    print("\n✅ CLEAN DATASET CONFIRMED")
else:
    print("\n❌ DO NOT DELETE ORIGINAL DATASET YET")