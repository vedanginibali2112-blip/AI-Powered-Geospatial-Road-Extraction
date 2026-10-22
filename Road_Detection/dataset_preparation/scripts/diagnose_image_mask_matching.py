from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "input" / "Segformer_Dataset"

IMAGE_DIR = DATASET_DIR / "images"
MASK_DIR = DATASET_DIR / "masks"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

images = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(IMAGE_EXTENSIONS)
]

masks = [
    f for f in os.listdir(MASK_DIR)
    if f.lower().endswith(".png")
]

# ============================================================
# CREATE NORMALIZED IMAGE NAMES
# ============================================================

image_lookup = {}

for f in images:

    base = os.path.splitext(f)[0]

    # Remove common duplicate suffixes
    base = base.replace(" (1)", "")
    base = base.replace(" (2)", "")
    base = base.replace(" (3)", "")

    image_lookup[base] = f


# ============================================================
# CREATE NORMALIZED MASK NAMES
# ============================================================

mask_lookup = {}

for f in masks:

    base = os.path.splitext(f)[0]

    # Remove _mask
    if base.endswith("_mask"):
        base = base[:-5]

    mask_lookup[base] = f


# ============================================================
# FIND UNMATCHED IMAGES
# ============================================================

unmatched_images = []

for normalized_name, original_name in image_lookup.items():

    if normalized_name not in mask_lookup:
        unmatched_images.append(original_name)


# ============================================================
# FIND UNMATCHED MASKS
# ============================================================

unmatched_masks = []

for normalized_name, original_name in mask_lookup.items():

    if normalized_name not in image_lookup:
        unmatched_masks.append(original_name)


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 80)
print("UNMATCHED IMAGES")
print("=" * 80)

print("Total:", len(unmatched_images))

for f in unmatched_images[:30]:
    print(f)


print("\n")
print("=" * 80)
print("UNMATCHED MASKS")
print("=" * 80)

print("Total:", len(unmatched_masks))

for f in unmatched_masks[:30]:
    print(f)


print("\n")
print("=" * 80)
print("DONE")
print("=" * 80)