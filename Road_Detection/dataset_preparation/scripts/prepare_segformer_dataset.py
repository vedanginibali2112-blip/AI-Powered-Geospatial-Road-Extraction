import os
from PIL import Image

# ============================================================
# INPUT FOLDERS
# ============================================================

from pathlib import Path
import os
from PIL import Image

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "input" / "Segformer_Dataset"

IMAGE_DIR = DATASET_DIR / "images"
MASK_DIR = DATASET_DIR / "masks"

OUTPUT_DIR = DATASET_DIR / "clean"

OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_MASK_DIR = OUTPUT_DIR / "masks"

OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

EXPECTED_SIZE = (512, 512)

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff"
)

# ============================================================
# GET FILES
# ============================================================

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(IMAGE_EXTENSIONS)
]

mask_files = [
    f for f in os.listdir(MASK_DIR)
    if f.lower().endswith(".png")
]

print("=" * 80)
print("PREPARING SEGFORMER DATASET")
print("=" * 80)

print(f"Images found : {len(image_files)}")
print(f"Masks found  : {len(mask_files)}")

# ============================================================
# CREATE MASK LOOKUP
#
# Example:
#
# Image:
# D1tile_000002_png.rf.ABC123 (1).jpg
#
# Mask:
# D1tile_000002_png.rf.ABC123_mask.png
#
# We store:
#
# D1tile_000002_png.rf.ABC123
#
# ============================================================

mask_lookup = {}

for mask_file in mask_files:

    mask_base = os.path.splitext(mask_file)[0]

    # Remove "_mask"
    if mask_base.endswith("_mask"):
        mask_base = mask_base[:-5]

    # Store mask using base name
    mask_lookup[mask_base] = mask_file


# ============================================================
# PROCESS IMAGE-MASK PAIRS
# ============================================================

matched = 0
unmatched = 0
bad_images = 0
bad_masks = 0

problem_images = []

for image_file in image_files:

    # Remove image extension
    image_base = os.path.splitext(image_file)[0]

    # --------------------------------------------------------
    # IMPORTANT FIX
    #
    # Your image contains:
    #
    # xxxxx (1).jpg
    #
    # Your mask contains:
    #
    # xxxxx_mask.png
    #
    # Therefore remove " (1)" from image filename.
    # --------------------------------------------------------

    if image_base.endswith(" (1)"):
        image_base = image_base[:-4]

    # --------------------------------------------------------
    # Find mask
    # --------------------------------------------------------

    if image_base not in mask_lookup:

        print(f"[NO MASK] {image_file}")

        unmatched += 1
        problem_images.append(image_file)

        continue

    mask_file = mask_lookup[image_base]

    image_path = os.path.join(
        IMAGE_DIR,
        image_file
    )

    mask_path = os.path.join(
        MASK_DIR,
        mask_file
    )

    # ========================================================
    # CHECK IMAGE
    # ========================================================

    try:

        with Image.open(image_path) as img:

            image_size = img.size

            if image_size != EXPECTED_SIZE:

                print(
                    f"[BAD IMAGE SIZE] "
                    f"{image_file} -> {image_size}"
                )

                bad_images += 1
                continue

            # Convert to RGB
            img_rgb = img.convert("RGB")

            # New sequential filename
            new_name = f"{matched + 1:06d}"

            output_image = os.path.join(
                OUTPUT_IMAGE_DIR,
                new_name + ".jpg"
            )

            img_rgb.save(
                output_image,
                "JPEG",
                quality=95
            )

    except Exception as e:

        print(
            f"[IMAGE ERROR] {image_file}: {e}"
        )

        bad_images += 1
        continue

    # ========================================================
    # CHECK MASK
    # ========================================================

    try:

        with Image.open(mask_path) as mask:

            mask_size = mask.size

            if mask_size != EXPECTED_SIZE:

                print(
                    f"[BAD MASK SIZE] "
                    f"{mask_file} -> {mask_size}"
                )

                if os.path.exists(output_image):
                    os.remove(output_image)

                bad_masks += 1
                continue

            # Convert mask to grayscale
            mask_gray = mask.convert("L")

            output_mask = os.path.join(
                OUTPUT_MASK_DIR,
                new_name + ".png"
            )

            mask_gray.save(
                output_mask,
                "PNG"
            )

    except Exception as e:

        print(
            f"[MASK ERROR] {mask_file}: {e}"
        )

        if os.path.exists(output_image):
            os.remove(output_image)

        bad_masks += 1
        continue

    # ========================================================
    # SUCCESSFUL PAIR
    # ========================================================

    matched += 1

    if matched % 100 == 0:

        print(
            f"Processed pairs: {matched}"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"Matched image-mask pairs : {matched}")
print(f"Images without mask      : {unmatched}")
print(f"Bad images               : {bad_images}")
print(f"Bad masks                : {bad_masks}")

print("\nOutput folders:")
print(OUTPUT_IMAGE_DIR)
print(OUTPUT_MASK_DIR)


# ============================================================
# OUTPUT VERIFICATION
# ============================================================

output_images = [
    f for f in os.listdir(OUTPUT_IMAGE_DIR)
    if f.lower().endswith(".jpg")
]

output_masks = [
    f for f in os.listdir(OUTPUT_MASK_DIR)
    if f.lower().endswith(".png")
]

print("\n")
print("=" * 80)
print("OUTPUT VERIFICATION")
print("=" * 80)

print(f"Output images : {len(output_images)}")
print(f"Output masks  : {len(output_masks)}")

if len(output_images) == len(output_masks) == matched:

    print("\nSUCCESS: Image and mask counts match.")

else:

    print("\nWARNING: Image and mask counts DO NOT match.")


print("\n")
print("=" * 80)
print("DONE")
print("=" * 80)