from pathlib import Path
import os
import re
from PIL import Image

# ============================================================
# INPUT FOLDERS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "input" / "Segformer_Dataset"

IMAGE_DIR = DATASET_DIR / "images"
MASK_DIR = DATASET_DIR / "masks"

# ============================================================
# OUTPUT FOLDERS
# ============================================================

OUTPUT_DIR = DATASET_DIR / "clean"

OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_MASK_DIR = OUTPUT_DIR / "masks"

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

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
# NORMALIZE IMAGE NAME
# ============================================================
#
# Examples:
#
# xxxxx.jpg
# xxxxx (1).jpg
# xxxxx (25).jpg
# xxxxx (125).jpg
#
# All become:
#
# xxxxx
#
# ============================================================

def normalize_image_name(filename):

    base = os.path.splitext(filename)[0]

    # Remove numbered suffix:
    # (1), (2), (25), (125), etc.
    base = re.sub(r"\s+\(\d+\)$", "", base)

    return base


# ============================================================
# NORMALIZE MASK NAME
# ============================================================
#
# Examples:
#
# xxxxx_mask.png
# xxxxx_mask (1).png
# xxxxx_mask (25).png
# xxxxx_mask (125).png
#
# All become:
#
# xxxxx
#
# ============================================================

def normalize_mask_name(filename):

    base = os.path.splitext(filename)[0]

    # First remove numbered suffix
    base = re.sub(r"\s+\(\d+\)$", "", base)

    # Then remove "_mask"
    if base.endswith("_mask"):
        base = base[:-5]

    return base


# ============================================================
# GET IMAGE FILES
# ============================================================

image_files = [
    f
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(IMAGE_EXTENSIONS)
]

# ============================================================
# GET MASK FILES
# ============================================================

mask_files = [
    f
    for f in os.listdir(MASK_DIR)
    if f.lower().endswith(".png")
]

print("=" * 80)
print("SEGFORMER DATASET PREPARATION")
print("=" * 80)

print(f"Total images found : {len(image_files)}")
print(f"Total masks found  : {len(mask_files)}")


# ============================================================
# BUILD IMAGE LOOKUP
# ============================================================

image_lookup = {}

for image_file in image_files:

    key = normalize_image_name(image_file)

    image_lookup.setdefault(key, []).append(image_file)


# ============================================================
# BUILD MASK LOOKUP
# ============================================================

mask_lookup = {}

for mask_file in mask_files:

    key = normalize_mask_name(mask_file)

    mask_lookup.setdefault(key, []).append(mask_file)


# ============================================================
# MATCH IMAGES AND MASKS
# ============================================================

matched_pairs = []

unmatched_images = []
unmatched_masks = []

# ------------------------------------------------------------
# MATCH IMAGES TO MASKS
# ------------------------------------------------------------

for key, image_list in image_lookup.items():

    if key not in mask_lookup:

        for image_file in image_list:
            unmatched_images.append(image_file)

        continue

    mask_list = mask_lookup[key]

    # --------------------------------------------------------
    # If there is exactly one image and one mask
    # --------------------------------------------------------

    if len(image_list) == 1 and len(mask_list) == 1:

        matched_pairs.append(
            (image_list[0], mask_list[0])
        )

    else:

        # ----------------------------------------------------
        # Match numbered duplicates
        #
        # Example:
        #
        # image:
        # xxxxx.jpg
        # xxxxx (1).jpg
        # xxxxx (2).jpg
        #
        # mask:
        # xxxxx_mask.png
        # xxxxx_mask (1).png
        # xxxxx_mask (2).png
        #
        # Match based on the number.
        # ----------------------------------------------------

        def get_number(filename):

            match = re.search(
                r"\s+\((\d+)\)\.[^.]+$",
                filename
            )

            if match:
                return int(match.group(1))

            return 0

        image_sorted = sorted(
            image_list,
            key=get_number
        )

        mask_sorted = sorted(
            mask_list,
            key=get_number
        )

        pair_count = min(
            len(image_sorted),
            len(mask_sorted)
        )

        for i in range(pair_count):

            matched_pairs.append(
                (
                    image_sorted[i],
                    mask_sorted[i]
                )
            )

        # Extra images
        if len(image_sorted) > pair_count:

            unmatched_images.extend(
                image_sorted[pair_count:]
            )

        # Extra masks
        if len(mask_sorted) > pair_count:

            unmatched_masks.extend(
                mask_sorted[pair_count:]
            )


# ============================================================
# FIND MASKS WITH NO IMAGE
# ============================================================

for key, mask_list in mask_lookup.items():

    if key not in image_lookup:

        unmatched_masks.extend(mask_list)


# ============================================================
# REMOVE DUPLICATE UNMATCHED ENTRIES
# ============================================================

unmatched_images = sorted(
    set(unmatched_images)
)

unmatched_masks = sorted(
    set(unmatched_masks)
)


# ============================================================
# PRINT MATCHING RESULT
# ============================================================

print("\n")
print("=" * 80)
print("MATCHING RESULT")
print("=" * 80)

print(
    f"Matched image-mask pairs : {len(matched_pairs)}"
)

print(
    f"Images without masks     : {len(unmatched_images)}"
)

print(
    f"Masks without images     : {len(unmatched_masks)}"
)


# ============================================================
# SHOW SAMPLE MATCHES
# ============================================================

print("\n")
print("=" * 80)
print("SAMPLE MATCHES")
print("=" * 80)

for image_file, mask_file in matched_pairs[:10]:

    print("\nIMAGE:")
    print(image_file)

    print("MASK:")
    print(mask_file)


# ============================================================
# PROCESS MATCHED PAIRS
# ============================================================

print("\n")
print("=" * 80)
print("PROCESSING 512 × 512 IMAGE-MASK PAIRS")
print("=" * 80)

processed = 0
bad_images = 0
bad_masks = 0

for image_file, mask_file in matched_pairs:

    image_path = os.path.join(
        IMAGE_DIR,
        image_file
    )

    mask_path = os.path.join(
        MASK_DIR,
        mask_file
    )

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    try:

        image = Image.open(image_path)

        if image.size != EXPECTED_SIZE:

            print(
                f"[BAD IMAGE SIZE] "
                f"{image_file} -> {image.size}"
            )

            bad_images += 1

            continue

        # Convert image to RGB
        image = image.convert("RGB")

    except Exception as e:

        print(
            f"[IMAGE ERROR] "
            f"{image_file}: {e}"
        )

        bad_images += 1

        continue

    # --------------------------------------------------------
    # Open mask
    # --------------------------------------------------------

    try:

        mask = Image.open(mask_path)

        if mask.size != EXPECTED_SIZE:

            print(
                f"[BAD MASK SIZE] "
                f"{mask_file} -> {mask.size}"
            )

            bad_masks += 1

            continue

        # Convert mask to grayscale
        mask = mask.convert("L")

    except Exception as e:

        print(
            f"[MASK ERROR] "
            f"{mask_file}: {e}"
        )

        bad_masks += 1

        continue

    # --------------------------------------------------------
    # Sequential output name
    # --------------------------------------------------------

    number = processed + 1

    image_output_name = (
        f"{number:06d}.jpg"
    )

    mask_output_name = (
        f"{number:06d}.png"
    )

    image_output_path = os.path.join(
        OUTPUT_IMAGE_DIR,
        image_output_name
    )

    mask_output_path = os.path.join(
        OUTPUT_MASK_DIR,
        mask_output_name
    )

    # --------------------------------------------------------
    # Save image
    # --------------------------------------------------------

    image.save(
        image_output_path,
        "JPEG",
        quality=95
    )

    # --------------------------------------------------------
    # Save mask
    # --------------------------------------------------------

    mask.save(
        mask_output_path,
        "PNG"
    )

    processed += 1

    if processed % 500 == 0:

        print(
            f"Processed: {processed}"
        )


# ============================================================
# FINAL OUTPUT COUNTS
# ============================================================

output_images = [
    f
    for f in os.listdir(OUTPUT_IMAGE_DIR)
    if f.lower().endswith(".jpg")
]

output_masks = [
    f
    for f in os.listdir(OUTPUT_MASK_DIR)
    if f.lower().endswith(".png")
]


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(
    f"Original images          : {len(image_files)}"
)

print(
    f"Original masks           : {len(mask_files)}"
)

print(
    f"Matched pairs            : {len(matched_pairs)}"
)

print(
    f"Successfully processed   : {processed}"
)

print(
    f"Images without masks     : {len(unmatched_images)}"
)

print(
    f"Masks without images    : {len(unmatched_masks)}"
)

print(
    f"Bad images               : {bad_images}"
)

print(
    f"Bad masks                : {bad_masks}"
)

print("\n")
print("=" * 80)
print("OUTPUT VERIFICATION")
print("=" * 80)

print(
    f"Output images : {len(output_images)}"
)

print(
    f"Output masks  : {len(output_masks)}"
)


# ============================================================
# FINAL CHECK
# ============================================================

if (
    len(output_images) == processed
    and
    len(output_masks) == processed
    and
    processed > 0
):

    print("\n")
    print("SUCCESS!")
    print("Image and mask dataset is ready.")
    print(
        "Every output image has a corresponding mask."
    )

else:

    print("\n")
    print("WARNING!")
    print(
        "Please review the numbers above."
    )


print("\n")
print("=" * 80)
print("OUTPUT LOCATION")
print("=" * 80)

print(OUTPUT_DIR)

print("\nDONE.")