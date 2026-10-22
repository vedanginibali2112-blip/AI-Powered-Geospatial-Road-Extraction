import os
import shutil
import random
import numpy as np
from PIL import Image

# ============================================================
# PATHS
# ============================================================
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "input" / "Segformer_Dataset"

IMAGE_DIR = DATASET_DIR / "images"
MASK_DIR = DATASET_DIR / "masks"

TRAIN_IMAGE_DIR = DATASET_DIR / "train" / "images"
TRAIN_MASK_DIR = DATASET_DIR / "train" / "masks"

VAL_IMAGE_DIR = DATASET_DIR / "val" / "images"
VAL_MASK_DIR = DATASET_DIR / "val" / "masks"

TEST_IMAGE_DIR = DATASET_DIR / "test" / "images"
TEST_MASK_DIR = DATASET_DIR / "test" / "masks"

EXPECTED_SIZE = (512, 512)
RANDOM_SEED = 42

# ============================================================
# EXACT SPLIT
# ============================================================

TARGET_TRAIN = 5048
TARGET_VAL = 631
TARGET_TEST = 632

TARGET_TOTAL = TARGET_TRAIN + TARGET_VAL + TARGET_TEST

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

output_dirs = [
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    VAL_IMAGE_DIR,
    VAL_MASK_DIR,
    TEST_IMAGE_DIR,
    TEST_MASK_DIR
]

for directory in output_dirs:
    os.makedirs(directory, exist_ok=True)

# ============================================================
# CLEAR OLD SPLIT DATA
# ============================================================

print("=" * 80)
print("CLEARING OLD TRAIN / VAL / TEST DATA")
print("=" * 80)

for directory in output_dirs:

    for filename in os.listdir(directory):

        filepath = os.path.join(directory, filename)

        if os.path.isfile(filepath):
            os.remove(filepath)

print("Old split data cleared.")

# ============================================================
# FIND IMAGE-MASK PAIRS
# ============================================================

print("\n" + "=" * 80)
print("SCANNING IMAGE-MASK PAIRS")
print("=" * 80)

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(".jpg")
]

image_files.sort()

valid_pairs = []
missing_masks = []
blank_masks = []

for image_file in image_files:

    image_name = os.path.splitext(image_file)[0]

    mask_file = image_name + ".png"

    mask_path = os.path.join(MASK_DIR, mask_file)

    # --------------------------------------------------------
    # CHECK MASK EXISTS
    # --------------------------------------------------------

    if not os.path.exists(mask_path):

        missing_masks.append(image_file)

        continue

    # --------------------------------------------------------
    # CHECK IMAGE SIZE
    # --------------------------------------------------------

    image_path = os.path.join(IMAGE_DIR, image_file)

    with Image.open(image_path) as image:

        if image.size != EXPECTED_SIZE:

            print(
                f"WARNING: Wrong image size: "
                f"{image_file} -> {image.size}"
            )

            continue

    # --------------------------------------------------------
    # CHECK MASK
    # --------------------------------------------------------

    with Image.open(mask_path) as mask:

        mask_np = np.array(mask.convert("L"))

        if mask_np.shape != (512, 512):

            print(
                f"WARNING: Wrong mask size: "
                f"{mask_file} -> {mask_np.shape}"
            )

            continue

        # Any non-zero pixel means foreground exists
        non_zero_pixels = np.count_nonzero(mask_np)

        # Completely black mask
        if non_zero_pixels == 0:

            blank_masks.append(mask_file)

            continue

    # --------------------------------------------------------
    # VALID PAIR
    # --------------------------------------------------------

    valid_pairs.append(
        (image_file, mask_file)
    )

# ============================================================
# SCAN RESULTS
# ============================================================

print("\nTotal JPG images       :", len(image_files))
print("Missing masks          :", len(missing_masks))
print("Blank masks removed    :", len(blank_masks))
print("Valid image-mask pairs :", len(valid_pairs))

# ============================================================
# CHECK ENOUGH DATA
# ============================================================

if len(valid_pairs) < TARGET_TOTAL:

    raise RuntimeError(
        "\nERROR!\n"
        f"Required pairs : {TARGET_TOTAL}\n"
        f"Available pairs: {len(valid_pairs)}\n"
        "Not enough valid non-blank image-mask pairs."
    )

# ============================================================
# RANDOM SHUFFLE
# ============================================================

random.seed(RANDOM_SEED)

random.shuffle(valid_pairs)

# Select exactly 6311 pairs
selected_pairs = valid_pairs[:TARGET_TOTAL]

# ============================================================
# SPLIT
# ============================================================

train_pairs = selected_pairs[:TARGET_TRAIN]

val_pairs = selected_pairs[
    TARGET_TRAIN:
    TARGET_TRAIN + TARGET_VAL
]

test_pairs = selected_pairs[
    TARGET_TRAIN + TARGET_VAL:
]

print("\n" + "=" * 80)
print("DATASET SPLIT")
print("=" * 80)

print("TRAIN :", len(train_pairs))
print("VAL   :", len(val_pairs))
print("TEST  :", len(test_pairs))
print("TOTAL :", len(selected_pairs))

# ============================================================
# PROCESS IMAGE + MASK
# ============================================================

def process_pair(
    image_file,
    mask_file,
    output_image_dir,
    output_mask_dir
):

    image_path = os.path.join(
        IMAGE_DIR,
        image_file
    )

    mask_path = os.path.join(
        MASK_DIR,
        mask_file
    )

    output_image_path = os.path.join(
        output_image_dir,
        image_file
    )

    output_mask_path = os.path.join(
        output_mask_dir,
        mask_file
    )

    # ========================================================
    # IMAGE
    # ========================================================

    with Image.open(image_path) as image:

        if image.size != EXPECTED_SIZE:

            raise ValueError(
                f"Wrong image size: "
                f"{image_file} -> {image.size}"
            )

        image = image.convert("RGB")

        image.save(
            output_image_path,
            "JPEG",
            quality=95
        )

    # ========================================================
    # MASK
    # ========================================================

    with Image.open(mask_path) as mask:

        if mask.size != EXPECTED_SIZE:

            raise ValueError(
                f"Wrong mask size: "
                f"{mask_file} -> {mask.size}"
            )

        # Convert to grayscale
        mask = mask.convert("L")

        mask_np = np.array(mask)

        # ----------------------------------------------------
        # IMPORTANT
        #
        # Every non-zero pixel becomes 255.
        #
        # 0     = BLACK background
        # > 0   = WHITE foreground/road
        # ----------------------------------------------------

        binary_mask = np.where(
            mask_np > 0,
            255,
            0
        ).astype(np.uint8)

        # Convert back to PIL image
        binary_mask = Image.fromarray(
            binary_mask,
            mode="L"
        )

        # Save as PNG
        binary_mask.save(
            output_mask_path,
            "PNG"
        )


# ============================================================
# PROCESS TRAIN
# ============================================================

print("\n" + "=" * 80)
print("PROCESSING TRAIN")
print("=" * 80)

for i, (img_f, msk_f) in enumerate(train_pairs, 1):

    process_pair(
        img_f,
        msk_f,
        TRAIN_IMAGE_DIR,
        TRAIN_MASK_DIR
    )

    if i % 500 == 0 or i == len(train_pairs):

        print(
            f"Train processed: "
            f"{i}/{len(train_pairs)}"
        )

# ============================================================
# PROCESS VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("PROCESSING VALIDATION")
print("=" * 80)

for i, (img_f, msk_f) in enumerate(val_pairs, 1):

    process_pair(
        img_f,
        msk_f,
        VAL_IMAGE_DIR,
        VAL_MASK_DIR
    )

    if i % 100 == 0 or i == len(val_pairs):

        print(
            f"Validation processed: "
            f"{i}/{len(val_pairs)}"
        )

# ============================================================
# PROCESS TEST
# ============================================================

print("\n" + "=" * 80)
print("PROCESSING TEST")
print("=" * 80)

for i, (img_f, msk_f) in enumerate(test_pairs, 1):

    process_pair(
        img_f,
        msk_f,
        TEST_IMAGE_DIR,
        TEST_MASK_DIR
    )

    if i % 100 == 0 or i == len(test_pairs):

        print(
            f"Test processed: "
            f"{i}/{len(test_pairs)}"
        )

# ============================================================
# VERIFY SPLIT
# ============================================================

def verify_split(
    split_name,
    image_dir,
    mask_dir,
    expected_count
):

    images = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(".jpg")
    ]

    masks = [
        f for f in os.listdir(mask_dir)
        if f.lower().endswith(".png")
    ]

    image_set = {
        os.path.splitext(f)[0]
        for f in images
    }

    mask_set = {
        os.path.splitext(f)[0]
        for f in masks
    }

    missing_mask = image_set - mask_set
    missing_image = mask_set - image_set

    blank_count = 0
    wrong_values = 0

    # --------------------------------------------------------
    # CHECK EVERY MASK
    # --------------------------------------------------------

    for mask_file in masks:

        mask_path = os.path.join(
            mask_dir,
            mask_file
        )

        with Image.open(mask_path) as mask:

            mask_np = np.array(
                mask.convert("L")
            )

            unique_values = np.unique(
                mask_np
            )

            # Must contain only 0 and 255
            if not set(unique_values).issubset(
                {0, 255}
            ):

                wrong_values += 1

            # Completely blank
            if np.count_nonzero(mask_np) == 0:

                blank_count += 1

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n" + split_name.upper())
    print("-" * 80)

    print("Expected images :", expected_count)
    print("Actual images   :", len(images))

    print("Expected masks  :", expected_count)
    print("Actual masks    :", len(masks))

    print("Missing masks   :", len(missing_mask))
    print("Missing images  :", len(missing_image))
    print("Blank masks     :", blank_count)
    print("Wrong values    :", wrong_values)

    if (
        len(images) == expected_count
        and
        len(masks) == expected_count
        and
        len(missing_mask) == 0
        and
        len(missing_image) == 0
        and
        blank_count == 0
        and
        wrong_values == 0
    ):

        print("✅ PERFECT")

        return True

    else:

        print("❌ PROBLEM FOUND")

        return False


# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 80)
print("FINAL DATASET VERIFICATION")
print("=" * 80)

train_ok = verify_split(
    "train",
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    TARGET_TRAIN
)

val_ok = verify_split(
    "val",
    VAL_IMAGE_DIR,
    VAL_MASK_DIR,
    TARGET_VAL
)

test_ok = verify_split(
    "test",
    TEST_IMAGE_DIR,
    TEST_MASK_DIR,
    TARGET_TEST
)

# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 80)
print("FINAL RESULT")
print("=" * 80)

if train_ok and val_ok and test_ok:

    print("""
🎉 DATASET CREATED SUCCESSFULLY!

Train : 5048 images + 5048 masks
Val   : 631 images + 631 masks
Test  : 632 images + 632 masks

All masks:
    0   = BLACK background
    255 = WHITE road/foreground

All masks are:
    512 x 512
    PNG
    Grayscale
    Binary
    Non-blank
""")

else:

    print("""
❌ DATASET HAS A PROBLEM.

Check the verification results above.
""")