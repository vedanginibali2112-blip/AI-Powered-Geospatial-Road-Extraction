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
# SETTING: MINIMUM REQUIRED ROAD PIXELS
# ============================================================
# Change this number to control what gets filtered out.
# If a mask has fewer road pixels than this threshold, it is skipped.
MIN_ROAD_PIXELS = 100 

# ============================================================
# EXACT TARGET SPLIT COUNTS
# ============================================================
TARGET_TRAIN = 5048
TARGET_VAL = 631
TARGET_TEST = 632
TARGET_TOTAL = TARGET_TRAIN + TARGET_VAL + TARGET_TEST  # Exactly 6,311

# ============================================================
# CREATE & CLEAN OUTPUT DIRECTORIES
# ============================================================

output_dirs = [
    TRAIN_IMAGE_DIR, TRAIN_MASK_DIR,
    VAL_IMAGE_DIR, VAL_MASK_DIR,
    TEST_IMAGE_DIR, TEST_MASK_DIR
]

for directory in output_dirs:
    os.makedirs(directory, exist_ok=True)

print("=" * 80)
print("CLEARING OLD SPLIT DATA")
print("=" * 80)

for directory in output_dirs:
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            os.remove(filepath)

print("Old split files cleared.")

# ============================================================
# FIND AND FILTER PAIRS
# ============================================================

image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(".jpg")]
image_files.sort()

valid_pairs = []
missing_masks = []
blank_masks_count = 0

print("\nScanning masks to exclude empty/blank images...")

for image_file in image_files:
    image_name = os.path.splitext(image_file)[0]
    mask_file = image_name + ".png"
    mask_path = os.path.join(MASK_DIR, mask_file)

    if os.path.exists(mask_path):
        with Image.open(mask_path) as mask:
            mask_gray = mask.convert("L")
            mask_np = np.array(mask_gray)
            
            # Count pixels that represent your road/target class.
            # This checks for both 255 (white) or 1 (raw index class).
            road_pixel_count = np.sum((mask_np == 255) | (mask_np == 1))
            
            # If the mask doesn't meet our road threshold, filter it out
            if road_pixel_count < MIN_ROAD_PIXELS:
                blank_masks_count += 1
                continue  

        valid_pairs.append((image_file, mask_file))
    else:
        missing_masks.append(image_file)

print(f"Total raw images analyzed: {len(image_files)}")
print(f"Filtered out blank/low-data masks: {blank_masks_count}")
print(f"Total valid pairs remaining with actual road data: {len(valid_pairs)}")

if len(valid_pairs) < TARGET_TOTAL:
    raise RuntimeError(
        f"Error: You requested a total split of {TARGET_TOTAL} pairs, "
        f"but only {len(valid_pairs)} pairs passed the filtering threshold."
    )

# ============================================================
# PRECISE RANDOM SAMPLING & SLICING
# ============================================================

random.seed(RANDOM_SEED)
random.shuffle(valid_pairs)

selected_pairs = valid_pairs[:TARGET_TOTAL]

train_pairs = selected_pairs[:TARGET_TRAIN]
val_pairs = selected_pairs[TARGET_TRAIN : TARGET_TRAIN + TARGET_VAL]
test_pairs = selected_pairs[TARGET_TRAIN + TARGET_VAL :]

print("\n" + "=" * 80)
print("EXACT TARGET DATASET SPLIT DEFINED")
print("=" * 80)
print(f"Train Expected      : {TARGET_TRAIN}  | Actual Split : {len(train_pairs)}")
print(f"Validation Expected : {TARGET_VAL}   | Actual Split : {len(val_pairs)}")
print(f"Test Expected       : {TARGET_TEST}   | Actual Split : {len(test_pairs)}")

# ============================================================
# PROCESSING ENGINE
# ============================================================

def process_pair(image_file, mask_file, output_image_dir, output_mask_dir):
    image_path = os.path.join(IMAGE_DIR, image_file)
    mask_path = os.path.join(MASK_DIR, mask_file)

    with Image.open(image_path) as image:
        if image.size != EXPECTED_SIZE:
            raise ValueError(f"Wrong image size: {image_file} -> {image.size}")
        image = image.convert("RGB")
        output_image_path = os.path.join(output_image_dir, image_file)
        image.save(output_image_path, "JPEG", quality=95)

    with Image.open(mask_path) as mask:
        if mask.size != EXPECTED_SIZE:
            raise ValueError(f"Wrong mask size: {mask_file} -> {mask.size}")
        mask = mask.convert("L")
        
        # 0 -> 0 background, 255 -> 1 class index
        mask = mask.point(lambda pixel: 1 if pixel == 255 else (pixel if pixel == 1 else 0))
        
        output_mask_path = os.path.join(output_mask_dir, mask_file)
        mask.save(output_mask_path, "PNG")

# Execute compilation
print("\nProcessing splits...")
for img_f, msk_f in train_pairs: process_pair(img_f, msk_f, TRAIN_IMAGE_DIR, TRAIN_MASK_DIR)
for img_f, msk_f in val_pairs: process_pair(img_f, msk_f, VAL_IMAGE_DIR, VAL_MASK_DIR)
for img_f, msk_f in test_pairs: process_pair(img_f, msk_f, TEST_IMAGE_DIR, TEST_MASK_DIR)

print("\n🎉 DATASET SPLIT COMPLETED SUCCESSFULLY WITH VERIFIED CONTENT!")
