import os
import cv2
import numpy as np

# ============================================================
# CHANGE THESE TWO PATHS
# ============================================================

IMAGE_DIR = r"input/images"
MASK_DIR  = r"input/masks"

# ============================================================
# IMAGE PROCESSING
# Images MUST be 3-channel RGB
# ============================================================

def process_images(folder):

    print("\n" + "=" * 70)
    print("CHECKING IMAGES")
    print("=" * 70)

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    ]

    total = len(files)
    already_3 = 0
    converted = 0
    failed = 0

    print(f"Total images found: {total}\n")

    for filename in files:

        path = os.path.join(folder, filename)

        # Read image without changing channels
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"[ERROR] Cannot read: {filename}")
            failed += 1
            continue

        # Check dimensions
        h, w = img.shape[:2]

        if (w, h) != (512, 512):
            print(f"[WARNING] {filename}: size = {w}x{h}")

        # ----------------------------------------------------
        # 3 CHANNELS
        # ----------------------------------------------------
        if len(img.shape) == 3 and img.shape[2] == 3:

            print(f"[OK] {filename} -> 3 channels")

            already_3 += 1

        # ----------------------------------------------------
        # 1 CHANNEL
        # ----------------------------------------------------
        elif len(img.shape) == 2:

            print(f"[FIX] {filename} -> 1 channel → 3 channels")

            # Grayscale → RGB
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # Save as RGB PNG
            cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            converted += 1

        # ----------------------------------------------------
        # 4 CHANNELS
        # ----------------------------------------------------
        elif len(img.shape) == 3 and img.shape[2] == 4:

            print(f"[FIX] {filename} -> 4 channels → 3 channels")

            # Remove alpha channel
            rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            cv2.imwrite(path, rgb)

            converted += 1

        else:

            print(
                f"[ERROR] {filename}: "
                f"unexpected shape {img.shape}"
            )

            failed += 1

    print("\nIMAGE SUMMARY")
    print("-" * 70)
    print(f"Total images       : {total}")
    print(f"Already 3-channel : {already_3}")
    print(f"Converted          : {converted}")
    print(f"Failed             : {failed}")


# ============================================================
# MASK PROCESSING
#
# IMPORTANT:
# Masks SHOULD remain 1-channel.
#
# We DO NOT convert masks to 3 channels.
# ============================================================

def process_masks(folder):

    print("\n" + "=" * 70)
    print("CHECKING MASKS")
    print("=" * 70)

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    ]

    total = len(files)
    one_channel = 0
    converted = 0
    failed = 0

    print(f"Total masks found: {total}\n")

    for filename in files:

        path = os.path.join(folder, filename)

        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if mask is None:
            print(f"[ERROR] Cannot read: {filename}")
            failed += 1
            continue

        h, w = mask.shape[:2]

        if (w, h) != (512, 512):
            print(f"[WARNING] {filename}: size = {w}x{h}")

        # ----------------------------------------------------
        # MASK ALREADY 1 CHANNEL
        # ----------------------------------------------------
        if len(mask.shape) == 2:

            unique_values = np.unique(mask)

            print(
                f"[OK] {filename} -> "
                f"1 channel | values: {unique_values[:20]}"
            )

            one_channel += 1

        # ----------------------------------------------------
        # MASK HAS 3 CHANNELS
        # Convert to grayscale
        # ----------------------------------------------------
        elif len(mask.shape) == 3 and mask.shape[2] == 3:

            print(f"[FIX] {filename} -> 3 channels → 1 channel")

            gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            cv2.imwrite(path, gray)

            converted += 1

        # ----------------------------------------------------
        # MASK HAS 4 CHANNELS
        # ----------------------------------------------------
        elif len(mask.shape) == 3 and mask.shape[2] == 4:

            print(f"[FIX] {filename} -> 4 channels → 1 channel")

            gray = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)

            cv2.imwrite(path, gray)

            converted += 1

        else:

            print(
                f"[ERROR] {filename}: "
                f"unexpected shape {mask.shape}"
            )

            failed += 1

    print("\nMASK SUMMARY")
    print("-" * 70)
    print(f"Total masks       : {total}")
    print(f"Already 1-channel : {one_channel}")
    print(f"Converted         : {converted}")
    print(f"Failed            : {failed}")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    if not os.path.exists(IMAGE_DIR):
        print(f"Image folder not found:\n{IMAGE_DIR}")
        exit()

    if not os.path.exists(MASK_DIR):
        print(f"Mask folder not found:\n{MASK_DIR}")
        exit()

    process_images(IMAGE_DIR)
    process_masks(MASK_DIR)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)