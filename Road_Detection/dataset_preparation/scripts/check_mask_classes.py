from pathlib import Path
import os
from PIL import Image
from collections import Counter

# ============================================================
# MASK FOLDER
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MASK_DIR = (
    PROJECT_ROOT
    / "input"
    / "Segformer_Dataset"
    / "clean"
    / "masks"
)

# ============================================================
# SETTINGS
# ============================================================

EXPECTED_SIZE = (512, 512)

# ============================================================
# FIND MASK FILES
# ============================================================

mask_files = [
    f
    for f in os.listdir(MASK_DIR)
    if f.lower().endswith(".png")
]

mask_files.sort()

print("=" * 80)
print("SEGFORMER MASK CLASS CHECK")
print("=" * 80)

print(f"Mask folder : {MASK_DIR}")
print(f"Total masks : {len(mask_files)}")

# ============================================================
# VARIABLES
# ============================================================

all_values = set()

value_pixel_counts = Counter()

masks_by_value = Counter()

empty_masks = []

wrong_size_masks = []

error_masks = []

# ============================================================
# CHECK EVERY MASK
# ============================================================

for index, mask_file in enumerate(mask_files, start=1):

    mask_path = os.path.join(
        MASK_DIR,
        mask_file
    )

    try:

        with Image.open(mask_path) as mask:

            # ------------------------------------------------
            # Check size
            # ------------------------------------------------

            if mask.size != EXPECTED_SIZE:

                wrong_size_masks.append(
                    (mask_file, mask.size)
                )

                continue

            # ------------------------------------------------
            # Convert to grayscale
            # ------------------------------------------------

            mask = mask.convert("L")

            # ------------------------------------------------
            # Get pixel values
            # ------------------------------------------------

            pixels = list(mask.getdata())

            unique_values = set(pixels)

            # ------------------------------------------------
            # Add values to global set
            # ------------------------------------------------

            all_values.update(unique_values)

            # ------------------------------------------------
            # Count pixels for each class
            # ------------------------------------------------

            value_pixel_counts.update(pixels)

            # ------------------------------------------------
            # Count how many masks contain each value
            # ------------------------------------------------

            for value in unique_values:

                masks_by_value[value] += 1

            # ------------------------------------------------
            # Check completely empty mask
            #
            # If only 0 exists, it means no labelled pixels.
            # ------------------------------------------------

            if unique_values == {0}:

                empty_masks.append(mask_file)

    except Exception as e:

        error_masks.append(
            (mask_file, str(e))
        )

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if index % 500 == 0:

        print(
            f"Checked {index}/{len(mask_files)} masks..."
        )


# ============================================================
# SORT CLASS VALUES
# ============================================================

sorted_values = sorted(all_values)

# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("FINAL MASK CLASS RESULTS")
print("=" * 80)

print(
    f"Total masks checked       : {len(mask_files)}"
)

print(
    f"Unique pixel values found : {len(sorted_values)}"
)

print(
    f"Pixel values              : {sorted_values}"
)

# ============================================================
# MIN / MAX
# ============================================================

if sorted_values:

    print(
        f"Minimum pixel value       : {min(sorted_values)}"
    )

    print(
        f"Maximum pixel value       : {max(sorted_values)}"
    )

# ============================================================
# PIXEL COUNT PER VALUE
# ============================================================

print("\n")
print("=" * 80)
print("PIXEL COUNT FOR EACH VALUE")
print("=" * 80)

for value in sorted_values:

    print(
        f"Value {value:3d} : "
        f"{value_pixel_counts[value]:,} pixels"
    )

# ============================================================
# MASK COUNT PER VALUE
# ============================================================

print("\n")
print("=" * 80)
print("NUMBER OF MASKS CONTAINING EACH VALUE")
print("=" * 80)

for value in sorted_values:

    print(
        f"Value {value:3d} : "
        f"{masks_by_value[value]:,} masks"
    )

# ============================================================
# EMPTY MASKS
# ============================================================

print("\n")
print("=" * 80)
print("EMPTY / BACKGROUND-ONLY MASKS")
print("=" * 80)

print(
    f"Background-only masks : {len(empty_masks)}"
)

if empty_masks:

    print("\nFirst 20 background-only masks:")

    for f in empty_masks[:20]:

        print(f)

# ============================================================
# WRONG SIZE MASKS
# ============================================================

print("\n")
print("=" * 80)
print("WRONG SIZE MASKS")
print("=" * 80)

print(
    f"Wrong size masks : {len(wrong_size_masks)}"
)

if wrong_size_masks:

    for f, size in wrong_size_masks[:20]:

        print(
            f"{f} -> {size}"
        )

# ============================================================
# ERROR MASKS
# ============================================================

print("\n")
print("=" * 80)
print("MASK READ ERRORS")
print("=" * 80)

print(
    f"Error masks : {len(error_masks)}"
)

if error_masks:

    for f, error in error_masks[:20]:

        print(
            f"{f} -> {error}"
        )

# ============================================================
# SEGFORMER RECOMMENDATION
# ============================================================

print("\n")
print("=" * 80)
print("SEGFORMER CHECK")
print("=" * 80)

if sorted_values == [0, 1]:

    print("GOOD: Masks contain exactly two classes: 0 and 1.")
    print("This is suitable for binary SegFormer segmentation.")

elif sorted_values:

    print(
        "INFO: Masks contain the following values:"
    )

    print(sorted_values)

    print(
        "\nWe need to determine whether these values"
    )

    print(
        "represent your intended segmentation classes."
    )

else:

    print(
        "WARNING: No pixel values were detected."
    )

# ============================================================
# DONE
# ============================================================

print("\n")
print("=" * 80)
print("CHECK COMPLETE")
print("=" * 80)