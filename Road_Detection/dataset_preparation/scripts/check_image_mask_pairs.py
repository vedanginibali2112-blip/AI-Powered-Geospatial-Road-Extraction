import os
from PIL import Image

# ============================================================
# CHANGE THESE PATHS
# ============================================================
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = str(PROJECT_ROOT / "input" / "images")
MASK_DIR  =  str(PROJECT_ROOT / "input" / "masks")

EXPECTED_SIZE = (512, 512)

VALID_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff"
)


# ============================================================
# GET FILES
# ============================================================

def get_files(folder):

    files = {}

    for filename in os.listdir(folder):

        if filename.lower().endswith(VALID_EXTENSIONS):

            name = os.path.splitext(filename)[0]

            files[name] = filename

    return files


# ============================================================
# MAIN
# ============================================================

def check_dataset():

    print("=" * 80)
    print("IMAGE - MASK DATASET CHECK")
    print("=" * 80)

    # --------------------------------------------------------
    # Check folders
    # --------------------------------------------------------

    if not os.path.exists(IMAGE_DIR):

        print("\nERROR: Image folder not found:")
        print(IMAGE_DIR)
        return

    if not os.path.exists(MASK_DIR):

        print("\nERROR: Mask folder not found:")
        print(MASK_DIR)
        return


    # --------------------------------------------------------
    # Get files
    # --------------------------------------------------------

    images = get_files(IMAGE_DIR)
    masks = get_files(MASK_DIR)

    print(f"\nTotal images : {len(images)}")
    print(f"Total masks  : {len(masks)}")


    # ========================================================
    # 1. IMAGE WITHOUT MASK
    # ========================================================

    missing_masks = sorted(
        set(images.keys()) - set(masks.keys())
    )

    print("\n")
    print("=" * 80)
    print("IMAGES WITHOUT MATCHING MASK")
    print("=" * 80)

    if not missing_masks:

        print("PASS - Every image has a matching mask.")

    else:

        print(
            f"FOUND {len(missing_masks)} IMAGE(S) WITHOUT MASK:"
        )

        for name in missing_masks:

            print(
                f"IMAGE: {images[name]}"
            )


    # ========================================================
    # 2. MASK WITHOUT IMAGE
    # ========================================================

    extra_masks = sorted(
        set(masks.keys()) - set(images.keys())
    )

    print("\n")
    print("=" * 80)
    print("MASKS WITHOUT MATCHING IMAGE")
    print("=" * 80)

    if not extra_masks:

        print("PASS - Every mask has a matching image.")

    else:

        print(
            f"FOUND {len(extra_masks)} MASK(S) WITHOUT IMAGE:"
        )

        for name in extra_masks:

            print(
                f"MASK: {masks[name]}"
            )


    # ========================================================
    # 3. CHECK MATCHED IMAGE + MASK SIZE
    # ========================================================

    wrong_images = []
    wrong_masks = []

    matched_names = sorted(
        set(images.keys()) & set(masks.keys())
    )

    print("\n")
    print("=" * 80)
    print("CHECKING 512 × 512 SIZE")
    print("=" * 80)

    for name in matched_names:

        image_filename = images[name]
        mask_filename = masks[name]

        image_path = os.path.join(
            IMAGE_DIR,
            image_filename
        )

        mask_path = os.path.join(
            MASK_DIR,
            mask_filename
        )


        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        try:

            with Image.open(image_path) as img:

                image_size = img.size

        except Exception as e:

            wrong_images.append(
                (
                    image_filename,
                    f"Cannot read image: {e}"
                )
            )

            continue


        # ----------------------------------------------------
        # MASK
        # ----------------------------------------------------

        try:

            with Image.open(mask_path) as mask:

                mask_size = mask.size

        except Exception as e:

            wrong_masks.append(
                (
                    mask_filename,
                    f"Cannot read mask: {e}"
                )
            )

            continue


        # ----------------------------------------------------
        # CHECK IMAGE SIZE
        # ----------------------------------------------------

        if image_size != EXPECTED_SIZE:

            wrong_images.append(
                (
                    image_filename,
                    image_size
                )
            )


        # ----------------------------------------------------
        # CHECK MASK SIZE
        # ----------------------------------------------------

        if mask_size != EXPECTED_SIZE:

            wrong_masks.append(
                (
                    mask_filename,
                    mask_size
                )
            )


    # ========================================================
    # 4. REPORT WRONG IMAGE SIZE
    # ========================================================

    print("\n")
    print("-" * 80)
    print("IMAGES THAT ARE NOT 512 × 512")
    print("-" * 80)

    if not wrong_images:

        print("PASS - All matched images are 512 × 512.")

    else:

        print(
            f"FOUND {len(wrong_images)} WRONG IMAGE(S):"
        )

        for filename, size in wrong_images:

            print(
                f"IMAGE: {filename}  →  SIZE: {size}"
            )


    # ========================================================
    # 5. REPORT WRONG MASK SIZE
    # ========================================================

    print("\n")
    print("-" * 80)
    print("MASKS THAT ARE NOT 512 × 512")
    print("-" * 80)

    if not wrong_masks:

        print("PASS - All matched masks are 512 × 512.")

    else:

        print(
            f"FOUND {len(wrong_masks)} WRONG MASK(S):"
        )

        for filename, size in wrong_masks:

            print(
                f"MASK: {filename}  →  SIZE: {size}"
            )


    # ========================================================
    # 6. FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"Total images                  : {len(images)}")
    print(f"Total masks                   : {len(masks)}")
    print(f"Images without masks          : {len(missing_masks)}")
    print(f"Masks without images          : {len(extra_masks)}")
    print(f"Images with wrong size        : {len(wrong_images)}")
    print(f"Masks with wrong size         : {len(wrong_masks)}")


    # --------------------------------------------------------
    # Overall result
    # --------------------------------------------------------

    if (
        len(missing_masks) == 0
        and len(extra_masks) == 0
        and len(wrong_images) == 0
        and len(wrong_masks) == 0
    ):

        print("\n" + "=" * 80)
        print("SUCCESS")
        print("=" * 80)

        print(
            "Every image has a matching mask and "
            "all are 512 × 512."
        )

    else:

        print("\n" + "=" * 80)
        print("PROBLEMS FOUND")
        print("=" * 80)

        print(
            "Review the lists above to identify the "
            "exact files that need correction."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    check_dataset()