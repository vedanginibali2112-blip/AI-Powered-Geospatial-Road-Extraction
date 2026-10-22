import os
import sys
import cv2
import numpy as np
from pathlib import Path

# ===========================================================
# 1. CONFIGURE YOUR INPUT AND OUTPUT PATHS
# ===========================================================
# The folder path exactly from your raw Manchester screenshot
RAW_MANCHESTER_ROOT = r"C:\Users\vedangini.bali\Downloads\archive\tiff"

# The single master folder where everything is being collected
MASTER_OUTPUT_ROOT = r"D:\RoadPlugin\master_collected_dataset"

def log(message):
    print(message)
    sys.stdout.flush()

log("🚀 Launching Master Data Collection Engine for Manchester Dataset...")

# We look inside all 3 folder pairs from the raw dataset
FOLDER_PAIRS = [
    ("train", "train_labels"),
    ("val", "val_labels"),
    ("test", "test_labels")
]

# Initialize Master Folders
MASTER_IMAGES_DIR = os.path.join(MASTER_OUTPUT_ROOT, "images")
MASTER_MASKS_DIR = os.path.join(MASTER_OUTPUT_ROOT, "masks")
os.makedirs(MASTER_IMAGES_DIR, exist_ok=True)
os.makedirs(MASTER_MASKS_DIR, exist_ok=True)

TILE_SIZE = 512
total_saved_pairs = 0
skipped_blank_masks = 0

# ===========================================================
# 2. CORE EXTRACTION AND COLLECTION LOOP
# ===========================================================
for img_folder, lbl_folder in FOLDER_PAIRS:
    src_img_dir = os.path.join(RAW_MANCHESTER_ROOT, img_folder)
    src_lbl_dir = os.path.join(RAW_MANCHESTER_ROOT, lbl_folder)
    
    if not os.path.exists(src_img_dir) or not os.path.exists(src_lbl_dir):
        log(f"⚠️ Skipping folder pair: {img_folder} / {lbl_folder} (Not found on disk)")
        continue
        
    log(f"\nCollecting tiles from raw folders: {img_folder}...")
    
    all_files = os.listdir(src_img_dir)
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg'))]
    
    for img_idx, img_name in enumerate(image_files):
        base_name, _ = os.path.splitext(img_name)
        
        # Find matching label extension
        lbl_name = None
        for ext in ['.tif', '.tiff', '.png', '.jpg']:
            test_lbl_name = base_name + ext
            if os.path.exists(os.path.join(src_lbl_dir, test_lbl_name)):
                lbl_name = test_lbl_name
                break
                
        if lbl_name is None:
            continue
            
        img_full_path = os.path.join(src_img_dir, img_name)
        lbl_full_path = os.path.join(src_lbl_dir, lbl_name)
        
        # Read maps into memory arrays
        img = cv2.imread(img_full_path)
        mask = cv2.imread(lbl_full_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            continue
            
        img_h, img_w = img.shape[:2]
        file_tile_counter = 0
        
        # Slice into 512x512 grids
        for y in range(0, img_h, TILE_SIZE):
            for x in range(0, img_w, TILE_SIZE):
                
                # Boundary drop
                if x + TILE_SIZE > img_w or y + TILE_SIZE > img_h:
                    continue
                    
                img_tile = img[y:y+TILE_SIZE, x:x+TILE_SIZE]
                mask_tile = mask[y:y+TILE_SIZE, x:x+TILE_SIZE]
                
                if np.max(img_tile) == 0:
                    continue
                    
                # Force pure binary road values (0 = Background, 255 = Road)
                _, binary_mask_255 = cv2.threshold(mask_tile, 1, 255, cv2.THRESH_BINARY)
                
                # 🚫 ROADS-ONLY FILTER: Skip if no road exists in this chunk
                if np.max(binary_mask_255) == 0:
                    skipped_blank_masks += 1
                    continue
                
                # Save into the unified master folders with a specific prefix
                clean_filename = f"manc_{img_folder}_{base_name}_tile_{file_tile_counter}.png"
                
                cv2.imwrite(os.path.join(MASTER_IMAGES_DIR, clean_filename), img_tile)
                cv2.imwrite(os.path.join(MASTER_MASKS_DIR, clean_filename), binary_mask_255)
                
                file_tile_counter += 1
                total_saved_pairs += 1
                
        if (img_idx + 1) % 15 == 0 or (img_idx + 1) == len(image_files):
            log(f"   ├─ Progress: [{img_idx+1}/{len(image_files)}] maps processed. Collected total: {total_saved_pairs}")

log("\n=======================================================")
log("🎉 MASTER COLLECTION COMPLETE!")
log("=======================================================")
log(f"Total Useful Road Tile Pairs Collected: {total_saved_pairs}")
log(f"Total Empty Background Tiles Dropped:   {skipped_blank_masks}")
log(f"Master Images Folder: {MASTER_IMAGES_DIR}")
log(f"Master Masks Folder:  {MASTER_MASKS_DIR}")
