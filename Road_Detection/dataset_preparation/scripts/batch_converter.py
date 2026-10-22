import os
import sys
import numpy as np
import cv2
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
import fiona
from shapely.geometry import shape, Polygon, LineString, box
from shapely.ops import transform, polygonize, unary_union
from pyproj import Transformer

# ===========================================================
# 1. DIRECTORY CONFIGURATIONS
# ===========================================================
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_ROOT = str(PROJECT_ROOT / "output" / "final_geospatial_dataset")

TIFF_FILES = [
    str(PROJECT_ROOT / "input" / "rasters" / "SM-GCP-M79.tif"),
    str(PROJECT_ROOT / "input" / "rasters" / "SM-GCP-M84.tif"),
    str(PROJECT_ROOT / "input" / "rasters" / "SM-GCP-M87.tif"),
    str(PROJECT_ROOT / "input" / "rasters" / "SM-GCP-M91.tif"),
    str(PROJECT_ROOT / "input" / "rasters" / "SM-GCP-M95.tif"),
]

SHAPEFILE_PATHS = [
    str(PROJECT_ROOT / "input" / "labels" / "polygon1.shp"),
    str(PROJECT_ROOT / "input" / "labels" / "polygon2.shp"),
    str(PROJECT_ROOT / "input" / "labels" / "polygon3.shp"),
    str(PROJECT_ROOT / "input" / "labels" / "polygon shp.shp"),
]

# ===========================================================
# 2. RUN WORKSPACE ARCHITECTURE INITIALIZATION
# ===========================================================
IMAGES_DIR = os.path.join(OUTPUT_ROOT, "images")
MASKS_DIR = os.path.join(OUTPUT_ROOT, "masks")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)

def log(message):
    print(message)
    sys.stdout.flush()

log("🚀 Launching Roads-Only Filtered Dataset Generator Pipeline...")

# ===========================================================
# 3. CORE PROCESSING ENGINE LOOP
# ===========================================================
total_saved_pairs = 0
skipped_no_road = 0
TILE_SIZE = 512

for tiff_idx, tiff_path in enumerate(TIFF_FILES):
    if not os.path.exists(tiff_path):
        log(f"❌ Error: TIFF file not found, skipping: {tiff_path}")
        continue
        
    map_nickname = os.path.splitext(os.path.basename(tiff_path))[0]
    log(f"\nProcessing File [{tiff_idx+1}/{len(TIFF_FILES)}]: {map_nickname}")
    
    with rasterio.open(tiff_path) as src:
        img_width = src.width
        img_height = src.height
        src_transform = src.transform
        src_crs = src.crs
        total_bands = src.count
        
        log(f"├─ Coordinate System Detected: {src_crs.to_string()}")
        
        # Bounding box around current map to prevent infinite-canvas filling anomalies
        bounds = src.bounds
        map_boundary_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        
        # ---------------------------------------------------
        # DYNAMIC CRS RE-PROJECTION & BOUNDARY CLIPPING
        # ---------------------------------------------------
        raw_lines = []
        true_polygons = []
        
        for shp_path in SHAPEFILE_PATHS:
            if not os.path.exists(shp_path):
                continue
            with fiona.open(shp_path, "r") as shp:
                shp_crs = shp.crs_wkt
                transformer = Transformer.from_crs(shp_crs, src_crs.to_wkt(), always_xy=True)
                
                for feature in shp:
                    if feature['geometry'] is not None:
                        geom = shape(feature['geometry'])
                        transformed_geom = transform(transformer.transform, geom)
                        
                        if not transformed_geom.intersects(map_boundary_box):
                            continue
                        clipped_geom = transformed_geom.intersection(map_boundary_box)
                        
                        if isinstance(clipped_geom, Polygon):
                            true_polygons.append(clipped_geom)
                        elif isinstance(clipped_geom, LineString):
                            raw_lines.append(clipped_geom)
                            
        # Natively close open line boundaries into solid polygons
        if raw_lines:
            merged_lines = unary_union(raw_lines)
            polygonized_loops = list(polygonize(merged_lines))
            true_polygons.extend(polygonized_loops)
            
        log(f"├─ Verified Solid Polygons Compiled: {len(true_polygons)}")
        log(f"└─ Tiling map grids and sorting road segments...")
        
        band_selection = [1, 2, 3] if total_bands >= 3 else list(range(1, total_bands + 1))
        file_tile_counter = 0
        
        for y in range(0, img_height, TILE_SIZE):
            for x in range(0, img_width, TILE_SIZE):
                
                if x + TILE_SIZE > img_width or y + TILE_SIZE > img_height:
                    continue
                    
                window = Window(col_off=x, row_off=y, width=TILE_SIZE, height=TILE_SIZE)
                tile_transform = rasterio.windows.transform(window, src_transform)
                
                rgb_data = src.read(band_selection, window=window)
                
                if total_bands == 1:
                    img_np = np.repeat(np.transpose(rgb_data, (1, 2, 0)), 3, axis=-1).astype(np.uint8)
                else:
                    img_np = np.transpose(rgb_data, (1, 2, 0)).astype(np.uint8)
                
                # Skip tiles that fall entirely into pitch-black spatial map border voids
                if np.max(img_np) == 0:
                    continue
                
                # Rasterize polygons to 512x512 array canvas
                mask_np = rasterize(
                    shapes=[(geom, 255) for geom in true_polygons],
                    out_shape=(TILE_SIZE, TILE_SIZE),
                    transform=tile_transform,
                    fill=0,
                    all_touched=False, # False ensures crisp, pixel-perfect edges for SegFormer
                    dtype=np.uint8
                )
                
                # ---------------------------------------------------
                # 🚫 THE SKIP FILTER: REMOVE TILES WITHOUT ROADS
                # ---------------------------------------------------
                if np.max(mask_np) == 0:
                    skipped_no_road += 1
                    continue # Drops both the image and mask arrays cleanly out of memory
                
                # Save out properties with matching 1-to-1 character layouts
                filename = f"{map_nickname}_tile_{file_tile_counter}.png"
                img_out_path = os.path.join(IMAGES_DIR, filename)
                mask_out_path = os.path.join(MASKS_DIR, filename)
                
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                cv2.imwrite(img_out_path, img_bgr)
                cv2.imwrite(mask_out_path, mask_np)
                
                file_tile_counter += 1
                total_saved_pairs += 1
                
                if total_saved_pairs % 200 == 0:
                    log(f"   ├─ Valid Pairs Saved: {total_saved_pairs} | Blank Background Tiles Skipped: {skipped_no_road}")

log("\n=======================================================")
log("🎉 DATASET BATCH CONVERSION SUCCESSFUL")
log("=======================================================")
log(f"Total Useful Dataset Pairs Stored:  {total_saved_pairs}")
log(f"Total Empty Backgrounds Discarded: {skipped_no_road}")
log(f"Clean Images Stored At: {IMAGES_DIR}")
log(f"Clean Masks Stored At:  {MASKS_DIR}")
