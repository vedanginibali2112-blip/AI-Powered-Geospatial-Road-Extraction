import os
import sys
import traceback

import cv2
import torch
import rasterio
import numpy as np
import geopandas as gpd
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from rasterio.windows import Window
from shapely.geometry import Polygon, shape
from rasterio.features import shapes
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ==========================================================
# Project Directories
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"
CONFIG_DIR = PROJECT_DIR / "config"
OUTPUT_DIR = PROJECT_DIR / "output"
TEMP_DIR = PROJECT_DIR / "temp"
LOG_DIR = PROJECT_DIR / "logs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def outpath(output_dir, filename):
    return str(output_dir / filename)


def smooth_mask(mask):
    m = (mask * 255).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    m = cv2.morphologyEx(
        m,
        cv2.MORPH_CLOSE,
        kernel,
    )

    m = cv2.morphologyEx(
        m,
        cv2.MORPH_OPEN,
        kernel,
    )

    m = cv2.bilateralFilter(
        m,
        d=7,
        sigmaColor=30,
        sigmaSpace=30,
    )

    _, m = cv2.threshold(
        m,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    return (m > 0).astype(np.uint8)


def smooth_segformer_fallback(mask):
    """
    Prepare SegFormer mask for fallback.
    """
    mask = (mask > 0).astype("uint8")

    if mask.sum() == 0:
        return mask

    fallback_mask = smooth_mask(mask)

    return (fallback_mask > 0).astype("uint8")



def mask_to_contour(mask):
    m = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        m,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        raise ValueError("No contour found in mask.")

    largest = max(
        contours,
        key=cv2.contourArea,
    )

    pts = largest.squeeze().astype(np.float64)

    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)

    return pts, m


def tangents_and_normals(points, mask, tan_win=5):
    n = len(points)
    h, w = mask.shape
    normals = np.zeros_like(points)

    for i in range(n):
        p_prev = points[(i - tan_win) % n]
        p_next = points[(i + tan_win) % n]

        tangent = p_next - p_prev
        norm = np.linalg.norm(tangent)

        if norm > 1e-6:
            tangent = tangent / norm
        else:
            tangent = np.array([1.0, 0.0])

        normal = np.array([-tangent[1], tangent[0]])

        probe = points[i] + normal * 3

        px = int(round(probe[0]))
        py = int(round(probe[1]))

        inside = (
            0 <= py < h
            and 0 <= px < w
            and mask[py, px] > 0
        )

        if inside:
            normal = -normal

        normals[i] = normal

    return normals


def snap_to_edge(
    points,
    normals,
    src,
    search_radius=5,
    confidence_ratio=1.25,
    pad=12,
):
    xs = points[:, 0]
    ys = points[:, 1]

    x_min = max(int(xs.min()) - pad, 0)
    y_min = max(int(ys.min()) - pad, 0)
    x_max = int(xs.max()) + pad
    y_max = int(ys.max()) + pad

    window_width = x_max - x_min
    window_height = y_max - y_min

    rgb = src.read(
        [1, 2, 3],
        window=Window(
            x_min,
            y_min,
            window_width,
            window_height,
        ),
        boundless=True,
        fill_value=0,
    ).transpose(1, 2, 0)

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    gx = cv2.Scharr(
        gray,
        cv2.CV_32F,
        1,
        0,
    )

    gy = cv2.Scharr(
        gray,
        cv2.CV_32F,
        0,
        1,
    )

    grad = np.sqrt(gx**2 + gy**2)

    snapped = points.copy()
    offset = np.array([x_min, y_min])

    for i, (point, normal) in enumerate(
        zip(points, normals)
    ):
        local_point = point - offset

        ox = int(round(local_point[0]))
        oy = int(round(local_point[1]))

        if (
            0 <= oy < window_height
            and 0 <= ox < window_width
        ):
            original_value = grad[oy, ox]
        else:
            original_value = 0.0

        best_value = original_value
        best_distance = 0

        for distance in range(0, search_radius + 1):
            candidate = local_point + normal * distance

            cxi = int(round(candidate[0]))
            cyi = int(round(candidate[1]))

            if (
                0 <= cyi < window_height
                and 0 <= cxi < window_width
            ):
                value = grad[cyi, cxi]

                if value > best_value:
                    best_value = value
                    best_distance = distance

        if (
            best_distance > 0
            and best_value > original_value * confidence_ratio
        ):
            snapped[i] = point + normal * best_distance

    return snapped


def light_smooth(points, win=3):
    n = len(points)
    kernel = np.ones(win) / win

    xs = np.convolve(
        np.r_[
            points[-win:, 0],
            points[:, 0],
            points[:win, 0],
        ],
        kernel,
        mode="same",
    )

    ys = np.convolve(
        np.r_[
            points[-win:, 1],
            points[:, 1],
            points[:win, 1],
        ],
        kernel,
        mode="same",
    )

    xs = xs[win:win + n]
    ys = ys[win:win + n]

    return np.column_stack([xs, ys])


def enforce_no_inward(
    smoothed_points,
    pre_smooth_points,
    original_contour,
):
    original_contour_float = (
        original_contour.astype(
            np.float32
        ).reshape(-1, 1, 2)
    )

    output = smoothed_points.copy()

    for i, point in enumerate(smoothed_points):
        test = cv2.pointPolygonTest(
            original_contour_float,
            (float(point[0]), float(point[1])),
            False,
        )

        if test > 0:
            output[i] = pre_smooth_points[i]

    return output

def resample_contour_by_distance(points, spacing=20.0):
    points = np.asarray(points, dtype=np.float64)

    if len(points) < 3:
        return points

    if np.linalg.norm(points[0] - points[-1]) < 1e-6:
        points = points[:-1]

    next_points = np.roll(points, -1, axis=0)

    segment_vectors = next_points - points

    segment_lengths = np.linalg.norm(
        segment_vectors,
        axis=1,
    )

    perimeter = segment_lengths.sum()

    if perimeter <= spacing:
        return points

    num_vertices = max(
        3,
        int(round(perimeter / spacing)),
    )

    actual_spacing = perimeter / num_vertices

    cumulative = np.concatenate(
        (
            [0.0],
            np.cumsum(segment_lengths),
        )
    )

    target_distances = (
        np.arange(num_vertices)
        * actual_spacing
    )

    resampled = []

    for distance in target_distances:

        segment_index = (
            np.searchsorted(
                cumulative,
                distance,
                side="right",
            ) - 1
        )

        segment_index = min(
            segment_index,
            len(points) - 1,
        )

        segment_start = cumulative[
            segment_index
        ]

        segment_length = segment_lengths[
            segment_index
        ]

        if segment_length <= 1e-6:
            resampled.append(
                points[segment_index]
            )
            continue

        ratio = (
            distance - segment_start
        ) / segment_length

        p1 = points[segment_index]

        p2 = points[
            (segment_index + 1)
            % len(points)
        ]

        new_point = (
            p1
            + ratio * (p2 - p1)
        )

        resampled.append(new_point)

    return np.asarray(
        resampled,
        dtype=np.float64,
    )
def refine_boundary(
    mask,
    src,
    tan_win=7,
    search_radius=3,
    confidence_ratio=1.15,
    smooth_win=6,
):
    contour, binary_mask = mask_to_contour(mask)

    print("Dense contour points:", len(contour))

    normals = tangents_and_normals(
        contour,
        binary_mask,
        tan_win=tan_win,
    )

    snapped = snap_to_edge(
        contour,
        normals,
        src,
        search_radius=search_radius,
        confidence_ratio=confidence_ratio,
    )

    moved = int(
        np.sum(
            np.any(
                snapped != contour,
                axis=1,
            )
        )
    )

    print(
        f"Vertices snapped to a stronger edge: "
        f"{moved} / {len(contour)}"
    )

    smoothed = light_smooth(
        snapped,
        win=smooth_win,
    )

    safe = enforce_no_inward(
        smoothed,
        snapped,
        contour,
    )

    rolled_back = int(
        np.sum(
            np.any(
                safe != smoothed,
                axis=1,
            )
        )
    )

    print(
        "Points rolled back by the no-inward clamp:",
        rolled_back,
    )

    resampled = resample_contour_by_distance(
        safe,
        spacing=50.0,
    )

    print(
        "Vertices after 50-pixel resampling:",
        len(resampled),
    )

    coordinates = [
        tuple(point)
        for point in resampled
    ]

    if coordinates[0] != coordinates[-1]:
        coordinates.append(
            coordinates[0]
        )

    return Polygon(coordinates).buffer(0)

def pixel_polygon_to_geo(polygon, src):
    if polygon.geom_type == "MultiPolygon":
        print(
            "Warning: Final geometry is MultiPolygon."
        )
        print("Keeping largest polygon part.")

        polygon = max(
            polygon.geoms,
            key=lambda p: p.area,
        )

    if polygon.geom_type != "Polygon":
        raise ValueError(
            f"Expected Polygon, got {polygon.geom_type}"
        )

    exterior = [
        tuple(src.transform * (x, y))
        for x, y in polygon.exterior.coords
    ]

    interiors = [
        [
            tuple(src.transform * (x, y))
            for x, y in ring.coords
        ]
        for ring in polygon.interiors
    ]

    return Polygon(exterior, interiors)


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print(
            "python road_pipeline.py "
            "<input_tif> <output_folder>"
        )
        sys.exit(1)

    tif_path = sys.argv[1]
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    src = None

    try:
        print("=" * 50)
        print("Torch Version :", torch.__version__)
        print(
            "CUDA Available:",
            torch.cuda.is_available(),
        )

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        if torch.cuda.is_available():
            print(
                "GPU :",
                torch.cuda.get_device_name(0),
            )

            print(
                "VRAM:",
                round(
                    torch.cuda.get_device_properties(
                        0
                    ).total_memory / 1024**3,
                    2,
                ),
                "GB",
            )

        print("=" * 50)

        processor = (
            SegformerImageProcessor.from_pretrained(
                "nvidia/"
                "segformer-b2-finetuned-ade-512-512"
            )
        )

        model = (
            SegformerForSemanticSegmentation
            .from_pretrained(
                "nvidia/"
                "segformer-b2-finetuned-ade-512-512",
                num_labels=2,
                ignore_mismatched_sizes=True,
            )
        )

        segformer_model = (
            MODEL_DIR
            / "segformer_b2_final_road_model.pth"
        )

        checkpoint = torch.load(
            segformer_model,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        model.to(device)
        model.eval()

        print("Road model loaded successfully")

        src = rasterio.open(tif_path)

        print("Width :", src.width)
        print("Height:", src.height)
        print("Bands :", src.count)
        print("Dtype :", src.dtypes)

        window_size = 512
        overlap = 64
        step = window_size - overlap

        height = src.height
        width = src.width

        prediction_mask = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        print("Height :", height)
        print("Width  :", width)
        print("STEP   :", step)
        print(
            "Prediction Mask Shape :",
            prediction_mask.shape,
        )

        processed = 0
        skipped = 0

        for y in tqdm(
            range(0, height, step),
            desc="Rows",
        ):
            for x in range(0, width, step):
                tile_width = min(
                    window_size,
                    width - x,
                )

                tile_height = min(
                    window_size,
                    height - y,
                )

                if (
                    tile_width < window_size // 2
                    or tile_height < window_size // 2
                ):
                    continue

                window = Window(
                    x,
                    y,
                    tile_width,
                    tile_height,
                )

                tile = src.read(
                    [1, 2, 3],
                    window=window,
                    boundless=True,
                    fill_value=255,
                )

                tile = np.transpose(
                    tile,
                    (1, 2, 0),
                )

                tile_std = tile.std()

                if tile_std < 1:
                    skipped += 1
                    continue

                inputs = processor(
                    images=Image.fromarray(tile),
                    return_tensors="pt",
                )

                inputs = {
                    key: value.to(device)
                    for key, value in inputs.items()
                }

                with torch.no_grad():
                    outputs = model(**inputs)

                    logits = F.interpolate(
                        outputs.logits,
                        size=(
                            tile_height,
                            tile_width,
                        ),
                        mode="bilinear",
                        align_corners=False,
                    )

                    pred = (
                        torch.argmax(
                            logits,
                            dim=1,
                        )
                        .squeeze()
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                    )

                    white_mask = np.all(
                        tile >= 250,
                        axis=2,
                    )

                    pred[white_mask] = 0

                prediction_mask[
                    y:y + tile_height,
                    x:x + tile_width,
                ] = pred

                processed += 1

                del inputs
                del outputs
                del logits

                if processed % 100 == 0:
                    print(
                        f"Processed: {processed} "
                        f"Skipped: {skipped}"
                    )

        print("\n===================================")
        print("Inference Finished")
        print("Processed Tiles :", processed)
        print("Skipped Tiles   :", skipped)
        print("===================================")

        mask = prediction_mask.copy()
        mask[mask > 0] = 1

        print("Mask Shape :", mask.shape)
        print(
            "Road Pixels :",
            np.sum(mask == 1),
        )
        print(
            "Background :",
            np.sum(mask == 0),
        )

        (
            num_labels,
            labels,
            stats,
            centroids,
        ) = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        if num_labels <= 1:
            raise ValueError(
                "No foreground component found."
            )

        areas = stats[
            1:,
            cv2.CC_STAT_AREA,
        ]

        largest_label = np.argmax(areas) + 1

        clean_mask = np.zeros_like(
            mask,
            dtype=np.uint8,
        )

        clean_mask[labels == largest_label] = 1

        print("====================================")
        print(
            "Total Components :",
            num_labels - 1,
        )
        print(
            "Largest Component Area :",
            areas.max(),
        )
        print(
            "Remaining Components :",
            1,
        )
        print("====================================")

        clean_mask = smooth_mask(clean_mask)
        print("Mask smoothing completed.")

        cv2.imwrite(
            outpath(
                output_dir,
                "CleanMask.png",
            ),
            clean_mask * 255,
        )

        print("Saved CleanMask.png")

        mask = (
            clean_mask > 0
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        if not contours:
            raise ValueError(
                "No contour found in clean mask."
            )

        largest_contour = max(
            contours,
            key=cv2.contourArea,
        )

        print(
            "Largest contour area:",
            cv2.contourArea(largest_contour),
        )

        contour_img = np.zeros_like(
            mask,
            dtype=np.uint8,
        )

        cv2.drawContours(
            contour_img,
            [largest_contour],
            -1,
            255,
            1,
        )

        cv2.imwrite(
            outpath(
                output_dir,
                "road_boundary.png",
            ),
            contour_img,
        )

        print("Boundary image saved.")

        clean_boundary_mask = np.zeros_like(
            mask,
            dtype=np.uint8,
        )

        cv2.drawContours(
            clean_boundary_mask,
            [largest_contour],
            -1,
            255,
            thickness=cv2.FILLED,
        )

        cv2.imwrite(
            outpath(
                output_dir,
                "road_clean_boundary_mask.png",
            ),
            clean_boundary_mask,
        )

        print("Clean boundary mask saved.")

        boundary_mask = np.zeros_like(
            mask,
            dtype=np.uint8,
        )

        contours2, _ = cv2.findContours(
            clean_boundary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        largest_contour2 = max(
            contours2,
            key=cv2.contourArea,
        )

        cv2.drawContours(
            boundary_mask,
            [largest_contour2],
            -1,
            255,
            1,
        )

        cv2.imwrite(
            outpath(
                output_dir,
                "road_clean_boundary.png",
            ),
            boundary_mask,
        )

        print(
            "Clean boundary extracted successfully."
        )
        print("======================================")
        print("Saved Files")
        print("1. road_boundary.png")
        print(
            "2. road_clean_boundary_mask.png"
        )
        print("3. road_clean_boundary.png")
        print("======================================")

        # ======================================================
        # SEGFORMER SHAPEFILE - ADDITION ONLY
        # ======================================================

        print("Creating SegFormer Shapefile...")
        print("======================================")

        segformer_mask = (
            clean_mask.copy().astype(np.uint8)
        )

        seg_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7),
        )

        segformer_mask = cv2.morphologyEx(
            segformer_mask,
            cv2.MORPH_CLOSE,
            seg_kernel,
            iterations=1,
        )

        (
            num_seg,
            labels_seg,
            stats_seg,
            _,
        ) = cv2.connectedComponentsWithStats(
            segformer_mask,
            connectivity=8,
        )

        if num_seg <= 1:
            print("No SegFormer component found.")

        else:
            largest_seg = (
                1
                + np.argmax(
                    stats_seg[
                        1:,
                        cv2.CC_STAT_AREA,
                    ]
                )
            )

            segformer_mask = (
                labels_seg == largest_seg
            ).astype(np.uint8)

            seg_contours, _ = cv2.findContours(
                segformer_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if not seg_contours:
                print(
                    "No SegFormer contour found."
                )

            else:
                seg_contour = max(
                    seg_contours,
                    key=cv2.contourArea,
                )

                print(
                    "SegFormer contour points:",
                    len(seg_contour),
                )

                epsilon = 3.0

                simplified_contour = (
                    cv2.approxPolyDP(
                        seg_contour,
                        epsilon,
                        True,
                    )
                )

                points = (
                    simplified_contour
                    .reshape(-1, 2)
                    .astype(np.float64)
                )

                smoothed_points = points.copy()

                for _ in range(2):
                    new_points = []

                    for i in range(
                        len(smoothed_points)
                    ):
                        p1 = smoothed_points[
                            (i - 1)
                            % len(smoothed_points)
                        ]

                        p2 = smoothed_points[i]

                        p3 = smoothed_points[
                            (i + 1)
                            % len(smoothed_points)
                        ]

                        new_point = (
                            0.25 * p1
                            + 0.50 * p2
                            + 0.25 * p3
                        )

                        new_points.append(new_point)

                    smoothed_points = np.array(
                        new_points
                    )

                coordinates = [
                    tuple(point)
                    for point in smoothed_points
                ]

                if coordinates[0] != coordinates[-1]:
                    coordinates.append(
                        coordinates[0]
                    )

                # --------------------------------------------------
                # PIXEL POLYGON
                # --------------------------------------------------

                seg_polygon_px = Polygon(
                    coordinates
                ).buffer(0)

                # --------------------------------------------------
                # HANDLE POLYGON / MULTIPOLYGON
                # --------------------------------------------------

                if (
                    seg_polygon_px.geom_type
                    == "MultiPolygon"
                ):
                    print(
                        "Warning: SegFormer geometry "
                        "is MultiPolygon."
                    )
                    print(
                        "Keeping largest polygon part."
                    )

                    seg_polygon_px = max(
                        seg_polygon_px.geoms,
                        key=lambda p: p.area,
                    )

                elif (
                    seg_polygon_px.geom_type
                    != "Polygon"
                ):
                    print(
                        "Invalid SegFormer geometry:",
                        seg_polygon_px.geom_type,
                    )
                    seg_polygon_px = None

                if seg_polygon_px is not None:
                    # --------------------------------------------------
                    # PIXEL TO GEO COORDINATES
                    # --------------------------------------------------

                    exterior_geo = [
                        tuple(
                            src.transform * (x, y)
                        )
                        for x, y in (
                            seg_polygon_px.exterior.coords
                        )
                    ]

                    interiors_geo = [
                        [
                            tuple(
                                src.transform * (x, y)
                            )
                            for x, y in ring.coords
                        ]
                        for ring in (
                            seg_polygon_px.interiors
                        )
                    ]

                    seg_polygon_geo = Polygon(
                        exterior_geo,
                        interiors_geo,
                    ).buffer(0)

                    # --------------------------------------------------
                    # HANDLE GEO MULTIPOLYGON
                    # --------------------------------------------------

                    if (
                        seg_polygon_geo.geom_type
                        == "MultiPolygon"
                    ):
                        print(
                            "Warning: Geo geometry "
                            "is MultiPolygon."
                        )
                        print(
                            "Keeping largest polygon part."
                        )

                        seg_polygon_geo = max(
                            seg_polygon_geo.geoms,
                            key=lambda p: p.area,
                        )

                    if (
                        seg_polygon_geo.geom_type
                        == "Polygon"
                    ):
                        segformer_gdf = gpd.GeoDataFrame(
                            {
                                "Class": ["Road"],
                                "Source": ["SegFormer"],
                            },
                            geometry=[seg_polygon_geo],
                            crs=src.crs,
                        )

                        segformer_shp = outpath(
                            output_dir,
                            "SegFormer.shp",
                        )

                        segformer_gdf.to_file(
                            segformer_shp,
                            driver="ESRI Shapefile",
                        )

                        print(
                            "SegFormer shapefile saved:"
                        )
                        print(segformer_shp)
                        print(
                            "======================================"
                        )
                    else:
                        print(
                            "Invalid Geo geometry:",
                            seg_polygon_geo.geom_type,
                        )

        # ======================================================
        # ADDITIONAL MASK CLEANING
        # ======================================================

        mask = clean_mask.copy().astype(np.uint8)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

        (
            num_labels,
            labels,
            stats,
            _,
        ) = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        if num_labels <= 1:
            raise ValueError(
                "No component found after cleaning."
            )

        largest_label = (
            1
            + np.argmax(
                stats[
                    1:,
                    cv2.CC_STAT_AREA,
                ]
            )
        )

        largest_mask = np.zeros_like(
            mask,
            dtype=np.uint8,
        )

        largest_mask[labels == largest_label] = 1

        contours, _ = cv2.findContours(
            largest_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        largest_contour = max(
            contours,
            key=cv2.contourArea,
        )

        print(
            "Contour Points:",
            len(largest_contour),
        )

        contour_img = np.zeros_like(
            largest_mask,
            dtype=np.uint8,
        )

        cv2.drawContours(
            contour_img,
            [largest_contour],
            -1,
            255,
            1,
        )

        cv2.imwrite(
            outpath(
                output_dir,
                "road_contour.png",
            ),
            contour_img,
        )

        print("Contour image saved.")

        # ======================================================
        # SAM2 REFINEMENT
        # ======================================================

        checkpoint = str(
            MODEL_DIR / "sam2.1_hiera_large.pt"
        )

        model_cfg = str(
            CONFIG_DIR / "sam2.1_hiera_l.yaml"
        )

        print("MODEL_CFG :", model_cfg)
        print("CHECKPOINT:", checkpoint)
        print(
            "TYPE(MODEL_CFG):",
            type(model_cfg),
        )
        print(
            "TYPE(CHECKPOINT):",
            type(checkpoint),
        )

        sam2_model = build_sam2(
            model_cfg,
            checkpoint,
            device=device,
        )

        predictor = SAM2ImagePredictor(
            sam2_model
        )

        print("SAM2 Loaded Successfully")

        sam_window_size = 1024
        sam_overlap = 128
        sam_step = (
            sam_window_size - sam_overlap
        )

        height, width = clean_mask.shape

        refined_mask = np.zeros_like(
            clean_mask,
            dtype=np.uint8,
        )

        for y in range(0, height, sam_step):
            for x in range(0, width, sam_step):
                window_width = min(
                    sam_window_size,
                    width - x,
                )

                window_height = min(
                    sam_window_size,
                    height - y,
                )

                if (
                    window_width
                    < sam_window_size // 2
                    or window_height
                    < sam_window_size // 2
                ):
                    continue

                window = Window(
                    col_off=x,
                    row_off=y,
                    width=window_width,
                    height=window_height,
                )

                rgb_window = src.read(
                    [1, 2, 3],
                    window=window,
                ).transpose(1, 2, 0)

                mask_window = clean_mask[
                    y:y + window_height,
                    x:x + window_width,
                ]

                if np.sum(mask_window) < 100:
                    continue

                ys, xs = np.where(
                    mask_window > 0
                )

                if len(xs) == 0:
                    continue

                box = np.array(
                    [
                        xs.min(),
                        ys.min(),
                        xs.max(),
                        ys.max(),
                    ],
                    dtype=np.float32,
                )

                predictor.set_image(rgb_window)

                masks, scores, logits = (
                    predictor.predict(
                        box=box,
                        multimask_output=False,
                    )
                )

                refined = masks[0].astype(
                    np.uint8
                )

                refined_mask[
                    y:y + window_height,
                    x:x + window_width,
                ] = np.maximum(
                    refined_mask[
                        y:y + window_height,
                        x:x + window_width,
                    ],
                    refined,
                )

        cv2.imwrite(
            outpath(
                output_dir,
                "SAM2Mask.png",
            ),
            refined_mask * 255,
        )

        print("SAM2 refinement completed.")

        mask = (
            refined_mask > 0
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            raise ValueError(
                "No contours found in refined_mask."
            )

        largest_contour = max(
            contours,
            key=cv2.contourArea,
        )

        print(
            "Largest Contour Area:",
            cv2.contourArea(largest_contour),
        )

        largest_mask = np.zeros_like(
            mask,
            dtype=np.uint8,
        )

        cv2.drawContours(
            largest_mask,
            [largest_contour],
            -1,
            1,
            thickness=cv2.FILLED,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (15, 15),
        )

        largest_mask = cv2.morphologyEx(
            largest_mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        (
            num_labels,
            labels,
            stats,
            _,
        ) = cv2.connectedComponentsWithStats(
            largest_mask,
            connectivity=8,
        )

        final_mask = np.zeros_like(
            largest_mask,
            dtype=np.uint8,
        )

        min_area = 1000

        for i in range(1, num_labels):
            area = stats[
                i,
                cv2.CC_STAT_AREA,
            ]

            if area >= min_area:
                final_mask[labels == i] = 1

        print(
            "Remaining Components:",
            np.max(final_mask),
        )

        save_path = outpath(
            output_dir,
            "SAM2Mask.png",
        )

        cv2.imwrite(
            save_path,
            final_mask * 255,
        )

        print("Cleaned SAM2 mask saved.")
        print(save_path)
        
        
        # ======================================================
        # HYBRID SEGFORMER + SAM2 FALLBACK
        # ======================================================

        print("Creating Hybrid SegFormer + SAM2 mask...")

        # Step 1: Prepare the SegFormer fallback
        segformer_fallback = smooth_segformer_fallback(
            clean_mask
        )

        # Step 2: Get the cleaned SAM2 result
        sam2_result = (
            final_mask > 0
        ).astype(np.uint8)

        # Step 3: Fill SAM2 missing pixels using SegFormer
        hybrid_mask = np.where(
            sam2_result > 0,
            sam2_result,
            segformer_fallback
        ).astype(np.uint8)

        # Step 4: Save the hybrid mask separately
        hybrid_path = outpath(
            output_dir,
            "HybridRoadMask.png"
        )

        cv2.imwrite(
            hybrid_path,
            hybrid_mask * 255
        )

        print("Hybrid mask saved:", hybrid_path)

        # Step 5: Use hybrid mask for the next steps
        final_mask = hybrid_mask.copy()

        print("Hybrid fallback applied successfully.")

        save_path = outpath(
            output_dir,
            "Road.png",
        )

        cv2.imwrite(
            save_path,
            final_mask * 255,
        )

        print(
            "Cleaned SAM2 mask saved to:"
        )
        print(save_path)

        # ======================================================
        # CREATE FINAL ROAD SHAPEFILE
        # ======================================================

        mask = (
            final_mask > 0
        ).astype(np.uint8)

        (
            num_labels,
            labels,
            stats,
            _,
        ) = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        if num_labels <= 1:
            raise ValueError(
                "No road component found."
            )

        largest_label = (
            1
            + np.argmax(
                stats[
                    1:,
                    cv2.CC_STAT_AREA,
                ]
            )
        )

        largest_mask = np.zeros_like(
            mask,
            dtype=np.uint8,
        )

        largest_mask[labels == largest_label] = 1

        print(
            "Largest Area:",
            stats[
                largest_label,
                cv2.CC_STAT_AREA,
            ],
        )

        polygons = []

        for geom, value in shapes(
            largest_mask,
            mask=(largest_mask == 1),
            transform=src.transform,
        ):
            if value == 1:
                polygons.append(shape(geom))

        print("Polygons:", len(polygons))

        refined_polygon_px = refine_boundary(
            final_mask,
            src,
        )

        geo_polygon = pixel_polygon_to_geo(
            refined_polygon_px,
            src,
        )

        gdf = gpd.GeoDataFrame(
            {"Class": ["Road"]},
            geometry=[geo_polygon],
            crs=src.crs,
        )

        gdf["geometry"] = (
            gdf.geometry.buffer(0)
        )

        gdf.to_file(
            outpath(
                output_dir,
                "Road.shp",
            ),
            driver="ESRI Shapefile",
        )

        print("Saved shapefile.")

    except Exception:
        traceback.print_exc()

    finally:
        if src is not None:
            src.close()


if __name__ == "__main__":
    main()