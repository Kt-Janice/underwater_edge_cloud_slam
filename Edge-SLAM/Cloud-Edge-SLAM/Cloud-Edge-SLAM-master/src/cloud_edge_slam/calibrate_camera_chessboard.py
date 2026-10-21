#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-stage single-camera checkerboard calibration.

Supports:
1. pinhole + radial-tangential
2. OpenCV fisheye

Stage 1: coarse screening
- Downscale every image.
- Run fast classic checkerboard detection first.
- If it fails, optionally run a lightweight SB detection on the downscaled image.
- Images that clearly cannot provide a complete checkerboard are rejected here.
- NO camera calibration and NO reprojection-error rejection are performed here.

Stage 2: fine calibration
- Only coarse candidates are processed at full resolution.
- Prefer findChessboardCornersSB with high-accuracy flags.
- Fall back to classic findChessboardCorners + cornerSubPix.
- Calibrate with the requested camera model.
- Iteratively reject the single worst reprojection-error view and recalibrate.

Board dimension convention
--------------------------
By default, --squares-rows / --squares-cols mean PRINTED CHECKER SQUARES.

Therefore:
    6 x 8 printed squares -> 5 x 7 inner corners
    7 x 10 printed squares -> 6 x 9 inner corners

If your numbers already mean OpenCV INNER CORNERS, use:
    --dimension-type inner-corners

OpenCV patternSize is always ordered as:
    (inner_columns, inner_rows)

Examples
--------
Fisheye, calib.io board with 6 x 8 printed squares:
    python3 calibrate_camera_chessboard_v3.py \
        --model fisheye \
        --images ./frames \
        --squares-rows 6 \
        --squares-cols 8 \
        --dimension-type squares \
        --square-size-mm 25 \
        --output ./result/fisheye_cam \
        --coarse-scale 0.4 \
        --max-view-error-px 2.0 \
        --save-detections

Pinhole, calib.io board with 7 x 10 printed squares:
    python3 calibrate_camera_chessboard.py \
        --model pinhole \
        --images ./frames \
        --squares-rows 7 \
        --squares-cols 10 \
        --dimension-type squares \
        --square-size-mm 23.8 \
        --output ./result/normal_cam \
        --coarse-scale 0.5 \
        --max-view-error-px 2.0 \
        --save-detections
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-stage pinhole/fisheye checkerboard calibration."
    )

    parser.add_argument(
        "--model",
        choices=["pinhole", "fisheye"],
        required=True,
        help="Camera model used for final calibration.",
    )

    parser.add_argument(
        "--images",
        required=True,
        help="Directory containing calibration images.",
    )

    parser.add_argument(
        "--squares-rows",
        type=int,
        required=True,
        help=(
            "Board rows. By default this means printed checker-square rows. "
            "See --dimension-type."
        ),
    )

    parser.add_argument(
        "--squares-cols",
        type=int,
        required=True,
        help=(
            "Board columns. By default this means printed checker-square columns. "
            "See --dimension-type."
        ),
    )

    parser.add_argument(
        "--dimension-type",
        choices=["squares", "inner-corners"],
        default="squares",
        help=(
            "Interpret --squares-rows/cols as printed squares or OpenCV inner "
            "corners. Default: squares."
        ),
    )

    parser.add_argument(
        "--square-size-mm",
        type=float,
        required=True,
        help="Measured physical checker-square side length in millimeters.",
    )

    parser.add_argument(
        "--output",
        default="./camera_calibration",
        help="Output file prefix.",
    )

    # ----------------------------
    # Stage 1: coarse screening
    # ----------------------------
    parser.add_argument(
        "--coarse-scale",
        type=float,
        default=0.4,
        help=(
            "Downscale factor used only for coarse checkerboard screening. "
            "Default: 0.4."
        ),
    )

    parser.add_argument(
        "--coarse-no-sb-fallback",
        action="store_true",
        help=(
            "Disable lightweight findChessboardCornersSB fallback during coarse "
            "screening. Faster, but may reject more difficult valid views."
        ),
    )

    parser.add_argument(
        "--candidate-stride",
        type=int,
        default=1,
        help=(
            "After coarse screening, keep every Nth candidate for fine processing. "
            "Default: 1 (keep all). Useful for highly redundant video frames."
        ),
    )

    # ----------------------------
    # Stage 2: fine calibration
    # ----------------------------
    parser.add_argument(
        "--min-valid-images",
        type=int,
        default=15,
        help="Minimum number of fine-stage images retained after rejection.",
    )

    parser.add_argument(
        "--max-view-error-px",
        type=float,
        default=2.0,
        help=(
            "Iteratively reject the single worst view until the worst retained "
            "view is <= this reprojection RMSE, or stopping conditions are met. "
            "Default: 2.0 px."
        ),
    )

    parser.add_argument(
        "--max-reject-rounds",
        type=int,
        default=30,
        help="Maximum number of iterative reprojection-rejection rounds.",
    )

    parser.add_argument(
        "--estimate-k3",
        action="store_true",
        help=(
            "Pinhole only: estimate k3. Default fixes k3=0 for a common "
            "four-parameter radial-tangential model."
        ),
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display full-resolution fine-stage corner detections.",
    )

    parser.add_argument(
        "--save-detections",
        action="store_true",
        help="Save full-resolution fine-stage checkerboard detection previews.",
    )

    return parser.parse_args()


def collect_images(directory):
    directory = Path(directory).expanduser().resolve()

    if not directory.is_dir():
        raise RuntimeError(f"Image directory does not exist: {directory}")

    images = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    images.sort()
    return images


def resolve_board_dimensions(rows, cols, dimension_type):
    if dimension_type == "squares":
        if rows < 3 or cols < 3:
            raise RuntimeError(
                "A checkerboard specified by printed squares must contain at "
                "least 3 x 3 squares."
            )

        inner_rows = rows - 1
        inner_cols = cols - 1
    else:
        if rows < 2 or cols < 2:
            raise RuntimeError(
                "OpenCV inner-corner dimensions must both be >= 2."
            )

        inner_rows = rows
        inner_cols = cols

    return inner_rows, inner_cols


def build_object_points(inner_cols, inner_rows, square_size_mm):
    points = np.zeros(
        (inner_rows * inner_cols, 3),
        dtype=np.float64,
    )

    grid = np.mgrid[
        0:inner_cols,
        0:inner_rows,
    ].T.reshape(-1, 2)

    points[:, :2] = grid.astype(np.float64)
    points[:, :2] *= float(square_size_mm)

    return points


def resize_for_coarse(gray, scale):
    if scale == 1.0:
        return gray

    return cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


def coarse_checkerboard_detect(gray_small, pattern_size, allow_sb_fallback):
    """
    Cheap first-stage detection.

    First:
        classic findChessboardCorners with FAST_CHECK.

    Fallback:
        lightweight findChessboardCornersSB on the already-downscaled image,
        without EXHAUSTIVE / ACCURACY flags.

    This stage only decides whether the image is worth expensive fine processing.
    """

    classic_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )

    try:
        found, _ = cv2.findChessboardCorners(
            gray_small,
            pattern_size,
            flags=classic_flags,
        )
    except cv2.error:
        found = False

    if found:
        return True, "CLASSIC_FAST"

    if (
        allow_sb_fallback
        and hasattr(cv2, "findChessboardCornersSB")
    ):
        sb_flags = 0

        if hasattr(cv2, "CALIB_CB_NORMALIZE_IMAGE"):
            sb_flags |= cv2.CALIB_CB_NORMALIZE_IMAGE

        try:
            found, _ = cv2.findChessboardCornersSB(
                gray_small,
                pattern_size,
                flags=sb_flags,
            )
        except cv2.error:
            found = False

        if found:
            return True, "SB_FAST"

    return False, "MISS"


def coarse_screen(
    image_paths,
    pattern_size,
    scale,
    allow_sb_fallback,
    csv_path,
):
    """
    Stage 1.

    All images are scanned cheaply. No calibration is performed.
    """

    candidates = []
    records = []
    expected_size = None

    start_time = time.time()

    for index, path in enumerate(image_paths, start=1):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            records.append(
                [str(path), 0, "READ_FAIL", "", "", ""]
            )

            print(
                f"[COARSE READ FAIL] "
                f"{index:04d}/{len(image_paths):04d} {path.name}"
            )
            continue

        original_size = (
            int(image.shape[1]),
            int(image.shape[0]),
        )

        if expected_size is None:
            expected_size = original_size

        if original_size != expected_size:
            records.append(
                [
                    str(path),
                    0,
                    "SIZE_MISMATCH",
                    original_size[0],
                    original_size[1],
                    "",
                ]
            )

            print(
                f"[COARSE SIZE FAIL] "
                f"{index:04d}/{len(image_paths):04d} "
                f"{path.name}: {original_size} != {expected_size}"
            )
            continue

        small = resize_for_coarse(
            image,
            scale,
        )

        found, method = coarse_checkerboard_detect(
            small,
            pattern_size,
            allow_sb_fallback,
        )

        records.append(
            [
                str(path),
                int(found),
                method,
                original_size[0],
                original_size[1],
                f"{scale:.6f}",
            ]
        )

        if found:
            candidates.append(path)
            label = "KEEP"
        else:
            label = "DROP"

        if (
            found
            or index == 1
            or index % 50 == 0
            or index == len(image_paths)
        ):
            print(
                f"[COARSE {label}] "
                f"{index:04d}/{len(image_paths):04d} "
                f"{path.name} ({method})"
            )

    elapsed = time.time() - start_time

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "image",
                "coarse_candidate",
                "coarse_method",
                "image_width",
                "image_height",
                "coarse_scale",
            ]
        )

        writer.writerows(records)

    return candidates, expected_size, elapsed


def fine_detect_sb(gray, pattern_size):
    if not hasattr(cv2, "findChessboardCornersSB"):
        return False, None

    flags = 0

    if hasattr(cv2, "CALIB_CB_NORMALIZE_IMAGE"):
        flags |= cv2.CALIB_CB_NORMALIZE_IMAGE

    if hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
        flags |= cv2.CALIB_CB_EXHAUSTIVE

    if hasattr(cv2, "CALIB_CB_ACCURACY"):
        flags |= cv2.CALIB_CB_ACCURACY

    try:
        found, corners = cv2.findChessboardCornersSB(
            gray,
            pattern_size,
            flags=flags,
        )
    except cv2.error:
        return False, None

    if not found:
        return False, None

    return True, corners.astype(np.float64)


def fine_detect_classic(gray, pattern_size):
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    try:
        found, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            flags=flags,
        )
    except cv2.error:
        return False, None

    if not found:
        return False, None

    criteria = (
        cv2.TERM_CRITERIA_EPS
        | cv2.TERM_CRITERIA_MAX_ITER,
        50,
        1e-4,
    )

    corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        criteria,
    )

    return True, corners.astype(np.float64)


def fine_detect_candidates(
    candidate_paths,
    pattern_size,
    object_template,
    show,
    save_detections,
    detections_dir,
):
    object_points = []
    image_points = []
    valid_paths = []

    detection_method = {}
    image_size = None

    start_time = time.time()

    for index, path in enumerate(candidate_paths, start=1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if image is None:
            detection_method[str(path)] = "READ_FAIL"

            print(
                f"[FINE READ FAIL] "
                f"{index:04d}/{len(candidate_paths):04d} {path.name}"
            )
            continue

        current_size = (
            int(image.shape[1]),
            int(image.shape[0]),
        )

        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise RuntimeError(
                "Fine-stage image resolution mismatch: "
                f"{path.name}: {current_size}, expected {image_size}"
            )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        found, corners = fine_detect_sb(
            gray,
            pattern_size,
        )

        method = "SB_FINE"

        if not found:
            found, corners = fine_detect_classic(
                gray,
                pattern_size,
            )
            method = "CLASSIC_FINE"

        if not found:
            detection_method[str(path)] = "FINE_MISS"

            print(
                f"[FINE MISS] "
                f"{index:04d}/{len(candidate_paths):04d} {path.name}"
            )
            continue

        detection_method[str(path)] = method

        object_points.append(
            object_template.copy()
        )

        image_points.append(
            corners
        )

        valid_paths.append(
            str(path)
        )

        print(
            f"[FINE OK-{method}] "
            f"{index:04d}/{len(candidate_paths):04d} {path.name}"
        )

        if show or save_detections:
            preview = image.copy()

            cv2.drawChessboardCorners(
                preview,
                pattern_size,
                corners.astype(np.float32),
                True,
            )

            if save_detections:
                cv2.imwrite(
                    str(detections_dir / path.name),
                    preview,
                )

            if show:
                display = preview
                max_width = 1400

                if display.shape[1] > max_width:
                    display_scale = (
                        max_width / display.shape[1]
                    )

                    display = cv2.resize(
                        display,
                        None,
                        fx=display_scale,
                        fy=display_scale,
                    )

                cv2.imshow(
                    "Fine checkerboard detection",
                    display,
                )

                key = cv2.waitKey(100)

                if key == 27:
                    print("[INFO] ESC pressed.")
                    break

    if show:
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time

    return (
        object_points,
        image_points,
        valid_paths,
        detection_method,
        image_size,
        elapsed,
    )


def calibrate_pinhole(
    object_points,
    image_points,
    image_size,
    estimate_k3,
):
    obj = [
        points.astype(np.float32)
        for points in object_points
    ]

    img = [
        points.astype(np.float32)
        for points in image_points
    ]

    flags = 0

    if not estimate_k3:
        flags |= cv2.CALIB_FIX_K3

    return cv2.calibrateCamera(
        obj,
        img,
        image_size,
        None,
        None,
        flags=flags,
    )


def calibrate_fisheye(
    object_points,
    image_points,
    image_size,
):
    fisheye_obj = []
    fisheye_img = []

    for object_point, image_point in zip(
        object_points,
        image_points,
    ):
        fisheye_obj.append(
            object_point.reshape(
                -1,
                1,
                3,
            ).astype(np.float64)
        )

        fisheye_img.append(
            image_point.reshape(
                -1,
                1,
                2,
            ).astype(np.float64)
        )

    camera_matrix = np.zeros(
        (3, 3),
        dtype=np.float64,
    )

    distortion = np.zeros(
        (4, 1),
        dtype=np.float64,
    )

    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_FIX_SKEW
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS
        | cv2.TERM_CRITERIA_MAX_ITER,
        200,
        1e-8,
    )

    return cv2.fisheye.calibrate(
        fisheye_obj,
        fisheye_img,
        image_size,
        camera_matrix,
        distortion,
        None,
        None,
        flags=flags,
        criteria=criteria,
    )


def calibrate_model(
    model,
    object_points,
    image_points,
    image_size,
    estimate_k3,
):
    if model == "pinhole":
        return calibrate_pinhole(
            object_points,
            image_points,
            image_size,
            estimate_k3,
        )

    return calibrate_fisheye(
        object_points,
        image_points,
        image_size,
    )


def project_pinhole(
    object_point,
    rvec,
    tvec,
    camera_matrix,
    distortion,
):
    projected, _ = cv2.projectPoints(
        object_point,
        rvec,
        tvec,
        camera_matrix,
        distortion,
    )

    return projected.reshape(-1, 2)


def project_fisheye(
    object_point,
    rvec,
    tvec,
    camera_matrix,
    distortion,
):
    obj = object_point.reshape(
        -1,
        1,
        3,
    ).astype(np.float64)

    projected, _ = cv2.fisheye.projectPoints(
        obj,
        rvec,
        tvec,
        camera_matrix,
        distortion,
    )

    return projected.reshape(-1, 2)


def calculate_reprojection_errors(
    model,
    object_points,
    image_points,
    rvecs,
    tvecs,
    camera_matrix,
    distortion,
):
    per_view_rmse = []
    total_squared_error = 0.0
    total_points = 0

    for (
        object_point,
        image_point,
        rvec,
        tvec,
    ) in zip(
        object_points,
        image_points,
        rvecs,
        tvecs,
    ):
        if model == "pinhole":
            projected = project_pinhole(
                object_point,
                rvec,
                tvec,
                camera_matrix,
                distortion,
            )
        else:
            projected = project_fisheye(
                object_point,
                rvec,
                tvec,
                camera_matrix,
                distortion,
            )

        observed = image_point.reshape(-1, 2)
        residual = observed - projected
        squared_error = np.sum(residual * residual, axis=1)

        rmse = math.sqrt(float(np.mean(squared_error)))

        per_view_rmse.append(rmse)
        total_squared_error += float(np.sum(squared_error))
        total_points += len(squared_error)

    global_rmse = math.sqrt(
        total_squared_error / total_points
    )

    return global_rmse, per_view_rmse


def iterative_rejection(
    model,
    all_object_points,
    all_image_points,
    all_paths,
    image_size,
    estimate_k3,
    min_valid_images,
    max_view_error_px,
    max_reject_rounds,
):
    active_indices = list(range(len(all_paths)))
    rejected_info = {}

    round_index = 0

    while True:
        current_object_points = [
            all_object_points[i]
            for i in active_indices
        ]

        current_image_points = [
            all_image_points[i]
            for i in active_indices
        ]

        (
            calibration_rms,
            camera_matrix,
            distortion,
            rvecs,
            tvecs,
        ) = calibrate_model(
            model,
            current_object_points,
            current_image_points,
            image_size,
            estimate_k3,
        )

        (
            reprojection_rmse,
            per_view_rmse,
        ) = calculate_reprojection_errors(
            model,
            current_object_points,
            current_image_points,
            rvecs,
            tvecs,
            camera_matrix,
            distortion,
        )

        worst_local_index = int(
            np.argmax(per_view_rmse)
        )

        worst_error = float(
            per_view_rmse[worst_local_index]
        )

        worst_global_index = active_indices[
            worst_local_index
        ]

        print()
        print(f"[CALIB ROUND {round_index}]")
        print(
            f"  active images       : "
            f"{len(active_indices)}"
        )
        print(
            f"  global RMSE         : "
            f"{reprojection_rmse:.6f} px"
        )
        print(
            f"  worst view          : "
            f"{Path(all_paths[worst_global_index]).name}"
        )
        print(
            f"  worst view RMSE     : "
            f"{worst_error:.6f} px"
        )

        if worst_error <= max_view_error_px:
            print(
                "[AUTO-REJECT] "
                "All retained views satisfy threshold."
            )
            break

        if len(active_indices) <= min_valid_images:
            print(
                "[AUTO-REJECT WARNING] "
                "Minimum retained-image count reached."
            )
            break

        if round_index >= max_reject_rounds:
            print(
                "[AUTO-REJECT WARNING] "
                "Maximum rejection rounds reached."
            )
            break

        rejected_info[
            worst_global_index
        ] = {
            "round": round_index + 1,
            "error": worst_error,
        }

        print(
            f"[AUTO-REJECT] Removing: "
            f"{Path(all_paths[worst_global_index]).name}"
        )

        del active_indices[
            worst_local_index
        ]

        round_index += 1

    final_object_points = [
        all_object_points[i]
        for i in active_indices
    ]

    final_image_points = [
        all_image_points[i]
        for i in active_indices
    ]

    (
        final_rms,
        camera_matrix,
        distortion,
        rvecs,
        tvecs,
    ) = calibrate_model(
        model,
        final_object_points,
        final_image_points,
        image_size,
        estimate_k3,
    )

    (
        final_reprojection_rmse,
        final_per_view_rmse,
    ) = calculate_reprojection_errors(
        model,
        final_object_points,
        final_image_points,
        rvecs,
        tvecs,
        camera_matrix,
        distortion,
    )

    return (
        active_indices,
        rejected_info,
        final_rms,
        final_reprojection_rmse,
        final_per_view_rmse,
        camera_matrix,
        distortion,
    )


def write_opencv_yaml(
    path,
    model,
    image_size,
    camera_matrix,
    distortion,
    calibration_rms,
    reprojection_rmse,
):
    fs = cv2.FileStorage(
        str(path),
        cv2.FILE_STORAGE_WRITE,
    )

    if not fs.isOpened():
        raise RuntimeError(
            f"Cannot open YAML: {path}"
        )

    fs.write("camera_model", model)
    fs.write("image_width", int(image_size[0]))
    fs.write("image_height", int(image_size[1]))
    fs.write("calibration_rms", float(calibration_rms))
    fs.write(
        "reprojection_rmse_px",
        float(reprojection_rmse),
    )
    fs.write("camera_matrix", camera_matrix)
    fs.write(
        "distortion_coefficients",
        distortion.reshape(1, -1),
    )

    fs.release()


def main():
    args = parse_args()

    if args.square_size_mm <= 0.0:
        print(
            "[ERROR] --square-size-mm must be > 0."
        )
        return 1

    if not (0.1 <= args.coarse_scale <= 1.0):
        print(
            "[ERROR] --coarse-scale must be in [0.1, 1.0]."
        )
        return 1

    if args.candidate_stride < 1:
        print(
            "[ERROR] --candidate-stride must be >= 1."
        )
        return 1

    if args.min_valid_images < 3:
        print(
            "[ERROR] --min-valid-images must be >= 3."
        )
        return 1

    if args.max_view_error_px <= 0.0:
        print(
            "[ERROR] --max-view-error-px must be > 0."
        )
        return 1

    try:
        inner_rows, inner_cols = resolve_board_dimensions(
            args.squares_rows,
            args.squares_cols,
            args.dimension_type,
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 1

    # OpenCV patternSize order:
    # (inner columns, inner rows)
    pattern_size = (
        inner_cols,
        inner_rows,
    )

    try:
        image_paths = collect_images(
            args.images
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 1

    if not image_paths:
        print("[ERROR] No images found.")
        return 1

    output_prefix = (
        Path(args.output)
        .expanduser()
        .resolve()
    )

    output_prefix.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    coarse_csv = Path(
        str(output_prefix)
        + "_coarse_screen.csv"
    )

    candidates_txt = Path(
        str(output_prefix)
        + "_candidates.txt"
    )

    fine_csv = Path(
        str(output_prefix)
        + "_per_view.csv"
    )

    rejected_txt = Path(
        str(output_prefix)
        + "_rejected.txt"
    )

    report_txt = Path(
        str(output_prefix)
        + "_report.txt"
    )

    opencv_yaml = Path(
        str(output_prefix)
        + "_opencv.yaml"
    )

    okvis_txt = Path(
        str(output_prefix)
        + "_okvis_values.txt"
    )

    detections_dir = Path(
        str(output_prefix)
        + "_detections"
    )

    if args.save_detections:
        detections_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    object_template = build_object_points(
        inner_cols,
        inner_rows,
        args.square_size_mm,
    )

    print(
        "============================================================"
    )
    print(
        "Two-stage camera calibration"
    )
    print(
        "============================================================"
    )
    print(
        f"Camera model          : {args.model}"
    )
    print(
        f"Images found          : {len(image_paths)}"
    )
    print(
        f"Dimension type        : {args.dimension_type}"
    )
    print(
        f"Input dimensions      : "
        f"{args.squares_rows} x {args.squares_cols}"
    )
    print(
        f"OpenCV inner corners  : "
        f"{inner_rows} rows x {inner_cols} cols"
    )
    print(
        f"OpenCV patternSize    : "
        f"({inner_cols}, {inner_rows})"
    )
    print(
        f"Checker size          : "
        f"{args.square_size_mm:.6f} mm"
    )
    print(
        f"Coarse scale          : "
        f"{args.coarse_scale:.3f}"
    )
    print(
        f"Coarse SB fallback    : "
        f"{not args.coarse_no_sb_fallback}"
    )
    print(
        f"Candidate stride      : "
        f"{args.candidate_stride}"
    )
    print(
        "============================================================"
    )

    # ============================================================
    # Stage 1: coarse screening
    # ============================================================

    print()
    print(
        "================ STAGE 1: COARSE SCREEN ================"
    )

    (
        coarse_candidates,
        coarse_image_size,
        coarse_elapsed,
    ) = coarse_screen(
        image_paths,
        pattern_size,
        args.coarse_scale,
        not args.coarse_no_sb_fallback,
        coarse_csv,
    )

    if args.candidate_stride > 1:
        fine_candidates = coarse_candidates[
            ::args.candidate_stride
        ]
    else:
        fine_candidates = coarse_candidates

    candidates_txt.write_text(
        "\n".join(
            str(path)
            for path in fine_candidates
        )
        + (
            "\n"
            if fine_candidates
            else ""
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"[COARSE SUMMARY] total       : {len(image_paths)}"
    )
    print(
        f"[COARSE SUMMARY] candidates  : {len(coarse_candidates)}"
    )
    print(
        f"[COARSE SUMMARY] fine input  : {len(fine_candidates)}"
    )
    print(
        f"[COARSE SUMMARY] elapsed     : {coarse_elapsed:.2f} s"
    )
    print(
        f"[COARSE SUMMARY] result CSV  : {coarse_csv}"
    )

    if len(fine_candidates) < args.min_valid_images:
        print()
        print(
            "[ERROR] Too few coarse candidates for fine calibration."
        )
        print(
            f"        Required at least: {args.min_valid_images}"
        )
        print(
            f"        Fine candidates  : {len(fine_candidates)}"
        )
        print(
            "        Try increasing --coarse-scale or allowing the "
            "coarse SB fallback."
        )
        return 1

    # ============================================================
    # Stage 2: full-resolution fine detection
    # ============================================================

    print()
    print(
        "================ STAGE 2: FINE DETECTION ================"
    )

    try:
        (
            object_points,
            image_points,
            valid_paths,
            fine_method,
            image_size,
            fine_elapsed,
        ) = fine_detect_candidates(
            fine_candidates,
            pattern_size,
            object_template,
            args.show,
            args.save_detections,
            detections_dir,
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 1

    fine_count = len(valid_paths)

    print()
    print(
        f"[FINE SUMMARY] input candidates : {len(fine_candidates)}"
    )
    print(
        f"[FINE SUMMARY] valid boards     : {fine_count}"
    )
    print(
        f"[FINE SUMMARY] elapsed          : {fine_elapsed:.2f} s"
    )

    if fine_count < args.min_valid_images:
        print(
            "[ERROR] Too few full-resolution fine detections."
        )
        return 1

    # ============================================================
    # Stage 2B: calibration + iterative rejection
    # ============================================================

    print()
    print(
        "================ STAGE 2B: CALIBRATION =================="
    )

    try:
        (
            active_indices,
            rejected_info,
            calibration_rms,
            reprojection_rmse,
            final_per_view_rmse,
            camera_matrix,
            distortion,
        ) = iterative_rejection(
            args.model,
            object_points,
            image_points,
            valid_paths,
            image_size,
            args.estimate_k3,
            args.min_valid_images,
            args.max_view_error_px,
            args.max_reject_rounds,
        )
    except cv2.error as exc:
        print(
            "[ERROR] OpenCV calibration failed:"
        )
        print(exc)
        return 1

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    d = distortion.reshape(-1)

    write_opencv_yaml(
        opencv_yaml,
        args.model,
        image_size,
        camera_matrix,
        distortion,
        calibration_rms,
        reprojection_rmse,
    )

    active_set = set(active_indices)

    final_error_map = {
        global_index: final_per_view_rmse[local_index]
        for local_index, global_index
        in enumerate(active_indices)
    }

    # ============================================================
    # Fine-stage CSV
    # ============================================================

    with fine_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "image",
                "fine_detector",
                "final_used",
                "rejected_round",
                "rejection_rmse_px",
                "final_reprojection_rmse_px",
            ]
        )

        for global_index, path_string in enumerate(valid_paths):
            method = fine_method.get(
                path_string,
                "",
            )

            if global_index in active_set:
                writer.writerow(
                    [
                        path_string,
                        method,
                        1,
                        "",
                        "",
                        f"{final_error_map[global_index]:.9f}",
                    ]
                )
            else:
                info = rejected_info.get(
                    global_index,
                    {},
                )

                rejected_error = info.get(
                    "error",
                    float("nan"),
                )

                writer.writerow(
                    [
                        path_string,
                        method,
                        0,
                        info.get("round", ""),
                        f"{rejected_error:.9f}",
                        "",
                    ]
                )

    # ============================================================
    # Rejected list
    # ============================================================

    rejected_lines = []

    for global_index, info in sorted(
        rejected_info.items(),
        key=lambda item: item[1]["round"],
    ):
        rejected_lines.append(
            f"round={info['round']:02d}, "
            f"rmse={info['error']:.6f}, "
            f"image={valid_paths[global_index]}"
        )

    rejected_txt.write_text(
        "\n".join(rejected_lines)
        + (
            "\n"
            if rejected_lines
            else ""
        ),
        encoding="utf-8",
    )

    # ============================================================
    # Report / model-specific output
    # ============================================================

    report = [
        "============================================================",
        "Final two-stage camera calibration report",
        "============================================================",
        f"camera_model                 : {args.model}",
        f"image_width                  : {image_size[0]}",
        f"image_height                 : {image_size[1]}",
        f"images_found                 : {len(image_paths)}",
        f"coarse_candidates            : {len(coarse_candidates)}",
        f"fine_candidates_after_stride : {len(fine_candidates)}",
        f"fine_checkerboards_detected  : {fine_count}",
        f"final_images_used            : {len(active_indices)}",
        f"images_auto_rejected         : {len(rejected_info)}",
        f"dimension_type               : {args.dimension_type}",
        (
            f"input_board_dimensions       : "
            f"{args.squares_rows} x {args.squares_cols}"
        ),
        (
            f"opencv_inner_corners         : "
            f"{inner_rows} rows x {inner_cols} cols"
        ),
        (
            f"opencv_pattern_size          : "
            f"({inner_cols}, {inner_rows})"
        ),
        f"checker_size_mm              : {args.square_size_mm:.9f}",
        f"coarse_scale                 : {args.coarse_scale:.9f}",
        (
            f"max_view_error_threshold_px  : "
            f"{args.max_view_error_px:.9f}"
        ),
        f"opencv_calibration_rms       : {calibration_rms:.9f}",
        f"reprojection_rmse_px         : {reprojection_rmse:.9f}",
        "",
        "Intrinsics:",
        f"fx = {fx:.12f}",
        f"fy = {fy:.12f}",
        f"cx = {cx:.12f}",
        f"cy = {cy:.12f}",
        "",
    ]

    if args.model == "pinhole":
        k1 = float(d[0])
        k2 = float(d[1])
        p1 = float(d[2])
        p2 = float(d[3])
        k3 = float(d[4]) if len(d) > 4 else 0.0

        report.extend(
            [
                "Pinhole radial-tangential distortion:",
                f"k1 = {k1:.12f}",
                f"k2 = {k2:.12f}",
                f"p1 = {p1:.12f}",
                f"p2 = {p2:.12f}",
                f"k3 = {k3:.12f}",
            ]
        )

        okvis_lines = [
            "# Pinhole radial-tangential calibration.",
            (
                f"image_dimension: "
                f"[{image_size[0]}, {image_size[1]}]"
            ),
            (
                f"focal_length: "
                f"[{fx:.12f}, {fy:.12f}]"
            ),
            (
                f"principal_point: "
                f"[{cx:.12f}, {cy:.12f}]"
            ),
            (
                "distortion_coefficients: "
                f"[{k1:.12f}, {k2:.12f}, "
                f"{p1:.12f}, {p2:.12f}]"
            ),
            "distortion_type: radialtangential",
        ]

    else:
        k1 = float(d[0])
        k2 = float(d[1])
        k3 = float(d[2])
        k4 = float(d[3])

        report.extend(
            [
                "OpenCV fisheye distortion:",
                f"k1 = {k1:.12f}",
                f"k2 = {k2:.12f}",
                f"k3 = {k3:.12f}",
                f"k4 = {k4:.12f}",
            ]
        )

        okvis_lines = [
            "# OpenCV fisheye calibration.",
            "# Verify the exact mathematical compatibility with",
            "# your OKVIS/SVIn2 equidistant/fisheye model before",
            "# copying these coefficients into the production config.",
            (
                f"image_dimension: "
                f"[{image_size[0]}, {image_size[1]}]"
            ),
            (
                f"focal_length: "
                f"[{fx:.12f}, {fy:.12f}]"
            ),
            (
                f"principal_point: "
                f"[{cx:.12f}, {cy:.12f}]"
            ),
            (
                "distortion_coefficients: "
                f"[{k1:.12f}, {k2:.12f}, "
                f"{k3:.12f}, {k4:.12f}]"
            ),
        ]

    report.extend(
        [
            "",
            "Output files:",
            f"Coarse CSV  : {coarse_csv}",
            f"Candidates  : {candidates_txt}",
            f"Fine CSV    : {fine_csv}",
            f"Rejected    : {rejected_txt}",
            f"OpenCV YAML : {opencv_yaml}",
            f"OKVIS values: {okvis_txt}",
            "============================================================",
        ]
    )

    report_txt.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    okvis_lines.extend(
        [
            "",
            f"# Calibration RMS: {calibration_rms:.9f} px",
            f"# Reprojection RMSE: {reprojection_rmse:.9f} px",
        ]
    )

    okvis_txt.write_text(
        "\n".join(okvis_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("\n".join(report))

    final_worst = sorted(
        [
            (
                valid_paths[global_index],
                final_error_map[global_index],
            )
            for global_index in active_indices
        ],
        key=lambda item: item[1],
        reverse=True,
    )

    print()
    print("Worst retained views:")

    for rank, (path_string, error) in enumerate(
        final_worst[:10],
        start=1,
    ):
        print(
            f"  {rank:02d}. "
            f"{Path(path_string).name}: "
            f"{error:.6f} px"
        )

    print()
    print("[DONE] Two-stage calibration completed.")

    if (
        final_worst
        and final_worst[0][1]
        > args.max_view_error_px
    ):
        print(
            "[WARNING] Final worst-view error is still above "
            "the requested threshold."
        )
        print(
            "          Prefer acquiring better viewpoint/edge coverage "
            "instead of repeatedly deleting more images."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
