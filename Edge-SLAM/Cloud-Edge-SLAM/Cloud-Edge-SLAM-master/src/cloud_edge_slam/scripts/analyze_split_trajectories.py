#!/usr/bin/env python3
"""Analyze split Cloud/Edge trajectories exported by cloud_edge_demo.cpp.

The script is diagnostic only: it reads existing text/CSV outputs and writes
summary CSV/TXT files into the selected result directory.
"""

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ASSOCIATION_MAX_DIFF = 0.01
DUPLICATE_TIME_EPS = 1e-9
IDENTICAL_POSITION_EPS = 1e-9
MIN_ALIGNMENT_ASSOCIATIONS = 3
TIME_BALANCED_BIN_SECONDS = 0.5
SIGNIFICANT_Z_ERROR_M = 0.02
SIGNIFICANT_ANGLE_DEG = 1.0

EXPECTED_INPUT_FILES = [
    "whole_map.txt",
    "whole_map_sorted_unique.txt",
    "whole_map_no_cloud.txt",
    "whole_map_cloud_only.txt",
    "map_1_all.txt",
    "map_1_edge_only.txt",
    "map_1_cloud_only.txt",
    "map_2_all.txt",
    "map_2_edge_only.txt",
    "map_2_cloud_only.txt",
    "keyframe_source_debug.csv",
    "keyframe_source_summary.txt",
    "okvis_wrapper.txt",
    "okvis_full_traj.txt",
    "cloud_merge_001_head_aligned_only.txt",
    "cloud_merge_001_sim3_aligned_only.txt",
    "cloud_merge_001_after_two_anchor_correction.txt",
    "cloud_merge_001_after_sim3_only_correction.txt",
    "cloud_merge_001_after_selected_correction.txt",
]

TUM_TRAJECTORY_FILES = [
    "whole_map.txt",
    "whole_map_sorted_unique.txt",
    "whole_map_no_cloud.txt",
    "whole_map_cloud_only.txt",
    "map_1_all.txt",
    "map_1_edge_only.txt",
    "map_1_cloud_only.txt",
    "map_2_all.txt",
    "map_2_edge_only.txt",
    "map_2_cloud_only.txt",
    "okvis_wrapper.txt",
    "okvis_full_traj.txt",
    "cloud_merge_001_head_aligned_only.txt",
    "cloud_merge_001_sim3_aligned_only.txt",
    "cloud_merge_001_after_two_anchor_correction.txt",
    "cloud_merge_001_after_sim3_only_correction.txt",
    "cloud_merge_001_after_selected_correction.txt",
]

INDEPENDENT_SE3_EVAL_FILES = [
    "okvis_wrapper.txt",
    "whole_map_no_cloud.txt",
    "whole_map_cloud_only.txt",
    "whole_map_sorted_unique.txt",
    "whole_map.txt",
    "map_1_edge_only.txt",
    "map_1_cloud_only.txt",
    "map_1_all.txt",
    "map_2_edge_only.txt",
    "cloud_merge_001_head_aligned_only.txt",
    "cloud_merge_001_sim3_aligned_only.txt",
    "cloud_merge_001_after_two_anchor_correction.txt",
    "cloud_merge_001_after_sim3_only_correction.txt",
    "cloud_merge_001_after_selected_correction.txt",
]

NO_CLOUD_ALIGNMENT_EVAL_FILES = [
    "whole_map_cloud_only.txt",
    "whole_map_sorted_unique.txt",
    "whole_map.txt",
    "map_1_cloud_only.txt",
    "map_1_all.txt",
    "cloud_merge_001_head_aligned_only.txt",
    "cloud_merge_001_sim3_aligned_only.txt",
    "cloud_merge_001_after_two_anchor_correction.txt",
    "cloud_merge_001_after_sim3_only_correction.txt",
    "cloud_merge_001_after_selected_correction.txt",
]

PAIR_OVERLAP_FILES = [
    ("whole_map_no_cloud.txt", "map_1_edge_only.txt"),
    ("whole_map_cloud_only.txt", "map_1_cloud_only.txt"),
    ("whole_map_sorted_unique.txt", "whole_map.txt"),
    ("whole_map_no_cloud.txt", "whole_map_sorted_unique.txt"),
]

AXIS_ERROR_FILES = [
    "okvis_full_traj.txt",
    "okvis_wrapper.txt",
    "whole_map.txt",
    "whole_map_sorted_unique.txt",
    "whole_map_no_cloud.txt",
    "whole_map_cloud_only.txt",
    "map_1_edge_only.txt",
    "map_1_cloud_only.txt",
    "map_1_all.txt",
    "cloud_merge_001_head_aligned_only.txt",
    "cloud_merge_001_sim3_aligned_only.txt",
    "cloud_merge_001_after_two_anchor_correction.txt",
    "cloud_merge_001_after_sim3_only_correction.txt",
    "cloud_merge_001_after_selected_correction.txt",
]

EDGE_FIXED_SE3_FILES = [
    "whole_map.txt",
    "whole_map_sorted_unique.txt",
    "whole_map_cloud_only.txt",
    "map_1_cloud_only.txt",
    "map_1_all.txt",
    "cloud_merge_001_head_aligned_only.txt",
    "cloud_merge_001_sim3_aligned_only.txt",
    "cloud_merge_001_after_two_anchor_correction.txt",
    "cloud_merge_001_after_sim3_only_correction.txt",
    "cloud_merge_001_after_selected_correction.txt",
]

SEGMENT_Z_ERROR_FILES = [
    "whole_map.txt",
    "map_1_all.txt",
]

TIME_BALANCED_FILES = [
    "whole_map.txt",
    "map_1_edge_only.txt",
    "map_1_cloud_only.txt",
    "map_1_all.txt",
    "okvis_full_traj.txt",
    "okvis_wrapper.txt",
    "cloud_merge_001_head_aligned_only.txt",
    "cloud_merge_001_sim3_aligned_only.txt",
    "cloud_merge_001_after_two_anchor_correction.txt",
    "cloud_merge_001_after_sim3_only_correction.txt",
    "cloud_merge_001_after_selected_correction.txt",
]

CLOUD_CORRECTION_STAGE_FILES = [
    "cloud_merge_001_head_aligned_only.txt",
    "cloud_merge_001_sim3_aligned_only.txt",
    "cloud_merge_001_after_two_anchor_correction.txt",
    "cloud_merge_001_after_sim3_only_correction.txt",
    "cloud_merge_001_after_selected_correction.txt",
]


@dataclass
class Trajectory:
    name: str
    path: str
    timestamps: np.ndarray
    positions: np.ndarray
    quaternions: np.ndarray
    duplicate_timestamp_count: int
    status: str


@dataclass
class AteResult:
    rmse: Optional[float]
    mean: Optional[float]
    max_error: Optional[float]
    associated_count: int
    status: str


@dataclass
class RigidTransform:
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, positions: np.ndarray) -> np.ndarray:
        if positions.size == 0:
            return positions.copy()
        return (self.rotation @ positions.T).T + self.translation

    def inverse(self) -> "RigidTransform":
        inverse_rotation = self.rotation.T
        inverse_translation = -inverse_rotation @ self.translation
        return RigidTransform(rotation=inverse_rotation, translation=inverse_translation)

    def compose(self, other: "RigidTransform") -> "RigidTransform":
        composed_rotation = self.rotation @ other.rotation
        composed_translation = self.rotation @ other.translation + self.translation
        return RigidTransform(rotation=composed_rotation, translation=composed_translation)


def format_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return "{:.9f}".format(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze split Cloud/Edge trajectories in one result directory."
    )
    parser.add_argument(
        "positional_result_dir",
        nargs="?",
        help="Optional result directory. --result_dir has priority if both are provided.",
    )
    parser.add_argument(
        "--result_dir",
        default=None,
        help="Directory containing exported trajectory files.",
    )
    parser.add_argument(
        "--gt_file",
        default=None,
        help="Ground-truth TUM trajectory file. GT metrics are skipped if omitted or missing.",
    )
    parser.add_argument(
        "--association_max_diff",
        type=float,
        default=ASSOCIATION_MAX_DIFF,
        help="Timestamp association threshold in seconds. Default: 0.01.",
    )
    return parser.parse_args()


def resolve_path(path: Optional[str], base_dir: str) -> Optional[str]:
    if path is None:
        return None
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(base_dir, expanded))


def read_tum_trajectory(path: str, name: str) -> Trajectory:
    if not os.path.exists(path):
        return Trajectory(
            name=name,
            path=path,
            timestamps=np.array([], dtype=float),
            positions=np.empty((0, 3), dtype=float),
            quaternions=np.empty((0, 4), dtype=float),
            duplicate_timestamp_count=0,
            status="missing",
        )

    timestamps: List[float] = []
    positions: List[List[float]] = []
    quaternions: List[List[float]] = []
    invalid_rows = 0

    with open(path, "r", encoding="utf-8", errors="replace") as input_file:
        for raw_line in input_file:
            line = raw_line.strip()
            if line == "":
                continue
            if line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 8:
                invalid_rows += 1
                continue

            try:
                timestamp = float(parts[0])
                tx = float(parts[1])
                ty = float(parts[2])
                tz = float(parts[3])
                qx = float(parts[4])
                qy = float(parts[5])
                qz = float(parts[6])
                qw = float(parts[7])
            except ValueError:
                invalid_rows += 1
                continue

            timestamps.append(timestamp)
            positions.append([tx, ty, tz])
            quaternions.append([qx, qy, qz, qw])

    if len(timestamps) == 0:
        status = "empty"
    elif invalid_rows > 0:
        status = "ok_with_invalid_rows"
    else:
        status = "ok"

    timestamp_array = np.array(timestamps, dtype=float)
    position_array = np.array(positions, dtype=float)
    quaternion_array = np.array(quaternions, dtype=float)
    duplicate_count = count_duplicate_timestamps(timestamp_array)

    return Trajectory(
        name=name,
        path=path,
        timestamps=timestamp_array,
        positions=position_array,
        quaternions=quaternion_array,
        duplicate_timestamp_count=duplicate_count,
        status=status,
    )


def count_duplicate_timestamps(timestamps: np.ndarray) -> int:
    if timestamps.size <= 1:
        return 0
    sorted_times = np.sort(timestamps)
    duplicate_count = 0
    previous_time = sorted_times[0]
    for i in range(1, sorted_times.size):
        if abs(sorted_times[i] - previous_time) < DUPLICATE_TIME_EPS:
            duplicate_count += 1
        else:
            previous_time = sorted_times[i]
    return duplicate_count


def trajectory_basic_metrics(traj: Trajectory) -> Dict[str, object]:
    metrics: Dict[str, object] = {
        "file_name": traj.name,
        "status": traj.status,
        "num_poses": int(traj.timestamps.size),
        "t_min": None,
        "t_max": None,
        "duration": None,
        "x_min": None,
        "x_max": None,
        "y_min": None,
        "y_max": None,
        "z_min": None,
        "z_max": None,
        "path_length": None,
        "mean_step": None,
        "max_step": None,
        "duplicate_timestamp_count": traj.duplicate_timestamp_count,
    }

    if traj.timestamps.size == 0:
        return metrics

    sorted_indices = np.argsort(traj.timestamps)
    sorted_times = traj.timestamps[sorted_indices]
    sorted_positions = traj.positions[sorted_indices]

    metrics["t_min"] = float(sorted_times[0])
    metrics["t_max"] = float(sorted_times[-1])
    metrics["duration"] = float(sorted_times[-1] - sorted_times[0])
    metrics["x_min"] = float(np.min(sorted_positions[:, 0]))
    metrics["x_max"] = float(np.max(sorted_positions[:, 0]))
    metrics["y_min"] = float(np.min(sorted_positions[:, 1]))
    metrics["y_max"] = float(np.max(sorted_positions[:, 1]))
    metrics["z_min"] = float(np.min(sorted_positions[:, 2]))
    metrics["z_max"] = float(np.max(sorted_positions[:, 2]))

    if sorted_positions.shape[0] <= 1:
        metrics["path_length"] = 0.0
        metrics["mean_step"] = 0.0
        metrics["max_step"] = 0.0
        return metrics

    steps = np.linalg.norm(np.diff(sorted_positions, axis=0), axis=1)
    metrics["path_length"] = float(np.sum(steps))
    metrics["mean_step"] = float(np.mean(steps))
    metrics["max_step"] = float(np.max(steps))
    return metrics


def associate_timestamps(
    times_a: np.ndarray,
    times_b: np.ndarray,
    max_diff: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if times_a.size == 0 or times_b.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    order_a = np.argsort(times_a)
    order_b = np.argsort(times_b)
    sorted_a = times_a[order_a]
    sorted_b = times_b[order_b]

    associated_a: List[int] = []
    associated_b: List[int] = []
    index_a = 0
    index_b = 0

    while index_a < sorted_a.size and index_b < sorted_b.size:
        time_a = sorted_a[index_a]
        time_b = sorted_b[index_b]
        diff = time_a - time_b
        abs_diff = abs(diff)

        if abs_diff <= max_diff:
            associated_a.append(int(order_a[index_a]))
            associated_b.append(int(order_b[index_b]))
            index_a += 1
            index_b += 1
        elif diff < 0.0:
            index_a += 1
        else:
            index_b += 1

    return np.array(associated_a, dtype=int), np.array(associated_b, dtype=int)


def estimate_rigid_transform(source_positions: np.ndarray, target_positions: np.ndarray) -> Optional[RigidTransform]:
    if source_positions.shape[0] < MIN_ALIGNMENT_ASSOCIATIONS:
        return None
    if target_positions.shape[0] < MIN_ALIGNMENT_ASSOCIATIONS:
        return None

    source_mean = np.mean(source_positions, axis=0)
    target_mean = np.mean(target_positions, axis=0)
    source_centered = source_positions - source_mean
    target_centered = target_positions - target_mean

    covariance = source_centered.T @ target_centered
    try:
        u_matrix, _, vt_matrix = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return None

    rotation = vt_matrix.T @ u_matrix.T
    if np.linalg.det(rotation) < 0.0:
        vt_matrix[-1, :] *= -1.0
        rotation = vt_matrix.T @ u_matrix.T

    translation = target_mean - rotation @ source_mean
    return RigidTransform(rotation=rotation, translation=translation)


def compute_ate_from_positions(estimated_positions: np.ndarray, gt_positions: np.ndarray) -> AteResult:
    if estimated_positions.shape[0] == 0:
        return AteResult(None, None, None, 0, "no_association")
    errors = np.linalg.norm(estimated_positions - gt_positions, axis=1)
    rmse = float(math.sqrt(np.mean(errors * errors)))
    mean = float(np.mean(errors))
    max_error = float(np.max(errors))
    return AteResult(rmse, mean, max_error, int(errors.size), "ok")


def estimate_alignment_transform_for_trajectory(
    traj: Trajectory,
    gt_traj: Optional[Trajectory],
    max_diff: float,
) -> Tuple[Optional[RigidTransform], int, str]:
    if traj.status == "missing":
        return None, 0, "missing"
    if gt_traj is None:
        return None, 0, "gt_missing"
    if gt_traj.status == "missing":
        return None, 0, "gt_missing"

    est_indices, gt_indices = associate_timestamps(traj.timestamps, gt_traj.timestamps, max_diff)
    if est_indices.size < MIN_ALIGNMENT_ASSOCIATIONS:
        return None, int(est_indices.size), "not_enough_associations"

    transform = estimate_rigid_transform(traj.positions[est_indices], gt_traj.positions[gt_indices])
    if transform is None:
        return None, int(est_indices.size), "alignment_failed"

    return transform, int(est_indices.size), "ok"


def compute_independent_se3_ate(traj: Trajectory, gt_traj: Optional[Trajectory], max_diff: float) -> AteResult:
    if traj.status == "missing":
        return AteResult(None, None, None, 0, "missing")
    if gt_traj is None:
        return AteResult(None, None, None, 0, "gt_missing")
    if gt_traj.status == "missing":
        return AteResult(None, None, None, 0, "gt_missing")

    est_indices, gt_indices = associate_timestamps(traj.timestamps, gt_traj.timestamps, max_diff)
    if est_indices.size < MIN_ALIGNMENT_ASSOCIATIONS:
        return AteResult(None, None, None, int(est_indices.size), "not_enough_associations")

    transform = estimate_rigid_transform(traj.positions[est_indices], gt_traj.positions[gt_indices])
    if transform is None:
        return AteResult(None, None, None, int(est_indices.size), "alignment_failed")

    aligned_positions = transform.apply(traj.positions[est_indices])
    return compute_ate_from_positions(aligned_positions, gt_traj.positions[gt_indices])


def build_no_cloud_alignment_transform(
    trajectories: Dict[str, Trajectory],
    gt_traj: Optional[Trajectory],
    max_diff: float,
) -> Tuple[Optional[RigidTransform], int, str]:
    if gt_traj is None:
        return None, 0, "gt_missing"
    if gt_traj.status == "missing":
        return None, 0, "gt_missing"

    no_cloud_traj = trajectories.get("whole_map_no_cloud.txt")
    if no_cloud_traj is None:
        return None, 0, "no_cloud_missing"
    if no_cloud_traj.status == "missing":
        return None, 0, "no_cloud_missing"

    no_cloud_indices, gt_indices = associate_timestamps(
        no_cloud_traj.timestamps,
        gt_traj.timestamps,
        max_diff,
    )
    if no_cloud_indices.size < MIN_ALIGNMENT_ASSOCIATIONS:
        return None, int(no_cloud_indices.size), "not_enough_associations"

    transform = estimate_rigid_transform(
        no_cloud_traj.positions[no_cloud_indices],
        gt_traj.positions[gt_indices],
    )
    if transform is None:
        return None, int(no_cloud_indices.size), "alignment_failed"

    return transform, int(no_cloud_indices.size), "ok"


def build_edge_fixed_alignment_transform(
    trajectories: Dict[str, Trajectory],
    gt_traj: Optional[Trajectory],
    max_diff: float,
) -> Tuple[Optional[RigidTransform], str, int, str]:
    for reference_file in ["map_1_edge_only.txt", "whole_map_no_cloud.txt"]:
        reference_traj = trajectories.get(reference_file)
        if reference_traj is None:
            continue
        transform, associated_count, status = estimate_alignment_transform_for_trajectory(
            reference_traj,
            gt_traj,
            max_diff,
        )
        if transform is not None:
            return transform, reference_file, associated_count, status

    return None, "none", 0, "edge_reference_unavailable"


def compute_ate_using_fixed_transform(
    traj: Trajectory,
    gt_traj: Optional[Trajectory],
    transform: Optional[RigidTransform],
    transform_status: str,
    max_diff: float,
) -> AteResult:
    if traj.status == "missing":
        return AteResult(None, None, None, 0, "missing")
    if gt_traj is None:
        return AteResult(None, None, None, 0, "gt_missing")
    if transform is None:
        return AteResult(None, None, None, 0, transform_status)

    est_indices, gt_indices = associate_timestamps(traj.timestamps, gt_traj.timestamps, max_diff)
    if est_indices.size == 0:
        return AteResult(None, None, None, 0, "no_association")

    aligned_positions = transform.apply(traj.positions[est_indices])
    return compute_ate_from_positions(aligned_positions, gt_traj.positions[gt_indices])


def compute_axis_metrics_from_errors(
    file_name: str,
    alignment_mode: str,
    segment_label: str,
    errors: np.ndarray,
    status: str,
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "file": file_name,
        "alignment_mode": alignment_mode,
        "segment_label": segment_label,
        "status": status,
        "num_associated": int(errors.shape[0]),
        "rmse_x": None,
        "rmse_y": None,
        "rmse_z": None,
        "mean_abs_x": None,
        "mean_abs_y": None,
        "mean_abs_z": None,
        "mean_signed_x": None,
        "mean_signed_y": None,
        "mean_signed_z": None,
        "max_abs_x": None,
        "max_abs_y": None,
        "max_abs_z": None,
        "rmse_xy": None,
        "rmse_xyz": None,
    }

    if errors.shape[0] == 0:
        return result

    squared_errors = errors * errors
    abs_errors = np.abs(errors)
    xy_errors = np.linalg.norm(errors[:, 0:2], axis=1)
    xyz_errors = np.linalg.norm(errors, axis=1)

    result["rmse_x"] = float(math.sqrt(np.mean(squared_errors[:, 0])))
    result["rmse_y"] = float(math.sqrt(np.mean(squared_errors[:, 1])))
    result["rmse_z"] = float(math.sqrt(np.mean(squared_errors[:, 2])))
    result["mean_abs_x"] = float(np.mean(abs_errors[:, 0]))
    result["mean_abs_y"] = float(np.mean(abs_errors[:, 1]))
    result["mean_abs_z"] = float(np.mean(abs_errors[:, 2]))
    result["mean_signed_x"] = float(np.mean(errors[:, 0]))
    result["mean_signed_y"] = float(np.mean(errors[:, 1]))
    result["mean_signed_z"] = float(np.mean(errors[:, 2]))
    result["max_abs_x"] = float(np.max(abs_errors[:, 0]))
    result["max_abs_y"] = float(np.max(abs_errors[:, 1]))
    result["max_abs_z"] = float(np.max(abs_errors[:, 2]))
    result["rmse_xy"] = float(math.sqrt(np.mean(xy_errors * xy_errors)))
    result["rmse_xyz"] = float(math.sqrt(np.mean(xyz_errors * xyz_errors)))
    return result


def get_cloud_interval(trajectories: Dict[str, Trajectory]) -> Tuple[Optional[float], Optional[float]]:
    cloud_traj = trajectories.get("map_1_cloud_only.txt")
    if cloud_traj is None:
        return None, None
    if cloud_traj.timestamps.size == 0:
        return None, None
    return float(np.min(cloud_traj.timestamps)), float(np.max(cloud_traj.timestamps))


def segment_label_for_timestamp(
    timestamp: float,
    cloud_t_min: Optional[float],
    cloud_t_max: Optional[float],
) -> str:
    if cloud_t_min is None or cloud_t_max is None:
        return "unknown"
    if timestamp < cloud_t_min:
        return "before_cloud"
    if timestamp <= cloud_t_max:
        return "inside_cloud"
    return "after_cloud"


def compute_aligned_axis_residuals(
    traj: Trajectory,
    gt_traj: Optional[Trajectory],
    transform: Optional[RigidTransform],
    transform_status: str,
    alignment_mode: str,
    max_diff: float,
    cloud_t_min: Optional[float],
    cloud_t_max: Optional[float],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    if traj.status == "missing":
        metric = compute_axis_metrics_from_errors(traj.name, alignment_mode, "all", np.empty((0, 3)), "missing")
        return metric, []
    if gt_traj is None or gt_traj.status == "missing":
        metric = compute_axis_metrics_from_errors(traj.name, alignment_mode, "all", np.empty((0, 3)), "gt_missing")
        return metric, []
    if transform is None:
        metric = compute_axis_metrics_from_errors(traj.name, alignment_mode, "all", np.empty((0, 3)), transform_status)
        return metric, []

    est_indices, gt_indices = associate_timestamps(traj.timestamps, gt_traj.timestamps, max_diff)
    if est_indices.size == 0:
        metric = compute_axis_metrics_from_errors(traj.name, alignment_mode, "all", np.empty((0, 3)), "no_association")
        return metric, []

    aligned_positions = transform.apply(traj.positions[est_indices])
    gt_positions = gt_traj.positions[gt_indices]
    errors = aligned_positions - gt_positions
    metric = compute_axis_metrics_from_errors(traj.name, alignment_mode, "all", errors, "ok")

    residual_rows: List[Dict[str, object]] = []
    for i in range(est_indices.size):
        timestamp = float(gt_traj.timestamps[gt_indices[i]])
        error = errors[i]
        segment_label = segment_label_for_timestamp(timestamp, cloud_t_min, cloud_t_max)
        is_cloud_interval = False
        if segment_label == "inside_cloud":
            is_cloud_interval = True

        error_xy = float(math.sqrt(error[0] * error[0] + error[1] * error[1]))
        error_xyz = float(math.sqrt(error[0] * error[0] + error[1] * error[1] + error[2] * error[2]))
        residual_rows.append(
            {
                "file": traj.name,
                "alignment_mode": alignment_mode,
                "timestamp": timestamp,
                "segment_label": segment_label,
                "error_x": float(error[0]),
                "error_y": float(error[1]),
                "error_z": float(error[2]),
                "error_xy": error_xy,
                "error_xyz": error_xyz,
                "is_cloud_interval": is_cloud_interval,
            }
        )

    return metric, residual_rows


def compute_segment_axis_metrics(
    file_name: str,
    alignment_mode: str,
    residual_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    segment_metrics: List[Dict[str, object]] = []
    for segment_label in ["before_cloud", "inside_cloud", "after_cloud"]:
        segment_errors: List[List[float]] = []
        for row in residual_rows:
            if row.get("file") != file_name:
                continue
            if row.get("alignment_mode") != alignment_mode:
                continue
            if row.get("segment_label") != segment_label:
                continue
            segment_errors.append(
                [
                    float(row["error_x"]),
                    float(row["error_y"]),
                    float(row["error_z"]),
                ]
            )

        if len(segment_errors) == 0:
            errors = np.empty((0, 3), dtype=float)
            status = "no_association"
        else:
            errors = np.array(segment_errors, dtype=float)
            status = "ok"
        segment_metrics.append(
            compute_axis_metrics_from_errors(file_name, alignment_mode, segment_label, errors, status)
        )
    return segment_metrics


def rotation_to_euler_zyx_deg(rotation: np.ndarray) -> Tuple[float, float, float]:
    sy = math.sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0])
    singular = sy < 1e-12
    if not singular:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cos_angle = (float(np.trace(rotation)) - 1.0) * 0.5
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def compare_alignment_transforms(
    edge_transform: Optional[RigidTransform],
    whole_transform: Optional[RigidTransform],
    edge_status: str,
    whole_status: str,
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "status": "ok",
        "translation_delta_x": None,
        "translation_delta_y": None,
        "translation_delta_z": None,
        "rotation_delta_angle_deg": None,
        "rotation_delta_roll_deg": None,
        "rotation_delta_pitch_deg": None,
        "rotation_delta_yaw_deg": None,
        "warning": "",
    }

    if edge_transform is None:
        result["status"] = edge_status
        return result
    if whole_transform is None:
        result["status"] = whole_status
        return result

    delta_transform = whole_transform.compose(edge_transform.inverse())
    roll_deg, pitch_deg, yaw_deg = rotation_to_euler_zyx_deg(delta_transform.rotation)

    result["translation_delta_x"] = float(delta_transform.translation[0])
    result["translation_delta_y"] = float(delta_transform.translation[1])
    result["translation_delta_z"] = float(delta_transform.translation[2])
    result["rotation_delta_angle_deg"] = rotation_angle_deg(delta_transform.rotation)
    result["rotation_delta_roll_deg"] = roll_deg
    result["rotation_delta_pitch_deg"] = pitch_deg
    result["rotation_delta_yaw_deg"] = yaw_deg

    if abs(float(result["translation_delta_z"])) > SIGNIFICANT_Z_ERROR_M:
        result["warning"] = "WARNING: whole_map independent alignment differs from edge-only alignment and may change visual Z behavior."
    if abs(roll_deg) > SIGNIFICANT_ANGLE_DEG or abs(pitch_deg) > SIGNIFICANT_ANGLE_DEG:
        result["warning"] = "WARNING: whole_map independent alignment differs from edge-only alignment and may change visual Z behavior."

    return result


def compute_time_balanced_metric_from_residuals(
    file_name: str,
    alignment_mode: str,
    residual_rows: Sequence[Dict[str, object]],
    bin_seconds: float,
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "file": file_name,
        "alignment_mode": alignment_mode,
        "bin_seconds": bin_seconds,
        "num_bins": 0,
        "time_balanced_rmse_x": None,
        "time_balanced_rmse_y": None,
        "time_balanced_rmse_z": None,
        "time_balanced_rmse_xyz": None,
        "time_balanced_mean_signed_z": None,
        "status": "ok",
    }

    selected_rows = []
    for row in residual_rows:
        if row.get("file") != file_name:
            continue
        if row.get("alignment_mode") != alignment_mode:
            continue
        selected_rows.append(row)

    if len(selected_rows) == 0:
        result["status"] = "no_residuals"
        return result

    t_min = min(float(row["timestamp"]) for row in selected_rows)
    bins: Dict[int, Dict[str, object]] = {}
    for row in selected_rows:
        timestamp = float(row["timestamp"])
        bin_index = int(math.floor((timestamp - t_min) / bin_seconds))
        bin_center = t_min + (float(bin_index) + 0.5) * bin_seconds
        distance_to_center = abs(timestamp - bin_center)
        current = bins.get(bin_index)
        if current is None or distance_to_center < float(current["distance_to_center"]):
            bins[bin_index] = {
                "distance_to_center": distance_to_center,
                "error_x": float(row["error_x"]),
                "error_y": float(row["error_y"]),
                "error_z": float(row["error_z"]),
            }

    errors = np.array(
        [[float(row["error_x"]), float(row["error_y"]), float(row["error_z"])] for row in bins.values()],
        dtype=float,
    )
    if errors.shape[0] == 0:
        result["status"] = "no_bins"
        return result

    xyz_errors = np.linalg.norm(errors, axis=1)
    result["num_bins"] = int(errors.shape[0])
    result["time_balanced_rmse_x"] = float(math.sqrt(np.mean(errors[:, 0] * errors[:, 0])))
    result["time_balanced_rmse_y"] = float(math.sqrt(np.mean(errors[:, 1] * errors[:, 1])))
    result["time_balanced_rmse_z"] = float(math.sqrt(np.mean(errors[:, 2] * errors[:, 2])))
    result["time_balanced_rmse_xyz"] = float(math.sqrt(np.mean(xyz_errors * xyz_errors)))
    result["time_balanced_mean_signed_z"] = float(np.mean(errors[:, 2]))
    return result


def compute_pair_overlap(
    pair: Tuple[str, str],
    trajectories: Dict[str, Trajectory],
    max_diff: float,
) -> Dict[str, object]:
    file_a, file_b = pair
    traj_a = trajectories.get(file_a)
    traj_b = trajectories.get(file_b)

    result: Dict[str, object] = {
        "pair_name": "{} vs {}".format(file_a, file_b),
        "file_a": file_a,
        "file_b": file_b,
        "status": "ok",
        "num_a": 0,
        "num_b": 0,
        "num_associated": 0,
        "association_ratio_a": None,
        "association_ratio_b": None,
        "mean_position_diff_on_common_timestamps": None,
        "max_position_diff_on_common_timestamps": None,
        "identity_status": "",
    }

    if traj_a is None or traj_a.status == "missing":
        result["status"] = "missing_a"
        return result
    if traj_b is None or traj_b.status == "missing":
        result["status"] = "missing_b"
        return result

    result["num_a"] = int(traj_a.timestamps.size)
    result["num_b"] = int(traj_b.timestamps.size)

    indices_a, indices_b = associate_timestamps(traj_a.timestamps, traj_b.timestamps, max_diff)
    result["num_associated"] = int(indices_a.size)

    if traj_a.timestamps.size > 0:
        result["association_ratio_a"] = float(indices_a.size / traj_a.timestamps.size)
    if traj_b.timestamps.size > 0:
        result["association_ratio_b"] = float(indices_b.size / traj_b.timestamps.size)

    if indices_a.size == 0:
        result["status"] = "no_association"
        return result

    position_diffs = np.linalg.norm(traj_a.positions[indices_a] - traj_b.positions[indices_b], axis=1)
    result["mean_position_diff_on_common_timestamps"] = float(np.mean(position_diffs))
    result["max_position_diff_on_common_timestamps"] = float(np.max(position_diffs))

    if float(np.max(position_diffs)) <= IDENTICAL_POSITION_EPS:
        result["identity_status"] = "IDENTICAL_ON_ASSOCIATED_TIMESTAMPS"

    return result


def nearest_index_by_time(times: np.ndarray, target_time: float) -> Optional[int]:
    if times.size == 0:
        return None
    return int(np.argmin(np.abs(times - target_time)))


def compute_anchor_continuity(
    trajectories: Dict[str, Trajectory],
) -> Dict[str, object]:
    cloud_file = "map_1_cloud_only.txt"
    edge_file = "map_1_edge_only.txt"
    cloud_traj = trajectories.get(cloud_file)
    edge_traj = trajectories.get(edge_file)

    result: Dict[str, object] = {
        "cloud_file": cloud_file,
        "edge_file": edge_file,
        "status": "ok",
        "cloud_start_time": None,
        "nearest_edge_to_cloud_start_time": None,
        "cloud_start_time_gap": None,
        "cloud_start_position_gap": None,
        "cloud_end_time": None,
        "nearest_edge_to_cloud_end_time": None,
        "cloud_end_time_gap": None,
        "cloud_end_position_gap": None,
        "attachment_warning": "",
    }

    if cloud_traj is None or cloud_traj.status == "missing":
        result["status"] = "cloud_missing"
        return result
    if edge_traj is None or edge_traj.status == "missing":
        result["status"] = "edge_missing"
        return result
    if cloud_traj.timestamps.size == 0:
        result["status"] = "cloud_empty"
        return result
    if edge_traj.timestamps.size == 0:
        result["status"] = "edge_empty"
        return result

    cloud_order = np.argsort(cloud_traj.timestamps)
    cloud_start_index = int(cloud_order[0])
    cloud_end_index = int(cloud_order[-1])
    cloud_start_time = float(cloud_traj.timestamps[cloud_start_index])
    cloud_end_time = float(cloud_traj.timestamps[cloud_end_index])

    start_edge_index = nearest_index_by_time(edge_traj.timestamps, cloud_start_time)
    end_edge_index = nearest_index_by_time(edge_traj.timestamps, cloud_end_time)
    if start_edge_index is None or end_edge_index is None:
        result["status"] = "edge_empty"
        return result

    start_time_gap = abs(float(edge_traj.timestamps[start_edge_index]) - cloud_start_time)
    end_time_gap = abs(float(edge_traj.timestamps[end_edge_index]) - cloud_end_time)
    start_position_gap = float(
        np.linalg.norm(cloud_traj.positions[cloud_start_index] - edge_traj.positions[start_edge_index])
    )
    end_position_gap = float(
        np.linalg.norm(cloud_traj.positions[cloud_end_index] - edge_traj.positions[end_edge_index])
    )

    result["cloud_start_time"] = cloud_start_time
    result["nearest_edge_to_cloud_start_time"] = float(edge_traj.timestamps[start_edge_index])
    result["cloud_start_time_gap"] = start_time_gap
    result["cloud_start_position_gap"] = start_position_gap
    result["cloud_end_time"] = cloud_end_time
    result["nearest_edge_to_cloud_end_time"] = float(edge_traj.timestamps[end_edge_index])
    result["cloud_end_time_gap"] = end_time_gap
    result["cloud_end_position_gap"] = end_position_gap

    edge_metrics = trajectory_basic_metrics(edge_traj)
    edge_mean_step = edge_metrics.get("mean_step")
    edge_max_step = edge_metrics.get("max_step")
    warning_threshold: Optional[float] = None
    if isinstance(edge_mean_step, float) and isinstance(edge_max_step, float):
        warning_threshold = max(edge_mean_step * 3.0, edge_max_step)

    if warning_threshold is not None:
        if start_position_gap > warning_threshold or end_position_gap > warning_threshold:
            result["attachment_warning"] = "WARNING: Cloud trajectory has large attachment gap to edge trajectory."

    return result


def write_metrics_csv(
    output_path: str,
    basic_metrics_by_file: Dict[str, Dict[str, object]],
    independent_ate_by_file: Dict[str, AteResult],
    no_cloud_ate_by_file: Dict[str, AteResult],
) -> None:
    fields = [
        "file_name",
        "status",
        "num_poses",
        "t_min",
        "t_max",
        "duration",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "z_min",
        "z_max",
        "path_length",
        "mean_step",
        "max_step",
        "duplicate_timestamp_count",
        "ate_independent_se3_rmse",
        "ate_independent_se3_mean",
        "ate_independent_se3_max",
        "num_gt_associated",
        "independent_se3_status",
        "ate_using_no_cloud_alignment_rmse",
        "ate_using_no_cloud_alignment_mean",
        "ate_using_no_cloud_alignment_max",
        "num_gt_associated_using_no_cloud_alignment",
        "no_cloud_alignment_status",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()

        for file_name in TUM_TRAJECTORY_FILES:
            basic = basic_metrics_by_file.get(file_name)
            if basic is None:
                basic = {
                    "file_name": file_name,
                    "status": "missing",
                    "num_poses": 0,
                    "duplicate_timestamp_count": "",
                }

            independent = independent_ate_by_file.get(
                file_name,
                AteResult(None, None, None, 0, "not_requested"),
            )
            no_cloud = no_cloud_ate_by_file.get(
                file_name,
                AteResult(None, None, None, 0, "not_requested"),
            )

            row: Dict[str, object] = {}
            for field in fields:
                row[field] = ""

            for field, value in basic.items():
                if field in row:
                    if isinstance(value, float):
                        row[field] = format_float(value)
                    else:
                        row[field] = value

            row["ate_independent_se3_rmse"] = format_float(independent.rmse)
            row["ate_independent_se3_mean"] = format_float(independent.mean)
            row["ate_independent_se3_max"] = format_float(independent.max_error)
            row["num_gt_associated"] = independent.associated_count
            row["independent_se3_status"] = independent.status
            row["ate_using_no_cloud_alignment_rmse"] = format_float(no_cloud.rmse)
            row["ate_using_no_cloud_alignment_mean"] = format_float(no_cloud.mean)
            row["ate_using_no_cloud_alignment_max"] = format_float(no_cloud.max_error)
            row["num_gt_associated_using_no_cloud_alignment"] = no_cloud.associated_count
            row["no_cloud_alignment_status"] = no_cloud.status

            writer.writerow(row)


def write_pair_overlap_csv(output_path: str, pair_results: Sequence[Dict[str, object]]) -> None:
    fields = [
        "pair_name",
        "file_a",
        "file_b",
        "status",
        "num_a",
        "num_b",
        "num_associated",
        "association_ratio_a",
        "association_ratio_b",
        "mean_position_diff_on_common_timestamps",
        "max_position_diff_on_common_timestamps",
        "identity_status",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for result in pair_results:
            row = {}
            for field in fields:
                value = result.get(field, "")
                if isinstance(value, float):
                    row[field] = format_float(value)
                else:
                    row[field] = value
            writer.writerow(row)


def write_anchor_continuity_csv(output_path: str, anchor_result: Dict[str, object]) -> None:
    fields = [
        "cloud_file",
        "edge_file",
        "status",
        "cloud_start_time",
        "nearest_edge_to_cloud_start_time",
        "cloud_start_time_gap",
        "cloud_start_position_gap",
        "cloud_end_time",
        "nearest_edge_to_cloud_end_time",
        "cloud_end_time_gap",
        "cloud_end_position_gap",
        "attachment_warning",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        row = {}
        for field in fields:
            value = anchor_result.get(field, "")
            if isinstance(value, float):
                row[field] = format_float(value)
            else:
                row[field] = value
        writer.writerow(row)


def write_axis_error_metrics_csv(output_path: str, axis_metrics: Sequence[Dict[str, object]]) -> None:
    fields = [
        "file",
        "alignment_mode",
        "segment_label",
        "status",
        "num_associated",
        "rmse_x",
        "rmse_y",
        "rmse_z",
        "mean_abs_x",
        "mean_abs_y",
        "mean_abs_z",
        "mean_signed_x",
        "mean_signed_y",
        "mean_signed_z",
        "max_abs_x",
        "max_abs_y",
        "max_abs_z",
        "rmse_xy",
        "rmse_xyz",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for metric in axis_metrics:
            row = {}
            for field in fields:
                value = metric.get(field, "")
                if isinstance(value, float):
                    row[field] = format_float(value)
                else:
                    row[field] = value
            writer.writerow(row)


def write_axis_residual_timeseries_csv(output_path: str, residual_rows: Sequence[Dict[str, object]]) -> None:
    fields = [
        "file",
        "alignment_mode",
        "timestamp",
        "segment_label",
        "error_x",
        "error_y",
        "error_z",
        "error_xy",
        "error_xyz",
        "is_cloud_interval",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for residual in residual_rows:
            row = {}
            for field in fields:
                value = residual.get(field, "")
                if isinstance(value, float):
                    row[field] = format_float(value)
                else:
                    row[field] = value
            writer.writerow(row)


def write_time_balanced_metrics_csv(output_path: str, metrics: Sequence[Dict[str, object]]) -> None:
    fields = [
        "file",
        "alignment_mode",
        "bin_seconds",
        "num_bins",
        "time_balanced_rmse_x",
        "time_balanced_rmse_y",
        "time_balanced_rmse_z",
        "time_balanced_rmse_xyz",
        "time_balanced_mean_signed_z",
        "status",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for metric in metrics:
            row = {}
            for field in fields:
                value = metric.get(field, "")
                if isinstance(value, float):
                    row[field] = format_float(value)
                else:
                    row[field] = value
            writer.writerow(row)


def write_alignment_transform_compare_txt(output_path: str, transform_compare: Dict[str, object]) -> None:
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write("[Alignment Transform Compare]\n")
        output_file.write("definition = T_delta = T_align_whole * inverse(T_align_edge)\n")
        for key in [
            "status",
            "translation_delta_x",
            "translation_delta_y",
            "translation_delta_z",
            "rotation_delta_angle_deg",
            "rotation_delta_roll_deg",
            "rotation_delta_pitch_deg",
            "rotation_delta_yaw_deg",
            "warning",
        ]:
            value = transform_compare.get(key, "")
            if isinstance(value, float):
                output_file.write("{} = {}\n".format(key, format_float(value)))
            else:
                output_file.write("{} = {}\n".format(key, value))


def find_axis_metric(
    axis_metrics: Sequence[Dict[str, object]],
    file_name: str,
    alignment_mode: str,
    segment_label: str = "all",
) -> Optional[Dict[str, object]]:
    for metric in axis_metrics:
        if metric.get("file") != file_name:
            continue
        if metric.get("alignment_mode") != alignment_mode:
            continue
        if metric.get("segment_label") != segment_label:
            continue
        return metric
    return None


def get_metric_float(metric: Optional[Dict[str, object]], field: str) -> Optional[float]:
    if metric is None:
        return None
    value = metric.get(field)
    if isinstance(value, float):
        return value
    return None


def find_time_balanced_metric(
    time_balanced_metrics: Sequence[Dict[str, object]],
    file_name: str,
    alignment_mode: str,
) -> Optional[Dict[str, object]]:
    for metric in time_balanced_metrics:
        if metric.get("file") != file_name:
            continue
        if metric.get("alignment_mode") != alignment_mode:
            continue
        return metric
    return None


def write_axis_error_summary_txt(
    output_path: str,
    axis_metrics: Sequence[Dict[str, object]],
    transform_compare: Dict[str, object],
    time_balanced_metrics: Sequence[Dict[str, object]],
    cloud_t_min: Optional[float],
    cloud_t_max: Optional[float],
    edge_fixed_reference_file: str,
    edge_fixed_status: str,
) -> None:
    overall_metrics = [
        metric
        for metric in axis_metrics
        if metric.get("segment_label") == "all" and metric.get("status") == "ok"
    ]

    max_rmse_z_metric: Optional[Dict[str, object]] = None
    max_mean_signed_z_metric: Optional[Dict[str, object]] = None
    for metric in overall_metrics:
        rmse_z = get_metric_float(metric, "rmse_z")
        mean_signed_z = get_metric_float(metric, "mean_signed_z")
        if rmse_z is not None:
            if max_rmse_z_metric is None:
                max_rmse_z_metric = metric
            else:
                old_rmse_z = get_metric_float(max_rmse_z_metric, "rmse_z")
                if old_rmse_z is None or rmse_z > old_rmse_z:
                    max_rmse_z_metric = metric
        if mean_signed_z is not None:
            if max_mean_signed_z_metric is None:
                max_mean_signed_z_metric = metric
            else:
                old_mean_signed_z = get_metric_float(max_mean_signed_z_metric, "mean_signed_z")
                if old_mean_signed_z is None or abs(mean_signed_z) > abs(old_mean_signed_z):
                    max_mean_signed_z_metric = metric

    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write("[Trajectory Axis Error Summary]\n")
        output_file.write("edge_fixed_reference_file = {}\n".format(edge_fixed_reference_file))
        output_file.write("edge_fixed_status = {}\n".format(edge_fixed_status))
        output_file.write("cloud_t_min = {}\n".format(format_float(cloud_t_min)))
        output_file.write("cloud_t_max = {}\n".format(format_float(cloud_t_max)))
        output_file.write("\n")

        output_file.write("[Largest Z Error]\n")
        if max_rmse_z_metric is not None:
            output_file.write(
                "max_rmse_z = {}, file = {}, alignment_mode = {}\n".format(
                    format_float(get_metric_float(max_rmse_z_metric, "rmse_z")),
                    max_rmse_z_metric.get("file", ""),
                    max_rmse_z_metric.get("alignment_mode", ""),
                )
            )
        else:
            output_file.write("max_rmse_z = missing\n")

        if max_mean_signed_z_metric is not None:
            output_file.write(
                "max_abs_mean_signed_z = {}, raw_mean_signed_z = {}, file = {}, alignment_mode = {}\n".format(
                    format_float(abs(get_metric_float(max_mean_signed_z_metric, "mean_signed_z"))),
                    format_float(get_metric_float(max_mean_signed_z_metric, "mean_signed_z")),
                    max_mean_signed_z_metric.get("file", ""),
                    max_mean_signed_z_metric.get("alignment_mode", ""),
                )
            )
        else:
            output_file.write("max_abs_mean_signed_z = missing\n")
        output_file.write("\n")

        output_file.write("[Alignment Transform Delta]\n")
        for key in [
            "translation_delta_x",
            "translation_delta_y",
            "translation_delta_z",
            "rotation_delta_angle_deg",
            "rotation_delta_roll_deg",
            "rotation_delta_pitch_deg",
            "rotation_delta_yaw_deg",
        ]:
            output_file.write("{} = {}\n".format(key, format_float(transform_compare.get(key))))
        if transform_compare.get("warning", "") != "":
            output_file.write("{}\n".format(transform_compare.get("warning")))
        output_file.write("\n")

        output_file.write("[Segment-wise Z Error]\n")
        for file_name in SEGMENT_Z_ERROR_FILES:
            for alignment_mode in ["independent_se3", "edge_fixed_se3"]:
                output_file.write("file = {}, alignment_mode = {}\n".format(file_name, alignment_mode))
                for segment_label in ["before_cloud", "inside_cloud", "after_cloud"]:
                    metric = find_axis_metric(axis_metrics, file_name, alignment_mode, segment_label)
                    output_file.write(
                        "  {}, num = {}, rmse_z = {}, mean_signed_z = {}, max_abs_z = {}, rmse_xyz = {}\n".format(
                            segment_label,
                            metric.get("num_associated", 0) if metric is not None else 0,
                            format_float(get_metric_float(metric, "rmse_z")),
                            format_float(get_metric_float(metric, "mean_signed_z")),
                            format_float(get_metric_float(metric, "max_abs_z")),
                            format_float(get_metric_float(metric, "rmse_xyz")),
                        )
                    )
        output_file.write("\n")

        output_file.write("[Time Balanced Metrics]\n")
        for metric in time_balanced_metrics:
            output_file.write(
                "{}, {}, bins = {}, rmse_z = {}, rmse_xyz = {}, mean_signed_z = {}, status = {}\n".format(
                    metric.get("file", ""),
                    metric.get("alignment_mode", ""),
                    metric.get("num_bins", 0),
                    format_float(metric.get("time_balanced_rmse_z")),
                    format_float(metric.get("time_balanced_rmse_xyz")),
                    format_float(metric.get("time_balanced_mean_signed_z")),
                    metric.get("status", ""),
                )
            )
        output_file.write("\n")

        output_file.write("[Cloud Correction Stage Comparison]\n")
        for file_name in CLOUD_CORRECTION_STAGE_FILES:
            output_file.write("file = {}\n".format(file_name))
            for alignment_mode in ["independent_se3", "edge_fixed_se3"]:
                metric = find_axis_metric(axis_metrics, file_name, alignment_mode, "all")
                time_balanced_metric = find_time_balanced_metric(time_balanced_metrics, file_name, alignment_mode)

                time_balanced_rmse_z = None
                if time_balanced_metric is not None:
                    time_balanced_value = time_balanced_metric.get("time_balanced_rmse_z")
                    if isinstance(time_balanced_value, float):
                        time_balanced_rmse_z = time_balanced_value

                output_file.write(
                    "  {}, num = {}, rmse_z = {}, mean_signed_z = {}, rmse_xyz = {}, time_balanced_rmse_z = {}, status = {}\n".format(
                        alignment_mode,
                        metric.get("num_associated", 0) if metric is not None else 0,
                        format_float(get_metric_float(metric, "rmse_z")),
                        format_float(get_metric_float(metric, "mean_signed_z")),
                        format_float(get_metric_float(metric, "rmse_xyz")),
                        format_float(time_balanced_rmse_z),
                        metric.get("status", "missing") if metric is not None else "missing",
                    )
                )
        head_edge_fixed = find_axis_metric(
            axis_metrics,
            "cloud_merge_001_head_aligned_only.txt",
            "edge_fixed_se3",
            "all",
        )
        sim3_edge_fixed = find_axis_metric(
            axis_metrics,
            "cloud_merge_001_sim3_aligned_only.txt",
            "edge_fixed_se3",
            "all",
        )
        after_edge_fixed = find_axis_metric(
            axis_metrics,
            "cloud_merge_001_after_two_anchor_correction.txt",
            "edge_fixed_se3",
            "all",
        )
        head_rmse_z = get_metric_float(head_edge_fixed, "rmse_z")
        sim3_rmse_z = get_metric_float(sim3_edge_fixed, "rmse_z")
        after_rmse_z = get_metric_float(after_edge_fixed, "rmse_z")
        if head_rmse_z is not None and sim3_rmse_z is not None:
            if sim3_rmse_z + SIGNIFICANT_Z_ERROR_M < head_rmse_z:
                output_file.write("Sim3-aligned-only has lower edge-fixed Z error than head-aligned-only.\n")
        if sim3_rmse_z is not None and after_rmse_z is not None:
            if after_rmse_z > sim3_rmse_z + SIGNIFICANT_Z_ERROR_M:
                output_file.write("Two-anchor correction has larger edge-fixed Z error than Sim3-aligned-only.\n")
        output_file.write("\n")

        output_file.write("[Diagnosis Hints]\n")
        whole_independent = find_axis_metric(axis_metrics, "whole_map.txt", "independent_se3", "all")
        edge_independent = find_axis_metric(axis_metrics, "map_1_edge_only.txt", "independent_se3", "all")

        whole_rmse_z = get_metric_float(whole_independent, "rmse_z")
        edge_rmse_z = get_metric_float(edge_independent, "rmse_z")
        whole_rmse_xyz = get_metric_float(whole_independent, "rmse_xyz")
        edge_rmse_xyz = get_metric_float(edge_independent, "rmse_xyz")

        if whole_rmse_z is not None and edge_rmse_z is not None:
            threshold = edge_rmse_z + max(SIGNIFICANT_Z_ERROR_M, edge_rmse_z * 0.2)
            if whole_rmse_z > threshold:
                output_file.write("whole_map has larger Z-axis error than edge-only trajectory.\n")

        if whole_rmse_xyz is not None and whole_rmse_z is not None and edge_rmse_xyz is not None:
            if whole_rmse_xyz <= edge_rmse_xyz * 1.1 and whole_rmse_z > max(SIGNIFICANT_Z_ERROR_M, whole_rmse_xyz * 0.5):
                output_file.write("Overall ATE RMSE hides Z-axis degradation.\n")

        if transform_compare.get("warning", "") != "":
            output_file.write("Independent SE3 alignment of whole_map changes the vertical alignment compared with edge-only alignment.\n")

        inside_metric = find_axis_metric(axis_metrics, "whole_map.txt", "edge_fixed_se3", "inside_cloud")
        before_metric = find_axis_metric(axis_metrics, "whole_map.txt", "edge_fixed_se3", "before_cloud")
        after_metric = find_axis_metric(axis_metrics, "whole_map.txt", "edge_fixed_se3", "after_cloud")
        inside_rmse_z = get_metric_float(inside_metric, "rmse_z")
        before_rmse_z = get_metric_float(before_metric, "rmse_z")
        after_rmse_z = get_metric_float(after_metric, "rmse_z")
        outside_values = []
        if before_rmse_z is not None:
            outside_values.append(before_rmse_z)
        if after_rmse_z is not None:
            outside_values.append(after_rmse_z)
        if inside_rmse_z is not None and len(outside_values) > 0:
            outside_max = max(outside_values)
            segment_delta_threshold = max(0.005, outside_max * 0.5)
            if inside_rmse_z > outside_max * 1.5 and inside_rmse_z - outside_max > segment_delta_threshold:
                output_file.write("Z-axis error is concentrated inside the CloudMap interval.\n")

        ordinary_whole = find_axis_metric(axis_metrics, "whole_map.txt", "independent_se3", "all")
        time_balanced_whole = find_time_balanced_metric(time_balanced_metrics, "whole_map.txt", "independent_se3")
        ordinary_rmse_xyz = get_metric_float(ordinary_whole, "rmse_xyz")
        time_balanced_rmse_xyz = None
        if time_balanced_whole is not None:
            value = time_balanced_whole.get("time_balanced_rmse_xyz")
            if isinstance(value, float):
                time_balanced_rmse_xyz = value
        if ordinary_rmse_xyz is not None and time_balanced_rmse_xyz is not None:
            if time_balanced_rmse_xyz > ordinary_rmse_xyz * 1.2 and time_balanced_rmse_xyz - ordinary_rmse_xyz > SIGNIFICANT_Z_ERROR_M:
                output_file.write("Dense CloudMap keyframes may bias ordinary RMSE evaluation.\n")


def extract_summary_sections(path: str, section_names: Iterable[str]) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    if not os.path.exists(path):
        return sections

    wanted = set(section_names)
    active_section: Optional[str] = None

    with open(path, "r", encoding="utf-8", errors="replace") as input_file:
        for raw_line in input_file:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                section_name = stripped[1:-1]
                if section_name in wanted:
                    active_section = section_name
                    sections[active_section] = [line]
                else:
                    active_section = None
                continue

            if active_section is not None:
                sections[active_section].append(line)

    return sections


def summarize_keyframe_source_debug(path: str) -> List[Dict[str, object]]:
    if not os.path.exists(path):
        return []

    grouped: Dict[Tuple[str, str], Dict[str, object]] = {}

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            map_id = row.get("map_id", "")
            is_cloud = row.get("is_cloud", "")
            timestamp_text = row.get("timestamp", "")
            key = (map_id, is_cloud)

            if key not in grouped:
                grouped[key] = {
                    "map_id": map_id,
                    "is_cloud": is_cloud,
                    "count": 0,
                    "t_min": None,
                    "t_max": None,
                }

            grouped[key]["count"] = int(grouped[key]["count"]) + 1
            try:
                timestamp = float(timestamp_text)
            except ValueError:
                continue

            current_min = grouped[key]["t_min"]
            current_max = grouped[key]["t_max"]
            if current_min is None or timestamp < float(current_min):
                grouped[key]["t_min"] = timestamp
            if current_max is None or timestamp > float(current_max):
                grouped[key]["t_max"] = timestamp

    return sorted(grouped.values(), key=lambda item: (str(item["map_id"]), str(item["is_cloud"])))


def is_pair_identical(pair_results: Sequence[Dict[str, object]], pair_name: str) -> bool:
    for result in pair_results:
        if result.get("pair_name") == pair_name:
            return result.get("identity_status") == "IDENTICAL_ON_ASSOCIATED_TIMESTAMPS"
    return False


def ate_warning_hidden_global_error(independent: AteResult, fixed: AteResult) -> bool:
    if independent.rmse is None:
        return False
    if fixed.rmse is None:
        return False
    minimum_delta = max(0.2, independent.rmse)
    if fixed.rmse > independent.rmse * 2.0 and fixed.rmse - independent.rmse > minimum_delta:
        return True
    return False


def trajectory_is_likely_normal(ate_result: AteResult, basic_metrics: Optional[Dict[str, object]]) -> bool:
    if ate_result.rmse is None:
        return False
    threshold = 0.5
    if basic_metrics is not None:
        mean_step = basic_metrics.get("mean_step")
        if isinstance(mean_step, float):
            threshold = max(threshold, mean_step * 3.0)
    if ate_result.rmse <= threshold:
        return True
    return False


def write_summary(
    output_path: str,
    result_dir: str,
    gt_file: Optional[str],
    association_max_diff: float,
    existing_files: Dict[str, bool],
    basic_metrics_by_file: Dict[str, Dict[str, object]],
    independent_ate_by_file: Dict[str, AteResult],
    no_cloud_ate_by_file: Dict[str, AteResult],
    pair_results: Sequence[Dict[str, object]],
    anchor_result: Dict[str, object],
    keyframe_debug_groups: Sequence[Dict[str, object]],
    extracted_sections: Dict[str, List[str]],
    no_cloud_transform_status: str,
    no_cloud_transform_count: int,
) -> None:
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write("[Trajectory Split Analysis Summary]\n")
        output_file.write("result_dir = {}\n".format(result_dir))
        output_file.write("gt_file = {}\n".format(gt_file if gt_file is not None else "missing"))
        output_file.write("association_max_diff = {:.9f}\n".format(association_max_diff))
        output_file.write("no_cloud_alignment_status = {}\n".format(no_cloud_transform_status))
        output_file.write("no_cloud_alignment_associations = {}\n".format(no_cloud_transform_count))
        output_file.write("\n")

        output_file.write("[Input File Status]\n")
        output_file.write("file,status\n")
        for file_name in EXPECTED_INPUT_FILES:
            if existing_files.get(file_name, False):
                output_file.write("{},present\n".format(file_name))
            else:
                output_file.write("{},missing\n".format(file_name))
        output_file.write("\n")

        output_file.write("[Trajectory Basic Metrics]\n")
        output_file.write("file,num_poses,t_min,t_max,duration,path_length,mean_step,max_step,duplicate_timestamp_count,status\n")
        for file_name in TUM_TRAJECTORY_FILES:
            metrics = basic_metrics_by_file.get(file_name)
            if metrics is None:
                output_file.write("{},0,,,,,,,,missing\n".format(file_name))
                continue
            output_file.write(
                "{},{},{},{},{},{},{},{},{},{}\n".format(
                    file_name,
                    metrics.get("num_poses", 0),
                    format_float(metrics.get("t_min")),
                    format_float(metrics.get("t_max")),
                    format_float(metrics.get("duration")),
                    format_float(metrics.get("path_length")),
                    format_float(metrics.get("mean_step")),
                    format_float(metrics.get("max_step")),
                    metrics.get("duplicate_timestamp_count", ""),
                    metrics.get("status", ""),
                )
            )
        output_file.write("\n")

        output_file.write("[Independent SE3 ATE]\n")
        output_file.write("file,rmse,mean,max,num_gt_associated,status\n")
        for file_name in INDEPENDENT_SE3_EVAL_FILES:
            ate = independent_ate_by_file.get(file_name)
            if ate is None:
                ate = AteResult(None, None, None, 0, "not_requested")
            output_file.write(
                "{},{},{},{},{},{}\n".format(
                    file_name,
                    format_float(ate.rmse),
                    format_float(ate.mean),
                    format_float(ate.max_error),
                    ate.associated_count,
                    ate.status,
                )
            )
        output_file.write("\n")

        output_file.write("[No Cloud Alignment ATE]\n")
        output_file.write("file,rmse,mean,max,num_gt_associated,status\n")
        for file_name in NO_CLOUD_ALIGNMENT_EVAL_FILES:
            ate = no_cloud_ate_by_file.get(file_name)
            if ate is None:
                ate = AteResult(None, None, None, 0, "not_requested")
            output_file.write(
                "{},{},{},{},{},{}\n".format(
                    file_name,
                    format_float(ate.rmse),
                    format_float(ate.mean),
                    format_float(ate.max_error),
                    ate.associated_count,
                    ate.status,
                )
            )
        output_file.write("\n")

        output_file.write("[Pair Overlap Summary]\n")
        output_file.write("pair,status,num_associated,mean_position_diff,max_position_diff,identity_status\n")
        for result in pair_results:
            output_file.write(
                "{},{},{},{},{},{}\n".format(
                    result.get("pair_name", ""),
                    result.get("status", ""),
                    result.get("num_associated", ""),
                    format_float(result.get("mean_position_diff_on_common_timestamps")),
                    format_float(result.get("max_position_diff_on_common_timestamps")),
                    result.get("identity_status", ""),
                )
            )
        output_file.write("\n")

        output_file.write("[Cloud Edge Anchor Continuity]\n")
        for key in [
            "status",
            "cloud_file",
            "edge_file",
            "cloud_start_time",
            "nearest_edge_to_cloud_start_time",
            "cloud_start_time_gap",
            "cloud_start_position_gap",
            "cloud_end_time",
            "nearest_edge_to_cloud_end_time",
            "cloud_end_time_gap",
            "cloud_end_position_gap",
            "attachment_warning",
        ]:
            value = anchor_result.get(key, "")
            if isinstance(value, float):
                output_file.write("{} = {}\n".format(key, format_float(value)))
            else:
                output_file.write("{} = {}\n".format(key, value))
        output_file.write("\n")

        output_file.write("[KeyFrame Source Debug CSV Summary]\n")
        if len(keyframe_debug_groups) == 0:
            output_file.write("missing\n")
        else:
            output_file.write("map_id,is_cloud,count,t_min,t_max\n")
            for group in keyframe_debug_groups:
                output_file.write(
                    "{},{},{},{},{}\n".format(
                        group.get("map_id", ""),
                        group.get("is_cloud", ""),
                        group.get("count", ""),
                        format_float(group.get("t_min")),
                        format_float(group.get("t_max")),
                    )
                )
        output_file.write("\n")

        output_file.write("[Copied From keyframe_source_summary.txt]\n")
        if len(extracted_sections) == 0:
            output_file.write("missing\n")
        else:
            for section_name in ["Atlas Summary", "Per Map Summary"]:
                lines = extracted_sections.get(section_name)
                if lines is None:
                    continue
                for line in lines:
                    output_file.write("{}\n".format(line))
                output_file.write("\n")

        output_file.write("[Diagnosis Hints]\n")
        if is_pair_identical(pair_results, "whole_map_no_cloud.txt vs map_1_edge_only.txt"):
            output_file.write("whole_map_no_cloud is equivalent to map_1_edge_only on associated timestamps.\n")

        if is_pair_identical(pair_results, "whole_map_cloud_only.txt vs map_1_cloud_only.txt"):
            output_file.write("whole_map_cloud_only is equivalent to map_1_cloud_only on associated timestamps.\n")

        map1_edge_ate = independent_ate_by_file.get("map_1_edge_only.txt")
        map1_edge_metrics = basic_metrics_by_file.get("map_1_edge_only.txt")
        if map1_edge_ate is not None:
            if trajectory_is_likely_normal(map1_edge_ate, map1_edge_metrics):
                output_file.write("Map 1 edge trajectory is likely normal.\n")

        map1_cloud_independent = independent_ate_by_file.get("map_1_cloud_only.txt")
        map1_cloud_fixed = no_cloud_ate_by_file.get("map_1_cloud_only.txt")
        if map1_cloud_independent is not None and map1_cloud_fixed is not None:
            if ate_warning_hidden_global_error(map1_cloud_independent, map1_cloud_fixed):
                output_file.write("WARNING: independent alignment hides global attachment error.\n")
                output_file.write("Cloud trajectory is inconsistent with the edge-defined global frame.\n")

        for file_name in NO_CLOUD_ALIGNMENT_EVAL_FILES:
            independent = independent_ate_by_file.get(file_name)
            fixed = no_cloud_ate_by_file.get(file_name)
            if independent is None or fixed is None:
                continue
            if ate_warning_hidden_global_error(independent, fixed):
                output_file.write(
                    "WARNING: independent alignment hides global attachment error for {}.\n".format(file_name)
                )

        attachment_warning = anchor_result.get("attachment_warning", "")
        if attachment_warning != "":
            output_file.write("{}\n".format(attachment_warning))
            output_file.write("Cloud trajectory may be attached to a wrong location or has poor boundary anchoring.\n")


def main() -> int:
    args = parse_args()
    cwd = os.getcwd()
    result_dir_arg = args.result_dir
    if result_dir_arg is None:
        result_dir_arg = args.positional_result_dir

    if result_dir_arg is None:
        raise SystemExit("ERROR: please provide --result_dir or a positional result directory.")

    result_dir = resolve_path(result_dir_arg, cwd)
    if result_dir is None:
        raise SystemExit("ERROR: invalid result directory.")
    if not os.path.isdir(result_dir):
        raise SystemExit("ERROR: result_dir does not exist or is not a directory: {}".format(result_dir))

    gt_file = resolve_path(args.gt_file, cwd)
    if gt_file is not None and not os.path.exists(gt_file):
        gt_file = None

    trajectories: Dict[str, Trajectory] = {}
    basic_metrics_by_file: Dict[str, Dict[str, object]] = {}
    existing_files: Dict[str, bool] = {}

    for file_name in EXPECTED_INPUT_FILES:
        file_path = os.path.join(result_dir, file_name)
        existing_files[file_name] = os.path.exists(file_path)

    for file_name in TUM_TRAJECTORY_FILES:
        file_path = os.path.join(result_dir, file_name)
        trajectory = read_tum_trajectory(file_path, file_name)
        trajectories[file_name] = trajectory
        basic_metrics_by_file[file_name] = trajectory_basic_metrics(trajectory)

    gt_traj: Optional[Trajectory] = None
    if gt_file is not None:
        gt_traj = read_tum_trajectory(gt_file, os.path.basename(gt_file))

    independent_ate_by_file: Dict[str, AteResult] = {}
    for file_name in INDEPENDENT_SE3_EVAL_FILES:
        trajectory = trajectories.get(file_name)
        if trajectory is None:
            independent_ate_by_file[file_name] = AteResult(None, None, None, 0, "missing")
            continue
        independent_ate_by_file[file_name] = compute_independent_se3_ate(
            trajectory,
            gt_traj,
            args.association_max_diff,
        )

    no_cloud_transform, no_cloud_transform_count, no_cloud_transform_status = build_no_cloud_alignment_transform(
        trajectories,
        gt_traj,
        args.association_max_diff,
    )

    no_cloud_ate_by_file: Dict[str, AteResult] = {}
    for file_name in NO_CLOUD_ALIGNMENT_EVAL_FILES:
        trajectory = trajectories.get(file_name)
        if trajectory is None:
            no_cloud_ate_by_file[file_name] = AteResult(None, None, None, 0, "missing")
            continue
        no_cloud_ate_by_file[file_name] = compute_ate_using_fixed_transform(
            trajectory,
            gt_traj,
            no_cloud_transform,
            no_cloud_transform_status,
            args.association_max_diff,
        )

    independent_transforms: Dict[str, Optional[RigidTransform]] = {}
    independent_transform_status: Dict[str, str] = {}
    independent_transform_counts: Dict[str, int] = {}
    axis_transform_files = sorted(set(AXIS_ERROR_FILES + TIME_BALANCED_FILES + ["map_1_edge_only.txt", "whole_map.txt"]))
    for file_name in axis_transform_files:
        trajectory = trajectories.get(file_name)
        if trajectory is None:
            independent_transforms[file_name] = None
            independent_transform_status[file_name] = "missing"
            independent_transform_counts[file_name] = 0
            continue

        transform, associated_count, status = estimate_alignment_transform_for_trajectory(
            trajectory,
            gt_traj,
            args.association_max_diff,
        )
        independent_transforms[file_name] = transform
        independent_transform_status[file_name] = status
        independent_transform_counts[file_name] = associated_count

    edge_fixed_transform, edge_fixed_reference_file, edge_fixed_count, edge_fixed_status = build_edge_fixed_alignment_transform(
        trajectories,
        gt_traj,
        args.association_max_diff,
    )

    cloud_t_min, cloud_t_max = get_cloud_interval(trajectories)

    axis_metrics: List[Dict[str, object]] = []
    residual_rows: List[Dict[str, object]] = []
    for file_name in AXIS_ERROR_FILES:
        trajectory = trajectories.get(file_name)
        if trajectory is None:
            continue

        metric, rows = compute_aligned_axis_residuals(
            trajectory,
            gt_traj,
            independent_transforms.get(file_name),
            independent_transform_status.get(file_name, "missing"),
            "independent_se3",
            args.association_max_diff,
            cloud_t_min,
            cloud_t_max,
        )
        axis_metrics.append(metric)
        residual_rows.extend(rows)

    for file_name in EDGE_FIXED_SE3_FILES:
        trajectory = trajectories.get(file_name)
        if trajectory is None:
            continue

        metric, rows = compute_aligned_axis_residuals(
            trajectory,
            gt_traj,
            edge_fixed_transform,
            edge_fixed_status,
            "edge_fixed_se3",
            args.association_max_diff,
            cloud_t_min,
            cloud_t_max,
        )
        axis_metrics.append(metric)
        residual_rows.extend(rows)

    for file_name in SEGMENT_Z_ERROR_FILES:
        axis_metrics.extend(compute_segment_axis_metrics(file_name, "independent_se3", residual_rows))
        axis_metrics.extend(compute_segment_axis_metrics(file_name, "edge_fixed_se3", residual_rows))

    edge_transform = independent_transforms.get("map_1_edge_only.txt")
    whole_transform = independent_transforms.get("whole_map.txt")
    transform_compare = compare_alignment_transforms(
        edge_transform,
        whole_transform,
        independent_transform_status.get("map_1_edge_only.txt", "missing"),
        independent_transform_status.get("whole_map.txt", "missing"),
    )

    time_balanced_metrics: List[Dict[str, object]] = []
    for file_name in TIME_BALANCED_FILES:
        time_balanced_metrics.append(
            compute_time_balanced_metric_from_residuals(
                file_name,
                "independent_se3",
                residual_rows,
                TIME_BALANCED_BIN_SECONDS,
            )
        )

        if file_name in EDGE_FIXED_SE3_FILES:
            time_balanced_metrics.append(
                compute_time_balanced_metric_from_residuals(
                    file_name,
                    "edge_fixed_se3",
                    residual_rows,
                    TIME_BALANCED_BIN_SECONDS,
                )
            )

    pair_results = [
        compute_pair_overlap(pair, trajectories, args.association_max_diff)
        for pair in PAIR_OVERLAP_FILES
    ]

    anchor_result = compute_anchor_continuity(trajectories)
    keyframe_debug_groups = summarize_keyframe_source_debug(
        os.path.join(result_dir, "keyframe_source_debug.csv")
    )
    extracted_sections = extract_summary_sections(
        os.path.join(result_dir, "keyframe_source_summary.txt"),
        ["Atlas Summary", "Per Map Summary"],
    )

    metrics_path = os.path.join(result_dir, "trajectory_split_metrics.csv")
    pair_overlap_path = os.path.join(result_dir, "trajectory_pair_overlap.csv")
    anchor_path = os.path.join(result_dir, "trajectory_anchor_continuity.csv")
    summary_path = os.path.join(result_dir, "trajectory_split_analysis_summary.txt")
    axis_summary_path = os.path.join(result_dir, "trajectory_axis_error_summary.txt")
    axis_metrics_path = os.path.join(result_dir, "trajectory_axis_error_metrics.csv")
    axis_residual_path = os.path.join(result_dir, "trajectory_axis_residual_timeseries.csv")
    transform_compare_path = os.path.join(result_dir, "trajectory_alignment_transform_compare.txt")
    time_balanced_path = os.path.join(result_dir, "trajectory_time_balanced_metrics.csv")

    write_metrics_csv(
        metrics_path,
        basic_metrics_by_file,
        independent_ate_by_file,
        no_cloud_ate_by_file,
    )
    write_pair_overlap_csv(pair_overlap_path, pair_results)
    write_anchor_continuity_csv(anchor_path, anchor_result)
    write_summary(
        summary_path,
        result_dir,
        gt_file,
        args.association_max_diff,
        existing_files,
        basic_metrics_by_file,
        independent_ate_by_file,
        no_cloud_ate_by_file,
        pair_results,
        anchor_result,
        keyframe_debug_groups,
        extracted_sections,
        no_cloud_transform_status,
        no_cloud_transform_count,
    )
    write_axis_error_metrics_csv(axis_metrics_path, axis_metrics)
    write_axis_residual_timeseries_csv(axis_residual_path, residual_rows)
    write_alignment_transform_compare_txt(transform_compare_path, transform_compare)
    write_time_balanced_metrics_csv(time_balanced_path, time_balanced_metrics)
    write_axis_error_summary_txt(
        axis_summary_path,
        axis_metrics,
        transform_compare,
        time_balanced_metrics,
        cloud_t_min,
        cloud_t_max,
        edge_fixed_reference_file,
        edge_fixed_status,
    )

    print("[Trajectory Split Analysis] Saved summary: {}".format(summary_path))
    print("[Trajectory Split Analysis] Saved metrics: {}".format(metrics_path))
    print("[Trajectory Split Analysis] Saved pair overlap: {}".format(pair_overlap_path))
    print("[Trajectory Split Analysis] Saved anchor continuity: {}".format(anchor_path))
    print("[Trajectory Axis Analysis] Saved summary: {}".format(axis_summary_path))
    print("[Trajectory Axis Analysis] Saved metrics: {}".format(axis_metrics_path))
    print("[Trajectory Axis Analysis] Saved residual timeseries: {}".format(axis_residual_path))
    print("[Trajectory Axis Analysis] Saved transform compare: {}".format(transform_compare_path))
    print("[Trajectory Axis Analysis] Saved time-balanced metrics: {}".format(time_balanced_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
