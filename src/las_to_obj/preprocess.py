from __future__ import annotations

from pathlib import Path

import numpy as np


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("open3d is required for point cloud preprocessing.") from exc
    return o3d


def _to_open3d_cloud(points: np.ndarray):
    o3d = _require_open3d()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return cloud


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(points) == 0 or voxel_size <= 0:
        return points
    cloud = _to_open3d_cloud(points)
    downsampled = cloud.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(downsampled.points)


def remove_statistical_outliers(
    points: np.ndarray, nb_neighbors: int, std_ratio: float
) -> np.ndarray:
    if len(points) == 0:
        return points
    cloud = _to_open3d_cloud(points)
    filtered, _ = cloud.remove_statistical_outlier(
        nb_neighbors=max(1, nb_neighbors), std_ratio=std_ratio
    )
    return np.asarray(filtered.points)


def write_debug_cloud(path: Path, points: np.ndarray) -> None:
    if len(points) == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d = _require_open3d()
    cloud = _to_open3d_cloud(points)
    o3d.io.write_point_cloud(str(path), cloud)

