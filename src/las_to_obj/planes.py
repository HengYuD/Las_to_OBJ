from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PlaneDetectionConfig
from .geometry import classify_plane, flatten_points_to_plane, make_plane_region, normalize
from .models import PlaneRegion


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("open3d is required for plane extraction.") from exc
    return o3d


def _to_open3d_cloud(points: np.ndarray):
    o3d = _require_open3d()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return cloud


@dataclass(slots=True)
class PlaneExtractionResult:
    regions: list[PlaneRegion]
    remaining_points: np.ndarray
    rejected_planes: int


def extract_plane_regions(
    points: np.ndarray,
    plane_config: PlaneDetectionConfig,
) -> PlaneExtractionResult:
    if len(points) == 0:
        return PlaneExtractionResult(regions=[], remaining_points=points, rejected_planes=0)

    remaining = points.copy()
    z_median = float(np.median(remaining[:, 2]))
    regions: list[PlaneRegion] = []
    rejected = 0

    for plane_index in range(plane_config.max_planes):
        if len(remaining) < plane_config.min_plane_points:
            break

        cloud = _to_open3d_cloud(remaining)
        plane_model, inliers = cloud.segment_plane(
            distance_threshold=plane_config.distance_threshold,
            ransac_n=plane_config.ransac_n,
            num_iterations=plane_config.num_iterations,
        )

        inlier_indices = np.asarray(inliers, dtype=int)
        if len(inlier_indices) < plane_config.min_plane_points:
            break

        plane_points = remaining[inlier_indices]
        flattened_points = flatten_points_to_plane(
            plane_points, tuple(float(v) for v in plane_model)
        )

        normal = normalize(np.asarray(plane_model[:3], dtype=float))
        label = classify_plane(
            normal=normal,
            centroid_z=float(flattened_points[:, 2].mean()),
            z_median=z_median,
            horizontal_angle_threshold_degrees=plane_config.horizontal_angle_threshold_degrees,
        )

        regions.append(
            make_plane_region(
                label=label,
                plane_model=tuple(float(v) for v in plane_model),
                flattened_points=flattened_points,
                source_name=f"{label}_{plane_index:03d}",
            )
        )

        keep_mask = np.ones(len(remaining), dtype=bool)
        keep_mask[inlier_indices] = False
        remaining = remaining[keep_mask]

    return PlaneExtractionResult(
        regions=regions,
        remaining_points=remaining,
        rejected_planes=rejected,
    )
