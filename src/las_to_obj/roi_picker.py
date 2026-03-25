from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .io import load_las_preview_points
from .preprocess import voxel_downsample


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("open3d is required for interactive ROI picking.") from exc
    return o3d


def pick_roi_interactively(
    input_las: Path,
    chunk_size: int = 1_000_000,
    max_preview_points: int = 400_000,
    preview_voxel_size: float = 0.08,
    padding: float = 0.0,
) -> dict[str, object]:
    preview_points = load_las_preview_points(
        las_path=input_las,
        chunk_size=chunk_size,
        max_points=max_preview_points,
    )
    if len(preview_points) == 0:
        raise ValueError("No preview points were loaded from the LAS file.")

    if preview_voxel_size > 0:
        preview_points = voxel_downsample(preview_points, preview_voxel_size)

    o3d = _require_open3d()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(preview_points)

    print("ROI picker instructions:")
    print("1. Hold Shift and left-click to pick ROI corner points.")
    print("2. Pick at least two points around the target volume.")
    print("3. Press Q to close the window when finished.")

    visualizer = o3d.visualization.VisualizerWithEditing()
    visualizer.create_window(window_name="LAS ROI Picker")
    visualizer.add_geometry(cloud)
    visualizer.run()
    visualizer.destroy_window()

    picked_indices = visualizer.get_picked_points()
    if len(picked_indices) < 2:
        raise ValueError("At least two picked points are required to build an ROI box.")

    picked_points = preview_points[np.asarray(picked_indices, dtype=int)]
    minimum = picked_points.min(axis=0) - float(padding)
    maximum = picked_points.max(axis=0) + float(padding)

    return {
        "enabled": True,
        "min": [float(value) for value in minimum],
        "max": [float(value) for value in maximum],
        "picked_point_count": int(len(picked_points)),
        "preview_point_count": int(len(preview_points)),
    }


def write_roi_json(output_path: Path, roi_payload: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(roi_payload, ensure_ascii=False, indent=2), encoding="utf-8")
