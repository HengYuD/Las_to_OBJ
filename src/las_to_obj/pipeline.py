from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .io import load_las_points, write_obj, write_report
from .planes import extract_plane_regions
from .postprocess import build_patches_from_regions
from .preprocess import remove_statistical_outliers, voxel_downsample, write_debug_cloud


def _save_debug_cloud(debug_dir: Path | None, name: str, points: np.ndarray) -> None:
    if debug_dir is None:
        return
    write_debug_cloud(debug_dir / name, points)


def run_pipeline(config: PipelineConfig) -> dict[str, object]:
    loaded = load_las_points(
        las_path=config.input_las,
        roi=config.roi,
        chunk_size=config.las_chunk_size,
    )

    if len(loaded.points) == 0:
        raise ValueError("No points remain after LAS loading and ROI filtering.")

    working = loaded.points.copy()
    origin_offset = np.zeros(3, dtype=float)
    if config.translate_to_local_origin:
        origin_offset = working.min(axis=0)
        working = working - origin_offset

    if config.debug.export_intermediate_clouds:
        _save_debug_cloud(config.debug_cloud_dir, "loaded_roi.ply", working)

    after_downsample = voxel_downsample(working, voxel_size=config.preprocess.voxel_size)
    if config.debug.export_intermediate_clouds:
        _save_debug_cloud(config.debug_cloud_dir, "downsampled.ply", after_downsample)

    after_denoise = remove_statistical_outliers(
        after_downsample,
        nb_neighbors=config.preprocess.statistical_outlier_neighbors,
        std_ratio=config.preprocess.statistical_outlier_std_ratio,
    )
    if config.debug.export_intermediate_clouds:
        _save_debug_cloud(config.debug_cloud_dir, "preprocessed.ply", after_denoise)

    extraction = extract_plane_regions(
        points=after_denoise,
        plane_config=config.plane_detection,
    )
    patches = build_patches_from_regions(
        regions=extraction.regions,
        mesh_config=config.mesh,
        trim_percent=config.plane_detection.extent_trim_percent,
    )

    if not patches:
        raise ValueError("No structural planes were extracted. Review ROI and thresholds.")

    write_obj(config.output_obj, patches)

    if config.debug.export_intermediate_clouds:
        _save_debug_cloud(
            config.debug_cloud_dir,
            "remaining_after_planes.ply",
            extraction.remaining_points,
        )

    label_counts: dict[str, int] = {}
    opening_counts: dict[str, int] = {"door": 0, "window": 0}
    for patch in patches:
        label_counts[patch.label] = label_counts.get(patch.label, 0) + 1
        for opening in patch.openings:
            opening_counts[opening.kind] = opening_counts.get(opening.kind, 0) + 1

    report = {
        "input_las": str(config.input_las),
        "output_obj": str(config.output_obj),
        "source_total_points": loaded.source_total_points,
        "roi_selected_points": loaded.selected_points,
        "downsampled_points": int(len(after_downsample)),
        "preprocessed_points": int(len(after_denoise)),
        "remaining_points": int(len(extraction.remaining_points)),
        "region_count": len(extraction.regions),
        "patch_count": len(patches),
        "rejected_planes": extraction.rejected_planes,
        "label_counts": label_counts,
        "opening_counts": opening_counts,
        "origin_offset": origin_offset.tolist(),
        "patches": [patch.to_report_dict() for patch in patches],
    }
    write_report(config.output_report, report)
    return report
