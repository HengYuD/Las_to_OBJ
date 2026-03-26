from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _as_triplet(value: Any, field_name: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} must be a list of 3 numbers.")
    return (float(value[0]), float(value[1]), float(value[2]))


def _resolve_path(base_dir: Path, raw: str | None) -> Path | None:
    if raw is None:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


@dataclass(slots=True)
class RoiConfig:
    enabled: bool = False
    minimum: tuple[float, float, float] | None = None
    maximum: tuple[float, float, float] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RoiConfig":
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", False)),
            minimum=_as_triplet(data.get("min"), "roi.min"),
            maximum=_as_triplet(data.get("max"), "roi.max"),
        )


@dataclass(slots=True)
class PreprocessConfig:
    voxel_size: float = 0.05
    statistical_outlier_neighbors: int = 24
    statistical_outlier_std_ratio: float = 2.5

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PreprocessConfig":
        data = data or {}
        return cls(
            voxel_size=float(data.get("voxel_size", 0.05)),
            statistical_outlier_neighbors=int(data.get("statistical_outlier_neighbors", 24)),
            statistical_outlier_std_ratio=float(data.get("statistical_outlier_std_ratio", 2.5)),
        )


@dataclass(slots=True)
class PlaneDetectionConfig:
    distance_threshold: float = 0.03
    ransac_n: int = 3
    num_iterations: int = 2000
    min_plane_points: int = 8000
    min_wall_points: int = 3000
    max_planes: int = 32
    max_horizontal_planes: int = 2
    horizontal_angle_threshold_degrees: float = 15.0
    extent_trim_percent: float = 2.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PlaneDetectionConfig":
        data = data or {}
        return cls(
            distance_threshold=float(data.get("distance_threshold", 0.03)),
            ransac_n=int(data.get("ransac_n", 3)),
            num_iterations=int(data.get("num_iterations", 2000)),
            min_plane_points=int(data.get("min_plane_points", 8000)),
            min_wall_points=int(data.get("min_wall_points", data.get("min_plane_points", 8000))),
            max_planes=int(data.get("max_planes", 32)),
            max_horizontal_planes=int(data.get("max_horizontal_planes", 2)),
            horizontal_angle_threshold_degrees=float(
                data.get("horizontal_angle_threshold_degrees", 15.0)
            ),
            extent_trim_percent=float(data.get("extent_trim_percent", 2.0)),
        )


@dataclass(slots=True)
class MeshConfig:
    min_plane_extent: float = 0.4
    merge_normal_angle_degrees: float = 5.0
    merge_plane_offset_threshold: float = 0.08
    merge_gap_threshold: float = 1.5
    merge_min_overlap: float = 0.25
    enable_opening_detection: bool = True
    opening_grid_resolution: float = 0.1
    opening_neighbor_fill_radius: int = 1
    opening_min_width: float = 0.5
    opening_min_height: float = 0.5
    opening_min_area: float = 0.35
    door_max_width: float = 2.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MeshConfig":
        data = data or {}
        return cls(
            min_plane_extent=float(data.get("min_plane_extent", 0.4)),
            merge_normal_angle_degrees=float(data.get("merge_normal_angle_degrees", 5.0)),
            merge_plane_offset_threshold=float(data.get("merge_plane_offset_threshold", 0.08)),
            merge_gap_threshold=float(data.get("merge_gap_threshold", 1.5)),
            merge_min_overlap=float(data.get("merge_min_overlap", 0.25)),
            enable_opening_detection=bool(data.get("enable_opening_detection", True)),
            opening_grid_resolution=float(data.get("opening_grid_resolution", 0.1)),
            opening_neighbor_fill_radius=int(data.get("opening_neighbor_fill_radius", 1)),
            opening_min_width=float(data.get("opening_min_width", 0.5)),
            opening_min_height=float(data.get("opening_min_height", 0.5)),
            opening_min_area=float(data.get("opening_min_area", 0.35)),
            door_max_width=float(data.get("door_max_width", 2.0)),
        )


@dataclass(slots=True)
class DebugConfig:
    export_intermediate_clouds: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "DebugConfig":
        data = data or {}
        return cls(export_intermediate_clouds=bool(data.get("export_intermediate_clouds", True)))


@dataclass(slots=True)
class PipelineConfig:
    input_las: Path
    output_obj: Path
    output_report: Path
    debug_cloud_dir: Path | None = None
    las_chunk_size: int = 1_000_000
    translate_to_local_origin: bool = True
    roi: RoiConfig = field(default_factory=RoiConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    plane_detection: PlaneDetectionConfig = field(default_factory=PlaneDetectionConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    @classmethod
    def from_file(cls, config_path: str | Path) -> "PipelineConfig":
        config_path = Path(config_path).resolve()
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        base_dir = config_path.parent

        if "input_las" not in payload:
            raise ValueError("Config must define input_las.")
        if "output_obj" not in payload:
            raise ValueError("Config must define output_obj.")
        if "output_report" not in payload:
            raise ValueError("Config must define output_report.")

        return cls(
            input_las=_resolve_path(base_dir, payload["input_las"]),
            output_obj=_resolve_path(base_dir, payload["output_obj"]),
            output_report=_resolve_path(base_dir, payload["output_report"]),
            debug_cloud_dir=_resolve_path(base_dir, payload.get("debug_cloud_dir")),
            las_chunk_size=int(payload.get("las_chunk_size", 1_000_000)),
            translate_to_local_origin=bool(payload.get("translate_to_local_origin", True)),
            roi=RoiConfig.from_dict(payload.get("roi")),
            preprocess=PreprocessConfig.from_dict(payload.get("preprocess")),
            plane_detection=PlaneDetectionConfig.from_dict(payload.get("plane_detection")),
            mesh=MeshConfig.from_dict(payload.get("mesh")),
            debug=DebugConfig.from_dict(payload.get("debug")),
        )
