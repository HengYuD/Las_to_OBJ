from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class LoadedPointCloud:
    points: np.ndarray
    source_total_points: int
    selected_points: int


@dataclass(slots=True)
class PlaneOpening:
    kind: str
    local_bounds: tuple[float, float, float, float]
    width: float
    height: float
    area: float

    def to_report_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "local_bounds": [
                float(self.local_bounds[0]),
                float(self.local_bounds[1]),
                float(self.local_bounds[2]),
                float(self.local_bounds[3]),
            ],
            "width": float(self.width),
            "height": float(self.height),
            "area": float(self.area),
        }


@dataclass(slots=True)
class PlaneRegion:
    label: str
    plane_model: tuple[float, float, float, float]
    flattened_points: np.ndarray
    source_names: list[str] = field(default_factory=list)

    @property
    def point_count(self) -> int:
        return int(len(self.flattened_points))

    @property
    def merged_from(self) -> int:
        return max(1, len(self.source_names))


@dataclass(slots=True)
class PlanePatch:
    name: str
    label: str
    plane_model: tuple[float, float, float, float]
    point_count: int
    centroid: np.ndarray
    normal: np.ndarray
    extents: tuple[float, float]
    vertices: np.ndarray
    faces: list[tuple[int, int, int]]
    openings: list[PlaneOpening] = field(default_factory=list)
    merged_from: int = 1

    def to_report_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "label": self.label,
            "point_count": self.point_count,
            "plane_model": list(self.plane_model),
            "centroid": self.centroid.tolist(),
            "normal": self.normal.tolist(),
            "extents": [float(self.extents[0]), float(self.extents[1])],
            "face_count": len(self.faces),
            "merged_from": int(self.merged_from),
            "opening_count": len(self.openings),
            "openings": [opening.to_report_dict() for opening in self.openings],
        }
