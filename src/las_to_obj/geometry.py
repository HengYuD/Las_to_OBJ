from __future__ import annotations

import math

import numpy as np

from .models import PlaneOpening, PlanePatch, PlaneRegion


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Zero-length vector cannot be normalized.")
    return vector / norm


def canonicalize_plane_model(
    plane_model: tuple[float, float, float, float] | list[float] | np.ndarray
) -> tuple[float, float, float, float]:
    plane_model = np.asarray(plane_model, dtype=float)
    normal = plane_model[:3]
    norm = float(np.linalg.norm(normal))
    if norm == 0.0:
        raise ValueError("Plane normal cannot be zero.")
    normal = normal / norm
    d = float(plane_model[3]) / norm

    dominant_axis = int(np.argmax(np.abs(normal)))
    if normal[dominant_axis] < 0:
        normal = -normal
        d = -d

    return (float(normal[0]), float(normal[1]), float(normal[2]), d)


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = normalize(normal)
    helper = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(normal, helper))) > 0.9:
        helper = np.array([1.0, 0.0, 0.0], dtype=float)
    tangent_u = normalize(np.cross(normal, helper))
    tangent_v = normalize(np.cross(normal, tangent_u))
    return tangent_u, tangent_v


def flatten_points_to_plane(
    points: np.ndarray, plane_model: tuple[float, float, float, float]
) -> np.ndarray:
    normal = normalize(np.asarray(plane_model[:3], dtype=float))
    d = float(plane_model[3])
    signed_distances = points @ normal + d
    return points - signed_distances[:, None] * normal[None, :]


def project_points_to_plane_axes(
    points: np.ndarray,
    origin: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
) -> np.ndarray:
    centered = points - origin
    return np.column_stack((centered @ tangent_u, centered @ tangent_v))


def classify_plane(
    normal: np.ndarray,
    centroid_z: float,
    z_median: float,
    horizontal_angle_threshold_degrees: float,
) -> str:
    normal = normalize(normal)
    cosine_threshold = math.cos(math.radians(horizontal_angle_threshold_degrees))
    if abs(float(normal[2])) >= cosine_threshold:
        return "floor" if centroid_z <= z_median else "ceiling"
    return "wall"


def compute_local_bounds(
    local_points: np.ndarray, trim_percent: float
) -> tuple[float, float, float, float]:
    u_values = local_points[:, 0]
    v_values = local_points[:, 1]
    if 0.0 < trim_percent < 50.0:
        low = trim_percent
        high = 100.0 - trim_percent
        u_min, u_max = np.percentile(u_values, [low, high])
        v_min, v_max = np.percentile(v_values, [low, high])
    else:
        u_min, u_max = float(u_values.min()), float(u_values.max())
        v_min, v_max = float(v_values.min()), float(v_values.max())
    return (float(u_min), float(u_max), float(v_min), float(v_max))


def make_plane_region(
    label: str,
    plane_model: tuple[float, float, float, float],
    flattened_points: np.ndarray,
    source_name: str,
) -> PlaneRegion:
    plane_model = canonicalize_plane_model(plane_model)
    return PlaneRegion(
        label=label,
        plane_model=plane_model,
        flattened_points=flattened_points,
        source_names=[source_name],
    )


def _mesh_from_rectangles(
    origin: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    normal: np.ndarray,
    rectangles: list[tuple[float, float, float, float]],
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []

    for rectangle in rectangles:
        u_min, u_max, v_min, v_max = rectangle
        base_index = len(vertices)
        local_vertices = [
            origin + u_min * tangent_u + v_min * tangent_v,
            origin + u_max * tangent_u + v_min * tangent_v,
            origin + u_max * tangent_u + v_max * tangent_v,
            origin + u_min * tangent_u + v_max * tangent_v,
        ]
        vertices.extend(local_vertices)

        winding = np.dot(
            np.cross(local_vertices[1] - local_vertices[0], local_vertices[2] - local_vertices[0]),
            normal,
        )
        if winding < 0:
            faces.extend([(base_index, base_index + 2, base_index + 1), (base_index, base_index + 3, base_index + 2)])
        else:
            faces.extend([(base_index, base_index + 1, base_index + 2), (base_index, base_index + 2, base_index + 3)])

    return np.asarray(vertices, dtype=float), faces


def build_patch_from_region(
    region: PlaneRegion,
    name: str,
    trim_percent: float,
    min_plane_extent: float,
    rectangles: list[tuple[float, float, float, float]] | None = None,
    openings: list[PlaneOpening] | None = None,
) -> PlanePatch | None:
    if len(region.flattened_points) == 0:
        return None

    plane_model = canonicalize_plane_model(region.plane_model)
    normal = normalize(np.asarray(plane_model[:3], dtype=float))
    centroid = region.flattened_points.mean(axis=0)
    tangent_u, tangent_v = plane_basis(normal)
    local_points = project_points_to_plane_axes(region.flattened_points, centroid, tangent_u, tangent_v)
    bounds = compute_local_bounds(local_points, trim_percent)
    extent_u = float(bounds[1] - bounds[0])
    extent_v = float(bounds[3] - bounds[2])

    if extent_u < min_plane_extent or extent_v < min_plane_extent:
        return None

    if rectangles is None:
        rectangles = [(bounds[0], bounds[1], bounds[2], bounds[3])]

    vertices, faces = _mesh_from_rectangles(
        origin=centroid,
        tangent_u=tangent_u,
        tangent_v=tangent_v,
        normal=normal,
        rectangles=rectangles,
    )

    return PlanePatch(
        name=name,
        label=region.label,
        plane_model=plane_model,
        point_count=region.point_count,
        centroid=centroid,
        normal=normal,
        extents=(extent_u, extent_v),
        vertices=vertices,
        faces=faces,
        openings=openings or [],
        merged_from=region.merged_from,
    )
