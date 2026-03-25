from __future__ import annotations

import math
from collections import deque

import numpy as np

from .config import MeshConfig
from .geometry import (
    build_patch_from_region,
    canonicalize_plane_model,
    compute_local_bounds,
    plane_basis,
    project_points_to_plane_axes,
)
from .models import PlaneOpening, PlanePatch, PlaneRegion


def _interval_gap(a: tuple[float, float], b: tuple[float, float]) -> float:
    if a[1] < b[0]:
        return float(b[0] - a[1])
    if b[1] < a[0]:
        return float(a[0] - b[1])
    return 0.0


def _interval_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _region_projection(
    region: PlaneRegion,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    plane_model = canonicalize_plane_model(region.plane_model)
    normal = np.asarray(plane_model[:3], dtype=float)
    centroid = region.flattened_points.mean(axis=0)
    tangent_u, tangent_v = plane_basis(normal)
    local_points = project_points_to_plane_axes(region.flattened_points, centroid, tangent_u, tangent_v)
    bounds = compute_local_bounds(local_points, trim_percent=0.0)
    return centroid, tangent_u, tangent_v, bounds


def _can_merge_regions(a: PlaneRegion, b: PlaneRegion, mesh_config: MeshConfig) -> bool:
    if a.label != "wall" or b.label != "wall":
        return False

    a_plane = canonicalize_plane_model(a.plane_model)
    b_plane = canonicalize_plane_model(b.plane_model)
    a_normal = np.asarray(a_plane[:3], dtype=float)
    b_normal = np.asarray(b_plane[:3], dtype=float)
    cosine = np.clip(float(np.dot(a_normal, b_normal)), -1.0, 1.0)
    angle = math.degrees(math.acos(cosine))
    if angle > mesh_config.merge_normal_angle_degrees:
        return False

    if abs(float(a_plane[3] - b_plane[3])) > mesh_config.merge_plane_offset_threshold:
        return False

    origin = np.vstack((a.flattened_points, b.flattened_points)).mean(axis=0)
    tangent_u, tangent_v = plane_basis(a_normal)
    a_local = project_points_to_plane_axes(a.flattened_points, origin, tangent_u, tangent_v)
    b_local = project_points_to_plane_axes(b.flattened_points, origin, tangent_u, tangent_v)
    a_bounds = compute_local_bounds(a_local, trim_percent=0.0)
    b_bounds = compute_local_bounds(b_local, trim_percent=0.0)

    gap_u = _interval_gap((a_bounds[0], a_bounds[1]), (b_bounds[0], b_bounds[1]))
    gap_v = _interval_gap((a_bounds[2], a_bounds[3]), (b_bounds[2], b_bounds[3]))
    overlap_u = _interval_overlap((a_bounds[0], a_bounds[1]), (b_bounds[0], b_bounds[1]))
    overlap_v = _interval_overlap((a_bounds[2], a_bounds[3]), (b_bounds[2], b_bounds[3]))

    return (
        (gap_u <= mesh_config.merge_gap_threshold and overlap_v >= mesh_config.merge_min_overlap)
        or (gap_v <= mesh_config.merge_gap_threshold and overlap_u >= mesh_config.merge_min_overlap)
        or (gap_u == 0.0 and gap_v == 0.0)
    )


def _merge_region_pair(a: PlaneRegion, b: PlaneRegion) -> PlaneRegion:
    points = np.vstack((a.flattened_points, b.flattened_points))
    a_plane = canonicalize_plane_model(a.plane_model)
    b_plane = canonicalize_plane_model(b.plane_model)
    normal = np.asarray(a_plane[:3], dtype=float) * a.point_count + np.asarray(
        b_plane[:3], dtype=float
    ) * b.point_count
    normal = normal / np.linalg.norm(normal)
    d = -float((points @ normal).mean())
    plane_model = canonicalize_plane_model((normal[0], normal[1], normal[2], d))
    return PlaneRegion(
        label=a.label,
        plane_model=plane_model,
        flattened_points=points,
        source_names=[*a.source_names, *b.source_names],
    )


def merge_coplanar_wall_regions(
    regions: list[PlaneRegion],
    mesh_config: MeshConfig,
) -> list[PlaneRegion]:
    pending = list(regions)
    merged: list[PlaneRegion] = []

    while pending:
        current = pending.pop(0)
        changed = True
        while changed:
            changed = False
            survivors: list[PlaneRegion] = []
            for candidate in pending:
                if _can_merge_regions(current, candidate, mesh_config):
                    current = _merge_region_pair(current, candidate)
                    changed = True
                else:
                    survivors.append(candidate)
            pending = survivors
        merged.append(current)

    return merged


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    result = mask.copy()
    for _ in range(max(0, radius)):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        neighborhoods = [
            padded[1 + dy : 1 + dy + result.shape[0], 1 + dx : 1 + dx + result.shape[1]]
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
        ]
        result = np.logical_or.reduce(neighborhoods)
    return result


def _connected_components(mask: np.ndarray) -> list[tuple[list[tuple[int, int]], bool, bool, bool, bool]]:
    rows, cols = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[tuple[list[tuple[int, int]], bool, bool, bool, bool]] = []

    for row in range(rows):
        for col in range(cols):
            if visited[row, col] or not mask[row, col]:
                continue
            queue = deque([(row, col)])
            visited[row, col] = True
            cells: list[tuple[int, int]] = []
            touches_bottom = False
            touches_top = False
            touches_left = False
            touches_right = False

            while queue:
                current_row, current_col = queue.popleft()
                cells.append((current_row, current_col))
                touches_bottom |= current_row == 0
                touches_top |= current_row == rows - 1
                touches_left |= current_col == 0
                touches_right |= current_col == cols - 1

                for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col
                    if not (0 <= next_row < rows and 0 <= next_col < cols):
                        continue
                    if visited[next_row, next_col] or not mask[next_row, next_col]:
                        continue
                    visited[next_row, next_col] = True
                    queue.append((next_row, next_col))

            components.append((cells, touches_bottom, touches_top, touches_left, touches_right))

    return components


def _component_bounds(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [cell[0] for cell in cells]
    cols = [cell[1] for cell in cells]
    return (min(rows), max(rows), min(cols), max(cols))


def _decompose_solid_mask(solid_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    mask = solid_mask.copy()
    rows, cols = mask.shape
    rectangles: list[tuple[int, int, int, int]] = []

    for row in range(rows):
        col = 0
        while col < cols:
            if not mask[row, col]:
                col += 1
                continue

            width = 1
            while col + width < cols and mask[row, col + width]:
                width += 1

            height = 1
            while row + height < rows and np.all(mask[row + height, col : col + width]):
                height += 1

            mask[row : row + height, col : col + width] = False
            rectangles.append((row, row + height - 1, col, col + width - 1))
            col += width

    return rectangles


def detect_wall_openings_and_rectangles(
    region: PlaneRegion,
    mesh_config: MeshConfig,
    trim_percent: float,
) -> tuple[list[PlaneOpening], list[tuple[float, float, float, float]]]:
    centroid, tangent_u, tangent_v, bounds = _region_projection(region)
    local_points = project_points_to_plane_axes(region.flattened_points, centroid, tangent_u, tangent_v)
    if trim_percent > 0:
        bounds = compute_local_bounds(local_points, trim_percent)

    width = float(bounds[1] - bounds[0])
    height = float(bounds[3] - bounds[2])
    if width < mesh_config.min_plane_extent or height < mesh_config.min_plane_extent:
        return [], []

    resolution = max(0.02, mesh_config.opening_grid_resolution)
    cols = max(1, int(math.ceil(width / resolution)))
    rows = max(1, int(math.ceil(height / resolution)))
    cell_u = width / cols
    cell_v = height / rows

    occupancy = np.zeros((rows, cols), dtype=bool)
    normalized = np.empty_like(local_points)
    normalized[:, 0] = (local_points[:, 0] - bounds[0]) / width
    normalized[:, 1] = (local_points[:, 1] - bounds[2]) / height
    col_indices = np.clip((normalized[:, 0] * cols).astype(int), 0, cols - 1)
    row_indices = np.clip((normalized[:, 1] * rows).astype(int), 0, rows - 1)
    occupancy[row_indices, col_indices] = True
    occupancy = _dilate(occupancy, mesh_config.opening_neighbor_fill_radius)

    empty_mask = ~occupancy
    components = _connected_components(empty_mask)
    openings: list[PlaneOpening] = []
    solid_mask = np.ones((rows, cols), dtype=bool)

    for cells, touches_bottom, touches_top, touches_left, touches_right in components:
        row_min, row_max, col_min, col_max = _component_bounds(cells)
        opening_width = (col_max - col_min + 1) * cell_u
        opening_height = (row_max - row_min + 1) * cell_v
        opening_area = opening_width * opening_height

        if opening_width < mesh_config.opening_min_width:
            continue
        if opening_height < mesh_config.opening_min_height:
            continue
        if opening_area < mesh_config.opening_min_area:
            continue

        kind: str | None = None
        if not (touches_bottom or touches_top or touches_left or touches_right):
            kind = "window"
        elif (touches_bottom ^ touches_top) and not touches_left and not touches_right:
            if opening_width <= mesh_config.door_max_width:
                kind = "door"

        if kind is None:
            continue

        u_min = bounds[0] + col_min * cell_u
        u_max = bounds[0] + (col_max + 1) * cell_u
        v_min = bounds[2] + row_min * cell_v
        v_max = bounds[2] + (row_max + 1) * cell_v
        opening = PlaneOpening(
            kind=kind,
            local_bounds=(u_min, u_max, v_min, v_max),
            width=opening_width,
            height=opening_height,
            area=opening_area,
        )
        openings.append(opening)
        solid_mask[row_min : row_max + 1, col_min : col_max + 1] = False

    if not openings:
        return [], [(bounds[0], bounds[1], bounds[2], bounds[3])]

    rectangles: list[tuple[float, float, float, float]] = []
    for row_min, row_max, col_min, col_max in _decompose_solid_mask(solid_mask):
        u_min = bounds[0] + col_min * cell_u
        u_max = bounds[0] + (col_max + 1) * cell_u
        v_min = bounds[2] + row_min * cell_v
        v_max = bounds[2] + (row_max + 1) * cell_v
        rectangles.append((u_min, u_max, v_min, v_max))

    return openings, rectangles


def build_patches_from_regions(
    regions: list[PlaneRegion],
    mesh_config: MeshConfig,
    trim_percent: float,
) -> list[PlanePatch]:
    merged_regions = merge_coplanar_wall_regions(regions, mesh_config)
    patches: list[PlanePatch] = []

    for index, region in enumerate(merged_regions):
        openings: list[PlaneOpening] = []
        rectangles = None
        if region.label == "wall" and mesh_config.enable_opening_detection:
            openings, rectangles = detect_wall_openings_and_rectangles(
                region=region,
                mesh_config=mesh_config,
                trim_percent=trim_percent,
            )

        patch = build_patch_from_region(
            region=region,
            name=f"{region.label}_{index:03d}",
            trim_percent=trim_percent,
            min_plane_extent=mesh_config.min_plane_extent,
            rectangles=rectangles,
            openings=openings,
        )
        if patch is not None:
            patches.append(patch)

    return patches
