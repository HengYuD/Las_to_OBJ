from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import numpy as np
except ImportError:  # pragma: no cover - environment-dependent
    np = None


@unittest.skipIf(np is None, "numpy is not installed in the current environment")
class PostprocessTests(unittest.TestCase):
    def test_merge_coplanar_wall_regions_merges_nearby_segments(self) -> None:
        from las_to_obj.config import MeshConfig
        from las_to_obj.geometry import make_plane_region
        from las_to_obj.postprocess import merge_coplanar_wall_regions

        wall_a = np.array([[0.0, y, z] for y in np.linspace(0.0, 1.0, 20) for z in np.linspace(0.0, 3.0, 10)])
        wall_b = np.array([[0.0, y, z] for y in np.linspace(1.2, 2.2, 20) for z in np.linspace(0.0, 3.0, 10)])
        regions = [
            make_plane_region("wall", (1.0, 0.0, 0.0, 0.0), wall_a, "wall_a"),
            make_plane_region("wall", (1.0, 0.0, 0.0, 0.0), wall_b, "wall_b"),
        ]

        merged = merge_coplanar_wall_regions(regions, MeshConfig(merge_gap_threshold=0.5))
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].merged_from, 2)

    def test_detect_wall_openings_finds_door_and_window(self) -> None:
        from las_to_obj.config import MeshConfig
        from las_to_obj.geometry import make_plane_region
        from las_to_obj.postprocess import detect_wall_openings_and_rectangles

        points: list[list[float]] = []
        for y in np.arange(0.0, 4.0, 0.05):
            for z in np.arange(0.0, 3.0, 0.05):
                in_door = 1.0 <= y <= 2.0 and 0.0 <= z <= 2.0
                in_window = 2.6 <= y <= 3.4 and 1.2 <= z <= 2.1
                if in_door or in_window:
                    continue
                points.append([0.0, y, z])

        region = make_plane_region(
            label="wall",
            plane_model=(1.0, 0.0, 0.0, 0.0),
            flattened_points=np.asarray(points, dtype=float),
            source_name="wall_000",
        )

        openings, rectangles = detect_wall_openings_and_rectangles(
            region=region,
            mesh_config=MeshConfig(
                opening_grid_resolution=0.1,
                opening_neighbor_fill_radius=0,
                opening_min_width=0.4,
                opening_min_height=0.4,
                opening_min_area=0.2,
                door_max_width=1.5,
            ),
            trim_percent=0.0,
        )

        self.assertEqual(sorted(opening.kind for opening in openings), ["door", "window"])
        self.assertGreater(len(rectangles), 1)


if __name__ == "__main__":
    unittest.main()
