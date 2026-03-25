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
class GeometryTests(unittest.TestCase):
    def test_classify_plane_distinguishes_floor_ceiling_and_wall(self) -> None:
        from las_to_obj.geometry import classify_plane

        self.assertEqual(classify_plane(np.array([0.0, 0.0, 1.0]), -1.0, 0.0, 15.0), "floor")
        self.assertEqual(classify_plane(np.array([0.0, 0.0, 1.0]), 3.0, 0.0, 15.0), "ceiling")
        self.assertEqual(classify_plane(np.array([1.0, 0.0, 0.0]), 1.0, 0.0, 15.0), "wall")

    def test_flatten_points_to_plane_moves_points_onto_plane(self) -> None:
        from las_to_obj.geometry import flatten_points_to_plane

        points = np.array(
            [
                [0.0, 0.0, 1.2],
                [1.0, 0.0, 0.8],
                [0.5, 1.0, 1.5],
            ]
        )
        flattened = flatten_points_to_plane(points, (0.0, 0.0, 1.0, -1.0))
        np.testing.assert_allclose(flattened[:, 2], np.ones(3), atol=1e-8)

    def test_build_patch_from_region_returns_mesh(self) -> None:
        from las_to_obj.geometry import build_patch_from_region, make_plane_region

        xs = np.linspace(-2.0, 2.0, 5)
        ys = np.linspace(-1.0, 1.0, 5)
        points = np.array([[x, y, 1.0] for x in xs for y in ys], dtype=float)
        region = make_plane_region(
            label="floor",
            plane_model=(0.0, 0.0, 1.0, -1.0),
            flattened_points=points,
            source_name="floor_000",
        )

        patch = build_patch_from_region(
            region=region,
            name="floor_000",
            trim_percent=0.0,
            min_plane_extent=0.2,
        )

        self.assertIsNotNone(patch)
        assert patch is not None
        self.assertEqual(patch.label, "floor")
        self.assertEqual(len(patch.vertices), 4)
        self.assertEqual(len(patch.faces), 2)
        self.assertEqual(sorted(round(extent, 6) for extent in patch.extents), [2.0, 4.0])


if __name__ == "__main__":
    unittest.main()

