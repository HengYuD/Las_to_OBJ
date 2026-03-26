from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from las_to_obj.config import PipelineConfig


class ConfigTests(unittest.TestCase):
    def test_from_file_resolves_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "pipeline.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input_las": "data/example.las",
                        "output_obj": "output/model.obj",
                        "output_report": "output/report.json",
                    }
                ),
                encoding="utf-8",
            )

            config = PipelineConfig.from_file(config_path)

            self.assertEqual(config.input_las, (tmp_path / "data/example.las").resolve())
            self.assertEqual(config.output_obj, (tmp_path / "output/model.obj").resolve())
            self.assertEqual(config.output_report, (tmp_path / "output/report.json").resolve())

    def test_plane_detection_supports_structure_filter_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "pipeline.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input_las": "data/example.las",
                        "output_obj": "output/model.obj",
                        "output_report": "output/report.json",
                        "plane_detection": {
                            "min_plane_points": 5000,
                            "min_wall_points": 1800,
                            "max_horizontal_planes": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = PipelineConfig.from_file(config_path)
            self.assertEqual(config.plane_detection.min_plane_points, 5000)
            self.assertEqual(config.plane_detection.min_wall_points, 1800)
            self.assertEqual(config.plane_detection.max_horizontal_planes, 1)


if __name__ == "__main__":
    unittest.main()
