from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PipelineConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert indoor LAS point clouds into lightweight OBJ structural meshes."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the LAS -> OBJ pipeline.")
    run_parser.add_argument("--config", required=True, help="Path to the JSON config file.")

    sample_parser = subparsers.add_parser(
        "print-sample-config", help="Print a sample pipeline config."
    )
    sample_parser.add_argument(
        "--output",
        help="Optional output path. If omitted, the sample JSON is printed to stdout.",
    )

    roi_parser = subparsers.add_parser("pick-roi", help="Interactively pick an ROI from a LAS preview.")
    source_group = roi_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--config", help="Path to an existing pipeline config.")
    source_group.add_argument("--input-las", help="Path to the LAS file.")
    roi_parser.add_argument("--chunk-size", type=int, default=1_000_000)
    roi_parser.add_argument("--max-preview-points", type=int, default=400_000)
    roi_parser.add_argument("--preview-voxel-size", type=float, default=0.08)
    roi_parser.add_argument("--padding", type=float, default=0.0)
    roi_parser.add_argument(
        "--output",
        help="Optional JSON file path for the ROI snippet. If omitted, the ROI is printed to stdout.",
    )
    roi_parser.add_argument(
        "--write-back",
        action="store_true",
        help="If --config is used, write the ROI back into that config file.",
    )

    return parser


def _sample_config() -> dict[str, object]:
    return {
        "input_las": "data/floor_01.las",
        "output_obj": "output/floor_01.obj",
        "output_report": "output/floor_01.report.json",
        "debug_cloud_dir": "output/debug/floor_01",
        "las_chunk_size": 1000000,
        "translate_to_local_origin": True,
        "roi": {
            "enabled": True,
            "min": [0.0, 0.0, -5.0],
            "max": [60.0, 45.0, 8.0],
        },
        "preprocess": {
            "voxel_size": 0.05,
            "statistical_outlier_neighbors": 24,
            "statistical_outlier_std_ratio": 2.5,
        },
        "plane_detection": {
            "distance_threshold": 0.03,
            "ransac_n": 3,
            "num_iterations": 2000,
            "min_plane_points": 8000,
            "max_planes": 32,
            "horizontal_angle_threshold_degrees": 15.0,
            "extent_trim_percent": 2.0,
        },
        "mesh": {
            "min_plane_extent": 0.4,
            "merge_normal_angle_degrees": 5.0,
            "merge_plane_offset_threshold": 0.08,
            "merge_gap_threshold": 1.5,
            "merge_min_overlap": 0.25,
            "enable_opening_detection": True,
            "opening_grid_resolution": 0.1,
            "opening_neighbor_fill_radius": 1,
            "opening_min_width": 0.5,
            "opening_min_height": 0.5,
            "opening_min_area": 0.35,
            "door_max_width": 2.0,
        },
        "debug": {"export_intermediate_clouds": True},
    }


def _write_roi_back_to_config(config_path: Path, roi_payload: dict[str, object]) -> None:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["roi"] = {
        "enabled": True,
        "min": roi_payload["min"],
        "max": roi_payload["max"],
    }
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        from .pipeline import run_pipeline

        config = PipelineConfig.from_file(args.config)
        report = run_pipeline(config)
        print(
            json.dumps(
                {
                    "output_obj": report["output_obj"],
                    "patch_count": report["patch_count"],
                    "label_counts": report["label_counts"],
                    "opening_counts": report["opening_counts"],
                    "remaining_points": report["remaining_points"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "print-sample-config":
        payload = json.dumps(_sample_config(), ensure_ascii=False, indent=2)
        if args.output:
            output_path = Path(args.output).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
        return 0

    if args.command == "pick-roi":
        from .roi_picker import pick_roi_interactively, write_roi_json

        config_path: Path | None = None
        if args.config:
            config_path = Path(args.config).resolve()
            config = PipelineConfig.from_file(config_path)
            input_las = config.input_las
        else:
            input_las = Path(args.input_las).resolve()

        roi_payload = pick_roi_interactively(
            input_las=input_las,
            chunk_size=args.chunk_size,
            max_preview_points=args.max_preview_points,
            preview_voxel_size=args.preview_voxel_size,
            padding=args.padding,
        )

        if args.output:
            write_roi_json(Path(args.output).resolve(), roi_payload)
        else:
            print(json.dumps(roi_payload, ensure_ascii=False, indent=2))

        if args.write_back:
            if config_path is None:
                parser.error("--write-back requires --config.")
            _write_roi_back_to_config(config_path, roi_payload)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
