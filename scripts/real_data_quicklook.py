#!/usr/bin/env python3
"""Calibrate one measured LACT event and reuse the existing event displays."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, help="Measured ROOT: waveforms or eventShow")
    parser.add_argument("--mapping-file", required=True, help="Measured camera mapping CSV")
    parser.add_argument("--calibration-file", help="LED led_cal ROOT; omitted means nominal peak conversion")
    parser.add_argument("--analysis-file", help="Matched official eventShow summary for raw waveforms")
    parser.add_argument("--event-id", required=True, type=int, help="LACT event ID, not entry index")
    pointing = parser.add_mutually_exclusive_group(required=True)
    pointing.add_argument("--pointing-file", help="Pointing table with MJD and altitude/azimuth columns")
    pointing.add_argument("--pointing-alt", type=float, help="Explicit telescope altitude in degrees")
    parser.add_argument("--pointing-az", type=float, help="Explicit telescope azimuth in degrees")
    parser.add_argument("--sampling-interval-ns", type=float, help="Verified waveform interval; omitted keeps sample indices")
    parser.add_argument("--km2a-file", help="Matched KM2A sidecar for this LACT build file")
    parser.add_argument("--wcda-file", help="Matched WCDA sidecar for this LACT build file")
    parser.add_argument("-c", "--config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "config" / "real_data.json")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("real_data_quicklook"))
    args = parser.parse_args(argv)
    if (args.pointing_alt is None) != (args.pointing_az is None):
        parser.error("--pointing-alt and --pointing-az must be supplied together")
    if args.sampling_interval_ns is not None and not 0 < args.sampling_interval_ns < float("inf"):
        parser.error("--sampling-interval-ns must be finite and positive")

    # Keep --help usable before the native pylast extension is installed/built.
    import matplotlib
    matplotlib.use("Agg")
    from pylast.calib import LactRealCalibrator
    from pylast.image import ImageProcessor
    from pylast.io import LactRealEventSource
    from pylast.visualize import (
        EventVisualizer, load_km2a_event_record, load_wcda_event_record,
        plot_raw_images, plot_clean_images, plot_image_detail, plot_event_cores,
        plot_km2a_event, plot_wcda_event,
    )

    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_config = dict(config.get("source", {}))
    if args.sampling_interval_ns is not None:
        source_config["sampling_interval_ns"] = args.sampling_interval_ns
    source = LactRealEventSource(
        args.input, args.mapping_file, config_str=source_config,
        analysis_file=args.analysis_file, pointing_file=args.pointing_file,
        pointing_alt=args.pointing_alt, pointing_az=args.pointing_az,
    )
    event = next((event for event in source if event.event_id == args.event_id), None)
    if event is None:
        raise SystemExit(f"LACT event_id={args.event_id} was not found in {args.input}")
    calibrator = LactRealCalibrator(args.calibration_file, config_str=config.get("calibrator", {}))
    calibrator(event)
    ImageProcessor(source.subarray, config_str=json.dumps(config.get("image_processor", {})))(event)
    print(f"event_id={event.event_id}; preprocessing={source.preprocessing}; "
          f"calibration={'LED area' if calibrator.has_calibration else 'nominal peak'}; "
          f"is_led={getattr(event, 'is_led', None) if getattr(event, 'is_led', None) is not None else 'unknown'}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / f"event_{event.event_id}"
    visualizer = EventVisualizer(source)
    external = {}
    for detector, path, load, standalone in (
        ("KM2A", args.km2a_file, load_km2a_event_record, plot_km2a_event),
        ("WCDA", args.wcda_file, load_wcda_event_record, plot_wcda_event),
    ):
        if path:
            record = load(path, event.event_id)
            if record is not None:
                external[detector] = record
                standalone_kwargs = dict(event=event, visualizer=visualizer) if detector == "KM2A" else {}
                standalone(path, event.event_id,
                           output_path=f"{prefix}_{detector.lower()}.png", show=False, **standalone_kwargs)
            else:
                print(f"No accepted {detector} match for event_id={event.event_id}; skipped")

    common = dict(visualizer=visualizer, external_reconstructions=external, show=False)
    plot_raw_images(event, output_path=f"{prefix}_raw.png", **common)
    plot_clean_images(event, output_path=f"{prefix}_clean.png", **common)
    # Public plot numbering is one-based: default native key 0 -> tel_id=1.
    plot_image_detail(event, tel_id=int(source.telescope_id) + 1,
                      output_path=f"{prefix}_detail.png", **common)
    if external:
        plot_event_cores(event, image_level="dl1", output_path=f"{prefix}_cores.png", **common)
    print(f"Plots written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
