#!/usr/bin/env python3
"""End-to-end pyLAST validation for LACT_sim measured electronics ROOT files."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

np = None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def image(event, level: str, tel_id: int) -> np.ndarray:
    container = getattr(event, level)
    require(container is not None, f"event has no {level} container")
    return np.asarray(container.tels[tel_id].image, dtype=np.float64)


def load_no_waveform(
    path: Path, expected_sum: float, source_type
) -> dict[str, float]:
    source = source_type(str(path))
    require(len(source) == 1, f"{path}: expected one event")
    event = source[0]
    require(event.event_id == 1909, f"{path}: event id is not 1909")
    dl0 = image(event, "dl0", 19)
    require(np.isclose(dl0.sum(), expected_sum, rtol=0, atol=1e-12),
            f"{path}: integrated DL0 sum differs")
    return {"dl0_sum_pe": float(dl0.sum())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("measured_root", type=Path)
    parser.add_argument("--saturation-off-root", type=Path, required=True)
    parser.add_argument("--saturation-on-root", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--package-dir",
        type=Path,
        help="wheel target directory to put first on sys.path",
    )
    parser.add_argument(
        "--dependency-dir",
        action="append",
        type=Path,
        default=[],
        help="additional dependency directory, useful together with python -S",
    )
    args = parser.parse_args()

    # Some shared validation machines prepend a development checkout through
    # sitecustomize.  Make the wheel under test authoritative when requested.
    for dependency_dir in reversed(args.dependency_dir):
        sys.path.insert(0, str(dependency_dir.resolve()))
    if args.package_dir:
        sys.path.insert(0, str(args.package_dir.resolve()))
    global np
    np = importlib.import_module("numpy")
    pylast = importlib.import_module("pylast")
    helper = importlib.import_module("pylast.helper")
    Calibrator = helper.Calibrator
    LactEventSource = helper.LactEventSource
    print(f"pylast_package={pylast.__file__}")
    print(f"pylast_helper={helper.__file__}")

    source = LactEventSource(str(args.measured_root))
    require(len(source) == 1, "measured ROOT must contain exactly one event")
    require(19 in source.subarray.tels, "telescope 19 is missing from subarray")
    readout = source.subarray.tels[19].camera.readout
    require(readout.reference_pulse_shape.shape == (1, 1101),
            "measured reference pulse shape differs")

    raw_event = source[0]
    require(raw_event.event_id == 1909, "measured ROOT event id is not 1909")
    require(raw_event.simulation is not None, "simulation truth is missing")
    true_image = np.asarray(
        raw_event.simulation.tels[19].true_image, dtype=np.float64
    )
    raw_waveform = np.asarray(raw_event.r1.tels[19].waveform, dtype=np.float64)
    require(raw_waveform.shape == (1664, 65), f"unexpected waveform shape {raw_waveform.shape}")
    direct_full_pe = float(raw_waveform.sum())

    full_event = source[0]
    full_calibrator = Calibrator(
        source.subarray,
        json.dumps({"image_extractor_type": "FullWaveFormExtractor"}),
    )
    full_calibrator(full_event)
    full_image = image(full_event, "dl0", 19)
    require(np.allclose(full_image.sum(), direct_full_pe, rtol=0, atol=1e-8),
            "FullWaveFormExtractor does not close to normalized R1 p.e. charge")

    local_event = source[0]
    local_config = {
        "image_extractor_type": "LocalPeakExtractor",
        "LocalPeakExtractor": {
            "window_width": 7,
            "window_shift": 3,
            "apply_correction": True,
        },
    }
    local_calibrator = Calibrator(source.subarray, json.dumps(local_config))
    local_calibrator(local_event)
    local_image = image(local_event, "dl0", 19)
    local_relative = float((local_image.sum() - full_image.sum()) / full_image.sum())
    require(abs(local_relative) < 0.01,
            f"LocalPeakExtractor corrected integral differs by {local_relative:.3%}")

    report = {
        "event_id": int(raw_event.event_id),
        "telescope_id": 19,
        "r1_sample_unit": "pe_charge_per_sample",
        "reference_pulse_points": int(readout.reference_pulse_shape.shape[1]),
        "waveform_shape": list(raw_waveform.shape),
        "true_cherenkov_image_sum_pe": float(true_image.sum()),
        "direct_full_waveform_pe": direct_full_pe,
        "full_waveform_extractor_pe": float(full_image.sum()),
        "full_waveform_closure_pe": float(full_image.sum() - direct_full_pe),
        "local_peak_corrected_pe": float(local_image.sum()),
        "local_peak_relative_to_full": local_relative,
        "saturation_off_waveform_off": load_no_waveform(
            args.saturation_off_root, 3235.0, LactEventSource
        ),
        "saturation_on_waveform_off": load_no_waveform(
            args.saturation_on_root, 3196.0, LactEventSource
        ),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
