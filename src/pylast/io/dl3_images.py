"""Scalar, per-telescope image diagnostics linked to the DL3 EVENTS table.

This module copies measured DL1 parameters; it does not reconstruct images,
infer DAQ triggers, or export simulation-only/default ``extra`` parameters.
"""

from collections.abc import Mapping
import math

import numpy as np

__all__ = ["event_to_dl3_image_rows", "IMAGE_DTYPES", "IMAGE_UNITS", "IMAGE_DESCRIPTIONS"]

# Photoelectron counts use FITS' count unit, with the calibration stated below.
_PARAMETERS = {
    "HILLAS_INTENSITY": ("hillas", "intensity", "ct", "Cleaned image charge in calibrated photoelectrons (p.e.)."),
    "HILLAS_LENGTH": ("hillas", "length", "deg", "Charge-weighted major-axis angular extent."),
    "HILLAS_WIDTH": ("hillas", "width", "deg", "Charge-weighted minor-axis angular extent."),
    "HILLAS_X": ("hillas", "x", "deg", "Charge centroid in the pylast camera x coordinate."),
    "HILLAS_Y": ("hillas", "y", "deg", "Charge centroid in the pylast camera y coordinate."),
    "HILLAS_R": ("hillas", "r", "deg", "Centroid distance from the camera origin."),
    "HILLAS_PHI": ("hillas", "phi", "deg", "Centroid polar angle in the camera coordinate system."),
    "HILLAS_PSI": ("hillas", "psi", "deg", "Major-axis orientation in the camera coordinate system."),
    "HILLAS_SKEWNESS": ("hillas", "skewness", "", "Standardized third longitudinal image moment."),
    "HILLAS_KURTOSIS": ("hillas", "kurtosis", "", "Standardized fourth longitudinal moment, not excess kurtosis."),
    "LEAKAGE_PIXELS_WIDTH_1": ("leakage", "pixels_width_1", "", "Fraction of positive cleaned pixels in the outermost border."),
    "LEAKAGE_PIXELS_WIDTH_2": ("leakage", "pixels_width_2", "", "Fraction of positive cleaned pixels in the outer two borders."),
    "LEAKAGE_INTENSITY_WIDTH_1": ("leakage", "intensity_width_1", "", "Fraction of cleaned charge in the outermost border."),
    "LEAKAGE_INTENSITY_WIDTH_2": ("leakage", "intensity_width_2", "", "Fraction of cleaned charge in the outer two borders."),
    "CONCENTRATION_COG": ("concentration", "concentration_cog", "", "Charge fraction within one pixel diameter of the centroid."),
    "CONCENTRATION_CORE": ("concentration", "concentration_core", "", "Charge fraction inside the Hillas ellipse."),
    "CONCENTRATION_PIXEL": ("concentration", "concentration_pixel", "", "Brightest-pixel fraction of cleaned charge."),
    "N_PIXELS": ("morphology", "n_pixels", "", "Number of pixels in the cleaning mask."),
    "N_ISLANDS": ("morphology", "n_islands", "", "Number of connected cleaned pixel islands."),
    "N_SMALL_ISLANDS": ("morphology", "n_small_islands", "", "Small-island count using the configured image processor definition."),
    "N_MEDIUM_ISLANDS": ("morphology", "n_medium_islands", "", "Medium-island count using the configured image processor definition."),
    "N_LARGE_ISLANDS": ("morphology", "n_large_islands", "", "Large-island count using the configured image processor definition."),
    "INTENSITY_MAX": ("intensity", "intensity_max", "ct", "Maximum cleaned pixel charge in p.e."),
    "INTENSITY_MEAN": ("intensity", "intensity_mean", "ct", "Image processor mean of positive cleaned pixel charges in p.e."),
    "INTENSITY_STD": ("intensity", "intensity_std", "ct", "Image processor population standard deviation of positive charges in p.e."),
    "INTENSITY_SKEWNESS": ("intensity", "intensity_skewness", "", "Image processor skewness of positive cleaned charges."),
    "INTENSITY_KURTOSIS": ("intensity", "intensity_kurtosis", "", "Image processor excess kurtosis of positive cleaned charges."),
}
_FLAGS = {
    "IMAGE_VALID": "valid_image", "IMAGE_CUT_PASS": "image_cut_pass",
    "RECONSTRUCTION_VALID": "reconstruction_valid", "QUALITY_PASS": "quality_pass",
    "PID_PASS": "pass_pid", "ACCEPTED": "accepted",
}
_SCORES = {
    "DISP_RECO": "disp_reco_deg", "GAMMANESS": "pid_gamma_score",
    "PID_THRESHOLD": "pid_threshold_gamma_score", "QUALITY_PROXY": "quality_proxy_log_error",
    "QUALITY_THRESHOLD": "quality_threshold_smooth",
}
IMAGE_DTYPES = {
    "OBS_ID": "int64", "EVENT_ID": "int64", "TEL_ID": "int32",
    "HAS_PARAMETERS": "int8", **{name: "float64" for name in _PARAMETERS},
    **{name: "int8" for name in _FLAGS},
    "USED_IN_DIRECTION": "int8", "USED_IN_ENERGY": "int8",
    **{name: "float64" for name in _SCORES},
    "MODEL_GROUP": "U128", "MODEL_BRANCH": "U128",
}
IMAGE_UNITS = {name: "" for name in IMAGE_DTYPES}
IMAGE_UNITS.update({name: spec[2] for name, spec in _PARAMETERS.items()})
IMAGE_UNITS["DISP_RECO"] = "deg"
IMAGE_DESCRIPTIONS = {
    "OBS_ID": "Observation identifier; foreign key to EVENTS.",
    "EVENT_ID": "Event identifier within OBS_ID; foreign key to EVENTS.",
    "TEL_ID": "Native telescope identifier, without adding one for display.",
    "HAS_PARAMETERS": "Finite positive Hillas size/length/width and finite centroid/orientation: 1=yes, 0=no.",
    **{name: spec[3] for name, spec in _PARAMETERS.items()},
    "IMAGE_VALID": "Matched model's image-validity flag, not a DAQ trigger; -1=unknown.",
    "IMAGE_CUT_PASS": "Matched model's image-selection result; -1=unknown.",
    "RECONSTRUCTION_VALID": "Matched per-image reconstruction-validity flag; -1=unknown.",
    "QUALITY_PASS": "Matched model quality-cut result; -1=unknown.",
    "PID_PASS": "Matched model gamma-selection result; -1=unknown.",
    "ACCEPTED": "Matched model's final per-image selection; -1=unknown.",
    "USED_IN_DIRECTION": "Membership in named geometry.telescopes, independent of final acceptance; -1=unknown.",
    "USED_IN_ENERGY": "Membership in named energy.telescopes, independent of final acceptance; -1=unknown.",
    "DISP_RECO": "Matched model's signed image-axis displacement prediction.",
    "GAMMANESS": "Matched model's gamma-class score, not a measured particle probability.",
    "PID_THRESHOLD": "Gamma-score threshold used for the matched model/group/branch.",
    "QUALITY_PROXY": "Prediction of log10(max(angular error / deg, 1e-4)); a quality proxy, not a measured error or confidence interval.",
    "QUALITY_THRESHOLD": "Quality-proxy threshold for the matched model/group/branch; lower proxy passes.",
    "MODEL_GROUP": "Training/threshold domain label, not a telescope multiplicity.",
    "MODEL_BRANCH": "Model selection branch, e.g. fixed; not inferred from the number of DL1 images.",
}


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def _flag(value):
    if value is None or isinstance(value, (float, np.floating)) and np.isnan(value):
        return -1
    if not isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)) or value not in (-1, 0, 1):
        raise ValueError("Prediction flags must be boolean or -1/NaN when unknown")
    return int(value)


def _predictions(prediction, event):
    if prediction is None:
        return {}
    if hasattr(prediction, "to_dict"):
        prediction = prediction.to_dict("records") if getattr(prediction, "ndim", None) == 2 else prediction.to_dict()
    records = [prediction] if isinstance(prediction, Mapping) else list(prediction)
    matched = {}
    for record in records:
        if not isinstance(record, Mapping) or "event_id" not in record or "telescope_id" not in record:
            raise ValueError("Each prediction requires event_id and telescope_id")
        if record["event_id"] != event.event_id or ("run_id" in record and record["run_id"] != event.run_id):
            raise ValueError("Prediction event/run identity does not match the event")
        tel_id = int(record["telescope_id"])
        if tel_id != record["telescope_id"] or tel_id in matched:
            raise ValueError("Prediction telescope_id must be integral and unique within the event")
        matched[tel_id] = record
    return matched


def _participants(event, level, name):
    if name is None:
        return None
    result = getattr(getattr(event, "dl2", None), level, {}).get(name)
    return None if result is None else set(result.telescopes)


def event_to_dl3_image_rows(event, prediction=None, *, obs_id, geometry_name=None, energy_name=None):
    """Copy one row per measured ``event.dl1.tels`` camera without moving pixels.

    ``prediction`` may be a mapping/Series, a sequence of mappings, or a
    DataFrame for this event. Scores are attached only by matching telescope ID;
    duplicate telescope or mismatched event/run keys raise rather than guessing.
    Named DL2 telescope membership is retained even when its acceptance is false.
    Unknown flags are -1; undefined numeric values are NaN; absent model labels
    are empty strings. Angles are converted from native radians to degrees.
    Photoelectron counts use unit ``ct`` with p.e. stated in the descriptions.
    """
    predictions = _predictions(prediction, event)
    direction = _participants(event, "geometry", geometry_name)
    energy = _participants(event, "energy", energy_name)
    cameras = getattr(getattr(event, "dl1", None), "tels", {})
    rows = []
    for tel_id, camera in sorted(cameras.items()):
        row = {name: ("" if dtype.startswith("U") else -1 if dtype == "int8" else math.nan)
               for name, dtype in IMAGE_DTYPES.items()}
        row.update(OBS_ID=int(obs_id), EVENT_ID=int(event.event_id), TEL_ID=int(tel_id))
        parameters = getattr(camera, "image_parameters", None)
        for name, (group, attribute, unit, _) in _PARAMETERS.items():
            value = _number(getattr(getattr(parameters, group, None), attribute, None))
            row[name] = math.degrees(value) if unit == "deg" else value
        core = [row[name] for name in ("HILLAS_INTENSITY", "HILLAS_LENGTH", "HILLAS_WIDTH",
                                       "HILLAS_X", "HILLAS_Y", "HILLAS_R", "HILLAS_PSI")]
        row["HAS_PARAMETERS"] = int(np.isfinite(core).all() and min(core[:3]) > 0)
        if not row["HAS_PARAMETERS"]:
            row.update({name: math.nan for name in _PARAMETERS})
        row["USED_IN_DIRECTION"] = -1 if direction is None else int(tel_id in direction)
        row["USED_IN_ENERGY"] = -1 if energy is None else int(tel_id in energy)
        record = predictions.get(tel_id, {})
        for name, key in _FLAGS.items():
            row[name] = _flag(record.get(key))
        for name, key in _SCORES.items():
            row[name] = _number(record.get(key))
        for name, key in (("MODEL_GROUP", "group"), ("MODEL_BRANCH", "branch")):
            value = record.get(key)
            if value is not None and not (isinstance(value, (float, np.floating)) and np.isnan(value)):
                if not isinstance(value, str) or len(value) > 128:
                    raise ValueError(f"{key} must be a string of at most 128 characters")
                row[name] = value
        rows.append(row)
    return rows
