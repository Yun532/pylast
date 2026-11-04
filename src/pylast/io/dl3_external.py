"""Scalar KM2A/WCDA sidecars for a LACT event list; no new coincidence search.

The sidecar ``event`` and Rabbit fields identify the original LACT trigger.
External MJD, deltaT and energy estimates remain explicitly raw: their time
scales, deltaT convention and energy units have not been established here.
"""

from pathlib import Path

import numpy as np

__all__ = ["DL3ExternalSource", "default_external_row", "EXTERNAL_DTYPES",
           "EXTERNAL_UNITS", "EXTERNAL_DESCRIPTIONS"]

# dtype, known unit (empty means unspecified), description without the prefix.
_COMMON = {
    "MATCH_STATE": ("U16", "", "Sidecar state: NOT_REQUESTED, NO_RECORD, UNMATCHED, MATCHED or AMBIGUOUS; MATCHED need not be a unique coincidence."),
    "SOURCE_ENTRY": ("int64", "", "Zero-based sidecar tree entry; -1 when no single source row is represented."),
    "N_RECORDS": ("int64", "", "Number of sidecar rows with this LACT event identifier."),
    "MULTI_CANDIDATE": ("bool", "", "Multiple same-ID rows or an upstream multiplicity greater than one, including upstream-resolved matches."),
    "IDENTITY_VERIFIED": ("bool", "", "Selected/source row has LACT Rabbit seconds/ticks equal to this event (all rows when unresolved); independent of reconstruction validity."),
    "LACT_RABBIT_TIME": ("int64", "s", "Sidecar RabbitTime: LACT identity Unix TAI integer seconds, not the external detector event time; -1 if unavailable."),
    "LACT_RABBIT_TICK": ("int64", "", "Sidecar Rabbittime: LACT identity subsecond count in 8 ns ticks; -1 if unavailable."),
    "MATCH_SUCCESS_RAW": ("float64", "", "Unmodified upstream match_success flag, NaN if absent."),
    "NMATCH_RAW": ("float64", "", "Unmodified upstream Nmatch multiplicity, NaN if absent."),
    "THETA_RAW": ("float64", "rad", "Unmodified native reconstruction zenith angle, including failure sentinel."),
    "PHI_RAW": ("float64", "rad", "Unmodified native reconstruction azimuth, before the documented detector transform."),
    "CORE_X_RAW": ("float64", "m", "Unmodified core x in the detector's native horizontal frame, including failure sentinel."),
    "CORE_Y_RAW": ("float64", "m", "Unmodified core y in the detector's native horizontal frame, including failure sentinel."),
    "ALT": ("float64", "deg", "Reconstructed geometric altitude; NaN unless a selected match has a valid direction."),
    "AZ": ("float64", "deg", "Upstream calibrated sky azimuth north through east in [0,360); absolute pointing not independently calibrated here; NaN unless valid and selected."),
    "CORE_EAST": ("float64", "m", "Core on the common LHAASO_ARRAY_MAP East axis; not surveyed geographic ENU. NaN unless valid and selected."),
    "CORE_NORTH": ("float64", "m", "Core on the common LHAASO_ARRAY_MAP North axis; not surveyed geographic ENU. NaN unless valid and selected."),
    "DIRECTION_VALID": ("bool", "", "Selected match has finite theta/phi and zenith in [0,pi]; distinct from matching and LACT acceptance."),
    "CORE_VALID": ("bool", "", "Selected match has finite core coordinates without the known -10000/-9999 failure sentinels."),
    "DELTA_T_RAW": ("float64", "", "Unmodified upstream deltaT; unit, sign convention and observed approximately 1e7 baseline are unverified. No offset subtracted."),
}
_KM2A = {
    "MKM2A_RAW": ("float64", "", "Unmodified upstream mkm2a matching flag/count."),
    "EVENT_MJD_RAW": ("float64", "d", "Original external Event/mjd; invalid source values occur. Time scale unknown; never used for identification or celestial conversion."),
    "REC_MJD_RAW": ("float64", "d", "Original external Rec/mjd; time scale unknown, not used for celestial-coordinate conversion."),
    "EVENT_ID_RAW": ("int64", "", "Original Event/ev_n; invalid negative source values occur. Not a verified event key; -1 if branch unavailable."),
    "REC_EVENT_ID_RAW": ("int64", "", "Original Rec/ev_n; invalid negative source values occur. Not a verified event key; -1 if branch unavailable."),
    "NP_E1_RAW": ("float64", "", "Original Rec/NpE1 shower particle estimator; normalization/unit unverified."),
    "NU_M4_RAW": ("float64", "", "Original Rec/NuM4 muon estimator; normalization/unit unverified."),
    "RMUE_RAW": ("float64", "", "Original rmue estimator; definition/unit unverified."),
    "GAMMAP_RAW": ("float64", "", "Original gammap classification; coding is not interpreted as a truth label."),
    "EREC_KM2A_RAW": ("float64", "", "Original Erec_km2a estimate; energy unit/calibration unverified, never substituted for LACT ENERGY."),
    "EREC_EU_RAW": ("float64", "", "Original Erec_eu estimate; energy unit/calibration unverified."),
}
_WCDA = {
    "MWCDA_RAW": ("float64", "", "Unmodified upstream mwcda matching flag/count."),
    "MJD_RAW": ("float64", "d", "Original external mjd; time scale unknown, not used for celestial-coordinate conversion."),
    "MJDNS4_RAW": ("float64", "", "Original mjdns4 time component; unit/convention unverified and not combined with mjd."),
    "IRUN_RAW": ("int64", "", "External irun identifier; -1 if unavailable."),
    "IEVENT_RAW": ("int64", "", "External ievent identifier; -1 if unavailable."),
    "ISEQ_RAW": ("int64", "", "External iseq identifier; -1 if unavailable."),
    "MATCHED_ENTRY_RAW": ("int64", "", "Unmodified matched_wcda_entry upstream selection index; -1 if unavailable."),
    "MATCH_SCORE_RAW": ("float64", "", "Original match_score; definition/unit unverified."),
    "NHIT_RAW": ("float64", "", "Original nhit hit multiplicity, including source sentinels."),
    "NHIT_EFF_RAW": ("float64", "", "Original nhiteff effective hit multiplicity, including source sentinels."),
    "GAMMAP_RAW": ("float64", "", "Original gammap classification; coding is not interpreted as a truth label."),
    "PINCNESS_RAW": ("float64", "", "Original pincness shower-shape estimator; normalization/unit unverified."),
    "F5W_RAW": ("float64", "", "Original f5w estimator; definition/unit unverified."),
    "EREC_WCDA_RAW": ("float64", "", "Original Erec_wcda estimate if supplied; energy unit/calibration unverified."),
    "EREC_RAW": ("float64", "", "Original Erec estimate if supplied; energy unit/calibration unverified."),
    "ENERGY_RAW": ("float64", "", "Original energy estimate if supplied; energy unit/calibration unverified."),
    "LOGE_RAW": ("float64", "", "Original logE estimate if supplied; logarithm base/reference energy unverified."),
}
_FIELDS = {f"{detector}_{name}": spec
           for detector, extras in (("KM2A", _KM2A), ("WCDA", _WCDA))
           for name, spec in {**_COMMON, **extras}.items()}
EXTERNAL_DTYPES = {name: spec[0] for name, spec in _FIELDS.items()}
EXTERNAL_UNITS = {name: spec[1] for name, spec in _FIELDS.items() if spec[1]}
EXTERNAL_DESCRIPTIONS = {name: f"{name.split('_', 1)[0]}: {spec[2]}"
                         for name, spec in _FIELDS.items()}

_IDENTITY = {"LACT_RABBIT_TIME": "RabbitTime", "LACT_RABBIT_TICK": "Rabbittime",
             "MATCH_SUCCESS_RAW": "match_success", "NMATCH_RAW": "Nmatch"}
_BRANCHES = {
    "KM2A": {**_IDENTITY, "THETA_RAW": "Rec/rec_theta", "PHI_RAW": "Rec/rec_phi",
             "CORE_X_RAW": "Rec/rec_x", "CORE_Y_RAW": "Rec/rec_y",
             "MKM2A_RAW": "mkm2a", "DELTA_T_RAW": "deltaTkm2a",
             "EVENT_MJD_RAW": "Event/mjd", "REC_MJD_RAW": "Rec/mjd",
             "EVENT_ID_RAW": "Event/ev_n", "REC_EVENT_ID_RAW": "Rec/ev_n",
             "NP_E1_RAW": "Rec/NpE1", "NU_M4_RAW": "Rec/NuM4", "RMUE_RAW": "rmue",
             "GAMMAP_RAW": "gammap", "EREC_KM2A_RAW": "Erec_km2a", "EREC_EU_RAW": "Erec_eu"},
    "WCDA": {**_IDENTITY, "THETA_RAW": "theta", "PHI_RAW": "phi",
             "CORE_X_RAW": "xc", "CORE_Y_RAW": "yc", "MWCDA_RAW": "mwcda",
             "DELTA_T_RAW": "deltaTwcda", "MJD_RAW": "mjd", "MJDNS4_RAW": "mjdns4",
             "IRUN_RAW": "irun", "IEVENT_RAW": "ievent", "ISEQ_RAW": "iseq",
             "MATCHED_ENTRY_RAW": "matched_wcda_entry", "MATCH_SCORE_RAW": "match_score",
             "NHIT_RAW": "nhit", "NHIT_EFF_RAW": "nhiteff", "GAMMAP_RAW": "gammap",
             "PINCNESS_RAW": "pincness", "F5W_RAW": "f5w", "EREC_WCDA_RAW": "Erec_wcda",
             "EREC_RAW": "Erec", "ENERGY_RAW": "energy", "LOGE_RAW": "logE"},
}


def default_external_row():
    """Return a fresh, complete two-detector schema with NOT_REQUESTED states."""
    row = {}
    for name, dtype in EXTERNAL_DTYPES.items():
        row[name] = ("NOT_REQUESTED" if dtype.startswith("U") else False if dtype == "bool"
                     else -1 if dtype == "int64" else float("nan"))
        if name.endswith("_N_RECORDS"):
            row[name] = 0
    return row


def _read_table(path, detector):
    import uproot

    required = ({"event", "Rec/rec_x", "Rec/rec_y", "Rec/rec_theta", "Rec/rec_phi"}
                if detector == "KM2A" else {"event", "theta", "phi", "xc", "yc"})
    flags = {"match_success", "Nmatch", "mkm2a" if detector == "KM2A" else "mwcda"}
    with uproot.open(path) as root:
        trees = [name for name, kind in root.classnames(recursive=True, cycle=False).items()
                 if ("TTree" in kind or "RNTuple" in kind)
                 and required.issubset(root[name].keys())
                 and flags.intersection(root[name].keys())]
        if len(trees) != 1:
            raise ValueError(f"Expected one {detector} matched tree in {path}, found {len(trees)}")
        tree = root[trees[0]]
        branches = {key: branch for key, branch in _BRANCHES[detector].items()
                    if branch in tree.keys()}
        data = {key: np.asarray(tree[branch].array(library="np"))
                for key, branch in {"event": "event", **branches}.items()}
        if any(values.ndim != 1 or len(values) != len(data["event"]) for values in data.values()):
            raise ValueError(f"Non-scalar or inconsistent {detector} sidecar branches")
    return data, {"file": path, "tree": trees[0], "branches": {"event": "event", **branches}}


def _row_state(data, index, detector):
    if "MATCH_SUCCESS_RAW" in data:
        flag = data["MATCH_SUCCESS_RAW"][index]
    else:
        flag = data.get("NMATCH_RAW", data.get(f"M{detector}_RAW"))[index]
    if flag == 1:
        return "MATCHED"
    if np.isfinite(flag) and flag <= 0:
        return "UNMATCHED"
    return "AMBIGUOUS"


def _rabbit(seconds, tick):
    """A legitimate Rabbit pair, or None for an absent/padding timestamp."""
    if (seconds is None or tick is None or not np.all(np.isfinite([seconds, tick]))
            or seconds <= 0 or not 0 <= tick < 125_000_000
            or int(seconds) != seconds or int(tick) != tick):
        return None
    return int(seconds), int(tick)


def _physical_coordinates(row, detector):
    prefix = detector + "_"
    theta, phi = row[prefix + "THETA_RAW"], row[prefix + "PHI_RAW"]
    x, y = row[prefix + "CORE_X_RAW"], row[prefix + "CORE_Y_RAW"]
    valid = bool(np.all(np.isfinite([theta, phi])) and 0 <= theta <= np.pi)
    row[prefix + "DIRECTION_VALID"] = valid
    if valid:
        az = (-(phi + np.pi - np.radians(.61)) if detector == "KM2A"
              else np.pi / 2 - np.radians(29.442) - phi)
        row[prefix + "AZ"] = float(np.degrees(az) % 360)
        row[prefix + "ALT"] = float(90 - np.degrees(theta))
    valid = bool(np.all(np.isfinite([x, y])) and x not in (-10000, -9999) and y not in (-10000, -9999))
    row[prefix + "CORE_VALID"] = valid
    if valid:
        if detector == "KM2A":
            # Both upstream joint-reconstruction versions use this map; the
            # 0.61 degree calibration belongs only to their sky-direction path.
            east, north = -y, x
        else:
            c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
            east, north = c * x - s * y - 7, s * x + c * y + 7.97
        row[prefix + "CORE_EAST"], row[prefix + "CORE_NORTH"] = float(east), float(north)


class DL3ExternalSource:
    """Read already matched scalar records without changing the LACT result.

    Files are loaded once when supplied; both omitted requires no uproot.
    Explicit match_success=1 accepts an upstream selection even when Nmatch>1.
    Without that flag, multiplicity one is required. Duplicate selections remain
    AMBIGUOUS and never supply authoritative physical coordinates. Selected
    records require exact LACT event_id and integer Rabbit identity agreement.
    """

    def __init__(self, km2a_file=None, wcda_file=None):
        self._tables, self._sources = {}, {}
        for detector, path in (("KM2A", km2a_file), ("WCDA", wcda_file)):
            if path is None:
                self._sources[detector] = {"enabled": False, "file": None}
                continue
            path = str(path)
            if "://" not in path:
                path = str(Path(path).resolve())
            data, source = _read_table(path, detector)
            self._tables[detector] = data
            self._sources[detector] = {"enabled": True, **source}

    @property
    def provenance(self):
        """JSON-compatible input/coordinate conventions, including unknowns."""
        return {
            "sources": self._sources,
            "matching": "Already matched input; no coincidence search. Explicit match_success takes precedence over counts; duplicate selections are AMBIGUOUS.",
            "identity": "event is LACT event_id; RabbitTime/Rabbittime are LACT Unix TAI seconds/8ns ticks, not external detector time. Selected rows require exact agreement.",
            "external_times": "Event/mjd, Rec/mjd, WCDA mjd and mjdns4 are preserved RAW with unknown time scale; not used for sky-coordinate conversion.",
            "raw_source_quality": "RAW does not imply valid: invalid KM2A MJD/IDs were verified in the development sample (event 225185). Only the checked LACT identity fields are authoritative association keys.",
            "delta_t": "RAW only; producer unit, sign and approximately 1e7 baseline/offset are unverified.",
            "external_energy": "RAW only; units, logarithm conventions and calibration unverified.",
            "coordinates": {
                "reference": "Upstream joint-reconstruction conventions; same as pylast.visualize.real_data and lhaaso_background. Array-map positions and calibrated sky azimuth are separate transforms.",
                "ground_frame": "LHAASO_ARRAY_MAP",
                "ground_frame_note": "Common local array map, not an independently surveyed geographic ENU frame. No 0.61 degree rotation of positions.",
                "KM2A": {"phi_offset_deg": .61, "azimuth_rad": "-(phi + pi - radians(phi_offset_deg)) modulo 2pi",
                         "upstream_phi_deg": "degrees(phi) - 0.61 + 270",
                         "core_east_m": "-y", "core_north_m": "x",
                         "reference": "Rec_KM2A_v4.1.7/LHAASOFit.cc:258-289; LHAASORec_AllDetector-v9.4.6/main.cc:271-289"},
                "WCDA": {"azimuth_offset_deg": 29.442, "azimuth_rad": "pi/2 - radians(azimuth_offset_deg) - phi modulo 2pi",
                         "core_rotation_deg": 30., "core_origin_east_m": -7., "core_origin_north_m": 7.97,
                         "core_east_m": "cos(rotation)*x - sin(rotation)*y - 7", "core_north_m": "sin(rotation)*x + cos(rotation)*y + 7.97"},
                "altitude_deg": "90 - degrees(theta); theta must be finite in [0, pi], phi finite",
                "invalid_core_sentinels": [-10000, -9999],
            },
        }

    def rows_for_event(self, event):
        """Return a fresh full flat row; raise on IO/schema or identity errors.

        Unmatched padding rows with missing/sentinel Rabbit are retained as
        UNMATCHED. Any legitimate Rabbit in a queried row is checked; selected
        rows cannot bypass verification by supplying a missing timestamp.
        """
        row = default_external_row()
        event_id = int(event.event_id)
        if event_id != event.event_id:
            raise ValueError("LACT event_id must be an integer")
        expected = _rabbit(getattr(event, "rabbit_time", None), getattr(event, "rabbit_tick", None))
        for detector, data in self._tables.items():
            prefix = detector + "_"
            indices = np.flatnonzero(data["event"] == event_id)
            row[prefix + "N_RECORDS"] = int(indices.size)
            row[prefix + "MATCH_STATE"] = "NO_RECORD"
            if not indices.size:
                continue
            states = [_row_state(data, i, detector) for i in indices]
            selected = [int(i) for i, state in zip(indices, states) if state == "MATCHED"]
            state = ("AMBIGUOUS" if len(selected) > 1 or "AMBIGUOUS" in states
                     else "MATCHED" if selected else "UNMATCHED")
            row[prefix + "MATCH_STATE"] = state
            row[prefix + "MULTI_CANDIDATE"] = bool(len(indices) > 1 or any(
                np.any(data[key][indices] > 1)
                for key in ("NMATCH_RAW", "MKM2A_RAW", "MWCDA_RAW") if key in data))
            verified = []
            for index in indices:
                actual = _rabbit(data["LACT_RABBIT_TIME"][index] if "LACT_RABBIT_TIME" in data else None,
                                 data["LACT_RABBIT_TICK"][index] if "LACT_RABBIT_TICK" in data else None)
                if actual is not None and (expected is None or actual != expected):
                    raise ValueError(f"Wrong {detector} sidecar for event_id={event_id}: LACT Rabbit {actual} != {expected}")
                if int(index) in selected and (actual is None or expected is None):
                    raise ValueError(f"Cannot verify {detector} LACT Rabbit identity for selected event_id={event_id}")
                verified.append(actual is not None and actual == expected)
            # Multiple unresolved rows have no representative scalar record.
            index = selected[0] if state == "MATCHED" else int(indices[0]) if len(indices) == 1 else None
            row[prefix + "IDENTITY_VERIFIED"] = (bool(verified[list(indices).index(index)])
                                                  if index is not None else bool(all(verified)))
            if index is not None:
                row[prefix + "SOURCE_ENTRY"] = index
                for name, values in data.items():
                    if name != "event":
                        dtype = EXTERNAL_DTYPES[prefix + name]
                        value = values[index]
                        row[prefix + name] = (int(value) if np.isfinite(value) and int(value) == value else -1
                                              ) if dtype == "int64" else float(value)
            if state == "MATCHED":
                _physical_coordinates(row, detector)
        return row
