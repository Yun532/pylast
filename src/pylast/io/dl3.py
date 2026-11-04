"""Export reconstructed events as the event-list component of GADF DL3.

This module never reconstructs events, invents good-time intervals, or creates
instrument responses. Missing observation metadata require ``preview=True``.
Astropy and (for ROOT output) uproot are imported only when exporting.
"""

from pathlib import Path
import json
import math
import uuid

import numpy as np

__all__ = ["event_to_dl3_row", "write_dl3_events"]

EVENT_UNITS = {
    "TIME": "s", "TIME_MJD": "d", "RABBIT_TIME": "s", "ENERGY": "TeV", "RA": "deg", "DEC": "deg",
    "ALT": "deg", "AZ": "deg", "POINTING_ALT": "deg", "POINTING_AZ": "deg", "THETA": "deg",
    "ANGSEP_LACT_KM2A": "deg", "ANGSEP_LACT_WCDA": "deg", "ANGSEP_KM2A_WCDA": "deg",
    "CORESEP_KM2A_WCDA": "m",
}
EVENT_DESCRIPTIONS = {
    "OBS_ID": "Observation identifier assigned by the observing/processing manifest.",
    "EVENT_ID": "LACT event identifier, unique within OBS_ID.",
    "RUN_ID": "Original native LACT run identifier; may be zero and is not OBS_ID.",
    "TIME": "LACT event TT seconds relative to MJDREFI + MJDREFF; not barycentric.",
    "TIME_MJD": "Original LACT UTC MJD; auxiliary double precision timestamp.",
    "RABBIT_TIME": "Original LACT Unix TAI integer seconds.",
    "RABBIT_TICK": "Original LACT subsecond integer ticks of 8 ns.",
    "IS_LED": "LED/calibration event flag: 0 false, 1 true, -1 unknown.",
    "RA": "LACT reconstructed ICRS right ascension, using the LACT event time.",
    "DEC": "LACT reconstructed ICRS declination, using the LACT event time.",
    "ALT": "LACT reconstructed altitude above the geometric horizon; no added refraction.",
    "AZ": "LACT reconstructed azimuth, north through east in [0, 360).",
    "ENERGY": "LACT reconstructed energy; native TeV. No external detector estimate is substituted.",
    "POINTING_ALT": "Array pointing altitude interpolated for this LACT event.",
    "POINTING_AZ": "Array pointing azimuth, north through east, for this LACT event.",
    "THETA": "Spherical separation between the LACT reconstructed direction and array pointing.",
    "GEOMETRY_VALID": "Native geometry validity; distinct from final event acceptance.",
    "ENERGY_VALID": "Native energy validity; distinct from final event acceptance.",
    "RECONSTRUCTION_VALID": "Prediction reconstruction status: 0 failed, 1 passed, -1 unknown.",
    "IMAGE_CUT_PASS": "Prediction image-selection status: 0 failed, 1 passed, -1 unknown.",
    "QUALITY_PASS": "Prediction quality-cut status: 0 failed, 1 passed, -1 unknown.",
    "PID_PASS": "Prediction particle-identification status: 0 failed, 1 passed, -1 unknown.",
    "ACCEPTED": "Original LACT model selection; external coincidence does not change this flag.",
    "GAMMANESS": "LACT gamma-likeness score; larger means more gamma-like, not a truth label.",
    "PID_THRESHOLD": "Gamma-score threshold applied by the LACT prediction recipe.",
    "QUALITY_PROXY": "Model prediction of log10(max(angular_error_deg, 1e-4)); not an event error or PSF.",
    "QUALITY_THRESHOLD": "Applied threshold in the same predicted log10-error convention as QUALITY_PROXY.",
    "DIRECTION_METHOD": "Native LACT direction reconstruction entry name.",
    "ENERGY_METHOD": "Native LACT energy reconstruction entry name.",
    "MODEL_GROUP": "Prediction model/threshold domain; e.g. 36tel_z20 is not an event telescope count.",
    "MODEL_BRANCH": "Prediction selection branch; fixed does not mean array exact-one.",
    "N_TELS_DIRECTION": "Number of telescope IDs listed by the direction result; -1 if unavailable.",
    "N_TELS_ENERGY": "Number of telescope IDs listed by the energy result; -1 if unavailable.",
    "KM2A_USED_IN_RECO": "Whether KM2A was used by this LACT reconstruction; false for this image-only exporter.",
    "WCDA_USED_IN_RECO": "Whether WCDA was used by this LACT reconstruction; false for this image-only exporter.",
    "ANGSEP_LACT_KM2A": "LACT-KM2A direction separation; not a truth angular error.",
    "ANGSEP_LACT_WCDA": "LACT-WCDA direction separation; not a truth angular error.",
    "ANGSEP_KM2A_WCDA": "KM2A-WCDA direction separation; not a truth angular error.",
    "CORESEP_KM2A_WCDA": "KM2A-WCDA horizontal core separation in LHAASO_ARRAY_MAP; not a surveyed geographic-ENU comparison.",
}


def _separation(alt1, az1, alt2, az2):
    """Spherical separation in degrees; NaNs remain NaNs."""
    a1, z1, a2, z2 = np.deg2rad([alt1, az1, alt2, az2])
    hav = np.sin((a1 - a2) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin((z1 - z2) / 2) ** 2
    return np.rad2deg(2 * np.arcsin(np.sqrt(np.clip(hav, 0, 1))))


def _write_root_table(root, name, table):
    data, types = {}, {}
    for column in table.colnames:
        values = np.asarray(table[column])
        if values.dtype.kind in "SU":
            data[column], types[column] = values.astype(str), "string"
        elif values.dtype.kind in "biuf":
            data[column], types[column] = values, values.dtype
        else:
            raise TypeError(f"{name}.{column} must be a numeric, boolean or text scalar column")
    root.mktree(name, types)
    if len(table):
        root[name].extend(data)


def event_to_dl3_row(event, prediction, *, obs_id, geometry_name, energy_name):
    """Copy one event and one prediction mapping into a scalar event-list row.

    Pass ``result.iloc[0]`` from a mono prediction. ``accepted`` must be supplied
    explicitly: native DL2 validity is not necessarily the final PID selection.
    Native geometry is radians, native energy is TeV, and Rabbit is Unix TAI
    seconds plus 8 ns ticks. No pixels are copied or moved out of the event.
    """
    if int(prediction["event_id"]) != int(event.event_id):
        raise ValueError("Prediction event_id does not match the event")
    if "accepted" not in prediction:
        raise ValueError("Prediction must explicitly provide accepted")
    if prediction["accepted"] not in (True, False):
        raise ValueError("accepted must be an explicit boolean selection")
    if event.dl2 is None:
        raise ValueError("Run reconstruction before exporting DL3")
    geometry = event.dl2.geometry[geometry_name]
    energy = event.dl2.energy[energy_name]

    def flag(key):
        value = prediction.get(key)
        if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
            return -1
        if value not in (-1, 0, 1):
            raise ValueError(f"{key} must be boolean, or -1/NaN when unknown")
        return int(value)

    estimate = float(energy.estimate_energy)
    # Unreconstructed native energies default to zero; zero is not a measurement.
    if not np.isfinite(estimate) or estimate <= 0 or flag("reconstruction_valid") == 0:
        estimate = math.nan
    led = getattr(event, "is_led", None)
    row = {
        "OBS_ID": int(obs_id), "EVENT_ID": int(event.event_id), "RUN_ID": int(event.run_id),
        "TIME_MJD": float(event.time_mjd),
        "RABBIT_TIME": int(event.rabbit_time), "RABBIT_TICK": int(event.rabbit_tick),
        "IS_LED": -1 if led is None else int(led),
        "ALT": math.degrees(geometry.alt), "AZ": math.degrees(geometry.az) % 360,
        "ENERGY": estimate,
        "POINTING_ALT": math.degrees(event.pointing.array_altitude),
        "POINTING_AZ": math.degrees(event.pointing.array_azimuth) % 360,
        "GEOMETRY_VALID": bool(geometry.is_valid), "ENERGY_VALID": bool(energy.energy_valid),
        "RECONSTRUCTION_VALID": flag("reconstruction_valid"),
        "IMAGE_CUT_PASS": flag("image_cut_pass"), "QUALITY_PASS": flag("quality_pass"),
        "PID_PASS": flag("pass_pid"), "ACCEPTED": bool(prediction["accepted"]),
        "GAMMANESS": float(prediction.get("pid_gamma_score", math.nan)),
        "PID_THRESHOLD": float(prediction.get("pid_threshold_gamma_score", math.nan)),
        "QUALITY_PROXY": float(prediction.get("quality_proxy_log_error", math.nan)),
        "QUALITY_THRESHOLD": float(prediction.get("quality_threshold_smooth", math.nan)),
        "DIRECTION_METHOD": str(geometry_name), "ENERGY_METHOD": str(energy_name),
        "MODEL_GROUP": str(prediction.get("group", "")),
        "MODEL_BRANCH": str(prediction.get("branch", "")),
        "N_TELS_DIRECTION": len(geometry.telescopes) if hasattr(geometry, "telescopes") else -1,
        "N_TELS_ENERGY": len(energy.telescopes) if hasattr(energy, "telescopes") else -1,
        "KM2A_USED_IN_RECO": False, "WCDA_USED_IN_RECO": False,
    }
    if row["ACCEPTED"] and (not np.isfinite([row["ALT"], row["AZ"], estimate]).all()
                            or not -90 <= row["ALT"] <= 90):
        raise ValueError("Accepted event has no finite direction or positive energy")
    return row


def write_dl3_events(rows, filename, *, obs_id, location=None, reference_time=None,
                     gti=None, livetime=None, observation_mode=None,
                     accepted_only=True, preview=False, overwrite=False, provenance=None,
                     image_rows=None):
    """Write one observation's scalar events to FITS or a flat ROOT EVENTS tree.

    ``location`` is a scalar Astropy EarthLocation; azimuth is north through east.
    ``gti`` is an Astropy Time array of shape (N, 2), with true good-time starts
    and stops. ``livetime`` is dead-time-corrected seconds. Neither quantity is
    inferred from event timestamps. ``observation_mode`` is POINTING or DRIFT.

    In preview mode unknown sky coordinates remain NaN and unknown exposure
    keywords/GTI are omitted. DL3STAT/MISSING/IRFAVAIL identify the limitations.
    Even with observation metadata this writes only the event-list component,
    not an IRF or a complete flux-analysis product. Nonaccepted records may be
    retained with ``accepted_only=False`` for reconstruction diagnostics.

    ``image_rows`` optionally supplies one image-parameter row per
    (OBS_ID, EVENT_ID, TEL_ID); it is filtered with EVENTS. POINTING records all
    input event pointings. SCHEMA and PROVENANCE make both output formats
    self-describing. External sky coordinates use the common LACT event time,
    never an external MJD whose time scale has not been established.
    """
    from astropy import units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.io import fits
    from astropy.table import Table
    from astropy.time import Time
    from astropy.utils import iers
    from .dl3_external import EXTERNAL_UNITS, EXTERNAL_DESCRIPTIONS
    from .dl3_images import IMAGE_DTYPES, IMAGE_UNITS, IMAGE_DESCRIPTIONS

    path = Path(filename)
    if path.suffix.lower() not in (".fits", ".fit", ".root"):
        raise ValueError("Output must be .fits, .fit, or .root")
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    records = list(rows)
    if records and any(set(row) != set(records[0]) for row in records):
        raise ValueError("All event rows must have the same column schema")
    table = Table(rows=records)
    required = {"OBS_ID", "EVENT_ID", "TIME_MJD", "RABBIT_TIME", "RABBIT_TICK",
                "ALT", "AZ", "ENERGY", "ACCEPTED", "POINTING_ALT", "POINTING_AZ"}
    if not required.issubset(table.colnames) or len(table) == 0:
        raise ValueError(f"Nonempty event rows must contain {sorted(required)}")
    if np.any(np.asarray(table["OBS_ID"], dtype=np.int64) != int(obs_id)):
        raise ValueError("Write one OBS_ID per file")
    if len(np.unique(table["EVENT_ID"])) != len(table):
        raise ValueError("Duplicate EVENT_ID within the observation")
    input_ids = set(int(value) for value in table["EVENT_ID"])
    accepted = np.asarray(table["ACCEPTED"])
    if accepted.dtype.kind not in "biuf" or not np.all(np.isin(accepted, (0, 1))):
        raise ValueError("ACCEPTED must contain explicit boolean selections")
    table["ACCEPTED"] = accepted.astype(bool)
    for name in ("OBS_ID", "EVENT_ID", "RUN_ID"):
        if name in table.colnames:
            values = np.asarray(table[name])
            if values.dtype.kind not in "iu" and (values.dtype.kind != "f" or
                    np.any(~np.isfinite(values)) or np.any(values != values.astype(np.int64))):
                raise ValueError(f"{name} must contain integer identifiers")
            table[name] = values.astype(np.int64)
    for name in ("RABBIT_TIME", "RABBIT_TICK"):
        values = np.asarray(table[name])
        if np.any(~np.isfinite(values)) or np.any(values < 0) or np.any(values != values.astype(np.int64)):
            raise ValueError(f"Invalid {name}")
        table[name] = values.astype(np.int64)
    for name in ("IS_LED", "RECONSTRUCTION_VALID", "IMAGE_CUT_PASS", "QUALITY_PASS", "PID_PASS"):
        if name in table.colnames:
            table[name] = np.asarray(table[name], dtype=np.int8)

    missing = [name for name, value in (("LOCATION", location), ("GTI", gti),
                                      ("LIVETIME", livetime), ("OBS_MODE", observation_mode)) if value is None]
    if missing and not preview:
        raise ValueError(f"Missing {', '.join(missing)}; supply them or explicitly use preview=True")
    if observation_mode is not None and observation_mode not in ("POINTING", "DRIFT"):
        raise ValueError("observation_mode must be POINTING or DRIFT")
    if location is not None and (not isinstance(location, EarthLocation) or not location.isscalar
                                 or not all(np.isfinite(value.to_value(u.m)) for value in location.geocentric)):
        raise ValueError("location must be a finite scalar EarthLocation")

    times = Time(np.asarray(table["RABBIT_TIME"], dtype=float),
                 np.asarray(table["RABBIT_TICK"], dtype=float) * 8e-9,
                 format="unix_tai", scale="tai").tt
    if not np.allclose(times.utc.mjd, table["TIME_MJD"], rtol=0, atol=1e-10):
        raise ValueError("Rabbit time disagrees with UTC TIME_MJD; check the input clock convention")
    ref = (Time(int(np.floor(np.min(times.mjd))), format="mjd", scale="tt")
           if reference_time is None else Time(reference_time).tt)
    if not ref.isscalar:
        raise ValueError("reference_time must be scalar")
    ref_i = int(np.floor(ref.mjd))
    ref_f = float((ref - Time(ref_i, format="mjd", scale="tt")).to_value(u.day))
    ref = Time(ref_i, ref_f, format="mjd", scale="tt")
    table["TIME"] = (times - ref).to_value(u.s)
    table["THETA"] = _separation(table["ALT"], table["AZ"], table["POINTING_ALT"], table["POINTING_AZ"])
    n_input = len(table)
    pointing_times = times
    pointing_alt = np.asarray(table["POINTING_ALT"], dtype=float)
    pointing_az = np.asarray(table["POINTING_AZ"], dtype=float)
    if not np.isfinite([pointing_alt, pointing_az]).all() or np.any(np.abs(pointing_alt) > 90):
        raise ValueError("Pointing angles must be finite degrees with altitude within [-90, 90]")
    pointing_table = Table({"OBS_ID": table["OBS_ID"].copy(), "EVENT_ID": table["EVENT_ID"].copy(),
                            "TIME": table["TIME"].copy(), "ALT_PNT": pointing_alt.copy(),
                            "AZ_PNT": pointing_az.copy(), "RA_PNT": np.full(n_input, np.nan),
                            "DEC_PNT": np.full(n_input, np.nan)})
    images = None
    if image_rows is not None:
        image_records = list(image_rows)
        if image_records:
            if any(set(item) != set(image_records[0]) for item in image_records):
                raise ValueError("All image rows must have the same column schema")
            images = Table(rows=image_records)
        else:
            images = Table(names=list(IMAGE_DTYPES), dtype=list(IMAGE_DTYPES.values()))
        keys = ("OBS_ID", "EVENT_ID", "TEL_ID")
        if not set(keys).issubset(images.colnames):
            raise ValueError("Image rows require OBS_ID, EVENT_ID and TEL_ID")
        for key in keys:
            values = np.asarray(images[key])
            if values.dtype.kind not in "iu" and (values.dtype.kind != "f" or
                    np.any(~np.isfinite(values)) or np.any(values != values.astype(np.int64))):
                raise ValueError(f"Image {key} must contain integer identifiers")
        image_keys = [tuple(int(item[key]) for key in keys) for item in images]
        if len(set(image_keys)) != len(image_keys):
            raise ValueError("Duplicate image key (OBS_ID, EVENT_ID, TEL_ID)")
        if any(key[0] != int(obs_id) or key[1] not in input_ids or key[2] < 0 for key in image_keys):
            raise ValueError("Image key does not belong to the supplied observation/events")
        for key in keys:
            images[key] = np.asarray(images[key], dtype=np.int64)
    if accepted_only:
        select = np.asarray(table["ACCEPTED"], dtype=bool)
        table, times = table[select], times[select]
    order = np.argsort(table["TIME"], kind="stable")
    table, times = table[order], times[order]
    if images is not None:
        images = images[np.isin(images["EVENT_ID"], table["EVENT_ID"])]
        rank = {int(value): index for index, value in enumerate(table["EVENT_ID"])}
        images = images[np.lexsort((images["TEL_ID"], [rank[int(value)] for value in images["EVENT_ID"]]))]
    if np.any(np.asarray(table["ACCEPTED"], dtype=bool) &
              (~np.isfinite(table["ENERGY"]) | (table["ENERGY"] <= 0)
               | ~np.isfinite(table["ALT"]) | ~np.isfinite(table["AZ"]))):
        raise ValueError("Accepted rows must have finite direction and positive energy")
    table["RA"], table["DEC"] = np.full(len(table), np.nan), np.full(len(table), np.nan)
    finite = np.isfinite(table["ALT"]) & np.isfinite(table["AZ"])
    if np.any(finite & (np.abs(table["ALT"]) > 90)):
        raise ValueError("ALT must be within [-90, 90] degrees")
    if location is not None and np.any(finite):
        # Use bundled IERS data deterministically, without a network request.
        with iers.conf.set_temp("auto_download", False):
            sky = SkyCoord(alt=np.asarray(table["ALT"])[finite] * u.deg,
                           az=np.asarray(table["AZ"])[finite] * u.deg,
                           frame=AltAz(obstime=times[finite], location=location, pressure=0 * u.hPa)).icrs
        table["RA"][finite], table["DEC"][finite] = sky.ra.deg, sky.dec.deg
    for detector in ("KM2A", "WCDA"):
        alt_key, az_key = f"{detector}_ALT", f"{detector}_AZ"
        if alt_key not in table.colnames:
            continue
        alt, az = np.asarray(table[alt_key], dtype=float), np.asarray(table[az_key], dtype=float)
        valid = np.isfinite(alt) & np.isfinite(az)
        if np.any(valid & (np.abs(alt) > 90)):
            raise ValueError(f"{alt_key} must be within [-90, 90] degrees")
        ra_key, dec_key = f"{detector}_RA", f"{detector}_DEC"
        table[ra_key], table[dec_key] = np.full(len(table), np.nan), np.full(len(table), np.nan)
        if location is not None and np.any(valid):
            with iers.conf.set_temp("auto_download", False):
                sky = SkyCoord(alt=alt[valid] * u.deg, az=az[valid] * u.deg,
                               frame=AltAz(obstime=times[valid], location=location, pressure=0 * u.hPa)).icrs
            table[ra_key][valid], table[dec_key][valid] = sky.ra.deg, sky.dec.deg
        table[f"ANGSEP_LACT_{detector}"] = _separation(table["ALT"], table["AZ"], alt, az)
    if {"KM2A_ALT", "WCDA_ALT"}.issubset(table.colnames):
        table["ANGSEP_KM2A_WCDA"] = _separation(table["KM2A_ALT"], table["KM2A_AZ"], table["WCDA_ALT"], table["WCDA_AZ"])
        table["CORESEP_KM2A_WCDA"] = np.hypot(table["KM2A_CORE_EAST"] - table["WCDA_CORE_EAST"],
                                             table["KM2A_CORE_NORTH"] - table["WCDA_CORE_NORTH"])

    header = {
        "HDUCLASS": "GADF", "HDUDOC": "https://gamma-astro-data-formats.readthedocs.io/en/v0.3/",
        "HDUVERS": "0.3", "HDUCLAS1": "EVENTS", "OBS_ID": int(obs_id),
        "MJDREFI": ref_i, "MJDREFF": ref_f, "TIMEUNIT": "s", "TIMESYS": "TT",
        "MJDTSYS": "UTC", "RBTSYS": "TAI", "RBTTICK": 8e-9,
        "TIMEREF": "TOPOCENTER", "RADECSYS": "ICRS", "EQUINOX": 2000.,
        "TELESCOP": "LACT", "INSTRUME": "LACT", "CREATOR": "pylast.write_dl3_events",
        "DL3STAT": "EVENTS_ONLY", "IRFAVAIL": False,
        "COORDOK": location is not None, "MISSING": ",".join(missing + ["IRF"]),
        "NINPUT": n_input, "NSELECT": len(table),
        "EV_CLASS": "ACCEPTED" if accepted_only else "ALL_RECO",
        "PYLSCHEM": "1.0", "EXTTIME": "LACT_EVENT",
        "EXTUSED": any(np.any(table[key]) for key in ("KM2A_USED_IN_RECO", "WCDA_USED_IN_RECO") if key in table.colnames),
    }
    if any(f"{detector}_CORE_EAST" in table.colnames for detector in ("KM2A", "WCDA")):
        header["EXTGFRM"] = "LHAASO_ARRAY_MAP"
    if observation_mode is not None:
        header["OBS_MODE"] = observation_mode
    if location is not None:
        header.update(GEOLON=float(location.lon.deg), GEOLAT=float(location.lat.deg),
                      ALTITUDE=float(location.height.to_value(u.m)))
        site = (provenance or {}).get("site") or {}
        if "approximate" in site:
            header["SITEAPP"] = bool(site["approximate"])
        if site.get("name"):
            header["SITENAME"] = str(site["name"])
        if site.get("source"):
            header["SITESRC"] = str(site["source"])
    if len(table):
        header.update(EVTMIN=float(np.min(table["TIME"])), EVTMAX=float(np.max(table["TIME"])))
    # Summarize all input pointings, independently of event cuts. Convert each
    # pointing at its own timestamp before averaging tracking sky directions.
    header["ALT_PNT"] = float(np.mean(pointing_alt))
    az = np.deg2rad(pointing_az)
    header["AZ_PNT"] = float(np.degrees(np.arctan2(np.mean(np.sin(az)), np.mean(np.cos(az)))) % 360)
    header["PNTSRC"] = "EVENTS"
    if location is not None:
        with iers.conf.set_temp("auto_download", False):
            pnt = SkyCoord(alt=pointing_alt * u.deg, az=pointing_az * u.deg,
                           frame=AltAz(obstime=pointing_times, location=location, pressure=0 * u.hPa)).icrs
        x, y, z = np.mean(pnt.cartesian.xyz.value, axis=1)
        header.update(RA_PNT=float(np.degrees(np.arctan2(y, x)) % 360),
                      DEC_PNT=float(np.degrees(np.arctan2(z, np.hypot(x, y)))))
        pointing_table["RA_PNT"], pointing_table["DEC_PNT"] = pnt.ra.deg, pnt.dec.deg

    gti_table = None
    if gti is not None:
        intervals = Time(gti).tt
        if intervals.ndim != 2 or intervals.shape[1] != 2 or intervals.shape[0] == 0:
            raise ValueError("gti must be a nonempty Time array of shape (N, 2)")
        offsets = (intervals - ref).to_value(u.s)
        offsets = offsets[np.argsort(offsets[:, 0], kind="stable")]
        if (not np.isfinite(offsets).all() or np.any(offsets[:, 1] <= offsets[:, 0])
                or np.any(offsets[1:, 0] < offsets[:-1, 1])):
            raise ValueError("GTI intervals must be finite, ordered and nonoverlapping with start < stop")
        covered = np.zeros(len(table), dtype=bool)
        for start, stop in offsets:
            covered |= (table["TIME"] >= start) & (table["TIME"] <= stop)
        if not np.all(covered):
            raise ValueError("Output events fall outside the supplied GTI")
        ontime = float(np.sum(offsets[:, 1] - offsets[:, 0]))
        header.update(TSTART=float(offsets[0, 0]), TSTOP=float(offsets[-1, 1]), ONTIME=ontime)
        header.update({"DATE-OBS": (ref + offsets[0, 0] * u.s).tt.isot,
                       "DATE-END": (ref + offsets[-1, 1] * u.s).tt.isot})
        gti_table = Table({"START": offsets[:, 0] * u.s, "STOP": offsets[:, 1] * u.s})
    if livetime is not None:
        if gti is None:
            raise ValueError("livetime requires real GTI intervals")
        live = float(u.Quantity(livetime, u.s).to_value(u.s))
        if not np.isfinite(live) or not 0 <= live <= header["ONTIME"]:
            raise ValueError("livetime must be finite and between 0 and ONTIME")
        header.update(LIVETIME=live, DEADC=live / header["ONTIME"])

    # Mandatory event-list columns come first for convenient inspection.
    first = ["EVENT_ID", "TIME", "RA", "DEC", "ENERGY"]
    table = table[first + [name for name in table.colnames if name not in first]]
    table.meta = header
    common = {key: header[key] for key in ("OBS_ID", "MJDREFI", "MJDREFF", "TIMEUNIT", "TIMESYS", "TIMEREF")}
    common.update({key: header[key] for key in ("GEOLON", "GEOLAT", "ALTITUDE", "SITEAPP", "SITENAME", "SITESRC", "RADECSYS")
                   if key in header})
    pointing_table = pointing_table[np.argsort(pointing_table["TIME"], kind="stable")]
    pointing_table.meta = dict(common, HDUCLASS="GADF", HDUCLAS1="POINTING", HDUVERS="0.3", PNTSRC="EVENTS")
    tables = {"EVENTS": table, "POINTING": pointing_table}
    if images is not None:
        images.meta = dict(HDUCLASS="PYLAST", HDUCLAS1="IMAGE_PARAMETERS", PYLSCHEM="1.0", OBS_ID=int(obs_id))
        tables["IMAGE_PARAMETERS"] = images
    if gti_table is not None:
        gti_table.meta = dict(common, HDUCLASS="GADF", HDUCLAS1="GTI", HDUVERS="0.3")
        tables["GTI"] = gti_table

    event_units = dict(EVENT_UNITS, **EXTERNAL_UNITS)
    descriptions = dict(EVENT_DESCRIPTIONS, **EXTERNAL_DESCRIPTIONS)
    for detector in ("KM2A", "WCDA"):
        for axis in ("RA", "DEC"):
            key = f"{detector}_{axis}"
            event_units[key] = "deg"
            descriptions[key] = f"{detector} ICRS {axis} using the common LACT event time, not an unverified external MJD."
    units_by_table = {
        "EVENTS": event_units, "IMAGE_PARAMETERS": IMAGE_UNITS,
        "POINTING": {"TIME": "s", "ALT_PNT": "deg", "AZ_PNT": "deg", "RA_PNT": "deg", "DEC_PNT": "deg"},
        "GTI": {"START": "s", "STOP": "s"},
    }
    descriptions_by_table = {
        "EVENTS": descriptions, "IMAGE_PARAMETERS": IMAGE_DESCRIPTIONS,
        "POINTING": {"OBS_ID": EVENT_DESCRIPTIONS["OBS_ID"], "EVENT_ID": EVENT_DESCRIPTIONS["EVENT_ID"],
                     "TIME": "Input LACT event TT timestamp; this is a sampled event-pointing series, not a GTI.",
                     "ALT_PNT": EVENT_DESCRIPTIONS["POINTING_ALT"], "AZ_PNT": EVENT_DESCRIPTIONS["POINTING_AZ"],
                     "RA_PNT": "ICRS pointing RA at the input event time.",
                     "DEC_PNT": "ICRS pointing DEC at the input event time."},
        "GTI": {"START": "True good-time interval start in TT seconds relative to MJDREF.",
                "STOP": "True good-time interval stop in TT seconds relative to MJDREF."},
    }
    table_metadata, schema_rows = {}, []
    for name, output in tables.items():
        columns, units = {}, {}
        for column in output.colnames:
            # Astropy maps signed int8 to FITS logical (L). Three-state flags
            # need a real integer column so -1 remains unknown, never True.
            if output[column].dtype == np.dtype("int8"):
                output[column] = np.asarray(output[column], dtype=np.int16)
            unit = units_by_table[name].get(column, "")
            description = descriptions_by_table[name].get(column, "User-supplied scalar; see processing provenance.")
            if unit:
                output[column].unit = unit
                units[column] = unit
            output[column].description = description
            dtype = str(output[column].dtype)
            columns[column] = {"dtype": dtype, "unit": unit, "description": description}
            schema_rows.append((name, column, dtype, unit, description))
        table_metadata[name] = {"header": dict(output.meta), "units": units, "columns": columns}
    schema = Table(rows=schema_rows, names=["TABLE", "COLUMN", "DTYPE", "UNIT", "DESCRIPTION"])
    schema.meta = dict(HDUCLASS="PYLAST", HDUCLAS1="SCHEMA", PYLSCHEM="1.0")
    metadata = {"schema_version": "1.0", "header": header, "units": table_metadata["EVENTS"]["units"],
                "provenance": provenance or {}, "tables": table_metadata}
    metadata_json = json.dumps(metadata, sort_keys=True, allow_nan=False)
    provenance_table = Table({"JSON": [metadata_json]})
    provenance_table.meta = dict(HDUCLASS="PYLAST", HDUCLAS1="PROVENANCE", PYLSCHEM="1.0")
    tables.update(SCHEMA=schema, PROVENANCE=provenance_table)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        if path.suffix.lower() == ".root":
            import uproot
            with uproot.recreate(temporary) as root:
                for name, output in tables.items():
                    _write_root_table(root, name, output)
                root["metadata"] = metadata_json
        else:
            primary = fits.PrimaryHDU()
            primary.header.update(CREATOR=header["CREATOR"], PYLSCHEM="1.0", DL3STAT="EVENTS_ONLY")
            hdus = [primary] + [fits.BinTableHDU(output, name=name) for name, output in tables.items()]
            fits.HDUList(hdus).writeto(temporary, checksum=True)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path
