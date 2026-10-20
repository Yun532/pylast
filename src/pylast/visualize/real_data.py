"""Matched KM2A/WCDA records and overlays for the existing event visualizer.

Inputs are already matched sidecar ROOT files: ``event`` is the LACT event ID.
This module does not perform a time coincidence search or recalibrate data.
Ground coordinates are East/North; camera projections use pylast's normal
camera convention through EventVisualizer, including its display axis order.
"""

from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .lhaaso_background import (
    DEFAULT_KM2A_ED_POS_FILE, DEFAULT_KM2A_MD_POS_FILE,
    draw_lhaaso_background, wcda_xy_to_en,
)

__all__ = [
    "load_km2a_event_record", "load_wcda_event_record", "list_matched_event_ids",
    "km2a_xy_to_en", "km2a_theta_phi_to_altaz", "wcda_theta_phi_to_altaz",
    "draw_external_camera_overlays", "draw_external_event_cores",
    "plot_km2a_event", "plot_wcda_event",
]

_COLORS = {"KM2A": "#2563eb", "WCDA": "#7c3aed"}
_KM2A_BRANCHES = {
    "event_id": "event", "rec_x": "Rec/rec_x", "rec_y": "Rec/rec_y",
    "rec_theta": "Rec/rec_theta", "rec_phi": "Rec/rec_phi",
    "rabbit_time": "RabbitTime", "rabbit_tick": "Rabbittime",
    "gammap": "gammap", "rmue": "rmue", "Erec_km2a": "Erec_km2a",
    "Erec_eu": "Erec_eu", "NpE1": "Rec/NpE1", "NuM4": "Rec/NuM4",
    "mkm2a": "mkm2a", "deltaTkm2a": "deltaTkm2a",
}
_WCDA_BRANCHES = {
    key: key for key in (
        "event", "match_success", "theta", "phi", "mjd", "xc", "yc", "nhit",
        "gammap", "Erec_wcda", "Erec", "energy", "logE", "pincness", "f5w",
        "mwcda", "Nmatch", "deltaTwcda",
    )
}
_WCDA_BRANCHES["event_id"] = _WCDA_BRANCHES.pop("event")
_WCDA_BRANCHES.update(rabbit_time="RabbitTime", rabbit_tick="Rabbittime")


def _open_tree(root, detector):
    required = ({"event", "Rec/rec_x", "Rec/rec_y"} if detector == "KM2A"
                else {"event", "theta", "phi", "mjd", "xc", "yc"})
    for name, classname in root.classnames(recursive=True).items():
        if "TTree" not in classname and "RNTuple" not in classname:
            continue
        tree = root[name]
        keys = set(tree.keys())
        if required.issubset(keys) and (detector != "WCDA" or keys.intersection({"match_success", "Nmatch", "mwcda"})):
            return tree
    raise KeyError(f"No {detector} matched tree with branches {sorted(required)}")


@lru_cache(maxsize=4)
def _load_table(path, detector):
    # Cache only scalar reconstruction branches, never complete hit/waveform arrays.
    import uproot
    branches = _KM2A_BRANCHES if detector == "KM2A" else _WCDA_BRANCHES
    with uproot.open(path) as root:
        tree = _open_tree(root, detector)
        data = {key: tree[branch].array(library="np")
                for key, branch in branches.items() if branch in tree.keys()}
    return data


def km2a_xy_to_en(x, y, phi_offset_deg=0.61):
    """Convert KM2A native coordinates to geographic East/North (metres)."""
    offset = np.radians(phi_offset_deg)
    x, y = np.asarray(x), np.asarray(y)
    return np.sin(offset) * x - np.cos(offset) * y, np.cos(offset) * x + np.sin(offset) * y


def km2a_theta_phi_to_altaz(theta, phi, phi_offset_deg=0.61):
    return (-(np.asarray(phi) + np.pi - np.radians(phi_offset_deg))) % (2 * np.pi), np.pi / 2 - np.asarray(theta)


def wcda_theta_phi_to_altaz(theta, phi, wcdaphi_deg=29.442):
    return (np.pi / 2 - np.radians(wcdaphi_deg) - np.asarray(phi)) % (2 * np.pi), np.pi / 2 - np.asarray(theta)


def _matched_mask(data, detector):
    flags = ("match_success", "Nmatch", "mwcda") if detector == "WCDA" else ("mkm2a",)
    for flag in flags:
        if flag in data:
            # Multiplicity 2 is ambiguous, not a successful unique match.
            return np.isclose(data[flag], 1, rtol=0, atol=1e-12)
    return np.ones(len(data["event_id"]), dtype=bool)


def _load_record(path, event_id, detector):
    path = str(Path(path).resolve())
    data = _load_table(path, detector)
    matches = np.asarray(data["event_id"]) == int(event_id)
    matches &= _matched_mask(data, detector)
    indices = np.flatnonzero(matches)
    if not indices.size:
        return None
    if indices.size > 1:
        raise ValueError(f"Ambiguous {detector} event_id={event_id}: {indices.size} matched entries in {path}")
    index = int(indices[0])
    record = {key: values[index].item() for key, values in data.items()}
    record.update(detector=detector, event_id=int(event_id), source_path=path, entry_index=index, match_success=1)
    if detector == "KM2A":
        x, y = record["rec_x"], record["rec_y"]
        theta, phi = record.get("rec_theta", np.nan), record.get("rec_phi", np.nan)
        east, north = km2a_xy_to_en(x, y)
        az, alt = km2a_theta_phi_to_altaz(theta, phi)
        if "rabbit_time" in record and "rabbit_tick" in record:
            record["mjd"] = 40587 + (record["rabbit_time"] + record["rabbit_tick"] * 8e-9 - 37) / 86400
    else:
        x, y = record["xc"], record["yc"]
        theta, phi = record["theta"], record["phi"]
        east, north = wcda_xy_to_en(x, y)
        az, alt = wcda_theta_phi_to_altaz(theta, phi)
    # A time match does not guarantee a successful detector reconstruction.
    # The supplied KM2A failure row uses -10000 cores/-10 angles; unmatched
    # WCDA rows use -9999. Keep the source values, omit these display positions.
    core_valid = bool(np.all(np.isfinite([x, y])) and x not in (-10000, -9999) and y not in (-10000, -9999))
    direction_valid = bool(np.all(np.isfinite([theta, phi])) and 0 <= theta <= np.pi)
    if not core_valid:
        east = north = np.nan
    if not direction_valid:
        az = alt = np.nan
    record.update(core_east_m=float(east), core_north_m=float(north),
                  azimuth_rad=float(az), altitude_rad=float(alt),
                  core_valid=core_valid, direction_valid=direction_valid)
    return record


def load_km2a_event_record(path, event_id):
    """Read one already matched KM2A reconstruction, or None if absent."""
    return _load_record(path, event_id, "KM2A")


def load_wcda_event_record(path, event_id):
    """Read one uniquely matched WCDA reconstruction, or None.

    Trusts an explicit match_success flag (including a resolved multiple match).
    Without that flag, requires Nmatch or legacy mwcda to have multiplicity one.
    """
    return _load_record(path, event_id, "WCDA")


def list_matched_event_ids(*, km2a_root_file=None, wcda_root_file=None):
    """List matched LACT event IDs by detector and their intersection."""
    result = {"KM2A": [], "WCDA": []}
    for detector, path in (("KM2A", km2a_root_file), ("WCDA", wcda_root_file)):
        if path is None:
            continue
        data = _load_table(str(Path(path).resolve()), detector)
        ids = data["event_id"][_matched_mask(data, detector)]
        result[detector] = sorted({int(value) for value in ids if np.isfinite(value)})
    result["both"] = sorted(set(result["KM2A"]) & set(result["WCDA"]))
    return result


def _records(records, event_id=None, event=None):
    for detector, record in records.items():
        detector = str(detector).upper()
        if detector not in _COLORS:
            raise ValueError("external_reconstructions keys must be KM2A or WCDA")
        if record is None:
            continue
        if event_id is not None and int(record["event_id"]) != int(event_id):
            raise ValueError(f"{detector} record is for event {record['event_id']}, expected {event_id}")
        if (getattr(event, "rabbit_time", 0) > 0
                and "rabbit_time" in record and "rabbit_tick" in record):
            expected = (event.rabbit_time, getattr(event, "rabbit_tick", 0))
            actual = (record["rabbit_time"], record["rabbit_tick"])
            if actual != expected:
                raise ValueError(
                    f"Wrong {detector} sidecar for event_id={record['event_id']}: "
                    f"LACT Rabbit timestamp {actual} does not match event {expected}"
                )
        yield detector, record


def _record_text(detector, record):
    parts = [detector]
    for key, label in (("gammap", "gammap"), ("nhit", "nhit")):
        value = record.get(key, np.nan)
        if key == "gammap" and value is not None and value < 0:
            continue
        if value is not None and np.isfinite(value):
            parts.append(f"{label}={value:.3g}")
    if not np.all(np.isfinite([record["azimuth_rad"], record["altitude_rad"]])):
        parts.append("invalid direction")
    return "  ".join(parts)


def draw_external_camera_overlays(visualizer, ax, event, tel_id, external_reconstructions):
    """Add matched directions/SDPs using the current camera projector and style."""
    from .event_visualizer import _camera_to_plot_xy, incident_point_on_camera
    pointing = getattr(event, "pointing", None)
    alt = getattr(pointing, "array_altitude", np.nan)
    az = getattr(pointing, "array_azimuth", np.nan)
    drawn = {}
    for detector, record in list(_records(external_reconstructions, getattr(event, "event_id", None), event=event)):
        drawn[detector] = False
        source_az, source_alt = record["azimuth_rad"], record["altitude_rad"]
        color = _COLORS[detector]
        if not np.all(np.isfinite([source_az, source_alt])):
            ax.plot([], [], color=color, linestyle="None", label=f"{detector} direction unavailable (invalid reconstruction)")
            continue
        if not np.all(np.isfinite([alt, az, source_az, source_alt])):
            continue
        _, _, camera_x, camera_y = incident_point_on_camera(
            source_az, np.pi / 2 - source_alt, az, np.pi / 2 - alt,
            focal_length=visualizer.tel_geoms[tel_id].focal_length,
        )
        x, y = _camera_to_plot_xy(camera_x, camera_y)
        limits = ax.get_xlim(), ax.get_ylim()
        ax.plot(x, y, marker="+", linestyle="None", color=color, ms=12, mew=2,
                label=f"{detector} direction", zorder=10004)
        visualizer._draw_sdp_line(
            ax, event, tel_id, source_az, source_alt,
            (record["core_east_m"], record["core_north_m"]),
            color=color, line_style="--" if detector == "KM2A" else ":",
            line_width=1.8, label=f"{detector} SDP",
        )
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        drawn[detector] = True
    return drawn


def draw_external_event_cores(ax, external_reconstructions, *, event_id=None, event=None, show_direction=True):
    """Add matched geographic core positions and arrival directions to an array view."""
    from .event_visualizer import _add_arrival_arrow
    points = {}
    for detector, record in list(_records(external_reconstructions, event_id, event=event)):
        x, y = record["core_east_m"], record["core_north_m"]
        if not np.all(np.isfinite([x, y])):
            ax.plot([], [], color=_COLORS[detector], linestyle="None", label=f"{detector} core unavailable")
            continue
        points[detector] = (x, y)
        color = _COLORS[detector]
        ax.scatter([x], [y], marker="+" if detector == "KM2A" else "x", s=160,
                   linewidths=2.2, color=color, zorder=10, label=_record_text(detector, record))
        if show_direction and np.isfinite(record["azimuth_rad"]):
            span = max(np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), 100)
            _add_arrival_arrow(ax, x, y, np.degrees(record["azimuth_rad"]), span,
                               color=color, label=f"{detector} direction", mode="incoming", line_style="--")
    if len(points) == 2:
        km2a, wcda = points["KM2A"], points["WCDA"]
        distance = np.hypot(km2a[0] - wcda[0], km2a[1] - wcda[1])
        ax.plot([km2a[0], wcda[0]], [km2a[1], wcda[1]], color="0.3", lw=1,
                label=f"Core separation {distance:.1f} m")
    return points


def _finish(fig, output_path, show):
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_wcda_event(path, event_id, *, output_path=None, show=True, ed_pos_file=None, md_pos_file=None):
    """Show a matched WCDA direction/core on the existing LHAASO array map.

    The matched WCDA files contain reconstruction scalars, not cell hit maps.
    """
    record = load_wcda_event_record(path, event_id)
    if record is None:
        raise ValueError(f"No matched WCDA event_id={event_id}")
    fig, ax = plt.subplots(figsize=(7.6, 7))
    draw_lhaaso_background(ax, ed_pos_file=ed_pos_file, md_pos_file=md_pos_file, set_limits=True)
    draw_external_event_cores(ax, {"WCDA": record}, event_id=event_id)
    ax.set(xlabel="East (m)", ylabel="North (m)", title=f"WCDA event_id={event_id}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25, linewidth=0.55)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    _finish(fig, output_path, show)
    return fig, ax


@lru_cache(maxsize=4)
def _detector_positions(path):
    rows = np.loadtxt(path, skiprows=1, usecols=(0, 1, 2, 3))
    return {int(row[0]): (row[1], row[2]) for row in rows if row[3] != 0}


def _hit_branch(tree, group, field):
    import re
    keys = list(tree.keys())
    preferred = [f"Event/{group}.{field}", f"Event/{group}/{field}", f"Event.{group}.{field}",
                 f"{group}.{field}", f"{group}/{field}", f"Event/{group}/LHHit.{field}"]
    for candidate in preferred:
        if candidate in keys:
            return candidate
    matches = [key for key in keys if group.lower() in key.lower()
               and field.lower() in re.split(r"[^a-z0-9]+", key.lower())]
    return min(matches, key=len) if matches else None


def _read_hits(tree, group, index, positions):
    import awkward as ak
    columns = {}
    for field in ("id", "pe", "status"):
        branch = _hit_branch(tree, group, field)
        if branch is None:
            if field == "status":
                continue
            raise KeyError(f"{group} {field} hit branch is unavailable")
        value = tree[branch].array(entry_start=index, entry_stop=index + 1, library="ak")
        columns[field] = np.asarray(ak.flatten(value, axis=None))
    ids, charge = columns["id"], columns["pe"]
    status = columns.get("status", np.full(ids.size, 5))
    if not (ids.size == charge.size == status.size):
        raise ValueError(f"Inconsistent {group} hit branch lengths")
    return np.asarray([(*positions[int(detector_id)], float(pe), int(detector_id))
                       for detector_id, pe, state in zip(ids, charge, status)
                       if int(detector_id) in positions and state >= 2 and pe >= 1], dtype=float).reshape(-1, 4)


def plot_km2a_event(path, event_id, *, output_path=None, show=True, ed_pos_file=None, md_pos_file=None,
                    event=None, visualizer=None):
    """Draw ED/MD hits on the same full array background as the core view.

    Pass the processed LACT event and its visualizer to include the actual
    telescope positions and pointing. Hit selection and reconstruction stay
    those of the supplied matched KM2A file.
    """
    import uproot
    from matplotlib.colors import LogNorm
    from .event_visualizer import (
        _add_array_compass, _add_telescope_direction_inset, _event_pointing_azimuth_deg,
    )
    record = load_km2a_event_record(path, event_id)
    if record is None:
        raise ValueError(f"No matched KM2A event_id={event_id}")
    list(_records({"KM2A": record}, event_id, event=event))
    if event is not None and int(event.event_id) != int(event_id):
        raise ValueError(f"LACT event is for event {event.event_id}, expected {event_id}")
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 7.7))
    try:
        with uproot.open(path) as root:
            tree = _open_tree(root, "KM2A")
            for ax, group, detector_file in zip(axes, ("HitsE", "HitsM"),
                    (ed_pos_file or DEFAULT_KM2A_ED_POS_FILE, md_pos_file or DEFAULT_KM2A_MD_POS_FILE)):
                positions = _detector_positions(str(detector_file))
                background = draw_lhaaso_background(
                    ax, ed_pos_file=ed_pos_file, md_pos_file=md_pos_file, set_limits=True,
                )
                if visualizer is not None:
                    for tel_id, geom in visualizer.tel_geoms.items():
                        ax.scatter([geom.pos_x], [geom.pos_y], s=116, facecolors="none",
                                   edgecolors="#d73027", linewidths=1.8, zorder=6,
                                   label="LACT telescope" if tel_id == min(visualizer.tel_geoms) else None)
                        ax.annotate(f"T{tel_id + 1}", (geom.pos_x, geom.pos_y), xytext=(3, 3),
                                    textcoords="offset points", fontsize=7.2, zorder=7)
                    # Match the existing core-view limits, including its telescope-based padding.
                    east = [geom.pos_x for geom in visualizer.tel_geoms.values()]
                    north = [geom.pos_y for geom in visualizer.tel_geoms.values()]
                    pad = 0.18 * max(float(np.ptp(east)), float(np.ptp(north)), 1.0)
                    east = np.concatenate([east, background["ed"][:, 0], background["md"][:, 0]])
                    north = np.concatenate([north, background["ed"][:, 1], background["md"][:, 1]])
                    ax.set_xlim(float(east.min()) - pad, float(east.max()) + pad)
                    ax.set_ylim(float(north.min()) - pad, float(north.max()) + pad)
                hits = _read_hits(tree, group, record["entry_index"], positions)
                panel = "ED" if group == "HitsE" else "MD"
                if hits.size:
                    points = ax.scatter(hits[:, 0], hits[:, 1], c=hits[:, 2], s=42, cmap="inferno",
                                        marker="s" if group == "HitsE" else "o", zorder=5,
                                        edgecolors="0.15", linewidths=0.35,
                                        norm=LogNorm(vmin=1, vmax=max(float(hits[:, 2].max()), 1.01)))
                    fig.colorbar(points, ax=ax, fraction=0.046, pad=0.04, label=f"{panel} charge [p.e.]")
                    maximum = hits[int(hits[:, 2].argmax())]
                    ax.text(0.98, 0.02, f"Max: {maximum[2]:.2f} p.e. (ID {int(maximum[3])})",
                            transform=ax.transAxes, ha="right", fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="0.55", alpha=0.94))
                else:
                    ax.text(0.5, 0.05, "No selected hits", transform=ax.transAxes, ha="center", fontsize=8)
                draw_external_event_cores(ax, {"KM2A": record}, event_id=event_id, event=event)
                if not record["direction_valid"]:
                    ax.text(0.02, 0.98, "KM2A: invalid reconstructed direction", transform=ax.transAxes,
                            va="top", fontsize=8, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
                ax.set(title=f"KM2A {panel}   Nhit={len(hits)}   Sum={hits[:, 2].sum():.2f} p.e.",
                       xlabel="East (m)", ylabel="North (m)")
                ax.set_aspect("equal", adjustable="box")
                ax.grid(True, alpha=0.25, linewidth=0.55)
                ax.tick_params(direction="in", top=True, right=True)
                x_min, x_max = ax.get_xlim()
                y_min, y_max = ax.get_ylim()
                length = 0.075 * min(x_max - x_min, y_max - y_min)
                _add_array_compass(ax, x_min + 0.07 * (x_max - x_min), y_min + 0.085 * (y_max - y_min), length)
                azimuth = _event_pointing_azimuth_deg(event, source=getattr(visualizer, "source", None))
                if azimuth is not None:
                    _add_telescope_direction_inset(ax, x_min + 0.075 * (x_max - x_min),
                                                  y_max - 0.105 * (y_max - y_min), length, azimuth)
                ax.legend(loc="upper right", fontsize=8)
        summary = [f"KM2A event_id={event_id}"]
        if record["direction_valid"]:
            summary.append(f"Zenith={np.degrees(record['rec_theta']):.2f} deg")
            summary.append(f"Azimuth={np.degrees(record['azimuth_rad']):.2f} deg")
        fig.suptitle("   |   ".join(summary), fontsize=13)
        fig.tight_layout()
        _finish(fig, output_path, show)
    except Exception:
        plt.close(fig)
        raise
    return fig, axes
