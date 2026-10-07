"""Frozen, image-only HGB DISP/energy reconstruction; no training or truth inputs.

Native DL2 energy is TeV; the frozen energy head predicts log10(GeV).
Quality information stays in the returned Python table, not hadroness.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..helper import AltAzFrame, Point2D, ReconstructedEnergy, ReconstructedGeometry, TelescopeFrame


FEATURES = [
    "log_size", "length_deg", "width_deg", "width_length", "leakage", "skewness",
    "kurtosis", "log_n_pixels", "concentration_cog", "concentration_core",
    "concentration_pixel", "max_fraction", "intensity_std_ratio", "intensity_skewness",
    "intensity_kurtosis", "leakage_pixels_width_2", "n_islands",
]
PROXY_FEATURES = FEATURES + ["energy_reco_log10", "disp_abs_reco_deg"]
PID_FEATURES = FEATURES + ["energy_reco_log10"]


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def empty_geometry():
    """Explicit NaNs: the older deployed C++ header has uninitialized scalars."""
    geometry = ReconstructedGeometry()
    geometry.is_valid = False
    geometry.set_telescopes([])
    for name in ("alt", "az", "alt_uncertainty", "az_uncertainty", "direction_error",
                 "core_x", "core_y", "core_pos_error", "tilted_core_x", "tilted_core_y",
                 "tilted_core_uncertainty_x", "tilted_core_uncertainty_y", "hmax", "xmax"):
        setattr(geometry, name, math.nan)
    return geometry


def nominal_hillas(x, y, psi, tel_az, tel_alt, array_az, array_alt):
    """Exactly the coordinate construction used by the training CSV exporter."""
    if not np.isfinite([x, y, psi, tel_az, tel_alt, array_az, array_alt]).all():
        return math.nan, math.nan, math.nan
    camera = TelescopeFrame(azimuth=tel_az, altitude=tel_alt)
    nominal = TelescopeFrame(azimuth=array_az, altitude=array_alt)
    sky = AltAzFrame()
    first = sky.transform_to(camera.transform_to(Point2D(x=x, y=y), sky), nominal)
    second = sky.transform_to(camera.transform_to(
        Point2D(x=x + math.cos(psi), y=y + math.sin(psi)), sky), nominal)
    return first.x, first.y, math.atan2(second.y - first.y, second.x - first.x)


def disp_to_sky(x, y, psi, disp_deg, array_az, array_alt):
    """Return (altitude, azimuth), in radians, using pyLAST's gnomonic inverse."""
    dx = math.radians(disp_deg)
    direction = TelescopeFrame(azimuth=array_az, altitude=array_alt).transform_to(
        Point2D(x=x + dx * math.cos(psi), y=y + dx * math.sin(psi)), AltAzFrame())
    return direction.altitude, direction.azimuth


def feature_table(rows):
    """17 float32 features. Position/origin variables never reach a model head."""
    f = rows.copy()
    if not set(FEATURES).issubset(f.columns):
        f["log_size"] = np.log10(f.hillas_intensity)
        f["length_deg"] = np.degrees(f.hillas_length)
        f["width_deg"] = np.degrees(f.hillas_width)
        f["width_length"] = f.hillas_width / f.hillas_length
        f["leakage"] = f.leakage_intensity_width_2
        f["log_n_pixels"] = np.log10(f.n_pixels.clip(lower=1))
        alignment = np.cos(f.nominal_psi - f.hillas_psi)
        good = np.isfinite(alignment)
        if (np.abs(alignment[good]) <= .99999).any():
            raise ValueError("Camera-to-nominal axis rotation is outside the frozen training convention")
        f["skewness"] = f.hillas_skewness * np.sign(alignment)
        f["kurtosis"] = f.hillas_kurtosis
        f["max_fraction"] = f.intensity_max / f.hillas_intensity
        f["intensity_std_ratio"] = f.intensity_std / (f.hillas_intensity / f.n_pixels)
    return f[FEATURES].replace([np.inf, -np.inf], np.nan).astype("float32")


def image_guard(rows, zenith_deg):
    """Frozen power4_0p08_guard acceptance, copied from the validated analysis."""
    intensity = rows.hillas_intensity.to_numpy()
    radius = np.degrees(rows.hillas_r.to_numpy())
    ratio = rows.hillas_width.to_numpy() / rows.hillas_length.to_numpy()
    leak = rows.leakage_intensity_width_2.to_numpy()
    if zenith_deg == 20:
        boundary = (1.5437882529682072 + 1.787895324035909 * leak**3
                    + .1731232100168901 * radius + 6.208810054456595 * ratio**3
                    + .6120753790903164 * radius * ratio)
        keep = leak < .25
    elif zenith_deg == 60:
        boundary = (1.410923941169068 + 2.705812157086978 * leak**3
                    + .4303231772253843 * np.sqrt(radius) + 2.1789238877870836 * ratio**3
                    + .3759453341765057 * radius * ratio)
        keep = (intensity > 150) & (radius < 3.5)
    else:
        raise ValueError("Only the frozen zenith 20/60 degree domains are supported")
    return keep & (radius > 0) & (np.log10(intensity) > boundary)


class MonoReconstructor:
    def __init__(self, subarray, config_str, model_dir=None):
        """Load frozen assets; relative asset paths may use a portable model_dir."""
        self.subarray = subarray
        self.config = json.loads(config_str) if isinstance(config_str, str) else dict(config_str)
        base = model_dir if model_dir is not None else self.config.get("model_dir")
        self.model_dir = Path(base).expanduser().resolve() if base is not None else None
        for key in ("bundle_path", "proxy_path", "threshold_path", "pid_bundle_path", "pid_cut_path"):
            if self.config.get(key):
                path = Path(self.config[key])
                if self.model_dir is not None and not path.is_absolute():
                    self.config[key] = str(self.model_dir / path)
        self.zenith_deg = int(self.config["zenith_deg"])
        if self.zenith_deg not in (20, 60):
            raise ValueError("Use a frozen z20 or z60 model; no zenith interpolation")
        self.telescope_id = self.config.get("telescope_id")
        self.use_fake_hillas = bool(self.config.get("use_fake_hillas", True))
        self.apply_quality = bool(self.config.get("apply_quality", True))
        self.apply_pid = bool(self.config.get("apply_pid", False))
        self.pid_mode = self.config.get("pid_mode", "fixed_tel7")
        self.geometry_name = self.config.get("geometry_name", "MonoDISPReconstructor")
        self.energy_name = self.config.get("energy_name", "MonoEnergyRegressor")
        if self.geometry_name == self.energy_name:
            raise ValueError("ROOT geometry/energy tree names must be globally different")
        expected = self.config.get("expected_sha256", {})
        self.model_metadata = {}
        for key in ("bundle_path", "proxy_path", "threshold_path", "pid_bundle_path", "pid_cut_path"):
            optional = key.startswith("pid_")
            if optional and (not self.config.get(key) or not Path(self.config[key]).is_file()):
                self.model_metadata[key] = {"available": False, "path": self.config.get(key)}
                continue
            path = Path(self.config[key])
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            actual = digest.hexdigest()
            wanted = expected.get(key, expected.get(key.removesuffix("_path")))
            if wanted and str(wanted).lower() != actual:
                raise ValueError(f"SHA256 mismatch for {key}: {path}")
            self.model_metadata[key] = {"available": True, "path": str(path.resolve()), "sha256": actual}
        self.bundle = joblib.load(self.config["bundle_path"])
        self.proxy = joblib.load(self.config["proxy_path"])
        if list(self.bundle["features"]) != FEATURES:
            raise ValueError("Frozen model must use the ordered 17-feature no-position schema")
        names = getattr(self.proxy, "feature_names_in_", PROXY_FEATURES)
        if list(names) != PROXY_FEATURES:
            raise ValueError("Quality proxy feature order does not match the frozen schema")
        thresholds = pd.read_csv(self.config["threshold_path"])
        self.threshold_energy = thresholds.energy_reco_log10.to_numpy(dtype=float)
        self.threshold_error = thresholds.threshold_log_error.to_numpy(dtype=float)
        if (len(self.threshold_energy) < 3
                or not np.isfinite([self.threshold_energy, self.threshold_error]).all()
                or not (np.diff(self.threshold_energy) > 0).all()):
            raise ValueError("Quality thresholds must be finite and strictly increasing in energy")
        self.pid_bundle = self.pid_cuts = None
        if all(self.model_metadata[key]["available"] for key in ("pid_bundle_path", "pid_cut_path")):
            self.pid_bundle = joblib.load(self.config["pid_bundle_path"])
            if (list(self.pid_bundle["features"]) != PID_FEATURES
                    or self.pid_bundle["gamma_label"] != 1
                    or self.pid_bundle["energy_unit"] != "log10(GeV)"
                    or int(self.pid_bundle["zenith_deg"]) != self.zenith_deg):
                raise ValueError("Mono PID must have the agreed pure 18-feature/log10GeV schema")
            self.pid_cuts = pd.read_csv(self.config["pid_cut_path"])
            required = ["branch", "energy_reco_log10", "threshold_gamma_score",
                        "cal_energy_min_log10", "cal_energy_max_log10", "threshold_interpolation"]
            if not set(required).issubset(self.pid_cuts.columns):
                raise ValueError("Mono PID calibration table is missing required schema columns")
            if not set(self.pid_cuts.branch).issubset({"fixed_tel7", "exact_one"}):
                raise ValueError("Unexpected Mono PID calibration branch")
        self.last_result = None

    def valid_images(self, rows):
        essential = ["hillas_intensity", "hillas_length", "hillas_width",
                     "nominal_x", "nominal_y", "nominal_psi"]
        valid = np.isfinite(rows[essential]).all(axis=1).to_numpy(copy=True)
        for name in ("hillas_intensity", "hillas_length", "hillas_width"):
            valid &= rows[name].to_numpy() > 0
        for name in ("has_cleaned_parameters", "triggered_after_nsb"):
            if name in rows:
                valid &= rows[name].fillna(False).astype(bool).to_numpy()
        return valid

    def guard_mask(self, rows):
        valid = self.valid_images(rows)
        mask = np.zeros(len(rows), dtype=bool)
        if valid.any():
            mask[valid] = image_guard(rows.loc[valid], self.zenith_deg)
        return mask

    def predict_table(self, rows, pid_mode=None):
        """Batch predict observed image rows; truth columns, if present, are ignored."""
        rows = pd.DataFrame(rows)
        result = rows[[c for c in ("run_id", "event_id", "telescope_id",
                                  "nominal_x", "nominal_y", "nominal_psi") if c in rows]].copy()
        valid = self.valid_images(rows)
        result["valid_image"] = valid
        result["image_cut_pass"] = self.guard_mask(rows)
        for name in ("disp_reco_deg", "energy_reco_log10", "energy_reco_GeV",
                     "quality_proxy_log_error", "quality_threshold_smooth"):
            result[name] = np.nan
        if valid.any():
            inputs = feature_table(rows.loc[valid])
            magnitude = self.bundle["magnitude"].predict(inputs)
            sign = self.bundle["sign"].predict(inputs)
            disp = magnitude * np.where(sign, 1., -1.)
            energy = self.bundle["energy"].predict(inputs)
            px = inputs.copy()
            px["energy_reco_log10"] = energy
            px["disp_abs_reco_deg"] = np.abs(disp)
            proxy = self.proxy.predict(px)
            limit = np.interp(energy, self.threshold_energy, self.threshold_error)
            result.loc[valid, ["disp_reco_deg", "energy_reco_log10", "energy_reco_GeV",
                               "quality_proxy_log_error", "quality_threshold_smooth"]] = np.column_stack(
                [disp, energy, 10. ** energy, proxy, limit])
        finite = np.isfinite(result[["disp_reco_deg", "energy_reco_GeV",
                                    "quality_proxy_log_error"]]).all(axis=1)
        result["reconstruction_valid"] = result.valid_image & finite
        result["quality_pass"] = finite & (result.quality_proxy_log_error < result.quality_threshold_smooth)
        result["accepted"] = result.image_cut_pass & finite
        if self.apply_quality:
            result["accepted"] &= result.quality_pass
        self._pid_decision(rows, result, pid_mode or self.pid_mode)
        if self.apply_pid:
            result["accepted"] &= result.pass_pid
        return result

    def _pid_decision(self, rows, result, mode):
        if mode not in ("fixed_tel7", "exact_one"):
            raise ValueError("pid_mode must be fixed_tel7 or exact_one")
        result["pid_mode"] = mode
        result["pid_gamma_score"] = np.nan
        result["pid_threshold_gamma_score"] = np.nan
        result["pid_valid"] = False
        result["pass_pid"] = False
        result["pid_status"] = "assets_unavailable"
        if self.pid_bundle is None or self.pid_cuts is None:
            return
        cuts = self.pid_cuts[self.pid_cuts.branch == mode].sort_values("energy_reco_log10")
        if cuts.empty:
            result["pid_status"] = "branch_not_calibrated"
            return
        nodes = cuts.energy_reco_log10.to_numpy(dtype=float)
        thresholds = cuts.threshold_gamma_score.to_numpy(dtype=float)
        lower = cuts.cal_energy_min_log10.to_numpy(dtype=float)
        upper = cuts.cal_energy_max_log10.to_numpy(dtype=float)
        if (not np.isfinite([nodes, thresholds, lower, upper]).all()
                or not (np.diff(nodes) > 0).all()
                or not np.all(lower == lower[0]) or not np.all(upper == upper[0])
                or lower[0] >= upper[0] or (thresholds < 0).any() or (thresholds > 1).any()
                or set(cuts.threshold_interpolation) != {"logit"}):
            raise ValueError("Invalid frozen Mono PID cut nodes or calibration domain")
        valid = result.reconstruction_valid.to_numpy(copy=True)
        if valid.any():
            x = feature_table(rows.loc[valid])
            x["energy_reco_log10"] = result.loc[valid, "energy_reco_log10"].to_numpy()
            model = self.pid_bundle["model"]
            labels = list(model.classes_)
            if labels.count(self.pid_bundle["gamma_label"]) != 1:
                raise ValueError("Mono PID model has no unique gamma class")
            score = model.predict_proba(x.astype(np.float32))[:, labels.index(self.pid_bundle["gamma_label"])]
            result.loc[valid, "pid_gamma_score"] = score
        clipped = np.clip(thresholds, 1e-12, 1 - 1e-12)
        logits = np.log(clipped / (1 - clipped))
        # PID calibration cached this derived feature as float32; reproduce that
        # precision for PID domain/cut only, keeping direction/E/proxy unchanged.
        energy = result.energy_reco_log10.to_numpy(dtype=np.float32).astype(float)
        interpolated = np.interp(energy, nodes, logits)
        result["pid_threshold_gamma_score"] = 1 / (1 + np.exp(-interpolated))
        supported = ((energy >= lower[0]) & (energy < upper[0])
                     & np.isfinite(result.pid_gamma_score.to_numpy()))
        supported &= result.image_cut_pass.to_numpy() & result.quality_pass.to_numpy()
        if mode == "fixed_tel7":
            supported &= (rows.telescope_id.to_numpy() == 7) if "telescope_id" in rows else self.telescope_id == 7
        result["pid_valid"] = supported
        result["pass_pid"] = supported & (result.pid_gamma_score >= result.pid_threshold_gamma_score)
        result["pid_status"] = np.where(supported, "calibrated", "outside_calibration_context")

    def event_table(self, event, telescope_ids=None):
        """Adapt the image-processor result without reading shower truth."""
        pointing = event.pointing
        if not hasattr(pointing, "tels"):
            raise RuntimeError("Mono adapter requires the read-only Pointing.tels binding patch")
        measured_zenith = 90 - math.degrees(pointing.array_altitude)
        if not math.isclose(measured_zenith, self.zenith_deg, abs_tol=.1):
            raise ValueError("Event pointing is outside this frozen model's zenith domain")
        if self.use_fake_hillas:
            if event.simulation is None:
                raise ValueError("use_fake_hillas=True requires simulated images, not shower truth")
            cameras = event.simulation.tels
            triggered = set(event.simulation.triggered_tels)
        else:
            if event.dl1 is None:
                raise ValueError("use_fake_hillas=False requires DL1 image parameters")
            cameras = event.dl1.tels
            triggered = set(cameras)
        ids = sorted(cameras) if telescope_ids is None else [int(i) for i in telescope_ids]
        output = []
        for tel_id in ids:
            camera = cameras.get(tel_id)
            if tel_id not in pointing.tels:
                raise ValueError(f"Missing measured telescope pointing for telescope {tel_id}")
            p = getattr(camera, "image_parameters", None)
            h = getattr(p, "hillas", None)
            row = {"run_id": event.run_id, "event_id": event.event_id, "telescope_id": tel_id,
                   "triggered_after_nsb": tel_id in triggered,
                   "has_cleaned_parameters": h is not None and _number(h.intensity) >= 50}
            for name in ("intensity", "length", "width", "x", "y", "r", "psi", "skewness", "kurtosis"):
                row[f"hillas_{name}"] = _number(getattr(h, name, math.nan))
            tp = pointing.tels[tel_id]
            row["nominal_x"], row["nominal_y"], row["nominal_psi"] = nominal_hillas(
                row["hillas_x"], row["hillas_y"], row["hillas_psi"], tp.azimuth, tp.altitude,
                pointing.array_azimuth, pointing.array_altitude)
            for group, names in {
                "leakage": ("intensity_width_2", "pixels_width_2"),
                "morphology": ("n_pixels", "n_islands"),
                "concentration": ("concentration_cog", "concentration_core", "concentration_pixel"),
                "intensity": ("intensity_max", "intensity_std", "intensity_skewness", "intensity_kurtosis"),
            }.items():
                parameters = getattr(p, group, None)
                for name in names:
                    column = f"leakage_{name}" if group == "leakage" else name
                    row[column] = _number(getattr(parameters, name, math.nan))
            output.append(row)
        if not output:
            return pd.DataFrame(columns=["run_id", "event_id", "telescope_id", "hillas_intensity",
                                         "hillas_length", "hillas_width", "nominal_x", "nominal_y",
                                         "nominal_psi", "has_cleaned_parameters", "triggered_after_nsb"])
        return pd.DataFrame(output)

    def store_result(self, event, result, geometry_name=None, energy_name=None, selected=False):
        if len(result) != 1:
            raise ValueError("One named mono DL2 result requires exactly one telescope row")
        row = result.iloc[0]
        geometry = empty_geometry()
        energy = ReconstructedEnergy()
        valid = bool(row.accepted if selected else row.image_cut_pass and row.reconstruction_valid)
        geometry.is_valid = valid
        energy.energy_valid = valid
        if bool(row.valid_image) and np.isfinite(row.disp_reco_deg):
            geometry.alt, geometry.az = disp_to_sky(
                row.nominal_x, row.nominal_y, row.nominal_psi, row.disp_reco_deg,
                event.pointing.array_azimuth, event.pointing.array_altitude)
            geometry.set_telescopes([int(row.telescope_id)])
        if np.isfinite(row.energy_reco_GeV):
            energy.estimate_energy = float(row.energy_reco_GeV) / 1000.
            energy.telescopes = [int(row.telescope_id)]
        # Mono has no reconstructed core/hmax; explicit NaNs prevent invented core=0.
        dl2 = event.ensure_dl2()
        dl2.add_geometry(geometry_name or self.geometry_name, geometry)
        dl2.add_energy(energy_name or self.energy_name, energy)

    def __call__(self, event, telescope_id=None, store=True, pid_mode=None):
        tel_id = self.telescope_id if telescope_id is None else telescope_id
        if tel_id is None:
            raise ValueError("Choose a fixed telescope_id or supply the hybrid exact-one candidate")
        result = self.predict_table(self.event_table(event, [tel_id]), pid_mode=pid_mode)
        if store:
            self.store_result(event, result)
        self.last_result = result
        return result
