"""Opt-in E20/PID23 inference with the pinned 2026-11-04 stereo release.

Requires the 36-telescope z20/z60 pipeline archive and its three separate
LightGBM TXT files. Call after image processing: the training inputs used
15/7.5 p.e. tailcuts and simulated noise, so applying the weights to measured
DL1 is exploratory. The frozen atmosphere yields vertical depth, not slant
Xmax. Its legacy height formula is kept to match the trained features.

In a native pyLAST process, use the same LightGBM build for Python and
pyLAST (4.7.0 in the validated Linux environment).

No shower truth is read, and no sensitivity/sky-source cut is applied here.
"""

import ast
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


E18 = ("log_intensity rec_impact_parameter intensity_mean intensity_max intensity_std "
       "leakage_pixels_width_1 leakage_pixels_width_2 hillas_length hillas_width hillas_r "
       "hillas_skewness hillas_kurtosis leakage_intensity_width_1 leakage_intensity_width_2 "
       "concentration_cog concentration_core concentration_pixel morphology_num_pixels").split()
XMAX = "rec_vertical_xmax_g_cm2"
E20 = E18 + [XMAX, "n_selected_images"]
PID23 = E18 + ["morphology_n_islands", "rec_energy", "tel_rec_energy", "rec_energy_std", XMAX]
_PREFIX = "LACT_36tel_E20_PID23_50h/"
_ASSET_SHA = {
    "frozen/scoring/native_current_full.py": "9e6a92209e07071cb8f9f75baeb24965b4201fd90be47f484754f3dd68b261ca",
    "frozen/scoring/height_math.py": "88636b36725cd9eaba2a47114dd9fb30290fd6c4aed58467f7ddf349d88d8a54",
    "frozen/scoring/atmprof1_gamma_verified.dat": "f9f76f2455b499f738e5c42f40ff1d88f33964e6106f318fe3970ac366d71aa3",
    "frozen/preparation/quality_recipe.json": "e7d3270948dcedbaf6a749f3b74befa441f840b6d2671fdd09551ecc3f24c418",
}


def _check(data, expected, label):
    if sha256(data).hexdigest() != expected:
        raise ValueError(f"Frozen stereo asset SHA256 differs: {label}")
    return data


def _kernels(source, names, **inputs):
    # Import only pinned numeric functions, without the upstream CLI/global side effects.
    functions = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    import math
    env = dict(np=np, math=math, **inputs)
    code = ast.Module(body=[functions[name] for name in names], type_ignores=[])
    exec(compile(code, "<frozen stereo numeric kernels>", "exec"), env)
    return env


class PortableStereoReconstructor:
    """Independent 36tel z20/z60 reconstruction; does not replace native Hillas.

    ``config`` is a mapping or JSON string containing ``group``,
    ``pipeline_archive`` and ``models`` with keys ``energy18``, ``energy20``,
    ``pid23``. Set ``use_fake_hillas`` only for the simulated image layer.
    With real inputs it consumes observed DL1, including telescope and array
    pointing, and never reads truth. Optionally pin model files with an
    ``expected_sha256`` mapping using the same model keys.
    ``__call__`` returns a per-event result, with E20 output energy and PID23
    score calculated with E18 predictors. It writes separate DL2 names
    unless called with ``store=False``.
    """

    geometry_name = "E20StereoDirection"
    energy_name = "E20StereoEnergy"
    particle_name = "PID23StereoParticle"

    def __init__(self, subarray, config):
        import lightgbm as lgb
        from scipy.interpolate import CubicSpline

        config = json.loads(config) if isinstance(config, str) else dict(config)
        self.group = config["group"]
        if self.group not in {"36tel_z20", "36tel_z60"}:
            raise ValueError("Frozen stereo archive supports only 36tel_z20 and 36tel_z60")
        self.zenith = int(self.group[-2:])
        self.use_fake_hillas = bool(config.get("use_fake_hillas", False))
        self.positions = {int(tel): np.asarray(pos, dtype=float).copy()
                          for tel, pos in subarray.tel_positions.items()}
        if set(self.positions) != set(range(36)) or not np.isfinite(list(self.positions.values())).all():
            raise ValueError("This stereo model requires the complete 36-telescope layout (IDs 0..35)")

        with ZipFile(config["pipeline_archive"]) as archive:
            assets = {Path(name).name: _check(archive.read(_PREFIX + name), digest, name)
                      for name, digest in _ASSET_SHA.items()}
        source = assets["native_current_full.py"]
        self._geometry = _kernels(source, ("reconstruct", "rotation", "geometry", "feature", "apply_cut"))
        self._energy18 = _kernels(source, ("add_energy",), ENERGY_FEATURES=E18)["add_energy"]
        self._energy20 = _kernels(source, ("add_energy",), ENERGY_FEATURES=E20)["add_energy"]
        height = {"__name__": "_pylast_frozen_stereo_height"}
        exec(compile(assets["height_math.py"], "<frozen stereo height>", "exec"), height)
        self._height, self._height_columns = height["reconstruct_height"], height["INPUT_COLUMNS"]
        profile = np.loadtxt(BytesIO(assets["atmprof1_gamma_verified.dat"]), dtype=np.float64)
        self._depth = CubicSpline(profile[:, 0], profile[:, 2], bc_type="natural", extrapolate=False)
        self._recipe = json.loads(assets["quality_recipe.json"])

        self.models, self.model_sha256 = {}, {}
        expected = config.get("expected_sha256", {})
        for name, order in (("energy18", E18), ("energy20", E20), ("pid23", PID23)):
            path = Path(config["models"][name])
            data = path.read_bytes()
            digest = sha256(data).hexdigest()
            if name in expected and digest != expected[name]:
                raise ValueError(f"Stereo model SHA256 differs for {name}: {path}")
            model = lgb.Booster(model_str=data.decode("utf-8"))
            if model.feature_name() != order or model.num_feature() != len(order):
                raise ValueError(f"Stereo model feature order differs for {name}")
            self.models[name] = model
            self.model_sha256[name] = digest
        self.last_result = None

    def _images(self, layer, pointing):
        rows = []
        for tel_id, camera in sorted(layer.tels.items()):
            pos, tp = self.positions.get(tel_id), pointing.tels.get(tel_id)
            if pos is None or tp is None:
                raise ValueError(f"Stereo image has no configured telescope/pointing: {tel_id}")
            params = camera.image_parameters
            h = params.hillas
            if not np.isfinite([h.intensity, h.length, h.width, h.r, h.x, h.y, h.psi]).all() or min(h.intensity, h.length, h.width, h.r) <= 0:
                continue
            x, y, psi = h.x, h.y, h.psi
            if (tp.azimuth, tp.altitude) != (pointing.array_azimuth, pointing.array_altitude):
                from .MonoReconstructor import nominal_hillas

                x, y, psi = nominal_hillas(x, y, psi, tp.azimuth, tp.altitude,
                                           pointing.array_azimuth, pointing.array_altitude)
            if not np.isfinite([x, y, psi]).all():
                continue
            row = dict(telescope_id=tel_id, array_az=pointing.array_azimuth, array_alt=pointing.array_altitude,
                       nominal_x=x, nominal_y=y, nominal_psi=psi,
                       tel_x=pos[0], tel_y=pos[1], tel_z=pos[2], log_intensity=np.log10(h.intensity),
                       morphology_num_pixels=params.morphology.n_pixels, morphology_n_islands=params.morphology.n_islands)
            for name in ("intensity", "length", "width", "r", "x", "y", "skewness", "kurtosis"):
                row["hillas_" + name] = getattr(h, name)
            for name in ("pixels_width_1", "pixels_width_2", "intensity_width_1", "intensity_width_2"):
                row["leakage_" + name] = getattr(params.leakage, name)
            for name in ("intensity_mean", "intensity_max", "intensity_std"):
                row[name] = getattr(params.intensity, name)
            for name in ("concentration_cog", "concentration_core", "concentration_pixel"):
                row[name] = getattr(params.concentration, name)
            rows.append(row)
        return rows

    def _finish(self, event, result, store, geometry=None):
        self.last_result = result
        if not store:
            return result

        from ..helper import ReconstructedEnergy, ReconstructedParticle
        from .MonoReconstructor import empty_geometry

        valid = result["reconstruction_valid"]
        tel_ids = result["selected_telescope_ids"]
        direction, energy, particle = empty_geometry(), ReconstructedEnergy(), ReconstructedParticle()
        direction.is_valid = energy.energy_valid = particle.is_valid = valid
        direction.alt = float("nan") if geometry is None else geometry["alt"]
        direction.az = float("nan") if geometry is None else geometry["az"]
        direction.core_x = result["reco_core_x_m"]
        direction.core_y = result["reco_core_y_m"]
        direction.hmax = result["hmax_m"]
        direction.xmax = result["reco_vertical_depth_g_cm2"]
        energy.estimate_energy = result["rec_energy_TeV"]
        particle.hadroness = result["pid23_hadronness"]
        if valid:
            direction.set_telescopes(tel_ids)
            energy.telescopes = tel_ids
        # estimate_energy_std is read-only in the current native binding; the
        # per-telescope energy spread and E18 energy remain in the result.
        dl2 = event.ensure_dl2()
        dl2.add_geometry(self.geometry_name, direction)
        dl2.add_energy(self.energy_name, energy)
        dl2.add_particle(self.particle_name, particle)
        return result

    def __call__(self, event, *, store=True):
        import pandas as pd

        result = dict(event_id=int(event.event_id), run_id=int(event.run_id),
                      reconstruction_valid=False, reason="", n_selected_images=0, selected_telescope_ids=[])
        result.update(dict.fromkeys(("reco_altitude_deg", "reco_azimuth_deg", "reco_core_x_m", "reco_core_y_m",
                                     "hmax_m", "reco_vertical_depth_g_cm2", "rec_energy_TeV", "rec_energy_std_TeV",
                                     "PID23_rec_energy_E18_TeV", "PID23_rec_energy_std_E18_TeV", "pid23_hadronness"), float("nan")))
        pointing = event.pointing
        if pointing is None or not np.isfinite([pointing.array_altitude, pointing.array_azimuth]).all():
            result["reason"] = "missing_pointing"
            return self._finish(event, result, store)
        if abs(90 - np.degrees(pointing.array_altitude) - self.zenith) > .1:
            result["reason"] = "outside_model_zenith"
            return self._finish(event, result, store)
        layer = event.simulation if self.use_fake_hillas else event.dl1
        rows = [] if layer is None else self._images(layer, pointing)
        frame = pd.DataFrame(rows)
        if len(frame):
            data = dict(intensity=frame.hillas_intensity, log10_intensity=frame.log_intensity,
                        leakage=frame.leakage_intensity_width_2, width_length=frame.hillas_width/frame.hillas_length,
                        r_deg=np.degrees(frame.hillas_r))
            frame = frame.loc[self._geometry["apply_cut"](data, self.zenith, self._recipe)].copy()
        n = len(frame)
        result.update(n_selected_images=n, selected_telescope_ids=frame.telescope_id.astype(int).tolist() if n else [])
        if n < 2 or (n < 4 and not np.degrees(frame.hillas_r).lt(3.).all()):
            result["reason"] = "fewer_than_two_images" if n < 2 else "low_multiplicity_radius_guard"
            return self._finish(event, result, store)
        geometry = self._geometry["geometry"](frame)
        if geometry is None:
            result["reason"] = "degenerate_geometry"
            return self._finish(event, result, store)
        frame = frame.assign(reco_alt=geometry["alt"], reco_az=geometry["az"], reco_core_x=geometry["core_x"],
                             reco_core_y=geometry["core_y"], rec_impact_parameter=geometry["impact"],
                             n_selected_images=n, event_uid=0)
        if not np.isfinite(frame[E18 + ["morphology_n_islands"]]).all().all():
            raise ValueError("Nonfinite selected stereo model feature; no imputation")
        height = float(self._height({key: frame[key].to_numpy() for key in self._height_columns}, [0, n])["hmax_m"][0])
        depth = float(self._depth(height / 1000))
        frame[XMAX] = depth if np.isfinite(depth) and depth >= 0 else float("nan")

        e18 = self._energy18(frame, self.models["energy18"], 1)
        e20 = self._energy20(frame, self.models["energy20"], 1)
        scores = self.models["pid23"].predict(e18[PID23], num_threads=1)
        intensity = frame.hillas_intensity
        score = float((scores * intensity).groupby(frame.event_uid, sort=False).sum().iloc[0] /
                      intensity.groupby(frame.event_uid, sort=False).sum().iloc[0])
        if not np.isfinite(scores).all() or not 0 <= score <= 1:
            raise ValueError("Frozen PID23 prediction outside [0, 1]")
        result.update(reconstruction_valid=True, reason="ok", reco_altitude_deg=float(np.degrees(geometry["alt"])),
                      reco_azimuth_deg=float(np.degrees(geometry["az"]) % 360), reco_core_x_m=float(geometry["core_x"]),
                      reco_core_y_m=float(geometry["core_y"]), hmax_m=height,
                      reco_vertical_depth_g_cm2=float(frame[XMAX].iloc[0]), rec_energy_TeV=float(e20.rec_energy.iloc[0]),
                      rec_energy_std_TeV=float(e20.rec_energy_std.iloc[0]), PID23_rec_energy_E18_TeV=float(e18.rec_energy.iloc[0]),
                      PID23_rec_energy_std_E18_TeV=float(e18.rec_energy_std.iloc[0]), pid23_hadronness=score)
        return self._finish(event, result, store, geometry)
