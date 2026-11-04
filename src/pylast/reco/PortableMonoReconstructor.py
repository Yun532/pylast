"""Use a trusted v3 portable mono bundle on an individual pyLAST image."""
from __future__ import annotations

import ast
import hashlib
import json
import operator
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd


_RUNTIME = "code/portable_mono.py"
_CUT_HELPER = "preselection_geometry_helper/code/original_s_runtime.py"
_RECIPE = "config/current_image_only_recipe.json"
_POSITIVE = ("hillas_intensity", "hillas_length", "hillas_width", "hillas_r")
_OBSERVED = (*_POSITIVE, "leakage_intensity_width_2", "nominal_x", "nominal_y", "nominal_psi")
_SCORES = ("disp_reco_deg", "energy_reco_log10", "energy_reco_GeV",
           "quality_proxy_log_error", "quality_threshold_smooth",
           "pid_gamma_score", "pid_threshold_gamma_score")


def _member_bytes(root, manifest, name):
    path = root / name
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Portable mono member lies outside its bundle: {name}")
    data = path.read_bytes()
    descriptor = manifest["files"][name]
    if len(data) != descriptor["bytes"] or hashlib.sha256(data).hexdigest() != descriptor["sha256"]:
        raise ValueError(f"Portable mono member does not match its manifest: {name}")
    return data


def _load_runtime(root, expected_sha):
    raw = (root / "manifest.json").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expected_sha is not None and expected_sha.lower() != sha:
        raise ValueError("Portable mono manifest does not match the configured SHA256")
    manifest = json.loads(raw)
    if manifest.get("schema") != "mono_refresh_portable_frame_v3":
        raise ValueError("Expected a v3 portable mono frame bundle")
    module = ModuleType("_pylast_portable_mono_bundle")
    module.__file__ = str(root / _RUNTIME)
    source = _member_bytes(root, manifest, _RUNTIME)
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    # The bundle loader checks all its members and validates fitted model metadata.
    return module.MonoRuntime(root, manifest_sha256=sha)


def _load_image_cut(root, manifest):
    # These are the two pure image-cut functions in the verified helper. Loading
    # the entire helper would import stereo LightGBM and change process thread settings.
    path = str(root / _CUT_HELPER)
    tree = ast.parse(_member_bytes(root, manifest, _CUT_HELPER), filename=path)
    wanted = {"feature", "apply_cut"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in functions} != wanted:
        raise ValueError("Portable mono image-cut helper is incomplete")
    namespace = {"np": np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), path, "exec"), namespace)
    return namespace["apply_cut"], json.loads(_member_bytes(root, manifest, _RECIPE))


def _telescope_id(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("telescope_id must be a nonnegative integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError("telescope_id must be a nonnegative integer") from exc
    if value < 0:
        raise ValueError("telescope_id must be a nonnegative integer")
    return value


class PortableMonoReconstructor:
    """Run v3 mono inference without shower truth or external detector data.

    Configuration names the trusted `bundle_dir`, `group`, optional
    `manifest_sha256`, and optional `telescope_id`. `threshold_branch`
    selects a calibration table ("fixed" or "exact_one"); it does not classify
    the event's array multiplicity or substitute its telescope identity.
    `predict_table` also works without native bindings if the 17 features
    are present alongside the observed image-cut and nominal Hillas columns.
    """

    def __init__(self, subarray, config_str, *, bundle_dir=None):
        config = json.loads(config_str) if isinstance(config_str, str) else dict(config_str)
        self.subarray = subarray
        self.telescope_id = _telescope_id(config.get("telescope_id"))
        self.group = config["group"]
        self.branch = config.get("threshold_branch", "fixed")
        if self.branch not in ("fixed", "exact_one"):
            raise ValueError("threshold_branch must be fixed or exact_one")
        self.geometry_name = config.get("geometry_name", "PortableMonoDirection")
        self.energy_name = config.get("energy_name", "PortableMonoEnergy")
        if self.geometry_name == self.energy_name:
            raise ValueError("Geometry and energy ROOT tree names must differ")
        self.use_fake_hillas = bool(config.get("use_fake_hillas", False))
        root = Path(bundle_dir if bundle_dir is not None else config["bundle_dir"]).resolve()
        self.runtime = _load_runtime(root, config.get("manifest_sha256"))
        manifest = self.runtime.manifest
        groups = manifest["inference_config"]["groups"]
        if self.group not in groups:
            raise ValueError(f"Unsupported portable mono group: {self.group}")
        self.zenith_deg = groups[self.group]["zenith"]
        self.reference_telescope_id = (
            manifest["inference_config"]["fixed_telescope_id"] if self.branch == "fixed" else None
        )
        self._apply_cut, self.recipe = _load_image_cut(root, manifest)
        self.runtime.models(self.zenith_deg)
        self.last_result = None
        self._mono = None

    @property
    def mono(self):
        if self._mono is None:
            from .MonoReconstructor import MonoReconstructor

            self._mono = MonoReconstructor.__new__(MonoReconstructor)
            self._mono.subarray = self.subarray
            self._mono.zenith_deg = self.zenith_deg
            self._mono.use_fake_hillas = self.use_fake_hillas
            self._mono.geometry_name = self.geometry_name
            self._mono.energy_name = self.energy_name
        return self._mono

    @staticmethod
    def _valid_image(rows):
        return (
            rows.has_cleaned_parameters.fillna(False).to_numpy(dtype=bool)
            & np.isfinite(rows[list(_OBSERVED)].to_numpy(dtype=float)).all(axis=1)
            & rows[list(_POSITIVE)].gt(0).all(axis=1).to_numpy()
        )

    def _image_masks(self, rows):
        valid = self._valid_image(rows)
        keep = np.zeros(len(rows), dtype=bool)
        if valid.any():
            good = rows.loc[valid]
            intensity = good.hillas_intensity.to_numpy()
            keep[valid] = self._apply_cut(
                dict(intensity=intensity, log10_intensity=np.log10(intensity),
                     leakage=good.leakage_intensity_width_2.to_numpy(),
                     width_length=(good.hillas_width / good.hillas_length).to_numpy(),
                     r_deg=np.degrees(good.hillas_r.to_numpy())),
                self.zenith_deg, self.recipe,
            )
        return valid, keep

    def _score_features(self, features):
        common = self.runtime.common
        x = features[common.FEATURES].astype("float32")
        if np.isinf(x.to_numpy()).any():
            raise ValueError("Infinite portable mono features are not permitted")
        (bundle, proxy, pid, qnodes, pnodes), _ = self.runtime.models(self.zenith_deg)
        disp, energy = common.reconstruction(x, bundle)
        derived = x.assign(energy_reco_log10=energy, disp_abs_reco_deg=np.abs(disp))
        quality = proxy["model"].predict(derived[common.PROXY_FEATURES])
        derived["energy_reco_log10"] = derived.energy_reco_log10.astype("float32")
        probability = pid["model"].predict_proba(derived[common.PID_FEATURES])[:, 1]
        values = pd.DataFrame(dict(
            disp_reco_deg=disp, energy_reco_log10=energy, energy_reco_GeV=10.0 ** energy,
            quality_proxy_log_error=quality, pid_gamma_score=probability,
            quality_threshold_smooth=common.lookup(qnodes, self.group, self.branch, energy),
            pid_threshold_gamma_score=common.lookup(pnodes, self.group, self.branch, energy),
        ), index=x.index)
        if (not np.isfinite(values.to_numpy()).all() or not values.energy_reco_GeV.gt(0).all()
                or not values.pid_gamma_score.between(0, 1).all()):
            raise ValueError("Invalid portable mono reconstruction or selection output")
        for label, nodes in (("quality", qnodes), ("pid", pnodes)):
            part = nodes.loc[nodes.group.eq(self.group) & nodes.branch.eq(self.branch)].iloc[0]
            values[label + "_calibration_in_domain"] = (
                (energy >= part.calibration_energy_min_log10) & (energy <= part.calibration_energy_max_log10)
            )
        return values

    def predict_table(self, rows):
        """Score one selected telescope row; do not infer an array routing state."""
        if len(rows) != 1 or _telescope_id(rows.telescope_id.iloc[0]) is None:
            raise ValueError("A portable mono prediction requires one real telescope row")
        if "zenith" in rows and not rows.zenith.eq(self.zenith_deg).all():
            raise ValueError("Input zenith group differs from the portable mono group")
        result = rows.copy()
        valid, keep = self._image_masks(rows)
        result["valid_image"], result["image_cut_pass"] = valid, keep
        result[list(_SCORES)] = np.nan
        domain = ["quality_calibration_in_domain", "pid_calibration_in_domain"]
        result[domain] = False
        if keep.any():
            eligible = rows.loc[keep]
            if not set(self.runtime.common.FEATURES).issubset(eligible.columns):
                from .MonoReconstructor import feature_table

                eligible = feature_table(eligible)
            scores = self._score_features(eligible)
            result.loc[keep, list(_SCORES)] = scores[list(_SCORES)].to_numpy()
            result.loc[keep, domain] = scores[domain].to_numpy()
        result["reconstruction_valid"] = keep
        result["quality_pass"] = keep & (result.quality_proxy_log_error < result.quality_threshold_smooth)
        result["pass_pid"] = keep & (result.pid_gamma_score >= result.pid_threshold_gamma_score)
        result["accepted"] = result.quality_pass & result.pass_pid
        result["group"], result["branch"] = self.group, self.branch
        result["threshold_reference_telescope_id"] = self.reference_telescope_id
        result["model_candidate_id"] = self.runtime.manifest["candidate_id"]
        result["model_manifest_sha256"] = self.runtime.manifest_sha
        return result

    def store_result(self, event, result):
        self.mono.store_result(event, result, selected=True)

    def __call__(self, event, *, telescope_id=None, store=True):
        chosen = self.telescope_id if telescope_id is None else _telescope_id(telescope_id)
        rows = self.mono.event_table(event, telescope_ids=None if chosen is None else [chosen])
        if chosen is None and len(rows) != 1:
            if len(rows):
                rows = rows.loc[self._valid_image(rows)]
            if len(rows) != 1:
                raise ValueError("Specify telescope_id unless there is one usable image")
        result = self.predict_table(rows)
        if store:
            self.store_result(event, result)
        self.last_result = result
        return result
