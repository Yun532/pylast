"""Small deterministic contract check; --native uses the isolated patched pyLAST.

Default coordinate doubles mirror CoordFrames.cpp, so this is not a ROOT smoke test.
No model training, data analysis, or large model loading occurs here.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import math
from pathlib import Path
import hashlib
import json
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace

RECO_SOURCE = Path(__file__).resolve().parents[1] / "src" / "pylast" / "reco"

import numpy as np
import pandas as pd


def fake_helper():
    helper = ModuleType("pylast.helper")

    class Point2D:
        def __init__(self, x, y):
            self.x, self.y = x, y

    class AltAzFrame:
        def transform_to(self, point, target):
            v = [math.cos(point.azimuth) * math.cos(point.altitude),
                 -math.sin(point.azimuth) * math.cos(point.altitude), math.sin(point.altitude)]
            x, y, z = target.rotation @ v
            return Point2D(-x / z, -y / z)

    class TelescopeFrame:
        def __init__(self, azimuth, altitude):
            ca, sa = math.cos(azimuth), math.sin(azimuth)
            cb, sb = math.sin(altitude), -math.cos(altitude)
            self.rotation = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]]) @ np.array(
                [[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])

        def transform_to(self, point, target):
            v = self.rotation.T @ np.array([-point.x, -point.y, 1.])
            v /= np.linalg.norm(v)
            return SimpleNamespace(azimuth=math.atan2(-v[1], v[0]) % (2 * math.pi),
                                   altitude=math.asin(v[2]))

    class ReconstructedGeometry:
        def __init__(self):
            self.is_valid = False
            self.alt = self.az = self.core_x = self.core_y = self.hmax = math.nan
            self.telescopes = []

        def set_telescopes(self, ids):
            self.telescopes = ids

    class ReconstructedEnergy:
        def __init__(self):
            self.energy_valid, self.estimate_energy, self.telescopes = False, 0., []

    for item in (Point2D, AltAzFrame, TelescopeFrame, ReconstructedGeometry, ReconstructedEnergy):
        setattr(helper, item.__name__, item)
    return helper


def load_modules(native):
    if native:
        sys.meta_path = [finder for finder in sys.meta_path
                         if "_pylast_editable" not in type(finder).__module__]
        import pylast.helper as helper
        assert Path(helper.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), helper.__file__
        return helper, importlib.import_module("pylast.reco.MonoReconstructor"), importlib.import_module("pylast.reco.HybridReconstructor")
    else:
        helper = fake_helper()
        for name in ("pylast", "pylast.reco"):
            module = ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
        sys.modules["pylast.helper"] = helper
    modules = []
    for name in ("MonoReconstructor", "HybridReconstructor"):
        spec = importlib.util.spec_from_file_location(f"pylast.reco.{name}", RECO_SOURCE / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    return helper, *modules


class ConstantHead:
    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.full(len(x), self.value)


class ConstantPID:
    classes_ = np.array([1, 0])  # Deliberately reversed: never assume gamma is column 1.

    def predict_proba(self, x):
        assert list(x) == [*self.features, "energy_reco_log10"]
        assert all(dtype == np.float32 for dtype in x.dtypes)
        return np.tile([.8, .2], (len(x), 1))


class FakeDL2:
    def __init__(self):
        self.geometry, self.energy = {}, {}

    def add_geometry(self, name, geometry):
        self.geometry[name] = geometry

    def add_energy(self, name, energy):
        self.energy[name] = energy



def portable_assets_contract(module):
    """Only invented tiny serialized heads; no production assets or training."""
    import joblib
    with TemporaryDirectory(prefix="pylast_mono_contract_") as temporary:
        root = Path(temporary)
        (root / "z20").mkdir()
        relative = dict(bundle_path="z20/reconstruction.joblib", proxy_path="z20/proxy.joblib",
                        threshold_path="z20/quality_cut.csv", pid_bundle_path="z20/pid.joblib",
                        pid_cut_path="z20/pid_cut.csv")
        bundle = dict(features=module.FEATURES, magnitude=ConstantHead(.4),
                      sign=ConstantHead(True), energy=ConstantHead(3.5))
        joblib.dump(bundle, root / relative["bundle_path"])
        joblib.dump(ConstantHead(-1.), root / relative["proxy_path"])
        pd.DataFrame(dict(energy_reco_log10=[2., 4., 6.],
                          threshold_log_error=[-.8, -.8, -.8])).to_csv(
                              root / relative["threshold_path"], index=False)
        pid = ConstantPID()
        pid.features = module.FEATURES
        joblib.dump(dict(model=pid, features=module.PID_FEATURES, gamma_label=1,
                         energy_unit="log10(GeV)", zenith_deg=20), root / relative["pid_bundle_path"])
        pd.DataFrame([
            dict(branch=branch, energy_reco_log10=e, threshold_gamma_score=.7,
                 cal_energy_min_log10=2., cal_energy_max_log10=6., threshold_interpolation="logit")
            for branch in ("fixed_tel7", "exact_one") for e in (2.5, 3.5, 5.5)
        ]).to_csv(root / relative["pid_cut_path"], index=False)
        expected = {k: hashlib.sha256((root / v).read_bytes()).hexdigest() for k, v in relative.items()}
        config = dict(relative, zenith_deg=20, telescope_id=7, apply_pid=True, expected_sha256=expected)
        explicit = module.MonoReconstructor(None, config, model_dir=root)
        nested = module.MonoReconstructor(None, json.dumps(dict(config, model_dir=str(root))))
        absolute = module.MonoReconstructor(None, {**config, **{k: str(root / v) for k, v in relative.items()}},
                                             model_dir=root / "unused")
        precedence = module.MonoReconstructor(None, dict(config, model_dir=str(root / "unused")), model_dir=root)
        assert config["bundle_path"] == relative["bundle_path"]  # caller configuration is not mutated
        for loaded in (explicit, nested, absolute, precedence):
            assert list(loaded.bundle["features"]) == module.FEATURES
            assert loaded.config["expected_sha256"] == expected
            for key, value in relative.items():
                assert Path(loaded.config[key]) == root / value
                assert loaded.model_metadata[key]["sha256"] == expected[key]
        broken = dict(config, expected_sha256={**expected, "proxy_path": "0" * 64})
        try:
            module.MonoReconstructor(None, broken, model_dir=root)
        except ValueError as error:
            assert "SHA256 mismatch" in str(error)
        else:
            raise AssertionError("Changed portable asset SHA accepted")


def processor_contract(helper):
    """Doubles verify optional dispatch only, not the native stereo algorithm."""
    calls = []
    class CProcessor:
        def __init__(self, subarray, config):
            self.config = config
            calls.append(("C_init", config))
        def __call__(self, event):
            calls.append(("C_event", event))
    class Mono:
        def __init__(self, subarray, config):
            self.config = config
            calls.append(("Mono_init", config))
        def __call__(self, event):
            calls.append(("Mono_event", event))
            return "fixed"
    class Hybrid:
        def __init__(self, subarray, stereo, mono, **kwargs):
            self.mono, self.stereo = mono, stereo
            calls.append(("Hybrid_init", kwargs))
        def __call__(self, event, run_stereo=True):
            assert run_stereo is False
            calls.append(("Hybrid_event", event))
            return "union"
    helper.ShowerProcessor = CProcessor
    package = sys.modules["pylast.reco"]
    names = ("MonoReconstructor", "HybridReconstructor")
    submodules = {name: importlib.import_module("pylast.reco." + name) for name in names}
    original = {name: getattr(submodules[name], name) for name in names}
    package_original = {name: getattr(package, name, None) for name in names}
    # Submodule-first imports leave package attributes as modules, not classes.
    # Patch the submodule classes, keeping exactly that problematic package state.
    for name, reconstructor in zip(names, (Mono, Hybrid)):
        setattr(package, name, submodules[name])
        setattr(submodules[name], name, reconstructor)
        assert isinstance(getattr(package, name), ModuleType)
    try:
        spec = importlib.util.spec_from_file_location("pylast.reco.ShowerProcessor", RECO_SOURCE / "ShowerProcessor.py")
        wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrapper)
        raw = ' { "GeometryReconstructionTypes": ["HillasReconstructor"], "unchanged": 3 } '
        default = wrapper.ShowerProcessor(None, raw)
        default("event")
        assert calls == [("C_init", raw), ("C_event", "event")]
        shared = {"bundle_path": "z20/reconstruction.joblib", "model_dir": "portable"}
        for stages in (["MonoReconstructor"],
                       ["HillasReconstructor", "HybridReconstructor"],
                       ["HillasReconstructor", "EnergyRegressor", "ParticleClassifier",
                        "MonoReconstructor", "HybridReconstructor"]):
            calls.clear()
            config = dict(GeometryReconstructionTypes=stages, MonoReconstructor=shared,
                          HybridReconstructor={}, HillasReconstructor={"ImageQuery": "unchanged_query"})
            processor = wrapper.ShowerProcessor(None, config)
            processor("event")
            c_names = [name for name in stages if name not in wrapper.ShowerProcessor.PY_RECONSTRUCTORS]
            assert sum(name == "C_event" for name, _ in calls) == bool(c_names)
            assert sum(name == "Mono_event" for name, _ in calls) == ("MonoReconstructor" in stages)
            assert sum(name == "Hybrid_event" for name, _ in calls) == ("HybridReconstructor" in stages)
            if c_names:
                native = json.loads(processor.c_shower_processor.config)
                assert native["GeometryReconstructionTypes"] == c_names
                assert native["HillasReconstructor"]["ImageQuery"] == "unchanged_query"
                assert "MonoReconstructor" not in native and "HybridReconstructor" not in native
            if processor.hybrid_reconstructor:
                assert processor.hybrid_reconstructor.mono is processor.mono_reconstructor
                assert processor.last_hybrid_result == "union"
            if "MonoReconstructor" in stages:
                assert processor.last_mono_result == "fixed"
        calls.clear()
        wrapped = wrapper.ShowerProcessor(None, json.dumps({"ShowerProcessor": config}))
        assert json.loads(wrapped.c_shower_processor.config)["ShowerProcessor"]["GeometryReconstructionTypes"] == c_names
        for broken in ({"GeometryReconstructionTypes": ["MonoReconstructor"]},
                       {"GeometryReconstructionTypes": ["HybridReconstructor"], "MonoReconstructor": shared},
                       {"GeometryReconstructionTypes": ["MonoReconstructor", "MonoReconstructor"],
                        "MonoReconstructor": shared}):
            try:
                wrapper.ShowerProcessor(None, broken)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid optional stage configuration accepted")
    finally:
        for name, value in original.items():
            setattr(submodules[name], name, value)
        for name, value in package_original.items():
            if value is None:
                delattr(package, name)
            else:
                setattr(package, name, value)


def lazy_stereo_import_contract():
    """A clean process deliberately refuses optional joblib/pandas imports."""
    program = '''
import builtins, importlib.util, pathlib, sys
from types import ModuleType
root = pathlib.Path(sys.argv[1])
package = ModuleType("pylast"); package.__path__ = [str(root.parent)]
sys.modules["pylast"] = package
helper = ModuleType("pylast.helper")
class CProcessor:
    def __init__(self, subarray, config): self.config = config
    def __call__(self, event): return None
helper.ShowerProcessor = CProcessor
sys.modules["pylast.helper"] = helper
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split(".")[0] in {"joblib", "pandas"}:
        raise ImportError("optional dependency deliberately unavailable")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
spec = importlib.util.spec_from_file_location("pylast.reco", root / "__init__.py",
                                            submodule_search_locations=[str(root)])
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert "pylast.reco.MonoReconstructor" not in sys.modules
assert "pylast.reco.HybridReconstructor" not in sys.modules
processor = module.ShowerProcessor(None)
processor("event")
assert processor.c_shower_processor.config is None
builtins.__import__ = original
for name in ("AltAzFrame", "Point2D", "ReconstructedEnergy", "ReconstructedGeometry", "TelescopeFrame"):
    setattr(helper, name, type(name, (), {}))
from pylast.reco import MonoReconstructor, HybridReconstructor
assert isinstance(MonoReconstructor, type) and isinstance(HybridReconstructor, type)
assert module.MonoReconstructor is MonoReconstructor
assert module.HybridReconstructor is HybridReconstructor
'''
    subprocess.run([sys.executable, "-B", "-c", program, str(RECO_SOURCE)], check=True)

def main(native=False):
    lazy_stereo_import_contract()
    helper, module, hybrid_module = load_modules(native)
    portable_assets_contract(module)
    if not native:
        processor_contract(helper)
    # Coordinate tests use the actual pyLAST helpers with --native.
    for zenith in (20, 60):
        for azimuth in (0., .7, 5.4):
            altitude = math.radians(90 - zenith)
            for psi in (.3, .3 + math.pi):
                x, y, axis = module.nominal_hillas(.012, -.009, psi,
                                                  azimuth, altitude, azimuth, altitude)
                assert np.allclose([x, y], [.012, -.009], atol=2e-14)
                assert abs(math.cos(axis - psi)) > 1 - 1e-13
                alt, az = module.disp_to_sky(x, y, axis, .4, azimuth, altitude)
                point = helper.AltAzFrame().transform_to(
                    SimpleNamespace(azimuth=az, altitude=alt) if not native else
                    helper.SphericalRepresentation(azimuth=az, altitude=alt),
                    helper.TelescopeFrame(azimuth=azimuth, altitude=altitude))
                expected = [x + math.radians(.4) * math.cos(axis),
                            y + math.radians(.4) * math.sin(axis)]
                assert np.allclose([point.x, point.y], expected, atol=3e-14)
    mono = module.MonoReconstructor.__new__(module.MonoReconstructor)
    mono.subarray = None
    mono.zenith_deg, mono.telescope_id = 20, 7
    mono.use_fake_hillas, mono.apply_quality = True, True
    mono.apply_pid, mono.pid_mode = False, "fixed_tel7"
    mono.pid_bundle = mono.pid_cuts = None
    mono.geometry_name, mono.energy_name = "MonoDISPReconstructor", "MonoEnergyRegressor"
    mono.bundle = {"features": module.FEATURES, "magnitude": ConstantHead(.4),
                   "sign": ConstantHead(True), "energy": ConstantHead(3.5)}
    mono.proxy = ConstantHead(-1.)
    mono.threshold_energy, mono.threshold_error = np.array([2., 4., 6.]), np.array([-.8, -.8, -.8])
    mono.last_result = None
    h = SimpleNamespace(intensity=1000., length=math.radians(.2), width=math.radians(.05),
                        x=.012, y=-.009, r=.015, psi=.3, skewness=.8, kurtosis=3.1)
    p = SimpleNamespace(hillas=h, leakage=SimpleNamespace(intensity_width_2=.05, pixels_width_2=.04),
                        morphology=SimpleNamespace(n_pixels=40., n_islands=1.),
                        concentration=SimpleNamespace(concentration_cog=.2, concentration_core=.4, concentration_pixel=.1),
                        intensity=SimpleNamespace(intensity_max=100., intensity_std=35., intensity_skewness=.6, intensity_kurtosis=3.))
    cameras = {i: SimpleNamespace(image_parameters=p) for i in range(36)}
    pointing = SimpleNamespace(array_azimuth=.7, array_altitude=math.radians(70),
                               tels={i: SimpleNamespace(azimuth=.7, altitude=math.radians(70)) for i in range(36)})
    event = SimpleNamespace(run_id=1, event_id=3, pointing=pointing,
                            simulation=SimpleNamespace(tels=cameras, triggered_tels=[7]), dl2=FakeDL2())
    event.ensure_dl2 = lambda: event.dl2
    rows = mono.event_table(event, [7])
    features = module.feature_table(rows)
    assert list(features) == module.FEATURES and features.shape == (1, 17)
    assert all(dtype == np.float32 for dtype in features.dtypes)
    assert features.skewness.iloc[0] == np.float32(.8)
    with_truth = rows.assign(true_alt=0., true_az=0., energy_GeV=1., true_impact_distance=999.)
    assert mono.predict_table(rows).equals(mono.predict_table(with_truth))
    result = mono(event)
    assert result.accepted.iloc[0]
    assert not result.pid_valid.iloc[0] and not result.pass_pid.iloc[0]
    assert math.isclose(event.dl2.energy[mono.energy_name].estimate_energy, 10**3.5 / 1000.)
    assert math.isnan(event.dl2.geometry[mono.geometry_name].core_x)
    mono_geometry = event.dl2.geometry[mono.geometry_name]
    mono_energy = event.dl2.energy[mono.energy_name]
    old = helper.ReconstructedGeometry()
    old.is_valid, old.alt, old.az = True, 1.21, .73
    old.set_telescopes([7, 9])
    old_energy = helper.ReconstructedEnergy()
    old_energy.energy_valid, old_energy.estimate_energy = True, 4.56
    event.dl2.add_geometry("HillasReconstructor", old)
    event.dl2.add_energy("EnergyRegressor", old_energy)
    hybrid = hybrid_module.HybridReconstructor(None, lambda e: None, mono)
    event.simulation.triggered_tels = [9]
    route = hybrid(event)
    assert route["branch"] == "mono_exact_one" and route["accepted"]
    assert event.dl2.geometry[mono.geometry_name] is mono_geometry
    assert event.dl2.energy[mono.energy_name] is mono_energy
    assert event.dl2.geometry["HillasReconstructor"].alt == 1.21
    event.simulation.triggered_tels = [7, 9]
    route = hybrid(event)
    assert route["branch"] == "stereo" and event.dl2.geometry["HybridDirection"].alt == 1.21
    assert event.dl2.energy["HybridEnergy"].estimate_energy == 4.56
    old.is_valid = False
    route = hybrid(event)
    assert route["branch"] == "rejected" and not event.dl2.geometry["HybridDirection"].is_valid
    event.simulation.triggered_tels = [7]
    mono.proxy = ConstantHead(-.4)
    rejected_quality = mono(event)
    assert not rejected_quality.accepted.iloc[0]
    assert event.dl2.geometry[mono.geometry_name].is_valid
    assert event.dl2.energy[mono.energy_name].energy_valid
    route = hybrid(event)
    assert not route["accepted"] and not event.dl2.geometry["HybridDirection"].is_valid
    mono.proxy = ConstantHead(-1.)
    pid = ConstantPID()
    pid.features = module.FEATURES
    mono.pid_bundle = {"model": pid, "features": module.PID_FEATURES, "gamma_label": 1}
    mono.pid_cuts = pd.DataFrame([
        dict(branch=branch, energy_reco_log10=e, threshold_gamma_score=threshold,
             cal_energy_min_log10=2., cal_energy_max_log10=6., threshold_interpolation="logit")
        for branch, threshold in [("fixed_tel7", .7), ("exact_one", .9)] for e in (2.5, 3.5, 5.5)])
    mono.apply_pid = True
    fixed = mono(event)
    assert fixed.pid_valid.iloc[0] and fixed.pass_pid.iloc[0] and fixed.accepted.iloc[0]
    route = hybrid(event)
    assert route["mono"].pid_mode.iloc[0] == "exact_one"
    assert route["mono"].pid_valid.iloc[0] and not route["accepted"]
    mono.bundle["energy"] = ConstantHead(6.)
    outside = mono(event)
    assert not outside.pid_valid.iloc[0] and not outside.pass_pid.iloc[0]
    # A value just below max rounds to the frozen calibration's float32 max;
    # it must be rejected without changing the reconstructed energy itself.
    below_max = np.nextafter(6., -np.inf)
    mono.bundle["energy"] = ConstantHead(below_max)
    boundary = mono(event)
    assert boundary.energy_reco_log10.iloc[0] == below_max
    assert not boundary.pid_valid.iloc[0] and not boundary.pass_pid.iloc[0]
    # Empty-camera input retains zero rows, including the exporter's object dtype.
    event.simulation.tels = {}
    event.simulation.triggered_tels = []
    empty_rows = mono.event_table(event)
    assert empty_rows.empty
    for rows in (empty_rows, pd.DataFrame()):
        valid = mono.valid_images(rows)
        assert valid.shape == (0,) and valid.dtype == np.bool_
        result = mono.predict_table(rows)
        assert result.empty and len(result) == 0
        assert {"accepted", "reconstruction_valid", "pid_valid", "pass_pid"} <= set(result)
        assert result.accepted.dtype == np.bool_
    for cameras_present in (False, True):
        event.simulation.tels = cameras if cameras_present else {}
        route = hybrid(event, run_stereo=False)
        assert route["n_guard_tel"] == 0 and route["branch"] == "rejected"
        assert route["mono"] is None and not route["accepted"]
        assert not event.dl2.geometry[hybrid.geometry_name].is_valid
        assert not event.dl2.energy[hybrid.energy_name].energy_valid
    print("PASS: 17 pure features, truth independence, coordinates/skew axis, GeV-to-TeV, exact-one union, stereo preservation")
    print("PASS: portable relative/absolute/JSON assets, SHA refusal, shared optional stages, lazy stereo imports")
    print("PASS: zero-row prediction and empty-camera/zero-guard hybrid rejection without fallback")
    print("coordinate_backend=" + ("native_pyLAST" if native else "C++-equation doubles; native verification still required"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", action="store_true")
    main(parser.parse_args().native)
