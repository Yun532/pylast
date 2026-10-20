"""Check measured and legacy CLI routing without loading compiled components."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class RealDataCliTest(unittest.TestCase):
    def test_real_and_legacy_pipeline_routing(self):
        for input_format, has_pointing in (("real", True), ("real", False), ("simtel", True), ("lact", True)):
            with self.subTest(input_format=input_format, pointing=has_pointing), tempfile.TemporaryDirectory() as directory:
                calls = []
                event = SimpleNamespace(pointing=object() if has_pointing else None)

                class Source:
                    subarray = object()
                    shower_array = object()
                    atmosphere_model = None

                    def __iter__(self):
                        return iter([event])

                source_factory = Mock(return_value=Source())
                calibrator = Mock(side_effect=lambda event: calls.append("calibrate"))
                image_processor = Mock(side_effect=lambda event: calls.append("image"))
                shower_processor = Mock(side_effect=lambda event: calls.append("shower"))
                writer = Mock(side_effect=lambda event: calls.append("write"))
                writer.close.side_effect = lambda: calls.append("close")
                calibration_factory = Mock(return_value=calibrator)
                shower_factory = Mock(return_value=shower_processor)
                writer_factory = Mock(return_value=writer)
                modules = {}
                for name in ("pylast", "pylast.helper", "pylast.io", "pylast.calib", "pylast.image", "pylast.reco"):
                    modules[name] = ModuleType(name)
                modules["pylast"].__path__ = []
                modules["pylast.helper"].register_exe = Mock()
                modules["pylast.helper"].Statistics = object
                for name in ("SimtelEventSource", "LactEventSource", "LactRealEventSource"):
                    setattr(modules["pylast.io"], name, source_factory)
                modules["pylast.io"].DataWriter = writer_factory
                modules["pylast.calib"].Calibrator = calibration_factory
                modules["pylast.calib"].LactRealCalibrator = calibration_factory
                modules["pylast.image"].ImageProcessor = Mock(return_value=image_processor)
                modules["pylast.reco"].ShowerProcessor = shower_factory
                config = {"shower_processor": {
                    "GeometryReconstructionTypes": ["HillasReconstructor"],
                    "HillasReconstructor": {"use_fake_hillas": True}},
                    "data_writer": {"write_simulated_camera": True}}
                config_path = Path(directory) / "config.json"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                args = SimpleNamespace(
                    input=["input.root"], output=["output.root"], config=str(config_path),
                    input_format=input_format, mapping_file="mapping.csv", calibration_file=None,
                    analysis_file="eventShow.root",
                    pointing_file=None, pointing_alt=70.0, pointing_az=12.0,
                    atmosphere_profile=None, subarray=None, plot_dir=None,
                )
                path = Path(__file__).parents[1] / "src/pylast/ulities/hillas_reco.py"
                with patch.dict(sys.modules, modules):
                    spec = importlib.util.spec_from_file_location("pylast.ulities.hillas_reco", path)
                    cli = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(cli)
                    self.assertEqual(cli.enhanced_hillas_reco(args), 0)
                    with self.assertRaisesRegex(ValueError, "unsupported|does not support"):
                        cli._real_shower_config({"GeometryReconstructionTypes": ["DispWeightedReconstructor"]})
                    mono_config = {"GeometryReconstructionTypes": ["MonoReconstructor", "HybridReconstructor"],
                                   "MonoReconstructor": {"use_fake_hillas": True},
                                   "HybridReconstructor": {"geometry_name": "HybridDirection"}}
                    measured = cli._real_shower_config(mono_config)
                    self.assertFalse(measured["MonoReconstructor"]["use_fake_hillas"])
                    self.assertEqual(measured["HybridReconstructor"], mono_config["HybridReconstructor"])
                    self.assertTrue(mono_config["MonoReconstructor"]["use_fake_hillas"])
                    mono_config["GeometryReconstructionTypes"] = ["HillasReconstructor", "HybridReconstructor"]
                    measured = cli._real_shower_config(mono_config)
                    self.assertFalse(measured["MonoReconstructor"]["use_fake_hillas"])
                    self.assertEqual(measured["HybridReconstructor"], mono_config["HybridReconstructor"])
                expected = ["image", "shower", "write", "close"]
                if input_format == "real" and not has_pointing:
                    expected.remove("shower")
                if input_format != "lact":
                    expected.insert(0, "calibrate")
                self.assertEqual(calls, expected)
                shower_config = json.loads(shower_factory.call_args.kwargs["config_str"])
                self.assertEqual(shower_config["HillasReconstructor"]["use_fake_hillas"], input_format != "real")
                if input_format == "real":
                    self.assertEqual(source_factory.call_args.args, ("input.root", "mapping.csv"))
                    self.assertEqual(source_factory.call_args.kwargs["pointing_alt"], 70.0)
                    self.assertEqual(source_factory.call_args.kwargs["analysis_file"], "eventShow.root")
                    writer_config = json.loads(writer_factory.call_args.kwargs["config_str"])
                    self.assertFalse(writer_config["write_simulated_camera"])
                    self.assertFalse(writer_config["write_simulation_config"])
                    self.assertTrue(writer_config["write_pointing"])
                    writer.write_all_simulation_shower.assert_not_called()
                else:
                    writer.write_all_simulation_shower.assert_called_once()


if __name__ == "__main__":
    unittest.main()
