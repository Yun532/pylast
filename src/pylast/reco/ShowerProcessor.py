"""Original C++ stereo processing, with explicit optional mono/hybrid stages."""
from ..helper import ShowerProcessor as CShowerProcessor
import json


class ShowerProcessor:
    C_RECONSTRUCTORS = ["HillasReconstructor", "EnergyRegressor", "ParticleClassifier"]
    PY_RECONSTRUCTORS = ["MonoReconstructor", "HybridReconstructor"]

    def __init__(self, subarray, config_str=None):
        self.subarray = subarray
        self.c_reconstructor_config = {}
        self.py_reconstructor_configs = {}
        self.c_shower_processor = None
        self.mono_reconstructor = None
        self.hybrid_reconstructor = None
        self.last_mono_result = None
        self.last_hybrid_result = None

        config = json.loads(config_str) if isinstance(config_str, str) else dict(config_str or {})
        section = config.get("ShowerProcessor", config)
        names = section.get("GeometryReconstructionTypes", [])
        python_names = [name for name in names if name in self.PY_RECONSTRUCTORS]
        if not python_names:
            # Preserve default stereo construction, including the original JSON string/None.
            native_config = json.dumps(config_str) if isinstance(config_str, dict) else config_str
            self.c_shower_processor = CShowerProcessor(subarray, native_config)
            return
        if len(set(python_names)) != len(python_names):
            raise ValueError("Duplicate Python reconstructor stage")
        if "MonoReconstructor" not in section:
            raise ValueError("Mono/hybrid stages require a shared MonoReconstructor configuration")

        c_names = [name for name in names if name not in self.PY_RECONSTRUCTORS]
        if "HybridReconstructor" in python_names and "HillasReconstructor" not in c_names:
            raise ValueError("HybridReconstructor requires the original C++ HillasReconstructor")
        self.py_reconstructor_configs = {name: section.get(name, {}) for name in python_names}
        self.c_reconstructor_config = {
            key: value for key, value in section.items() if key not in self.PY_RECONSTRUCTORS
        }
        self.c_reconstructor_config["GeometryReconstructionTypes"] = c_names
        if c_names:
            native_config = dict(config)
            if "ShowerProcessor" in config:
                native_config["ShowerProcessor"] = self.c_reconstructor_config
            else:
                native_config = self.c_reconstructor_config
            self.c_shower_processor = CShowerProcessor(subarray, json.dumps(native_config))

        # Optional model dependencies are imported only for explicitly requested stages.
        from .MonoReconstructor import MonoReconstructor
        self.mono_reconstructor = MonoReconstructor(subarray, section["MonoReconstructor"])
        if "HybridReconstructor" in python_names:
            from .HybridReconstructor import HybridReconstructor
            self.hybrid_reconstructor = HybridReconstructor(
                subarray, self.c_shower_processor, self.mono_reconstructor,
                **section.get("HybridReconstructor", {}),
            )

    def __call__(self, event):
        """One C++ stereo pass, then optional fixed-mono and exact-one routing."""
        if self.c_shower_processor is not None:
            self.c_shower_processor(event)
        if "MonoReconstructor" in self.py_reconstructor_configs:
            self.last_mono_result = self.mono_reconstructor(event)
        if self.hybrid_reconstructor is not None:
            # Hybrid's exact-one call uses store=False; any fixed-mono DL2 stays intact.
            self.last_hybrid_result = self.hybrid_reconstructor(event, run_stereo=False)
