from .ShowerProcessor import ShowerProcessor

__all__ = ["ShowerProcessor"]


def __getattr__(name):
    """Keep optional model libraries out of the normal stereo-only import path."""
    if name in ("MonoReconstructor", "HybridReconstructor"):
        from importlib import import_module
        value = getattr(import_module("." + name, __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
