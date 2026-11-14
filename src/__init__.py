"""
Package initialization for defect inspection core modules.

Purpose: Exposes the data pipeline, training workflow, and inference utilities.
Inputs: None.
Outputs: Exported core submodules for data processing, training, and inference.
"""

"""Package initialization for defect inspection core modules."""

__all__ = ["data", "train", "inference"]


def __getattr__(name: str):
    if name in ("data", "train", "inference"):
        import importlib

        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

