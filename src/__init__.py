"""
Package initialization for defect inspection core modules.

Purpose: Exposes the data pipeline, training workflow, and inference utilities.
Inputs: None.
Outputs: Exported core submodules for data processing, training, and inference.
"""

from . import data
from . import inference
from . import train

__all__ = ["data", "train", "inference"]

