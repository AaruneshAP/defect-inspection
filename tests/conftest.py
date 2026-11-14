"""Pytest config: skip tests that need git-ignored data, caches or weights when those are absent."""

from pathlib import Path

import pytest

from src.inspection_log import missing_databricks_vars

REQUIREMENTS = {
    "requires_data": Path("data/yolo"),
    "requires_cache": Path("data/cache/val_detections.parquet"),
    "requires_weights": Path("models/best.pt"),
}


def pytest_collection_modifyitems(config, items):
    """Add a skip marker to tests whose required local files do not exist."""
    for item in items:
        for marker, path in REQUIREMENTS.items():
            if marker in item.keywords and not path.exists():
                item.add_marker(pytest.mark.skip(reason=f"{path} not available (git-ignored)"))
    missing = missing_databricks_vars() if any("requires_databricks" in i.keywords for i in items) else []
    for item in items:
        if "requires_databricks" in item.keywords and missing:
            item.add_marker(pytest.mark.skip(reason=f"Databricks variables not set: {', '.join(missing)}"))
