"""Pytest config: skip tests that need git-ignored data, caches or weights when those are absent."""

from pathlib import Path

import pytest

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
