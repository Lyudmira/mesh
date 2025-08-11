"""Verify that required Python packages can be imported.

The test prints the resolved version for each package so success or
failure can be easily inspected when running ``pytest -s``.
"""

import importlib


def test_python_packages():
    modules = ["openvdb", "open3d", "pymeshlab", "igl"]
    for name in modules:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        print(f"{name} OK (version: {version})")
