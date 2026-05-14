import importlib.util
import sys
from pathlib import Path

import torch  # noqa: F401 - load libtorch before pybind extension modules.


def load_cpybin(module_name):
    canonical_name = f"cpp_to_py.cpybin.{module_name}"
    if canonical_name in sys.modules:
        return sys.modules[canonical_name]

    cpybin_dir = Path(__file__).resolve().parents[1] / "cpp_to_py" / "cpybin"
    matches = sorted(cpybin_dir.glob(f"{module_name}*.so"))
    if not matches:
        raise ImportError(f"Cannot find cpp_to_py/cpybin extension: {module_name}")

    spec = importlib.util.spec_from_file_location(canonical_name, matches[0])
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load cpp_to_py/cpybin extension: {matches[0]}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[canonical_name] = module
    spec.loader.exec_module(module)
    return module
