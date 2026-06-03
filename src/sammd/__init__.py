"""Public API for SAMMD."""

from importlib.metadata import PackageNotFoundError, version

from sammd.builders import build_system
from sammd.config import SAMMDConfig, load_config, load_config_dict

try:
    __version__ = version("sammd")
except PackageNotFoundError:  # pragma: no cover - editable source tree fallback
    __version__ = "0.0.0"

__all__ = [
    "SAMMDConfig",
    "__version__",
    "build_system",
    "load_config",
    "load_config_dict",
]
