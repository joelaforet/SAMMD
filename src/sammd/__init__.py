"""Public API for SAMMD."""

from importlib.metadata import PackageNotFoundError, version

from sammd.core.builders import build_system
from sammd.core.config import SAMMDConfig, load_config, load_config_dict


def _register_interchange_plugins_if_available() -> None:
    """Register SAMMD Interchange plugins when optional OpenFF dependencies exist."""

    try:
        from sammd.backends.interchange_plugins import register_interchange_plugin_collection

        register_interchange_plugin_collection()
    except ImportError:
        # OpenFF Interchange is optional for the dependency-light public API.
        return

try:
    __version__ = version("sammd")
except PackageNotFoundError:  # pragma: no cover - editable source tree fallback
    __version__ = "0.1.0"

_register_interchange_plugins_if_available()

__all__ = [
    "SAMMDConfig",
    "__version__",
    "build_system",
    "load_config",
    "load_config_dict",
]
