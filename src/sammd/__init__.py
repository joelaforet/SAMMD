"""Public API for SAMMD."""

from importlib.metadata import PackageNotFoundError, version

from sammd.builders import build_system
from sammd.config import SAMMDConfig, load_config, load_config_dict
from sammd.openmm_runtime import (
    add_position_restraints,
    add_sulfur_metal_lj_scaling,
    create_langevin_integrator,
    create_openmm_simulation,
    require_openmm,
)

try:
    __version__ = version("sammd")
except PackageNotFoundError:  # pragma: no cover - editable source tree fallback
    __version__ = "0.0.0"

__all__ = [
    "SAMMDConfig",
    "__version__",
    "add_position_restraints",
    "add_sulfur_metal_lj_scaling",
    "build_system",
    "create_langevin_integrator",
    "create_openmm_simulation",
    "load_config",
    "load_config_dict",
    "require_openmm",
]
