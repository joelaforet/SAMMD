"""Public API for SAMMD."""

from importlib.metadata import PackageNotFoundError, version

from sammd.builders import build_system
from sammd.config import SAMMDConfig, load_config, load_config_dict
from sammd.sam import SAMPlacement, SAMPlacementPlan, plan_sam_placements
from sammd.surfaces import BindingSite, SurfaceSlab, generate_binding_sites, plan_pd111_slab

try:
    __version__ = version("sammd")
except PackageNotFoundError:  # pragma: no cover - editable source tree fallback
    __version__ = "0.0.0"

__all__ = [
    "BindingSite",
    "SAMMDConfig",
    "SAMPlacement",
    "SAMPlacementPlan",
    "SurfaceSlab",
    "__version__",
    "build_system",
    "generate_binding_sites",
    "load_config",
    "load_config_dict",
    "plan_pd111_slab",
    "plan_sam_placements",
]
