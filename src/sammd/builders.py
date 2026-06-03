"""Lightweight public build API placeholders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sammd.config import SAMMDConfig, load_config, load_config_dict


def build_system(config: SAMMDConfig | str | Path | dict[str, Any]) -> None:
    """Validate input and report that backend system construction is deferred.

    Parameters
    ----------
    config
        Validated config, path to a YAML config, or parsed config mapping.

    Raises
    ------
    NotImplementedError
        Always raised for this scaffolding milestone after configuration validation.
    """

    if isinstance(config, SAMMDConfig):
        loaded_config = config
    elif isinstance(config, str | Path):
        loaded_config = load_config(config)
    elif isinstance(config, dict):
        loaded_config = load_config_dict(config)
    else:
        msg = "config must be a SAMMDConfig, path, or configuration mapping"
        raise TypeError(msg)

    msg = (
        "Full OpenFF/OpenMM system construction is not implemented in the SAMMD "
        f"scaffolding milestone for {loaded_config.surface.metal}({loaded_config.surface.facet})."
    )
    raise NotImplementedError(msg)
