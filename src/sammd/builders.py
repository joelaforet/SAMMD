"""Lightweight public build-plan composition for SAMMD MVP workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sammd.config import SAMMDConfig, load_config, load_config_dict
from sammd.io import OutputPaths, plan_output_paths, slab_to_atom_records, write_mmcif
from sammd.sam import SAMPlacementPlan, plan_sam_placements
from sammd.solvation import SolutionPlan, plan_solution_composition
from sammd.surfaces import BindingSite, SurfaceSlab, generate_binding_sites, plan_pd111_slab

DEFAULT_SOLVENT_PADDING_NM = 3.0
OPENMM_CONSTRUCTION_IMPLEMENTED = False


@dataclass(frozen=True)
class PlanningBox:
    """Derived MVP box dimensions used only for composition counts."""

    lateral_size_nm: tuple[float, float]
    slab_thickness_nm: float
    solvent_padding_nm: float
    dimensions_nm: tuple[float, float, float]
    volume_nm3: float


@dataclass(frozen=True)
class SAMMDBuildPlan:
    """Lightweight system build artifact assembled from deterministic planners."""

    config: SAMMDConfig
    slab: SurfaceSlab
    binding_sites: tuple[BindingSite, ...]
    sam_placements: SAMPlacementPlan
    solution: SolutionPlan
    output_paths: OutputPaths
    planning_box: PlanningBox
    openmm_construction_implemented: bool = OPENMM_CONSTRUCTION_IMPLEMENTED

    @property
    def full_construction_available(self) -> bool:
        """Return whether full backend construction is available in this milestone."""

        return self.openmm_construction_implemented

    def require_full_construction(self) -> None:
        """Raise a clear error for workflows that need OpenFF/OpenMM objects."""

        if self.openmm_construction_implemented:
            return
        msg = (
            "Full OpenFF/OpenMM construction is not implemented yet; this object is a "
            "lightweight deterministic build plan."
        )
        raise NotImplementedError(msg)

    def write_planned_slab_mmcif(
        self, path: str | Path | None = None, *, overwrite: bool = False
    ) -> Path:
        """Write the planned Pd slab as an mmCIF visualization artifact.

        Parameters
        ----------
        path
            Optional destination path. Defaults to the planned topology path.
        overwrite
            Whether an existing destination may be replaced.

        Returns
        -------
        Path
            Written mmCIF path.
        """

        destination = self.output_paths.topology if path is None else Path(path)
        return write_mmcif(
            destination,
            slab_to_atom_records(self.slab),
            data_name="sammd_planned_slab",
            cell_lengths_nm=self.planning_box.dimensions_nm,
            overwrite=overwrite,
        )

    def write_planned_topology(self, *, overwrite: bool = False) -> Path:
        """Write the current lightweight planned topology artifact."""

        return self.write_planned_slab_mmcif(overwrite=overwrite)


def build_system(
    config: SAMMDConfig | str | Path | dict[str, Any],
    output_dir: str | Path | None = None,
    seed: int | None = None,
) -> SAMMDBuildPlan:
    """Compose a deterministic lightweight SAMMD build plan.

    Parameters
    ----------
    config
        Validated config, path to a YAML config, or parsed config mapping.
    output_dir
        Base directory for resolving planned output artifact paths.
    seed
        Optional override for deterministic SAM site and component choices.

    Returns
    -------
    SAMMDBuildPlan
        Lightweight artifact containing slab, SAM, solution, and output plans.
    """

    loaded_config = _load_build_config(config)
    active_seed = loaded_config.simulation.seed if seed is None else seed
    if active_seed < 0:
        msg = "seed must be non-negative"
        raise ValueError(msg)

    slab = plan_pd111_slab(
        loaded_config.surface.slab.lateral_size_nm,
        loaded_config.surface.slab.layers,
        centered=loaded_config.surface.slab.centered,
    )
    binding_sites = generate_binding_sites(slab, loaded_config.sam.anchor.site)
    planning_box = _derive_planning_box(loaded_config, slab)
    sam_placements = plan_sam_placements(
        loaded_config.sam,
        binding_sites,
        planning_box.lateral_size_nm[0] * planning_box.lateral_size_nm[1],
        seed=active_seed,
    )
    solution = plan_solution_composition(loaded_config, planning_box.volume_nm3)
    output_paths = plan_output_paths(loaded_config, "." if output_dir is None else output_dir)

    return SAMMDBuildPlan(
        config=loaded_config,
        slab=slab,
        binding_sites=binding_sites,
        sam_placements=sam_placements,
        solution=solution,
        output_paths=output_paths,
        planning_box=planning_box,
    )


def _load_build_config(config: SAMMDConfig | str | Path | dict[str, Any]) -> SAMMDConfig:
    """Normalize supported build inputs into a validated configuration object."""

    if isinstance(config, SAMMDConfig):
        return config
    if isinstance(config, str | Path):
        return load_config(config)
    if isinstance(config, dict):
        return load_config_dict(config)
    msg = "config must be a SAMMDConfig, path, or configuration mapping"
    raise TypeError(msg)


def _derive_planning_box(config: SAMMDConfig, slab: SurfaceSlab) -> PlanningBox:
    """Derive the MVP box used for count planning from adjusted slab dimensions."""

    padding_nm = getattr(config.solvent, "padding_nm", DEFAULT_SOLVENT_PADDING_NM)
    box_z_nm = slab.slab_extent_nm[2] + 2.0 * padding_nm
    dimensions_nm = (
        slab.lateral_size_nm[0],
        slab.lateral_size_nm[1],
        box_z_nm,
    )
    volume_nm3 = dimensions_nm[0] * dimensions_nm[1] * dimensions_nm[2]
    return PlanningBox(
        lateral_size_nm=slab.lateral_size_nm,
        slab_thickness_nm=slab.slab_extent_nm[2],
        solvent_padding_nm=padding_nm,
        dimensions_nm=dimensions_nm,
        volume_nm3=volume_nm3,
    )
