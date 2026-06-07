"""Lightweight public build-plan composition for SAMMD MVP workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sammd.config import SAMMDConfig, load_config, load_config_dict
from sammd.io import (
    AtomRecord,
    OutputPaths,
    plan_output_paths,
    safe_write_text,
    slab_to_atom_records,
    write_mmcif,
)
from sammd.sam import SAMPlacementPlan, plan_sam_placements
from sammd.solvation import SolutionPlan, plan_solution_composition
from sammd.surfaces import BindingSite, SurfaceSlab, generate_binding_sites, plan_pd111_slab

DEFAULT_SOLVENT_PADDING_NM = 3.0
SLAB_CUTOFF_BUFFER_NM = 0.5
OPENMM_CONSTRUCTION_IMPLEMENTED = False


@dataclass(frozen=True)
class CompositionPlanningBox:
    """Approximate MVP volume used only for solution composition counts.

    These dimensions are not final simulation cell vectors. The volume includes the
    commensurate slab thickness plus solvent padding and intentionally omits SAM and
    packed molecule extents until full backend construction is implemented.
    """

    lateral_size_nm: tuple[float, float]
    slab_thickness_nm: float
    solvent_padding_nm: float
    approximate_dimensions_nm: tuple[float, float, float]
    count_planning_volume_nm3: float


@dataclass(frozen=True)
class SAMMDBuildPlan:
    """Lightweight system build artifact assembled from deterministic planners."""

    config: SAMMDConfig
    slab: SurfaceSlab
    binding_sites: tuple[BindingSite, ...]
    sam_placements: SAMPlacementPlan
    solution: SolutionPlan
    output_paths: OutputPaths
    composition_planning_box: CompositionPlanningBox
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

    def write_topology_cif(
        self, path: str | Path | None = None, *, overwrite: bool = False
    ) -> Path:
        """Write the configured topology CIF for inspecting the built plan.

        Parameters
        ----------
        path
            Optional destination path. Defaults to the configured ``topology.cif`` path.
        overwrite
            Whether an existing destination may be replaced.

        Returns
        -------
        Path
            Written topology CIF path.
        """

        if path is None and self.output_paths.topology is None:
            msg = "topology output path is not configured"
            raise ValueError(msg)
        destination = self.output_paths.topology if path is None else Path(path)
        assert destination is not None
        return write_mmcif(
            destination,
            _topology_atom_records(self),
            data_name="sammd_topology",
            cell_lengths_nm=self.composition_planning_box.approximate_dimensions_nm,
            overwrite=overwrite,
        )

    def build_summary(self) -> dict[str, object]:
        """Return a machine-readable summary of the planned system build."""

        return {
            "experiment": self.config.experiment.model_dump(mode="json"),
            "surface": {
                "metal": self.slab.metal,
                "facet": self.slab.facet,
                "requested_lateral_size_nm": list(self.slab.requested_lateral_size_nm),
                "effective_lateral_size_nm": list(self.slab.lateral_size_nm),
                "automatic_layers": self.slab.layers,
                "metal_atoms": len(self.slab.positions_nm),
            },
            "sam": {
                "grafting_density_nm2_per_molecule": self.config.sam.grafting_density,
                "molecules_total": len(self.sam_placements.placements),
                "molecules_per_side": self.sam_placements.selected_sites_per_side,
                "components": [
                    component.model_dump(mode="json") for component in self.config.sam.components
                ],
            },
            "solution": {
                "count_planning_volume_nm3": self.solution.box_volume_nm3,
                "molecule_counts": self.solution.molecule_counts,
                "warnings": list(self.solution.warnings),
            },
            "parameterization": self.config.parameterization.model_dump(mode="json"),
            "outputs": {
                key: str(value) if value is not None else None
                for key, value in self.output_paths.__dict__.items()
            },
            "full_construction_available": self.full_construction_available,
        }

    def write_build_summary(
        self, path: str | Path | None = None, *, overwrite: bool = False
    ) -> Path:
        """Write the JSON build summary artifact."""

        if path is None and self.output_paths.build_summary is None:
            msg = "build summary output path is not configured"
            raise ValueError(msg)
        destination = self.output_paths.build_summary if path is None else Path(path)
        assert destination is not None
        return safe_write_text(
            destination,
            json.dumps(self.build_summary(), indent=2, ensure_ascii=False) + "\n",
            overwrite=overwrite,
        )

    def write_resolved_config(
        self, path: str | Path | None = None, *, overwrite: bool = False
    ) -> Path:
        """Write the validated YAML config used for the build."""

        if path is None and self.output_paths.resolved_config is None:
            msg = "resolved config output path is not configured"
            raise ValueError(msg)
        destination = self.output_paths.resolved_config if path is None else Path(path)
        assert destination is not None
        return safe_write_text(
            destination,
            yaml.safe_dump(self.config.model_dump(mode="json"), sort_keys=False),
            overwrite=overwrite,
        )

    def write_planned_topology(self, *, overwrite: bool = False) -> Path:
        """Write the configured topology CIF for the deterministic build plan."""

        return self.write_topology_cif(overwrite=overwrite)


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
    active_seed = loaded_config.experiment.seed if seed is None else seed
    if active_seed < 0:
        msg = "seed must be non-negative"
        raise ValueError(msg)

    slab_layers = _auto_slab_layers(loaded_config)
    slab = plan_pd111_slab(
        loaded_config.surface.lateral_size,
        slab_layers,
    )
    binding_sites = generate_binding_sites(slab)
    composition_planning_box = _derive_composition_planning_box(loaded_config, slab)
    sam_placements = plan_sam_placements(
        loaded_config.sam,
        binding_sites,
        composition_planning_box.lateral_size_nm[0] * composition_planning_box.lateral_size_nm[1],
        seed=active_seed,
    )
    solution = plan_solution_composition(
        loaded_config, composition_planning_box.count_planning_volume_nm3
    )
    output_paths = plan_output_paths(
        loaded_config,
        loaded_config.outputs.directory if output_dir is None else output_dir,
    )

    return SAMMDBuildPlan(
        config=loaded_config,
        slab=slab,
        binding_sites=binding_sites,
        sam_placements=sam_placements,
        solution=solution,
        output_paths=output_paths,
        composition_planning_box=composition_planning_box,
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


def _topology_atom_records(plan: SAMMDBuildPlan) -> tuple[AtomRecord, ...]:
    """Return atom records for the current inspectable topology CIF."""

    records = list(slab_to_atom_records(plan.slab, chain_id="A"))
    serial = len(records) + 1
    for residue_id, placement in enumerate(plan.sam_placements.placements, start=1):
        records.append(
            AtomRecord(
                serial=serial,
                atom_name="S",
                element="S",
                residue_name=placement.component_residue_name,
                residue_id=residue_id,
                chain_id="B" if placement.side == "bottom" else "C",
                component_label=f"SAM {placement.component_name} {placement.side}",
                coordinates_nm=placement.position_nm,
            )
        )
        serial += 1
    return tuple(records)


def _auto_slab_layers(config: SAMMDConfig) -> int:
    """Choose a slab thick enough for the configured nonbonded cutoff."""

    from sammd.surfaces import get_fcc_surface_metadata

    metadata = get_fcc_surface_metadata(config.surface.metal, config.surface.facet)
    minimum_thickness_nm = config.parameterization.nonbonded_cutoff + SLAB_CUTOFF_BUFFER_NM
    layers = 2
    while metadata.slab_thickness_nm(layers) <= minimum_thickness_nm:
        layers += 1
    return layers


def _derive_composition_planning_box(
    config: SAMMDConfig, slab: SurfaceSlab
) -> CompositionPlanningBox:
    """Derive approximate dimensions used only for solution count planning."""

    padding_nm = getattr(config.solvent, "padding", DEFAULT_SOLVENT_PADDING_NM)
    box_z_nm = slab.slab_extent_nm[2] + 2.0 * padding_nm
    approximate_dimensions_nm = (
        slab.lateral_size_nm[0],
        slab.lateral_size_nm[1],
        box_z_nm,
    )
    count_planning_volume_nm3 = (
        approximate_dimensions_nm[0] * approximate_dimensions_nm[1] * approximate_dimensions_nm[2]
    )
    return CompositionPlanningBox(
        lateral_size_nm=slab.lateral_size_nm,
        slab_thickness_nm=slab.slab_extent_nm[2],
        solvent_padding_nm=padding_nm,
        approximate_dimensions_nm=approximate_dimensions_nm,
        count_planning_volume_nm3=count_planning_volume_nm3,
    )
