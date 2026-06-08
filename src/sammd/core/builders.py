"""Lightweight public build-plan composition for SAMMD MVP workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sammd.core.config import SAMMDConfig, load_config, load_config_dict
from sammd.core.io import (
    AtomRecord,
    OutputPaths,
    plan_output_paths,
    safe_write_text,
    slab_to_atom_records,
    write_mmcif,
)
from sammd.model.metal_sulfur import default_metal_sulfur_interaction
from sammd.model.sam import SAMPlacementPlan, plan_sam_placements
from sammd.model.solvation import SolutionPlan, plan_solution_composition
from sammd.model.surfaces import BindingSite, SurfaceSlab, generate_binding_sites, plan_fcc111_slab

DEFAULT_SOLVENT_PADDING_NM = 3.0
DEFAULT_SAM_EXTENDED_LENGTH_NM = 0.95
SLAB_CUTOFF_BUFFER_NM = 0.5
OPENMM_CONSTRUCTION_IMPLEMENTED = False

ENGINE_EXPORT_PLAN: dict[str, dict[str, object]] = {
    "openmm": {
        "status": "reserved",
        "available": False,
        "handoff": "Interchange primary; system.xml convenience export",
        "teaching_scope": "student OpenMM Python API path",
    },
    "gromacs": {
        "status": "reserved",
        "available": False,
        "handoff": "future downstream export from Interchange",
        "teaching_scope": "not taught in beginner workflow",
    },
    "lammps": {
        "status": "reserved",
        "available": False,
        "handoff": "future downstream export from Interchange",
        "teaching_scope": "not taught in beginner workflow",
    },
    "amber": {
        "status": "reserved",
        "available": False,
        "handoff": "future downstream export from Interchange",
        "teaching_scope": "not taught in beginner workflow",
    },
}


@dataclass(frozen=True)
class SAMLengthEstimate:
    """Approximate fully extended SAM length used for deterministic box planning."""

    component_name: str
    residue_name: str
    smiles: str
    configured_length_nm: float | None
    heuristic_length_nm: float
    length_nm: float
    source: str


@dataclass(frozen=True)
class BoxPlan:
    """Unified orthorhombic build box for counts and topology metadata."""

    lateral_size_nm: tuple[float, float]
    dimensions_nm: tuple[float, float, float]
    bounds_nm: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    volume_nm3: float
    solvent_padding_nm: float
    sam_extended_length_nm: float
    slab_center_nm: tuple[float, float, float]
    sam_length_estimates: tuple[SAMLengthEstimate, ...]


@dataclass(frozen=True)
class SAMMDBuildPlan:
    """Lightweight system build artifact assembled from deterministic planners."""

    config: SAMMDConfig
    slab: SurfaceSlab
    binding_sites: tuple[BindingSite, ...]
    sam_placements: SAMPlacementPlan
    solution: SolutionPlan
    output_paths: OutputPaths
    box_plan: BoxPlan
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
            cell_lengths_nm=self.box_plan.dimensions_nm,
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
                "metal_sulfur_interaction": default_metal_sulfur_interaction().to_summary(),
                "placements": [
                    {
                        "component_name": placement.component_name,
                        "residue_name": placement.component_residue_name,
                        "side": placement.side,
                        "site_kind": placement.site_kind,
                        "site_position_nm": list(placement.anchor_pose.site_position_nm),
                        "sulfur_position_nm": list(placement.anchor_pose.sulfur_position_nm),
                        "normal": list(placement.anchor_pose.normal),
                        "axis_direction": list(placement.anchor_pose.axis_direction),
                        "azimuth_rad": placement.anchor_pose.azimuth_rad,
                        "sulfur_height_nm": placement.anchor_pose.sulfur_height_nm,
                        "nearest_metal_atom_indices": list(
                            placement.anchor_pose.nearest_metal_atom_indices
                        ),
                        "attachment_mode": placement.anchor_pose.attachment_mode,
                        "metal_sulfur_interaction": (
                            placement.anchor_pose.metal_sulfur_interaction.to_summary()
                        ),
                    }
                    for placement in self.sam_placements.placements
                ],
                "components": [
                    component.model_dump(mode="json") for component in self.config.sam.components
                ],
            },
            "box": {
                "lateral_size_nm": list(self.box_plan.lateral_size_nm),
                "dimensions_nm": list(self.box_plan.dimensions_nm),
                "bounds_nm": [list(bounds) for bounds in self.box_plan.bounds_nm],
                "volume_nm3": self.box_plan.volume_nm3,
                "solvent_padding_nm": self.box_plan.solvent_padding_nm,
                "sam_extended_length_nm": self.box_plan.sam_extended_length_nm,
                "slab_center_nm": list(self.box_plan.slab_center_nm),
                "sam_length_estimates": [
                    {
                        "component_name": estimate.component_name,
                        "residue_name": estimate.residue_name,
                        "smiles": estimate.smiles,
                        "configured_length_nm": estimate.configured_length_nm,
                        "heuristic_length_nm": estimate.heuristic_length_nm,
                        "length_nm": estimate.length_nm,
                        "source": estimate.source,
                    }
                    for estimate in self.box_plan.sam_length_estimates
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
            "artifacts": {
                "topology": self._artifact_summary("topology", "current"),
                "build_summary": self._artifact_summary("build_summary", "current"),
                "resolved_config": self._artifact_summary("resolved_config", "current"),
                "positions": self._artifact_summary("positions", "reserved"),
                "openff_interchange": self._artifact_summary(
                    "openff_interchange", "reserved"
                ),
                "openmm_system": self._artifact_summary("openmm_system", "reserved"),
                "anchor_metadata": self._artifact_summary("anchor_metadata", "reserved"),
            },
            "engine_exports": ENGINE_EXPORT_PLAN,
            "full_construction_available": self.full_construction_available,
        }

    def _artifact_summary(self, key: str, status: str) -> dict[str, object]:
        """Return path and lightweight-release availability for one artifact."""

        path = getattr(self.output_paths, key)
        summary: dict[str, object] = {
            "path": str(path) if path is not None else None,
            "status": status,
            "available": status == "current",
        }
        if key == "openff_interchange":
            summary.update(
                {
                    "constructed": False,
                    "format": "json",
                    "save_method": "Interchange.model_dump_json",
                    "load_method": "Interchange.model_validate_json",
                    "openff_interchange_package_version": None,
                    "compatibility_caveat": (
                        "Pre-1.0 OpenFF Interchange JSON compatibility is not "
                        "guaranteed across versions."
                    ),
                }
            )
        return summary

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
    slab = plan_fcc111_slab(
        loaded_config.surface.metal,
        loaded_config.surface.lateral_size,
        slab_layers,
    )
    binding_sites = generate_binding_sites(slab)
    box_plan = _derive_box_plan(loaded_config, slab)
    sam_placements = plan_sam_placements(
        loaded_config.sam,
        binding_sites,
        box_plan.lateral_size_nm[0] * box_plan.lateral_size_nm[1],
        seed=active_seed,
    )
    solution = plan_solution_composition(loaded_config, box_plan.volume_nm3)
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
        box_plan=box_plan,
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
                coordinates_nm=placement.anchor_pose.sulfur_position_nm,
            )
        )
        serial += 1
    return tuple(records)


def _auto_slab_layers(config: SAMMDConfig) -> int:
    """Choose a slab thick enough for the configured nonbonded cutoff."""

    from sammd.model.surfaces import get_fcc_surface_metadata

    metadata = get_fcc_surface_metadata(config.surface.metal, config.surface.facet)
    minimum_thickness_nm = config.parameterization.nonbonded_cutoff + SLAB_CUTOFF_BUFFER_NM
    layers = 2
    while metadata.slab_thickness_nm(layers) <= minimum_thickness_nm:
        layers += 1
    return layers


def _derive_box_plan(config: SAMMDConfig, slab: SurfaceSlab) -> BoxPlan:
    """Derive the single deterministic box used for counts and metadata."""

    padding_nm = getattr(config.solvent, "padding", DEFAULT_SOLVENT_PADDING_NM)
    sam_length_estimates = tuple(
        _estimate_sam_length(component) for component in config.sam.components
    )
    sam_extended_length_nm = max(estimate.length_nm for estimate in sam_length_estimates)
    z_min = slab.bottom_z_nm - sam_extended_length_nm - padding_nm
    z_max = slab.top_z_nm + sam_extended_length_nm + padding_nm
    box_z_nm = z_max - z_min
    dimensions_nm = (slab.lateral_size_nm[0], slab.lateral_size_nm[1], box_z_nm)
    bounds_nm = (
        (-dimensions_nm[0] / 2.0, dimensions_nm[0] / 2.0),
        (-dimensions_nm[1] / 2.0, dimensions_nm[1] / 2.0),
        (z_min, z_max),
    )
    volume_nm3 = dimensions_nm[0] * dimensions_nm[1] * dimensions_nm[2]
    return BoxPlan(
        lateral_size_nm=slab.lateral_size_nm,
        dimensions_nm=dimensions_nm,
        bounds_nm=bounds_nm,
        volume_nm3=volume_nm3,
        solvent_padding_nm=padding_nm,
        sam_extended_length_nm=sam_extended_length_nm,
        slab_center_nm=(0.0, 0.0, 0.0),
        sam_length_estimates=sam_length_estimates,
    )


def _estimate_sam_length(component: Any) -> SAMLengthEstimate:
    """Estimate fully extended SAM length without chemistry toolkit dependencies."""

    heuristic_length_nm = _estimate_smiles_extended_length_nm(component.smiles)
    configured_length_nm = component.extended_length_nm
    if configured_length_nm is not None:
        length_nm = configured_length_nm
        source = "configured"
    elif heuristic_length_nm > DEFAULT_SAM_EXTENDED_LENGTH_NM:
        length_nm = heuristic_length_nm
        source = "smiles_heuristic"
    else:
        length_nm = DEFAULT_SAM_EXTENDED_LENGTH_NM
        source = "minimum_default"
    return SAMLengthEstimate(
        component_name=component.name,
        residue_name=component.residue_name,
        smiles=component.smiles,
        configured_length_nm=configured_length_nm,
        heuristic_length_nm=heuristic_length_nm,
        length_nm=length_nm,
        source=source,
    )


def _estimate_smiles_extended_length_nm(smiles: str) -> float:
    """Return a conservative heavy-atom path estimate from a simple SMILES scan."""

    heavy_atoms = 0
    index = 0
    organic_subset = {"B", "C", "N", "O", "P", "S", "F", "I"}
    aromatic_subset = {"b", "c", "n", "o", "p", "s"}
    while index < len(smiles):
        char = smiles[index]
        if char == "[":
            end = smiles.find("]", index + 1)
            token = smiles[index + 1 : end if end != -1 else len(smiles)]
            if token and not token.startswith("H"):
                heavy_atoms += 1
            index = len(smiles) if end == -1 else end + 1
            continue
        if char in organic_subset or char in aromatic_subset:
            if index + 1 < len(smiles) and smiles[index : index + 2] in {"Cl", "Br"}:
                index += 2
            else:
                index += 1
            heavy_atoms += 1
            continue
        index += 1
    if heavy_atoms <= 1:
        return 0.0
    # Approximate an all-trans heavy-atom chain; branching/rings make this conservative.
    return 0.15 + 0.154 * (heavy_atoms - 1)
