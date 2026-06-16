"""OpenFF Interchange construction for SAMMD system exports."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

from sammd.backends.forcefields import get_fcc_metal_parameters
from sammd.backends.openff import (
    PreparedMoleculeTemplate,
    force_field_inputs_from_config,
    prepare_molecule_template,
    require_openff_interchange,
    require_openff_toolkit,
)
from sammd.backends.packmol import (
    PackmolMoleculeTemplate,
    PackmolSolventComponent,
    pack_fixed_solute_with_solvent_components,
    packmol_file_stem,
)
from sammd.core.io import AtomRecord, safe_write_text
from sammd.model.sam import SAMPlacement
from sammd.model.solvation import SolutionPlan, plan_solution_composition
from sammd.utils.geometry import (
    Vector3,
    add_vectors,
    centroid,
    distance,
    matvec,
    rotate_about_axis,
    rotation_matrix,
    subtract_vectors,
)

MAX_RESIDUES_PER_CHAIN = 9999
# Component chains follow PolyzyMD semantic roles rather than construction order
CHAIN_LETTERS = "ABCDEFGHIJKLNOPQRSTUVWXYZ"
ROLE_CHAIN_STARTS = {
    "reactant": "B",
    "sam": "C",
    "solvent": "D",
}
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackendExportResult:
    """Artifacts and metadata from a completed Interchange export."""

    interchange: Any
    openmm_topology: Any
    metal_sulfur_collection: Any
    positions: Any
    positions_nm: tuple[Vector3, ...]
    sulfur_indices: tuple[int, ...]
    metal_indices: tuple[int, ...]
    anchor_pairs: tuple[tuple[int, int], ...]
    component_ranges: dict[str, dict[str, object]]
    files: dict[str, Path]
    openff_toolkit_version: str | None
    openff_interchange_version: str | None
    runtime_solvent_geometry: RuntimeSolventGeometry | None = None


@dataclass(frozen=True)
class RuntimeSolventGeometry:
    """Actual fixed-solute geometry used for runtime solvent packing."""

    solvent_boundary_z_bounds_nm: tuple[float, float]
    fixed_solute_z_bounds_nm: tuple[float, float]
    solvent_regions_nm: tuple[
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]], ...
    ]
    solvent_count_planning_volume_nm3: float
    solvent_padding_nm: float
    solvent_padding_per_face_nm: float
    solvent_clearance_nm: float
    dimensions_nm: Vector3
    z_shift_nm: float
    molecule_counts: dict[str, int]
    coordinate_shift_nm: Vector3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class _ResidueIdentity:
    chain_id: str
    residue_id: int
    residue_name: str


class _ComponentResidueAssigner:
    """Assign one stable residue identity per chemically meaningful molecule."""

    def __init__(self) -> None:
        self._role_states: dict[str, dict[str, object]] = {}
        self._states: dict[str, dict[str, object]] = {}
        self._component_ranges: dict[str, dict[str, object]] = {}

    @property
    def component_ranges(self) -> dict[str, dict[str, object]]:
        return {key: dict(value) for key, value in self._component_ranges.items()}

    def allocate(self, component_name: str, residue_name: str) -> _ResidueIdentity:
        role = _component_role(component_name)
        role_state = self._role_states.get(role)
        if role_state is None:
            role_state = {
                "chain_ids": [ROLE_CHAIN_STARTS[role]],
                "next_residue_id": 1,
                "next_chain_index": CHAIN_LETTERS.index(ROLE_CHAIN_STARTS[role]),
            }
            self._role_states[role] = role_state

        state = self._states.get(component_name)
        if state is None:
            state = {
                "residue_name": residue_name,
                "residue_count": 0,
                "chain_ids": [],
            }
            self._states[component_name] = state

        role_chain_ids = role_state["chain_ids"]
        next_residue_id = int(role_state["next_residue_id"])
        if next_residue_id > MAX_RESIDUES_PER_CHAIN:
            role_chain_ids.append(self._next_chain_id(role_state))
            next_residue_id = 1
        chain_id = str(role_chain_ids[-1])

        component_chain_ids = state["chain_ids"]
        if chain_id not in component_chain_ids:
            component_chain_ids.append(chain_id)

        identity = _ResidueIdentity(
            chain_id=chain_id,
            residue_id=next_residue_id,
            residue_name=residue_name,
        )
        role_state["next_residue_id"] = next_residue_id + 1
        state["residue_count"] = int(state["residue_count"]) + 1
        self._component_ranges[component_name] = {
            "residue_name": residue_name,
            "residue_count": state["residue_count"],
            "chain_ids": tuple(component_chain_ids),
            "max_residues_per_chain": MAX_RESIDUES_PER_CHAIN,
        }
        return identity

    def _next_chain_id(self, role_state: dict[str, object]) -> str:
        next_chain_index = int(role_state["next_chain_index"]) + 1
        if next_chain_index >= len(CHAIN_LETTERS):
            msg = "Interchange export exceeded available one-character chain identifiers"
            raise RuntimeError(msg)
        chain_id = CHAIN_LETTERS[next_chain_index]
        role_state["next_chain_index"] = next_chain_index
        return chain_id


def _component_role(component_name: str) -> str:
    """Return the semantic chain role for a component label."""

    if component_name.startswith("reactant:"):
        return "reactant"
    if component_name.startswith("sam:"):
        return "sam"
    if component_name.startswith("solvent:"):
        return "solvent"
    msg = f"unknown Interchange component role for {component_name!r}"
    raise ValueError(msg)


ProgressCallback = Callable[[str], None]


def _progress(callback: ProgressCallback | None, message: str) -> None:
    LOGGER.info(message)
    if callback is not None:
        callback(message)


def _timed_progress(
    callback: ProgressCallback | None,
    message: str,
    start_time: float,
) -> float:
    now = perf_counter()
    _progress(callback, f"{message} ({now - start_time:.2f}s)")
    return now


def export_interchange_backend(
    plan: Any,
    *,
    overwrite: bool = False,
    progress: ProgressCallback | None = None,
) -> BackendExportResult:
    """Construct Interchange-backed artifacts and write them to configured paths."""

    _progress(progress, "Building OpenFF Interchange export")
    result = build_interchange_backend(plan, progress=progress)
    paths = plan.output_paths
    _require_paths(paths)

    stage_start = perf_counter()
    _progress(progress, "Writing interchange.json")
    safe_write_text(
        paths.openff_interchange,
        result.interchange.model_dump_json(indent=2) + "\n",
        overwrite=overwrite,
    )
    _timed_progress(progress, "  interchange.json written", stage_start)
    stage_start = perf_counter()
    _progress(progress, "Writing solvated_system.cif")
    _write_pdbx(
        paths.solvated_system,
        result.openmm_topology,
        result.positions,
        overwrite=overwrite,
    )
    _timed_progress(progress, "  solvated_system.cif written", stage_start)
    stage_start = perf_counter()
    _progress(progress, "Writing solvated_system_pymol.pdb")
    _write_pdb(
        paths.pymol_system,
        result.openmm_topology,
        result.positions,
        overwrite=overwrite,
    )
    _timed_progress(progress, "  solvated_system_pymol.pdb written", stage_start)
    stage_start = perf_counter()
    _progress(progress, "Writing anchor_metadata.json")
    safe_write_text(
        paths.anchor_metadata,
        json.dumps(_anchor_metadata(result), indent=2, sort_keys=True) + "\n",
        overwrite=overwrite,
    )
    _timed_progress(progress, "  anchor_metadata.json written", stage_start)
    return BackendExportResult(
        interchange=result.interchange,
        openmm_topology=result.openmm_topology,
        metal_sulfur_collection=result.metal_sulfur_collection,
        positions=result.positions,
        positions_nm=result.positions_nm,
        sulfur_indices=result.sulfur_indices,
        metal_indices=result.metal_indices,
        anchor_pairs=result.anchor_pairs,
        component_ranges=result.component_ranges,
        files={
            "solvated_system": paths.solvated_system,
            "pymol_system": paths.pymol_system,
            "openff_interchange": paths.openff_interchange,
            "anchor_metadata": paths.anchor_metadata,
        },
        openff_toolkit_version=result.openff_toolkit_version,
        openff_interchange_version=result.openff_interchange_version,
        runtime_solvent_geometry=result.runtime_solvent_geometry,
    )


def backend_build_summary(plan: Any, result: BackendExportResult) -> dict[str, object]:
    """Return a build summary updated for completed Interchange exports."""

    summary = plan.build_summary()
    artifacts = dict(summary["artifacts"])
    for key in ("solvated_system", "pymol_system", "openff_interchange", "anchor_metadata"):
        artifact = dict(artifacts[key])
        artifact["status"] = "current"
        artifact["available"] = True
        if key == "openff_interchange":
            artifact["constructed"] = True
            artifact["openff_interchange_package_version"] = result.openff_interchange_version
        artifacts[key] = artifact
    summary["artifacts"] = artifacts
    summary["backend_export"] = {
        "mode": "openff_interchange_with_plugin_pair_overrides",
        "openff_toolkit_version": result.openff_toolkit_version,
        "openff_interchange_version": result.openff_interchange_version,
        "atom_count": len(result.positions_nm),
        "metal_atom_count": len(result.metal_indices),
        "sam_sulfur_count": len(result.sulfur_indices),
        "sulfur_metal_pair_count": len(result.anchor_pairs),
        "metal_sulfur_override": _anchor_metadata(result),
    }
    runtime_solvent_geometry = getattr(result, "runtime_solvent_geometry", None)
    if runtime_solvent_geometry is not None:
        geometry = runtime_solvent_geometry
        box_summary = dict(summary["box"])
        box_summary.update(
            {
                "dimensions_nm": list(geometry.dimensions_nm),
                "bounds_nm": [
                    [0.0, geometry.dimensions_nm[0]],
                    [0.0, geometry.dimensions_nm[1]],
                    [0.0, geometry.dimensions_nm[2]],
                ],
                "volume_nm3": (
                    geometry.dimensions_nm[0]
                    * geometry.dimensions_nm[1]
                    * geometry.dimensions_nm[2]
                ),
                "actual_solvent_boundary_z_bounds_nm": list(
                    geometry.solvent_boundary_z_bounds_nm
                ),
                "actual_fixed_solute_z_bounds_nm": list(geometry.fixed_solute_z_bounds_nm),
                "solvent_packing_regions_nm": [
                    [list(axis_bounds) for axis_bounds in region]
                    for region in geometry.solvent_regions_nm
                ],
                "coordinate_shift_nm": list(geometry.coordinate_shift_nm),
                "z_shift_nm": geometry.z_shift_nm,
                "solvent_count_planning_volume_nm3": (
                    geometry.solvent_count_planning_volume_nm3
                ),
                "solvent_padding_nm": geometry.solvent_padding_nm,
                "solvent_padding_per_face_nm": geometry.solvent_padding_per_face_nm,
                "solvent_clearance_nm": geometry.solvent_clearance_nm,
            }
        )
        summary["box"] = box_summary
        solution_summary = dict(summary["solution"])
        solution_summary.update(
            {
                "count_planning_volume_nm3": geometry.solvent_count_planning_volume_nm3,
                "molecule_counts": geometry.molecule_counts,
            }
        )
        summary["solution"] = solution_summary
    return summary


def build_interchange_backend(
    plan: Any,
    *,
    progress: ProgressCallback | None = None,
) -> BackendExportResult:
    """Build an OpenFF Interchange export from a SAMMD plan."""

    if plan.solution.salts:
        names = ", ".join(salt.name for salt in plan.solution.salts)
        msg = f"Interchange export does not yet support salts: {names}"
        raise NotImplementedError(msg)

    stage_start = perf_counter()
    _progress(progress, "Importing OpenFF/OpenMM dependencies")
    toolkit = require_openff_toolkit()
    interchange_module = require_openff_interchange()
    modules = _require_openmm()
    unit = _openff_unit_module()
    from sammd.backends.interchange_plugins import (
        create_metal_sulfur_lj_collection,
        register_interchange_plugin_collection,
    )

    register_interchange_plugin_collection()
    _timed_progress(progress, "  OpenFF/OpenMM dependencies ready", stage_start)

    stage_start = perf_counter()
    _progress(progress, "Loading force fields")
    force_field_inputs = [str(item) for item in force_field_inputs_from_config(plan.config)]
    force_field = toolkit.ForceField(*force_field_inputs)
    _timed_progress(progress, "  force fields ready", stage_start)
    molecules: list[Any] = []
    positions_nm: list[Vector3] = []
    atom_records: list[AtomRecord] = []
    sulfur_indices: list[int] = []
    metal_indices: list[int] = []
    anchor_pairs: list[tuple[int, int]] = []
    residue_assigner = _ComponentResidueAssigner()

    stage_start = perf_counter()
    _progress(progress, f"Preparing {len(plan.slab.positions_nm)} metal atoms")
    shift_nm = tuple(dimension / 2.0 for dimension in plan.box_plan.dimensions_nm)

    pd_template = _metal_atom_molecule(toolkit, plan.slab.metal)
    stage_start = _timed_progress(progress, "  metal template ready", stage_start)
    for metal_atom_index, pos in enumerate(plan.slab.positions_nm, start=1):
        metal_identity = _ResidueIdentity(
            chain_id="M",
            residue_id=metal_atom_index,
            residue_name=_metal_residue_name(plan.slab.metal),
        )
        _append_identified_molecule(
            molecules,
            positions_nm,
            atom_records,
            pd_template,
            [tuple(position + shift for position, shift in zip(pos, shift_nm, strict=True))],
            metal_identity,
            component_label="metal_slab",
        )
    metal_indices.extend(range(len(plan.slab.positions_nm)))
    _timed_progress(progress, "  metal block appended", stage_start)

    stage_start = perf_counter()
    _progress(progress, f"Preparing {len(plan.sam_placements.placements)} SAM molecules")
    templates: dict[tuple[str, str], Any] = {}
    for placement_index, placement in enumerate(plan.sam_placements.placements, start=1):
        template_key = (placement.component_name, placement.component_smiles)
        template = templates.get(template_key)
        if template is None:
            template = _prepare_template(
                toolkit,
                placement.component_smiles,
                placement.component_name,
                plan.config.parameterization.charge_model,
            )
            templates[template_key] = template
            stage_start = _timed_progress(
                progress,
                f"  SAM template ready: {placement.component_name}",
                stage_start,
            )
        molecule, transformed = _placed_sam_molecule(template, placement, shift_nm)
        start = len(positions_nm)
        _append_identified_molecule(
            molecules,
            positions_nm,
            atom_records,
            molecule,
            transformed,
            residue_assigner.allocate(
                f"sam:{placement.component_name}",
                placement.component_residue_name,
            ),
            component_label=f"sam:{placement.component_name}",
        )
        sulfur_index = start + _single_atom_index(template.atom_symbols, "S", "SAM molecule")
        sulfur_indices.append(sulfur_index)
        anchor_pairs.extend(
            (sulfur_index, int(metal_index))
            for metal_index in placement.anchor_pose.nearest_metal_atom_indices
        )
        if placement_index % 25 == 0 or placement_index == len(plan.sam_placements.placements):
            stage_start = _timed_progress(
                progress,
                f"  placed {placement_index}/{len(plan.sam_placements.placements)} SAM molecules",
                stage_start,
            )

    solvent_boundary_records = tuple(atom_records)

    reactant_count = sum(reactant.count for reactant in plan.solution.reactants)
    stage_start = perf_counter()
    _progress(progress, f"Preparing {reactant_count} reactants")
    for reactant in plan.solution.reactants:
        template = _prepare_template(
            toolkit,
            reactant.smiles,
            reactant.name,
            plan.config.parameterization.charge_model,
        )
        stage_start = _timed_progress(
            progress,
            f"  reactant template ready: {reactant.name}",
            stage_start,
        )
        for transformed in _molecule_centers_above_solute(
            template,
            reactant.count,
            plan.box_plan.dimensions_nm,
            tuple(positions_nm),
            clearance_nm=0.55,
        ):
            molecule = deepcopy(template.molecule)
            _append_identified_molecule(
                molecules,
                positions_nm,
                atom_records,
                molecule,
                transformed,
                residue_assigner.allocate(
                    f"reactant:{reactant.name}",
                    reactant.residue_name or "RCT",
                ),
                component_label=f"reactant:{reactant.name}",
            )
    _timed_progress(progress, "  reactants placed", stage_start)

    runtime_geometry = _runtime_solvent_geometry(
        plan,
        tuple(atom_records),
        solvent_boundary_records=solvent_boundary_records,
    )
    if runtime_geometry.coordinate_shift_nm != (0.0, 0.0, 0.0):
        positions_nm[:] = [
            _shift_position(position, runtime_geometry.coordinate_shift_nm)
            for position in positions_nm
        ]
        atom_records[:] = [
            _shift_atom_record(record, runtime_geometry.coordinate_shift_nm)
            for record in atom_records
        ]
    packmol_solute_records = tuple(
        _wrap_atom_record_xy(record, runtime_geometry.dimensions_nm) for record in atom_records
    )
    _ensure_positions_inside_box(
        tuple(record.coordinates_nm for record in packmol_solute_records),
        runtime_geometry.dimensions_nm,
        context="fixed solute Packmol",
    )

    runtime_solution = plan_solution_composition(
        plan.config,
        runtime_geometry.solvent_count_planning_volume_nm3,
    )
    runtime_geometry = _runtime_geometry_with_counts(
        runtime_geometry,
        runtime_solution,
        plan.solution,
    )

    solvent_count = sum(solvent.count for solvent in runtime_solution.solvent_components)
    stage_start = perf_counter()
    _progress(progress, f"Preparing {solvent_count} solvent molecules")
    solvent_templates: list[tuple[Any, PreparedMoleculeTemplate]] = []
    for solvent in runtime_solution.solvent_components:
        if solvent.smiles is None:
            msg = (
                f"solvent component {solvent.name!r} requires explicit SMILES for "
                "Interchange export"
            )
            raise ValueError(msg)
        template = _prepare_template(
            toolkit,
            solvent.smiles,
            solvent.name,
            plan.config.parameterization.charge_model,
        )
        stage_start = _timed_progress(
            progress,
            f"  solvent template ready: {solvent.name}",
            stage_start,
        )
        solvent_templates.append((solvent, template))

    packed_solvent = pack_fixed_solute_with_solvent_components(
        solute_records=packmol_solute_records,
        solvent_components=tuple(
            PackmolSolventComponent(
                name=solvent.name,
                template=PackmolMoleculeTemplate(
                    residue_name=solvent.residue_name or "SOL",
                    positions_nm=template.positions_nm,
                    atom_symbols=template.atom_symbols,
                    atom_names=tuple(
                        _atom_name(symbol, index)
                        for index, symbol in enumerate(template.atom_symbols, 1)
                    ),
                ),
                count=solvent.count,
            )
            for solvent, template in solvent_templates
        ),
        dimensions_nm=runtime_geometry.dimensions_nm,
        working_dir=_packmol_working_dir(plan, "mixed_solvent"),
        solvent_regions_nm=runtime_geometry.solvent_regions_nm,
        tolerance_angstrom=plan.config.packing.packmol.tolerance,
        nloop=plan.config.packing.packmol.nloop,
    )
    if solvent_templates:
        stage_start = _timed_progress(
            progress,
            "  PACKMOL solvent positions ready",
            stage_start,
        )

    for solvent, template in solvent_templates:
        centered_solvent = packed_solvent.get(solvent.name, ())
        for solvent_index, transformed in enumerate(centered_solvent, start=1):
            molecule = deepcopy(template.molecule)
            _append_identified_molecule(
                molecules,
                positions_nm,
                atom_records,
                molecule,
                transformed,
                residue_assigner.allocate(
                    f"solvent:{solvent.name}",
                    solvent.residue_name or "SOL",
                ),
                component_label=f"solvent:{solvent.name}",
            )
            if solvent_index % 250 == 0 or solvent_index == len(centered_solvent):
                stage_start = _timed_progress(
                    progress,
                    f"  placed {solvent_index}/{len(centered_solvent)} {solvent.name} molecules",
                    stage_start,
                )

    stage_start = perf_counter()
    _progress(progress, f"Creating OpenFF Topology ({len(molecules)} molecules)")
    topology = toolkit.Topology.from_molecules(molecules)
    stage_start = _timed_progress(progress, "  OpenFF topology object ready", stage_start)
    topology.box_vectors = [
        [runtime_geometry.dimensions_nm[0], 0.0, 0.0],
        [0.0, runtime_geometry.dimensions_nm[1], 0.0],
        [0.0, 0.0, runtime_geometry.dimensions_nm[2]],
    ] * unit.nanometer
    positions = positions_nm * unit.nanometer
    _timed_progress(progress, f"  positions array ready ({len(positions_nm)} atoms)", stage_start)
    stage_start = perf_counter()
    _progress(progress, "Parameterizing with OpenFF Interchange")
    interchange = interchange_module.Interchange.from_smirnoff(
        force_field,
        topology,
        box=topology.box_vectors,
        positions=positions,
        charge_from_molecules=_unique_charge_molecules(molecules),
    )
    _timed_progress(progress, "  OpenFF Interchange ready", stage_start)
    stage_start = perf_counter()
    _progress(progress, f"Recording {len(anchor_pairs)} sulfur-metal pair overrides")
    metal_sulfur_collection = create_metal_sulfur_lj_collection(tuple(anchor_pairs))
    interchange.collections[metal_sulfur_collection.type] = metal_sulfur_collection
    _timed_progress(progress, "  sulfur-metal override collection attached", stage_start)
    stage_start = perf_counter()
    _progress(progress, "Exporting OpenMM topology and positions")
    openmm_topology = interchange.to_openmm_topology()
    _apply_openmm_atom_identities(openmm_topology, tuple(atom_records))
    _label_openmm_metal_atoms(openmm_topology, tuple(metal_indices), plan.slab.metal)
    _ensure_openmm_atom_names(openmm_topology)
    openmm_positions = modules.unit.Quantity(
        [modules.openmm.Vec3(*position) for position in positions_nm],
        modules.unit.nanometer,
    )
    _timed_progress(progress, "  OpenMM topology and positions ready", stage_start)
    return BackendExportResult(
        interchange=interchange,
        openmm_topology=openmm_topology,
        metal_sulfur_collection=metal_sulfur_collection,
        positions=openmm_positions,
        positions_nm=tuple(positions_nm),
        sulfur_indices=tuple(sulfur_indices),
        metal_indices=tuple(metal_indices),
        anchor_pairs=tuple(anchor_pairs),
        component_ranges={
            "metal_slab": {
                "residue_name": _metal_residue_name(plan.slab.metal),
                "residue_count": len(plan.slab.positions_nm),
                "chain_ids": ("M",),
                "max_residues_per_chain": MAX_RESIDUES_PER_CHAIN,
            },
            **residue_assigner.component_ranges,
        },
        files={},
        openff_toolkit_version=_package_version("openff-toolkit"),
        openff_interchange_version=_package_version("openff-interchange"),
        runtime_solvent_geometry=runtime_geometry,
    )


def _prepare_template(
    toolkit: Any,
    smiles: str,
    name: str,
    charge_model: str,
) -> PreparedMoleculeTemplate:
    return prepare_molecule_template(
        smiles,
        name,
        charge_model,
        toolkit=toolkit,
        allow_undefined_stereo=True,
    )


def _metal_atom_molecule(toolkit: Any, symbol: str) -> Any:
    get_fcc_metal_parameters(symbol)
    molecule = toolkit.Molecule.from_smiles(f"[{symbol}]", allow_undefined_stereo=True)
    molecule.name = _metal_residue_name(symbol)
    molecule.atoms[0].name = symbol
    molecule.assign_partial_charges("zeros")
    return molecule


def _placed_sam_molecule(
    template: PreparedMoleculeTemplate,
    placement: SAMPlacement,
    shift_nm: Vector3,
) -> tuple[Any, tuple[Vector3, ...]]:
    sulfur_index = _single_atom_index(template.atom_symbols, "S", "SAM molecule")
    axis_index = _terminal_heavy_axis_index(template, sulfur_index)
    target_sulfur = tuple(
        value + shift
        for value, shift in zip(placement.anchor_pose.sulfur_position_nm, shift_nm, strict=True)
    )
    transformed = _orient_template_by_anchor(
        template.positions_nm,
        anchor_index=sulfur_index,
        axis_index=axis_index,
        target_anchor_nm=target_sulfur,
        target_direction=placement.anchor_pose.axis_direction,
        azimuth_rad=placement.anchor_pose.azimuth_rad,
    )
    return deepcopy(template.molecule), transformed


def _molecule_centers_above_solute(
    template: PreparedMoleculeTemplate,
    count: int,
    dimensions_nm: Vector3,
    solute_positions_nm: tuple[Vector3, ...],
    *,
    clearance_nm: float,
) -> tuple[tuple[Vector3, ...], ...]:
    if count <= 0:
        return ()
    columns = max(1, math.ceil(count ** (1.0 / 3.0)))
    rows = max(1, math.ceil(math.sqrt(count / columns)))
    layers = max(1, math.ceil(count / (columns * rows)))
    solute_top_z = max((position[2] for position in solute_positions_nm), default=0.0)
    template_center = centroid(template.positions_nm)
    template_radius = max(distance(template_center, position) for position in template.positions_nm)
    z_start = solute_top_z + clearance_nm + template_radius
    z_stop = max(z_start, dimensions_nm[2] - template_radius - 0.15)
    z_spacing = max(0.22, (z_stop - z_start) / max(1, layers - 1))
    centered = []
    for index in range(count):
        col = index % columns
        row = (index // columns) % rows
        layer = index // (columns * rows)
        center = (
            dimensions_nm[0] * (col + 0.5) / columns,
            dimensions_nm[1] * (row + 0.5) / rows,
            min(dimensions_nm[2] - template_radius - 0.05, z_start + layer * z_spacing),
        )
        centered.append(_center_template(template.positions_nm, center))
    return tuple(centered)


def _packmol_working_dir(plan: Any, solvent_name: str) -> Path:
    paths = getattr(plan, "output_paths", None)
    solvated_system = getattr(paths, "solvated_system", None)
    if solvated_system is None:
        return Path(tempfile.mkdtemp(prefix="sammd-packmol-"))
    return Path(solvated_system).parent / "packmol" / packmol_file_stem(solvent_name)


def _runtime_solvent_regions(plan: Any) -> tuple[Any, ...]:
    """Return planned solvent regions in the shifted runtime coordinate frame."""

    box_lowers = tuple(axis_bounds[0] for axis_bounds in plan.box_plan.bounds_nm)
    return tuple(
        tuple(
            (axis_bounds[0] - axis_lower, axis_bounds[1] - axis_lower)
            for axis_bounds, axis_lower in zip(region, box_lowers, strict=True)
        )
        for region in plan.box_plan.solvent_packing_regions_nm
    )


def _runtime_solvent_geometry(
    plan: Any,
    fixed_solute_records: tuple[AtomRecord, ...],
    *,
    solvent_boundary_records: tuple[AtomRecord, ...] | None = None,
) -> RuntimeSolventGeometry:
    """Derive runtime solvent regions from slab/SAM z extents."""

    if not fixed_solute_records:
        msg = "runtime solvent geometry requires fixed-solute atom records"
        raise ValueError(msg)
    boundary_records = solvent_boundary_records or fixed_solute_records
    if not boundary_records:
        msg = "runtime solvent geometry requires solvent-boundary atom records"
        raise ValueError(msg)
    padding_nm = float(getattr(plan.config.solvent, "padding", 3.0))
    padding_per_face_nm = padding_nm / 2.0
    clearance_nm = float(plan.config.packing.packmol.tolerance) / 10.0
    boundary_min_z = min(record.coordinates_nm[2] for record in boundary_records)
    boundary_max_z = max(record.coordinates_nm[2] for record in boundary_records)
    fixed_min_z = min(record.coordinates_nm[2] for record in fixed_solute_records)
    fixed_max_z = max(record.coordinates_nm[2] for record in fixed_solute_records)
    solvent_z_min = boundary_min_z - padding_per_face_nm
    solvent_z_max = boundary_max_z + padding_per_face_nm
    final_z_min = min(solvent_z_min, fixed_min_z - clearance_nm)
    final_z_max = max(solvent_z_max, fixed_max_z + clearance_nm)
    coordinate_shift_nm = (0.0, 0.0, -final_z_min)
    z_shift_nm = coordinate_shift_nm[2]
    dimensions_nm = (
        plan.box_plan.dimensions_nm[0],
        plan.box_plan.dimensions_nm[1],
        final_z_max - final_z_min,
    )
    boundary_bounds = (boundary_min_z + z_shift_nm, boundary_max_z + z_shift_nm)
    fixed_bounds = (fixed_min_z + z_shift_nm, fixed_max_z + z_shift_nm)
    bottom_region = (
        (0.0, dimensions_nm[0]),
        (0.0, dimensions_nm[1]),
        (solvent_z_min + z_shift_nm, boundary_bounds[0] - clearance_nm),
    )
    top_region = (
        (0.0, dimensions_nm[0]),
        (0.0, dimensions_nm[1]),
        (boundary_bounds[1] + clearance_nm, solvent_z_max + z_shift_nm),
    )
    solvent_regions_nm = tuple(
        region for region in (bottom_region, top_region) if region[2][1] > region[2][0]
    )
    count_volume_nm3 = sum(_region_volume_nm3(region) for region in solvent_regions_nm)
    return RuntimeSolventGeometry(
        solvent_boundary_z_bounds_nm=boundary_bounds,
        fixed_solute_z_bounds_nm=fixed_bounds,
        solvent_regions_nm=solvent_regions_nm,
        solvent_count_planning_volume_nm3=count_volume_nm3,
        solvent_padding_nm=padding_nm,
        solvent_padding_per_face_nm=padding_per_face_nm,
        solvent_clearance_nm=clearance_nm,
        dimensions_nm=dimensions_nm,
        z_shift_nm=z_shift_nm,
        molecule_counts={},
        coordinate_shift_nm=coordinate_shift_nm,
    )


def _ensure_positions_inside_box(
    positions_nm: tuple[Vector3, ...],
    dimensions_nm: Vector3,
    *,
    context: str,
) -> None:
    """Validate that coordinates lie inside the zero-origin runtime box."""

    tolerance_nm = 1.0e-9
    for atom_index, position in enumerate(positions_nm, start=1):
        for axis, coordinate, dimension in zip("xyz", position, dimensions_nm, strict=True):
            if coordinate < -tolerance_nm or coordinate > dimension + tolerance_nm:
                msg = (
                    f"{context} atom {atom_index} {axis}-coordinate {coordinate:g} nm "
                    f"lies outside runtime box dimension {dimension:g} nm"
                )
                raise ValueError(msg)


def _runtime_geometry_with_counts(
    geometry: RuntimeSolventGeometry,
    solvent_solution: SolutionPlan,
    fixed_solution: SolutionPlan,
) -> RuntimeSolventGeometry:
    """Return runtime solvent geometry annotated with solution counts."""

    molecule_counts: dict[str, int] = {}
    for component in solvent_solution.solvent_components:
        molecule_counts[component.name] = molecule_counts.get(component.name, 0) + component.count
    for component in fixed_solution.reactants:
        molecule_counts[component.name] = molecule_counts.get(component.name, 0) + component.count
    for salt in fixed_solution.salts:
        molecule_counts[salt.cation] = molecule_counts.get(salt.cation, 0) + salt.cation_count
        molecule_counts[salt.anion] = molecule_counts.get(salt.anion, 0) + salt.anion_count

    return RuntimeSolventGeometry(
        solvent_boundary_z_bounds_nm=geometry.solvent_boundary_z_bounds_nm,
        fixed_solute_z_bounds_nm=geometry.fixed_solute_z_bounds_nm,
        solvent_regions_nm=geometry.solvent_regions_nm,
        solvent_count_planning_volume_nm3=geometry.solvent_count_planning_volume_nm3,
        solvent_padding_nm=geometry.solvent_padding_nm,
        solvent_padding_per_face_nm=geometry.solvent_padding_per_face_nm,
        solvent_clearance_nm=geometry.solvent_clearance_nm,
        dimensions_nm=geometry.dimensions_nm,
        z_shift_nm=geometry.z_shift_nm,
        molecule_counts=molecule_counts,
        coordinate_shift_nm=geometry.coordinate_shift_nm,
    )


def _region_volume_nm3(
    region: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> float:
    """Return the volume of one orthorhombic region."""

    return (
        (region[0][1] - region[0][0])
        * (region[1][1] - region[1][0])
        * (region[2][1] - region[2][0])
    )


def _shift_position_z(position: Vector3, shift_nm: float) -> Vector3:
    """Return a position shifted along z."""

    return (position[0], position[1], position[2] + shift_nm)


def _shift_position(position: Vector3, shift_nm: Vector3) -> Vector3:
    """Return a position shifted by a three-axis vector."""

    return (
        position[0] + shift_nm[0],
        position[1] + shift_nm[1],
        position[2] + shift_nm[2],
    )


def _wrap_position_xy(position: Vector3, dimensions_nm: Vector3) -> Vector3:
    """Return a copy of a position imaged into the primary XY cell."""

    return (
        position[0] % dimensions_nm[0],
        position[1] % dimensions_nm[1],
        position[2],
    )


def _shift_atom_record_z(record: AtomRecord, shift_nm: float) -> AtomRecord:
    """Return an atom record shifted along z."""

    return AtomRecord(
        serial=record.serial,
        atom_name=record.atom_name,
        element=record.element,
        residue_name=record.residue_name,
        residue_id=record.residue_id,
        chain_id=record.chain_id,
        component_label=record.component_label,
        coordinates_nm=_shift_position_z(record.coordinates_nm, shift_nm),
    )


def _shift_atom_record(record: AtomRecord, shift_nm: Vector3) -> AtomRecord:
    """Return an atom record shifted by a three-axis vector."""

    return AtomRecord(
        serial=record.serial,
        atom_name=record.atom_name,
        element=record.element,
        residue_name=record.residue_name,
        residue_id=record.residue_id,
        chain_id=record.chain_id,
        component_label=record.component_label,
        coordinates_nm=_shift_position(record.coordinates_nm, shift_nm),
    )


def _wrap_atom_record_xy(record: AtomRecord, dimensions_nm: Vector3) -> AtomRecord:
    """Return an atom record imaged into the primary XY cell."""

    return AtomRecord(
        serial=record.serial,
        atom_name=record.atom_name,
        element=record.element,
        residue_name=record.residue_name,
        residue_id=record.residue_id,
        chain_id=record.chain_id,
        component_label=record.component_label,
        coordinates_nm=_wrap_position_xy(record.coordinates_nm, dimensions_nm),
    )


def _append_identified_molecule(
    molecules: list[Any],
    positions_nm: list[Vector3],
    atom_records: list[AtomRecord],
    molecule: Any,
    molecule_positions_nm: list[Vector3] | tuple[Vector3, ...],
    identity: _ResidueIdentity,
    *,
    component_label: str,
) -> None:
    molecule = deepcopy(molecule)
    local_names, local_symbols = _assign_openff_atom_identity(molecule, identity)
    molecules.append(molecule)
    for atom_name, symbol, position in zip(
        local_names,
        local_symbols,
        molecule_positions_nm,
        strict=True,
    ):
        positions_nm.append(position)
        atom_records.append(
            AtomRecord(
                serial=len(atom_records) + 1,
                atom_name=atom_name,
                element=symbol,
                residue_name=identity.residue_name,
                residue_id=identity.residue_id,
                chain_id=identity.chain_id,
                component_label=component_label,
                coordinates_nm=position,
            )
        )


def _assign_openff_atom_identity(
    molecule: Any, identity: _ResidueIdentity
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    names = []
    symbols = []
    for atom_index, atom in enumerate(molecule.atoms, start=1):
        symbol = getattr(atom, "symbol", None) or "X"
        atom_name = getattr(atom, "name", "") or _atom_name(symbol, atom_index)
        atom.name = atom_name
        atom.metadata.update(
            {
                "chain_id": identity.chain_id,
                "residue_number": str(identity.residue_id),
                "residue_id": str(identity.residue_id),
                "residue_name": identity.residue_name,
            }
        )
        names.append(atom_name)
        symbols.append(symbol)
    return tuple(names), tuple(symbols)


def _atom_name(symbol: str, atom_index: int) -> str:
    return f"{symbol}{atom_index}"[:4]


def _unique_charge_molecules(molecules: list[Any]) -> list[Any]:
    seen: set[str] = set()
    unique = []
    for molecule in molecules:
        key = molecule.to_smiles(mapped=False)
        if key not in seen:
            seen.add(key)
            unique.append(molecule)
    return unique


def _single_atom_index(symbols: tuple[str, ...], symbol: str, label: str) -> int:
    matches = [index for index, atom_symbol in enumerate(symbols) if atom_symbol == symbol]
    if len(matches) != 1:
        msg = f"{label} must contain exactly one {symbol} atom"
        raise ValueError(msg)
    return matches[0]


def _terminal_heavy_axis_index(template: PreparedMoleculeTemplate, anchor_index: int) -> int:
    heavy_indices = [
        index
        for index, symbol in enumerate(template.atom_symbols)
        if index != anchor_index and symbol != "H"
    ]
    if not heavy_indices:
        msg = "SAM molecule must contain a heavy atom beyond sulfur"
        raise ValueError(msg)
    return max(
        heavy_indices,
        key=lambda index: distance(
            template.positions_nm[anchor_index],
            template.positions_nm[index],
        ),
    )


def _anchor_metadata(result: BackendExportResult) -> dict[str, object]:
    """Return anchor override metadata with runtime frame audit fields."""

    from sammd.backends.interchange_plugins import metal_sulfur_lj_override_summary

    metadata = metal_sulfur_lj_override_summary(result.anchor_pairs)
    metadata["sulfur_indices"] = list(result.sulfur_indices)
    metadata["metal_indices"] = list(result.metal_indices)
    runtime_solvent_geometry = getattr(result, "runtime_solvent_geometry", None)
    if runtime_solvent_geometry is not None:
        metadata["runtime_coordinate_frame"] = _runtime_coordinate_frame_metadata(
            runtime_solvent_geometry
        )
    return metadata


def _runtime_coordinate_frame_metadata(geometry: RuntimeSolventGeometry) -> dict[str, object]:
    """Return serializable metadata for the zero-origin Packmol runtime frame."""

    return {
        "coordinate_shift_nm": list(geometry.coordinate_shift_nm),
        "z_shift_nm": geometry.z_shift_nm,
        "dimensions_nm": list(geometry.dimensions_nm),
        "bounds_nm": [
            [0.0, geometry.dimensions_nm[0]],
            [0.0, geometry.dimensions_nm[1]],
            [0.0, geometry.dimensions_nm[2]],
        ],
        "actual_solvent_boundary_z_bounds_nm": list(geometry.solvent_boundary_z_bounds_nm),
        "actual_fixed_solute_z_bounds_nm": list(geometry.fixed_solute_z_bounds_nm),
        "solvent_packing_regions_nm": [
            [list(axis_bounds) for axis_bounds in region]
            for region in geometry.solvent_regions_nm
        ],
        "solvent_count_planning_volume_nm3": geometry.solvent_count_planning_volume_nm3,
        "solvent_padding_nm": geometry.solvent_padding_nm,
        "solvent_padding_per_face_nm": geometry.solvent_padding_per_face_nm,
        "solvent_clearance_nm": geometry.solvent_clearance_nm,
    }


def _write_pdbx(path: Path, topology: Any, positions: Any, *, overwrite: bool) -> None:
    """Write OpenMM PDBx/mmCIF text to SAMMD's stable ``.cif`` artifact path."""

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=destination.parent, delete=False
        ) as handle:
            temporary_name = handle.name
            _require_openmm().app.PDBxFile.writeFile(topology, positions, handle, keepIds=True)
            # PyMOL treats EOF inside the final atom_site loop as a truncated loop.
            handle.write("\n#\n")
        if overwrite:
            os.replace(temporary_name, destination)
        else:
            os.link(temporary_name, destination)
            os.unlink(temporary_name)
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def _write_pdb(path: Path, topology: Any, positions: Any, *, overwrite: bool) -> None:
    """Write PDB text with explicit CONECT records for PyMOL visualization."""

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=destination.parent, delete=False
        ) as handle:
            temporary_name = handle.name
            _require_openmm().app.PDBFile.writeFile(topology, positions, handle, keepIds=True)
        if overwrite:
            os.replace(temporary_name, destination)
        else:
            os.link(temporary_name, destination)
            os.unlink(temporary_name)
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def _ensure_openmm_atom_names(topology: Any) -> None:
    """Fill missing atom names so PDBx/mmCIF rows have all declared fields."""

    for atom in topology.atoms():
        if getattr(atom, "name", None):
            continue
        element = getattr(atom, "element", None)
        symbol = getattr(element, "symbol", None) or "X"
        atom.name = symbol


def _apply_openmm_atom_identities(
    topology: Any,
    atom_records: tuple[AtomRecord, ...],
) -> None:
    """Repair OpenMM atom/residue labels using the append-order identity ledger."""

    atoms = tuple(topology.atoms())
    if len(atoms) != len(atom_records):
        msg = (
            "OpenMM topology atom count does not match Interchange export identity ledger: "
            f"{len(atoms)} topology atoms, {len(atom_records)} atom records"
        )
        raise ValueError(msg)
    for atom, record in zip(atoms, atom_records, strict=True):
        atom.name = record.atom_name
        atom.residue.name = record.residue_name
        atom.residue.id = str(record.residue_id)
        atom.residue.chain.id = record.chain_id


def _label_openmm_metal_atoms(topology: Any, metal_indices: tuple[int, ...], symbol: str) -> None:
    """Assign metal atom/residue labels that produce valid, readable PDBx rows."""

    residue_name = _metal_residue_name(symbol)
    metal_index_set = set(metal_indices)
    for atom in topology.atoms():
        if atom.index not in metal_index_set:
            continue
        atom.name = symbol
        atom.residue.name = residue_name
        atom.residue.id = str(atom.index + 1)
        atom.residue.chain.id = "M"


def _metal_residue_name(symbol: str) -> str:
    """Return a three-character metal residue name, e.g. Pd -> Pdx."""

    if len(symbol) >= 3:
        return symbol[:3]
    return symbol + "x" * (3 - len(symbol))


def _require_paths(paths: Any) -> None:
    for key in (
        "solvated_system",
        "pymol_system",
        "openff_interchange",
        "anchor_metadata",
    ):
        if getattr(paths, key) is None:
            msg = f"{key} output path is not configured"
            raise ValueError(msg)


def _require_openmm() -> SimpleNamespace:
    """Import OpenMM lazily for structure writing helpers only."""

    try:
        import openmm
        from openmm import app, unit
    except ImportError as error:
        msg = (
            "OpenMM is required to write solvated_system.cif from the Interchange export. "
            "Install and run from an environment with OpenMM available."
        )
        raise ImportError(msg) from error
    return SimpleNamespace(openmm=openmm, app=app, unit=unit)


def _package_version(distribution_name: str) -> str | None:
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return None


def _openff_unit_module() -> Any:
    from openff.units import unit

    return unit


def _orient_template_by_anchor(
    positions_nm: tuple[Vector3, ...],
    *,
    anchor_index: int,
    axis_index: int,
    target_anchor_nm: Vector3,
    target_direction: Vector3,
    azimuth_rad: float,
) -> tuple[Vector3, ...]:
    anchor = positions_nm[anchor_index]
    source_vector = subtract_vectors(positions_nm[axis_index], anchor)
    rotation = rotation_matrix(source_vector, target_direction)
    return tuple(
        add_vectors(
            target_anchor_nm,
            rotate_about_axis(
                matvec(rotation, subtract_vectors(position, anchor)),
                target_direction,
                azimuth_rad,
            ),
        )
        for position in positions_nm
    )


def _center_template(positions_nm: tuple[Vector3, ...], center_nm: Vector3) -> tuple[Vector3, ...]:
    current_center = centroid(positions_nm)
    return tuple(
        add_vectors(center_nm, subtract_vectors(position, current_center))
        for position in positions_nm
    )
