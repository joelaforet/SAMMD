"""OpenFF Interchange backend construction for SAMMD system exports."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from time import perf_counter
from typing import Any

from sammd.backends.forcefields import get_fcc_metal_parameters
from sammd.backends.openff import (
    force_field_inputs_from_config,
    require_openff_interchange,
    require_openff_toolkit,
)
from sammd.backends.openmm_runtime import add_sulfur_metal_lj_exceptions, require_openmm
from sammd.core.io import safe_write_text
from sammd.model.metal_sulfur import METAL_SULFUR_EPSILON_KCAL_MOL, METAL_SULFUR_SIGMA_NM
from sammd.model.sam import SAMPlacement
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

KCAL_TO_KJ = 4.184


@dataclass(frozen=True)
class BackendExportResult:
    """Artifacts and metadata from a completed Interchange backend export."""

    interchange: Any
    openmm_topology: Any
    openmm_system: Any
    positions: Any
    positions_nm: tuple[Vector3, ...]
    sulfur_indices: tuple[int, ...]
    metal_indices: tuple[int, ...]
    anchor_pairs: tuple[tuple[int, int], ...]
    component_ranges: dict[str, dict[str, int]]
    files: dict[str, Path]
    openff_toolkit_version: str | None
    openff_interchange_version: str | None


@dataclass(frozen=True)
class _MoleculeTemplate:
    molecule: Any
    positions_nm: tuple[Vector3, ...]
    atom_symbols: tuple[str, ...]


ProgressCallback = Callable[[str], None]


def _progress(callback: ProgressCallback | None, message: str) -> None:
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

    _progress(progress, "Building OpenFF Interchange and patched OpenMM System")
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
    _progress(progress, "Writing system.xml")
    safe_write_text(
        paths.openmm_system,
        require_openmm().openmm.XmlSerializer.serialize(result.openmm_system),
        overwrite=overwrite,
    )
    _timed_progress(progress, "  system.xml written", stage_start)
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
        openmm_system=result.openmm_system,
        positions=result.positions,
        positions_nm=result.positions_nm,
        sulfur_indices=result.sulfur_indices,
        metal_indices=result.metal_indices,
        anchor_pairs=result.anchor_pairs,
        component_ranges=result.component_ranges,
        files={
            "solvated_system": paths.solvated_system,
            "openff_interchange": paths.openff_interchange,
            "openmm_system": paths.openmm_system,
            "anchor_metadata": paths.anchor_metadata,
        },
        openff_toolkit_version=result.openff_toolkit_version,
        openff_interchange_version=result.openff_interchange_version,
    )


def backend_build_summary(plan: Any, result: BackendExportResult) -> dict[str, object]:
    """Return a build summary updated for completed backend exports."""

    summary = plan.build_summary()
    artifacts = dict(summary["artifacts"])
    for key in ("solvated_system", "openff_interchange", "openmm_system", "anchor_metadata"):
        artifact = dict(artifacts[key])
        artifact["status"] = "current"
        artifact["available"] = True
        if key == "openff_interchange":
            artifact["constructed"] = True
            artifact["openff_interchange_package_version"] = result.openff_interchange_version
        artifacts[key] = artifact
    engine_exports = dict(summary["engine_exports"])
    openmm_export = dict(engine_exports["openmm"])
    openmm_export["status"] = "current"
    openmm_export["available"] = True
    engine_exports["openmm"] = openmm_export
    summary["artifacts"] = artifacts
    summary["engine_exports"] = engine_exports
    summary["full_construction_available"] = True
    summary["backend_export"] = {
        "mode": "openff_interchange_with_openmm_pair_overrides",
        "openff_toolkit_version": result.openff_toolkit_version,
        "openff_interchange_version": result.openff_interchange_version,
        "atom_count": len(result.positions_nm),
        "metal_atom_count": len(result.metal_indices),
        "sam_sulfur_count": len(result.sulfur_indices),
        "sulfur_metal_pair_count": len(result.anchor_pairs),
        "metal_sulfur_override": _anchor_metadata(result),
    }
    return summary


def build_interchange_backend(
    plan: Any,
    *,
    progress: ProgressCallback | None = None,
) -> BackendExportResult:
    """Build an OpenFF Interchange and patched OpenMM system from a SAMMD plan."""

    if plan.solution.salts:
        names = ", ".join(salt.name for salt in plan.solution.salts)
        msg = f"Interchange backend export does not yet support salts: {names}"
        raise NotImplementedError(msg)

    stage_start = perf_counter()
    _progress(progress, "Importing OpenFF/OpenMM dependencies")
    toolkit = require_openff_toolkit()
    interchange_module = require_openff_interchange()
    modules = require_openmm()
    unit = _openff_unit_module()
    _timed_progress(progress, "  OpenFF/OpenMM dependencies ready", stage_start)

    stage_start = perf_counter()
    _progress(progress, "Loading force fields")
    force_field_inputs = [str(item) for item in force_field_inputs_from_config(plan.config)]
    force_field = toolkit.ForceField(*force_field_inputs)
    _timed_progress(progress, "  force fields ready", stage_start)
    molecules: list[Any] = []
    positions_nm: list[Vector3] = []
    sulfur_indices: list[int] = []
    metal_indices: list[int] = []
    anchor_pairs: list[tuple[int, int]] = []
    component_ranges: dict[str, dict[str, int]] = {}

    stage_start = perf_counter()
    _progress(progress, f"Preparing {len(plan.slab.positions_nm)} metal atoms")
    shift_nm = tuple(dimension / 2.0 for dimension in plan.box_plan.dimensions_nm)

    pd_template = _metal_atom_molecule(toolkit, plan.slab.metal)
    stage_start = _timed_progress(progress, "  metal template ready", stage_start)
    _append_molecule_block(
        molecules,
        positions_nm,
        component_ranges,
        "metal_slab",
        [pd_template] * len(plan.slab.positions_nm),
        [
            tuple(position + shift for position, shift in zip(pos, shift_nm, strict=True))
            for pos in plan.slab.positions_nm
        ],
    )
    metal_indices.extend(range(len(plan.slab.positions_nm)))
    _timed_progress(progress, "  metal block appended", stage_start)

    stage_start = perf_counter()
    _progress(progress, f"Preparing {len(plan.sam_placements.placements)} SAM molecules")
    templates: dict[tuple[str, str], _MoleculeTemplate] = {}
    for placement_index, placement in enumerate(plan.sam_placements.placements, start=1):
        template_key = (placement.component_name, placement.component_smiles)
        template = templates.get(template_key)
        if template is None:
            template = _molecule_template(
                toolkit,
                placement.component_smiles,
                placement.component_name,
                plan.config,
                progress=progress,
            )
            templates[template_key] = template
            stage_start = _timed_progress(
                progress,
                f"  SAM template ready: {placement.component_name}",
                stage_start,
            )
        molecule, transformed = _placed_sam_molecule(template, placement, shift_nm)
        start = len(positions_nm)
        molecules.append(molecule)
        positions_nm.extend(transformed)
        sulfur_index = start + _single_atom_index(template.atom_symbols, "S", "SAM molecule")
        sulfur_indices.append(sulfur_index)
        anchor_pairs.extend(
            (sulfur_index, int(metal_index))
            for metal_index in placement.anchor_pose.nearest_metal_atom_indices
        )
        _record_range(component_ranges, f"sam:{placement.component_name}", start, len(positions_nm))
        if placement_index % 25 == 0 or placement_index == len(plan.sam_placements.placements):
            stage_start = _timed_progress(
                progress,
                f"  placed {placement_index}/{len(plan.sam_placements.placements)} SAM molecules",
                stage_start,
            )

    reactant_count = sum(reactant.count for reactant in plan.solution.reactants)
    stage_start = perf_counter()
    _progress(progress, f"Preparing {reactant_count} reactants")
    for reactant in plan.solution.reactants:
        template = _molecule_template(
            toolkit,
            reactant.smiles,
            reactant.name,
            plan.config,
            progress=progress,
        )
        stage_start = _timed_progress(
            progress,
            f"  reactant template ready: {reactant.name}",
            stage_start,
        )
        for transformed in _simple_molecule_centers(
            template,
            reactant.count,
            plan.box_plan.dimensions_nm,
            z_fraction=0.72,
        ):
            molecule = deepcopy(template.molecule)
            start = len(positions_nm)
            molecules.append(molecule)
            positions_nm.extend(transformed)
            _record_range(component_ranges, f"reactant:{reactant.name}", start, len(positions_nm))
    _timed_progress(progress, "  reactants placed", stage_start)

    solvent_count = sum(solvent.count for solvent in plan.solution.solvent_components)
    stage_start = perf_counter()
    _progress(progress, f"Preparing {solvent_count} solvent molecules")
    for solvent in plan.solution.solvent_components:
        if solvent.smiles is None:
            msg = f"solvent component {solvent.name!r} requires explicit SMILES for backend export"
            raise ValueError(msg)
        template = _molecule_template(
            toolkit,
            solvent.smiles,
            solvent.name,
            plan.config,
            progress=progress,
        )
        stage_start = _timed_progress(
            progress,
            f"  solvent template ready: {solvent.name}",
            stage_start,
        )
        centered_solvent = _simple_molecule_centers(
            template,
            solvent.count,
            plan.box_plan.dimensions_nm,
            z_fraction=0.50,
        )
        stage_start = _timed_progress(
            progress,
            f"  solvent centers ready: {solvent.name}",
            stage_start,
        )
        for solvent_index, transformed in enumerate(centered_solvent, start=1):
            molecule = deepcopy(template.molecule)
            start = len(positions_nm)
            molecules.append(molecule)
            positions_nm.extend(transformed)
            _record_range(component_ranges, f"solvent:{solvent.name}", start, len(positions_nm))
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
        [plan.box_plan.dimensions_nm[0], 0.0, 0.0],
        [0.0, plan.box_plan.dimensions_nm[1], 0.0],
        [0.0, 0.0, plan.box_plan.dimensions_nm[2]],
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
    _progress(progress, "Exporting OpenMM System")
    openmm_system = interchange.to_openmm()
    _timed_progress(progress, "  OpenMM System ready", stage_start)
    stage_start = perf_counter()
    _progress(progress, f"Applying {len(anchor_pairs)} sulfur-metal pair overrides")
    add_sulfur_metal_lj_exceptions(
        openmm_system,
        tuple(anchor_pairs),
        sigma_nm=METAL_SULFUR_SIGMA_NM,
        epsilon_kj_mol=METAL_SULFUR_EPSILON_KCAL_MOL * KCAL_TO_KJ,
        unit_module=modules.unit,
    )
    _timed_progress(progress, "  sulfur-metal overrides applied", stage_start)
    stage_start = perf_counter()
    _progress(progress, "Exporting OpenMM topology and positions")
    openmm_topology = interchange.to_openmm_topology()
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
        openmm_system=openmm_system,
        positions=openmm_positions,
        positions_nm=tuple(positions_nm),
        sulfur_indices=tuple(sulfur_indices),
        metal_indices=tuple(metal_indices),
        anchor_pairs=tuple(anchor_pairs),
        component_ranges=component_ranges,
        files={},
        openff_toolkit_version=_package_version("openff-toolkit"),
        openff_interchange_version=_package_version("openff-interchange"),
    )


def _molecule_template(
    toolkit: Any,
    smiles: str,
    name: str,
    config: Any,
    *,
    progress: ProgressCallback | None = None,
) -> _MoleculeTemplate:
    stage_start = perf_counter()
    _progress(progress, f"  template {name}: parsing SMILES")
    molecule = toolkit.Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    molecule.name = name
    stage_start = _timed_progress(progress, f"  template {name}: SMILES parsed", stage_start)
    _progress(progress, f"  template {name}: generating conformer")
    molecule.generate_conformers(n_conformers=1)
    stage_start = _timed_progress(progress, f"  template {name}: conformer ready", stage_start)
    _progress(
        progress,
        f"  template {name}: assigning {config.parameterization.charge_model} charges",
    )
    molecule.assign_partial_charges(config.parameterization.charge_model)
    stage_start = _timed_progress(progress, f"  template {name}: charges assigned", stage_start)
    conformer = molecule.conformers[0]
    positions = tuple(
        (
            conformer[index][0].m_as("nanometer"),
            conformer[index][1].m_as("nanometer"),
            conformer[index][2].m_as("nanometer"),
        )
        for index in range(molecule.n_atoms)
    )
    _timed_progress(progress, f"  template {name}: positions extracted", stage_start)
    return _MoleculeTemplate(
        molecule=molecule,
        positions_nm=positions,
        atom_symbols=tuple(atom.symbol for atom in molecule.atoms),
    )


def _metal_atom_molecule(toolkit: Any, symbol: str) -> Any:
    get_fcc_metal_parameters(symbol)
    molecule = toolkit.Molecule.from_smiles(f"[{symbol}]", allow_undefined_stereo=True)
    molecule.name = _metal_residue_name(symbol)
    molecule.atoms[0].name = symbol
    molecule.assign_partial_charges("zeros")
    return molecule


def _placed_sam_molecule(
    template: _MoleculeTemplate,
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


def _simple_molecule_centers(
    template: _MoleculeTemplate,
    count: int,
    dimensions_nm: Vector3,
    *,
    z_fraction: float,
) -> tuple[tuple[Vector3, ...], ...]:
    if count <= 0:
        return ()
    columns = max(1, math.ceil(math.sqrt(count)))
    rows = max(1, math.ceil(count / columns))
    centered = []
    for index in range(count):
        col = index % columns
        row = index // columns
        center = (
            dimensions_nm[0] * (col + 0.5) / columns,
            dimensions_nm[1] * (row + 0.5) / rows,
            dimensions_nm[2] * min(0.95, z_fraction + 0.02 * (index % 5)),
        )
        centered.append(_center_template(template.positions_nm, center))
    return tuple(centered)


def _append_molecule_block(
    molecules: list[Any],
    positions_nm: list[Vector3],
    component_ranges: dict[str, dict[str, int]],
    name: str,
    block_molecules: list[Any],
    block_positions: list[Vector3],
) -> None:
    start = len(positions_nm)
    molecules.extend(deepcopy(molecule) for molecule in block_molecules)
    positions_nm.extend(block_positions)
    _record_range(component_ranges, name, start, len(positions_nm))


def _record_range(
    component_ranges: dict[str, dict[str, int]], name: str, start: int, stop: int
) -> None:
    existing = component_ranges.get(name)
    if existing is None:
        component_ranges[name] = {"start": start, "stop": stop}
    else:
        existing["stop"] = stop


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


def _terminal_heavy_axis_index(template: _MoleculeTemplate, anchor_index: int) -> int:
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
    return {
        "mode": "openmm_nonbonded_exception_post_export",
        "sulfur_indices": list(result.sulfur_indices),
        "metal_indices": list(result.metal_indices),
        "sulfur_metal_pairs": [list(pair) for pair in result.anchor_pairs],
        "sigma_nm": METAL_SULFUR_SIGMA_NM,
        "epsilon_kj_mol": METAL_SULFUR_EPSILON_KCAL_MOL * KCAL_TO_KJ,
    }


def _write_pdbx(path: Path, topology: Any, positions: Any, *, overwrite: bool) -> None:
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
            require_openmm().app.PDBxFile.writeFile(topology, positions, handle, keepIds=True)
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


def _ensure_openmm_atom_names(topology: Any) -> None:
    """Fill missing atom names so PDBx/mmCIF rows have all declared fields."""

    for atom in topology.atoms():
        if getattr(atom, "name", None):
            continue
        element = getattr(atom, "element", None)
        symbol = getattr(element, "symbol", None) or "X"
        atom.name = symbol


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
        "openff_interchange",
        "openmm_system",
        "anchor_metadata",
    ):
        if getattr(paths, key) is None:
            msg = f"{key} output path is not configured"
            raise ValueError(msg)


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
