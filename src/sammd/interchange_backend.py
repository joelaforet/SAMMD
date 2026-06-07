"""OpenFF Interchange backend construction for SAMMD system exports."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from sammd.forcefields import get_fcc_metal_parameters
from sammd.io import safe_write_text
from sammd.metal_sulfur import METAL_SULFUR_EPSILON_KCAL_MOL, METAL_SULFUR_SIGMA_NM
from sammd.openff import (
    force_field_inputs_from_config,
    require_openff_interchange,
    require_openff_toolkit,
)
from sammd.openmm_runtime import add_sulfur_metal_lj_exceptions, require_openmm
from sammd.sam import SAMPlacement

KCAL_TO_KJ = 4.184
Vector3 = tuple[float, float, float]


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


def export_interchange_backend(plan: Any, *, overwrite: bool = False) -> BackendExportResult:
    """Construct Interchange-backed artifacts and write them to configured paths."""

    result = build_interchange_backend(plan)
    paths = plan.output_paths
    _require_paths(paths)

    safe_write_text(
        paths.openff_interchange,
        result.interchange.model_dump_json(indent=2) + "\n",
        overwrite=overwrite,
    )
    _write_pdbx(paths.topology, result.openmm_topology, result.positions, overwrite=overwrite)
    _write_pdbx(paths.positions, result.openmm_topology, result.positions, overwrite=overwrite)
    safe_write_text(
        paths.openmm_system,
        require_openmm().openmm.XmlSerializer.serialize(result.openmm_system),
        overwrite=overwrite,
    )
    safe_write_text(
        paths.anchor_metadata,
        json.dumps(_anchor_metadata(result), indent=2, sort_keys=True) + "\n",
        overwrite=overwrite,
    )
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
            "topology": paths.topology,
            "positions": paths.positions,
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
    for key in ("topology", "positions", "openff_interchange", "openmm_system", "anchor_metadata"):
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


def build_interchange_backend(plan: Any) -> BackendExportResult:
    """Build an OpenFF Interchange and patched OpenMM system from a SAMMD plan."""

    toolkit = require_openff_toolkit()
    interchange_module = require_openff_interchange()
    modules = require_openmm()
    unit = _openff_unit_module()

    force_field_inputs = [str(item) for item in force_field_inputs_from_config(plan.config)]
    force_field = toolkit.ForceField(*force_field_inputs)
    molecules: list[Any] = []
    positions_nm: list[Vector3] = []
    sulfur_indices: list[int] = []
    metal_indices: list[int] = []
    anchor_pairs: list[tuple[int, int]] = []
    component_ranges: dict[str, dict[str, int]] = {}

    shift_nm = tuple(dimension / 2.0 for dimension in plan.box_plan.dimensions_nm)

    pd_template = _metal_atom_molecule(toolkit, plan.slab.metal)
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

    templates: dict[tuple[str, str], _MoleculeTemplate] = {}
    for placement in plan.sam_placements.placements:
        template = templates.setdefault(
            (placement.component_name, placement.component_smiles),
            _molecule_template(
                toolkit,
                placement.component_smiles,
                placement.component_name,
                plan.config,
            ),
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

    for reactant in plan.solution.reactants:
        template = _molecule_template(toolkit, reactant.smiles, reactant.name, plan.config)
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

    for solvent in plan.solution.solvent_components:
        if solvent.smiles is None:
            msg = f"solvent component {solvent.name!r} requires explicit SMILES for backend export"
            raise ValueError(msg)
        template = _molecule_template(toolkit, solvent.smiles, solvent.name, plan.config)
        for transformed in _simple_molecule_centers(
            template,
            solvent.count,
            plan.box_plan.dimensions_nm,
            z_fraction=0.50,
        ):
            molecule = deepcopy(template.molecule)
            start = len(positions_nm)
            molecules.append(molecule)
            positions_nm.extend(transformed)
            _record_range(component_ranges, f"solvent:{solvent.name}", start, len(positions_nm))

    topology = toolkit.Topology.from_molecules(molecules)
    topology.box_vectors = [
        [plan.box_plan.dimensions_nm[0], 0.0, 0.0],
        [0.0, plan.box_plan.dimensions_nm[1], 0.0],
        [0.0, 0.0, plan.box_plan.dimensions_nm[2]],
    ] * unit.nanometer
    positions = positions_nm * unit.nanometer
    interchange = interchange_module.Interchange.from_smirnoff(
        force_field,
        topology,
        box=topology.box_vectors,
        positions=positions,
        charge_from_molecules=_unique_charge_molecules(molecules),
    )
    openmm_system = interchange.to_openmm()
    add_sulfur_metal_lj_exceptions(
        openmm_system,
        tuple(anchor_pairs),
        sigma_nm=METAL_SULFUR_SIGMA_NM,
        epsilon_kj_mol=METAL_SULFUR_EPSILON_KCAL_MOL * KCAL_TO_KJ,
        unit_module=modules.unit,
    )
    openmm_topology = interchange.to_openmm_topology()
    openmm_positions = modules.unit.Quantity(
        [modules.openmm.Vec3(*position) for position in positions_nm],
        modules.unit.nanometer,
    )
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


def _molecule_template(toolkit: Any, smiles: str, name: str, config: Any) -> _MoleculeTemplate:
    molecule = toolkit.Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    molecule.name = name
    molecule.generate_conformers(n_conformers=1)
    molecule.assign_partial_charges(config.parameterization.charge_model)
    conformer = molecule.conformers[0]
    positions = tuple(
        (
            conformer[index][0].m_as("nanometer"),
            conformer[index][1].m_as("nanometer"),
            conformer[index][2].m_as("nanometer"),
        )
        for index in range(molecule.n_atoms)
    )
    return _MoleculeTemplate(
        molecule=molecule,
        positions_nm=positions,
        atom_symbols=tuple(atom.symbol for atom in molecule.atoms),
    )


def _metal_atom_molecule(toolkit: Any, symbol: str) -> Any:
    get_fcc_metal_parameters(symbol)
    molecule = toolkit.Molecule.from_smiles(f"[{symbol}]", allow_undefined_stereo=True)
    molecule.name = symbol
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
        key=lambda index: _distance(
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
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        require_openmm().app.PDBxFile.writeFile(topology, positions, handle, keepIds=True)


def _require_paths(paths: Any) -> None:
    for key in ("topology", "positions", "openff_interchange", "openmm_system", "anchor_metadata"):
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
    source_vector = _subtract_vectors(positions_nm[axis_index], anchor)
    rotation = _rotation_matrix(source_vector, target_direction)
    return tuple(
        _add_vectors(
            target_anchor_nm,
            _rotate_about_axis(
                _matvec(rotation, _subtract_vectors(position, anchor)),
                target_direction,
                azimuth_rad,
            ),
        )
        for position in positions_nm
    )


def _center_template(positions_nm: tuple[Vector3, ...], center_nm: Vector3) -> tuple[Vector3, ...]:
    current_center = _centroid(positions_nm)
    return tuple(
        _add_vectors(center_nm, _subtract_vectors(position, current_center))
        for position in positions_nm
    )


def _rotation_matrix(source: Vector3, target: Vector3) -> tuple[Vector3, Vector3, Vector3]:
    source_unit = _normalize(source)
    target_unit = _normalize(target)
    cross = _cross_product(source_unit, target_unit)
    sine = _norm(cross)
    cosine = _dot_product(source_unit, target_unit)
    if sine < 1.0e-12:
        if cosine > 0.0:
            return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        orthogonal = _normalize((1.0, 0.0, 0.0) if abs(source_unit[0]) < 0.9 else (0.0, 1.0, 0.0))
        cross = _normalize(_cross_product(source_unit, orthogonal))
        return _axis_angle_matrix(cross, math.pi)
    skew = _skew_matrix(cross)
    skew_squared = _matrix_multiply(skew, skew)
    return _matrix_add(
        _matrix_add(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), skew),
        _matrix_scale(skew_squared, (1.0 - cosine) / (sine * sine)),
    )


def _axis_angle_matrix(axis: Vector3, angle_rad: float) -> tuple[Vector3, Vector3, Vector3]:
    skew = _skew_matrix(axis)
    return _matrix_add(
        _matrix_add(
            _matrix_scale(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), math.cos(angle_rad)),
            _matrix_scale(skew, math.sin(angle_rad)),
        ),
        _matrix_scale(_outer_product(axis), 1.0 - math.cos(angle_rad)),
    )


def _rotate_about_axis(vector: Vector3, axis: Vector3, angle_rad: float) -> Vector3:
    unit_axis = _normalize(axis)
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    parallel = _scale_vector(unit_axis, _dot_product(unit_axis, vector) * (1.0 - cosine))
    cross = _cross_product(unit_axis, vector)
    return _add_vectors(
        _add_vectors(_scale_vector(vector, cosine), _scale_vector(cross, sine)),
        parallel,
    )


def _add_vectors(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _subtract_vectors(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _scale_vector(vector: Vector3, scale: float) -> Vector3:
    return (vector[0] * scale, vector[1] * scale, vector[2] * scale)


def _distance(left: Vector3, right: Vector3) -> float:
    return _norm(_subtract_vectors(left, right))


def _norm(vector: Vector3) -> float:
    return math.sqrt(_dot_product(vector, vector))


def _normalize(vector: Vector3) -> Vector3:
    vector_norm = _norm(vector)
    if vector_norm < 1.0e-15:
        msg = "cannot normalize a zero vector"
        raise ValueError(msg)
    return (vector[0] / vector_norm, vector[1] / vector_norm, vector[2] / vector_norm)


def _dot_product(left: Vector3, right: Vector3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _cross_product(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _centroid(positions: tuple[Vector3, ...]) -> Vector3:
    count = len(positions)
    return (
        sum(position[0] for position in positions) / count,
        sum(position[1] for position in positions) / count,
        sum(position[2] for position in positions) / count,
    )


def _matvec(matrix: tuple[Vector3, Vector3, Vector3], vector: Vector3) -> Vector3:
    return tuple(_dot_product(row, vector) for row in matrix)  # type: ignore[return-value]


def _skew_matrix(vector: Vector3) -> tuple[Vector3, Vector3, Vector3]:
    x, y, z = vector
    return ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0))


def _outer_product(vector: Vector3) -> tuple[Vector3, Vector3, Vector3]:
    return tuple(
        tuple(vector[row] * vector[column] for column in range(3)) for row in range(3)
    )  # type: ignore[return-value]


def _matrix_multiply(
    left: tuple[Vector3, Vector3, Vector3],
    right: tuple[Vector3, Vector3, Vector3],
) -> tuple[Vector3, Vector3, Vector3]:
    columns = tuple((right[0][i], right[1][i], right[2][i]) for i in range(3))
    return tuple(tuple(_dot_product(row, column) for column in columns) for row in left)  # type: ignore[return-value]


def _matrix_add(
    left: tuple[Vector3, Vector3, Vector3],
    right: tuple[Vector3, Vector3, Vector3],
) -> tuple[Vector3, Vector3, Vector3]:
    return tuple(
        tuple(left[row][column] + right[row][column] for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _matrix_scale(
    matrix: tuple[Vector3, Vector3, Vector3], scale: float
) -> tuple[Vector3, Vector3, Vector3]:
    return tuple(tuple(value * scale for value in row) for row in matrix)  # type: ignore[return-value]
