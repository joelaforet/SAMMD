"""Dependency-free PACKMOL input planning and execution helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Sized
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any

from sammd.core.io import AtomRecord

Vector3 = tuple[float, float, float]
BoxBounds = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class PackmolStructure:
    """One molecular structure entry in a PACKMOL job."""

    name: str
    path: str | Path
    count: int
    fixed: bool = False
    atom_count: int | None = None
    output_group: str | None = None
    inside_box_bounds_nm: BoxBounds | None = None


@dataclass(frozen=True)
class PackmolJob:
    """PACKMOL input plan with orthorhombic bounds in nanometers."""

    output_path: str | Path
    structures: tuple[PackmolStructure, ...] | list[PackmolStructure]
    box_bounds_nm: BoxBounds
    tolerance_angstrom: float = 1.8
    nloop: int = 200
    filetype: str = "pdb"
    movebadrandom: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "structures", tuple(self.structures))


@dataclass(frozen=True)
class PackmolResult:
    """Result from running PACKMOL."""

    output_path: Path
    stdout_path: Path
    returncode: int
    stdout: str
    grouped_positions_nm: dict[str, tuple[Vector3, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class PackmolMoleculeTemplate:
    """Atom labels and coordinates for one molecule written as a PACKMOL PDB."""

    residue_name: str
    positions_nm: tuple[Vector3, ...]
    atom_symbols: tuple[str, ...]
    atom_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not (
            len(self.positions_nm) == len(self.atom_symbols) == len(self.atom_names)
        ):
            msg = "PACKMOL molecule template fields must have matching atom counts"
            raise ValueError(msg)


@dataclass(frozen=True)
class PackmolSolventComponent:
    """One solvent component to place in a shared PACKMOL reservoir job."""

    name: str
    template: PackmolMoleculeTemplate
    count: int


def zero_origin_box_bounds(dimensions_nm: Any) -> BoxBounds:
    """Return ``((0, x), (0, y), (0, z))`` bounds from dimensions in nanometers."""

    values = getattr(dimensions_nm, "dimensions_nm", dimensions_nm)
    dimensions = _validate_vector3(values, "dimensions_nm")
    if any(value <= 0.0 for value in dimensions):
        msg = "dimensions_nm values must be positive finite numbers"
        raise ValueError(msg)
    return ((0.0, dimensions[0]), (0.0, dimensions[1]), (0.0, dimensions[2]))


def build_packmol_input(job: PackmolJob) -> str:
    """Render deterministic PACKMOL input text for a validated job."""

    _validate_job(job)
    lines = [
        f"tolerance {_format_float(job.tolerance_angstrom)}",
        f"filetype {job.filetype}",
        f"output {Path(job.output_path)}",
        f"nloop {job.nloop}",
    ]
    if job.movebadrandom:
        lines.append("movebadrandom")
    lines.append("")

    for structure in job.structures:
        lines.extend(
            [
                f"structure {Path(structure.path)}",
                f"  number {structure.count}",
            ]
        )
        if structure.fixed:
            lines.append("  fixed 0. 0. 0. 0. 0. 0.")
        else:
            bounds_nm = structure.inside_box_bounds_nm or job.box_bounds_nm
            bounds_angstrom = tuple((lower * 10.0, upper * 10.0) for lower, upper in bounds_nm)
            box_tokens = (
                bounds_angstrom[0][0],
                bounds_angstrom[1][0],
                bounds_angstrom[2][0],
                bounds_angstrom[0][1],
                bounds_angstrom[1][1],
                bounds_angstrom[2][1],
            )
            lines.append("  inside box " + " ".join(_format_float(value) for value in box_tokens))
        lines.extend(["end structure", ""])
    return "\n".join(lines).rstrip() + "\n"


def write_packmol_input(job: PackmolJob, path: str | Path, *, overwrite: bool = False) -> Path:
    """Write PACKMOL input text atomically, refusing overwrite by default."""

    return _safe_write_text(path, build_packmol_input(job), overwrite=overwrite)


def run_packmol(
    job: PackmolJob,
    input_path: str | Path,
    working_dir: str | Path,
    stdout_path: str | Path,
    *,
    executable: str = "packmol",
) -> PackmolResult:
    """Run PACKMOL and capture stdout to a file."""

    resolved_executable = shutil.which(executable)
    if resolved_executable is None:
        msg = f"PACKMOL executable not found: {executable!r}"
        raise RuntimeError(msg)

    _validate_job(job)
    workdir = Path(working_dir)
    workdir.mkdir(parents=True, exist_ok=True)
    stdout_destination = Path(stdout_path)
    stdout_destination.parent.mkdir(parents=True, exist_ok=True)
    with Path(input_path).open("r", encoding="utf-8") as stdin_handle:
        completed = subprocess.run(
            [resolved_executable],
            cwd=workdir,
            stdin=stdin_handle,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    stdout_destination.write_text(completed.stdout, encoding="utf-8")
    return PackmolResult(
        output_path=Path(job.output_path),
        stdout_path=stdout_destination,
        returncode=completed.returncode,
        stdout=completed.stdout,
    )


def pack_fixed_solute_with_solvent(
    *,
    solute_records: tuple[AtomRecord, ...] | list[AtomRecord],
    solvent_template: PackmolMoleculeTemplate,
    solvent_name: str,
    solvent_count: int,
    dimensions_nm: Any,
    working_dir: str | Path,
    solvent_regions_nm: tuple[BoxBounds, ...] | list[BoxBounds] | None = None,
    tolerance_angstrom: float = 1.8,
    nloop: int = 200,
) -> tuple[tuple[Vector3, ...], ...]:
    """Pack solvent in explicit regions around one fixed solute and return positions."""

    if solvent_count <= 0:
        return ()
    solute_atoms = tuple(solute_records)
    if not solute_atoms:
        msg = "fixed-solute packing requires at least one solute atom record"
        raise ValueError(msg)

    box_bounds_nm = zero_origin_box_bounds(dimensions_nm)
    regions_nm = _require_explicit_solvent_regions(solvent_regions_nm)

    workdir = Path(working_dir)
    workdir.mkdir(parents=True, exist_ok=True)
    solute_path = workdir / "fixed_solute.pdb"
    solvent_path = workdir / f"{packmol_file_stem(solvent_name)}.pdb"
    output_path = workdir / "packmol_output.pdb"
    input_path = workdir / "packmol_input.inp"
    stdout_path = workdir / "packmol_stdout.log"

    write_atom_records_pdb(solute_path, solute_atoms)
    write_template_pdb(solvent_path, solvent_template)
    solvent_counts = split_count_by_region_volume(solvent_count, regions_nm)
    solvent_structures = tuple(
        PackmolStructure(
            f"{solvent_name}_{region_index}",
            solvent_path.name,
            region_count,
            inside_box_bounds_nm=region,
        )
        for region_index, (region, region_count) in enumerate(
            zip(regions_nm, solvent_counts, strict=True),
            start=1,
        )
        if region_count > 0
    )
    job = PackmolJob(
        output_path=output_path.name,
        structures=(
            PackmolStructure("solute", solute_path.name, 1, fixed=True),
            *solvent_structures,
        ),
        box_bounds_nm=box_bounds_nm,
        tolerance_angstrom=tolerance_angstrom,
        nloop=nloop,
    )
    input_path.write_text(build_packmol_input(job), encoding="utf-8")
    result = run_packmol(job, input_path, workdir, stdout_path)
    if result.returncode != 0 or "Success!" not in result.stdout:
        msg = f"PACKMOL failed while placing {solvent_name}; see {stdout_path}"
        raise RuntimeError(msg)

    packed_positions = read_pdb_positions_nm(output_path)
    atoms_per_solvent = len(solvent_template.positions_nm)
    solvent_atom_count = solvent_count * atoms_per_solvent
    expected_atom_count = len(solute_atoms) + solvent_atom_count
    if len(packed_positions) != expected_atom_count:
        msg = (
            f"PACKMOL output contains {len(packed_positions)} atoms, expected "
            f"{expected_atom_count} ({len(solute_atoms)} solute + "
            f"{solvent_atom_count} solvent)"
        )
        raise RuntimeError(msg)

    solvent_start = len(solute_atoms)
    solvent_stop = solvent_start + solvent_atom_count
    return tuple(
        packed_positions[index : index + atoms_per_solvent]
        for index in range(solvent_start, solvent_stop, atoms_per_solvent)
    )


def pack_fixed_solute_with_solvent_components(
    *,
    solute_records: tuple[AtomRecord, ...] | list[AtomRecord],
    solvent_components: tuple[PackmolSolventComponent, ...] | list[PackmolSolventComponent],
    dimensions_nm: Any,
    working_dir: str | Path,
    solvent_regions_nm: tuple[BoxBounds, ...] | list[BoxBounds] | None = None,
    tolerance_angstrom: float = 1.8,
    nloop: int = 200,
) -> dict[str, tuple[tuple[Vector3, ...], ...]]:
    """Pack all solvent components against one fixed solute in shared regions.

    The same explicit solvent regions are applied to every component so mixed
    solvents occupy common reservoirs instead of sequentially shrinking them.
    """

    raw_components = tuple(solvent_components)
    for component in raw_components:
        _validate_solvent_component(component)
    components = tuple(component for component in raw_components if component.count > 0)
    if not components:
        return {}
    solute_atoms = tuple(solute_records)
    if not solute_atoms:
        msg = "fixed-solute packing requires at least one solute atom record"
        raise ValueError(msg)

    box_bounds_nm = zero_origin_box_bounds(dimensions_nm)
    regions_nm = _require_explicit_solvent_regions(solvent_regions_nm)

    workdir = Path(working_dir)
    workdir.mkdir(parents=True, exist_ok=True)
    solute_path = workdir / "fixed_solute.pdb"
    output_path = workdir / "packmol_output.pdb"
    input_path = workdir / "packmol_input.inp"
    stdout_path = workdir / "packmol_stdout.log"

    write_atom_records_pdb(solute_path, solute_atoms)
    solvent_structures: list[PackmolStructure] = []
    component_atom_counts: dict[str, int] = {}
    component_region_counts: dict[str, tuple[int, ...]] = {}
    for component_index, component in enumerate(components):
        solvent_path = workdir / f"{packmol_file_stem(component.name)}.pdb"
        write_template_pdb(solvent_path, component.template)
        component_atom_counts[component.name] = len(component.template.positions_nm)
        region_counts = split_count_by_region_volume(
            component.count,
            regions_nm,
            tie_break_offset=component_index,
        )
        component_region_counts[component.name] = region_counts
        solvent_structures.extend(
            PackmolStructure(
                f"{component.name}_{region_index}",
                solvent_path.name,
                region_count,
                inside_box_bounds_nm=region,
            )
            for region_index, (region, region_count) in enumerate(
                zip(regions_nm, region_counts, strict=True),
                start=1,
            )
            if region_count > 0
        )

    job = PackmolJob(
        output_path=output_path.name,
        structures=(
            PackmolStructure("solute", solute_path.name, 1, fixed=True),
            *solvent_structures,
        ),
        box_bounds_nm=box_bounds_nm,
        tolerance_angstrom=tolerance_angstrom,
        nloop=nloop,
    )
    input_path.write_text(build_packmol_input(job), encoding="utf-8")
    result = run_packmol(job, input_path, workdir, stdout_path)
    if result.returncode != 0 or "Success!" not in result.stdout:
        msg = f"PACKMOL failed while placing solvent mixture; see {stdout_path}"
        raise RuntimeError(msg)

    packed_positions = read_pdb_positions_nm(output_path)
    solvent_atom_count = sum(
        component.count * component_atom_counts[component.name] for component in components
    )
    expected_atom_count = len(solute_atoms) + solvent_atom_count
    if len(packed_positions) != expected_atom_count:
        msg = (
            f"PACKMOL output contains {len(packed_positions)} atoms, expected "
            f"{expected_atom_count} ({len(solute_atoms)} solute + {solvent_atom_count} solvent)"
        )
        raise RuntimeError(msg)

    offset = len(solute_atoms)
    packed_by_component: dict[str, list[tuple[Vector3, ...]]] = {
        component.name: [] for component in components
    }
    for component in components:
        atoms_per_solvent = component_atom_counts[component.name]
        for region_count in component_region_counts[component.name]:
            for _ in range(region_count):
                stop = offset + atoms_per_solvent
                packed_by_component[component.name].append(packed_positions[offset:stop])
                offset = stop
    return {name: tuple(positions) for name, positions in packed_by_component.items()}


def solvent_regions_around_solute(
    solute_records: tuple[AtomRecord, ...] | list[AtomRecord],
    box_bounds_nm: BoxBounds,
    clearance_nm: float,
) -> tuple[BoxBounds, ...]:
    """Return bottom and top solvent regions outside actual solute z extents."""

    _validate_bounds(box_bounds_nm)
    _validate_positive_finite_number(clearance_nm, "clearance_nm")
    solute_atoms = tuple(solute_records)
    if not solute_atoms:
        msg = "solvent region planning requires at least one solute atom record"
        raise ValueError(msg)
    solute_z_values = tuple(record.coordinates_nm[2] for record in solute_atoms)
    bottom_region = (
        box_bounds_nm[0],
        box_bounds_nm[1],
        (box_bounds_nm[2][0], min(solute_z_values) - clearance_nm),
    )
    top_region = (
        box_bounds_nm[0],
        box_bounds_nm[1],
        (max(solute_z_values) + clearance_nm, box_bounds_nm[2][1]),
    )
    return tuple(region for region in (bottom_region, top_region) if region[2][1] > region[2][0])


def split_count_by_region_volume(
    count: int,
    regions_nm: tuple[BoxBounds, ...],
    *,
    tie_break_offset: int = 0,
) -> tuple[int, ...]:
    """Split a molecule count across regions proportional to region volume.

    Parameters
    ----------
    count
        Number of molecules to split.
    regions_nm
        Candidate solvent packing regions in nanometers.
    tie_break_offset
        Cyclic region offset used to distribute equal remainders across components.
    """

    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        msg = "count must be a non-negative integer"
        raise ValueError(msg)
    if not regions_nm:
        msg = "at least one solvent packing region is required"
        raise ValueError(msg)
    volumes = tuple(_region_volume_nm3(region) for region in regions_nm)
    total_volume = sum(volumes)
    if total_volume <= 0.0:
        msg = "solvent packing regions must have positive total volume"
        raise ValueError(msg)
    raw_counts = tuple(count * volume / total_volume for volume in volumes)
    floor_counts = [int(raw_count) for raw_count in raw_counts]
    remaining = count - sum(floor_counts)
    if isinstance(tie_break_offset, bool) or not isinstance(tie_break_offset, int):
        msg = "tie_break_offset must be an integer"
        raise ValueError(msg)
    region_count = len(regions_nm)
    remainders = sorted(
        (
            (-(raw_count - floor_count), (index - tie_break_offset) % region_count, index)
            for index, (raw_count, floor_count) in enumerate(
                zip(raw_counts, floor_counts, strict=True)
            )
        ),
    )
    for _, _, index in remainders[:remaining]:
        floor_counts[index] += 1
    return tuple(floor_counts)


def _require_explicit_solvent_regions(
    solvent_regions_nm: tuple[BoxBounds, ...] | list[BoxBounds] | None,
) -> tuple[BoxBounds, ...]:
    """Return validated explicit regions for high-level SAMMD solvent packing.

    Parameters
    ----------
    solvent_regions_nm
        Explicit solvent packing regions in nanometers.

    Returns
    -------
    tuple[BoxBounds, ...]
        Validated immutable region bounds.
    """

    if solvent_regions_nm is None:
        msg = "explicit solvent_regions_nm are required for SAMMD solvent packing"
        raise ValueError(msg)
    regions_nm = tuple(solvent_regions_nm)
    if not regions_nm:
        msg = "at least one explicit solvent packing region is required"
        raise ValueError(msg)
    for region in regions_nm:
        _validate_bounds(region)
    return regions_nm


def read_pdb_positions_nm(path: str | Path) -> tuple[Vector3, ...]:
    """Read ATOM/HETATM coordinates from a PDB file and return nanometers."""

    positions: list[Vector3] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        try:
            x_angstrom = float(line[30:38])
            y_angstrom = float(line[38:46])
            z_angstrom = float(line[46:54])
        except ValueError as error:
            msg = f"invalid PDB coordinate fields on line {line_number}"
            raise ValueError(msg) from error
        positions.append((x_angstrom / 10.0, y_angstrom / 10.0, z_angstrom / 10.0))
    return tuple(positions)


def pdb_atom_line(
    serial: int,
    atom_name: str,
    residue_name: str,
    residue_id: int,
    coordinates_nm: Vector3,
    *,
    element: str,
    chain_id: str = "A",
) -> str:
    """Format one PDB HETATM line from coordinates in nanometers."""

    if serial <= 0:
        msg = "serial must be positive"
        raise ValueError(msg)
    if residue_id <= 0:
        msg = "residue_id must be positive"
        raise ValueError(msg)
    coordinates = _validate_vector3(coordinates_nm, "coordinates_nm")
    x_angstrom, y_angstrom, z_angstrom = (coordinate * 10.0 for coordinate in coordinates)
    return (
        f"HETATM{serial:5d} {atom_name:<4.4s} {residue_name:>3.3s} {chain_id:1.1s}"
        f"{residue_id:4d}    {x_angstrom:8.3f}{y_angstrom:8.3f}{z_angstrom:8.3f}"
        f"  1.00  0.00          {element:>2.2s}"
    )


def write_atom_records_pdb(path: str | Path, records: tuple[AtomRecord, ...]) -> Path:
    """Write AtomRecord entries as PACKMOL-readable HETATM PDB rows."""

    lines = [
        pdb_atom_line(
            serial=index,
            atom_name=record.atom_name,
            residue_name=record.residue_name,
            residue_id=record.residue_id,
            coordinates_nm=record.coordinates_nm,
            element=record.element,
            chain_id=record.chain_id,
        )
        for index, record in enumerate(records, 1)
    ]
    destination = Path(path)
    destination.write_text("\n".join([*lines, "END", ""]), encoding="utf-8")
    return destination


def write_template_pdb(path: str | Path, template: PackmolMoleculeTemplate) -> Path:
    """Write one molecule template as PACKMOL-readable HETATM PDB rows."""

    lines = [
        pdb_atom_line(
            serial=index,
            atom_name=atom_name,
            residue_name=template.residue_name,
            residue_id=1,
            coordinates_nm=position,
            element=symbol,
            chain_id="A",
        )
        for index, (position, symbol, atom_name) in enumerate(
            zip(
                template.positions_nm,
                template.atom_symbols,
                template.atom_names,
                strict=True,
            ),
            1,
        )
    ]
    destination = Path(path)
    destination.write_text("\n".join([*lines, "END", ""]), encoding="utf-8")
    return destination


def packmol_file_stem(name: str) -> str:
    """Return a filesystem-safe lowercase stem for PACKMOL scratch files."""

    stem = "".join(character if character.isalnum() else "_" for character in name.lower())
    return stem.strip("_") or "solvent"


def _validate_job(job: PackmolJob) -> None:
    if not job.structures:
        msg = "at least one PACKMOL structure is required"
        raise ValueError(msg)
    _validate_path(job.output_path, "output_path")
    _validate_bounds(job.box_bounds_nm)
    _validate_positive_finite_number(job.tolerance_angstrom, "tolerance_angstrom")
    if isinstance(job.nloop, bool) or not isinstance(job.nloop, int) or job.nloop <= 0:
        msg = "nloop must be a positive integer"
        raise ValueError(msg)
    if not isinstance(job.filetype, str) or not job.filetype.strip():
        msg = "filetype must be a non-empty string"
        raise ValueError(msg)
    if not isinstance(job.movebadrandom, bool):
        msg = "movebadrandom must be a boolean"
        raise ValueError(msg)
    for structure in job.structures:
        _validate_structure(structure)


def _validate_structure(structure: PackmolStructure) -> None:
    if not isinstance(structure, PackmolStructure):
        msg = "job structures must be PackmolStructure entries"
        raise TypeError(msg)
    if not str(structure.name).strip():
        msg = "structure name must be a non-empty string"
        raise ValueError(msg)
    _validate_path(structure.path, f"structure '{structure.name}' path")
    if (
        isinstance(structure.count, bool)
        or not isinstance(structure.count, int)
        or structure.count <= 0
    ):
        msg = f"structure '{structure.name}' count must be a positive integer"
        raise ValueError(msg)
    if structure.atom_count is not None and (
        isinstance(structure.atom_count, bool)
        or not isinstance(structure.atom_count, int)
        or structure.atom_count <= 0
    ):
        msg = f"structure '{structure.name}' atom_count must be a positive integer when provided"
        raise ValueError(msg)
    if not isinstance(structure.fixed, bool):
        msg = f"structure '{structure.name}' fixed must be a boolean"
        raise ValueError(msg)
    if structure.inside_box_bounds_nm is not None:
        _validate_bounds(structure.inside_box_bounds_nm)


def _validate_solvent_component(component: PackmolSolventComponent) -> None:
    """Reject invalid solvent component metadata before writing scratch files."""

    if not isinstance(component, PackmolSolventComponent):
        msg = "solvent components must be PackmolSolventComponent entries"
        raise TypeError(msg)
    if not str(component.name).strip():
        msg = "solvent component name must be non-empty"
        raise ValueError(msg)
    if (
        isinstance(component.count, bool)
        or not isinstance(component.count, int)
        or component.count < 0
    ):
        msg = f"solvent component {component.name!r} count must be a non-negative integer"
        raise ValueError(msg)


def _region_volume_nm3(region: BoxBounds) -> float:
    _validate_bounds(region)
    return (
        (region[0][1] - region[0][0])
        * (region[1][1] - region[1][0])
        * (region[2][1] - region[2][0])
    )


def _validate_bounds(bounds_nm: Any) -> None:
    if not isinstance(bounds_nm, Sized):
        msg = "box_bounds_nm must contain exactly three axis bounds"
        raise ValueError(msg)
    if len(bounds_nm) != 3:
        msg = "box_bounds_nm must contain exactly three axis bounds"
        raise ValueError(msg)
    for axis, bounds in zip(("x", "y", "z"), bounds_nm, strict=True):
        if not isinstance(bounds, Sized):
            msg = f"box_bounds_nm {axis}-axis bounds must contain exactly two values"
            raise ValueError(msg)
        if len(bounds) != 2:
            msg = f"box_bounds_nm {axis}-axis bounds must contain exactly two values"
            raise ValueError(msg)
        lower, upper = bounds
        if (
            isinstance(lower, bool)
            or isinstance(upper, bool)
        ):
            msg = f"box_bounds_nm {axis}-axis bounds must be finite with upper > lower"
            raise ValueError(msg)
        try:
            lower_finite = isfinite(lower)
            upper_finite = isfinite(upper)
            upper_after_lower = upper > lower
        except TypeError as error:
            msg = f"box_bounds_nm {axis}-axis bounds must be numeric"
            raise TypeError(msg) from error
        if not lower_finite or not upper_finite or not upper_after_lower:
            msg = f"box_bounds_nm {axis}-axis bounds must be finite with upper > lower"
            raise ValueError(msg)


def _validate_path(path: Any, name: str) -> None:
    if isinstance(path, bool):
        msg = f"{name} must be a non-empty path"
        raise TypeError(msg)
    try:
        path_text = os.fspath(path)
    except TypeError as error:
        msg = f"{name} must be a non-empty path"
        raise TypeError(msg) from error
    if not isinstance(path_text, str) or not path_text.strip():
        msg = f"{name} must be a non-empty path"
        raise ValueError(msg)


def _validate_positive_finite_number(value: Any, name: str) -> None:
    if isinstance(value, bool):
        msg = f"{name} must be a positive finite number"
        raise ValueError(msg)
    try:
        finite = isfinite(value)
        positive = value > 0.0
    except TypeError as error:
        msg = f"{name} must be a numeric positive finite number"
        raise TypeError(msg) from error
    if not finite or not positive:
        msg = f"{name} must be a positive finite number"
        raise ValueError(msg)


def _validate_vector3(values: Any, name: str) -> Vector3:
    if not isinstance(values, Sized):
        msg = f"{name} must contain exactly three values"
        raise ValueError(msg)
    if len(values) != 3:
        msg = f"{name} must contain exactly three values"
        raise ValueError(msg)
    if any(isinstance(value, bool) for value in values):
        msg = f"{name} values must be finite"
        raise ValueError(msg)
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        msg = f"{name} values must be numeric"
        raise TypeError(msg) from error
    if any(not isfinite(value) for value in vector):
        msg = f"{name} values must be finite"
        raise ValueError(msg)
    return vector


def _format_float(value: float) -> str:
    return f"{value:.6g}"


def _safe_write_text(path: str | Path, text: str, *, overwrite: bool = False) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        msg = f"refusing to overwrite existing file: {destination}"
        raise FileExistsError(msg)

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=destination.parent, delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(text)
        if overwrite:
            os.replace(temporary_name, destination)
        else:
            os.link(temporary_name, destination)
            os.unlink(temporary_name)
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()
    return destination
