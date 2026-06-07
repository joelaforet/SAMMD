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

    bounds_angstrom = tuple(
        (lower * 10.0, upper * 10.0) for lower, upper in job.box_bounds_nm
    )
    box_tokens = (
        bounds_angstrom[0][0],
        bounds_angstrom[1][0],
        bounds_angstrom[2][0],
        bounds_angstrom[0][1],
        bounds_angstrom[1][1],
        bounds_angstrom[2][1],
    )
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
