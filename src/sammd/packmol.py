"""Packmol input, execution, and lightweight PDB helpers for SAMMD workflows."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sammd.geometry import Vector3

PACKMOL_TOLERANCE_ANGSTROM = 1.8
PACKMOL_NLOOP = 200


@dataclass(frozen=True)
class PackmolSolutionPositions:
    """Packmol-generated positions grouped by molecule type."""

    solvent_positions_nm: tuple[tuple[Vector3, ...], ...]


def pack_solution_with_packmol(
    *,
    topology: Any,
    solute_positions_nm: tuple[Vector3, ...],
    solvent_template: Any,
    solvent_name: str,
    solvent_residue_name: str,
    solvent_count: int,
    box_dimensions_nm: Vector3,
    working_dir: Path,
) -> PackmolSolutionPositions:
    """Use Packmol to place solvent molecules around prebuilt solute coordinates."""

    working_dir.mkdir(parents=True, exist_ok=True)
    solute_path = working_dir / "fixed_solute.pdb"
    solvent_path = working_dir / f"{solvent_name}.pdb"
    output_path = working_dir / "packmol_output.pdb"
    input_path = working_dir / "packmol_input.inp"
    stdout_path = working_dir / "packmol_stdout.log"

    write_topology_pdb(solute_path, topology, solute_positions_nm)
    write_molecule_template_pdb(
        solvent_path,
        solvent_template,
        residue_name=solvent_residue_name,
    )

    input_text = build_packmol_input(
        solute_path=solute_path,
        solvent_path=solvent_path,
        output_path=output_path,
        solvent_count=solvent_count,
        box_dimensions_nm=box_dimensions_nm,
    )
    input_path.write_text(input_text, encoding="utf-8")
    run_packmol(input_path, working_dir, stdout_path)

    packed_positions = read_pdb_positions_nm(output_path)
    n_solute_atoms = len(solute_positions_nm)
    n_solvent_atoms = solvent_count * len(solvent_template.atoms)
    expected_atoms = n_solute_atoms + n_solvent_atoms
    if len(packed_positions) != expected_atoms:
        msg = (
            f"Packmol output contains {len(packed_positions)} atoms, expected "
            f"{expected_atoms} ({n_solute_atoms} solute + {n_solvent_atoms} solvent)"
        )
        raise RuntimeError(msg)

    solvent_start = n_solute_atoms
    solvent_stop = solvent_start + n_solvent_atoms
    atoms_per_solvent = len(solvent_template.atoms)
    solvent_positions = tuple(
        packed_positions[index : index + atoms_per_solvent]
        for index in range(solvent_start, solvent_stop, atoms_per_solvent)
    )
    return PackmolSolutionPositions(solvent_positions_nm=solvent_positions)


def build_packmol_input(
    *,
    solute_path: Path,
    solvent_path: Path,
    output_path: Path,
    solvent_count: int,
    box_dimensions_nm: Vector3,
    tolerance_angstrom: float = PACKMOL_TOLERANCE_ANGSTROM,
    nloop: int = PACKMOL_NLOOP,
) -> str:
    """Build Packmol input text for fixed-solute solvent placement."""

    box_angstrom = tuple(length * 10.0 for length in box_dimensions_nm)
    lines = [
        f"tolerance {tolerance_angstrom:.3f}",
        "filetype pdb",
        f"output {output_path.name}",
        "movebadrandom",
        f"nloop {nloop}",
        "",
        f"structure {solute_path.name}",
        "  number 1",
        "  fixed 0. 0. 0. 0. 0. 0.",
        "end structure",
        "",
        f"structure {solvent_path.name}",
        f"  number {solvent_count}",
        f"  inside box 0. 0. 0. {box_angstrom[0]:.6f} "
        f"{box_angstrom[1]:.6f} {box_angstrom[2]:.6f}",
        "end structure",
        "",
    ]
    return "\n".join(lines)


def run_packmol(input_path: Path, working_dir: Path, stdout_path: Path) -> None:
    """Execute Packmol and keep stdout for debugging."""

    packmol = shutil.which("packmol")
    if packmol is None:
        msg = "Packmol executable not found; run through the SAMMD science environment."
        raise RuntimeError(msg)
    with input_path.open("r", encoding="utf-8") as handle:
        result = subprocess.run(
            [packmol],
            stdin=handle,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=working_dir,
            check=False,
        )
    stdout = result.stdout.decode("utf-8", errors="replace")
    stdout_path.write_text(stdout, encoding="utf-8")
    if result.returncode != 0 or "Success!" not in stdout:
        msg = f"Packmol failed; see {stdout_path}"
        raise RuntimeError(msg)


def write_topology_pdb(path: Path, topology: Any, positions_nm: tuple[Vector3, ...]) -> None:
    """Write a minimal PDB for Packmol fixed-solute coordinates."""

    records = []
    for serial, (atom, position) in enumerate(zip(topology.atoms(), positions_nm, strict=True), 1):
        residue = atom.residue
        records.append(
            pdb_atom_line(
                serial=serial,
                atom_name=atom.name,
                residue_name=residue.name,
                chain_id=residue.chain.id,
                residue_id=int(residue.id or 1),
                position_nm=position,
                element=atom.element.symbol,
            )
        )
    path.write_text("\n".join([*records, "END", ""]), encoding="utf-8")


def write_molecule_template_pdb(path: Path, template: Any, *, residue_name: str) -> None:
    """Write one molecule template PDB for Packmol."""

    records = [
        pdb_atom_line(
            serial=index,
            atom_name=atom.name,
            residue_name=residue_name,
            chain_id="A",
            residue_id=1,
            position_nm=position,
            element=atom.element,
        )
        for index, (atom, position) in enumerate(
            zip(template.atoms, template.positions_nm, strict=True),
            1,
        )
    ]
    path.write_text("\n".join([*records, "END", ""]), encoding="utf-8")


def pdb_atom_line(
    *,
    serial: int,
    atom_name: str,
    residue_name: str,
    chain_id: str,
    residue_id: int,
    position_nm: Vector3,
    element: str,
) -> str:
    """Format one simple HETATM line for Packmol input."""

    x_angstrom, y_angstrom, z_angstrom = (coordinate * 10.0 for coordinate in position_nm)
    return (
        f"HETATM{serial:5d} {atom_name[:4]:<4s} {residue_name[:3]:>3s} {chain_id[:1]:1s}"
        f"{residue_id:4d}    {x_angstrom:8.3f}{y_angstrom:8.3f}{z_angstrom:8.3f}"
        f"  1.00  0.00          {element[:2]:>2s}"
    )


def read_pdb_positions_nm(path: Path) -> tuple[Vector3, ...]:
    """Read HETATM/ATOM coordinates from a PDB file as nanometer tuples."""

    positions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        positions.append(
            (
                float(line[30:38]) * 0.1,
                float(line[38:46]) * 0.1,
                float(line[46:54]) * 0.1,
            )
        )
    return tuple(positions)
