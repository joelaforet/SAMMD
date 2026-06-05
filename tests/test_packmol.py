"""Tests for Packmol helper functions."""

from pathlib import Path

import pytest

from sammd.packmol import build_packmol_input, pdb_atom_line, read_pdb_positions_nm


def test_packmol_input_packs_solvent_around_fixed_solute() -> None:
    """Packmol input should pack solvent around a fixed solute instead of a lattice."""

    text = build_packmol_input(
        solute_path=Path("fixed_pd_sam.pdb"),
        solvent_path=Path("ethanol.pdb"),
        output_path=Path("packmol_output.pdb"),
        solvent_count=25,
        box_dimensions_nm=(2.2, 2.4, 9.5),
    )

    assert "structure ethanol.pdb" in text
    assert "  number 25" in text
    assert "structure fixed_pd_sam.pdb" in text
    assert "fixed 0. 0. 0. 0. 0. 0." in text
    assert "inside box 0. 0." in text
    assert "nloop 200" in text


def test_pdb_atom_line_and_position_reader_round_trip_nm(tmp_path: Path) -> None:
    """Write simple PDB coordinates in angstroms and read them back as nanometers."""

    path = tmp_path / "molecule.pdb"
    line = pdb_atom_line(
        serial=1,
        atom_name="C1",
        residue_name="EOH",
        chain_id="A",
        residue_id=1,
        position_nm=(0.1234, 0.5, 1.0),
        element="C",
    )
    path.write_text(f"{line}\nEND\n", encoding="utf-8")

    positions = read_pdb_positions_nm(path)

    assert len(positions) == 1
    assert positions[0] == pytest.approx((0.1234, 0.5, 1.0))
