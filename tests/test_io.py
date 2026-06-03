"""Tests for output planning and mmCIF scaffolds."""

import shlex

import pytest

from sammd.config import OutputConfig, SAMMDConfig
from sammd.io import (
    AtomRecord,
    format_mmcif,
    plan_output_paths,
    safe_write_text,
    slab_to_atom_records,
)
from sammd.surfaces import plan_pd111_slab


def test_output_paths_default_under_base_directory(tmp_path) -> None:
    """Resolve default visualization and reporter artifacts under a base directory."""

    paths = plan_output_paths(SAMMDConfig(), tmp_path)

    assert paths.topology == tmp_path / "topology.cif"
    assert paths.trajectory == tmp_path / "trajectory.dcd"
    assert paths.thermodynamics == tmp_path / "thermodynamics.csv"
    assert paths.checkpoint is None
    assert paths.state is None


def test_output_paths_support_user_overrides(tmp_path) -> None:
    """Resolve configured relative and optional runtime output paths."""

    config = OutputConfig(
        topology="viz/system.cif",
        trajectory="traj/run.dcd",
        thermodynamics="reports/state.csv",
        checkpoint="restart.chk",
        state="restart.xml",
    )

    paths = plan_output_paths(config, tmp_path)

    assert paths.topology == tmp_path / "viz/system.cif"
    assert paths.trajectory == tmp_path / "traj/run.dcd"
    assert paths.thermodynamics == tmp_path / "reports/state.csv"
    assert paths.checkpoint == tmp_path / "restart.chk"
    assert paths.state == tmp_path / "restart.xml"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"topology": "topology.pdb"}, "topology output"),
        ({"trajectory": "trajectory.xtc"}, "trajectory output"),
        ({"thermodynamics": "thermodynamics.txt"}, "thermodynamics output"),
    ],
)
def test_output_path_suffix_validation_rejects_wrong_extensions(tmp_path, kwargs, message) -> None:
    """Reject practical output suffix mistakes before runtime construction."""

    config = OutputConfig(**kwargs)

    with pytest.raises(ValueError, match=message):
        plan_output_paths(config, tmp_path)


def test_safe_write_text_refuses_overwrite_and_writes_content(tmp_path) -> None:
    """Write text atomically and require explicit overwrite for existing files."""

    path = tmp_path / "nested" / "artifact.txt"

    safe_write_text(path, "first\n")

    assert path.read_text(encoding="utf-8") == "first\n"

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        safe_write_text(path, "second\n")

    safe_write_text(path, "second\n", overwrite=True)
    assert path.read_text(encoding="utf-8") == "second\n"


def test_mmcif_writer_formats_required_atom_site_fields_and_angstrom_coordinates() -> None:
    """Produce parseable mmCIF text with Angstrom coordinates from nanometers."""

    text = format_mmcif(
        (
            AtomRecord(
                serial=1,
                atom_name="Pd1",
                element="Pd",
                residue_name="PDT",
                residue_id=1,
                chain_id="M",
                component_label="metal_top_layer",
                coordinates_nm=(0.1, -0.2, 0.3),
            ),
        ),
        cell_lengths_nm=(1.0, 2.0, 3.0),
    )

    assert "_cell.length_a 10.000000" in text
    assert "_atom_site.Cartn_x" in text
    assert "_atom_site.Cartn_y" in text
    assert "_atom_site.Cartn_z" in text
    atom_line = next(line for line in text.splitlines() if line.startswith("HETATM"))
    tokens = shlex.split(atom_line)
    assert tokens[:8] == ["HETATM", "1", "Pd", "Pd1", "PDT", "M", "metal_top_layer", "1"]
    assert tokens[8:11] == ["1.000000", "-2.000000", "3.000000"]


def test_slab_to_atom_records_gives_deterministic_pd_layer_labels() -> None:
    """Label Pd slab atoms with deterministic metal component roles."""

    slab = plan_pd111_slab((0.4, 0.4), 3)
    records = slab_to_atom_records(slab)

    assert len(records) == len(slab.positions_nm)
    assert {record.element for record in records} == {"Pd"}
    assert records[0].serial == 1
    assert records[0].atom_name == "Pd1"
    assert records[0].residue_name == "PDB"
    assert records[0].component_label == "metal_bottom_layer"
    assert records[-1].residue_name == "PDT"
    assert records[-1].component_label == "metal_top_layer"
    assert any(record.component_label == "metal_slab" for record in records)
