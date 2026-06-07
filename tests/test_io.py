"""Tests for output planning and mmCIF scaffolds."""

from math import inf, nan

import pytest

from sammd.config import OutputFilesConfig, OutputsConfig, SAMMDConfig
from sammd.io import (
    AtomRecord,
    format_mmcif,
    plan_output_paths,
    safe_write_text,
    slab_to_atom_records,
)
from sammd.surfaces import plan_pd111_slab


def _base_atom_record(**overrides) -> AtomRecord:
    """Create a valid atom record with focused test overrides."""

    values = {
        "serial": 1,
        "atom_name": "Pd1",
        "element": "Pd",
        "residue_name": "PDT",
        "residue_id": 1,
        "chain_id": "M",
        "component_label": "metal_top_layer",
        "coordinates_nm": (0.1, -0.2, 0.3),
    }
    values.update(overrides)
    return AtomRecord(**values)


def test_output_paths_default_under_base_directory(tmp_path) -> None:
    """Resolve default visualization and reporter artifacts under a base directory."""

    paths = plan_output_paths(SAMMDConfig(), tmp_path)

    assert paths.topology == tmp_path / "topology.cif"
    assert paths.positions == tmp_path / "positions.cif"
    assert paths.openff_interchange == tmp_path / "interchange.json"
    assert paths.openmm_system == tmp_path / "system.xml"
    assert paths.anchor_metadata == tmp_path / "anchor_metadata.json"
    assert paths.build_summary == tmp_path / "build_summary.json"
    assert paths.resolved_config == tmp_path / "resolved_config.yaml"


def test_output_paths_support_user_overrides(tmp_path) -> None:
    """Resolve configured relative and optional runtime output paths."""

    config = OutputsConfig(
        files=OutputFilesConfig(
            topology="viz/system.cif",
            positions="coords/positions.cif",
            openff_interchange="interchange/system.json",
            openmm_system="openmm/system.xml",
            anchor_metadata="metadata/anchor_metadata.json",
            build_summary="reports/build_summary.json",
            resolved_config="reports/resolved_config.yaml",
        )
    )

    paths = plan_output_paths(config, tmp_path)

    assert paths.topology == tmp_path / "viz/system.cif"
    assert paths.positions == tmp_path / "coords/positions.cif"
    assert paths.openff_interchange == tmp_path / "interchange/system.json"
    assert paths.openmm_system == tmp_path / "openmm/system.xml"
    assert paths.anchor_metadata == tmp_path / "metadata/anchor_metadata.json"
    assert paths.build_summary == tmp_path / "reports/build_summary.json"
    assert paths.resolved_config == tmp_path / "reports/resolved_config.yaml"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"topology": "topology.pdb"}, "topology output"),
        ({"positions": "positions.pdb"}, "positions output"),
        ({"openff_interchange": "interchange.xml"}, "OpenFF Interchange output"),
        ({"openmm_system": "system.json"}, "OpenMM system output"),
        ({"anchor_metadata": "anchor_metadata.yaml"}, "anchor metadata output"),
        ({"build_summary": "build_summary.txt"}, "build summary output"),
        ({"resolved_config": "resolved_config.json"}, "resolved config output"),
    ],
)
def test_output_path_suffix_validation_rejects_wrong_extensions(tmp_path, kwargs, message) -> None:
    """Reject practical output suffix mistakes before runtime construction."""

    config = OutputsConfig(files=OutputFilesConfig(**kwargs))

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
            _base_atom_record(),
        ),
        cell_lengths_nm=(1.0, 2.0, 3.0),
    )

    assert "_cell.length_a 10.000000" in text
    assert "_entity.id" in text
    assert "_sammd_entity.component_label" in text
    assert "_atom_site.Cartn_x" in text
    assert "_atom_site.Cartn_y" in text
    assert "_atom_site.Cartn_z" in text
    assert "_atom_site.occupancy" in text
    assert "_atom_site.B_iso_or_equiv" in text
    atom_line = next(line for line in text.splitlines() if line.startswith("HETATM"))
    tokens = _split_cif_line(atom_line)
    assert tokens[:8] == ["HETATM", "1", "Pd", "Pd1", "PDT", "M", "1", "1"]
    assert tokens[8:11] == ["1.000000", "-2.000000", "3.000000"]
    assert tokens[11:13] == ["1.00", "0.00"]


@pytest.mark.parametrize("data_name", ["", "bad name", "bad'name", 'bad"name', "bad\nname"])
def test_mmcif_writer_rejects_invalid_data_names(data_name) -> None:
    """Reject unsafe mmCIF data block names before writing text."""

    with pytest.raises(ValueError, match="data_name"):
        format_mmcif((_base_atom_record(),), data_name=data_name)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"serial": 0}, "serial must be positive"),
        ({"residue_id": 0}, "residue_id must be positive"),
        ({"atom_name": ""}, "atom_name must be a non-empty string"),
        ({"element": ""}, "element must be a non-empty string"),
        ({"residue_name": ""}, "residue_name must be a non-empty string"),
        ({"chain_id": ""}, "chain_id must be a non-empty string"),
        ({"component_label": ""}, "component_label must be a non-empty string"),
        ({"atom_name": "Pd\n1"}, "atom_name must not contain control characters"),
        ({"coordinates_nm": (0.0, 1.0)}, "coordinates_nm must contain exactly three"),
        ({"coordinates_nm": (0.0, 1.0, nan)}, "coordinates_nm values must be finite"),
        ({"coordinates_nm": (0.0, 1.0, inf)}, "coordinates_nm values must be finite"),
    ],
)
def test_mmcif_writer_rejects_invalid_atom_records(overrides, message) -> None:
    """Validate atom records before emitting potentially invalid mmCIF text."""

    with pytest.raises(ValueError, match=message):
        format_mmcif((_base_atom_record(**overrides),))


def test_mmcif_writer_rejects_duplicate_atom_serials() -> None:
    """Reject duplicate atom IDs before formatting atom-site rows."""

    records = (
        _base_atom_record(serial=1),
        _base_atom_record(serial=1, atom_name="Pd2", residue_id=2),
    )

    with pytest.raises(ValueError, match="duplicate serial 1"):
        format_mmcif(records)


def test_mmcif_writer_quotes_safe_strings_and_uses_numeric_entity_ids() -> None:
    """Quote string values while keeping atom-site entity IDs internally consistent."""

    records = (
        _base_atom_record(
            atom_name="Pd top",
            residue_name="Pd's",
            component_label="metal top layer",
        ),
        _base_atom_record(
            serial=2,
            atom_name="#Pd2",
            residue_id=2,
            component_label="metal slab",
            coordinates_nm=(0.2, 0.0, 0.0),
        ),
        _base_atom_record(
            serial=3,
            atom_name="Pd3",
            residue_id=3,
            component_label="metal top layer",
            coordinates_nm=(0.3, 0.0, 0.0),
        ),
    )

    text = format_mmcif(records)
    lines = text.splitlines()
    entity_rows = _loop_rows(lines, "_entity.id")
    sammd_entity_rows = _loop_rows(lines, "_sammd_entity.id")
    atom_rows = [_split_cif_line(line) for line in lines if line.startswith("HETATM")]

    assert entity_rows == [
        ["1", "non-polymer", "metal top layer"],
        ["2", "non-polymer", "metal slab"],
    ]
    assert sammd_entity_rows == [["1", "metal top layer"], ["2", "metal slab"]]
    assert [row[6] for row in atom_rows] == ["1", "2", "1"]
    assert atom_rows[0][3] == "Pd top"
    assert atom_rows[0][4] == "Pd's"
    assert atom_rows[1][3] == "#Pd2"
    assert "metal top layer" not in {row[6] for row in atom_rows}


def test_mmcif_writer_quotes_reserved_tokens_and_double_quotes() -> None:
    """Round-trip reserved CIF tokens and values containing double quotes."""

    records = (
        _base_atom_record(atom_name="?", residue_name='Pd "top"', component_label="."),
    )

    text = format_mmcif(records)
    entity_rows = _loop_rows(text.splitlines(), "_entity.id")
    atom_row = next(
        _split_cif_line(line) for line in text.splitlines() if line.startswith("HETATM")
    )

    assert entity_rows == [["1", "non-polymer", "."]]
    assert atom_row[3] == "?"
    assert atom_row[4] == 'Pd "top"'


def test_mmcif_writer_rejects_values_with_both_quote_types() -> None:
    """Fail clearly rather than emitting invalid mixed-quote CIF values."""

    record = _base_atom_record(atom_name='Pd "top\'')

    with pytest.raises(ValueError, match="both single and double quotes"):
        format_mmcif((record,))


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


def _loop_rows(lines: list[str], first_field: str) -> list[list[str]]:
    """Collect rows for the loop identified by its first field."""

    start = lines.index(first_field)
    index = start
    while index < len(lines) and lines[index].startswith("_"):
        index += 1
    rows: list[list[str]] = []
    while index < len(lines) and lines[index] != "#":
        rows.append(_split_cif_line(lines[index]))
        index += 1
    return rows


def _split_cif_line(line: str) -> list[str]:
    """Split the simple one-line CIF values emitted by the writer."""

    tokens: list[str] = []
    index = 0
    while index < len(line):
        while index < len(line) and line[index].isspace():
            index += 1
        if index >= len(line):
            break
        if line[index] in {"'", '"'}:
            quote = line[index]
            index += 1
            start = index
            while index < len(line) and line[index] != quote:
                index += 1
            if index >= len(line):
                msg = "unterminated CIF quote in test helper"
                raise ValueError(msg)
            tokens.append(line[start:index])
            index += 1
        else:
            start = index
            while index < len(line) and not line[index].isspace():
                index += 1
            tokens.append(line[start:index])
    return tokens
