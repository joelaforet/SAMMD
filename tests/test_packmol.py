"""Tests for dependency-free PACKMOL planning helpers."""

from math import inf, nan
from types import SimpleNamespace

import pytest

from sammd.backends.packmol import (
    PackmolJob,
    PackmolMoleculeTemplate,
    PackmolStructure,
    build_packmol_input,
    pack_fixed_solute_with_solvent,
    pdb_atom_line,
    read_pdb_positions_nm,
    run_packmol,
    solvent_regions_around_solute,
    split_count_by_region_volume,
    write_packmol_input,
    zero_origin_box_bounds,
)
from sammd.core.builders import BoxPlan
from sammd.core.io import AtomRecord


def test_packmol_input_renders_fixed_solute_and_free_solvent() -> None:
    """Render fixed structures and free structures with Angstrom box bounds."""

    job = PackmolJob(
        output_path="packed.pdb",
        structures=(
            PackmolStructure("slab", "solute.pdb", 1, fixed=True),
            PackmolStructure("water", "water.pdb", 37),
        ),
        box_bounds_nm=((-1.0, 1.0), (-2.0, 2.0), (0.0, 3.0)),
    )

    text = build_packmol_input(job)

    assert text == (
        "tolerance 1.8\n"
        "filetype pdb\n"
        "output packed.pdb\n"
        "nloop 200\n"
        "movebadrandom\n"
        "\n"
        "structure solute.pdb\n"
        "  number 1\n"
        "  fixed 0. 0. 0. 0. 0. 0.\n"
        "end structure\n"
        "\n"
        "structure water.pdb\n"
        "  number 37\n"
        "  inside box -10 -20 0 10 20 30\n"
        "end structure\n"
    )


def test_multiple_free_structures_keep_deterministic_input_order() -> None:
    """Do not reorder free structures while rendering PACKMOL inputs."""

    job = PackmolJob(
        output_path="packed.pdb",
        structures=[
            PackmolStructure("ethanol", "ethanol.pdb", 2),
            PackmolStructure("water", "water.pdb", 3),
            PackmolStructure("ion", "chloride.pdb", 1),
        ],
        box_bounds_nm=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        movebadrandom=False,
    )

    lines = build_packmol_input(job).splitlines()

    assert [line for line in lines if line.startswith("structure ")] == [
        "structure ethanol.pdb",
        "structure water.pdb",
        "structure chloride.pdb",
    ]
    assert "movebadrandom" not in lines


def test_tolerance_and_nloop_are_rendered_from_job() -> None:
    """Expose PACKMOL tolerance and loop controls directly in input text."""

    job = PackmolJob(
        output_path="packed.pdb",
        structures=(PackmolStructure("water", "water.pdb", 1),),
        box_bounds_nm=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        tolerance_angstrom=2.25,
        nloop=500,
    )

    text = build_packmol_input(job)

    assert "tolerance 2.25\n" in text
    assert "nloop 500\n" in text


def test_zero_origin_bounds_accept_plain_dimensions_and_box_plan() -> None:
    """Create PACKMOL-friendly zero-origin bounds from common dimension providers."""

    assert zero_origin_box_bounds((1.0, 2.0, 3.0)) == ((0.0, 1.0), (0.0, 2.0), (0.0, 3.0))

    box_plan = BoxPlan(
        lateral_size_nm=(1.0, 2.0),
        dimensions_nm=(1.0, 2.0, 3.0),
        bounds_nm=((-0.5, 0.5), (-1.0, 1.0), (-1.5, 1.5)),
        volume_nm3=6.0,
        solvent_padding_nm=1.0,
        solvent_padding_per_face_nm=0.5,
        solvent_packing_regions_nm=(((0.0, 1.0), (0.0, 2.0), (0.0, 0.5)),),
        solvent_count_planning_volume_nm3=1.0,
        solvent_packing_warnings=(),
        sam_extended_length_nm=0.5,
        slab_center_nm=(0.0, 0.0, 0.0),
        sam_length_estimates=(),
    )

    assert zero_origin_box_bounds(box_plan) == ((0.0, 1.0), (0.0, 2.0), (0.0, 3.0))


def test_packmol_input_renders_two_explicit_solvent_regions() -> None:
    """Render separate solvent blocks when structures define inside-box bounds."""

    job = PackmolJob(
        output_path="packed.pdb",
        structures=(
            PackmolStructure("solute", "solute.pdb", 1, fixed=True),
            PackmolStructure(
                "water_bottom",
                "water.pdb",
                10,
                inside_box_bounds_nm=((0.0, 2.0), (0.0, 2.0), (0.0, 1.5)),
            ),
            PackmolStructure(
                "water_top",
                "water.pdb",
                10,
                inside_box_bounds_nm=((0.0, 2.0), (0.0, 2.0), (3.5, 5.0)),
            ),
        ),
        box_bounds_nm=((0.0, 2.0), (0.0, 2.0), (0.0, 5.0)),
    )

    text = build_packmol_input(job)

    assert text.count("structure water.pdb") == 2
    assert "inside box 0 0 0 20 20 15" in text
    assert "inside box 0 0 35 20 20 50" in text
    assert "inside box 0 0 0 20 20 50" not in text


def test_actual_solute_extents_define_bottom_and_top_solvent_regions() -> None:
    """Use generated atom coordinates instead of full-box solvent packing."""

    records = (
        AtomRecord(1, "C", "C", "SAM", 1, "A", "sam", (0.0, 0.0, 1.7)),
        AtomRecord(2, "C", "C", "SAM", 1, "A", "sam", (0.0, 0.0, 3.4)),
    )

    regions = solvent_regions_around_solute(
        records,
        ((0.0, 2.0), (0.0, 2.0), (0.0, 5.0)),
        clearance_nm=0.18,
    )

    assert regions[0][2] == pytest.approx((0.0, 1.52))
    assert regions[1][2] == pytest.approx((3.58, 5.0))
    assert regions[0][:2] == ((0.0, 2.0), (0.0, 2.0))
    assert regions[1][:2] == ((0.0, 2.0), (0.0, 2.0))
    assert split_count_by_region_volume(10, regions) == (5, 5)


@pytest.mark.parametrize(
    ("job", "message"),
    [
        (
            PackmolJob("out.pdb", (), ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
            "at least one PACKMOL structure",
        ),
        (
            PackmolJob(
                "out.pdb",
                (PackmolStructure("water", "water.pdb", 0),),
                ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            ),
            "count must be a positive integer",
        ),
        (
            PackmolJob(
                "out.pdb",
                (PackmolStructure("water", "water.pdb", True),),
                ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            ),
            "count must be a positive integer",
        ),
        (
            PackmolJob(
                "out.pdb",
                (PackmolStructure("water", "water.pdb", 1, atom_count=True),),
                ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            ),
            "atom_count must be a positive integer",
        ),
        (
            PackmolJob(
                "out.pdb",
                (PackmolStructure("water", "water.pdb", 1, fixed="False"),),
                ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            ),
            "fixed must be a boolean",
        ),
        (
            PackmolJob(
                "out.pdb",
                (PackmolStructure("water", "water.pdb", 1, fixed=0),),
                ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            ),
            "fixed must be a boolean",
        ),
        (
            PackmolJob(
                "out.pdb",
                (PackmolStructure("water", "", 1),),
                ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            ),
            "path must be a non-empty path",
        ),
        (
            PackmolJob(
                "out.pdb",
                (PackmolStructure("water", "   ", 1),),
                ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            ),
            "path must be a non-empty path",
        ),
        (
            PackmolJob(
                "   ",
                (PackmolStructure("water", "water.pdb", 1),),
                ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            ),
            "output_path must be a non-empty path",
        ),
        (
            PackmolJob(
                "out.pdb",
                (PackmolStructure("water", "water.pdb", 1),),
                ((0.0, 0.0), (0.0, 1.0), (0.0, 1.0)),
            ),
            "x-axis bounds must be finite with upper > lower",
        ),
        (
            PackmolJob(
                "out.pdb",
                (PackmolStructure("water", "water.pdb", 1),),
                ((0.0, inf), (0.0, 1.0), (0.0, 1.0)),
            ),
            "x-axis bounds must be finite with upper > lower",
        ),
        (
            PackmolJob(
                "out.pdb",
                (PackmolStructure("water", "water.pdb", 1),),
                ((False, 1.0), (0.0, 1.0), (0.0, 1.0)),
            ),
            "x-axis bounds must be finite with upper > lower",
        ),
        (
            PackmolJob(
                "out.pdb",
                (PackmolStructure("water", "water.pdb", 1),),
                ((0.0, True), (0.0, 1.0), (0.0, 1.0)),
            ),
            "x-axis bounds must be finite with upper > lower",
        ),
        (
            PackmolJob(
                "out.pdb",
                (PackmolStructure("water", "water.pdb", 1),),
                True,
            ),
            "box_bounds_nm must contain exactly three axis bounds",
        ),
        (
            PackmolJob(
                "out.pdb",
                (PackmolStructure("water", "water.pdb", 1),),
                (True, (0.0, 1.0), (0.0, 1.0)),
            ),
            "x-axis bounds must contain exactly two values",
        ),
        (
            PackmolJob(
                "out.pdb",
                (PackmolStructure("water", "water.pdb", 1),),
                (0.0, (0.0, 1.0), (0.0, 1.0)),
            ),
            "x-axis bounds must contain exactly two values",
        ),
        (
            PackmolJob(
                "out.pdb",
                (PackmolStructure("water", "water.pdb", 1),),
                ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                tolerance_angstrom=nan,
            ),
            "tolerance_angstrom must be a positive finite number",
        ),
        (
            PackmolJob(
                "out.pdb",
                (PackmolStructure("water", "water.pdb", 1),),
                ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                tolerance_angstrom=True,
            ),
            "tolerance_angstrom must be a positive finite number",
        ),
        (
            PackmolJob(
                "out.pdb",
                (PackmolStructure("water", "water.pdb", 1),),
                ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                nloop=0,
            ),
            "nloop must be a positive integer",
        ),
        (
            PackmolJob(
                "out.pdb",
                (PackmolStructure("water", "water.pdb", 1),),
                ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                nloop=True,
            ),
            "nloop must be a positive integer",
        ),
        (
            PackmolJob(
                "out.pdb",
                (PackmolStructure("water", "water.pdb", 1),),
                ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                filetype="",
            ),
            "filetype must be a non-empty string",
        ),
        (
            PackmolJob(
                "out.pdb",
                (PackmolStructure("water", "water.pdb", 1),),
                ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                filetype=True,
            ),
            "filetype must be a non-empty string",
        ),
        (
            PackmolJob(
                "out.pdb",
                (PackmolStructure("water", "water.pdb", 1),),
                ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                filetype=123,
            ),
            "filetype must be a non-empty string",
        ),
        (
            PackmolJob(
                "out.pdb",
                (PackmolStructure("water", "water.pdb", 1),),
                ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                movebadrandom="no",
            ),
            "movebadrandom must be a boolean",
        ),
        (
            PackmolJob(
                "out.pdb",
                (PackmolStructure("water", "water.pdb", 1),),
                ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                movebadrandom=1,
            ),
            "movebadrandom must be a boolean",
        ),
    ],
)
def test_invalid_packmol_jobs_fail_clearly(job, message) -> None:
    """Reject invalid counts, bounds, tolerance, and nloop before execution."""

    with pytest.raises(ValueError, match=message):
        build_packmol_input(job)


@pytest.mark.parametrize(
    ("job", "message"),
    [
        (
            PackmolJob(
                True,
                (PackmolStructure("water", "water.pdb", 1),),
                ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            ),
            "output_path must be a non-empty path",
        ),
        (
            PackmolJob(
                "out.pdb",
                (PackmolStructure("water", True, 1),),
                ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            ),
            "structure 'water' path must be a non-empty path",
        ),
        (
            PackmolJob(
                "out.pdb",
                (PackmolStructure("water", "water.pdb", 1),),
                (("low", 1.0), (0.0, 1.0), (0.0, 1.0)),
            ),
            "x-axis bounds must be numeric",
        ),
        (
            PackmolJob(
                "out.pdb",
                (PackmolStructure("water", "water.pdb", 1),),
                ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                tolerance_angstrom="tight",
            ),
            "tolerance_angstrom must be a numeric positive finite number",
        ),
        (
            PackmolJob(
                "out.pdb",
                (object(),),
                ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            ),
            "job structures must be PackmolStructure entries",
        ),
    ],
)
def test_invalid_packmol_job_types_fail_clearly(job, message) -> None:
    """Reject invalid path and numeric types before Path or isfinite failures."""

    with pytest.raises(TypeError, match=message):
        build_packmol_input(job)


@pytest.mark.parametrize("dimensions", [(0.0, 1.0, 1.0), (1.0, nan, 1.0), (True, 1.0, 1.0)])
def test_zero_origin_bounds_reject_invalid_dimensions(dimensions) -> None:
    """Reject invalid dimensions used to create packing bounds."""

    with pytest.raises(ValueError, match="dimensions_nm"):
        zero_origin_box_bounds(dimensions)


def test_zero_origin_bounds_reject_scalar_dimensions() -> None:
    """Reject scalar dimensions before raw len failures."""

    with pytest.raises(ValueError, match="dimensions_nm must contain exactly three values"):
        zero_origin_box_bounds(1.0)


def test_zero_origin_bounds_reject_non_numeric_dimensions() -> None:
    """Reject non-numeric dimensions with a helper-specific error."""

    with pytest.raises(TypeError, match="dimensions_nm values must be numeric"):
        zero_origin_box_bounds(("wide", 1.0, 1.0))


def test_write_packmol_input_refuses_overwrite(tmp_path) -> None:
    """Use safe file writing semantics for rendered PACKMOL inputs."""

    job = PackmolJob(
        output_path="packed.pdb",
        structures=(PackmolStructure("water", "water.pdb", 1),),
        box_bounds_nm=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    )
    path = tmp_path / "packmol.inp"

    write_packmol_input(job, path)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_packmol_input(job, path)


def test_missing_packmol_executable_raises_clear_runtime_error(tmp_path) -> None:
    """Fail before subprocess execution when PACKMOL is unavailable."""

    job = PackmolJob(
        output_path="packed.pdb",
        structures=(PackmolStructure("water", "water.pdb", 1),),
        box_bounds_nm=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    )

    with pytest.raises(RuntimeError, match="PACKMOL executable not found"):
        run_packmol(
            job,
            tmp_path / "packmol.inp",
            tmp_path,
            tmp_path / "packmol.out",
            executable="definitely-not-packmol",
        )


def test_pdb_coordinate_parser_returns_nanometers(tmp_path) -> None:
    """Convert fixed-width PDB Angstrom coordinates to nanometers."""

    pdb_path = tmp_path / "atoms.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                pdb_atom_line(1, "O", "HOH", 1, (0.1, 0.2, 0.3), element="O"),
                pdb_atom_line(2, "H1", "HOH", 1, (-0.1, 0.0, 1.2345), element="H"),
                "END",
            ]
        ),
        encoding="utf-8",
    )

    positions = read_pdb_positions_nm(pdb_path)

    assert positions[0] == pytest.approx((0.1, 0.2, 0.3))
    assert positions[1] == pytest.approx((-0.1, 0.0, 1.2345))


def test_fixed_solute_helper_uses_packmol_around_full_box(tmp_path, monkeypatch) -> None:
    """High-level packing keeps solute fixed and allows solvent throughout box bounds."""

    import sammd.backends.packmol as packmol_backend

    packed_positions = (
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5),
        (0.6, 0.5, 0.5),
        (0.8, 0.7, 0.6),
        (0.9, 0.7, 0.6),
    )

    def fake_run_packmol(job, input_path, working_dir, stdout_path):
        return SimpleNamespace(returncode=0, stdout="Success!")

    monkeypatch.setattr(packmol_backend, "run_packmol", fake_run_packmol)
    monkeypatch.setattr(packmol_backend, "read_pdb_positions_nm", lambda path: packed_positions)

    solvent_positions = pack_fixed_solute_with_solvent(
        solute_records=(
            AtomRecord(1, "Pd", "Pd", "Pdx", 1, "M", "metal_slab", (0.0, 0.0, 0.0)),
        ),
        solvent_template=PackmolMoleculeTemplate(
            residue_name="EOH",
            positions_nm=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)),
            atom_symbols=("C", "O"),
            atom_names=("C1", "O2"),
        ),
        solvent_name="ethanol",
        solvent_count=2,
        dimensions_nm=(2.0, 2.0, 2.0),
        working_dir=tmp_path,
    )

    assert solvent_positions == (packed_positions[1:3], packed_positions[3:5])
    input_text = (tmp_path / "packmol_input.inp").read_text(encoding="utf-8")
    assert "structure fixed_solute.pdb" in input_text
    assert "fixed 0. 0. 0. 0. 0. 0." in input_text
    assert "structure ethanol.pdb" in input_text
    assert "inside box 0 0 0 20 20 20" in input_text
