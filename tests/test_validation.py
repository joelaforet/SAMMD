"""Tests for dependency-free validation gates."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from sammd.core.builders import build_system
from sammd.core.io import OutputPaths
from sammd.core.validation import (
    validate_build_plan,
    validate_output_paths,
    validate_topology_cif_text,
)


def test_default_build_plan_passes_validation(tmp_path: Path) -> None:
    """The default plan satisfies all current gates."""

    plan = build_system({}, output_dir=tmp_path)

    report = validate_build_plan(plan)
    output_report = validate_output_paths(plan.output_paths)

    assert report.passed
    assert output_report.passed
    assert report.to_summary()["passed"] is True


def test_current_validation_gates_do_not_require_export_artifacts(tmp_path: Path) -> None:
    """Interchange reload/export/minimization gates stay deferred for this release."""

    plan = build_system({}, output_dir=tmp_path)
    report = validate_build_plan(plan)
    output_report = validate_output_paths(plan.output_paths)
    current_gate_ids = {gate.gate_id for gate in report.gates + output_report.gates}
    deferred_export_gate_ids = {
        "interchange_json_reload",
        "interchange_openmm_export",
        "topology_positions_system_atom_counts",
        "system_xml_deserializes_particle_count",
        "minimization_energy_finite_not_increased",
    }

    assert current_gate_ids.isdisjoint(deferred_export_gate_ids)
    assert not plan.output_paths.solvated_system.exists()
    assert not plan.output_paths.openff_interchange.exists()
    assert not plan.output_paths.anchor_metadata.exists()


def test_bad_box_volume_fails_validation(tmp_path: Path) -> None:
    """Box dimensions, bounds, and volume must agree."""

    plan = build_system({}, output_dir=tmp_path)
    bad_box = replace(plan.box_plan, volume_nm3=plan.box_plan.volume_nm3 + 1.0)
    bad_plan = replace(plan, box_plan=bad_box)

    report = validate_build_plan(bad_plan)

    assert not report.passed
    assert _gate(report, "box_dimensions_bounds_volume").passed is False


def test_bad_sam_nearest_metal_pairs_fail_validation(tmp_path: Path) -> None:
    """Each SAM anchor needs the configured number of slab-local metal indices."""

    plan = build_system({}, output_dir=tmp_path)
    first = plan.sam_placements.placements[0]
    bad_pose = replace(
        first.anchor_pose,
        nearest_metal_atom_indices=(0, len(plan.slab.positions_nm)),
    )
    bad_placement = replace(first, anchor_pose=bad_pose)
    bad_placements = (bad_placement, *plan.sam_placements.placements[1:])
    bad_plan = replace(
        plan,
        sam_placements=replace(plan.sam_placements, placements=bad_placements),
    )

    report = validate_build_plan(bad_plan)
    gate = _gate(report, "metal_s_pair_count_and_indices")

    assert not report.passed
    assert gate.passed is False
    assert gate.details["failures"]


def test_bad_output_suffix_fails_validation(tmp_path: Path) -> None:
    """Current and reserved path fields use stable artifact suffixes."""

    paths = OutputPaths(
        sam_grafting_density=tmp_path / "sam_grafting_density.txt",
        pymol_system=tmp_path / "solvated_system_pymol.cif",
        anchor_metadata=tmp_path / "anchor_metadata.yaml",
    )

    report = validate_output_paths(paths)

    gate = _gate(report, "output_path_suffixes")


    assert not report.passed
    assert gate.details["failures"] == {
        "sam_grafting_density": "expected .cif, found .txt",
        "pymol_system": "expected .pdb, found .cif",
        "anchor_metadata": "expected .json, found .yaml",
    }


def test_topology_cif_atom_count_mismatch_fails_validation(tmp_path: Path) -> None:
    """Inspection topology CIF atom count is checked without mmCIF dependencies."""

    plan = build_system({}, output_dir=tmp_path)
    path = plan.write_topology_cif()
    text = path.read_text(encoding="utf-8")

    report = validate_topology_cif_text(text, expected_atom_count=1)
    gate = _gate(report, "topology_cif_atom_count")

    assert not report.passed
    expected_actual = len(plan.slab.positions_nm) + len(plan.sam_placements.placements)
    assert gate.details["actual"] == expected_actual


def test_topology_cif_cell_mismatch_fails_validation(tmp_path: Path) -> None:
    """Cell lengths are parsed from the current topology CIF header."""

    plan = build_system({}, output_dir=tmp_path)
    path = plan.write_topology_cif()
    text = path.read_text(encoding="utf-8")
    bad_box = (
        plan.box_plan.dimensions_nm[0] + 1.0,
        plan.box_plan.dimensions_nm[1],
        plan.box_plan.dimensions_nm[2],
    )

    report = validate_topology_cif_text(text, expected_box_nm=bad_box)

    assert not report.passed
    assert _gate(report, "topology_cif_cell_lengths").passed is False


def test_module_import_avoids_heavy_optional_modules() -> None:
    """Validation import should not pull heavy OpenMM/OpenFF-style modules."""

    heavy_modules = ("openmm", "openff", "rdkit", "mbuild", "MDAnalysis", "parmed", "pdbfixer")
    code = (
        "import json, sys; "
        "import sammd.core.validation; "
        f"heavy_modules = {heavy_modules!r}; "
        "print(json.dumps([name for name in heavy_modules if name in sys.modules]))"
    )
    src_path = Path(__file__).resolve().parents[1] / "src"
    pythonpath = os.pathsep.join(filter(None, (str(src_path), os.environ.get("PYTHONPATH", ""))))
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": pythonpath},
        text=True,
    )

    assert json.loads(result.stdout) == []


def _gate(report, gate_id):
    return next(gate for gate in report.gates if gate.gate_id == gate_id)
