"""Tests for import-light build-plan composition."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sammd.core.builders import (
    DEFAULT_SAM_EXTENDED_LENGTH_NM,
    DEFAULT_SOLVENT_PADDING_NM,
    build_system,
)
from sammd.core.config import CONFIG_TEMPLATE, SAMMDConfig
from sammd.model.metal_sulfur import (
    METAL_SULFUR_EPSILON_KCAL_MOL,
    METAL_SULFUR_EPSILON_KJ_MOL,
    METAL_SULFUR_INTERACTION_MODE,
    METAL_SULFUR_PAIRS_PER_ANCHOR,
    METAL_SULFUR_SIGMA_NM,
)
from sammd.model.solvation import round_half_up
from sammd.model.surfaces import plan_fcc111_slab, plan_pd111_slab


def test_build_system_accepts_config_dict_and_yaml_path(tmp_path) -> None:
    """Normalize all public config inputs into equivalent build plans."""

    config = SAMMDConfig()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG_TEMPLATE, encoding="utf-8")

    object_plan = build_system(config, output_dir=tmp_path, seed=31)
    dict_plan = build_system(config.model_dump(), output_dir=tmp_path, seed=31)
    path_plan = build_system(config_path, output_dir=tmp_path, seed=31)

    assert object_plan.slab == dict_plan.slab == path_plan.slab
    assert object_plan.sam_placements == dict_plan.sam_placements == path_plan.sam_placements
    assert object_plan.solution == dict_plan.solution == path_plan.solution


def test_default_build_plan_contains_schema_artifacts(tmp_path) -> None:
    """Compose slab, binding, SAM, solution, and build output path plans."""

    plan = build_system(SAMMDConfig(), output_dir=tmp_path)

    assert plan.slab.metal == "Pd"
    assert plan.slab.facet == "111"
    assert plan.slab.layers == 8
    assert {site.site_kind for site in plan.binding_sites} == {"fcc_hollow"}
    assert {placement.side for placement in plan.sam_placements.placements} == {"bottom", "top"}
    assert plan.sam_placements.placements[0].component_residue_name == "PTL"
    assert plan.solution.solvent_components[0].name == "ethanol"
    assert plan.solution.solvent_components[0].residue_name == "EOH"
    assert plan.solution.reactants[0].name == "cinnamaldehyde"
    assert plan.solution.reactants[0].residue_name == "CIN"
    assert plan.output_paths.sam_grafting_density == tmp_path / "sam_grafting_density.cif"
    assert plan.output_paths.solvated_system == tmp_path / "solvated_system.cif"
    assert plan.output_paths.pymol_system == tmp_path / "solvated_system_pymol.pdb"
    assert plan.output_paths.openff_interchange == tmp_path / "interchange.json"
    assert plan.output_paths.anchor_metadata == tmp_path / "anchor_metadata.json"
    assert not plan.full_construction_available
    artifacts = plan.build_summary()["artifacts"]
    assert artifacts["sam_grafting_density"] == {
        "path": str(tmp_path / "sam_grafting_density.cif"),
        "status": "current",
        "available": True,
    }
    assert artifacts["anchor_metadata"] == {
        "path": str(tmp_path / "anchor_metadata.json"),
        "status": "reserved",
        "available": False,
    }
    assert artifacts["pymol_system"] == {
        "path": str(tmp_path / "solvated_system_pymol.pdb"),
        "status": "reserved",
        "available": False,
    }
    assert artifacts["openff_interchange"] == {
        "path": str(tmp_path / "interchange.json"),
        "status": "reserved",
        "available": False,
        "constructed": False,
        "format": "json",
        "save_method": "Interchange.model_dump_json",
        "load_method": "Interchange.model_validate_json",
        "openff_interchange_package_version": None,
        "compatibility_caveat": (
            "Pre-1.0 OpenFF Interchange JSON compatibility is not guaranteed across versions."
        ),
    }
    engine_exports = plan.build_summary()["engine_exports"]
    assert set(engine_exports) == {"openmm", "gromacs", "lammps", "amber"}
    assert engine_exports["openmm"] == {
        "status": "reserved",
        "available": False,
        "handoff": "Use Interchange.to_openmm() downstream when needed",
        "teaching_scope": "student OpenMM Python API path",
    }
    for engine in ["gromacs", "lammps", "amber"]:
        assert engine_exports[engine] == {
            "status": "reserved",
            "available": False,
            "handoff": "future downstream export from Interchange",
            "teaching_scope": "not taught in beginner workflow",
        }
    interaction = plan.build_summary()["sam"]["metal_sulfur_interaction"]
    assert interaction["mode"] == METAL_SULFUR_INTERACTION_MODE
    assert interaction["site_kind"] == "fcc_hollow"
    assert interaction["pairs_per_anchor"] == METAL_SULFUR_PAIRS_PER_ANCHOR
    assert interaction["sigma_nm"] == METAL_SULFUR_SIGMA_NM
    assert interaction["epsilon_kcal_mol"] == METAL_SULFUR_EPSILON_KCAL_MOL
    assert interaction["epsilon_kj_mol"] == METAL_SULFUR_EPSILON_KJ_MOL
    with pytest.raises(NotImplementedError, match="OpenFF/OpenMM construction is not implemented"):
        plan.require_full_construction()


def test_build_system_accepts_registered_non_pd_surface(tmp_path) -> None:
    """Build a deterministic plan for a registered non-Pd Fcc(111) surface."""

    config = SAMMDConfig(surface={"metal": "Pt", "lateral_size": (1.0, 1.0)})
    plan = build_system(config, output_dir=tmp_path)
    expected_slab = plan_fcc111_slab("Pt", (1.0, 1.0), plan.slab.layers)

    assert plan.slab.metal == "Pt"
    assert plan.slab.facet == "111"
    assert plan.slab.labels[:3] == ("Pt1", "Pt2", "Pt3")
    assert plan.slab.lateral_size_nm == expected_slab.lateral_size_nm
    assert len(plan.binding_sites) > 0


def test_box_plan_uses_adjusted_commensurate_lateral_dimensions() -> None:
    """Base the unified box on the effective slab cell rather than requested dimensions."""

    requested_lateral_size = (5.0, 5.0)
    config = SAMMDConfig(surface={"lateral_size": requested_lateral_size})
    slab = plan_pd111_slab(requested_lateral_size, 8)

    plan = build_system(config)

    assert plan.slab.requested_lateral_size_nm == requested_lateral_size
    assert plan.box_plan.lateral_size_nm == slab.lateral_size_nm
    assert plan.box_plan.lateral_size_nm != requested_lateral_size
    assert plan.sam_placements.lateral_area_nm2 == pytest.approx(
        slab.lateral_size_nm[0] * slab.lateral_size_nm[1]
    )


def test_default_box_plan_uses_sam_tip_padding_and_centered_slab() -> None:
    """Compute z bounds from centered slab faces, SAM length, and solvent padding."""

    plan = build_system(SAMMDConfig())
    expected_z_min = (
        plan.slab.bottom_z_nm
        - DEFAULT_SAM_EXTENDED_LENGTH_NM
        - DEFAULT_SOLVENT_PADDING_NM
    )
    expected_z_max = (
        plan.slab.top_z_nm + DEFAULT_SAM_EXTENDED_LENGTH_NM + DEFAULT_SOLVENT_PADDING_NM
    )
    expected_z = expected_z_max - expected_z_min

    assert plan.box_plan.slab_center_nm == (0.0, 0.0, 0.0)
    assert plan.box_plan.bounds_nm[2] == pytest.approx((expected_z_min, expected_z_max))
    assert plan.box_plan.dimensions_nm == pytest.approx(
        (plan.slab.lateral_size_nm[0], plan.slab.lateral_size_nm[1], expected_z)
    )
    assert plan.box_plan.sam_extended_length_nm == DEFAULT_SAM_EXTENDED_LENGTH_NM
    assert plan.box_plan.sam_length_estimates[0].source == "minimum_default"


def test_default_solvation_counts_are_deterministic() -> None:
    """Convert the default unified box volume into stable molecule counts."""

    plan = build_system(SAMMDConfig())
    count_planning_volume_nm3 = plan.box_plan.volume_nm3
    expected_ethanol = round_half_up(
        0.789 * count_planning_volume_nm3 * 1.0e-24 * 1000.0 / 46.06844 * 6.02214076e23
    )

    assert plan.box_plan.solvent_padding_nm == DEFAULT_SOLVENT_PADDING_NM
    assert plan.solution.box_volume_nm3 == pytest.approx(plan.box_plan.volume_nm3)
    assert plan.solution.solvent_components[0].count == expected_ethanol
    assert plan.solution.reactants[0].count == 1


def test_configured_sam_length_override_changes_box_and_solution_counts() -> None:
    """Use optional configured SAM length before the dependency-free heuristic."""

    default_plan = build_system(SAMMDConfig())
    longer_plan = build_system(
        SAMMDConfig(
            sam={
                "components": [
                    {
                        "name": "propanethiol",
                        "residue_name": "PTL",
                        "smiles": "CCCS",
                        "fraction": 1.0,
                        "extended_length_nm": 1.75,
                    }
                ]
            }
        )
    )

    assert longer_plan.box_plan.sam_extended_length_nm == 1.75
    assert longer_plan.box_plan.sam_length_estimates[0].source == "configured"
    assert longer_plan.box_plan.dimensions_nm[2] > default_plan.box_plan.dimensions_nm[2]
    assert longer_plan.solution.box_volume_nm3 == pytest.approx(longer_plan.box_plan.volume_nm3)
    assert (
        longer_plan.solution.solvent_components[0].count
        > default_plan.solution.solvent_components[0].count
    )


def test_build_plan_writes_topology_cif_and_refuses_overwrite(tmp_path) -> None:
    """Write readable build artifacts with safe overwrite behavior."""

    plan = build_system(SAMMDConfig(), output_dir=tmp_path)

    written_path = plan.write_topology_cif()
    summary_path = plan.write_build_summary()
    resolved_config_path = plan.write_resolved_config()
    text = written_path.read_text(encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert written_path == tmp_path / "sam_grafting_density.cif"
    assert summary_path == tmp_path / "build_summary.json"
    assert resolved_config_path == tmp_path / "resolved_config.yaml"
    assert text.startswith("data_sammd_topology")
    assert "_cell.length_a" in text
    assert f"_cell.length_c {plan.box_plan.dimensions_nm[2] * 10.0:.6f}" in text
    assert "_atom_site.Cartn_x" in text
    assert text.count("HETATM") == len(plan.slab.positions_nm) + len(
        plan.sam_placements.placements
    )
    assert " PTL " in text
    first_placement = plan.sam_placements.placements[0]
    first_sulfur_z_angstrom = first_placement.anchor_pose.sulfur_position_nm[2] * 10.0
    first_site_z_angstrom = first_placement.position_nm[2] * 10.0
    first_sam_row = next(
        line
        for line in text.splitlines()
        if f" {first_placement.component_residue_name} " in line and " S " in line
    )
    assert f"{first_sulfur_z_angstrom:.6f}" in first_sam_row
    assert f"{first_site_z_angstrom:.6f}" not in first_sam_row
    assert summary["experiment"]["name"] == "propanethiol_cinnamaldehyde_pd111"
    assert len(summary["sam"]["placements"]) == len(plan.sam_placements.placements)
    assert summary["sam"]["placements"][0] == {
        "component_name": first_placement.component_name,
        "residue_name": first_placement.component_residue_name,
        "side": first_placement.side,
        "site_kind": first_placement.site_kind,
        "site_position_nm": list(first_placement.anchor_pose.site_position_nm),
        "sulfur_position_nm": list(first_placement.anchor_pose.sulfur_position_nm),
        "normal": list(first_placement.anchor_pose.normal),
        "axis_direction": list(first_placement.anchor_pose.axis_direction),
        "azimuth_rad": first_placement.anchor_pose.azimuth_rad,
        "sulfur_height_nm": first_placement.anchor_pose.sulfur_height_nm,
        "nearest_metal_atom_indices": list(first_placement.anchor_pose.nearest_metal_atom_indices),
        "attachment_mode": first_placement.anchor_pose.attachment_mode,
        "metal_sulfur_interaction": (
            first_placement.anchor_pose.metal_sulfur_interaction.to_summary()
        ),
    }
    assert summary["box"]["volume_nm3"] == pytest.approx(plan.box_plan.volume_nm3)
    assert summary["box"]["slab_center_nm"] == [0.0, 0.0, 0.0]
    assert summary["solution"]["count_planning_volume_nm3"] == pytest.approx(
        plan.box_plan.volume_nm3
    )
    assert summary["solution"]["molecule_counts"]["cinnamaldehyde"] == 1
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        plan.write_topology_cif()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        plan.write_build_summary()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        plan.write_resolved_config()
    plan.write_topology_cif(overwrite=True)
    plan.write_build_summary(overwrite=True)
    plan.write_resolved_config(overwrite=True)


def test_build_system_hides_slab_thickness_from_config() -> None:
    """Choose a geometry-derived slab thickness instead of asking users for layers."""

    default_plan = build_system(SAMMDConfig(parameterization={"nonbonded_cutoff": 1.0}))
    thicker_plan = build_system(SAMMDConfig(parameterization={"nonbonded_cutoff": 2.0}))

    assert default_plan.slab.layers == 8
    assert default_plan.slab.slab_extent_nm[2] > 1.5
    assert thicker_plan.slab.layers > default_plan.slab.layers
    assert thicker_plan.slab.slab_extent_nm[2] > 2.5


def test_seed_override_deterministically_controls_sam_site_choices() -> None:
    """Allow callers to override the config seed without changing other inputs."""

    first = build_system(SAMMDConfig(), seed=101)
    second = build_system(SAMMDConfig(), seed=101)
    different = build_system(SAMMDConfig(), seed=202)

    assert first.sam_placements == second.sam_placements
    assert first.sam_placements.seed == 101
    assert different.sam_placements.seed == 202
    assert first.sam_placements.placements != different.sam_placements.placements


def test_build_import_avoids_heavy_optional_modules() -> None:
    """Check import-time behavior in a fresh process for heavy optional modules."""

    heavy_modules = ("openmm", "openff", "rdkit", "mbuild", "MDAnalysis", "parmed", "pdbfixer")
    code = (
        "import json, sys; "
        "import sammd, sammd.core.builders, sammd.model.metal_sulfur; "
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
