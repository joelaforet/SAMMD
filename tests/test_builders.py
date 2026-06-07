"""Tests for lightweight build-plan composition."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sammd.builders import DEFAULT_SOLVENT_PADDING_NM, build_system
from sammd.config import CONFIG_TEMPLATE, SAMMDConfig
from sammd.solvation import round_half_up
from sammd.surfaces import plan_pd111_slab


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
    assert plan.output_paths.topology == tmp_path / "topology.cif"
    assert plan.output_paths.positions == tmp_path / "positions.cif"
    assert plan.output_paths.openff_interchange == tmp_path / "interchange.json"
    assert plan.output_paths.openmm_system == tmp_path / "system.xml"
    assert not plan.full_construction_available
    with pytest.raises(NotImplementedError, match="OpenFF/OpenMM construction is not implemented"):
        plan.require_full_construction()


def test_planning_box_uses_adjusted_commensurate_lateral_dimensions() -> None:
    """Base count volume on the effective slab cell rather than requested dimensions."""

    requested_lateral_size = (5.0, 5.0)
    config = SAMMDConfig(surface={"lateral_size": requested_lateral_size})
    slab = plan_pd111_slab(requested_lateral_size, 8)

    plan = build_system(config)

    assert plan.slab.requested_lateral_size_nm == requested_lateral_size
    assert plan.composition_planning_box.lateral_size_nm == slab.lateral_size_nm
    assert plan.composition_planning_box.lateral_size_nm != requested_lateral_size
    assert plan.sam_placements.lateral_area_nm2 == pytest.approx(
        slab.lateral_size_nm[0] * slab.lateral_size_nm[1]
    )


def test_default_solvation_counts_are_deterministic() -> None:
    """Convert the default derived planning box into stable molecule counts."""

    plan = build_system(SAMMDConfig())
    count_planning_volume_nm3 = plan.composition_planning_box.count_planning_volume_nm3
    expected_ethanol = round_half_up(
        0.789 * count_planning_volume_nm3 * 1.0e-24 * 1000.0 / 46.06844 * 6.02214076e23
    )

    assert plan.composition_planning_box.solvent_padding_nm == DEFAULT_SOLVENT_PADDING_NM
    assert plan.solution.solvent_components[0].count == expected_ethanol
    assert plan.solution.reactants[0].count == 1


def test_build_plan_writes_topology_cif_and_refuses_overwrite(tmp_path) -> None:
    """Write readable build artifacts with safe overwrite behavior."""

    plan = build_system(SAMMDConfig(), output_dir=tmp_path)

    written_path = plan.write_topology_cif()
    summary_path = plan.write_build_summary()
    resolved_config_path = plan.write_resolved_config()
    text = written_path.read_text(encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert written_path == tmp_path / "topology.cif"
    assert summary_path == tmp_path / "build_summary.json"
    assert resolved_config_path == tmp_path / "resolved_config.yaml"
    assert text.startswith("data_sammd_topology")
    assert "_cell.length_a" in text
    assert "_atom_site.Cartn_x" in text
    assert text.count("HETATM") == len(plan.slab.positions_nm) + len(
        plan.sam_placements.placements
    )
    assert " PTL " in text
    assert summary["experiment"]["name"] == "propanethiol_cinnamaldehyde_pd111"
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


def test_build_import_avoids_heavy_backend_modules() -> None:
    """Check import-time behavior in a fresh process for heavy backend modules."""

    heavy_modules = ("openmm", "openff", "rdkit", "mbuild", "MDAnalysis", "parmed", "pdbfixer")
    code = (
        "import json, sys; "
        "import sammd, sammd.builders; "
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
