"""Tests for lightweight build-plan composition."""

import sys

import pytest

from sammd.builders import DEFAULT_SOLVENT_PADDING_NM, build_system
from sammd.config import CONFIG_TEMPLATE, SAMMDConfig
from sammd.solvation import round_half_up
from sammd.surfaces import plan_pd111_slab


def test_build_system_accepts_config_dict_and_yaml_path(tmp_path) -> None:
    """Normalize all public config inputs into equivalent build plans."""

    config = SAMMDConfig()
    config_path = tmp_path / "sammd.yaml"
    config_path.write_text(CONFIG_TEMPLATE, encoding="utf-8")

    object_plan = build_system(config, output_dir=tmp_path, seed=31)
    dict_plan = build_system(config.model_dump(), output_dir=tmp_path, seed=31)
    path_plan = build_system(config_path, output_dir=tmp_path, seed=31)

    assert object_plan.slab == dict_plan.slab == path_plan.slab
    assert object_plan.sam_placements == dict_plan.sam_placements == path_plan.sam_placements
    assert object_plan.solution == dict_plan.solution == path_plan.solution


def test_default_build_plan_contains_all_lightweight_artifacts(tmp_path) -> None:
    """Compose slab, binding, SAM, solution, and output path plans."""

    plan = build_system(SAMMDConfig(), output_dir=tmp_path)

    assert plan.slab.metal == "Pd"
    assert plan.slab.facet == "111"
    assert {site.site_kind for site in plan.binding_sites} == {"fcc_hollow"}
    assert {placement.side for placement in plan.sam_placements.placements} == {"bottom", "top"}
    assert plan.solution.solvent_components[0].name == "water"
    assert plan.solution.reactants[0].name == "cinnamaldehyde"
    assert plan.output_paths.topology == tmp_path / "topology.cif"
    assert not plan.full_construction_available
    with pytest.raises(NotImplementedError, match="OpenFF/OpenMM construction is not implemented"):
        plan.require_full_construction()


def test_planning_box_uses_adjusted_commensurate_lateral_dimensions() -> None:
    """Base count volume on the effective slab cell rather than requested dimensions."""

    requested_lateral_size_nm = (5.0, 5.0)
    config = SAMMDConfig()
    slab = plan_pd111_slab(requested_lateral_size_nm, config.surface.slab.layers)

    plan = build_system(config)

    assert plan.slab.requested_lateral_size_nm == requested_lateral_size_nm
    assert plan.planning_box.lateral_size_nm == slab.lateral_size_nm
    assert plan.planning_box.lateral_size_nm != requested_lateral_size_nm
    assert plan.sam_placements.lateral_area_nm2 == pytest.approx(
        slab.lateral_size_nm[0] * slab.lateral_size_nm[1]
    )


def test_default_solvation_counts_are_deterministic() -> None:
    """Convert the default derived planning box into stable molecule counts."""

    plan = build_system(SAMMDConfig())
    expected_water = round_half_up(
        0.997 * plan.planning_box.volume_nm3 * 1.0e-24 * 1000.0 / 18.01528 * 6.02214076e23
    )
    expected_reactant = round_half_up(
        0.05 * plan.planning_box.volume_nm3 * 1.0e-24 * 6.02214076e23
    )

    assert plan.planning_box.solvent_padding_nm == DEFAULT_SOLVENT_PADDING_NM
    assert plan.solution.solvent_components[0].count == expected_water
    assert plan.solution.reactants[0].count == expected_reactant


def test_build_plan_writes_planned_slab_mmcif_and_refuses_overwrite(tmp_path) -> None:
    """Write a readable Pd slab visualization artifact with safe overwrite behavior."""

    plan = build_system(SAMMDConfig(), output_dir=tmp_path)

    written_path = plan.write_planned_slab_mmcif()
    text = written_path.read_text(encoding="utf-8")

    assert written_path == tmp_path / "topology.cif"
    assert text.startswith("data_sammd_planned_slab")
    assert "_atom_site.Cartn_x" in text
    assert text.count("HETATM") == len(plan.slab.positions_nm)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        plan.write_planned_topology()
    plan.write_planned_topology(overwrite=True)


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
    """Keep import-time behavior independent of heavy molecular backends."""

    heavy_modules = ("openmm", "openff", "rdkit", "mbuild", "MDAnalysis", "parmed", "pdbfixer")

    assert all(module_name not in sys.modules for module_name in heavy_modules)
