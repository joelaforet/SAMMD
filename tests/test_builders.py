"""Tests for lightweight build-plan composition."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sammd.builders import DEFAULT_SOLVENT_PADDING_NM, build_system
from sammd.config import CONFIG_TEMPLATE, AnchorConfig, SAMComponentConfig, SAMConfig, SAMMDConfig
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
    assert plan.planned_slab_mmcif_path == tmp_path / "planned_slab.cif"
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
    assert plan.composition_planning_box.lateral_size_nm == slab.lateral_size_nm
    assert plan.composition_planning_box.lateral_size_nm != requested_lateral_size_nm
    assert plan.sam_placements.lateral_area_nm2 == pytest.approx(
        slab.lateral_size_nm[0] * slab.lateral_size_nm[1]
    )


def test_default_solvation_counts_are_deterministic() -> None:
    """Convert the default derived planning box into stable molecule counts."""

    plan = build_system(SAMMDConfig())
    count_planning_volume_nm3 = plan.composition_planning_box.count_planning_volume_nm3
    expected_water = round_half_up(
        0.997 * count_planning_volume_nm3 * 1.0e-24 * 1000.0 / 18.01528 * 6.02214076e23
    )
    expected_reactant = round_half_up(
        0.05 * count_planning_volume_nm3 * 1.0e-24 * 6.02214076e23
    )

    assert plan.composition_planning_box.solvent_padding_nm == DEFAULT_SOLVENT_PADDING_NM
    assert plan.solution.solvent_components[0].count == expected_water
    assert plan.solution.reactants[0].count == expected_reactant


def test_build_plan_writes_planned_slab_mmcif_and_refuses_overwrite(tmp_path) -> None:
    """Write a readable Pd slab visualization artifact with safe overwrite behavior."""

    plan = build_system(SAMMDConfig(), output_dir=tmp_path)

    written_path = plan.write_planned_slab_mmcif()
    text = written_path.read_text(encoding="utf-8")

    assert written_path == tmp_path / "planned_slab.cif"
    assert text.startswith("data_sammd_planned_slab_only")
    assert "_atom_site.Cartn_x" in text
    assert text.count("HETATM") == len(plan.slab.positions_nm)
    with pytest.raises(NotImplementedError, match="write_planned_slab_mmcif"):
        plan.write_planned_topology()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        plan.write_planned_slab_mmcif()
    plan.write_planned_slab_mmcif(overwrite=True)


@pytest.mark.parametrize(
    ("config_data", "message"),
    [
        ({"surface": {"slab": {"centered": False}}}, "centered=False is not implemented"),
        (
            {"surface": {"slab": {"double_sided": False}}},
            "double_sided=False is not implemented",
        ),
    ],
)
def test_build_system_rejects_unsupported_slab_options(config_data, message) -> None:
    """Fail clearly for schema-allowed slab options outside the MVP geometry."""

    with pytest.raises(NotImplementedError, match=message):
        build_system(config_data)


@pytest.mark.parametrize("site_kind", ["bridge", "atop"])
def test_build_system_rejects_unimplemented_anchor_sites(site_kind: str) -> None:
    """Fail before planning for schema-allowed anchor sites outside the MVP."""

    config = SAMMDConfig(sam=SAMConfig(anchor=AnchorConfig(site=site_kind)))

    with pytest.raises(NotImplementedError, match=f"{site_kind}.*not implemented"):
        build_system(config)


def test_build_system_rejects_unimplemented_component_anchor_sites() -> None:
    """Validate component-specific anchor requests before surface planning."""

    config = SAMMDConfig(
        sam=SAMConfig(
            components=[
                SAMComponentConfig(
                    name="atop-component",
                    smiles="CCCS",
                    fraction=1.0,
                    anchor=AnchorConfig(site="atop"),
                )
            ]
        )
    )

    with pytest.raises(NotImplementedError, match=r"atop.*not implemented"):
        build_system(config)


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
