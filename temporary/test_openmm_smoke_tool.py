"""Temporary tests for the developer-only OpenMM smoke runner."""

from __future__ import annotations

import importlib.util
import math
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from sammd.core.builders import build_system
from sammd.core.config import SAMMDConfig
from sammd.utils.geometry import norm


def packmol_structure_blocks(text: str) -> list[str]:
    """Return rendered Packmol structure blocks from input text."""

    blocks = []
    current = []
    for line in text.splitlines():
        if line.startswith("structure "):
            current = [line]
        elif current:
            current.append(line)
            if line == "end structure":
                blocks.append("\n".join(current))
                current = []
    return blocks


def load_smoke_tool():
    """Load the sibling temporary/openmm_smoke.py module."""

    path = Path(__file__).resolve().with_name("openmm_smoke.py")
    spec = importlib.util.spec_from_file_location("sammd_openmm_smoke_tool", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_smoke_tool_import_does_not_import_heavy_science_modules() -> None:
    """Keep the smoke tool importable before a CUDA environment is active."""

    sys.modules.pop("openmm", None)
    sys.modules.pop("rdkit", None)

    load_smoke_tool()

    assert "openmm" not in sys.modules
    assert "rdkit" not in sys.modules


def test_auto_solvent_count_uses_sammd_solution_plan() -> None:
    """Auto solvent count should use the SAMMD solution composition plan."""

    smoke = load_smoke_tool()
    config = SAMMDConfig.model_validate(
        {
            "surface": {"lateral_size": [2.0, 2.0]},
            "solvent": {
                "components": [
                    {
                        "name": "ethanol",
                        "residue_name": "EOH",
                        "smiles": "CCO",
                        "mole_fraction": 1.0,
                        "density": smoke.ETHANOL_DENSITY_G_ML,
                        "molar_mass": smoke.ETHANOL_MASS_G_MOL,
                    }
                ]
            },
        }
    )
    plan = build_system(config)

    count = smoke.resolve_solvent_count("auto", plan, "ethanol")
    expected = plan.solution.solvent_components[0].count

    assert count == expected


def test_rotation_matrix_maps_source_vector_to_target_vector() -> None:
    """SAM orientation helper should rotate the anchor bond onto the surface normal."""

    smoke = load_smoke_tool()
    matrix = smoke.rotation_matrix((1.0, 0.0, 0.0), (0.0, 0.0, -1.0))
    rotated = smoke.matvec(matrix, (1.0, 0.0, 0.0))

    assert rotated[0] == pytest.approx(0.0, abs=1.0e-12)
    assert rotated[1] == pytest.approx(0.0, abs=1.0e-12)
    assert rotated[2] == pytest.approx(-1.0, abs=1.0e-12)
    assert math.isclose(norm(rotated), 1.0)


def test_component_residue_assigner_wraps_after_9999_residues() -> None:
    """Follow PolyzyMD's one-repeat-unit-per-residue chain wrapping convention."""

    smoke = load_smoke_tool()
    assigner = smoke.ComponentResidueAssigner()

    identities = assigner.allocate("ethanol", "EOH", 10000)
    next_identity = assigner.allocate("reactant", "CIN", 1)[0]

    assert identities[0] == smoke.ResidueIdentity("A", 1, "EOH")
    assert identities[9998] == smoke.ResidueIdentity("A", 9999, "EOH")
    assert identities[9999] == smoke.ResidueIdentity("B", 1, "EOH")
    assert next_identity == smoke.ResidueIdentity("C", 1, "CIN")
    assert assigner.component_ranges["ethanol"] == {
        "residue_name": "EOH",
        "residue_count": 10000,
        "first_chain_id": "A",
        "last_chain_id": "B",
        "max_residues_per_chain": 9999,
    }


def test_default_run_schedule_records_300_frames_with_2fs_timestep() -> None:
    """Default smoke schedule should prioritize the requested trajectory frame count."""

    smoke = load_smoke_tool()

    schedule = smoke.resolve_run_schedule(
        duration_ns=5.0,
        timestep_fs=2.0,
        steps=None,
        frames=300,
        report_interval=None,
    )

    assert schedule.total_steps == 2500200
    assert schedule.report_interval == 8334
    assert schedule.frames == 300
    assert schedule.simulated_duration_ns == pytest.approx(5.0004)


def test_packmol_input_packs_solvent_around_fixed_solute() -> None:
    """Packmol input should pack solvent in the planned shifted regions."""

    smoke = load_smoke_tool()
    config = SAMMDConfig.model_validate({"surface": {"lateral_size": [2.0, 2.0]}})
    plan = build_system(config)
    box = smoke.derive_box_dimensions(plan, 3.0)
    regions = smoke.runtime_solvent_packing_regions(plan)

    assert box == plan.box_plan.dimensions_nm
    assert regions[0][0] == pytest.approx((0.0, box[0]))
    assert regions[0][1] == pytest.approx((0.0, box[1]))
    assert regions[0][2] == pytest.approx((0.0, 1.5))
    assert regions[1][0] == pytest.approx((0.0, box[0]))
    assert regions[1][1] == pytest.approx((0.0, box[1]))
    assert regions[1][2] == pytest.approx((box[2] - 1.5, box[2]))
    text = smoke.build_packmol_input(
        solute_path=Path("fixed_pd_sam.pdb"),
        solvent_path=Path("ethanol.pdb"),
        output_path=Path("packmol_output.pdb"),
        solvent_count=25,
        box_dimensions_nm=box,
        solvent_regions_nm=regions,
    )
    blocks = packmol_structure_blocks(text)
    solvent_blocks = [block for block in blocks if block.startswith("structure ethanol.pdb")]

    assert len(solvent_blocks) == 2
    assert "  number 13" in solvent_blocks[0]
    assert "  number 12" in solvent_blocks[1]
    assert "  inside box 0 0 0 22.0052 23.8213 15" in solvent_blocks[0]
    top_region_start_angstrom = regions[1][2][0] * 10.0
    top_region_stop_angstrom = regions[1][2][1] * 10.0
    assert (
        f"  inside box 0 0 {top_region_start_angstrom:.4f} "
        f"22.0052 23.8213 {top_region_stop_angstrom:.4f}"
    ) in solvent_blocks[1]
    assert "structure fixed_pd_sam.pdb" in text
    assert "fixed 0. 0. 0. 0. 0. 0." in text
    assert "  inside box 0 0 0 22.0052 23.8213 68.3212" not in text
    assert "nloop 200" in text


def test_runtime_solvent_regions_exclude_shifted_fixed_solute_envelope() -> None:
    """Shifted smoke solvent regions should stay outside planned fixed-solute extents."""

    smoke = load_smoke_tool()
    plan = build_system(SAMMDConfig.model_validate({"surface": {"lateral_size": [2.0, 2.0]}}))
    box = smoke.derive_box_dimensions(plan, 3.0)
    shift_z = box[2] / 2.0
    regions = smoke.runtime_solvent_packing_regions(plan)
    bottom_anchor_z = min(
        placement.anchor_pose.sulfur_position_nm[2]
        for placement in plan.sam_placements.placements
        if placement.side == "bottom"
    )
    top_anchor_z = max(
        placement.anchor_pose.sulfur_position_nm[2]
        for placement in plan.sam_placements.placements
        if placement.side == "top"
    )
    bottom_solute_limit_z = bottom_anchor_z - plan.box_plan.sam_extended_length_nm + shift_z
    top_solute_limit_z = top_anchor_z + plan.box_plan.sam_extended_length_nm + shift_z

    assert regions[0][2][1] == pytest.approx(bottom_solute_limit_z)
    assert regions[1][2][0] == pytest.approx(top_solute_limit_z)


def test_packmol_input_rejects_explicit_full_box_solvent_region() -> None:
    """Smoke Packmol input should not accept old full-box solvent packing."""

    smoke = load_smoke_tool()
    box = (2.0, 2.0, 2.0)

    with pytest.raises(ValueError, match="must not reproduce full-box solvent packing"):
        smoke.build_packmol_input(
            solute_path=Path("fixed_pd_sam.pdb"),
            solvent_path=Path("ethanol.pdb"),
            output_path=Path("packmol_output.pdb"),
            solvent_count=25,
            box_dimensions_nm=box,
            solvent_regions_nm=(smoke.zero_origin_box_bounds(box),),
        )


def test_committed_packmol_artifacts_do_not_use_single_full_box_solvent() -> None:
    """Committed smoke artifacts should not contradict planned solvent regions."""

    repo_root = Path(__file__).resolve().parents[1]
    artifact_paths = sorted(repo_root.glob("outputs/**/packmol_input.inp"))

    if not artifact_paths:
        pytest.skip("no generated Packmol artifacts are present in this checkout")
    for artifact_path in artifact_paths:
        text = artifact_path.read_text(encoding="utf-8")
        blocks = packmol_structure_blocks(text)
        solvent_blocks = [
            block
            for block in blocks
            if "fixed 0. 0. 0. 0. 0. 0." not in block and "inside box" in block
        ]
        assert len(solvent_blocks) != 1, artifact_path


def test_validate_args_rejects_invalid_solvent_count() -> None:
    """Catch invalid smoke-only CLI values before optional imports."""

    smoke = load_smoke_tool()
    args = Namespace(
        lateral_size_nm=2.0,
        solvent_padding_nm=3.0,
        timestep_fs=0.5,
        friction_per_ps=1.0,
        pd_s_sigma_angstrom=2.2,
        pd_s_epsilon_kcal_mol=1.0,
        duration_ns=5.0,
        sulfur_height_nm=0.0,
        seed=1,
        steps=1,
        frames=300,
        minimize_iterations=1,
        report_interval=1,
        reactant_count=None,
        solvent_count="zero",
        water_count=None,
    )

    with pytest.raises(SystemExit, match="--solvent-count"):
        smoke.validate_args(args)


def test_validate_args_rejects_zero_steps() -> None:
    """Smoke runs should not silently accept zero integration steps."""

    smoke = load_smoke_tool()
    args = Namespace(
        lateral_size_nm=2.0,
        solvent_padding_nm=3.0,
        timestep_fs=0.5,
        friction_per_ps=1.0,
        pd_s_sigma_angstrom=2.2,
        pd_s_epsilon_kcal_mol=1.0,
        duration_ns=5.0,
        sulfur_height_nm=0.0,
        seed=1,
        steps=0,
        frames=300,
        minimize_iterations=1,
        report_interval=1,
        reactant_count=None,
        solvent_count="auto",
        water_count=None,
    )

    with pytest.raises(SystemExit, match="--steps must be positive"):
        smoke.validate_args(args)


def test_seed_help_distinguishes_canonical_and_sensitivity_runs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Document that alternate smoke seeds create distinct validation systems."""

    smoke = load_smoke_tool()

    with pytest.raises(SystemExit) as error:
        smoke.parse_args(["--help"])

    help_text = capsys.readouterr().out
    assert error.value.code == 0
    assert "Finite-system construction and velocity seed" in help_text
    assert "canonical smoke validation" in help_text
    assert "seed-sensitivity checks" in help_text


def test_smoke_summary_records_seed_provenance() -> None:
    """Smoke summaries should identify canonical and alternate seed runs."""

    smoke = load_smoke_tool()
    fake_system = SimpleNamespace(getNumParticles=lambda: 2)
    fake_scaling = SimpleNamespace(
        pairs_added=1,
        sigma_nm=(0.22,),
        epsilon_delta_kj_mol=(8.368,),
    )
    fake_build = SimpleNamespace(
        pd_indices=(0,),
        sam_count=1,
        solvent_count=1,
        reactant_count=1,
        system=fake_system,
        platform_dimensions_nm=(2.0, 2.0, 6.0),
        anchor_pairs=((1, 0),),
        anchor_scaling=fake_scaling,
        component_chain_ranges={},
        ensemble="NVT",
        pressure_bar=1.0,
        temperature_k=300.0,
    )
    fake_plan = SimpleNamespace(
        sam_placements=SimpleNamespace(selected_sites_per_side=1),
    )
    fake_paths = SimpleNamespace(
        output_dir=Path("outputs"),
        topology=Path("topology.cif"),
        trajectory=Path("trajectory.dcd"),
        thermodynamics=Path("thermo.csv"),
        checkpoint=Path("state.chk"),
        state_xml=Path("state.xml"),
        system_xml=Path("system.xml"),
        anchor_metadata=Path("anchor.json"),
        summary=Path("summary.json"),
        packmol_dir=Path("packmol"),
    )
    schedule = smoke.RunSchedule(
        requested_duration_ns=1.0,
        simulated_duration_ns=1.0,
        total_steps=500,
        report_interval=10,
        frames=50,
        timestep_fs=2.0,
    )
    energy = smoke.EnergyRecord(0.0, 0.0, 300.0)

    summary = smoke.smoke_summary(
        plan=fake_plan,
        smoke_build=fake_build,
        paths=fake_paths,
        platform_name="CPU",
        platform_errors=(),
        initial=energy,
        minimized=energy,
        final=energy,
        pd_displacements=(0.0,),
        sulfur_displacements=(0.0,),
        schedule=schedule,
        minimize_iterations=0,
        seed=1234,
    )

    assert summary["run"]["build_seed"] == 1234
    assert summary["run"]["velocity_seed"] == 1234
    assert summary["run"]["canonical_validation_seed"] == smoke.DEFAULT_SEED


def test_smoke_tool_defaults_match_canonical_metal_sulfur_strategy() -> None:
    """Keep the smoke pair override aligned with dependency-free SAM metadata."""

    smoke = load_smoke_tool()

    assert pytest.approx(2.2) == smoke.DEFAULT_PD_S_SIGMA_ANGSTROM
    assert pytest.approx(2.0) == smoke.DEFAULT_PD_S_EPSILON_KCAL_MOL
