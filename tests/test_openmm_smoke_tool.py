"""Temporary tests for the developer-only OpenMM smoke runner."""

from __future__ import annotations

import importlib.util
import math
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

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
    """Load the temporary/openmm_smoke.py module from the repository root."""

    path = Path(__file__).resolve().parents[1] / "temporary" / "openmm_smoke.py"
    spec = importlib.util.spec_from_file_location("sammd_openmm_smoke_tool", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_openmm_minimization_lowers_finite_repulsive_energy() -> None:
    """OpenMM minimization should relax a lightweight finite steric clash."""

    openmm = pytest.importorskip("openmm")
    unit = openmm.unit
    system = openmm.System()
    system.addParticle(39.9 * unit.amu)
    system.addParticle(39.9 * unit.amu)
    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    for _ in range(2):
        nonbonded.addParticle(0.0, 0.34 * unit.nanometer, 0.8 * unit.kilojoule_per_mole)
    system.addForce(nonbonded)
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    context = openmm.Context(system, integrator)
    context.setPositions(
        [
            openmm.Vec3(0.0, 0.0, 0.0),
            openmm.Vec3(0.15, 0.0, 0.0),
        ]
        * unit.nanometer
    )

    initial = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
    openmm.LocalEnergyMinimizer.minimize(context)
    minimized = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )

    assert math.isfinite(initial)
    assert math.isfinite(minimized)
    assert minimized < initial


def actual_fixed_solute_positions(smoke, plan) -> tuple[tuple[float, float, float], ...]:
    """Construct fixed-solute positions after slab, SAM, and reactant placement."""

    positions = list(actual_solvent_boundary_positions(smoke, plan))
    box = smoke.derive_box_dimensions(plan, 3.0)
    shift_nm = tuple(dimension / 2.0 for dimension in box)
    reactant_template = smoke.MoleculeTemplate(
        name="reactant",
        smiles="C=CC=O",
        atoms=(),
        bonds=(),
        bond_parameters=(),
        angle_parameters=(),
        torsion_parameters=(),
        constraints=(),
        exception_parameters=(),
        positions_nm=((0.0, 0.0, -0.2), (0.0, 0.0, 0.2)),
        charge_model="test",
        force_field="test",
    )
    for molecule_positions in smoke.place_reactants_above_surface(
        plan,
        reactant_template,
        1,
        shift_nm,
        box,
    ):
        positions.extend(molecule_positions)
    return tuple(positions)


def actual_solvent_boundary_positions(smoke, plan) -> tuple[tuple[float, float, float], ...]:
    """Construct slab and SAM positions that define planar solvent boundaries."""

    box = smoke.derive_box_dimensions(plan, 3.0)
    shift_nm = tuple(dimension / 2.0 for dimension in box)
    positions = [smoke.add_vectors(position, shift_nm) for position in plan.slab.positions_nm]
    positions.extend(
        smoke.add_vectors(
            smoke.add_vectors(placement.position_nm, smoke.scale_vector(placement.normal, 0.18)),
            shift_nm,
        )
        for placement in plan.sam_placements.placements
    )
    return tuple(positions)


def test_smoke_tool_import_does_not_import_heavy_science_modules() -> None:
    """Keep the smoke tool importable before a CUDA environment is active."""

    sys.modules.pop("openmm", None)
    sys.modules.pop("rdkit", None)

    load_smoke_tool()

    assert "openmm" not in sys.modules
    assert "rdkit" not in sys.modules


def test_auto_solvent_count_uses_actual_solvent_region_volume() -> None:
    """Auto solvent count should use actual solvent packing region volume."""

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
    _, _, regions = smoke.actual_solvent_packing_geometry(
        plan,
        actual_fixed_solute_positions(smoke, plan),
        solvent_padding_nm=3.0,
        solvent_boundary_positions_nm=actual_solvent_boundary_positions(smoke, plan),
    )
    actual_volume = smoke.solvent_packing_volume_nm3(regions)

    count = smoke.resolve_solvent_count("auto", plan, "ethanol", actual_volume)
    expected = smoke.plan_solution_composition(config, actual_volume).solvent_components[0].count

    assert count == expected
    assert count != plan.solution.solvent_components[0].count


def test_rotation_matrix_maps_source_vector_to_target_vector() -> None:
    """SAM orientation helper should rotate the anchor bond onto the surface normal."""

    smoke = load_smoke_tool()
    matrix = smoke.rotation_matrix((1.0, 0.0, 0.0), (0.0, 0.0, -1.0))
    rotated = smoke.matvec(matrix, (1.0, 0.0, 0.0))

    assert rotated[0] == pytest.approx(0.0, abs=1.0e-12)
    assert rotated[1] == pytest.approx(0.0, abs=1.0e-12)
    assert rotated[2] == pytest.approx(-1.0, abs=1.0e-12)
    assert math.isclose(norm(rotated), 1.0)


def test_component_residue_assigner_uses_semantic_component_chains() -> None:
    """Follow PolyzyMD's semantic chain convention for each component role."""

    smoke = load_smoke_tool()
    assigner = smoke.ComponentResidueAssigner()

    metal_identity = assigner.allocate("palladium_slab", "PD", 1)[0]
    sam_identity = assigner.allocate("propanethiolate_sam", "PTL", 1)[0]
    reactant_identity = assigner.allocate("cinnamaldehyde", "CIN", 1)[0]
    solvent_identities = assigner.allocate("ethanol", "EOH", 10000)

    assert metal_identity == smoke.ResidueIdentity("M", 1, "PD")
    assert sam_identity == smoke.ResidueIdentity("C", 1, "PTL")
    assert reactant_identity == smoke.ResidueIdentity("B", 1, "CIN")
    assert solvent_identities[0] == smoke.ResidueIdentity("D", 1, "EOH")
    assert solvent_identities[9998] == smoke.ResidueIdentity("D", 9999, "EOH")
    assert solvent_identities[9999] == smoke.ResidueIdentity("E", 1, "EOH")
    assert assigner.component_ranges["ethanol"] == {
        "residue_name": "EOH",
        "residue_count": 10000,
        "chain_ids": ("D", "E"),
        "max_residues_per_chain": 9999,
    }


def test_component_residue_assigner_continues_same_role_namespaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple components in the same role should share chain/residue namespaces."""

    smoke = load_smoke_tool()
    monkeypatch.setattr(smoke, "MAX_RESIDUES_PER_CHAIN", 2)
    assigner = smoke.ComponentResidueAssigner()

    first_solvent = assigner.allocate("solvent:ethanol", "EOH", 3)
    second_solvent = assigner.allocate("solvent:water", "HOH", 2)
    first_sam = assigner.allocate("sam:propanethiol", "PTL", 2)
    second_sam = assigner.allocate("sam:mercaptoethanol", "MCE", 1)
    first_reactant = assigner.allocate("reactant:cinnamaldehyde", "CIN", 2)
    second_reactant = assigner.allocate("reactant:benzaldehyde", "BEN", 1)

    assert first_solvent == (
        smoke.ResidueIdentity("D", 1, "EOH"),
        smoke.ResidueIdentity("D", 2, "EOH"),
        smoke.ResidueIdentity("E", 1, "EOH"),
    )
    assert second_solvent == (
        smoke.ResidueIdentity("E", 2, "HOH"),
        smoke.ResidueIdentity("F", 1, "HOH"),
    )
    assert first_sam == (
        smoke.ResidueIdentity("C", 1, "PTL"),
        smoke.ResidueIdentity("C", 2, "PTL"),
    )
    assert second_sam == (smoke.ResidueIdentity("D", 1, "MCE"),)
    assert first_reactant == (
        smoke.ResidueIdentity("B", 1, "CIN"),
        smoke.ResidueIdentity("B", 2, "CIN"),
    )
    assert second_reactant == (smoke.ResidueIdentity("C", 1, "BEN"),)
    assert assigner.component_ranges["solvent:water"] == {
        "residue_name": "HOH",
        "residue_count": 2,
        "chain_ids": ("E", "F"),
        "max_residues_per_chain": 2,
    }
    assert assigner.component_ranges["sam:mercaptoethanol"]["chain_ids"] == ("D",)
    assert assigner.component_ranges["reactant:benzaldehyde"]["chain_ids"] == ("C",)


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


def test_packmol_input_packs_solvent_around_actual_fixed_solute() -> None:
    """Packmol input should pack solvent in actual fixed-solute regions."""

    smoke = load_smoke_tool()
    config = SAMMDConfig.model_validate({"surface": {"lateral_size": [2.0, 2.0]}})
    plan = build_system(config)
    fixed_positions = actual_fixed_solute_positions(smoke, plan)
    boundary_positions = actual_solvent_boundary_positions(smoke, plan)
    box, z_shift, regions = smoke.actual_solvent_packing_geometry(
        plan,
        fixed_positions,
        3.0,
        solvent_boundary_positions_nm=boundary_positions,
    )

    assert regions[0][0] == pytest.approx((0.0, box[0]))
    assert regions[0][1] == pytest.approx((0.0, box[1]))
    shifted_boundary_bottom = min(position[2] for position in boundary_positions) + z_shift
    shifted_boundary_top = max(position[2] for position in boundary_positions) + z_shift
    assert regions[0][2][0] == pytest.approx(shifted_boundary_bottom - 1.5)
    assert regions[1][0] == pytest.approx((0.0, box[0]))
    assert regions[1][1] == pytest.approx((0.0, box[1]))
    assert regions[1][2][1] == pytest.approx(shifted_boundary_top + 1.5)
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
    x_stop_angstrom = box[0] * 10.0
    y_stop_angstrom = box[1] * 10.0
    bottom_region_stop_angstrom = regions[0][2][1] * 10.0
    assert (
        f"  inside box 0 0 0 {x_stop_angstrom:g} {y_stop_angstrom:g} "
        f"{bottom_region_stop_angstrom:g}"
    ) in solvent_blocks[0]
    top_region_start_angstrom = regions[1][2][0] * 10.0
    top_region_stop_angstrom = regions[1][2][1] * 10.0
    assert (
        f"  inside box 0 0 {top_region_start_angstrom:g} "
        f"{x_stop_angstrom:g} {y_stop_angstrom:g} {top_region_stop_angstrom:g}"
    ) in solvent_blocks[1]
    assert "structure fixed_pd_sam.pdb" in text
    assert "fixed 0. 0. 0. 0. 0. 0." in text
    assert (
        f"  inside box 0 0 0 {x_stop_angstrom:g} {y_stop_angstrom:g} "
        f"{box[2] * 10.0:g}"
    ) not in text
    assert "nloop 200" in text


def test_packmol_input_uses_non_default_configured_tolerance() -> None:
    """Render Packmol input with the active runtime clearance tolerance."""

    smoke = load_smoke_tool()

    text = smoke.build_packmol_input(
        solute_path=Path("fixed_pd_sam.pdb"),
        solvent_path=Path("ethanol.pdb"),
        output_path=Path("packmol_output.pdb"),
        solvent_count=1,
        box_dimensions_nm=(2.0, 2.0, 2.0),
        solvent_regions_nm=(((0.0, 2.0), (0.0, 2.0), (0.0, 0.5)),),
        tolerance_angstrom=2.7,
    )

    assert text.startswith("tolerance 2.7\n")


def test_actual_solvent_regions_exclude_generated_fixed_solute_envelope() -> None:
    """Smoke solvent regions should stay outside actual slab/SAM extents."""

    smoke = load_smoke_tool()
    plan = build_system(SAMMDConfig.model_validate({"surface": {"lateral_size": [2.0, 2.0]}}))
    fixed_positions = actual_fixed_solute_positions(smoke, plan)
    boundary_positions = actual_solvent_boundary_positions(smoke, plan)
    _, z_shift, regions = smoke.actual_solvent_packing_geometry(
        plan,
        fixed_positions,
        3.0,
        solvent_boundary_positions_nm=boundary_positions,
    )
    shifted_min_z = min(position[2] for position in boundary_positions) + z_shift
    shifted_max_z = max(position[2] for position in boundary_positions) + z_shift
    clearance_nm = smoke.PACKMOL_TOLERANCE_ANGSTROM / 10.0

    assert regions[0][2][1] == pytest.approx(shifted_min_z - clearance_nm)
    assert regions[1][2][0] == pytest.approx(shifted_max_z + clearance_nm)


def test_smoke_reactant_does_not_define_global_solvent_reservoir() -> None:
    """A high reactant should remain fixed without lifting the whole top reservoir."""

    smoke = load_smoke_tool()
    plan = build_system(SAMMDConfig.model_validate({"surface": {"lateral_size": [2.0, 2.0]}}))
    boundary_positions = actual_solvent_boundary_positions(smoke, plan)
    high_reactant_positions = (
        *boundary_positions,
        (boundary_positions[0][0], boundary_positions[0][1], 6.0),
    )

    _, z_shift, regions = smoke.actual_solvent_packing_geometry(
        plan,
        high_reactant_positions,
        3.0,
        solvent_boundary_positions_nm=boundary_positions,
    )
    shifted_boundary_top = max(position[2] for position in boundary_positions) + z_shift
    clearance_nm = smoke.PACKMOL_TOLERANCE_ANGSTROM / 10.0

    assert regions[1][2][0] == pytest.approx(shifted_boundary_top + clearance_nm)
    assert regions[1][2][0] < high_reactant_positions[-1][2] + z_shift + clearance_nm


def test_smoke_high_reactant_extends_runtime_box_without_lifting_solvent_region() -> None:
    """Smoke geometry should contain a high reactant but keep solvent near SAM."""

    smoke = load_smoke_tool()
    plan = build_system(SAMMDConfig.model_validate({"surface": {"lateral_size": [2.0, 2.0]}}))
    boundary_positions = actual_solvent_boundary_positions(smoke, plan)
    high_reactant_positions = (
        *boundary_positions,
        (boundary_positions[0][0], boundary_positions[0][1], 6.0),
    )

    box, z_shift, regions = smoke.actual_solvent_packing_geometry(
        plan,
        high_reactant_positions,
        3.0,
        solvent_boundary_positions_nm=boundary_positions,
    )
    shifted_boundary_top = max(position[2] for position in boundary_positions) + z_shift
    shifted_reactant_top = high_reactant_positions[-1][2] + z_shift
    clearance_nm = smoke.PACKMOL_TOLERANCE_ANGSTROM / 10.0

    assert regions[1][2][0] == pytest.approx(shifted_boundary_top + clearance_nm)
    assert regions[1][2][1] == pytest.approx(shifted_boundary_top + 1.5)
    assert shifted_reactant_top < box[2]
    assert box[2] == pytest.approx(shifted_reactant_top + clearance_nm)


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
        solvent_name="methanol",
        solvent_residue_name="MOH",
        solvent_smiles="CO",
        reactant_count=1,
        system=fake_system,
        platform_dimensions_nm=(2.0, 2.0, 6.0),
        anchor_pairs=((1, 0),),
        anchor_scaling=fake_scaling,
        component_chain_ranges={},
        ensemble="NVT",
        pressure_bar=1.0,
        temperature_k=300.0,
        runtime_solvent_geometry=smoke.RuntimeSolventGeometry(
            solvent_boundary_z_bounds_nm=(1.0, 3.0),
            fixed_solute_z_bounds_nm=(0.8, 3.2),
            solvent_regions_nm=(((0.0, 2.0), (0.0, 2.0), (0.0, 0.8)),),
            solvent_count_planning_volume_nm3=3.2,
            solvent_padding_nm=2.0,
            solvent_padding_per_face_nm=1.0,
            solvent_clearance_nm=0.2,
            dimensions_nm=(2.0, 2.0, 4.0),
            z_shift_nm=0.5,
            molecule_counts={"ethanol": 1},
        ),
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
    assert summary["system"]["solvent"] == {
        "name": "methanol",
        "residue_name": "MOH",
        "smiles": "CO",
        "molecules": 1,
    }


def test_smoke_summary_records_runtime_solvent_geometry_metadata() -> None:
    """Expose actual solvent packing geometry in smoke summaries."""

    smoke = load_smoke_tool()
    geometry = smoke.RuntimeSolventGeometry(
        solvent_boundary_z_bounds_nm=(1.0, 3.0),
        fixed_solute_z_bounds_nm=(0.8, 3.2),
        solvent_regions_nm=(((0.0, 2.0), (0.0, 2.0), (0.0, 0.8)),),
        solvent_count_planning_volume_nm3=3.2,
        solvent_padding_nm=2.0,
        solvent_padding_per_face_nm=1.0,
        solvent_clearance_nm=0.2,
        dimensions_nm=(2.0, 2.0, 4.0),
        z_shift_nm=0.5,
        molecule_counts={"ethanol": 7},
    )
    fake_build = SimpleNamespace(
        pd_indices=(0,),
        sam_count=1,
        solvent_count=7,
        solvent_name="ethanol",
        solvent_residue_name="EOH",
        solvent_smiles="CCO",
        reactant_count=1,
        system=SimpleNamespace(getNumParticles=lambda: 2),
        platform_dimensions_nm=geometry.dimensions_nm,
        anchor_pairs=(),
        anchor_scaling=SimpleNamespace(pairs_added=0, sigma_nm=(), epsilon_delta_kj_mol=()),
        component_chain_ranges={},
        ensemble="NVT",
        pressure_bar=1.0,
        temperature_k=300.0,
        runtime_solvent_geometry=geometry,
    )
    fake_plan = SimpleNamespace(sam_placements=SimpleNamespace(selected_sites_per_side=1))
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
    schedule = smoke.RunSchedule(1.0, 1.0, 500, 10, 50, 2.0)
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

    assert summary["system"]["solvent_packing_regions_nm"] == [
        [[0.0, 2.0], [0.0, 2.0], [0.0, 0.8]]
    ]
    assert summary["system"]["solvent_count_planning_volume_nm3"] == pytest.approx(3.2)
    assert summary["system"]["solvent_padding_per_face_nm"] == pytest.approx(1.0)
    assert summary["system"]["solvent_clearance_nm"] == pytest.approx(0.2)
    assert summary["system"]["solvent_z_shift_nm"] == pytest.approx(0.5)
    assert summary["solution"]["molecule_counts"] == {"ethanol": 7}


def test_smoke_build_config_geometry_summary_uses_library_names() -> None:
    """Serialize runtime solvent geometry for smoke build configuration."""

    smoke = load_smoke_tool()
    geometry = smoke.RuntimeSolventGeometry(
        solvent_boundary_z_bounds_nm=(1.0, 3.0),
        fixed_solute_z_bounds_nm=(0.8, 3.2),
        solvent_regions_nm=(((0.0, 2.0), (0.0, 2.0), (0.0, 0.8)),),
        solvent_count_planning_volume_nm3=3.2,
        solvent_padding_nm=2.0,
        solvent_padding_per_face_nm=1.0,
        solvent_clearance_nm=0.2,
        dimensions_nm=(2.0, 2.0, 4.0),
        z_shift_nm=0.5,
        molecule_counts={"ethanol": 7},
    )

    summary = smoke.runtime_solvent_geometry_summary(geometry)

    assert summary["solvent_packing_regions_nm"] == [
        [[0.0, 2.0], [0.0, 2.0], [0.0, 0.8]]
    ]
    assert summary["solvent_count_planning_volume_nm3"] == pytest.approx(3.2)
    assert summary["solvent_padding_per_face_nm"] == pytest.approx(1.0)
    assert summary["molecule_counts"] == {"ethanol": 7}


def test_smoke_build_config_uses_configured_solvent_metadata(tmp_path: Path) -> None:
    """Serialize the actual configured solvent metadata in build config."""

    smoke = load_smoke_tool()
    geometry = smoke.RuntimeSolventGeometry(
        solvent_boundary_z_bounds_nm=(1.0, 3.0),
        fixed_solute_z_bounds_nm=(0.8, 3.2),
        solvent_regions_nm=(((0.0, 2.0), (0.0, 2.0), (0.0, 0.8)),),
        solvent_count_planning_volume_nm3=3.2,
        solvent_padding_nm=2.0,
        solvent_padding_per_face_nm=1.0,
        solvent_clearance_nm=0.2,
        dimensions_nm=(2.0, 2.0, 4.0),
        z_shift_nm=0.5,
        molecule_counts={"methanol": 4},
    )
    smoke_build = SimpleNamespace(
        solvent_count=4,
        solvent_name="methanol",
        solvent_residue_name="MOH",
        solvent_smiles="CO",
        runtime_solvent_geometry=geometry,
    )
    args = Namespace(
        platform="CPU",
        seed=7,
        pd_s_sigma_angstrom=2.2,
        pd_s_epsilon_kcal_mol=2.0,
        duration_ns=1.0,
        steps=None,
        timestep_fs=2.0,
        friction_per_ps=1.0,
        minimize_iterations=0,
        sulfur_height_nm=0.18,
    )
    path = tmp_path / "build_config.yaml"

    smoke.write_build_config(
        path,
        SAMMDConfig.model_validate(
            {
                "solvent": {
                    "components": [
                        {
                            "name": "methanol",
                            "residue_name": "MOH",
                            "smiles": "CO",
                            "mole_fraction": 1.0,
                            "density": 0.792,
                            "molar_mass": 32.04,
                        }
                    ]
                }
            }
        ),
        args,
        smoke_build,
        "CCS",
        1,
        smoke.RunSchedule(1.0, 1.0, 500, 10, 50, 2.0),
    )

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert payload["smoke_overrides"]["solvent_name"] == "methanol"
    assert payload["smoke_overrides"]["solvent_residue_name"] == "MOH"
    assert payload["smoke_overrides"]["solvent_smiles"] == "CO"


def test_smoke_tool_defaults_match_canonical_metal_sulfur_strategy() -> None:
    """Keep the smoke pair override aligned with dependency-free SAM metadata."""

    smoke = load_smoke_tool()

    assert pytest.approx(2.2) == smoke.DEFAULT_PD_S_SIGMA_ANGSTROM
    assert pytest.approx(2.0) == smoke.DEFAULT_PD_S_EPSILON_KCAL_MOL
