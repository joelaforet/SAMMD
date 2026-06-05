"""Run a small real OpenMM smoke system for SAMMD.

This script is intentionally outside the public package API. It exercises the current
deterministic planners with optional science dependencies from the pixi ``science``
environment and builds a pragmatic OpenMM system directly for backend validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

from sammd._openmm_backend import _SmokeBuild as SmokeBuild
from sammd._openmm_backend import build_openmm_smoke_system
from sammd._openmm_templates import (
    _NAGL_CHARGE_MODEL,
    _OPENFF_FORCE_FIELD,
    molecule_template_from_smiles,
    require_openff_modules,
    require_openmm_modules,
)
from sammd.builders import build_system
from sammd.config import SAMMDConfig
from sammd.geometry import indexed_displacements, indexed_reference_displacements
from sammd.io import safe_write_text
from sammd.openmm_build import OpenMMSmokeBuilder, OpenMMSmokeBuildOptions
from sammd.openmm_runtime import (
    EnergyRecord,
    create_simulation_with_platform_fallback,
    positions_to_nm,
    read_energy,
)
from sammd.workflow import (
    CANONICAL_SMOKE_SOLVENT_NAME,
    CANONICAL_SMOKE_SOLVENT_RESIDUE_NAME,
    CANONICAL_SMOKE_SOLVENT_SMILES,
    DEFAULT_SMOKE_OUTPUT_DIR,
    RunSchedule,
    SmokePaths,
    load_smoke_config,
    prepare_outputs,
    resolve_run_schedule,
    smoke_paths,
)
from sammd.workflow import (
    ETHANOL_DENSITY_G_ML as WORKFLOW_ETHANOL_DENSITY_G_ML,
)
from sammd.workflow import (
    ETHANOL_MASS_G_MOL as WORKFLOW_ETHANOL_MASS_G_MOL,
)

ETHANOL_MASS_G_MOL = WORKFLOW_ETHANOL_MASS_G_MOL
ETHANOL_DENSITY_G_ML = WORKFLOW_ETHANOL_DENSITY_G_ML
SOLVENT_NAME = CANONICAL_SMOKE_SOLVENT_NAME
SOLVENT_RESIDUE_NAME = CANONICAL_SMOKE_SOLVENT_RESIDUE_NAME
SOLVENT_SMILES = CANONICAL_SMOKE_SOLVENT_SMILES
DEFAULT_OUTPUT_DIR = DEFAULT_SMOKE_OUTPUT_DIR
DEFAULT_LATERAL_SIZE_NM = 2.0
DEFAULT_SOLVENT_PADDING_NM = 3.0
DEFAULT_SOLVENT_COUNT = "auto"
DEFAULT_DURATION_NS = 5.0
DEFAULT_MINIMIZE_ITERATIONS = 0
DEFAULT_TRAJECTORY_FRAMES = 300
DEFAULT_TIMESTEP_FS = 2.0
DEFAULT_FRICTION_PER_PS = 1.0
DEFAULT_SEED = 2026
DEFAULT_SULFUR_HEIGHT_NM = 0.18
DEFAULT_PD_S_SIGMA_ANGSTROM = 2.2
REFERENCE_PD_S_EPSILON_KCAL_MOL = 1.0
DEFAULT_PD_S_EPSILON_KCAL_MOL = 2.0
OPENFF_FORCE_FIELD = _OPENFF_FORCE_FIELD
NAGL_CHARGE_MODEL = _NAGL_CHARGE_MODEL
PASS_MAX_TEMPERATURE_K = 600.0
PASS_MAX_PD_DISPLACEMENT_NM = 0.15
PASS_MAX_SULFUR_DISPLACEMENT_NM = 0.70


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the real-system smoke workflow."""

    args = parse_args(argv)
    modules = require_openmm_modules()
    openff_modules = require_openff_modules()
    paths = smoke_paths(Path(args.output_dir))
    prepare_outputs(paths, overwrite=args.overwrite)

    config = load_smoke_config(
        args.config,
        lateral_size_nm=args.lateral_size_nm,
        solvent_padding_nm=args.solvent_padding_nm,
        seed=args.seed,
        timestep_fs=args.timestep_fs,
        reporter_interval_steps=args.report_interval or 1,
    )
    plan = build_system(config, output_dir=paths.output_dir, seed=args.seed)
    schedule = resolve_run_schedule(
        duration_ns=args.duration_ns,
        timestep_fs=args.timestep_fs,
        steps=args.steps,
        frames=args.frames,
        report_interval=args.report_interval,
    )
    solvent_component = config.solvent.components[0]
    solvent_count = resolve_solvent_count(args.solvent_count, plan, solvent_component.name)
    reactant = config.reactants[0]
    reactant_count = resolve_reactant_count(args.reactant_count, plan, reactant.name)
    sam_template = molecule_template_from_smiles(
        modules,
        openff_modules,
        config.sam.components[0].smiles,
        config.sam.components[0].name,
    )
    reactant_template = molecule_template_from_smiles(
        modules,
        openff_modules,
        reactant.smiles,
        reactant.name,
    )
    solvent_template = molecule_template_from_smiles(
        modules,
        openff_modules,
        solvent_component.smiles or SOLVENT_SMILES,
        solvent_component.name,
    )

    smoke_build = (
        OpenMMSmokeBuilder.from_plan(
            modules=modules,
            plan=plan,
            construction_fn=build_openmm_smoke_system,
        )
        .add_surface()
        .add_sam_layer(sam_template)
        .add_reactants(reactant_template, count=reactant_count)
        .add_solvent(solvent_template, count=solvent_count)
        .finalize(
            OpenMMSmokeBuildOptions(
                sulfur_height_nm=args.sulfur_height_nm,
                solvent_padding_nm=args.solvent_padding_nm,
                packmol_working_dir=paths.packmol_dir,
                pressure_bar=config.simulation.pressure_bar,
                temperature_k=config.simulation.temperature_k,
                pd_s_sigma_nm=args.pd_s_sigma_angstrom * 0.1,
                pd_s_epsilon_kcal_mol=args.pd_s_epsilon_kcal_mol,
            )
        )
    )

    write_build_config(
        paths.build_config,
        config,
        args,
        solvent_count,
        sam_template.smiles,
        reactant_count,
        schedule,
    )
    write_pdbx(paths.topology, modules.app, smoke_build.topology, smoke_build.positions_quantity)
    safe_write_text(
        paths.system_xml,
        modules.openmm.XmlSerializer.serialize(smoke_build.system),
        overwrite=True,
    )
    safe_write_text(
        paths.anchor_metadata,
        json.dumps(anchor_metadata(smoke_build), indent=2, sort_keys=True) + "\n",
        overwrite=True,
    )

    selection = create_simulation_with_platform_fallback(
        smoke_build.topology,
        smoke_build.system,
        smoke_build.positions_quantity,
        platform_name=args.platform,
        temperature_k=config.simulation.temperature_k,
        timestep_fs=args.timestep_fs,
        friction_per_ps=args.friction_per_ps,
        openmm_module=modules.openmm,
        app_module=modules.app,
        unit_module=modules.unit,
    )
    simulation = selection.simulation
    simulation.reporters.append(
        modules.app.DCDReporter(
            str(paths.trajectory),
            schedule.report_interval,
            enforcePeriodicBox=True,
        )
    )
    simulation.reporters.append(
        modules.app.StateDataReporter(
            str(paths.thermodynamics),
            schedule.report_interval,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            speed=True,
            separator=",",
        )
    )

    initial = read_energy(simulation, include_kinetic=False, unit_module=modules.unit)
    simulation.minimizeEnergy(maxIterations=args.minimize_iterations)
    minimized = read_energy(simulation, include_kinetic=False, unit_module=modules.unit)
    minimized_positions = simulation.context.getState(getPositions=True).getPositions()
    write_pdbx(paths.minimized_positions, modules.app, smoke_build.topology, minimized_positions)

    simulation.context.setVelocitiesToTemperature(config.simulation.temperature_k, args.seed)
    if schedule.total_steps > 0:
        simulation.step(schedule.total_steps)
    final = read_energy(simulation, include_kinetic=True, unit_module=modules.unit)
    final_state = simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
    final_positions = final_state.getPositions()
    write_pdbx(paths.final_positions, modules.app, smoke_build.topology, final_positions)
    safe_write_text(
        paths.state_xml,
        modules.openmm.XmlSerializer.serialize(final_state),
        overwrite=True,
    )
    simulation.saveCheckpoint(str(paths.checkpoint))

    final_positions_nm = positions_to_nm(final_positions, unit_module=modules.unit)
    pd_displacements = indexed_displacements(
        final_positions_nm,
        smoke_build.positions_nm,
        smoke_build.pd_indices,
    )
    sulfur_displacements = indexed_reference_displacements(
        final_positions_nm,
        smoke_build.sulfur_indices,
        smoke_build.sulfur_reference_positions_nm,
    )
    summary = smoke_summary(
        plan=plan,
        smoke_build=smoke_build,
        paths=paths,
        platform_name=selection.platform_name,
        platform_errors=selection.errors,
        initial=initial,
        minimized=minimized,
        final=final,
        pd_displacements=pd_displacements,
        sulfur_displacements=sulfur_displacements,
        schedule=schedule,
        minimize_iterations=args.minimize_iterations,
        seed=args.seed,
    )
    write_summary(paths.summary, summary)
    write_summary_csv(paths.output_dir / "smoke_metrics.csv", summary)

    print(json.dumps(summary["pass_fail"], indent=2, sort_keys=True))
    print(f"Smoke summary: {paths.summary}")
    return 0 if summary["pass_fail"]["passed"] else 1


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse smoke workflow arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional SAMMD YAML config to use")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Replace prior smoke outputs")
    parser.add_argument("--platform", default="auto", help="auto, CUDA, CPU, or Reference")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "Finite-system construction and velocity seed. Non-default seeds change SAM/site "
            "placement and initial velocities, so use the default seed 2026 for the canonical "
            "smoke validation and alternate seeds for seed-sensitivity checks."
        ),
    )
    parser.add_argument("--lateral-size-nm", type=float, default=DEFAULT_LATERAL_SIZE_NM)
    parser.add_argument("--solvent-padding-nm", type=float, default=DEFAULT_SOLVENT_PADDING_NM)
    parser.add_argument("--solvent-count", default=DEFAULT_SOLVENT_COUNT)
    parser.add_argument("--water-count", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--reactant-count", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--duration-ns", type=float, default=DEFAULT_DURATION_NS)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--minimize-iterations", type=int, default=DEFAULT_MINIMIZE_ITERATIONS)
    parser.add_argument("--frames", type=int, default=DEFAULT_TRAJECTORY_FRAMES)
    parser.add_argument(
        "--report-interval",
        type=int,
        default=None,
        help="Override trajectory/thermodynamics interval; otherwise derived from --frames.",
    )
    parser.add_argument("--timestep-fs", type=float, default=DEFAULT_TIMESTEP_FS)
    parser.add_argument("--friction-per-ps", type=float, default=DEFAULT_FRICTION_PER_PS)
    parser.add_argument("--sulfur-height-nm", type=float, default=DEFAULT_SULFUR_HEIGHT_NM)
    parser.add_argument("--pd-s-sigma-angstrom", type=float, default=DEFAULT_PD_S_SIGMA_ANGSTROM)
    parser.add_argument(
        "--pd-s-epsilon-kcal-mol",
        type=float,
        default=DEFAULT_PD_S_EPSILON_KCAL_MOL,
    )
    args = parser.parse_args(argv)
    if args.water_count is not None:
        args.solvent_count = args.water_count
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    """Reject nonphysical CLI values before heavy construction starts."""

    positive_values = {
        "lateral-size-nm": args.lateral_size_nm,
        "solvent-padding-nm": args.solvent_padding_nm,
        "timestep-fs": args.timestep_fs,
        "friction-per-ps": args.friction_per_ps,
        "pd-s-sigma-angstrom": args.pd_s_sigma_angstrom,
        "pd-s-epsilon-kcal-mol": args.pd_s_epsilon_kcal_mol,
        "duration-ns": args.duration_ns,
    }
    for name, value in positive_values.items():
        if not math.isfinite(value) or value <= 0:
            raise SystemExit(f"--{name} must be positive and finite")
    if not math.isfinite(args.sulfur_height_nm) or args.sulfur_height_nm < 0:
        raise SystemExit("--sulfur-height-nm must be non-negative and finite")
    if args.seed < 0:
        raise SystemExit("--seed must be non-negative")
    if args.steps is not None and args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.minimize_iterations < 0:
        raise SystemExit("--minimize-iterations must be non-negative")
    if args.frames <= 0:
        raise SystemExit("--frames must be positive")
    if args.report_interval is not None and args.report_interval <= 0:
        raise SystemExit("--report-interval must be positive")
    if args.reactant_count is not None and args.reactant_count <= 0:
        raise SystemExit("--reactant-count must be positive")
    if args.solvent_count != "auto":
        try:
            solvent_count = int(args.solvent_count)
        except ValueError as error:
            raise SystemExit("--solvent-count must be 'auto' or a positive integer") from error
        if solvent_count <= 0:
            raise SystemExit("--solvent-count must be 'auto' or a positive integer")


def resolve_solvent_count(solvent_count: str, plan: Any, solvent_name: str) -> int:
    """Resolve explicit or plan-derived solvent count."""

    if solvent_count != "auto":
        return int(solvent_count)
    for component in plan.solution.solvent_components:
        if component.name == solvent_name:
            return max(1, component.count)
    raise ValueError(f"solvent component {solvent_name!r} was not planned")


def resolve_reactant_count(reactant_count: int | None, plan: Any, reactant_name: str) -> int:
    """Resolve explicit or concentration-derived reactant count."""

    if reactant_count is not None:
        return reactant_count
    for reactant in plan.solution.reactants:
        if reactant.name == reactant_name:
            return max(1, reactant.count)
    raise ValueError(f"reactant {reactant_name!r} was not planned")


def smoke_summary(
    *,
    plan: Any,
    smoke_build: SmokeBuild,
    paths: SmokePaths,
    platform_name: str,
    platform_errors: tuple[str, ...],
    initial: EnergyRecord,
    minimized: EnergyRecord,
    final: EnergyRecord,
    pd_displacements: tuple[float, ...],
    sulfur_displacements: tuple[float, ...],
    schedule: RunSchedule,
    minimize_iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Assemble pass/fail and provenance metrics."""

    max_pd_displacement = max(pd_displacements) if pd_displacements else math.inf
    mean_pd_displacement = (
        sum(pd_displacements) / len(pd_displacements) if pd_displacements else math.inf
    )
    max_sulfur_displacement = max(sulfur_displacements) if sulfur_displacements else math.inf
    energies_are_finite = all(
        math.isfinite(value)
        for value in (
            initial.potential_energy_kj_mol,
            minimized.potential_energy_kj_mol,
            final.potential_energy_kj_mol,
        )
    )
    minimization_lowered_energy = (
        minimized.potential_energy_kj_mol <= initial.potential_energy_kj_mol
    )
    temperature_is_sane = (
        final.temperature_k is None or final.temperature_k < PASS_MAX_TEMPERATURE_K
    )
    pd_is_stable = max_pd_displacement < PASS_MAX_PD_DISPLACEMENT_NM
    sulfur_is_stable = max_sulfur_displacement < PASS_MAX_SULFUR_DISPLACEMENT_NM
    passed = all(
        (
            energies_are_finite,
            minimization_lowered_energy,
            temperature_is_sane,
            pd_is_stable,
            sulfur_is_stable,
        )
    )

    return {
        "system": {
            "pd_atoms": len(smoke_build.pd_indices),
            "pd_mobility": "mobile_unrestrained",
            "sam_molecules": smoke_build.sam_count,
            "sam_molecules_per_face": plan.sam_placements.selected_sites_per_side,
            "solvent": {
                "name": SOLVENT_NAME,
                "residue_name": SOLVENT_RESIDUE_NAME,
                "smiles": SOLVENT_SMILES,
                "molecules": smoke_build.solvent_count,
            },
            "reactant_molecules": smoke_build.reactant_count,
            "total_atoms": smoke_build.system.getNumParticles(),
            "box_dimensions_nm": list(smoke_build.platform_dimensions_nm),
            "anchor_pairs": len(smoke_build.anchor_pairs),
            "anchor_pairs_added": smoke_build.anchor_scaling.pairs_added,
            "pd_s_anchor_sigma_nm": smoke_build.anchor_scaling.sigma_nm[0]
            if smoke_build.anchor_scaling.sigma_nm
            else None,
            "pd_s_anchor_epsilon_kj_mol": smoke_build.anchor_scaling.epsilon_delta_kj_mol[0]
            if smoke_build.anchor_scaling.epsilon_delta_kj_mol
            else None,
            "component_chain_ranges": smoke_build.component_chain_ranges,
        },
        "run": {
            "platform": platform_name,
            "platform_errors": list(platform_errors),
            "build_seed": seed,
            "velocity_seed": seed,
            "canonical_validation_seed": DEFAULT_SEED,
            "ensemble": smoke_build.ensemble,
            "pressure_bar": smoke_build.pressure_bar,
            "temperature_k": smoke_build.temperature_k,
            "requested_duration_ns": schedule.requested_duration_ns,
            "simulated_duration_ns": schedule.simulated_duration_ns,
            "timestep_fs": schedule.timestep_fs,
            "steps": schedule.total_steps,
            "trajectory_frames": schedule.frames,
            "report_interval_steps": schedule.report_interval,
            "minimize_iterations": minimize_iterations,
        },
        "energies": {
            "initial_potential_kj_mol": initial.potential_energy_kj_mol,
            "minimized_potential_kj_mol": minimized.potential_energy_kj_mol,
            "final_potential_kj_mol": final.potential_energy_kj_mol,
            "final_kinetic_kj_mol": final.kinetic_energy_kj_mol,
            "final_temperature_k": final.temperature_k,
        },
        "stability": {
            "max_pd_displacement_nm": max_pd_displacement,
            "mean_pd_displacement_nm": mean_pd_displacement,
            "max_sulfur_anchor_displacement_nm": max_sulfur_displacement,
        },
        "pass_fail": {
            "passed": passed,
            "energies_are_finite": energies_are_finite,
            "minimization_lowered_energy": minimization_lowered_energy,
            "temperature_is_sane": temperature_is_sane,
            "pd_slab_is_stable": pd_is_stable,
            "sam_sulfur_anchors_are_stable": sulfur_is_stable,
        },
        "files": {
            "output_dir": str(paths.output_dir),
            "topology": str(paths.topology),
            "trajectory": str(paths.trajectory),
            "thermodynamics": str(paths.thermodynamics),
            "checkpoint": str(paths.checkpoint),
            "state_xml": str(paths.state_xml),
            "system_xml": str(paths.system_xml),
            "anchor_metadata": str(paths.anchor_metadata),
            "summary": str(paths.summary),
            "packmol_dir": str(paths.packmol_dir),
        },
    }


def write_build_config(
    path: Path,
    config: SAMMDConfig,
    args: argparse.Namespace,
    solvent_count: int,
    sam_template_smiles: str,
    reactant_count: int,
    schedule: RunSchedule,
) -> None:
    """Write the validated config plus smoke-only runtime arguments."""

    payload = {
        "sammd_config": config.model_dump(mode="json"),
        "smoke_overrides": {
            "platform": args.platform,
            "build_seed": args.seed,
            "velocity_seed": args.seed,
            "canonical_validation_seed": DEFAULT_SEED,
            "solvent_name": SOLVENT_NAME,
            "solvent_residue_name": SOLVENT_RESIDUE_NAME,
            "solvent_smiles": SOLVENT_SMILES,
            "solvent_count": solvent_count,
            "sam_template_smiles": sam_template_smiles,
            "organic_force_field": OPENFF_FORCE_FIELD,
            "organic_charge_model": NAGL_CHARGE_MODEL,
            "pd_s_sigma_angstrom": args.pd_s_sigma_angstrom,
            "pd_s_epsilon_kcal_mol": args.pd_s_epsilon_kcal_mol,
            "pd_s_reference_epsilon_kcal_mol": REFERENCE_PD_S_EPSILON_KCAL_MOL,
            "pd_mobility": "mobile_unrestrained",
            "reactant_count": reactant_count,
            "duration_ns": args.duration_ns,
            "requested_steps": args.steps,
            "timestep_fs": schedule.timestep_fs,
            "friction_per_ps": args.friction_per_ps,
            "steps": schedule.total_steps,
            "simulated_duration_ns": schedule.simulated_duration_ns,
            "trajectory_frames": schedule.frames,
            "report_interval_steps": schedule.report_interval,
            "minimize_iterations": args.minimize_iterations,
            "sulfur_height_nm": args.sulfur_height_nm,
        },
    }
    safe_write_text(path, yaml.safe_dump(payload, sort_keys=False), overwrite=True)


def anchor_metadata(smoke_build: SmokeBuild) -> dict[str, Any]:
    """Return serializable sulfur-metal anchor metadata."""

    return {
        "sulfur_indices": list(smoke_build.sulfur_indices),
        "sulfur_reference_positions_nm": [
            list(position) for position in smoke_build.sulfur_reference_positions_nm
        ],
        "sulfur_metal_pairs": [list(pair) for pair in smoke_build.anchor_pairs],
        "pairs_added": smoke_build.anchor_scaling.pairs_added,
        "sigma_nm": list(smoke_build.anchor_scaling.sigma_nm),
        "epsilon_kj_mol": list(smoke_build.anchor_scaling.epsilon_delta_kj_mol),
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    """Write the JSON smoke summary."""

    safe_write_text(path, json.dumps(summary, indent=2, sort_keys=True) + "\n", overwrite=True)


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    """Write a compact one-row metric CSV for quick inspection."""

    row = {
        "passed": summary["pass_fail"]["passed"],
        "platform": summary["run"]["platform"],
        "total_atoms": summary["system"]["total_atoms"],
        "pd_atoms": summary["system"]["pd_atoms"],
        "sam_molecules": summary["system"]["sam_molecules"],
        "solvent_molecules": summary["system"]["solvent"]["molecules"],
        "initial_potential_kj_mol": summary["energies"]["initial_potential_kj_mol"],
        "minimized_potential_kj_mol": summary["energies"]["minimized_potential_kj_mol"],
        "final_potential_kj_mol": summary["energies"]["final_potential_kj_mol"],
        "final_temperature_k": summary["energies"]["final_temperature_k"],
        "max_pd_displacement_nm": summary["stability"]["max_pd_displacement_nm"],
        "max_sulfur_anchor_displacement_nm": summary["stability"][
            "max_sulfur_anchor_displacement_nm"
        ],
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def write_pdbx(path: Path, app: Any, topology: Any, positions: Any) -> None:
    """Write an OpenMM PDBx/mmCIF coordinate file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        app.PDBxFile.writeFile(topology, positions, handle, keepIds=True)


if __name__ == "__main__":
    sys.exit(main())
