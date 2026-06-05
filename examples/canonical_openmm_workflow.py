"""Notebook-shaped SAMMD OpenMM workflow prototype.

This script builds and briefly runs a small Pd(111)/thiol-SAM/reactant/ethanol
system. It is a smoke validation workflow, not a production simulation. The same
sections are intended to become notebook cells after the remaining private
backend calls are replaced by public SAMMD teaching APIs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# These private imports are the main reason this remains an instructor-facing
# prototype rather than an undergraduate-ready notebook.
from sammd._openmm_backend import build_openmm_smoke_system
from sammd._openmm_templates import (
    molecule_template_from_smiles,
    require_openff_modules,
    require_openmm_modules,
)
from sammd.builders import build_system
from sammd.io import safe_write_text
from sammd.openmm_build import OpenMMSmokeBuilder, OpenMMSmokeBuildOptions
from sammd.openmm_runtime import create_simulation_with_platform_fallback, read_energy
from sammd.workflow import (
    CANONICAL_SMOKE_SOLVENT_SMILES,
    canonical_smoke_config,
    prepare_outputs,
    resolve_run_schedule,
    smoke_paths,
)


def main() -> int:
    """Run the compact teaching workflow."""

    args = parse_args()

    # --- Science choices ---
    # This block is what a future notebook should expose to students. In this
    # prototype, several values still flow through the canonical demo helper
    # rather than a polished public user-facing API.
    lateral_size_nm = 2.0
    solvent_padding_nm = 3.0
    seed = 2026
    timestep_fs = 2.0
    sulfur_height_nm = 0.18
    pd_s_sigma_nm = 0.22
    pd_s_epsilon_kcal_mol = 2.0

    print("SAMMD canonical OpenMM workflow prototype")
    print("Goal: build and sanity-check a small thiol SAM on Pd(111).")
    print("This is a smoke test, not a production MD study.\n")

    paths = smoke_paths(args.output_dir)
    prepare_outputs(paths, overwrite=args.overwrite)
    print(f"Writing outputs to: {paths.output_dir}")

    config = canonical_smoke_config(
        lateral_size_nm=lateral_size_nm,
        solvent_padding_nm=solvent_padding_nm,
        seed=seed,
        timestep_fs=timestep_fs,
        reporter_interval_steps=1,
    )
    print("Configured science system:")
    print(f"  surface: {config.surface.metal}({config.surface.facet})")
    print(f"  SAM: {config.sam.components[0].name} ({config.sam.components[0].smiles})")
    reactant_concentration = config.reactants[0].concentration_millimolar
    print(f"  reactant: {config.reactants[0].name} at {reactant_concentration:g} mM")
    print(f"  solvent: {config.solvent.components[0].name}, mole fraction 1.0\n")

    plan = build_system(config, output_dir=paths.output_dir, seed=seed)
    schedule = resolve_run_schedule(
        duration_ns=args.duration_ns,
        timestep_fs=timestep_fs,
        steps=None,
        frames=args.frames,
        report_interval=None,
    )
    print("Resolved short smoke schedule:")
    print(f"  steps: {schedule.total_steps}")
    print(f"  timestep: {schedule.timestep_fs:g} fs")
    print(f"  simulated time: {schedule.simulated_duration_ns:g} ns")
    print(f"  trajectory frames: {schedule.frames}\n")

    print("Loading OpenMM/OpenFF science dependencies...")
    modules = require_openmm_modules()
    openff_modules = require_openff_modules()

    sam_component = config.sam.components[0]
    reactant = config.reactants[0]
    solvent = config.solvent.components[0]
    sam_template = molecule_template_from_smiles(
        modules,
        openff_modules,
        sam_component.smiles,
        sam_component.name,
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
        solvent.smiles or CANONICAL_SMOKE_SOLVENT_SMILES,
        solvent.name,
    )

    print("Building staged OpenMM system:")
    print("  1. add_surface()")
    print("  2. add_sam_layer()")
    print("  3. add_reactants()")
    print("  4. add_solvent()")
    smoke_build = (
        OpenMMSmokeBuilder.from_plan(
            modules=modules,
            plan=plan,
            construction_fn=build_openmm_smoke_system,
        )
        .add_surface()
        .add_sam_layer(sam_template)
        .add_reactants(reactant_template, count=plan.solution.reactants[0].count)
        .add_solvent(solvent_template, count=plan.solution.solvent_components[0].count)
        .finalize(
            OpenMMSmokeBuildOptions(
                sulfur_height_nm=sulfur_height_nm,
                solvent_padding_nm=solvent_padding_nm,
                packmol_working_dir=paths.packmol_dir,
                pressure_bar=config.simulation.pressure_bar,
                temperature_k=config.simulation.temperature_k,
                pd_s_sigma_nm=pd_s_sigma_nm,
                pd_s_epsilon_kcal_mol=pd_s_epsilon_kcal_mol,
            )
        )
    )
    print(f"Built {smoke_build.system.getNumParticles()} atoms.")
    print(f"  Pd atoms: {len(smoke_build.pd_indices)}")
    print(f"  SAM molecules: {smoke_build.sam_count}")
    print(f"  reactant molecules: {smoke_build.reactant_count}")
    print(f"  solvent molecules: {smoke_build.solvent_count}\n")

    write_pdbx(paths.topology, modules.app, smoke_build.topology, smoke_build.positions_quantity)
    safe_write_text(
        paths.system_xml, modules.openmm.XmlSerializer.serialize(smoke_build.system), overwrite=True
    )

    print("Requesting CUDA OpenMM simulation; falling back if unavailable...")
    selection = create_simulation_with_platform_fallback(
        smoke_build.topology,
        smoke_build.system,
        smoke_build.positions_quantity,
        platform_name="CUDA",
        temperature_k=config.simulation.temperature_k,
        timestep_fs=timestep_fs,
        friction_per_ps=1.0,
        openmm_module=modules.openmm,
        app_module=modules.app,
        unit_module=modules.unit,
    )
    simulation = selection.simulation
    simulation.reporters.append(
        modules.app.DCDReporter(
            str(paths.trajectory), schedule.report_interval, enforcePeriodicBox=True
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
            separator=",",
        )
    )
    print(f"Platform: {selection.platform_name}\n")

    print("Minimizing and running a short smoke simulation...")
    initial = read_energy(simulation, include_kinetic=False, unit_module=modules.unit)
    simulation.minimizeEnergy(maxIterations=0)
    minimized = read_energy(simulation, include_kinetic=False, unit_module=modules.unit)
    simulation.context.setVelocitiesToTemperature(config.simulation.temperature_k, seed)
    simulation.step(schedule.total_steps)
    final = read_energy(simulation, include_kinetic=True, unit_module=modules.unit)

    summary = {
        "platform": selection.platform_name,
        "atoms": smoke_build.system.getNumParticles(),
        "steps": schedule.total_steps,
        "simulated_duration_ns": schedule.simulated_duration_ns,
        "initial_potential_kj_mol": initial.potential_energy_kj_mol,
        "minimized_potential_kj_mol": minimized.potential_energy_kj_mol,
        "final_potential_kj_mol": final.potential_energy_kj_mol,
        "final_temperature_k": final.temperature_k,
        "topology": str(paths.topology),
        "trajectory": str(paths.trajectory),
        "thermodynamics": str(paths.thermodynamics),
    }
    safe_write_text(paths.summary, json.dumps(summary, indent=2) + "\n", overwrite=True)

    print("Smoke simulation completed.")
    print(f"Initial potential energy: {initial.potential_energy_kj_mol:.3g} kJ/mol")
    print(f"Minimized potential energy: {minimized.potential_energy_kj_mol:.3g} kJ/mol")
    print(f"Final temperature: {final.temperature_k:.2f} K")
    print(
        "Expected smoke-run signs: minimization succeeds, no NaNs, "
        "and final temperature is near target."
    )
    print("\nImportant: this confirms the workflow runs; it does not prove scientific convergence.")
    print(f"Summary: {paths.summary}")
    print(f"Trajectory: {paths.trajectory}")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse small teaching-example controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/openmm_smoke/canonical_teaching_example"),
    )
    parser.add_argument("--duration-ns", type=float, default=0.1)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_pdbx(path: Path, app: object, topology: object, positions: object) -> None:
    """Write an OpenMM topology/positions pair as PDBx/mmCIF."""

    with path.open("w", encoding="utf-8") as handle:
        app.PDBxFile.writeFile(topology, positions, handle, keepIds=True)


if __name__ == "__main__":
    raise SystemExit(main())
