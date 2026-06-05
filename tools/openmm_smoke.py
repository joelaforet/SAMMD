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
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sammd.builders import build_system
from sammd.config import SAMMDConfig, load_config
from sammd.forcefields import get_fcc_metal_parameters
from sammd.geometry import (
    Vector3,
    add_vectors,
    centroid,
    distance,
    indexed_displacements,
    indexed_reference_displacements,
    matvec,
    rotate_about_axis,
    rotation_matrix,
    scale_vector,
    subtract_vectors,
)
from sammd.io import safe_write_text
from sammd.openmm_runtime import AnchorScalingMetadata, create_langevin_integrator
from sammd.topology import ComponentResidueAssigner, ResidueIdentity, get_or_add_chain

KCAL_TO_KJ = 4.184
ETHANOL_MASS_G_MOL = 46.06844
ETHANOL_DENSITY_G_ML = 0.789
SOLVENT_NAME = "ethanol"
SOLVENT_RESIDUE_NAME = "EOH"
SOLVENT_SMILES = "CCO"
DEFAULT_OUTPUT_DIR = "outputs/openmm_smoke/pd111_propanethiol_cinnamaldehyde"
DEFAULT_LATERAL_SIZE_NM = 2.0
DEFAULT_SOLVENT_PADDING_NM = 3.0
DEFAULT_SOLVENT_COUNT = "auto"
DEFAULT_DURATION_NS = 5.0
DEFAULT_MINIMIZE_ITERATIONS = 0
DEFAULT_TRAJECTORY_FRAMES = 300
DEFAULT_TIMESTEP_FS = 2.0
DEFAULT_FRICTION_PER_PS = 1.0
DEFAULT_SULFUR_HEIGHT_NM = 0.18
DEFAULT_PD_S_SIGMA_ANGSTROM = 2.2
REFERENCE_PD_S_EPSILON_KCAL_MOL = 1.0
DEFAULT_PD_S_EPSILON_KCAL_MOL = 2.0
OPENFF_FORCE_FIELD = "openff-2.2.1.offxml"
NAGL_CHARGE_MODEL = "openff-gnn-am1bcc-1.0.0.pt"
SAM_TAIL_CLEARANCE_NM = 0.95
PACKMOL_TOLERANCE_ANGSTROM = 1.8
PACKMOL_NLOOP = 200
PASS_MAX_TEMPERATURE_K = 600.0
PASS_MAX_PD_DISPLACEMENT_NM = 0.15
PASS_MAX_SULFUR_DISPLACEMENT_NM = 0.70

@dataclass(frozen=True)
class BondParameter:
    """OpenFF-derived harmonic bond parameter."""

    atom1: int
    atom2: int
    length_nm: float
    k_kj_mol_nm2: float


@dataclass(frozen=True)
class AngleParameter:
    """OpenFF-derived harmonic angle parameter."""

    atom1: int
    atom2: int
    atom3: int
    angle_rad: float
    k_kj_mol_rad2: float


@dataclass(frozen=True)
class TorsionParameter:
    """OpenFF-derived periodic torsion parameter."""

    atom1: int
    atom2: int
    atom3: int
    atom4: int
    periodicity: int
    phase_rad: float
    k_kj_mol: float


@dataclass(frozen=True)
class ConstraintParameter:
    """OpenFF-derived distance constraint."""

    atom1: int
    atom2: int
    distance_nm: float


@dataclass(frozen=True)
class ExceptionParameter:
    """OpenFF-derived intramolecular nonbonded exception."""

    atom1: int
    atom2: int
    chargeprod_e2: float
    sigma_nm: float
    epsilon_kj_mol: float


@dataclass(frozen=True)
class AtomTemplate:
    """Per-atom template values used for direct OpenMM construction."""

    name: str
    element: str
    charge_e: float
    sigma_nm: float
    epsilon_kj_mol: float


@dataclass(frozen=True)
class MoleculeTemplate:
    """Small molecule coordinates and topology from RDKit."""

    name: str
    smiles: str
    atoms: tuple[AtomTemplate, ...]
    bonds: tuple[tuple[int, int], ...]
    bond_parameters: tuple[BondParameter, ...]
    angle_parameters: tuple[AngleParameter, ...]
    torsion_parameters: tuple[TorsionParameter, ...]
    constraints: tuple[ConstraintParameter, ...]
    exception_parameters: tuple[ExceptionParameter, ...]
    positions_nm: tuple[Vector3, ...]
    charge_model: str
    force_field: str


@dataclass(frozen=True)
class SmokePaths:
    """Resolved files written by the smoke workflow."""

    output_dir: Path
    build_config: Path
    topology: Path
    minimized_positions: Path
    final_positions: Path
    trajectory: Path
    thermodynamics: Path
    checkpoint: Path
    state_xml: Path
    system_xml: Path
    anchor_metadata: Path
    summary: Path
    packmol_dir: Path


@dataclass(frozen=True)
class SmokeBuild:
    """Constructed OpenMM objects plus index metadata for stability checks."""

    topology: Any
    system: Any
    positions_nm: tuple[Vector3, ...]
    positions_quantity: Any
    pd_indices: tuple[int, ...]
    sulfur_indices: tuple[int, ...]
    sulfur_reference_positions_nm: tuple[Vector3, ...]
    anchor_pairs: tuple[tuple[int, int], ...]
    anchor_scaling: Any
    solvent_count: int
    reactant_count: int
    sam_count: int
    platform_dimensions_nm: Vector3
    component_chain_ranges: dict[str, dict[str, object]]
    ensemble: str
    pressure_bar: float
    temperature_k: float


@dataclass(frozen=True)
class PlatformSelection:
    """OpenMM platform selection result."""

    simulation: Any
    platform_name: str
    errors: tuple[str, ...]


@dataclass(frozen=True)
class EnergyRecord:
    """Potential/kinetic/temperature values extracted from an OpenMM State."""

    potential_energy_kj_mol: float
    kinetic_energy_kj_mol: float | None
    temperature_k: float | None


@dataclass(frozen=True)
class RunSchedule:
    """Resolved integration and reporting schedule."""

    requested_duration_ns: float
    simulated_duration_ns: float
    total_steps: int
    report_interval: int
    frames: int
    timestep_fs: float


@dataclass(frozen=True)
class PackmolSolutionPositions:
    """Packmol-generated positions grouped by molecule type."""

    solvent_positions_nm: tuple[tuple[Vector3, ...], ...]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the real-system smoke workflow."""

    args = parse_args(argv)
    modules = require_openmm_modules()
    openff_modules = require_openff_modules()
    paths = smoke_paths(Path(args.output_dir))
    prepare_outputs(paths, overwrite=args.overwrite)

    config = load_smoke_config(args)
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
        args.seed,
    )
    reactant_template = molecule_template_from_smiles(
        modules,
        openff_modules,
        reactant.smiles,
        reactant.name,
        args.seed + 17,
    )
    solvent_template = molecule_template_from_smiles(
        modules,
        openff_modules,
        solvent_component.smiles or SOLVENT_SMILES,
        solvent_component.name,
        args.seed + 31,
    )

    smoke_build = build_openmm_smoke_system(
        modules,
        plan,
        sam_template,
        reactant_template,
        solvent_template,
        solvent_count=solvent_count,
        reactant_count=reactant_count,
        sulfur_height_nm=args.sulfur_height_nm,
        solvent_padding_nm=args.solvent_padding_nm,
        packmol_working_dir=paths.packmol_dir,
        pressure_bar=config.simulation.pressure_bar,
        temperature_k=config.simulation.temperature_k,
        pd_s_sigma_nm=args.pd_s_sigma_angstrom * 0.1,
        pd_s_epsilon_kcal_mol=args.pd_s_epsilon_kcal_mol,
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
        modules,
        smoke_build,
        platform_name=args.platform,
        temperature_k=config.simulation.temperature_k,
        timestep_fs=args.timestep_fs,
        friction_per_ps=args.friction_per_ps,
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

    initial = read_energy(modules, simulation, include_kinetic=False)
    simulation.minimizeEnergy(maxIterations=args.minimize_iterations)
    minimized = read_energy(modules, simulation, include_kinetic=False)
    minimized_positions = simulation.context.getState(getPositions=True).getPositions()
    write_pdbx(paths.minimized_positions, modules.app, smoke_build.topology, minimized_positions)

    simulation.context.setVelocitiesToTemperature(config.simulation.temperature_k, args.seed)
    if schedule.total_steps > 0:
        simulation.step(schedule.total_steps)
    final = read_energy(modules, simulation, include_kinetic=True)
    final_state = simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
    final_positions = final_state.getPositions()
    write_pdbx(paths.final_positions, modules.app, smoke_build.topology, final_positions)
    safe_write_text(
        paths.state_xml,
        modules.openmm.XmlSerializer.serialize(final_state),
        overwrite=True,
    )
    simulation.saveCheckpoint(str(paths.checkpoint))

    final_positions_nm = positions_to_nm(modules, final_positions)
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
    parser.add_argument("--seed", type=int, default=2026)
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
    if args.steps is not None and args.steps < 0:
        raise SystemExit("--steps must be non-negative")
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


def require_openmm_modules() -> Any:
    """Import OpenMM lazily with guidance for pixi science users."""

    try:
        import openmm
        from openmm import app, unit
    except ImportError as error:
        msg = "OpenMM is required; run this through `pixi run -e science real-system-smoke`."
        raise SystemExit(msg) from error
    return type("OpenMMModules", (), {"openmm": openmm, "app": app, "unit": unit})


def require_openff_modules() -> Any:
    """Import OpenFF Toolkit/NAGL lazily for proper small-molecule parameters."""

    try:
        from openff.toolkit import ForceField, Molecule, ToolkitRegistry
        from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper
        from openff.toolkit.utils.rdkit_wrapper import RDKitToolkitWrapper
    except ImportError as error:
        msg = (
            "OpenFF Toolkit with NAGL support is required; run this through "
            "`pixi run -e science real-system-smoke`."
        )
        raise SystemExit(msg) from error
    return type(
        "OpenFFModules",
        (),
        {
            "ForceField": ForceField,
            "Molecule": Molecule,
            "ToolkitRegistry": ToolkitRegistry,
            "NAGLToolkitWrapper": NAGLToolkitWrapper,
            "RDKitToolkitWrapper": RDKitToolkitWrapper,
        },
    )


def smoke_paths(output_dir: Path) -> SmokePaths:
    """Resolve deterministic smoke output paths."""

    return SmokePaths(
        output_dir=output_dir,
        build_config=output_dir / "build_config.yaml",
        topology=output_dir / "topology.cif",
        minimized_positions=output_dir / "minimized_positions.cif",
        final_positions=output_dir / "final_positions.cif",
        trajectory=output_dir / "trajectory.dcd",
        thermodynamics=output_dir / "thermodynamics.csv",
        checkpoint=output_dir / "checkpoint.chk",
        state_xml=output_dir / "state.xml",
        system_xml=output_dir / "system.xml",
        anchor_metadata=output_dir / "anchor_metadata.json",
        summary=output_dir / "smoke_summary.json",
        packmol_dir=output_dir / "packmol",
    )


def prepare_outputs(paths: SmokePaths, *, overwrite: bool) -> None:
    """Create output directory and enforce safe overwrite semantics."""

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths.__dict__.values():
        if path == paths.output_dir:
            continue
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing smoke output: {path}")
        if path.exists() and overwrite:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    metrics_path = paths.output_dir / "smoke_metrics.csv"
    if metrics_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing smoke output: {metrics_path}")
    if metrics_path.exists() and overwrite:
        metrics_path.unlink()


def load_smoke_config(args: argparse.Namespace) -> SAMMDConfig:
    """Load user config or create a compact default real-system smoke config."""

    if args.config is not None:
        return load_config(args.config)
    return SAMMDConfig.model_validate(
        {
            "surface": {
                "slab": {
                    "lateral_size_nm": [args.lateral_size_nm, args.lateral_size_nm],
                }
            },
            "solvent": {
                "padding_nm": args.solvent_padding_nm,
                "components": [
                    {
                        "name": SOLVENT_NAME,
                        "smiles": SOLVENT_SMILES,
                        "mole_fraction": 1.0,
                        "density_g_ml": ETHANOL_DENSITY_G_ML,
                        "molar_mass_g_mol": ETHANOL_MASS_G_MOL,
                    }
                ],
            },
            "reporters": {"interval_steps": args.report_interval or 1},
            "simulation": {
                "seed": args.seed,
                "timestep_fs": args.timestep_fs,
            },
        }
    )


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


def resolve_run_schedule(
    *,
    duration_ns: float,
    timestep_fs: float,
    steps: int | None,
    frames: int,
    report_interval: int | None,
) -> RunSchedule:
    """Resolve integer MD steps and reporter cadence for an exact frame count."""

    if report_interval is not None:
        total_steps = (
            steps if steps is not None else max(1, round(duration_ns * 1.0e6 / timestep_fs))
        )
        resolved_frames = total_steps // report_interval
        return RunSchedule(
            requested_duration_ns=duration_ns,
            simulated_duration_ns=total_steps * timestep_fs / 1.0e6,
            total_steps=total_steps,
            report_interval=report_interval,
            frames=resolved_frames,
            timestep_fs=timestep_fs,
        )

    if steps is not None:
        total_steps = steps
        report_interval = max(1, round(steps / frames))
        resolved_frames = total_steps // report_interval
    else:
        requested_steps = max(1, round(duration_ns * 1.0e6 / timestep_fs))
        report_interval = max(1, math.ceil(requested_steps / frames))
        total_steps = report_interval * frames
        resolved_frames = frames
    return RunSchedule(
        requested_duration_ns=duration_ns,
        simulated_duration_ns=total_steps * timestep_fs / 1.0e6,
        total_steps=total_steps,
        report_interval=report_interval,
        frames=resolved_frames,
        timestep_fs=timestep_fs,
    )


def molecule_template_from_smiles(
    modules: Any,
    openff_modules: Any,
    smiles: str,
    name: str,
    seed: int,
) -> MoleculeTemplate:
    """Build an OpenFF/NAGL-parameterized molecule template."""

    unit = modules.unit
    openff_molecule = openff_modules.Molecule.from_smiles(
        smiles,
        allow_undefined_stereo=True,
    )
    openff_molecule.name = name
    openff_molecule.generate_conformers(
        n_conformers=1,
        toolkit_registry=openff_toolkit_registry(openff_modules),
    )
    openff_molecule.assign_partial_charges(
        NAGL_CHARGE_MODEL,
        toolkit_registry=openff_toolkit_registry(openff_modules),
    )
    force_field = openff_modules.ForceField(OPENFF_FORCE_FIELD)
    openmm_system = force_field.create_openmm_system(
        openff_molecule.to_topology(),
        charge_from_molecules=[openff_molecule],
    )
    (
        nonbonded,
        exception_parameters,
        bond_parameters,
        angle_parameters,
        torsion_parameters,
    ) = extract_openff_forces(modules, openmm_system)
    constraints = tuple(
        ConstraintParameter(
            atom1=openmm_system.getConstraintParameters(index)[0],
            atom2=openmm_system.getConstraintParameters(index)[1],
            distance_nm=openmm_system.getConstraintParameters(index)[2].value_in_unit(unit.nanometer),
        )
        for index in range(openmm_system.getNumConstraints())
    )
    atoms = tuple(
        AtomTemplate(
            name=f"{atom.symbol}{index + 1}",
            element=atom.symbol,
            charge_e=charge,
            sigma_nm=sigma_nm,
            epsilon_kj_mol=epsilon_kj_mol,
        )
        for index, (atom, (charge, sigma_nm, epsilon_kj_mol)) in enumerate(
            zip(openff_molecule.atoms, nonbonded, strict=True)
        )
    )
    conformer = openff_molecule.conformers[0]
    positions = tuple(
        (
            conformer[index][0].m_as("nanometer"),
            conformer[index][1].m_as("nanometer"),
            conformer[index][2].m_as("nanometer"),
        )
        for index in range(openff_molecule.n_atoms)
    )
    bonds = tuple(
        tuple(sorted((bond.atom1_index, bond.atom2_index))) for bond in openff_molecule.bonds
    )
    return MoleculeTemplate(
        name=name,
        smiles=smiles,
        atoms=atoms,
        bonds=bonds,
        bond_parameters=bond_parameters,
        angle_parameters=angle_parameters,
        torsion_parameters=torsion_parameters,
        constraints=constraints,
        exception_parameters=exception_parameters,
        positions_nm=positions,
        charge_model=NAGL_CHARGE_MODEL,
        force_field=OPENFF_FORCE_FIELD,
    )


def openff_toolkit_registry(openff_modules: Any) -> Any:
    """Return ToolkitRegistry with NAGL enabled for partial charges."""

    return openff_modules.ToolkitRegistry(
        [openff_modules.NAGLToolkitWrapper(), openff_modules.RDKitToolkitWrapper()]
    )


def extract_openff_forces(
    modules: Any,
    system: Any,
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[ExceptionParameter, ...],
    tuple[BondParameter, ...],
    tuple[AngleParameter, ...],
    tuple[TorsionParameter, ...],
]:
    """Extract OpenFF-generated nonbonded and bonded parameters from an OpenMM System."""

    openmm = modules.openmm
    unit = modules.unit
    nonbonded: list[tuple[float, float, float]] = []
    exceptions: list[ExceptionParameter] = []
    bonds: list[BondParameter] = []
    angles: list[AngleParameter] = []
    torsions: list[TorsionParameter] = []
    for force_index in range(system.getNumForces()):
        force = system.getForce(force_index)
        if isinstance(force, openmm.NonbondedForce):
            nonbonded = [
                (
                    charge.value_in_unit(unit.elementary_charge),
                    sigma.value_in_unit(unit.nanometer),
                    epsilon.value_in_unit(unit.kilojoule_per_mole),
                )
                for charge, sigma, epsilon in (
                    force.getParticleParameters(index)
                    for index in range(force.getNumParticles())
                )
            ]
            exceptions = [
                ExceptionParameter(
                    atom1=atom1,
                    atom2=atom2,
                    chargeprod_e2=chargeprod.value_in_unit(
                        unit.elementary_charge**2
                    ),
                    sigma_nm=sigma.value_in_unit(unit.nanometer),
                    epsilon_kj_mol=epsilon.value_in_unit(unit.kilojoule_per_mole),
                )
                for atom1, atom2, chargeprod, sigma, epsilon in (
                    force.getExceptionParameters(index)
                    for index in range(force.getNumExceptions())
                )
            ]
        elif isinstance(force, openmm.HarmonicBondForce):
            bonds = [
                BondParameter(
                    atom1=atom1,
                    atom2=atom2,
                    length_nm=length.value_in_unit(unit.nanometer),
                    k_kj_mol_nm2=k.value_in_unit(
                        unit.kilojoule_per_mole / unit.nanometer**2
                    ),
                )
                for atom1, atom2, length, k in (
                    force.getBondParameters(index) for index in range(force.getNumBonds())
                )
            ]
        elif isinstance(force, openmm.HarmonicAngleForce):
            angles = [
                AngleParameter(
                    atom1=atom1,
                    atom2=atom2,
                    atom3=atom3,
                    angle_rad=angle.value_in_unit(unit.radian),
                    k_kj_mol_rad2=k.value_in_unit(
                        unit.kilojoule_per_mole / unit.radian**2
                    ),
                )
                for atom1, atom2, atom3, angle, k in (
                    force.getAngleParameters(index) for index in range(force.getNumAngles())
                )
            ]
        elif isinstance(force, openmm.PeriodicTorsionForce):
            torsions = [
                TorsionParameter(
                    atom1=atom1,
                    atom2=atom2,
                    atom3=atom3,
                    atom4=atom4,
                    periodicity=periodicity,
                    phase_rad=phase.value_in_unit(unit.radian),
                    k_kj_mol=k.value_in_unit(unit.kilojoule_per_mole),
                )
                for atom1, atom2, atom3, atom4, periodicity, phase, k in (
                    force.getTorsionParameters(index)
                    for index in range(force.getNumTorsions())
                )
            ]
    if not nonbonded:
        raise RuntimeError("OpenFF template did not contain a NonbondedForce")
    return tuple(nonbonded), tuple(exceptions), tuple(bonds), tuple(angles), tuple(torsions)


def build_openmm_smoke_system(
    modules: Any,
    plan: Any,
    sam_template: MoleculeTemplate,
    reactant_template: MoleculeTemplate,
    solvent_template: MoleculeTemplate,
    *,
    solvent_count: int,
    reactant_count: int,
    sulfur_height_nm: float,
    solvent_padding_nm: float,
    packmol_working_dir: Path,
    pressure_bar: float,
    temperature_k: float,
    pd_s_sigma_nm: float,
    pd_s_epsilon_kcal_mol: float,
) -> SmokeBuild:
    """Build the direct OpenMM topology/system/positions for the smoke run."""

    openmm = modules.openmm
    app = modules.app
    unit = modules.unit
    topology = app.Topology()
    system = openmm.System()
    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.PME)
    nonbonded.setCutoffDistance(plan.config.simulation.nonbonded_cutoff_nm * unit.nanometer)
    nonbonded.setUseSwitchingFunction(True)
    nonbonded.setSwitchingDistance(
        0.9 * plan.config.simulation.nonbonded_cutoff_nm * unit.nanometer
    )
    nonbonded.setUseDispersionCorrection(True)
    bond_force = openmm.HarmonicBondForce()
    angle_force = openmm.HarmonicAngleForce()
    torsion_force = openmm.PeriodicTorsionForce()

    atom_handles: list[Any] = []
    positions_nm: list[Vector3] = []
    all_bonds: list[tuple[int, int]] = []
    pd_indices: list[int] = []
    sulfur_indices: list[int] = []
    sulfur_references: list[Vector3] = []
    anchor_pairs: list[tuple[int, int]] = []
    residue_assigner = ComponentResidueAssigner()
    chain_cache: dict[str, Any] = {}

    box_dimensions_nm = derive_box_dimensions(plan, solvent_padding_nm)
    shift_nm = tuple(dimension / 2.0 for dimension in box_dimensions_nm)
    set_periodic_box(modules, topology, system, box_dimensions_nm)
    pd_identities = residue_assigner.allocate(
        "palladium_slab",
        "PD",
        len(plan.slab.positions_nm),
    )
    sam_identities = residue_assigner.allocate(
        "propanethiolate_sam",
        "PTL",
        len(plan.sam_placements.placements),
    )

    add_pd_slab(
        modules,
        topology,
        system,
        nonbonded,
        atom_handles,
        chain_cache,
        positions_nm,
        pd_indices,
        plan,
        shift_nm,
        pd_identities,
    )
    add_sam_layer(
        modules,
        topology,
        system,
        nonbonded,
        bond_force,
        angle_force,
        torsion_force,
        atom_handles,
        chain_cache,
        positions_nm,
        all_bonds,
        sulfur_indices,
        sulfur_references,
        anchor_pairs,
        plan,
        sam_template,
        shift_nm,
        sulfur_height_nm,
        sam_identities,
    )
    reactant_identities = residue_assigner.allocate(
        "cinnamaldehyde",
        "CIN",
        reactant_count,
    )
    reactant_positions = place_reactants_above_surface(
        plan,
        reactant_template,
        reactant_count,
        shift_nm,
        box_dimensions_nm,
    )
    add_reactants(
        modules,
        topology,
        system,
        nonbonded,
        bond_force,
        angle_force,
        torsion_force,
        atom_handles,
        chain_cache,
        positions_nm,
        all_bonds,
        reactant_template,
        reactant_positions,
        reactant_identities,
    )
    packed_solution = pack_solution_with_packmol(
        topology=topology,
        solute_positions_nm=tuple(positions_nm),
        solvent_template=solvent_template,
        solvent_count=solvent_count,
        box_dimensions_nm=box_dimensions_nm,
        working_dir=packmol_working_dir,
    )
    solvent_identities = residue_assigner.allocate(
        SOLVENT_NAME,
        SOLVENT_RESIDUE_NAME,
        solvent_count,
    )
    placed_solvent = add_solvent_molecules(
        modules,
        topology,
        system,
        nonbonded,
        bond_force,
        angle_force,
        torsion_force,
        atom_handles,
        chain_cache,
        positions_nm,
        all_bonds,
        solvent_template,
        packed_solution.solvent_positions_nm,
        solvent_identities,
    )
    system.addForce(nonbonded)
    system.addForce(bond_force)
    system.addForce(angle_force)
    system.addForce(torsion_force)
    system.addForce(openmm.CMMotionRemover())

    anchor_scaling = add_sulfur_metal_lj_exceptions(
        nonbonded,
        tuple(anchor_pairs),
        sigma_nm=pd_s_sigma_nm,
        epsilon_kcal_mol=pd_s_epsilon_kcal_mol,
        unit=unit,
    )
    positions_quantity = unit.Quantity(
        [openmm.Vec3(*position) for position in positions_nm],
        unit.nanometer,
    )

    return SmokeBuild(
        topology=topology,
        system=system,
        positions_nm=tuple(positions_nm),
        positions_quantity=positions_quantity,
        pd_indices=tuple(pd_indices),
        sulfur_indices=tuple(sulfur_indices),
        sulfur_reference_positions_nm=tuple(sulfur_references),
        anchor_pairs=tuple(anchor_pairs),
        anchor_scaling=anchor_scaling,
        solvent_count=placed_solvent,
        reactant_count=reactant_count,
        sam_count=len(plan.sam_placements.placements),
        platform_dimensions_nm=box_dimensions_nm,
        component_chain_ranges=residue_assigner.component_ranges,
        ensemble="NVT",
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
    )


def derive_box_dimensions(plan: Any, solvent_padding_nm: float) -> Vector3:
    """Return periodic box lengths for slab, SAM, and two solvent regions."""

    return (
        plan.slab.lateral_size_nm[0],
        plan.slab.lateral_size_nm[1],
        plan.slab.slab_extent_nm[2] + 2.0 * (SAM_TAIL_CLEARANCE_NM + solvent_padding_nm),
    )


def set_periodic_box(modules: Any, topology: Any, system: Any, dimensions_nm: Vector3) -> None:
    """Apply orthorhombic periodic box vectors to topology and system."""

    openmm = modules.openmm
    unit = modules.unit
    vectors = (
        openmm.Vec3(dimensions_nm[0], 0.0, 0.0),
        openmm.Vec3(0.0, dimensions_nm[1], 0.0),
        openmm.Vec3(0.0, 0.0, dimensions_nm[2]),
    )
    topology.setPeriodicBoxVectors(vectors)
    system.setDefaultPeriodicBoxVectors(*(vector * unit.nanometer for vector in vectors))


def add_sulfur_metal_lj_exceptions(
    nonbonded: Any,
    pairs: tuple[tuple[int, int], ...],
    *,
    sigma_nm: float,
    epsilon_kcal_mol: float,
    unit: Any,
) -> AnchorScalingMetadata:
    """Override selected Pd-S pairs with literature-style LJ chemisorption mimic."""

    epsilon_kj_mol = epsilon_kcal_mol * KCAL_TO_KJ
    sigmas = []
    epsilons = []
    for sulfur_index, metal_index in pairs:
        nonbonded.addException(
            sulfur_index,
            metal_index,
            0.0 * unit.elementary_charge**2,
            sigma_nm * unit.nanometer,
            epsilon_kj_mol * unit.kilojoule_per_mole,
            replace=True,
        )
        sigmas.append(sigma_nm)
        epsilons.append(epsilon_kj_mol)
    return AnchorScalingMetadata(
        pairs_requested=len(pairs),
        pairs_added=len(pairs),
        force_added=False,
        scale_factor=1.0,
        force_index=None,
        sigma_nm=tuple(sigmas),
        epsilon_delta_kj_mol=tuple(epsilons),
    )


def pack_solution_with_packmol(
    *,
    topology: Any,
    solute_positions_nm: tuple[Vector3, ...],
    solvent_template: MoleculeTemplate,
    solvent_count: int,
    box_dimensions_nm: Vector3,
    working_dir: Path,
) -> PackmolSolutionPositions:
    """Use Packmol to place solvent molecules around Pd+SAM+reactant solute."""

    working_dir.mkdir(parents=True, exist_ok=True)
    solute_path = working_dir / "fixed_solute.pdb"
    solvent_path = working_dir / f"{SOLVENT_NAME}.pdb"
    output_path = working_dir / "packmol_output.pdb"
    input_path = working_dir / "packmol_input.inp"
    stdout_path = working_dir / "packmol_stdout.log"

    write_topology_pdb(solute_path, topology, solute_positions_nm)
    write_molecule_template_pdb(solvent_path, solvent_template, residue_name=SOLVENT_RESIDUE_NAME)

    input_text = build_packmol_input(
        solute_path=solute_path,
        solvent_path=solvent_path,
        output_path=output_path,
        solvent_count=solvent_count,
        box_dimensions_nm=box_dimensions_nm,
    )
    input_path.write_text(input_text, encoding="utf-8")
    run_packmol(input_path, working_dir, stdout_path)

    packed_positions = read_pdb_positions_nm(output_path)
    n_solute_atoms = len(solute_positions_nm)
    n_solvent_atoms = solvent_count * len(solvent_template.atoms)
    expected_atoms = n_solute_atoms + n_solvent_atoms
    if len(packed_positions) != expected_atoms:
        msg = (
            f"Packmol output contains {len(packed_positions)} atoms, expected "
            f"{expected_atoms} ({n_solute_atoms} solute + {n_solvent_atoms} solvent)"
        )
        raise RuntimeError(msg)

    solvent_start = n_solute_atoms
    solvent_stop = solvent_start + n_solvent_atoms
    atoms_per_solvent = len(solvent_template.atoms)
    solvent_positions = tuple(
        packed_positions[index : index + atoms_per_solvent]
        for index in range(solvent_start, solvent_stop, atoms_per_solvent)
    )
    return PackmolSolutionPositions(solvent_positions_nm=solvent_positions)


def build_packmol_input(
    *,
    solute_path: Path,
    solvent_path: Path,
    output_path: Path,
    solvent_count: int,
    box_dimensions_nm: Vector3,
) -> str:
    """Build Packmol input text for the smoke solvent placement."""

    box_angstrom = tuple(length * 10.0 for length in box_dimensions_nm)

    lines = [
        f"tolerance {PACKMOL_TOLERANCE_ANGSTROM:.3f}",
        "filetype pdb",
        f"output {output_path.name}",
        "movebadrandom",
        f"nloop {PACKMOL_NLOOP}",
        "",
        f"structure {solute_path.name}",
        "  number 1",
        "  fixed 0. 0. 0. 0. 0. 0.",
        "end structure",
        "",
        f"structure {solvent_path.name}",
        f"  number {solvent_count}",
        f"  inside box 0. 0. 0. {box_angstrom[0]:.6f} {box_angstrom[1]:.6f} {box_angstrom[2]:.6f}",
        "end structure",
        "",
    ]
    return "\n".join(lines)


def run_packmol(input_path: Path, working_dir: Path, stdout_path: Path) -> None:
    """Execute Packmol and keep stdout for debugging."""

    packmol = shutil.which("packmol")
    if packmol is None:
        msg = "Packmol executable not found; run through `pixi run -e science real-system-smoke`."
        raise RuntimeError(msg)
    with input_path.open("r", encoding="utf-8") as handle:
        result = subprocess.run(
            [packmol],
            stdin=handle,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=working_dir,
            check=False,
        )
    stdout = result.stdout.decode("utf-8", errors="replace")
    stdout_path.write_text(stdout, encoding="utf-8")
    if result.returncode != 0 or "Success!" not in stdout:
        msg = f"Packmol failed; see {stdout_path}"
        raise RuntimeError(msg)


def write_topology_pdb(path: Path, topology: Any, positions_nm: tuple[Vector3, ...]) -> None:
    """Write a minimal PDB for Packmol fixed-solute coordinates."""

    records = []
    for serial, (atom, position) in enumerate(zip(topology.atoms(), positions_nm, strict=True), 1):
        residue = atom.residue
        records.append(
            pdb_atom_line(
                serial=serial,
                atom_name=atom.name,
                residue_name=residue.name,
                chain_id=residue.chain.id,
                residue_id=int(residue.id or 1),
                position_nm=position,
                element=atom.element.symbol,
            )
        )
    path.write_text("\n".join([*records, "END", ""]), encoding="utf-8")


def write_molecule_template_pdb(
    path: Path,
    template: MoleculeTemplate,
    *,
    residue_name: str,
) -> None:
    """Write one molecule template PDB for Packmol."""

    records = [
        pdb_atom_line(
            serial=index,
            atom_name=atom.name,
            residue_name=residue_name,
            chain_id="A",
            residue_id=1,
            position_nm=position,
            element=atom.element,
        )
        for index, (atom, position) in enumerate(
            zip(template.atoms, template.positions_nm, strict=True),
            1,
        )
    ]
    path.write_text("\n".join([*records, "END", ""]), encoding="utf-8")


def pdb_atom_line(
    *,
    serial: int,
    atom_name: str,
    residue_name: str,
    chain_id: str,
    residue_id: int,
    position_nm: Vector3,
    element: str,
) -> str:
    """Format one simple HETATM line for Packmol input."""

    x_angstrom, y_angstrom, z_angstrom = (coordinate * 10.0 for coordinate in position_nm)
    return (
        f"HETATM{serial:5d} {atom_name[:4]:<4s} {residue_name[:3]:>3s} {chain_id[:1]:1s}"
        f"{residue_id:4d}    {x_angstrom:8.3f}{y_angstrom:8.3f}{z_angstrom:8.3f}"
        f"  1.00  0.00          {element[:2]:>2s}"
    )


def read_pdb_positions_nm(path: Path) -> tuple[Vector3, ...]:
    """Read HETATM/ATOM coordinates from a PDB file as nanometer tuples."""

    positions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        positions.append(
            (
                float(line[30:38]) * 0.1,
                float(line[38:46]) * 0.1,
                float(line[46:54]) * 0.1,
            )
        )
    return tuple(positions)


def add_pd_slab(
    modules: Any,
    topology: Any,
    system: Any,
    nonbonded: Any,
    atom_handles: list[Any],
    chain_cache: dict[str, Any],
    positions_nm: list[Vector3],
    pd_indices: list[int],
    plan: Any,
    shift_nm: Vector3,
    residue_identities: tuple[ResidueIdentity, ...],
) -> None:
    """Add Pd atoms with CHARMM-INTERFACE LJ parameters."""

    unit = modules.unit
    pd_element = element_by_symbol(modules, "Pd")
    pd_parameters = get_fcc_metal_parameters("Pd")
    sigma_nm = pd_parameters.sigma_angstrom * 0.1
    epsilon_kj_mol = pd_parameters.openff_epsilon_kcal_mol * KCAL_TO_KJ
    for position, residue_identity in zip(plan.slab.positions_nm, residue_identities, strict=True):
        chain = get_or_add_chain(topology, chain_cache, residue_identity.chain_id)
        residue = topology.addResidue(
            residue_identity.residue_name,
            chain,
            id=str(residue_identity.residue_id),
        )
        atom = topology.addAtom("Pd", pd_element, residue)
        atom_handles.append(atom)
        system.addParticle(pd_element.mass)
        nonbonded.addParticle(
            0.0 * unit.elementary_charge,
            sigma_nm * unit.nanometer,
            epsilon_kj_mol * unit.kilojoule_per_mole,
        )
        pd_indices.append(len(positions_nm))
        positions_nm.append(add_vectors(position, shift_nm))


def add_sam_layer(
    modules: Any,
    topology: Any,
    system: Any,
    nonbonded: Any,
    bond_force: Any,
    angle_force: Any,
    torsion_force: Any,
    atom_handles: list[Any],
    chain_cache: dict[str, Any],
    positions_nm: list[Vector3],
    all_bonds: list[tuple[int, int]],
    sulfur_indices: list[int],
    sulfur_references: list[Vector3],
    anchor_pairs: list[tuple[int, int]],
    plan: Any,
    template: MoleculeTemplate,
    shift_nm: Vector3,
    sulfur_height_nm: float,
    residue_identities: tuple[ResidueIdentity, ...],
) -> None:
    """Add all planned propanethiol SAM molecules."""

    sulfur_index = sulfur_atom_index(template)
    axis_index = terminal_heavy_axis_index(template, sulfur_index)
    placed_sam_atoms: list[tuple[Vector3, str, str]] = []
    for placement_index, (placement, residue_identity) in enumerate(
        zip(
            plan.sam_placements.placements,
            residue_identities,
            strict=True,
        )
    ):
        target_sulfur = add_vectors(
            add_vectors(placement.position_nm, scale_vector(placement.normal, sulfur_height_nm)),
            shift_nm,
        )
        transformed = orient_template_by_anchor(
            template,
            anchor_index=sulfur_index,
            axis_index=axis_index,
            target_anchor_nm=target_sulfur,
            target_direction=placement.normal,
            azimuth_rad=select_sam_azimuth_rad(
                template,
                target_sulfur,
                placement.normal,
                placement.side,
                placed_sam_atoms,
                placement_index,
            ),
        )
        placed_sam_atoms.extend(
            (position, atom.element, placement.side)
            for position, atom in zip(transformed, template.atoms, strict=True)
        )
        global_indices = add_template_molecule(
            modules,
            topology,
            system,
            nonbonded,
            bond_force,
            angle_force,
            torsion_force,
            atom_handles,
            chain_cache,
            positions_nm,
            all_bonds,
            template,
            transformed,
            residue_identity,
        )
        global_sulfur = global_indices[sulfur_index]
        sulfur_indices.append(global_sulfur)
        sulfur_references.append(target_sulfur)
        nearest_metals = placement.anchor_metadata["nearest_metal_atom_indices"]
        anchor_pairs.extend((global_sulfur, int(metal_index)) for metal_index in nearest_metals)


def add_reactants(
    modules: Any,
    topology: Any,
    system: Any,
    nonbonded: Any,
    bond_force: Any,
    angle_force: Any,
    torsion_force: Any,
    atom_handles: list[Any],
    chain_cache: dict[str, Any],
    positions_nm: list[Vector3],
    all_bonds: list[tuple[int, int]],
    template: MoleculeTemplate,
    molecule_positions_nm: tuple[tuple[Vector3, ...], ...],
    residue_identities: tuple[ResidueIdentity, ...],
) -> None:
    """Add Packmol-placed cinnamaldehyde molecule(s)."""

    for transformed, residue_identity in zip(
        molecule_positions_nm,
        residue_identities,
        strict=True,
    ):
        add_template_molecule(
            modules,
            topology,
            system,
            nonbonded,
            bond_force,
            angle_force,
            torsion_force,
            atom_handles,
            chain_cache,
            positions_nm,
            all_bonds,
            template,
            transformed,
            residue_identity,
        )


def place_reactants_above_surface(
    plan: Any,
    template: MoleculeTemplate,
    reactant_count: int,
    shift_nm: Vector3,
    box_dimensions_nm: Vector3,
) -> tuple[tuple[Vector3, ...], ...]:
    """Place reactant molecule(s) in the upper solvent before Packmol solvent packing."""

    top_z = plan.slab.top_z_nm + SAM_TAIL_CLEARANCE_NM + 0.75
    centers = solvent_centers(reactant_count, plan.slab.lateral_size_nm, top_z)
    return tuple(
        center_template(template, clamp_to_box(add_vectors(center, shift_nm), box_dimensions_nm))
        for center in centers
    )


def solvent_centers(
    count: int,
    lateral_size_nm: tuple[float, float],
    z_nm: float,
) -> tuple[Vector3, ...]:
    """Return deterministic reactant centers in the upper solvent region."""

    centers = []
    for index in range(count):
        offset = (index - (count - 1) / 2.0) * 0.35
        x = max(-0.35 * lateral_size_nm[0], min(0.35 * lateral_size_nm[0], offset))
        y = 0.20 * lateral_size_nm[1] * ((-1) ** index)
        centers.append((x, y, z_nm + 0.20 * index))
    return tuple(centers)


def add_solvent_molecules(
    modules: Any,
    topology: Any,
    system: Any,
    nonbonded: Any,
    bond_force: Any,
    angle_force: Any,
    torsion_force: Any,
    atom_handles: list[Any],
    chain_cache: dict[str, Any],
    positions_nm: list[Vector3],
    all_bonds: list[tuple[int, int]],
    template: MoleculeTemplate,
    molecule_positions_nm: tuple[tuple[Vector3, ...], ...],
    residue_identities: tuple[ResidueIdentity, ...],
) -> int:
    """Add Packmol-placed OpenFF solvent molecules."""

    for solvent_positions, residue_identity in zip(
        molecule_positions_nm,
        residue_identities,
        strict=True,
    ):
        add_template_molecule(
            modules,
            topology,
            system,
            nonbonded,
            bond_force,
            angle_force,
            torsion_force,
            atom_handles,
            chain_cache,
            positions_nm,
            all_bonds,
            template,
            solvent_positions,
            residue_identity,
        )
    return len(molecule_positions_nm)


def add_template_molecule(
    modules: Any,
    topology: Any,
    system: Any,
    nonbonded: Any,
    bond_force: Any,
    angle_force: Any,
    torsion_force: Any,
    atom_handles: list[Any],
    chain_cache: dict[str, Any],
    positions_nm: list[Vector3],
    all_bonds: list[tuple[int, int]],
    template: MoleculeTemplate,
    transformed_positions_nm: tuple[Vector3, ...],
    residue_identity: ResidueIdentity,
) -> tuple[int, ...]:
    """Add one RDKit-derived molecule template to topology and system."""

    unit = modules.unit
    chain = get_or_add_chain(topology, chain_cache, residue_identity.chain_id)
    residue = topology.addResidue(
        residue_identity.residue_name,
        chain,
        id=str(residue_identity.residue_id),
    )
    global_indices = []
    local_atoms = []
    for atom, position in zip(template.atoms, transformed_positions_nm, strict=True):
        element = element_by_symbol(modules, atom.element)
        atom_handle = topology.addAtom(atom.name, element, residue)
        atom_handles.append(atom_handle)
        local_atoms.append(atom_handle)
        global_indices.append(len(positions_nm))
        system.addParticle(element.mass)
        nonbonded.addParticle(
            atom.charge_e * unit.elementary_charge,
            atom.sigma_nm * unit.nanometer,
            atom.epsilon_kj_mol * unit.kilojoule_per_mole,
        )
        positions_nm.append(position)

    for exception in template.exception_parameters:
        nonbonded.addException(
            global_indices[exception.atom1],
            global_indices[exception.atom2],
            exception.chargeprod_e2 * unit.elementary_charge**2,
            exception.sigma_nm * unit.nanometer,
            exception.epsilon_kj_mol * unit.kilojoule_per_mole,
        )
    for local_i, local_j in template.bonds:
        global_i = global_indices[local_i]
        global_j = global_indices[local_j]
        topology.addBond(local_atoms[local_i], local_atoms[local_j])
        all_bonds.append((global_i, global_j))
    for constraint in template.constraints:
        system.addConstraint(
            global_indices[constraint.atom1],
            global_indices[constraint.atom2],
            constraint.distance_nm * unit.nanometer,
        )
    for bond in template.bond_parameters:
        bond_force.addBond(
            global_indices[bond.atom1],
            global_indices[bond.atom2],
            bond.length_nm * unit.nanometer,
            bond.k_kj_mol_nm2 * unit.kilojoule_per_mole / unit.nanometer**2,
        )
    for angle in template.angle_parameters:
        angle_force.addAngle(
            global_indices[angle.atom1],
            global_indices[angle.atom2],
            global_indices[angle.atom3],
            angle.angle_rad * unit.radian,
            angle.k_kj_mol_rad2 * unit.kilojoule_per_mole / unit.radian**2,
        )
    for torsion in template.torsion_parameters:
        torsion_force.addTorsion(
            global_indices[torsion.atom1],
            global_indices[torsion.atom2],
            global_indices[torsion.atom3],
            global_indices[torsion.atom4],
            torsion.periodicity,
            torsion.phase_rad * unit.radian,
            torsion.k_kj_mol * unit.kilojoule_per_mole,
        )
    return tuple(global_indices)


def element_by_symbol(modules: Any, symbol: str) -> Any:
    """Return an OpenMM app Element by symbol."""

    return modules.app.Element.getBySymbol(symbol)


def sulfur_atom_index(template: MoleculeTemplate) -> int:
    """Return the sulfur atom index for a SAM template."""

    matches = [index for index, atom in enumerate(template.atoms) if atom.element == "S"]
    if len(matches) != 1:
        raise ValueError("SAM template must contain exactly one sulfur atom")
    return matches[0]


def terminal_heavy_axis_index(template: MoleculeTemplate, anchor_index: int) -> int:
    """Return the heavy atom farthest from the SAM anchor for molecular-axis alignment."""

    heavy_indices = [
        index
        for index, atom in enumerate(template.atoms)
        if index != anchor_index and atom.element != "H"
    ]
    if not heavy_indices:
        raise ValueError("SAM template must contain a heavy atom beyond the anchor")
    return max(
        heavy_indices,
        key=lambda index: distance(
            template.positions_nm[anchor_index], template.positions_nm[index]
        ),
    )


def sam_azimuth_rad(placement_index: int) -> float:
    """Return a deterministic per-SAM rotation around the sulfur-anchor axis."""

    golden_angle_rad = math.pi * (3.0 - math.sqrt(5.0))
    return placement_index * golden_angle_rad


def select_sam_azimuth_rad(
    template: MoleculeTemplate,
    target_anchor_nm: Vector3,
    target_direction: Vector3,
    side: str,
    placed_sam_atoms: list[tuple[Vector3, str, str]],
    placement_index: int,
) -> float:
    """Choose an around-axis SAM rotation that avoids prior same-side H clashes."""

    same_side_atoms = [
        (position, element)
        for position, element, placed_side in placed_sam_atoms
        if placed_side == side
    ]
    if not same_side_atoms:
        return sam_azimuth_rad(placement_index)

    candidate_count = 24
    candidates = [
        sam_azimuth_rad(placement_index) + 2.0 * math.pi * index / candidate_count
        for index in range(candidate_count)
    ]
    return max(
        candidates,
        key=lambda angle: score_sam_azimuth(
            template,
            target_anchor_nm,
            target_direction,
            angle,
            same_side_atoms,
        ),
    )


def score_sam_azimuth(
    template: MoleculeTemplate,
    target_anchor_nm: Vector3,
    target_direction: Vector3,
    azimuth_rad: float,
    same_side_atoms: list[tuple[Vector3, str]],
) -> tuple[float, float]:
    """Score an azimuth by its closest H-involving inter-SAM contact."""

    candidate_atoms = [
        (
            add_vectors(
                target_anchor_nm,
                rotate_about_axis(position, target_direction, azimuth_rad),
            ),
            atom.element,
        )
        for position, atom in zip(
            anchor_relative_positions(template, target_direction), template.atoms, strict=True
        )
    ]
    closest_h_contact = math.inf
    closest_any_contact = math.inf
    for candidate_position, candidate_element in candidate_atoms:
        for placed_position, placed_element in same_side_atoms:
            contact = distance(candidate_position, placed_position)
            closest_any_contact = min(closest_any_contact, contact)
            if "H" in (candidate_element, placed_element):
                closest_h_contact = min(closest_h_contact, contact)
    return closest_h_contact, closest_any_contact


def anchor_relative_positions(
    template: MoleculeTemplate,
    target_direction: Vector3,
) -> tuple[Vector3, ...]:
    """Return template coordinates relative to its sulfur anchor."""

    sulfur_index = sulfur_atom_index(template)
    sulfur_position = template.positions_nm[sulfur_index]
    axis_index = terminal_heavy_axis_index(template, sulfur_index)
    source_vector = subtract_vectors(template.positions_nm[axis_index], sulfur_position)
    rotation = rotation_matrix(source_vector, target_direction)
    return tuple(
        matvec(rotation, subtract_vectors(position, sulfur_position))
        for position in template.positions_nm
    )


def orient_template_by_anchor(
    template: MoleculeTemplate,
    *,
    anchor_index: int,
    axis_index: int,
    target_anchor_nm: Vector3,
    target_direction: Vector3,
    azimuth_rad: float = 0.0,
) -> tuple[Vector3, ...]:
    """Rotate and translate a template so its anchor-to-axis vector points outward."""

    anchor = template.positions_nm[anchor_index]
    source_vector = subtract_vectors(template.positions_nm[axis_index], anchor)
    rotation = rotation_matrix(source_vector, target_direction)
    return tuple(
        add_vectors(
            target_anchor_nm,
            rotate_about_axis(
                matvec(rotation, subtract_vectors(position, anchor)),
                target_direction,
                azimuth_rad,
            ),
        )
        for position in template.positions_nm
    )


def center_template(template: MoleculeTemplate, center_nm: Vector3) -> tuple[Vector3, ...]:
    """Translate a template so its coordinate centroid is at ``center_nm``."""

    current_center = centroid(template.positions_nm)
    return tuple(
        add_vectors(center_nm, subtract_vectors(position, current_center))
        for position in template.positions_nm
    )


def create_simulation_with_platform_fallback(
    modules: Any,
    smoke_build: SmokeBuild,
    *,
    platform_name: str,
    temperature_k: float,
    timestep_fs: float,
    friction_per_ps: float,
) -> PlatformSelection:
    """Create an OpenMM Simulation, falling back from CUDA to CPU in auto mode."""

    openmm = modules.openmm
    app = modules.app
    unit = modules.unit
    available = available_platform_names(openmm)
    candidates = auto_platform_candidates(available) if platform_name == "auto" else [platform_name]
    errors = []
    for candidate in candidates:
        if candidate not in available:
            errors.append(f"{candidate}: not available")
            continue
        try:
            integrator = create_langevin_integrator(
                temperature_k,
                friction_per_ps,
                timestep_fs,
                openmm_module=openmm,
                unit_module=unit,
            )
            platform = openmm.Platform.getPlatformByName(candidate)
            simulation = app.Simulation(
                smoke_build.topology,
                smoke_build.system,
                integrator,
                platform,
            )
            simulation.context.setPositions(smoke_build.positions_quantity)
            simulation.context.getState(getEnergy=True)
        except Exception as error:
            errors.append(f"{candidate}: {error}")
            continue
        return PlatformSelection(
            simulation=simulation,
            platform_name=candidate,
            errors=tuple(errors),
        )
    formatted_errors = "\n".join(errors) or "no candidate platforms"
    raise RuntimeError(f"could not create an OpenMM context:\n{formatted_errors}")


def available_platform_names(openmm: Any) -> tuple[str, ...]:
    """Return installed OpenMM platform names."""

    return tuple(
        openmm.Platform.getPlatform(index).getName()
        for index in range(openmm.Platform.getNumPlatforms())
    )


def auto_platform_candidates(available: tuple[str, ...]) -> list[str]:
    """Prefer CUDA for the real smoke, but keep CPU fallback usable."""

    preferred = ["CUDA", "OpenCL", "CPU", "Reference"]
    return [platform for platform in preferred if platform in available]


def read_energy(modules: Any, simulation: Any, *, include_kinetic: bool) -> EnergyRecord:
    """Extract potential energy and optional temperature from a Simulation."""

    unit = modules.unit
    state = simulation.context.getState(getEnergy=True)
    potential = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    if not include_kinetic:
        return EnergyRecord(float(potential), None, None)
    kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    mobile_particles = tuple(
        index
        for index in range(simulation.system.getNumParticles())
        if simulation.system.getParticleMass(index).value_in_unit(unit.dalton) > 0.0
    )
    mobile_particle_set = set(mobile_particles)
    mobile_constraints = sum(
        1
        for index in range(simulation.system.getNumConstraints())
        if simulation.system.getConstraintParameters(index)[0] in mobile_particle_set
        and simulation.system.getConstraintParameters(index)[1] in mobile_particle_set
    )
    degrees_of_freedom = max(1, 3 * len(mobile_particles) - mobile_constraints - 3)
    gas_constant = 0.00831446261815324
    temperature = 2.0 * kinetic / (degrees_of_freedom * gas_constant)
    return EnergyRecord(float(potential), float(kinetic), float(temperature))


def positions_to_nm(modules: Any, positions: Any) -> tuple[Vector3, ...]:
    """Convert OpenMM positions to plain nanometer tuples."""

    unit = modules.unit
    return tuple(tuple(vector) for vector in positions.value_in_unit(unit.nanometer))


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


def clamp_to_box(position: Vector3, box_dimensions_nm: Vector3) -> Vector3:
    """Clamp a coordinate just inside the orthorhombic box."""

    margin = 0.05
    return tuple(
        max(margin, min(length - margin, coordinate))
        for coordinate, length in zip(position, box_dimensions_nm, strict=True)
    )


if __name__ == "__main__":
    sys.exit(main())
