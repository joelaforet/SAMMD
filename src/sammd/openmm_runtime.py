"""Optional OpenMM runtime helpers for SAMMD simulations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from sammd.reporting import create_openmm_reporters, prepare_reporter_output_directories

DEFAULT_PD_RESTRAINT_K_KJ_MOL_NM2 = 10000.0
POSITION_RESTRAINT_EXPRESSION = "0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
LJ_SCALING_EXPRESSION = "4*epsilon_delta*((sigma/r)^12-(sigma/r)^6)"


@dataclass(frozen=True)
class OpenMMModules:
    """Lazy OpenMM module bundle used by runtime helpers."""

    openmm: Any
    app: Any
    unit: Any


@dataclass(frozen=True)
class LangevinIntegratorConfig:
    """User-facing Langevin integrator values."""

    temperature_k: float = 300.0
    friction_per_ps: float = 1.0
    timestep_fs: float = 2.0


@dataclass(frozen=True)
class PlatformSelection:
    """OpenMM simulation selected from platform fallback candidates."""

    simulation: Any
    platform_name: str
    errors: tuple[str, ...]


@dataclass(frozen=True)
class EnergyRecord:
    """Potential, kinetic, and temperature values read from an OpenMM context."""

    potential_energy_kj_mol: float
    kinetic_energy_kj_mol: float | None
    temperature_k: float | None


@dataclass(frozen=True)
class AnchorScalingMetadata:
    """Summary of sulfur-metal nonbonded proxy force construction."""

    pairs_requested: int
    pairs_added: int
    force_added: bool
    scale_factor: float
    force_index: int | None = None
    sigma_nm: tuple[float, ...] = ()
    epsilon_delta_kj_mol: tuple[float, ...] = ()


def require_openmm() -> OpenMMModules:
    """Import OpenMM lazily and raise a clear setup error when unavailable.

    Returns
    -------
    OpenMMModules
        OpenMM, OpenMM app, and unit modules.
    """

    try:
        import openmm
        from openmm import app, unit
    except ImportError as error:
        msg = (
            "OpenMM is required for SAMMD runtime simulation helpers. Install and run from "
            "the SAMMD science/pixi environment before creating integrators, simulations, "
            "or runtime forces."
        )
        raise ImportError(msg) from error
    return OpenMMModules(openmm=openmm, app=app, unit=unit)


def create_langevin_integrator(
    temperature_k: float,
    friction_per_ps: float,
    timestep_fs: float,
    *,
    openmm_module: Any | None = None,
    unit_module: Any | None = None,
) -> Any:
    """Create an OpenMM Langevin integrator from simple user-facing values.

    Parameters
    ----------
    temperature_k
        Temperature in kelvin.
    friction_per_ps
        Collision rate in inverse picoseconds.
    timestep_fs
        Integration timestep in femtoseconds.
    openmm_module
        Optional injected OpenMM-like module for tests.
    unit_module
        Optional injected OpenMM unit-like module for tests.

    Returns
    -------
    Any
        OpenMM ``LangevinIntegrator`` instance.
    """

    _validate_positive_finite(temperature_k, "temperature_k")
    _validate_positive_finite(friction_per_ps, "friction_per_ps")
    _validate_positive_finite(timestep_fs, "timestep_fs")
    modules = None if openmm_module is not None and unit_module is not None else require_openmm()
    openmm = openmm_module if openmm_module is not None else modules.openmm
    unit = unit_module if unit_module is not None else modules.unit
    return openmm.LangevinIntegrator(
        temperature_k * unit.kelvin,
        friction_per_ps / unit.picosecond,
        timestep_fs * unit.femtoseconds,
    )


def create_openmm_simulation(
    topology: Any,
    system: Any,
    positions: Any,
    reporting_config: Any,
    output_paths: Any,
    *,
    integrator: Any | None = None,
    integrator_config: LangevinIntegratorConfig | None = None,
    platform_name: str | None = None,
    total_steps: int | None = None,
    openmm_module: Any | None = None,
    app_module: Any | None = None,
    unit_module: Any | None = None,
    prepare_reporter_directories: bool = False,
) -> Any:
    """Create and configure an OpenMM ``Simulation`` from backend artifacts.

    Parameters
    ----------
    topology
        Existing OpenMM topology.
    system
        Existing OpenMM system.
    positions
        Existing OpenMM-compatible positions.
    reporting_config
        SAMMD reporter configuration.
    output_paths
        SAMMD resolved output paths.
    integrator
        Optional prebuilt OpenMM integrator.
    integrator_config
        Optional simple integrator values used when ``integrator`` is omitted.
    platform_name
        Optional OpenMM platform name.
    total_steps
        Total expected simulation steps for progress reporters.
    openmm_module, app_module, unit_module
        Optional injected OpenMM-like modules for tests.
    prepare_reporter_directories
        Whether to create reporter output directories before attaching reporters.

    Returns
    -------
    Any
        Configured OpenMM ``Simulation`` with positions and reporters attached.
    """

    modules = None if openmm_module is not None and app_module is not None else require_openmm()
    openmm = openmm_module if openmm_module is not None else modules.openmm
    app = app_module if app_module is not None else modules.app
    unit = unit_module if unit_module is not None else (
        modules.unit if modules is not None else None
    )

    if integrator is None:
        config = integrator_config or LangevinIntegratorConfig()
        integrator = create_langevin_integrator(
            config.temperature_k,
            config.friction_per_ps,
            config.timestep_fs,
            openmm_module=openmm,
            unit_module=unit,
        )
    if platform_name is None:
        simulation = app.Simulation(topology, system, integrator)
    else:
        platform = openmm.Platform.getPlatformByName(platform_name)
        simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    if prepare_reporter_directories:
        prepare_reporter_output_directories(output_paths)
    simulation.reporters.extend(
        create_openmm_reporters(
            reporting_config,
            output_paths,
            total_steps=total_steps,
            app_module=app,
        )
    )
    return simulation


def create_simulation_with_platform_fallback(
    topology: Any,
    system: Any,
    positions: Any,
    *,
    platform_name: str,
    temperature_k: float,
    timestep_fs: float,
    friction_per_ps: float,
    openmm_module: Any | None = None,
    app_module: Any | None = None,
    unit_module: Any | None = None,
) -> PlatformSelection:
    """Create an OpenMM Simulation, falling back from accelerated platforms in auto mode."""

    modules = None if openmm_module is not None and app_module is not None else require_openmm()
    openmm = openmm_module if openmm_module is not None else modules.openmm
    app = app_module if app_module is not None else modules.app
    unit = unit_module if unit_module is not None else (
        modules.unit if modules is not None else None
    )
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
            simulation = app.Simulation(topology, system, integrator, platform)
            simulation.context.setPositions(positions)
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
    msg = f"could not create an OpenMM context:\n{formatted_errors}"
    raise RuntimeError(msg)


def available_platform_names(openmm_module: Any | None = None) -> tuple[str, ...]:
    """Return installed OpenMM platform names."""

    openmm = openmm_module if openmm_module is not None else require_openmm().openmm
    return tuple(
        openmm.Platform.getPlatform(index).getName()
        for index in range(openmm.Platform.getNumPlatforms())
    )


def auto_platform_candidates(available: tuple[str, ...]) -> list[str]:
    """Prefer accelerated OpenMM platforms while keeping CPU fallbacks usable."""

    preferred = ["CUDA", "OpenCL", "CPU", "Reference"]
    return [platform for platform in preferred if platform in available]


def read_energy(
    simulation: Any,
    *,
    include_kinetic: bool,
    unit_module: Any | None = None,
) -> EnergyRecord:
    """Extract potential energy and optional instantaneous temperature from a Simulation."""

    unit = unit_module if unit_module is not None else require_openmm().unit
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


def positions_to_nm(
    positions: Any,
    *,
    unit_module: Any | None = None,
) -> tuple[tuple[float, float, float], ...]:
    """Convert OpenMM positions to plain nanometer tuples."""

    unit = unit_module if unit_module is not None else require_openmm().unit
    return tuple(tuple(vector) for vector in positions.value_in_unit(unit.nanometer))


def add_position_restraints(
    system: Any,
    atom_indices: list[int] | tuple[int, ...],
    positions_nm: Any,
    *,
    k_kj_mol_nm2: float = DEFAULT_PD_RESTRAINT_K_KJ_MOL_NM2,
    openmm_module: Any | None = None,
) -> Any:
    """Add harmonic positional restraints to selected atoms.

    The force uses ``0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)`` with coordinates in
    nanometers and ``k`` in kJ mol^-1 nm^-2.
    """

    _validate_positive_finite(k_kj_mol_nm2, "k_kj_mol_nm2")
    indices = _validate_atom_indices(system, atom_indices)
    reference_positions = _validate_positions_nm(positions_nm, expected_count=len(indices))
    openmm = openmm_module if openmm_module is not None else require_openmm().openmm

    force = openmm.CustomExternalForce(POSITION_RESTRAINT_EXPRESSION)
    force.addGlobalParameter("k", k_kj_mol_nm2)
    for parameter_name in ("x0", "y0", "z0"):
        force.addPerParticleParameter(parameter_name)
    for atom_index, position in zip(indices, reference_positions, strict=True):
        force.addParticle(atom_index, list(position))
    system.addForce(force)
    return force


def add_sulfur_metal_lj_scaling(
    system: Any,
    pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    scale_factor: float,
    openmm_module: Any | None = None,
    unit_module: Any | None = None,
) -> AnchorScalingMetadata:
    """Add a pairwise LJ correction for explicit sulfur-metal anchor pairs.

    The correction is ``(scale_factor - 1)`` times the existing Lorentz-Berthelot
    12-6 LJ pair energy for each supplied pair.
    """

    _validate_positive_finite(scale_factor, "scale_factor")
    validated_pairs = _validate_pairs(system, pairs)
    nonbonded_force = _find_nonbonded_force(system)
    if nonbonded_force is None:
        msg = "system must contain an OpenMM NonbondedForce for sulfur-metal LJ scaling"
        raise ValueError(msg)
    _reject_pairs_with_nonbonded_exceptions(nonbonded_force, validated_pairs)
    if scale_factor == 1.0:
        return AnchorScalingMetadata(
            pairs_requested=len(validated_pairs),
            pairs_added=0,
            force_added=False,
            scale_factor=scale_factor,
        )

    modules = None if openmm_module is not None and unit_module is not None else require_openmm()
    openmm = openmm_module if openmm_module is not None else modules.openmm
    unit = unit_module if unit_module is not None else modules.unit

    correction_force = openmm.CustomBondForce(LJ_SCALING_EXPRESSION)
    correction_force.addPerBondParameter("sigma")
    correction_force.addPerBondParameter("epsilon_delta")
    if hasattr(correction_force, "setUsesPeriodicBoundaryConditions"):
        correction_force.setUsesPeriodicBoundaryConditions(True)

    sigma_values = []
    epsilon_delta_values = []
    for sulfur_index, metal_index in validated_pairs:
        _, sulfur_sigma, sulfur_epsilon = nonbonded_force.getParticleParameters(sulfur_index)
        _, metal_sigma, metal_epsilon = nonbonded_force.getParticleParameters(metal_index)
        sulfur_sigma_nm = _quantity_to_float(sulfur_sigma, unit.nanometer)
        metal_sigma_nm = _quantity_to_float(metal_sigma, unit.nanometer)
        sigma_nm = (sulfur_sigma_nm + metal_sigma_nm) / 2.0
        epsilon_kj_mol = math.sqrt(
            _quantity_to_float(sulfur_epsilon, unit.kilojoule_per_mole)
            * _quantity_to_float(metal_epsilon, unit.kilojoule_per_mole)
        )
        epsilon_delta = (scale_factor - 1.0) * epsilon_kj_mol
        correction_force.addBond(sulfur_index, metal_index, [sigma_nm, epsilon_delta])
        sigma_values.append(sigma_nm)
        epsilon_delta_values.append(epsilon_delta)

    force_index = system.addForce(correction_force)
    return AnchorScalingMetadata(
        pairs_requested=len(validated_pairs),
        pairs_added=len(validated_pairs),
        force_added=True,
        scale_factor=scale_factor,
        force_index=force_index,
        sigma_nm=tuple(sigma_values),
        epsilon_delta_kj_mol=tuple(epsilon_delta_values),
    )


def _validate_positive_finite(value: float, name: str) -> None:
    """Validate a positive finite scalar value."""

    if not math.isfinite(value) or value <= 0:
        msg = f"{name} must be positive and finite"
        raise ValueError(msg)


def _validate_atom_indices(
    system: Any,
    atom_indices: list[int] | tuple[int, ...],
    *,
    allow_duplicates: bool = False,
) -> tuple[int, ...]:
    """Validate atom indices against system size when available."""

    indices = tuple(atom_indices)
    if not indices:
        msg = "atom_indices must contain at least one atom"
        raise ValueError(msg)
    particle_count = system.getNumParticles() if hasattr(system, "getNumParticles") else None
    for atom_index in indices:
        if isinstance(atom_index, bool) or not isinstance(atom_index, int) or atom_index < 0:
            msg = "atom indices must be non-negative integers"
            raise ValueError(msg)
        if particle_count is not None and atom_index >= particle_count:
            msg = f"atom index {atom_index} is outside system particle count {particle_count}"
            raise ValueError(msg)
    if not allow_duplicates and len(set(indices)) != len(indices):
        msg = "atom_indices must not contain duplicates"
        raise ValueError(msg)
    return indices


def _validate_pairs(
    system: Any,
    pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    """Validate explicit sulfur-metal pair indices."""

    try:
        pair_items = tuple(pairs)
    except TypeError as error:
        msg = "pairs must be an iterable of sulfur-metal index pairs"
        raise ValueError(msg) from error
    if not pair_items:
        msg = "pairs must contain at least one sulfur-metal pair"
        raise ValueError(msg)

    validated_pairs = []
    seen_pairs: set[frozenset[int]] = set()
    for pair in pair_items:
        try:
            pair_tuple = tuple(pair)
        except TypeError as error:
            msg = "each sulfur-metal pair must contain exactly two atom indices"
            raise ValueError(msg) from error
        if len(pair_tuple) != 2:
            msg = "each sulfur-metal pair must contain exactly two atom indices"
            raise ValueError(msg)
        sulfur_index, metal_index = pair_tuple
        for atom_index in (sulfur_index, metal_index):
            if isinstance(atom_index, bool) or not isinstance(atom_index, int):
                msg = "pair atom indices must be integers and not booleans"
                raise ValueError(msg)
        if sulfur_index == metal_index:
            msg = "sulfur-metal pairs must not contain self-pairs"
            raise ValueError(msg)
        normalized_pair = frozenset((sulfur_index, metal_index))
        if normalized_pair in seen_pairs:
            msg = "sulfur-metal pairs must not contain duplicate or reversed duplicate pairs"
            raise ValueError(msg)
        seen_pairs.add(normalized_pair)
        validated_pairs.append((sulfur_index, metal_index))

    validated_pairs_tuple = tuple(validated_pairs)
    flattened = [atom_index for pair in validated_pairs_tuple for atom_index in pair]
    _validate_atom_indices(system, tuple(flattened), allow_duplicates=True)
    return validated_pairs_tuple


def _validate_positions_nm(
    positions_nm: Any,
    *,
    expected_count: int,
) -> tuple[tuple[float, float, float], ...]:
    """Validate reference coordinates as finite nanometer triples."""

    positions = tuple(_as_coordinate_tuple(position) for position in positions_nm)
    if len(positions) != expected_count:
        msg = "positions_nm length must match atom_indices length"
        raise ValueError(msg)
    for position in positions:
        if len(position) != 3 or not all(math.isfinite(coordinate) for coordinate in position):
            msg = "positions_nm must contain finite xyz coordinate triples"
            raise ValueError(msg)
    return positions


def _as_coordinate_tuple(position: Any) -> tuple[float, float, float]:
    """Convert a coordinate-like object into a numeric xyz tuple."""

    if hasattr(position, "value_in_unit"):
        modules = require_openmm()
        position = position.value_in_unit(modules.unit.nanometer)
    values = tuple(position)
    return tuple(_quantity_to_float(value, None) for value in values)


def _find_nonbonded_force(system: Any) -> Any | None:
    """Return the first NonbondedForce-like object in a system."""

    for force_index in range(system.getNumForces()):
        force = system.getForce(force_index)
        if force.__class__.__name__ == "NonbondedForce":
            return force
    return None


def _reject_pairs_with_nonbonded_exceptions(
    nonbonded_force: Any,
    pairs: tuple[tuple[int, int], ...],
) -> None:
    """Reject LJ scaling pairs covered by existing NonbondedForce exceptions."""

    if not hasattr(nonbonded_force, "getNumExceptions"):
        return
    requested_pairs = {frozenset(pair) for pair in pairs}
    for exception_index in range(nonbonded_force.getNumExceptions()):
        exception_parameters = nonbonded_force.getExceptionParameters(exception_index)
        exception_pair = frozenset((int(exception_parameters[0]), int(exception_parameters[1])))
        if exception_pair in requested_pairs:
            msg = (
                "sulfur-metal LJ scaling does not support pairs with existing "
                "NonbondedForce exceptions or exclusions"
            )
            raise ValueError(msg)


def _quantity_to_float(value: Any, unit: Any | None) -> float:
    """Convert an OpenMM quantity or plain scalar to ``float``."""

    if unit is not None and hasattr(value, "value_in_unit"):
        return float(value.value_in_unit(unit))
    return float(value)
