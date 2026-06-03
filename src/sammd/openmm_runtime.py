"""Optional OpenMM runtime helpers for SAMMD simulations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from sammd.reporting import create_openmm_reporters

DEFAULT_PD_RESTRAINT_K_KJ_MOL_NM2 = 10000.0
POSITION_RESTRAINT_EXPRESSION = "0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
LJ_SCALING_EXPRESSION = "epsilon_delta*((sigma/r)^12-(sigma/r)^6)"


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
    simulation.reporters.extend(
        create_openmm_reporters(
            reporting_config,
            output_paths,
            total_steps=total_steps,
            app_module=app,
        )
    )
    return simulation


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
    nonbonded_force = _find_nonbonded_force(system)
    if nonbonded_force is None:
        msg = "system must contain an OpenMM NonbondedForce for sulfur-metal LJ scaling"
        raise ValueError(msg)

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
        epsilon_delta = (scale_factor - 1.0) * 4.0 * epsilon_kj_mol
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
) -> tuple[int, ...]:
    """Validate atom indices against system size when available."""

    indices = tuple(atom_indices)
    if not indices:
        msg = "atom_indices must contain at least one atom"
        raise ValueError(msg)
    particle_count = system.getNumParticles() if hasattr(system, "getNumParticles") else None
    for atom_index in indices:
        if not isinstance(atom_index, int) or atom_index < 0:
            msg = "atom indices must be non-negative integers"
            raise ValueError(msg)
        if particle_count is not None and atom_index >= particle_count:
            msg = f"atom index {atom_index} is outside system particle count {particle_count}"
            raise ValueError(msg)
    return indices


def _validate_pairs(
    system: Any,
    pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    """Validate explicit sulfur-metal pair indices."""

    validated_pairs = tuple((int(sulfur), int(metal)) for sulfur, metal in pairs)
    if not validated_pairs:
        msg = "pairs must contain at least one sulfur-metal pair"
        raise ValueError(msg)
    flattened = [atom_index for pair in validated_pairs for atom_index in pair]
    _validate_atom_indices(system, tuple(flattened))
    return validated_pairs


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


def _quantity_to_float(value: Any, unit: Any | None) -> float:
    """Convert an OpenMM quantity or plain scalar to ``float``."""

    if unit is not None and hasattr(value, "value_in_unit"):
        return float(value.value_in_unit(unit))
    return float(value)
