"""Lightweight solution composition planning for SAMMD systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import floor, isfinite
from typing import Any

from sammd.config import SAMMDConfig

AVOGADRO_CONSTANT_MOL_INV = 6.02214076e23
NM3_TO_L = 1.0e-24
L_TO_ML = 1000.0
VOLUME_FRACTION_TOLERANCE = 1.0e-12

DEFAULT_WATER_DENSITY_G_ML = 0.997
DEFAULT_WATER_MOLAR_MASS_G_MOL = 18.01528

KNOWN_SOLVENT_PROPERTIES: dict[str, dict[str, float]] = {
    "water": {
        "density_g_ml": DEFAULT_WATER_DENSITY_G_ML,
        "molar_mass_g_mol": DEFAULT_WATER_MOLAR_MASS_G_MOL,
    },
    "tip3p": {
        "density_g_ml": DEFAULT_WATER_DENSITY_G_ML,
        "molar_mass_g_mol": DEFAULT_WATER_MOLAR_MASS_G_MOL,
    },
    "ethanol": {
        "molar_mass_g_mol": 46.06844,
    },
}


@dataclass(frozen=True)
class SolventComponentSpec:
    """Simple solvent component input for composition planning."""

    name: str
    volume_fraction: float
    smiles: str | None = None
    density_g_ml: float | None = None
    molar_mass_g_mol: float | None = None


@dataclass(frozen=True)
class SaltSpec:
    """Simple neutral salt input for composition planning."""

    cation: str = "Na+"
    anion: str = "Cl-"
    concentration_molar: float = 0.0
    neutralize: bool = True


@dataclass(frozen=True)
class ReactantSpec:
    """Simple reactant input for composition planning."""

    name: str
    smiles: str
    concentration_molar: float


@dataclass(frozen=True)
class PlannedMolecule:
    """Planned count and source metadata for one molecular component."""

    name: str
    role: str
    count: int
    smiles: str | None = None
    volume_fraction: float | None = None
    concentration_molar: float | None = None
    density_g_ml: float | None = None
    molar_mass_g_mol: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlannedSalt:
    """Planned ion counts for one neutral salt."""

    cation: str
    anion: str
    cation_count: int
    anion_count: int
    concentration_molar: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SolutionPlan:
    """Deterministic solution composition plan for later packing and parameterization."""

    box_volume_nm3: float
    solvent_components: tuple[PlannedMolecule, ...]
    salts: tuple[PlannedSalt, ...]
    reactants: tuple[PlannedMolecule, ...]
    warnings: tuple[str, ...] = ()

    @property
    def molecule_counts(self) -> dict[str, int]:
        """Return aggregate non-ion molecule counts by component name."""

        counts: dict[str, int] = {}
        for component in (*self.solvent_components, *self.reactants):
            counts[component.name] = counts.get(component.name, 0) + component.count
        return counts


def round_half_up(value: float) -> int:
    """Round a non-negative molecule expectation with explicit half-up behavior."""

    if not isfinite(value) or value < 0:
        msg = "molecule count expectation must be a finite non-negative value"
        raise ValueError(msg)
    return floor(value + 0.5)


def plan_solution_composition(config: SAMMDConfig, box_volume_nm3: float) -> SolutionPlan:
    """Plan solvent, salt, and reactant counts from a validated SAMMD configuration.

    Parameters
    ----------
    config
        Validated SAMMD configuration.
    box_volume_nm3
        Simulation box volume in nm^3.

    Returns
    -------
    SolutionPlan
        Deterministic molecule and ion counts with metadata for later placement.
    """

    return plan_solution_components(
        box_volume_nm3=box_volume_nm3,
        solvent_components=config.solvent.components,
        salts=config.salts,
        reactants=config.reactants,
        water_model=config.solvent.water_model,
    )


def plan_solution_components(
    box_volume_nm3: float,
    solvent_components: list[Any] | tuple[Any, ...],
    salts: list[Any] | tuple[Any, ...] = (),
    reactants: list[Any] | tuple[Any, ...] = (),
    water_model: str = "TIP3P",
) -> SolutionPlan:
    """Plan solution counts from config-like or dataclass component inputs."""

    _validate_box_volume(box_volume_nm3)
    total_volume_fraction = sum(
        float(_get_required(component, "volume_fraction")) for component in solvent_components
    )
    if total_volume_fraction > 1.0 + VOLUME_FRACTION_TOLERANCE:
        msg = "solvent component volume fractions must not exceed 1.0"
        raise ValueError(msg)

    warnings: list[str] = []
    planned_solvents = tuple(
        _plan_solvent_component(component, box_volume_nm3, water_model)
        for component in solvent_components
    )
    planned_salts = tuple(_plan_salt(salt, box_volume_nm3) for salt in salts)
    planned_reactants = tuple(_plan_reactant(reactant, box_volume_nm3) for reactant in reactants)

    for component in (*planned_solvents, *planned_reactants):
        if component.count == 0:
            warnings.append(
                f"{component.role} component '{component.name}' rounded to zero molecules "
                "for this box"
            )
    for salt in planned_salts:
        if salt.cation_count == 0 and salt.anion_count == 0:
            warnings.append(
                f"salt '{salt.cation}/{salt.anion}' rounded to zero ion pairs for this box"
            )

    return SolutionPlan(
        box_volume_nm3=box_volume_nm3,
        solvent_components=planned_solvents,
        salts=planned_salts,
        reactants=planned_reactants,
        warnings=tuple(warnings),
    )


def _plan_solvent_component(
    component: Any, box_volume_nm3: float, water_model: str
) -> PlannedMolecule:
    """Plan one volume-fraction solvent component."""

    name = str(_get_required(component, "name"))
    volume_fraction = float(_get_required(component, "volume_fraction"))
    if volume_fraction < 0:
        msg = f"solvent component '{name}' volume_fraction must be non-negative"
        raise ValueError(msg)
    density_g_ml = _resolve_float(component, "density_g_ml", name, "density_g_ml")
    molar_mass_g_mol = _resolve_float(component, "molar_mass_g_mol", name, "molar_mass_g_mol")
    is_water = name.lower() == "water"
    if is_water:
        density_g_ml = density_g_ml or DEFAULT_WATER_DENSITY_G_ML
        molar_mass_g_mol = molar_mass_g_mol or DEFAULT_WATER_MOLAR_MASS_G_MOL
    if density_g_ml is None:
        msg = f"solvent component '{name}' requires density_g_ml for volume-fraction planning"
        raise ValueError(msg)
    if molar_mass_g_mol is None:
        msg = f"solvent component '{name}' requires molar_mass_g_mol for count planning"
        raise ValueError(msg)
    if density_g_ml <= 0 or molar_mass_g_mol <= 0:
        msg = f"solvent component '{name}' density and molar mass must be positive"
        raise ValueError(msg)

    component_volume_l = box_volume_nm3 * NM3_TO_L * volume_fraction
    moles = density_g_ml * component_volume_l * L_TO_ML / molar_mass_g_mol
    count = round_half_up(moles * AVOGADRO_CONSTANT_MOL_INV)
    smiles = _get_optional(component, "smiles")
    metadata = {"water_model": water_model} if is_water else {}
    return PlannedMolecule(
        name=name,
        role="solvent",
        count=count,
        smiles=smiles,
        volume_fraction=volume_fraction,
        density_g_ml=density_g_ml,
        molar_mass_g_mol=molar_mass_g_mol,
        metadata=metadata,
    )


def _plan_salt(salt: Any, box_volume_nm3: float) -> PlannedSalt:
    """Plan matched ion counts for one neutral salt."""

    cation = str(_get_required(salt, "cation"))
    anion = str(_get_required(salt, "anion"))
    concentration_molar = float(_get_required(salt, "concentration_molar"))
    if concentration_molar < 0:
        msg = f"salt '{cation}/{anion}' concentration_molar must be non-negative"
        raise ValueError(msg)
    ion_pairs = round_half_up(
        concentration_molar * box_volume_nm3 * NM3_TO_L * AVOGADRO_CONSTANT_MOL_INV
    )
    return PlannedSalt(
        cation=cation,
        anion=anion,
        cation_count=ion_pairs,
        anion_count=ion_pairs,
        concentration_molar=concentration_molar,
        metadata={"neutralize": bool(_get_optional(salt, "neutralize", True))},
    )


def _plan_reactant(reactant: Any, box_volume_nm3: float) -> PlannedMolecule:
    """Plan one molarity-based reactant count."""

    name = str(_get_required(reactant, "name"))
    concentration_molar = float(_get_required(reactant, "concentration_molar"))
    if concentration_molar < 0:
        msg = f"reactant '{name}' concentration_molar must be non-negative"
        raise ValueError(msg)
    count = round_half_up(
        concentration_molar * box_volume_nm3 * NM3_TO_L * AVOGADRO_CONSTANT_MOL_INV
    )
    return PlannedMolecule(
        name=name,
        role="reactant",
        count=count,
        smiles=str(_get_required(reactant, "smiles")),
        concentration_molar=concentration_molar,
    )


def _validate_box_volume(box_volume_nm3: float) -> None:
    """Validate the simulation box volume used for composition planning."""

    if not isfinite(box_volume_nm3) or box_volume_nm3 <= 0:
        msg = "box_volume_nm3 must be a finite positive value"
        raise ValueError(msg)


def _resolve_float(component: Any, attr: str, name: str, property_name: str) -> float | None:
    """Resolve an optional numeric property from input or known component metadata."""

    value = _get_optional(component, attr)
    if value is not None:
        return float(value)
    properties = KNOWN_SOLVENT_PROPERTIES.get(name.lower(), {})
    value = properties.get(property_name)
    return float(value) if value is not None else None


def _get_required(component: Any, attr: str) -> Any:
    """Return a required attribute from dataclass, Pydantic model, or mapping inputs."""

    value = _get_optional(component, attr)
    if value is None:
        msg = f"component is missing required '{attr}'"
        raise ValueError(msg)
    return value


def _get_optional(component: Any, attr: str, default: Any = None) -> Any:
    """Return an optional attribute from dataclass, Pydantic model, or mapping inputs."""

    if isinstance(component, dict):
        return component.get(attr, default)
    return getattr(component, attr, default)
