"""Deterministic solution composition planning for SAMMD systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import floor, isfinite
from typing import Any

from sammd.core.config import (
    KNOWN_COSOLVENT_MOLAR_MASSES_G_MOL,
    MOLE_FRACTION_TOLERANCE,
    SAMMDConfig,
)

AVOGADRO_CONSTANT_MOL_INV = 6.02214076e23
NM3_TO_L = 1.0e-24
L_TO_ML = 1000.0

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
        "molar_mass_g_mol": KNOWN_COSOLVENT_MOLAR_MASSES_G_MOL["ethanol"],
    },
}


@dataclass(frozen=True)
class SolventComponentSpec:
    """Simple solvent component input for composition planning."""

    name: str
    mole_fraction: float
    residue_name: str = "SOL"
    smiles: str | None = None
    density: float | None = None
    molar_mass: float | None = None


@dataclass(frozen=True)
class IonSpec:
    """Simple ion input for explicit salt stoichiometry planning."""

    name: str
    residue_name: str
    smiles: str
    count_per_formula_unit: int


@dataclass(frozen=True)
class SaltSpec:
    """Simple neutral salt input for composition planning."""

    name: str = "sodium_chloride"
    concentration: float = 0.0
    cation: IonSpec = field(
        default_factory=lambda: IonSpec("sodium", "SOD", "[Na+]", 1)
    )
    anion: IonSpec = field(
        default_factory=lambda: IonSpec("chloride", "CLA", "[Cl-]", 1)
    )


@dataclass(frozen=True)
class ReactantSpec:
    """Simple reactant input for composition planning."""

    name: str
    smiles: str
    residue_name: str = "RCT"
    count: int | None = None
    concentration: float | None = None
    initial_height_above_sam: float = 0.3


@dataclass(frozen=True)
class PlannedMolecule:
    """Planned count and source metadata for one molecular component."""

    name: str
    role: str
    count: int
    residue_name: str | None = None
    smiles: str | None = None
    mole_fraction: float | None = None
    concentration_millimolar: float | None = None
    concentration_molar: float | None = None
    density_g_ml: float | None = None
    molar_mass_g_mol: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlannedSalt:
    """Planned ion counts for one neutral salt."""

    name: str
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
        for salt in self.salts:
            counts[salt.cation] = counts.get(salt.cation, 0) + salt.cation_count
            counts[salt.anion] = counts.get(salt.anion, 0) + salt.anion_count
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
        Volume in nm^3 used for solution count planning. Build planners may pass an approximate
        composition-planning volume rather than a final simulation cell volume.

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
    )


def plan_solution_components(
    box_volume_nm3: float,
    solvent_components: list[Any] | tuple[Any, ...],
    salts: list[Any] | tuple[Any, ...] = (),
    reactants: list[Any] | tuple[Any, ...] = (),
) -> SolutionPlan:
    """Plan solution counts from a count-planning volume and component inputs."""

    _validate_count_planning_volume(box_volume_nm3)
    total_mole_fraction = sum(
        float(_get_required(component, "mole_fraction")) for component in solvent_components
    )
    if abs(total_mole_fraction - 1.0) > MOLE_FRACTION_TOLERANCE:
        msg = "solvent component mole fractions must sum to 1.0"
        raise ValueError(msg)

    warnings: list[str] = []
    planned_solvents = _plan_solvent_components(solvent_components, box_volume_nm3)
    planned_salts = tuple(_plan_salt(salt, box_volume_nm3) for salt in salts)
    planned_reactants = tuple(
        _plan_reactant(reactant, box_volume_nm3, warnings) for reactant in reactants
    )

    for component in (*planned_solvents, *planned_reactants):
        if component.count == 0:
            warnings.append(
                f"{component.role} component '{component.name}' rounded to zero molecules "
                "for this count-planning volume"
            )
    for salt in planned_salts:
        if salt.cation_count == 0 and salt.anion_count == 0:
            warnings.append(
                f"salt '{salt.cation}/{salt.anion}' rounded to zero ion pairs for this "
                "count-planning volume"
            )

    return SolutionPlan(
        box_volume_nm3=box_volume_nm3,
        solvent_components=planned_solvents,
        salts=planned_salts,
        reactants=planned_reactants,
        warnings=tuple(warnings),
    )


def _plan_solvent_components(
    solvent_components: list[Any] | tuple[Any, ...], box_volume_nm3: float
) -> tuple[PlannedMolecule, ...]:
    """Plan solvent counts from solvent-only mole fractions.

    The mixture volume is estimated as an ideal sum of pure-component molar volumes,
    which lets experimental mole fractions produce finite molecule counts for a target
    planning volume without requiring a mixture-density model.
    """

    prepared = tuple(
        _prepare_solvent_component(component) for component in solvent_components
    )
    mixture_molar_volume_ml_mol = sum(
        component["mole_fraction"] * component["molar_mass_g_mol"] / component["density_g_ml"]
        for component in prepared
    )
    if mixture_molar_volume_ml_mol <= 0:
        msg = "solvent mixture molar volume must be positive"
        raise ValueError(msg)
    total_solvent_moles = box_volume_nm3 * NM3_TO_L * L_TO_ML / mixture_molar_volume_ml_mol
    total_expected_count = total_solvent_moles * AVOGADRO_CONSTANT_MOL_INV

    return tuple(
        PlannedMolecule(
            name=str(component["name"]),
            role="solvent",
            count=round_half_up(total_expected_count * component["mole_fraction"]),
            residue_name=component["residue_name"],
            smiles=component["smiles"],
            mole_fraction=component["mole_fraction"],
            density_g_ml=component["density_g_ml"],
            molar_mass_g_mol=component["molar_mass_g_mol"],
            metadata=component["metadata"],
        )
        for component in prepared
    )


def _prepare_solvent_component(component: Any) -> dict[str, Any]:
    """Resolve and validate one solvent component before mixture count planning."""

    name = str(_get_required(component, "name"))
    residue_name = str(_get_required(component, "residue_name"))
    mole_fraction = float(_get_required(component, "mole_fraction"))
    if mole_fraction < 0:
        msg = f"solvent component '{name}' mole_fraction must be non-negative"
        raise ValueError(msg)
    density_g_ml = _resolve_solvent_float_property(
        component, "density", name, "density_g_ml"
    )
    molar_mass_g_mol = _resolve_solvent_float_property(
        component, "molar_mass", name, "molar_mass_g_mol"
    )
    is_water = name.lower() == "water"
    if is_water:
        density_g_ml = density_g_ml or DEFAULT_WATER_DENSITY_G_ML
        molar_mass_g_mol = molar_mass_g_mol or DEFAULT_WATER_MOLAR_MASS_G_MOL
    if density_g_ml is None:
        msg = f"solvent component '{name}' requires density for mole-fraction planning"
        raise ValueError(msg)
    if molar_mass_g_mol is None:
        msg = f"solvent component '{name}' requires molar_mass for count planning"
        raise ValueError(msg)
    if density_g_ml <= 0 or molar_mass_g_mol <= 0:
        msg = f"solvent component '{name}' density and molar mass must be positive"
        raise ValueError(msg)

    smiles = _get_optional(component, "smiles")
    metadata = {"is_water": True} if is_water else {}
    return {
        "name": name,
        "residue_name": residue_name,
        "smiles": smiles,
        "mole_fraction": mole_fraction,
        "density_g_ml": density_g_ml,
        "molar_mass_g_mol": molar_mass_g_mol,
        "metadata": metadata,
    }


def _plan_salt(salt: Any, box_volume_nm3: float) -> PlannedSalt:
    """Plan explicitly stoichiometric ion counts for one salt."""

    name = str(_get_required(salt, "name"))
    cation = _get_required(salt, "cation")
    anion = _get_required(salt, "anion")
    concentration_molar = float(_get_required(salt, "concentration"))
    if concentration_molar < 0:
        msg = f"salt '{name}' concentration must be non-negative"
        raise ValueError(msg)
    formula_units = round_half_up(
        concentration_molar * box_volume_nm3 * NM3_TO_L * AVOGADRO_CONSTANT_MOL_INV
    )
    cation_count_per_unit = int(_get_required(cation, "count_per_formula_unit"))
    anion_count_per_unit = int(_get_required(anion, "count_per_formula_unit"))
    return PlannedSalt(
        name=name,
        cation=str(_get_required(cation, "name")),
        anion=str(_get_required(anion, "name")),
        cation_count=formula_units * cation_count_per_unit,
        anion_count=formula_units * anion_count_per_unit,
        concentration_molar=concentration_molar,
        metadata={
            "formula_units": formula_units,
            "cation_residue_name": _get_required(cation, "residue_name"),
            "cation_smiles": _get_required(cation, "smiles"),
            "cation_count_per_formula_unit": cation_count_per_unit,
            "anion_residue_name": _get_required(anion, "residue_name"),
            "anion_smiles": _get_required(anion, "smiles"),
            "anion_count_per_formula_unit": anion_count_per_unit,
        },
    )


def _plan_reactant(
    reactant: Any, box_volume_nm3: float, warnings: list[str]
) -> PlannedMolecule:
    """Plan one count- or molarity-based reactant count."""

    name = str(_get_required(reactant, "name"))
    residue_name = str(_get_required(reactant, "residue_name"))
    configured_count = _get_optional(reactant, "count")
    configured_concentration = _get_optional(reactant, "concentration")
    if configured_count is not None:
        count = int(configured_count)
        if count < 0:
            msg = f"reactant '{name}' count must be non-negative"
            raise ValueError(msg)
        concentration_millimolar = None
        concentration_molar = None
        expected_count = float(count)
        realized_concentration_millimolar = (
            count / (box_volume_nm3 * NM3_TO_L * AVOGADRO_CONSTANT_MOL_INV) * 1000.0
        )
    elif configured_concentration is not None:
        concentration_millimolar = float(configured_concentration)
        if concentration_millimolar < 0:
            msg = f"reactant '{name}' concentration must be non-negative"
            raise ValueError(msg)
        concentration_molar = concentration_millimolar / 1000.0
        expected_count = concentration_molar * box_volume_nm3 * NM3_TO_L * AVOGADRO_CONSTANT_MOL_INV
        count = round_half_up(expected_count)
        if concentration_millimolar > 0 and count == 0:
            count = 1
        if concentration_millimolar > 0 and count == 1:
            warnings.append(
                f"SAMMD will only place 1 molecule of reactant '{name}' in this box; "
                f"requested concentration {concentration_millimolar:g} mM. "
                "Consider increasing the slab size."
            )
        realized_concentration_millimolar = (
            count / (box_volume_nm3 * NM3_TO_L * AVOGADRO_CONSTANT_MOL_INV) * 1000.0
        )
    else:
        msg = f"reactant '{name}' must define count or concentration"
        raise ValueError(msg)
    return PlannedMolecule(
        name=name,
        role="reactant",
        count=count,
        residue_name=residue_name,
        smiles=str(_get_required(reactant, "smiles")),
        concentration_millimolar=concentration_millimolar,
        concentration_molar=concentration_molar,
        metadata={
            "expected_count": expected_count,
            "realized_concentration_millimolar": realized_concentration_millimolar,
            "initial_height_above_sam_nm": _get_required(reactant, "initial_height_above_sam"),
        },
    )


def _validate_count_planning_volume(box_volume_nm3: float) -> None:
    """Validate the volume used for solution composition count planning."""

    if not isfinite(box_volume_nm3) or box_volume_nm3 <= 0:
        msg = "count-planning volume must be a finite positive value"
        raise ValueError(msg)


def _resolve_solvent_float_property(
    component: Any, attr: str, name: str, property_name: str
) -> float | None:
    """Resolve an optional solvent property from input or known solvent metadata."""

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
