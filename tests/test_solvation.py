"""Tests for lightweight solution composition planning."""

import pytest

from sammd.config import (
    SaltConfig,
    SAMMDConfig,
    SolventComponentConfig,
    SolventConfig,
)
from sammd.solvation import (
    AVOGADRO_CONSTANT_MOL_INV,
    L_TO_ML,
    NM3_TO_L,
    ReactantSpec,
    SaltSpec,
    SolventComponentSpec,
    plan_solution_components,
    plan_solution_composition,
    round_half_up,
)


def test_water_only_default_gives_positive_count() -> None:
    """Plan the exact TIP3P water count for a known box volume."""

    box_volume_nm3 = 125.0
    density_g_ml = 0.997
    molar_mass_g_mol = 18.01528
    expected_count = round_half_up(
        density_g_ml
        * box_volume_nm3
        * NM3_TO_L
        * L_TO_ML
        / molar_mass_g_mol
        * AVOGADRO_CONSTANT_MOL_INV
    )

    plan = plan_solution_composition(SAMMDConfig(), box_volume_nm3=box_volume_nm3)

    assert plan.solvent_components[0].name == "water"
    assert plan.solvent_components[0].count == expected_count
    assert plan.solvent_components[0].metadata["water_model"] == "TIP3P"


def test_water_ethanol_volume_fraction_counts() -> None:
    """Plan both water and ethanol counts for a 50/50 liquid mixture."""

    box_volume_nm3 = 125.0
    water_fraction = 0.5
    ethanol_fraction = 0.5
    water_expected = round_half_up(
        0.997
        * box_volume_nm3
        * water_fraction
        * NM3_TO_L
        * L_TO_ML
        / 18.01528
        * AVOGADRO_CONSTANT_MOL_INV
    )
    ethanol_expected = round_half_up(
        0.789
        * box_volume_nm3
        * ethanol_fraction
        * NM3_TO_L
        * L_TO_ML
        / 46.06844
        * AVOGADRO_CONSTANT_MOL_INV
    )
    config = SAMMDConfig(
        solvent=SolventConfig(
            components=[
                SolventComponentConfig(name="water", volume_fraction=water_fraction),
                SolventComponentConfig(
                    name="ethanol",
                    smiles="CCO",
                    volume_fraction=ethanol_fraction,
                    density_g_ml=0.789,
                ),
            ]
        )
    )

    plan = plan_solution_composition(config, box_volume_nm3=box_volume_nm3)
    counts = plan.molecule_counts

    assert counts["water"] == water_expected
    assert counts["ethanol"] == ethanol_expected
    assert sum(component.volume_fraction or 0.0 for component in plan.solvent_components) == 1.0


def test_missing_cosolvent_density_fails_clearly() -> None:
    """Reject schema-bypassed volume-fraction co-solvents without density."""

    with pytest.raises(ValueError, match="requires density_g_ml"):
        plan_solution_components(
            box_volume_nm3=125.0,
            solvent_components=[SolventComponentSpec(name="methanol", volume_fraction=1.0)],
        )


def test_nacl_molarity_produces_equal_ion_counts() -> None:
    """Plan matched Na+ and Cl- counts for neutral NaCl."""

    box_volume_nm3 = 125.0
    concentration_molar = 0.1
    expected_count = round_half_up(
        concentration_molar * box_volume_nm3 * NM3_TO_L * AVOGADRO_CONSTANT_MOL_INV
    )
    config = SAMMDConfig(salts=[SaltConfig(concentration_molar=concentration_molar)])

    plan = plan_solution_composition(config, box_volume_nm3=box_volume_nm3)

    assert plan.salts[0].cation == "Na+"
    assert plan.salts[0].anion == "Cl-"
    assert plan.salts[0].cation_count == plan.salts[0].anion_count
    assert plan.salts[0].cation_count == expected_count


def test_cinnamaldehyde_default_count_is_deterministic() -> None:
    """Plan the default cinnamaldehyde reactant with deterministic rounding."""

    box_volume_nm3 = 125.0
    concentration_molar = 0.05
    expected_count = round_half_up(
        concentration_molar * box_volume_nm3 * NM3_TO_L * AVOGADRO_CONSTANT_MOL_INV
    )

    plan = plan_solution_composition(SAMMDConfig(), box_volume_nm3=box_volume_nm3)

    assert plan.reactants[0].name == "cinnamaldehyde"
    assert plan.reactants[0].count == expected_count


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.49, 0),
        (0.5, 1),
        (1.5, 2),
    ],
)
def test_half_up_rounding_boundary_behavior(value: float, expected: int) -> None:
    """Round exact half-count boundaries away from zero."""

    assert round_half_up(value) == expected


def test_zero_concentration_can_produce_zero_count() -> None:
    """Allow zero counts from very small or zero target concentrations."""

    plan = plan_solution_components(
        box_volume_nm3=1.0,
        solvent_components=[SolventComponentSpec(name="water", volume_fraction=1.0)],
        salts=[SaltSpec(concentration_molar=0.0)],
        reactants=[ReactantSpec(name="trace", smiles="C", concentration_molar=0.0)],
    )

    assert plan.salts[0].cation_count == 0
    assert plan.salts[0].anion_count == 0
    assert plan.reactants[0].count == 0
    assert plan.warnings


def test_generic_reactant_formula_count() -> None:
    """Plan an exact generic reactant count for a known box and molarity."""

    box_volume_nm3 = 250.0
    concentration_molar = 0.2
    expected_count = round_half_up(
        concentration_molar * box_volume_nm3 * NM3_TO_L * AVOGADRO_CONSTANT_MOL_INV
    )

    plan = plan_solution_components(
        box_volume_nm3=box_volume_nm3,
        solvent_components=[SolventComponentSpec(name="water", volume_fraction=1.0)],
        reactants=[
            ReactantSpec(name="generic", smiles="C", concentration_molar=concentration_molar)
        ],
    )

    assert plan.reactants[0].count == expected_count


def test_volume_fraction_tolerance_matches_config_schema() -> None:
    """Accept the same near-one volume-fraction total in config and raw planner."""

    components = [
        SolventComponentConfig(name="water", volume_fraction=0.5),
        SolventComponentConfig(
            name="ethanol",
            volume_fraction=0.4999995,
            density_g_ml=0.789,
        ),
    ]
    config = SAMMDConfig(solvent=SolventConfig(components=components))

    plan = plan_solution_composition(config, box_volume_nm3=125.0)
    raw_plan = plan_solution_components(
        box_volume_nm3=125.0,
        solvent_components=config.solvent.components,
        reactants=config.reactants,
    )

    assert sum(component.volume_fraction or 0.0 for component in plan.solvent_components) < 1.0
    assert raw_plan.molecule_counts == plan.molecule_counts


@pytest.mark.parametrize("box_volume_nm3", [0.0, -1.0])
def test_invalid_box_volume_fails(box_volume_nm3: float) -> None:
    """Reject non-positive box volumes before count planning."""

    with pytest.raises(ValueError, match="count-planning volume must be a finite positive value"):
        plan_solution_composition(SAMMDConfig(), box_volume_nm3=box_volume_nm3)


@pytest.mark.parametrize(
    "components",
    [
        [SolventComponentSpec(name="water", volume_fraction=0.75)],
        [
            SolventComponentSpec(name="water", volume_fraction=0.75),
            SolventComponentSpec(
                name="ethanol",
                volume_fraction=0.5,
                density_g_ml=0.789,
            ),
        ],
    ],
)
def test_raw_volume_fractions_must_sum_to_one(
    components: list[SolventComponentSpec],
) -> None:
    """Reject schema-bypassed solvent volume fraction totals not equal to one."""

    with pytest.raises(ValueError, match=r"volume fractions must sum to 1\.0"):
        plan_solution_components(
            box_volume_nm3=125.0,
            solvent_components=components,
        )
