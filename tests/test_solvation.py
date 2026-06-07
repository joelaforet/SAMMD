"""Tests for lightweight solution composition planning."""

import pytest

from sammd.config import IonConfig, SaltConfig, SAMMDConfig, SolventComponentConfig, SolventConfig
from sammd.solvation import (
    AVOGADRO_CONSTANT_MOL_INV,
    L_TO_ML,
    NM3_TO_L,
    IonSpec,
    ReactantSpec,
    SaltSpec,
    SolventComponentSpec,
    plan_solution_components,
    plan_solution_composition,
    round_half_up,
)


def test_default_ethanol_gives_positive_count() -> None:
    """Plan the default ethanol count for a known box volume."""

    box_volume_nm3 = 125.0
    expected_count = round_half_up(
        0.789 * box_volume_nm3 * NM3_TO_L * L_TO_ML / 46.06844 * AVOGADRO_CONSTANT_MOL_INV
    )

    plan = plan_solution_composition(SAMMDConfig(), box_volume_nm3=box_volume_nm3)

    assert plan.solvent_components[0].name == "ethanol"
    assert plan.solvent_components[0].residue_name == "EOH"
    assert plan.solvent_components[0].count == expected_count


def test_water_ethanol_mole_fraction_counts() -> None:
    """Plan both water and ethanol counts for a 50/50 solvent mole-fraction mixture."""

    box_volume_nm3 = 125.0
    water_mole_fraction = 0.5
    ethanol_mole_fraction = 0.5
    mixture_molar_volume_ml_mol = (
        water_mole_fraction * 18.01528 / 0.997
        + ethanol_mole_fraction * 46.06844 / 0.789
    )
    total_expected = (
        box_volume_nm3
        * NM3_TO_L
        * L_TO_ML
        / mixture_molar_volume_ml_mol
        * AVOGADRO_CONSTANT_MOL_INV
    )
    water_expected = round_half_up(total_expected * water_mole_fraction)
    ethanol_expected = round_half_up(total_expected * ethanol_mole_fraction)
    config = SAMMDConfig(
        solvent=SolventConfig(
            components=[
                SolventComponentConfig(
                    name="water", residue_name="HOH", smiles="O", mole_fraction=water_mole_fraction
                ),
                SolventComponentConfig(
                    name="ethanol",
                    residue_name="EOH",
                    smiles="CCO",
                    mole_fraction=ethanol_mole_fraction,
                    density=0.789,
                ),
            ]
        )
    )

    plan = plan_solution_composition(config, box_volume_nm3=box_volume_nm3)
    counts = plan.molecule_counts

    assert counts["water"] == water_expected
    assert counts["ethanol"] == ethanol_expected
    assert sum(component.mole_fraction or 0.0 for component in plan.solvent_components) == 1.0


def test_missing_cosolvent_density_fails_clearly() -> None:
    """Reject schema-bypassed mole-fraction co-solvents without density."""

    with pytest.raises(ValueError, match="requires density"):
        plan_solution_components(
            box_volume_nm3=125.0,
            solvent_components=[
                SolventComponentSpec(name="methanol", residue_name="MET", mole_fraction=1.0)
            ],
        )


def test_salt_stoichiometry_produces_ion_counts() -> None:
    """Plan explicit cation and anion counts for non-1:1 salts."""

    box_volume_nm3 = 125.0
    concentration = 0.1
    formula_units = round_half_up(
        concentration * box_volume_nm3 * NM3_TO_L * AVOGADRO_CONSTANT_MOL_INV
    )
    config = SAMMDConfig(
        salts=[
            SaltConfig(
                name="sodium_sulfate",
                concentration=concentration,
                cation=IonConfig(
                    name="sodium",
                    residue_name="SOD",
                    smiles="[Na+]",
                    count_per_formula_unit=2,
                ),
                anion=IonConfig(
                    name="sulfate",
                    residue_name="SUL",
                    smiles="O=S(=O)([O-])[O-]",
                    count_per_formula_unit=1,
                ),
            )
        ]
    )

    plan = plan_solution_composition(config, box_volume_nm3=box_volume_nm3)

    assert plan.salts[0].name == "sodium_sulfate"
    assert plan.salts[0].cation == "sodium"
    assert plan.salts[0].anion == "sulfate"
    assert plan.salts[0].cation_count == 2 * formula_units
    assert plan.salts[0].anion_count == formula_units
    assert plan.salts[0].metadata["cation_residue_name"] == "SOD"
    assert plan.salts[0].metadata["anion_residue_name"] == "SUL"


def test_default_reactant_count_is_direct() -> None:
    """Default config places one cinnamaldehyde reactant by explicit count."""

    plan = plan_solution_composition(SAMMDConfig(), box_volume_nm3=125.0)

    assert plan.reactants[0].name == "cinnamaldehyde"
    assert plan.reactants[0].residue_name == "CIN"
    assert plan.reactants[0].count == 1
    assert plan.reactants[0].concentration_millimolar is None


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
        solvent_components=[
            SolventComponentSpec(name="water", residue_name="HOH", mole_fraction=1.0)
        ],
        salts=[
            SaltSpec(
                concentration=0.0,
                cation=IonSpec("sodium", "SOD", "[Na+]", 1),
                anion=IonSpec("chloride", "CLA", "[Cl-]", 1),
            )
        ],
        reactants=[ReactantSpec(name="trace", residue_name="TRC", smiles="C", concentration=0.0)],
    )

    assert plan.salts[0].cation_count == 0
    assert plan.salts[0].anion_count == 0
    assert plan.reactants[0].count == 0
    assert plan.warnings


def test_generic_reactant_concentration_count() -> None:
    """Plan an exact generic reactant count for a known box and molarity."""

    box_volume_nm3 = 250.0
    concentration = 200.0
    expected_count = round_half_up(
        concentration / 1000.0 * box_volume_nm3 * NM3_TO_L * AVOGADRO_CONSTANT_MOL_INV
    )

    plan = plan_solution_components(
        box_volume_nm3=box_volume_nm3,
        solvent_components=[
            SolventComponentSpec(name="water", residue_name="HOH", mole_fraction=1.0)
        ],
        reactants=[
            ReactantSpec(
                name="generic",
                residue_name="GEN",
                smiles="C",
                concentration=concentration,
            )
        ],
    )

    assert plan.reactants[0].count == expected_count


def test_mole_fraction_tolerance_matches_config_schema() -> None:
    """Accept the same near-one mole-fraction total in config and raw planner."""

    components = [
        SolventComponentConfig(name="water", residue_name="HOH", smiles="O", mole_fraction=0.5),
        SolventComponentConfig(
            name="ethanol",
            residue_name="EOH",
            mole_fraction=0.4999995,
            density=0.789,
        ),
    ]
    config = SAMMDConfig(solvent=SolventConfig(components=components))

    plan = plan_solution_composition(config, box_volume_nm3=125.0)
    raw_plan = plan_solution_components(
        box_volume_nm3=125.0,
        solvent_components=config.solvent.components,
        reactants=config.reactants,
    )

    assert sum(component.mole_fraction or 0.0 for component in plan.solvent_components) < 1.0
    assert raw_plan.molecule_counts == plan.molecule_counts


@pytest.mark.parametrize("box_volume_nm3", [0.0, -1.0])
def test_invalid_box_volume_fails(box_volume_nm3: float) -> None:
    """Reject non-positive box volumes before count planning."""

    with pytest.raises(ValueError, match="count-planning volume must be a finite positive value"):
        plan_solution_composition(SAMMDConfig(), box_volume_nm3=box_volume_nm3)


@pytest.mark.parametrize(
    "components",
    [
        [SolventComponentSpec(name="water", residue_name="HOH", mole_fraction=0.75)],
        [
            SolventComponentSpec(name="water", residue_name="HOH", mole_fraction=0.75),
            SolventComponentSpec(
                name="ethanol",
                residue_name="EOH",
                mole_fraction=0.5,
                density=0.789,
            ),
        ],
    ],
)
def test_raw_mole_fractions_must_sum_to_one(
    components: list[SolventComponentSpec],
) -> None:
    """Reject schema-bypassed solvent mole fraction totals not equal to one."""

    with pytest.raises(ValueError, match=r"mole fractions must sum to 1\.0"):
        plan_solution_components(
            box_volume_nm3=125.0,
            solvent_components=components,
        )


def test_nonzero_reactant_concentration_places_one_molecule_and_warns() -> None:
    """Keep finite-size reactant boxes nonempty and warn when only one is placed."""

    plan = plan_solution_components(
        box_volume_nm3=1.0,
        solvent_components=[
            SolventComponentSpec(name="water", residue_name="HOH", mole_fraction=1.0)
        ],
        reactants=[ReactantSpec(name="trace", residue_name="TRC", smiles="C", concentration=1.0)],
    )

    assert plan.reactants[0].count == 1
    assert plan.reactants[0].metadata["expected_count"] < 1.0
    assert plan.reactants[0].metadata["realized_concentration_millimolar"] > 1.0
    assert "SAMMD will only place 1 molecule" in plan.warnings[0]
