"""Tests for INTERFACE Fcc metal parameters."""

import pytest

from sammd.forcefields import FCC_METAL_LJ_REGISTRY, get_fcc_metal_parameters, sigma_from_rmin_half


def test_registry_contains_all_fcc_metals() -> None:
    """Reproduce the CHARMM-INTERFACE Fcc metal set."""

    assert set(FCC_METAL_LJ_REGISTRY) == {"Ag", "Al", "Au", "Cu", "Ni", "Pb", "Pd", "Pt"}


def test_registry_reproduces_pd_values() -> None:
    """Validate Pd parameters from the project scope table."""

    pd = get_fcc_metal_parameters("Pd")
    assert pd.source_epsilon_kcal_mol == -6.15
    assert pd.openff_epsilon_kcal_mol == 6.15
    assert pd.rmin_half_angstrom == 1.4095


def test_epsilon_sign_conversion_and_sigma_helper() -> None:
    """Store CHARMM epsilon as negative and OpenFF epsilon as positive."""

    for parameters in FCC_METAL_LJ_REGISTRY.values():
        assert parameters.source_epsilon_kcal_mol < 0
        assert parameters.openff_epsilon_kcal_mol == abs(parameters.source_epsilon_kcal_mol)
        assert parameters.sigma_angstrom == pytest.approx(
            2 * parameters.rmin_half_angstrom / 2 ** (1 / 6)
        )
    assert sigma_from_rmin_half(1.4095) == pytest.approx(2.511443, rel=1e-6)


def test_unsupported_metal_fails_clearly() -> None:
    """Reject metals outside the lightweight Fcc registry."""

    with pytest.raises(ValueError, match="unsupported Fcc metal"):
        get_fcc_metal_parameters("Fe")
