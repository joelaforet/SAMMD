"""Tests for SAMMD OpenFF Interchange plugin collections."""

from __future__ import annotations

import importlib.metadata

import pytest

from sammd.backends import interchange_plugins as plugins
from sammd.model.metal_sulfur import METAL_SULFUR_EPSILON_KCAL_MOL, METAL_SULFUR_SIGMA_NM


def test_metal_sulfur_override_summary_records_selected_pairs_and_parameters() -> None:
    """Dependency-light metadata mirrors the plugin collection export contract."""

    summary = plugins.metal_sulfur_lj_override_summary(((5, 1), (5, 2)))

    assert summary["mode"] == "openff_interchange_plugin_collection"
    assert summary["collection_type"] == plugins.SAMMD_METAL_SULFUR_COLLECTION_TYPE
    assert summary["sulfur_metal_pairs"] == [[5, 1], [5, 2]]
    assert summary["sigma_nm"] == METAL_SULFUR_SIGMA_NM == 0.22
    assert summary["epsilon_kcal_mol"] == METAL_SULFUR_EPSILON_KCAL_MOL == 2.0
    assert summary["charge_product"] == 0
    assert summary["openmm_exception_replace"] is True


def test_metal_sulfur_plugin_collection_builds_exact_topology_keys() -> None:
    """Build exact TopologyKey pair mappings when OpenFF Interchange is installed."""

    pytest.importorskip("openff.interchange")

    collection = plugins.create_metal_sulfur_lj_collection(((7, 0), (7, 3)))

    assert collection.type == plugins.SAMMD_METAL_SULFUR_COLLECTION_TYPE
    assert collection.is_plugin is True
    assert collection.expression == plugins.SAMMD_METAL_SULFUR_EXPRESSION
    assert [key.atom_indices for key in collection.key_map] == [(7, 0), (7, 3)]
    for potential in collection.potentials.values():
        assert potential.parameters["sigma"].m_as("nanometer") == pytest.approx(0.22)
        assert potential.parameters["epsilon"].m_as("kilocalorie_per_mole") == pytest.approx(2.0)
        assert potential.parameters["charge_product"].m == pytest.approx(0.0)


def test_interchange_plugin_collection_entry_point_is_registered() -> None:
    """Expose the collection through Interchange's plugin entry point group."""

    entry_points = importlib.metadata.entry_points(
        group="openff.interchange.plugins.collections"
    )
    assert any(
        entry_point.value
        == "sammd.backends.interchange_plugins:SAMMDMetalSulfurLJCollection"
        for entry_point in entry_points
    )
