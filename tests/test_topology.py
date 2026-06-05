"""Tests for topology identity helpers."""

import pytest

from sammd.topology import (
    SAMMD_CHAIN_CONVENTION,
    ComponentResidueAssigner,
    ResidueIdentity,
    get_or_add_chain,
)


def test_component_residue_assigner_wraps_after_9999_residues() -> None:
    """Follow SAMMD's one-repeat-unit-per-residue chain wrapping convention."""

    assigner = ComponentResidueAssigner()

    identities = assigner.allocate("ethanol", "EOH", 10000)
    next_identity = assigner.allocate("reactant", "CIN", 1)[0]

    assert identities[0] == ResidueIdentity("A", 1, "EOH")
    assert identities[9998] == ResidueIdentity("A", 9999, "EOH")
    assert identities[9999] == ResidueIdentity("B", 1, "EOH")
    assert next_identity == ResidueIdentity("C", 1, "CIN")
    assert assigner.component_ranges["ethanol"] == {
        "residue_name": "EOH",
        "residue_count": 10000,
        "first_chain_id": "A",
        "last_chain_id": "B",
        "max_residues_per_chain": 9999,
    }


def test_component_residue_assigner_rejects_invalid_counts() -> None:
    """Require at least one residue for each allocated component."""

    assigner = ComponentResidueAssigner()

    with pytest.raises(ValueError, match="residue_count must be positive"):
        assigner.allocate("empty", "EMP", 0)


def test_get_or_add_chain_reuses_existing_chain() -> None:
    """Cache OpenMM-like topology chains by identifier."""

    class FakeTopology:
        def __init__(self) -> None:
            self.added: list[str] = []

        def addChain(self, chain_id: str) -> str:  # noqa: N802 - OpenMM-style test double
            self.added.append(chain_id)
            return f"chain-{chain_id}"

    topology = FakeTopology()
    cache: dict[str, str] = {}

    first = get_or_add_chain(topology, cache, "A")
    second = get_or_add_chain(topology, cache, "A")

    assert first == second == "chain-A"
    assert topology.added == ["A"]


def test_sammd_surface_chain_convention_is_documented() -> None:
    """Document the SAMMD-specific surface chain convention in metadata."""

    assert SAMMD_CHAIN_CONVENTION == {
        "A": "Pd slab",
        "B": "SAM",
        "C": "reactant",
        "D+": "solvent",
    }
