"""Topology identity helpers for SAMMD visualization-friendly outputs.

SAMMD surface systems intentionally use A=Pd slab, B=SAM, C=reactant, and D+=solvent.
This project-specific convention differs from PolyzyMD protein/substrate/polymer chain
semantics because SAMMD systems do not contain a protein chain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

MAX_RESIDUES_PER_CHAIN = 9999
CHAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SAMMD_CHAIN_CONVENTION = {
    "A": "Pd slab",
    "B": "SAM",
    "C": "reactant",
    "D+": "solvent",
}


@dataclass(frozen=True)
class ResidueIdentity:
    """PDBx identity for one chemically meaningful repeat unit."""

    chain_id: str
    residue_id: int
    residue_name: str


class ComponentResidueAssigner:
    """Assign SAMMD surface-system wrapping chain/residue identifiers by component."""

    def __init__(self) -> None:
        self._next_chain_index = 0
        self._component_ranges: dict[str, dict[str, object]] = {}

    @property
    def component_ranges(self) -> dict[str, dict[str, object]]:
        """Return serializable chain/residue ranges assigned so far."""

        return dict(self._component_ranges)

    def allocate(
        self,
        component_name: str,
        residue_name: str,
        residue_count: int,
    ) -> tuple[ResidueIdentity, ...]:
        """Allocate one residue per repeat unit, wrapping chains every 9999 residues."""

        if residue_count <= 0:
            msg = "residue_count must be positive"
            raise ValueError(msg)
        chains_needed = math.ceil(residue_count / MAX_RESIDUES_PER_CHAIN)
        start_chain_index = self._next_chain_index
        stop_chain_index = start_chain_index + chains_needed - 1
        if stop_chain_index >= len(CHAIN_LETTERS):
            msg = "topology exceeded available one-character chain identifiers"
            raise RuntimeError(msg)

        identities = tuple(
            ResidueIdentity(
                chain_id=CHAIN_LETTERS[start_chain_index + index // MAX_RESIDUES_PER_CHAIN],
                residue_id=index % MAX_RESIDUES_PER_CHAIN + 1,
                residue_name=residue_name,
            )
            for index in range(residue_count)
        )
        self._component_ranges[component_name] = {
            "residue_name": residue_name,
            "residue_count": residue_count,
            "first_chain_id": CHAIN_LETTERS[start_chain_index],
            "last_chain_id": CHAIN_LETTERS[stop_chain_index],
            "max_residues_per_chain": MAX_RESIDUES_PER_CHAIN,
        }
        self._next_chain_index += chains_needed
        return identities


def get_or_add_chain(topology: Any, chain_cache: dict[str, Any], chain_id: str) -> Any:
    """Return an existing OpenMM topology chain or add it by identifier."""

    chain = chain_cache.get(chain_id)
    if chain is None:
        chain = topology.addChain(chain_id)
        chain_cache[chain_id] = chain
    return chain
