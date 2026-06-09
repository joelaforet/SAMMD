"""OpenFF Interchange plugin collections owned by SAMMD."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from sammd.model.metal_sulfur import (
    KCAL_TO_KJ,
    METAL_SULFUR_EPSILON_KCAL_MOL,
    METAL_SULFUR_SIGMA_NM,
)

SAMMD_METAL_SULFUR_COLLECTION_TYPE = "SAMMDMetalSulfurLJ"
SAMMD_METAL_SULFUR_EXPRESSION = "4*epsilon*((sigma/r)**12-(sigma/r)**6)"

try:  # pragma: no cover - exercised only in optional OpenFF environments
    from openff.interchange.components.potentials import Potential
    from openff.interchange.models import PotentialKey, TopologyKey
    from openff.interchange.smirnoff import SMIRNOFFCollection
    from openff.units import unit
except ImportError as error:  # pragma: no cover - dependency-light tests assert helpers
    OPENFF_PLUGIN_IMPORT_ERROR: ImportError | None = error
    Potential = PotentialKey = TopologyKey = unit = None  # type: ignore[assignment]

    class SMIRNOFFCollection:  # type: ignore[no-redef]
        """Placeholder base so this module remains importable without OpenFF."""

else:  # pragma: no cover - OpenFF is not installed in the default test env
    OPENFF_PLUGIN_IMPORT_ERROR = None


class SAMMDMetalSulfurLJCollection(SMIRNOFFCollection):
    """Exact SAMMD-selected sulfur-metal LJ overrides for Interchange exports."""

    type: str = SAMMD_METAL_SULFUR_COLLECTION_TYPE
    is_plugin: bool = True
    expression: str = SAMMD_METAL_SULFUR_EXPRESSION

    @classmethod
    def allowed_parameter_handlers(cls) -> list[type]:
        """No SMIRNOFF handler owns these exact topology-index pairs."""

        return []

    @classmethod
    def supported_parameters(cls) -> list[str]:
        """Return the parameter names stored by this plugin collection."""

        return ["id", "sigma", "epsilon", "charge_product"]

    @classmethod
    def potential_parameters(cls) -> list[str]:
        """Return parameters stored on each selected pair potential."""

        return ["sigma", "epsilon", "charge_product"]

    def store_matches(self, *args: Any, **kwargs: Any) -> None:
        """Disable SMIRNOFF matching; SAMMD provides exact topology pairs."""

        raise NotImplementedError("SAMMD metal-S pairs are assigned by atom index")

    def store_potentials(self, *args: Any, **kwargs: Any) -> None:
        """Disable SMIRNOFF parameter storage; SAMMD provides exact potentials."""

        raise NotImplementedError("SAMMD metal-S potentials are assigned by atom index")

    @classmethod
    def from_anchor_pairs(
        cls,
        anchor_pairs: Iterable[tuple[int, int]],
        *,
        sigma_nm: float = METAL_SULFUR_SIGMA_NM,
        epsilon_kcal_mol: float = METAL_SULFUR_EPSILON_KCAL_MOL,
    ) -> SAMMDMetalSulfurLJCollection:
        """Build a plugin collection from exact sulfur-metal atom-index pairs."""

        if OPENFF_PLUGIN_IMPORT_ERROR is not None:
            raise ImportError(
                "OpenFF Interchange is required to construct the SAMMD metal-S plugin collection."
            ) from OPENFF_PLUGIN_IMPORT_ERROR
        validated_pairs = validate_anchor_pairs(anchor_pairs)
        _validate_positive_finite(sigma_nm, "sigma_nm")
        _validate_positive_finite(epsilon_kcal_mol, "epsilon_kcal_mol")

        collection = cls()
        for sulfur_index, metal_index in validated_pairs:
            topology_key = TopologyKey(atom_indices=(sulfur_index, metal_index))
            potential_key = PotentialKey(
                id=f"{SAMMD_METAL_SULFUR_COLLECTION_TYPE}-{sulfur_index}-{metal_index}",
                associated_handler=SAMMD_METAL_SULFUR_COLLECTION_TYPE,
            )
            collection.key_map[topology_key] = potential_key
            collection.potentials[potential_key] = Potential(
                parameters={
                    "sigma": sigma_nm * unit.nanometer,
                    "epsilon": epsilon_kcal_mol * unit.kilocalorie_per_mole,
                    "charge_product": 0.0 * unit.elementary_charge**2,
                }
            )
        return collection

    def to_summary(self) -> dict[str, object]:
        """Return JSON-friendly metadata for this plugin collection."""

        return metal_sulfur_lj_override_summary(_pairs_from_collection(self))

    @classmethod
    def check_openmm_requirements(cls, combine_nonbonded_forces: bool) -> None:
        """Require a combined NonbondedForce that can receive exceptions."""

        assert combine_nonbonded_forces

    def modify_openmm_forces(
        self,
        interchange: Any,
        system: Any,
        add_constrained_forces: bool,
        constrained_pairs: set[tuple[int, ...]],
        particle_map: dict[Any, int],
    ) -> None:
        """Apply selected pair overrides as OpenMM NonbondedForce exceptions."""

        del interchange, add_constrained_forces, constrained_pairs
        nonbonded_force = _find_nonbonded_force(system)
        if nonbonded_force is None:
            msg = "system must contain an OpenMM NonbondedForce for SAMMD metal-S overrides"
            raise ValueError(msg)

        openmm_unit = _require_openmm_unit()
        for topology_key, potential_key in self.key_map.items():
            sulfur_index, metal_index = topology_key.atom_indices
            potential = self.potentials[potential_key]
            sigma_nm = _quantity_to_float(potential.parameters["sigma"], "nanometer")
            epsilon_kj_mol = _quantity_to_float(
                potential.parameters["epsilon"],
                "kilojoule_per_mole",
            )
            nonbonded_force.addException(
                _map_particle_index(sulfur_index, particle_map),
                _map_particle_index(metal_index, particle_map),
                0.0 * openmm_unit.elementary_charge**2,
                sigma_nm * openmm_unit.nanometer,
                epsilon_kj_mol * openmm_unit.kilojoule_per_mole,
                replace=True,
            )


def create_metal_sulfur_lj_collection(
    anchor_pairs: Iterable[tuple[int, int]],
) -> SAMMDMetalSulfurLJCollection:
    """Return the SAMMD Interchange plugin collection for selected anchor pairs."""

    return SAMMDMetalSulfurLJCollection.from_anchor_pairs(anchor_pairs)


def metal_sulfur_lj_override_summary(
    anchor_pairs: Iterable[tuple[int, int]],
) -> dict[str, object]:
    """Return dependency-free metadata matching the plugin collection parameters."""

    validated_pairs = validate_anchor_pairs(anchor_pairs)
    return {
        "mode": "openff_interchange_plugin_collection",
        "collection_type": SAMMD_METAL_SULFUR_COLLECTION_TYPE,
        "sulfur_metal_pairs": [list(pair) for pair in validated_pairs],
        "sigma_nm": METAL_SULFUR_SIGMA_NM,
        "epsilon_kcal_mol": METAL_SULFUR_EPSILON_KCAL_MOL,
        "epsilon_kj_mol": METAL_SULFUR_EPSILON_KCAL_MOL * KCAL_TO_KJ,
        "charge_product": 0,
        "openmm_exception_replace": True,
    }


def validate_anchor_pairs(
    anchor_pairs: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Validate exact sulfur-metal atom-index pairs."""

    try:
        pair_items = tuple(anchor_pairs)
    except TypeError as error:
        msg = "anchor_pairs must be an iterable of sulfur-metal index pairs"
        raise ValueError(msg) from error
    if not pair_items:
        msg = "anchor_pairs must contain at least one sulfur-metal pair"
        raise ValueError(msg)

    validated_pairs: list[tuple[int, int]] = []
    seen_pairs: set[frozenset[int]] = set()
    for pair in pair_items:
        try:
            sulfur_index, metal_index = tuple(pair)
        except (TypeError, ValueError) as error:
            msg = "each sulfur-metal pair must contain exactly two atom indices"
            raise ValueError(msg) from error
        for atom_index in (sulfur_index, metal_index):
            if isinstance(atom_index, bool) or not isinstance(atom_index, int) or atom_index < 0:
                msg = "pair atom indices must be non-negative integers and not booleans"
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
    return tuple(validated_pairs)


def _pairs_from_collection(collection: SAMMDMetalSulfurLJCollection) -> tuple[tuple[int, int], ...]:
    return tuple(tuple(key.atom_indices) for key in collection.key_map)


def _validate_positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        msg = f"{name} must be positive and finite"
        raise ValueError(msg)


def _find_nonbonded_force(system: Any) -> Any | None:
    for force_index in range(system.getNumForces()):
        force = system.getForce(force_index)
        if force.__class__.__name__ == "NonbondedForce":
            return force
    return None


def _map_particle_index(atom_index: int, particle_map: dict[Any, int]) -> int:
    return int(particle_map.get(atom_index, atom_index))


def _quantity_to_float(value: Any, unit_name: str) -> float:
    if hasattr(value, "m_as"):
        return float(value.m_as(unit_name))
    if hasattr(value, "to"):
        converted = value.to(unit_name)
        magnitude = getattr(converted, "m", converted)
        return float(magnitude)
    return float(value)


def _require_openmm_unit() -> Any:
    from openmm import unit as openmm_unit

    return openmm_unit
