"""Tests for SAMMD OpenFF Interchange plugin collections."""

from __future__ import annotations

import importlib.metadata
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from sammd.backends import interchange_plugins as plugins
from sammd.model.metal_sulfur import (
    KCAL_TO_KJ,
    METAL_SULFUR_EPSILON_KCAL_MOL,
    METAL_SULFUR_SIGMA_NM,
)


class FakeQuantity:
    def __init__(self, value: float, unit_name: str) -> None:
        self.value = float(value)
        self.unit_name = unit_name

    @property
    def m(self) -> float:
        return self.value

    def m_as(self, unit_name: str) -> float:
        if self.unit_name == "kilocalorie_per_mole" and unit_name == "kilojoule_per_mole":
            return self.value * KCAL_TO_KJ
        return self.value

    def to(self, unit_name: str) -> FakeQuantity:
        return FakeQuantity(self.m_as(unit_name), unit_name)

    def value_in_unit(self, unit: FakeUnit) -> float:
        return self.m_as(unit.name)


class FakeUnit:
    def __init__(self, name: str) -> None:
        self.name = name

    def __rmul__(self, value: float) -> FakeQuantity:
        return FakeQuantity(value, self.name)

    def __pow__(self, power: int) -> FakeUnit:
        return FakeUnit(f"{self.name}^{power}")


@dataclass(frozen=True)
class FakePotentialKey:
    id: str
    associated_handler: str


@dataclass(frozen=True)
class FakeTopologyKey:
    atom_indices: tuple[int, int]


class FakePotential:
    def __init__(self, parameters: dict[str, FakeQuantity]) -> None:
        self.parameters = parameters


class FakeSMIRNOFFCollection:
    def __init__(self) -> None:
        self.key_map: dict[FakeTopologyKey, FakePotentialKey] = {}
        self.potentials: dict[FakePotentialKey, FakePotential] = {}

    @classmethod
    def model_validate(cls, data: Any) -> dict[str, Any]:
        return {"class": cls.__name__, "data": data}


class ToOnlyQuantity:
    def to(self, unit_name: str) -> SimpleNamespace:
        return SimpleNamespace(m=3.25, unit_name=unit_name)


class FakeSystem:
    def __init__(self, forces: list[Any]) -> None:
        self._forces = forces

    def getNumForces(self) -> int:  # noqa: N802 - mirrors OpenMM's API
        return len(self._forces)

    def getForce(self, force_index: int) -> Any:  # noqa: N802 - mirrors OpenMM's API
        return self._forces[force_index]


class NonbondedForce:
    def __init__(self) -> None:
        self.exceptions: list[tuple[Any, ...]] = []

    def addException(self, *args: Any, **kwargs: Any) -> None:  # noqa: N802 - mirrors OpenMM's API
        self.exceptions.append((*args, kwargs))


def install_fake_openff_and_openmm(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    unit_module = SimpleNamespace(
        nanometer=FakeUnit("nanometer"),
        kilocalorie_per_mole=FakeUnit("kilocalorie_per_mole"),
        elementary_charge=FakeUnit("elementary_charge"),
    )
    openmm_unit_module = SimpleNamespace(
        nanometer=FakeUnit("nanometer"),
        kilojoule_per_mole=FakeUnit("kilojoule_per_mole"),
        elementary_charge=FakeUnit("elementary_charge"),
    )

    potentials_module = ModuleType("openff.interchange.components.potentials")

    def validate_collections(v: Any, handler: Any, info: Any) -> dict[str, Any]:
        return {"original": v, "handler": handler, "info": info}

    potentials_module.Potential = FakePotential
    potentials_module.validate_collections = validate_collections

    components_module = ModuleType("openff.interchange.components")
    components_module.potentials = potentials_module

    models_module = ModuleType("openff.interchange.models")
    models_module.PotentialKey = FakePotentialKey
    models_module.TopologyKey = FakeTopologyKey

    smirnoff_module = ModuleType("openff.interchange.smirnoff")
    smirnoff_module.SMIRNOFFCollection = FakeSMIRNOFFCollection
    for class_name in (
        "SMIRNOFFAngleCollection",
        "SMIRNOFFBondCollection",
        "SMIRNOFFConstraintCollection",
        "SMIRNOFFElectrostaticsCollection",
        "SMIRNOFFImproperTorsionCollection",
        "SMIRNOFFProperTorsionCollection",
        "SMIRNOFFvdWCollection",
        "SMIRNOFFVirtualSiteCollection",
    ):
        setattr(smirnoff_module, class_name, type(class_name, (FakeSMIRNOFFCollection,), {}))

    modules = {
        "openff": ModuleType("openff"),
        "openff.interchange": ModuleType("openff.interchange"),
        "openff.interchange.components": components_module,
        "openff.interchange.components.potentials": potentials_module,
        "openff.interchange.models": models_module,
        "openff.interchange.smirnoff": smirnoff_module,
        "openff.units": ModuleType("openff.units"),
        "openmm": ModuleType("openmm"),
    }
    modules["openff.units"].unit = unit_module
    modules["openmm"].unit = openmm_unit_module

    for module_name, module in modules.items():
        monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(plugins, "_SAMMD_METAL_SULFUR_LJ_COLLECTION", None)
    monkeypatch.setattr(plugins, "_INTERCHANGE_PLUGIN_COLLECTION_REGISTERED", False)
    return potentials_module


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


def test_lazy_collection_attribute_reports_missing_openff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plugins, "_SAMMD_METAL_SULFUR_LJ_COLLECTION", None)
    monkeypatch.setitem(sys.modules, "openff.interchange.components.potentials", None)

    with pytest.raises(AttributeError, match="missing"):
        plugins.__getattr__("missing")
    with pytest.raises(ImportError, match="OpenFF Interchange is required"):
        plugins.__getattr__("SAMMDMetalSulfurLJCollection")


@pytest.mark.parametrize(
    ("anchor_pairs", "message"),
    [
        (object(), "must be an iterable"),
        ([], "at least one"),
        ([(1, 2, 3)], "exactly two"),
        ([(-1, 2)], "non-negative integers"),
        ([(True, 2)], "not booleans"),
        ([(1, 1)], "self-pairs"),
        ([(1, 2), (2, 1)], "duplicate"),
    ],
)
def test_validate_anchor_pairs_rejects_invalid_inputs(anchor_pairs: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        plugins.validate_anchor_pairs(anchor_pairs)


def test_plugin_collection_helpers_work_with_fake_optional_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_openff_and_openmm(monkeypatch)

    collection_cls = plugins.__getattr__("SAMMDMetalSulfurLJCollection")
    cached_collection_cls = plugins.__getattr__("SAMMDMetalSulfurLJCollection")

    assert cached_collection_cls is collection_cls
    assert collection_cls.type == plugins.SAMMD_METAL_SULFUR_COLLECTION_TYPE
    assert collection_cls.expression == plugins.SAMMD_METAL_SULFUR_EXPRESSION
    assert collection_cls.allowed_parameter_handlers() == []
    assert collection_cls.supported_parameters() == ["id", "sigma", "epsilon", "charge_product"]
    assert collection_cls.potential_parameters() == ["sigma", "epsilon", "charge_product"]
    with pytest.raises(NotImplementedError, match="assigned by atom index"):
        collection_cls().store_matches()
    with pytest.raises(NotImplementedError, match="assigned by atom index"):
        collection_cls().store_potentials()

    collection = collection_cls.from_anchor_pairs(
        ((2, 3),),
        sigma_nm=0.3,
        epsilon_kcal_mol=1.5,
    )
    assert collection.to_summary()["sulfur_metal_pairs"] == [[2, 3]]
    assert [key.atom_indices for key in collection.key_map] == [(2, 3)]
    potential = next(iter(collection.potentials.values()))
    assert potential.parameters["sigma"].m_as("nanometer") == pytest.approx(0.3)
    assert potential.parameters["epsilon"].m_as("kilojoule_per_mole") == pytest.approx(
        1.5 * KCAL_TO_KJ
    )

    nonbonded_force = NonbondedForce()
    collection.modify_openmm_forces(
        interchange=None,
        system=FakeSystem([object(), nonbonded_force]),
        add_constrained_forces=False,
        constrained_pairs=set(),
        particle_map={2: 8},
    )
    sulfur_index, metal_index, charge_product, sigma, epsilon, kwargs = (
        nonbonded_force.exceptions[0]
    )
    assert (sulfur_index, metal_index) == (8, 3)
    assert charge_product.value_in_unit(FakeUnit("elementary_charge^2")) == pytest.approx(0.0)
    assert sigma.value_in_unit(FakeUnit("nanometer")) == pytest.approx(0.3)
    assert epsilon.value_in_unit(FakeUnit("kilojoule_per_mole")) == pytest.approx(
        1.5 * KCAL_TO_KJ
    )
    assert kwargs == {"replace": True}
    with pytest.raises(ValueError, match="NonbondedForce"):
        collection.modify_openmm_forces(None, FakeSystem([]), False, set(), {})
    with pytest.raises(AssertionError):
        collection_cls.check_openmm_requirements(False)
    collection_cls.check_openmm_requirements(True)
    with pytest.raises(ValueError, match="sigma_nm"):
        collection_cls.from_anchor_pairs(((2, 3),), sigma_nm=float("nan"))


def test_dependency_light_serialization_helpers_cover_conversion_branches() -> None:
    assert plugins._quantity_to_float(FakeQuantity(2, "nanometer"), "nanometer") == pytest.approx(2)
    assert plugins._quantity_to_float(ToOnlyQuantity(), "nanometer") == pytest.approx(3.25)
    assert plugins._quantity_to_float("4.5", "nanometer") == pytest.approx(4.5)
    assert plugins._map_particle_index(7, {7: 11}) == 11
    assert plugins._map_particle_index(7, {}) == 7


def test_register_interchange_plugin_collection_patches_validator_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    potentials_module = install_fake_openff_and_openmm(monkeypatch)

    plugins.register_interchange_plugin_collection()
    assert plugins._INTERCHANGE_PLUGIN_COLLECTION_REGISTERED is True
    assert potentials_module.validate_collections._sammd_accepts_metal_sulfur_plugin

    result = potentials_module.validate_collections(
        {
            "Bonds": {"bond": True},
            "SAMMDMetalSulfurLJ": {"plugin": True},
        },
        None,
        SimpleNamespace(mode="python"),
    )
    assert result["Bonds"] == {"class": "SMIRNOFFBondCollection", "data": {"bond": True}}
    assert result["SAMMDMetalSulfurLJ"] == {
        "class": "SAMMDMetalSulfurLJCollection",
        "data": {"plugin": True},
    }

    plugins.register_interchange_plugin_collection()
    monkeypatch.setattr(plugins, "_INTERCHANGE_PLUGIN_COLLECTION_REGISTERED", False)
    plugins.register_interchange_plugin_collection()
    assert plugins._INTERCHANGE_PLUGIN_COLLECTION_REGISTERED is True
    with pytest.raises(ValueError, match="not implemented"):
        plugins._validate_collections_with_sammd_plugins(
            {},
            None,
            SimpleNamespace(mode="json-unavailable"),
        )


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


def test_interchange_plugin_collection_reload_applies_openmm_exception() -> None:
    """Persisted Interchange JSON reloads and still applies metal-S exceptions."""

    pytest.importorskip("openff.interchange")
    pytest.importorskip("openff.toolkit")
    pytest.importorskip("openmm")

    from openff.interchange import Interchange
    from openff.toolkit import ForceField, Molecule
    from openff.units import unit
    from openmm import NonbondedForce
    from openmm import unit as openmm_unit

    plugins.register_interchange_plugin_collection()

    molecule = Molecule.from_smiles("C")
    molecule.generate_conformers(n_conformers=1)
    molecule.assign_partial_charges("zeros")
    topology = molecule.to_topology()
    topology.box_vectors = [[3, 0, 0], [0, 3, 0], [0, 0, 3]] * unit.nanometer
    force_field = ForceField("openff_unconstrained-2.2.1.offxml")
    interchange = Interchange.from_smirnoff(
        force_field,
        topology,
        positions=molecule.conformers[0],
        charge_from_molecules=[molecule],
    )
    collection = plugins.create_metal_sulfur_lj_collection(((0, 1),))
    interchange.collections[collection.type] = collection

    reloaded = Interchange.model_validate_json(interchange.model_dump_json())
    system = reloaded.to_openmm(combine_nonbonded_forces=True)

    nonbonded_force = next(
        force for force in system.getForces() if isinstance(force, NonbondedForce)
    )
    exceptions = [
        nonbonded_force.getExceptionParameters(index)
        for index in range(nonbonded_force.getNumExceptions())
    ]
    _, _, charge_product, sigma, epsilon = next(
        params for params in exceptions if {int(params[0]), int(params[1])} == {0, 1}
    )
    assert charge_product.value_in_unit(openmm_unit.elementary_charge**2) == pytest.approx(0.0)
    assert sigma.value_in_unit(openmm_unit.nanometer) == pytest.approx(METAL_SULFUR_SIGMA_NM)
    assert epsilon.value_in_unit(openmm_unit.kilocalorie_per_mole) == pytest.approx(
        METAL_SULFUR_EPSILON_KCAL_MOL
    )
