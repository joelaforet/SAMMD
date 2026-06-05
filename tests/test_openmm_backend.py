"""Tests for private OpenMM backend helpers that avoid science dependencies."""

# ruff: noqa: N802

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sammd import _openmm_backend as backend
from sammd import _openmm_templates as templates


class FakeQuantity:
    """Minimal unit-bearing value for extracted fake OpenMM parameters."""

    def __init__(self, value: float) -> None:
        """Store a scalar value returned by ``value_in_unit``."""

        self.value = value

    def value_in_unit(self, unit: object) -> float:
        """Return the stored scalar independent of the requested fake unit."""

        return self.value


class FakeUnitValue:
    """Minimal fake unit supporting powers and division."""

    def __pow__(self, power: int) -> FakeUnitValue:
        """Return this unit for exponentiation."""

        return self

    def __truediv__(self, other: object) -> FakeUnitValue:
        """Return this unit for division."""

        return self


class FakeUnit:
    """Minimal unit object supporting arithmetic used by extractors."""

    elementary_charge = FakeUnitValue()
    nanometer = FakeUnitValue()
    kilojoule_per_mole = FakeUnitValue()
    radian = FakeUnitValue()


class FakeNonbondedForce:
    """Fake NonbondedForce with particles and one exception."""

    def getNumParticles(self) -> int:
        """Return fake particle count."""

        return 2

    def getParticleParameters(self, index: int) -> tuple[FakeQuantity, FakeQuantity, FakeQuantity]:
        """Return deterministic nonbonded parameters."""

        return FakeQuantity(0.1 + index), FakeQuantity(0.2 + index), FakeQuantity(0.3 + index)

    def getNumExceptions(self) -> int:
        """Return fake exception count."""

        return 1

    def getExceptionParameters(
        self,
        index: int,
    ) -> tuple[int, int, FakeQuantity, FakeQuantity, FakeQuantity]:
        """Return deterministic exception parameters."""

        return 0, 1, FakeQuantity(0.4), FakeQuantity(0.5), FakeQuantity(0.6)


class FakeBondForce:
    """Fake harmonic bond force."""

    def getNumBonds(self) -> int:
        """Return fake bond count."""

        return 1

    def getBondParameters(self, index: int) -> tuple[int, int, FakeQuantity, FakeQuantity]:
        """Return deterministic bond parameters."""

        return 0, 1, FakeQuantity(0.7), FakeQuantity(0.8)


class FakeAngleForce:
    """Fake harmonic angle force."""

    def getNumAngles(self) -> int:
        """Return fake angle count."""

        return 1

    def getAngleParameters(
        self,
        index: int,
    ) -> tuple[int, int, int, FakeQuantity, FakeQuantity]:
        """Return deterministic angle parameters."""

        return 0, 1, 2, FakeQuantity(0.9), FakeQuantity(1.0)


class FakeTorsionForce:
    """Fake periodic torsion force."""

    def getNumTorsions(self) -> int:
        """Return fake torsion count."""

        return 1

    def getTorsionParameters(
        self,
        index: int,
    ) -> tuple[int, int, int, int, int, FakeQuantity, FakeQuantity]:
        """Return deterministic torsion parameters."""

        return 0, 1, 2, 3, 2, FakeQuantity(1.1), FakeQuantity(1.2)


class FakeSystem:
    """Fake OpenMM System exposing forces and one constraint."""

    def __init__(self, forces: tuple[object, ...] | None = None) -> None:
        """Store fake forces for extraction."""

        self.forces = forces or (
            FakeNonbondedForce(),
            FakeBondForce(),
            FakeAngleForce(),
            FakeTorsionForce(),
        )

    def getNumForces(self) -> int:
        """Return fake force count."""

        return len(self.forces)

    def getForce(self, index: int) -> object:
        """Return one fake force."""

        return self.forces[index]

    def getNumConstraints(self) -> int:
        """Return fake constraint count."""

        return 1

    def getConstraintParameters(self, index: int) -> tuple[int, int, FakeQuantity]:
        """Return deterministic constraint parameters."""

        return 0, 1, FakeQuantity(0.13)


class FakeCoordinate:
    """Fake OpenFF coordinate scalar."""

    def __init__(self, value: float) -> None:
        """Store a coordinate value."""

        self.value = value

    def m_as(self, unit: str) -> float:
        """Return a coordinate in the requested fake unit."""

        assert unit == "nanometer"
        return self.value


def test_extract_openff_forces_reads_all_supported_force_types() -> None:
    """Fake OpenMM forces should be converted into private parameter dataclasses."""

    modules = fake_modules()

    nonbonded, exceptions, bonds, angles, torsions = templates.extract_openff_forces(
        modules,
        FakeSystem(),
    )

    assert nonbonded == ((0.1, 0.2, 0.3), (1.1, 1.2, 1.3))
    assert exceptions == (templates._ExceptionParameter(0, 1, 0.4, 0.5, 0.6),)
    assert bonds == (templates._BondParameter(0, 1, 0.7, 0.8),)
    assert angles == (templates._AngleParameter(0, 1, 2, 0.9, 1.0),)
    assert torsions == (templates._TorsionParameter(0, 1, 2, 3, 2, 1.1, 1.2),)


def test_extract_openff_forces_requires_nonbonded_force() -> None:
    """Templates without nonbonded parameters should be rejected."""

    with pytest.raises(RuntimeError, match="NonbondedForce"):
        templates.extract_openff_forces(fake_modules(), FakeSystem(forces=(FakeBondForce(),)))


def test_molecule_template_from_smiles_builds_private_template_without_seed_parameter() -> None:
    """Template construction should no longer expose an unused seed argument."""

    assert "seed" not in inspect.signature(templates.molecule_template_from_smiles).parameters

    openff_modules = fake_openff_modules()
    template = templates.molecule_template_from_smiles(
        fake_modules(),
        openff_modules,
        "CS",
        "methanethiol",
    )

    assert isinstance(template, templates._MoleculeTemplate)
    assert template.name == "methanethiol"
    assert template.bonds == ((0, 1),)
    assert template.positions_nm == ((0.0, 0.1, 0.2), (0.3, 0.4, 0.5))
    assert template.charge_model == templates._NAGL_CHARGE_MODEL
    assert template.force_field == templates._OPENFF_FORCE_FIELD
    assert openff_modules.last_molecule.generate_kwargs["n_conformers"] == 1
    assert openff_modules.last_molecule.generate_kwargs["toolkit_registry"] is not None


def test_solvent_position_boundary_uses_injected_placement() -> None:
    """Solvent placement can be tested without invoking Packmol."""

    calls = []

    def placement_fn(
        request: backend._SolventPlacementRequest,
    ) -> tuple[tuple[backend.Vector3, ...], ...]:
        """Record placement request and return precomputed solvent positions."""

        calls.append(request)
        return (((0.1, 0.2, 0.3),), ((0.4, 0.5, 0.6),))

    positions = backend.resolve_solvent_positions(
        topology="topology",
        solute_positions_nm=((0.0, 0.0, 0.0),),
        solvent_template=minimal_template(),
        solvent_count=2,
        box_dimensions_nm=(1.0, 2.0, 3.0),
        working_dir=Path("packmol"),
        solvent_placement_fn=placement_fn,
    )

    assert positions == (((0.1, 0.2, 0.3),), ((0.4, 0.5, 0.6),))
    assert calls[0].solvent_name == backend.SOLVENT_NAME
    assert calls[0].solvent_residue_name == backend.SOLVENT_RESIDUE_NAME
    assert calls[0].solvent_count == 2


def test_solvent_position_boundary_rejects_wrong_count() -> None:
    """Injected placement should still satisfy requested molecule count."""

    with pytest.raises(RuntimeError, match="returned 1 molecules, expected 2"):
        backend.resolve_solvent_positions(
            topology="topology",
            solute_positions_nm=(),
            solvent_template=minimal_template(),
            solvent_count=2,
            box_dimensions_nm=(1.0, 2.0, 3.0),
            working_dir=Path("packmol"),
            solvent_placement_fn=lambda request: (((0.1, 0.2, 0.3),),),
        )


def test_backend_named_constants_drive_pure_placement_helpers() -> None:
    """Pure placement helpers should expose former magic constants by name."""

    centers = backend.solvent_centers(3, (2.0, 4.0), 1.5)

    assert pytest.approx(0.75) == backend.REACTANT_SURFACE_CLEARANCE_NM
    assert pytest.approx(0.35) == backend.REACTANT_LATERAL_SPACING_NM
    assert pytest.approx(0.20) == backend.REACTANT_Y_FRACTION
    assert pytest.approx(0.20) == backend.REACTANT_Z_SPACING_NM
    assert backend.SAM_AZIMUTH_CANDIDATE_COUNT == 24
    assert pytest.approx(0.05) == backend.BOX_CLAMP_MARGIN_NM
    assert centers[0] == pytest.approx((-0.35, 0.8, 1.5))
    assert centers[1] == pytest.approx((0.0, -0.8, 1.7))
    assert centers[2] == pytest.approx((0.35, 0.8, 1.9))
    assert backend.clamp_to_box((-1.0, 0.5, 5.0), (1.0, 1.0, 1.0)) == pytest.approx(
        (0.05, 0.5, 0.95)
    )


def test_backend_geometry_helpers_transform_templates() -> None:
    """Pure SAM orientation helpers should work with private templates."""

    template = minimal_template(
        atoms=(
            templates._AtomTemplate("S1", "S", 0.0, 0.1, 0.2),
            templates._AtomTemplate("C1", "C", 0.0, 0.1, 0.2),
            templates._AtomTemplate("H1", "H", 0.0, 0.1, 0.2),
        ),
        positions=((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    )

    assert backend.sulfur_atom_index(template) == 0
    assert backend.terminal_heavy_axis_index(template, 0) == 1
    assert backend.center_template(template, (1.0, 1.0, 1.0))[0] == pytest.approx(
        (1.0, 2 / 3, 2 / 3)
    )
    oriented = backend.orient_template_by_anchor(
        template,
        anchor_index=0,
        axis_index=1,
        target_anchor_nm=(1.0, 2.0, 3.0),
        target_direction=(0.0, 0.0, 1.0),
    )

    assert oriented[0] == pytest.approx((1.0, 2.0, 3.0))
    assert oriented[1] == pytest.approx((1.0, 2.0, 4.0))
    assert backend.sam_azimuth_rad(1) > 0.0
    assert backend.select_sam_azimuth_rad(
        template,
        (1.0, 2.0, 3.0),
        (0.0, 0.0, 1.0),
        "top",
        [],
        2,
    ) == backend.sam_azimuth_rad(2)


def test_backend_template_validation_errors_are_clear() -> None:
    """Template orientation validation should reject missing or ambiguous anchors."""

    with pytest.raises(ValueError, match="exactly one sulfur"):
        backend.sulfur_atom_index(minimal_template())
    with pytest.raises(ValueError, match="heavy atom"):
        backend.terminal_heavy_axis_index(
            minimal_template(atoms=(templates._AtomTemplate("S1", "S", 0.0, 0.1, 0.2),)),
            0,
        )


def fake_modules() -> SimpleNamespace:
    """Return fake OpenMM module bundle for extraction tests."""

    return SimpleNamespace(
        openmm=SimpleNamespace(
            NonbondedForce=FakeNonbondedForce,
            HarmonicBondForce=FakeBondForce,
            HarmonicAngleForce=FakeAngleForce,
            PeriodicTorsionForce=FakeTorsionForce,
        ),
        unit=FakeUnit(),
    )


def fake_openff_modules() -> SimpleNamespace:
    """Return fake OpenFF module bundle for molecule-template tests."""

    modules = SimpleNamespace()

    class FakeMolecule:
        """Fake OpenFF molecule with atoms, bonds, charges, and conformers."""

        def __init__(self) -> None:
            """Create a minimal fake molecule."""

            self.name = ""
            self.atoms = [SimpleNamespace(symbol="C"), SimpleNamespace(symbol="S")]
            self.bonds = [SimpleNamespace(atom1_index=1, atom2_index=0)]
            self.n_atoms = 2
            self.conformers = [
                [
                    [FakeCoordinate(0.0), FakeCoordinate(0.1), FakeCoordinate(0.2)],
                    [FakeCoordinate(0.3), FakeCoordinate(0.4), FakeCoordinate(0.5)],
                ]
            ]
            self.generate_kwargs: dict[str, Any] = {}

        @classmethod
        def from_smiles(cls, smiles: str, allow_undefined_stereo: bool) -> FakeMolecule:
            """Return and record one fake molecule."""

            assert smiles == "CS"
            assert allow_undefined_stereo
            molecule = cls()
            modules.last_molecule = molecule
            return molecule

        def generate_conformers(self, **kwargs: Any) -> None:
            """Record conformer-generation keyword arguments."""

            self.generate_kwargs = kwargs

        def assign_partial_charges(self, model: str, toolkit_registry: object) -> None:
            """Validate charge assignment inputs."""

            assert model == templates._NAGL_CHARGE_MODEL
            assert toolkit_registry is modules.last_registry

        def to_topology(self) -> str:
            """Return a fake topology sentinel."""

            return "topology"

    class FakeForceField:
        """Fake OpenFF force field returning fake OpenMM systems."""

        def __init__(self, force_field: str) -> None:
            """Validate the requested force field."""

            assert force_field == templates._OPENFF_FORCE_FIELD

        def create_openmm_system(
            self,
            topology: str,
            charge_from_molecules: list[Any],
        ) -> FakeSystem:
            """Return a fake system for extracted parameters."""

            assert topology == "topology"
            assert charge_from_molecules == [modules.last_molecule]
            return FakeSystem()

    def registry(wrappers: list[object]) -> object:
        """Record toolkit wrappers and return a registry sentinel."""

        modules.last_wrappers = wrappers
        modules.last_registry = object()
        return modules.last_registry

    modules.Molecule = FakeMolecule
    modules.ForceField = FakeForceField
    modules.ToolkitRegistry = registry
    modules.NAGLToolkitWrapper = object
    modules.RDKitToolkitWrapper = object
    modules.last_molecule = None
    modules.last_registry = None
    return modules


def minimal_template(
    *,
    atoms: tuple[templates._AtomTemplate, ...] | None = None,
    positions: tuple[backend.Vector3, ...] | None = None,
) -> templates._MoleculeTemplate:
    """Return a private molecule template for pure backend helper tests."""

    atoms = atoms or (templates._AtomTemplate("C1", "C", 0.0, 0.1, 0.2),)
    positions = positions or ((0.0, 0.0, 0.0),)
    return templates._MoleculeTemplate(
        name="template",
        smiles="C",
        atoms=atoms,
        bonds=(),
        bond_parameters=(),
        angle_parameters=(),
        torsion_parameters=(),
        constraints=(),
        exception_parameters=(),
        positions_nm=positions,
        charge_model=templates._NAGL_CHARGE_MODEL,
        force_field=templates._OPENFF_FORCE_FIELD,
    )
