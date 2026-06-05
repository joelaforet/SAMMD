"""Private OpenMM smoke molecule templates and OpenFF extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sammd.geometry import Vector3

_OPENFF_FORCE_FIELD = "openff-2.2.1.offxml"
_NAGL_CHARGE_MODEL = "openff-gnn-am1bcc-1.0.0.pt"

@dataclass(frozen=True)
class _BondParameter:
    """OpenFF-derived harmonic bond parameter."""

    atom1: int
    atom2: int
    length_nm: float
    k_kj_mol_nm2: float


@dataclass(frozen=True)
class _AngleParameter:
    """OpenFF-derived harmonic angle parameter."""

    atom1: int
    atom2: int
    atom3: int
    angle_rad: float
    k_kj_mol_rad2: float


@dataclass(frozen=True)
class _TorsionParameter:
    """OpenFF-derived periodic torsion parameter."""

    atom1: int
    atom2: int
    atom3: int
    atom4: int
    periodicity: int
    phase_rad: float
    k_kj_mol: float


@dataclass(frozen=True)
class _ConstraintParameter:
    """OpenFF-derived distance constraint."""

    atom1: int
    atom2: int
    distance_nm: float


@dataclass(frozen=True)
class _ExceptionParameter:
    """OpenFF-derived intramolecular nonbonded exception."""

    atom1: int
    atom2: int
    chargeprod_e2: float
    sigma_nm: float
    epsilon_kj_mol: float


@dataclass(frozen=True)
class _AtomTemplate:
    """Per-atom template values used for direct OpenMM construction."""

    name: str
    element: str
    charge_e: float
    sigma_nm: float
    epsilon_kj_mol: float


@dataclass(frozen=True)
class _MoleculeTemplate:
    """Small molecule coordinates and topology from RDKit."""

    name: str
    smiles: str
    atoms: tuple[_AtomTemplate, ...]
    bonds: tuple[tuple[int, int], ...]
    bond_parameters: tuple[_BondParameter, ...]
    angle_parameters: tuple[_AngleParameter, ...]
    torsion_parameters: tuple[_TorsionParameter, ...]
    constraints: tuple[_ConstraintParameter, ...]
    exception_parameters: tuple[_ExceptionParameter, ...]
    positions_nm: tuple[Vector3, ...]
    charge_model: str
    force_field: str


def require_openmm_modules() -> Any:
    """Import OpenMM lazily with guidance for pixi science users."""

    try:
        import openmm
        from openmm import app, unit
    except ImportError as error:
        msg = "OpenMM is required; run this through `pixi run -e science real-system-smoke`."
        raise SystemExit(msg) from error
    return type("OpenMMModules", (), {"openmm": openmm, "app": app, "unit": unit})


def require_openff_modules() -> Any:
    """Import OpenFF Toolkit/NAGL lazily for proper small-molecule parameters."""

    try:
        from openff.toolkit import ForceField, Molecule, ToolkitRegistry
        from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper
        from openff.toolkit.utils.rdkit_wrapper import RDKitToolkitWrapper
    except ImportError as error:
        msg = (
            "OpenFF Toolkit with NAGL support is required; run this through "
            "`pixi run -e science real-system-smoke`."
        )
        raise SystemExit(msg) from error
    return type(
        "OpenFFModules",
        (),
        {
            "ForceField": ForceField,
            "Molecule": Molecule,
            "ToolkitRegistry": ToolkitRegistry,
            "NAGLToolkitWrapper": NAGLToolkitWrapper,
            "RDKitToolkitWrapper": RDKitToolkitWrapper,
        },
    )

def molecule_template_from_smiles(
    modules: Any,
    openff_modules: Any,
    smiles: str,
    name: str,
    seed: int,
) -> _MoleculeTemplate:
    """Build an OpenFF/NAGL-parameterized molecule template."""

    unit = modules.unit
    openff_molecule = openff_modules.Molecule.from_smiles(
        smiles,
        allow_undefined_stereo=True,
    )
    openff_molecule.name = name
    openff_molecule.generate_conformers(
        n_conformers=1,
        toolkit_registry=openff_toolkit_registry(openff_modules),
    )
    openff_molecule.assign_partial_charges(
        _NAGL_CHARGE_MODEL,
        toolkit_registry=openff_toolkit_registry(openff_modules),
    )
    force_field = openff_modules.ForceField(_OPENFF_FORCE_FIELD)
    openmm_system = force_field.create_openmm_system(
        openff_molecule.to_topology(),
        charge_from_molecules=[openff_molecule],
    )
    (
        nonbonded,
        exception_parameters,
        bond_parameters,
        angle_parameters,
        torsion_parameters,
    ) = extract_openff_forces(modules, openmm_system)
    constraints = tuple(
        _ConstraintParameter(
            atom1=openmm_system.get_ConstraintParameters(index)[0],
            atom2=openmm_system.get_ConstraintParameters(index)[1],
            distance_nm=openmm_system.get_ConstraintParameters(index)[2].value_in_unit(unit.nanometer),
        )
        for index in range(openmm_system.getNumConstraints())
    )
    atoms = tuple(
        _AtomTemplate(
            name=f"{atom.symbol}{index + 1}",
            element=atom.symbol,
            charge_e=charge,
            sigma_nm=sigma_nm,
            epsilon_kj_mol=epsilon_kj_mol,
        )
        for index, (atom, (charge, sigma_nm, epsilon_kj_mol)) in enumerate(
            zip(openff_molecule.atoms, nonbonded, strict=True)
        )
    )
    conformer = openff_molecule.conformers[0]
    positions = tuple(
        (
            conformer[index][0].m_as("nanometer"),
            conformer[index][1].m_as("nanometer"),
            conformer[index][2].m_as("nanometer"),
        )
        for index in range(openff_molecule.n_atoms)
    )
    bonds = tuple(
        tuple(sorted((bond.atom1_index, bond.atom2_index))) for bond in openff_molecule.bonds
    )
    return _MoleculeTemplate(
        name=name,
        smiles=smiles,
        atoms=atoms,
        bonds=bonds,
        bond_parameters=bond_parameters,
        angle_parameters=angle_parameters,
        torsion_parameters=torsion_parameters,
        constraints=constraints,
        exception_parameters=exception_parameters,
        positions_nm=positions,
        charge_model=_NAGL_CHARGE_MODEL,
        force_field=_OPENFF_FORCE_FIELD,
    )


def openff_toolkit_registry(openff_modules: Any) -> Any:
    """Return ToolkitRegistry with NAGL enabled for partial charges."""

    return openff_modules.ToolkitRegistry(
        [openff_modules.NAGLToolkitWrapper(), openff_modules.RDKitToolkitWrapper()]
    )


def extract_openff_forces(
    modules: Any,
    system: Any,
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[_ExceptionParameter, ...],
    tuple[_BondParameter, ...],
    tuple[_AngleParameter, ...],
    tuple[_TorsionParameter, ...],
]:
    """Extract OpenFF-generated nonbonded and bonded parameters from an OpenMM System."""

    openmm = modules.openmm
    unit = modules.unit
    nonbonded: list[tuple[float, float, float]] = []
    exceptions: list[_ExceptionParameter] = []
    bonds: list[_BondParameter] = []
    angles: list[_AngleParameter] = []
    torsions: list[_TorsionParameter] = []
    for force_index in range(system.getNumForces()):
        force = system.getForce(force_index)
        if isinstance(force, openmm.NonbondedForce):
            nonbonded = [
                (
                    charge.value_in_unit(unit.elementary_charge),
                    sigma.value_in_unit(unit.nanometer),
                    epsilon.value_in_unit(unit.kilojoule_per_mole),
                )
                for charge, sigma, epsilon in (
                    force.getParticleParameters(index)
                    for index in range(force.getNumParticles())
                )
            ]
            exceptions = [
                _ExceptionParameter(
                    atom1=atom1,
                    atom2=atom2,
                    chargeprod_e2=chargeprod.value_in_unit(
                        unit.elementary_charge**2
                    ),
                    sigma_nm=sigma.value_in_unit(unit.nanometer),
                    epsilon_kj_mol=epsilon.value_in_unit(unit.kilojoule_per_mole),
                )
                for atom1, atom2, chargeprod, sigma, epsilon in (
                    force.get_ExceptionParameters(index)
                    for index in range(force.getNumExceptions())
                )
            ]
        elif isinstance(force, openmm.HarmonicBondForce):
            bonds = [
                _BondParameter(
                    atom1=atom1,
                    atom2=atom2,
                    length_nm=length.value_in_unit(unit.nanometer),
                    k_kj_mol_nm2=k.value_in_unit(
                        unit.kilojoule_per_mole / unit.nanometer**2
                    ),
                )
                for atom1, atom2, length, k in (
                    force.get_BondParameters(index) for index in range(force.getNumBonds())
                )
            ]
        elif isinstance(force, openmm.HarmonicAngleForce):
            angles = [
                _AngleParameter(
                    atom1=atom1,
                    atom2=atom2,
                    atom3=atom3,
                    angle_rad=angle.value_in_unit(unit.radian),
                    k_kj_mol_rad2=k.value_in_unit(
                        unit.kilojoule_per_mole / unit.radian**2
                    ),
                )
                for atom1, atom2, atom3, angle, k in (
                    force.get_AngleParameters(index) for index in range(force.getNumAngles())
                )
            ]
        elif isinstance(force, openmm.PeriodicTorsionForce):
            torsions = [
                _TorsionParameter(
                    atom1=atom1,
                    atom2=atom2,
                    atom3=atom3,
                    atom4=atom4,
                    periodicity=periodicity,
                    phase_rad=phase.value_in_unit(unit.radian),
                    k_kj_mol=k.value_in_unit(unit.kilojoule_per_mole),
                )
                for atom1, atom2, atom3, atom4, periodicity, phase, k in (
                    force.get_TorsionParameters(index)
                    for index in range(force.getNumTorsions())
                )
            ]
    if not nonbonded:
        raise RuntimeError("OpenFF template did not contain a NonbondedForce")
    return tuple(nonbonded), tuple(exceptions), tuple(bonds), tuple(angles), tuple(torsions)
