"""Private direct OpenMM construction backend for the smoke workflow."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sammd._openmm_templates import _MoleculeTemplate
from sammd.forcefields import get_fcc_metal_parameters
from sammd.geometry import (
    Vector3,
    add_vectors,
    centroid,
    distance,
    matvec,
    rotate_about_axis,
    rotation_matrix,
    scale_vector,
    subtract_vectors,
)
from sammd.openmm_runtime import AnchorScalingMetadata
from sammd.packmol import pack_solution_with_packmol
from sammd.topology import ComponentResidueAssigner, ResidueIdentity, get_or_add_chain
from sammd.workflow import (
    CANONICAL_SMOKE_SOLVENT_NAME,
    CANONICAL_SMOKE_SOLVENT_RESIDUE_NAME,
)

KCAL_TO_KJ = 4.184
SOLVENT_NAME = CANONICAL_SMOKE_SOLVENT_NAME
SOLVENT_RESIDUE_NAME = CANONICAL_SMOKE_SOLVENT_RESIDUE_NAME
SAM_TAIL_CLEARANCE_NM = 0.95

@dataclass(frozen=True)
class _SmokeBuild:
    """Constructed OpenMM objects plus index metadata for stability checks."""

    topology: Any
    system: Any
    positions_nm: tuple[Vector3, ...]
    positions_quantity: Any
    pd_indices: tuple[int, ...]
    sulfur_indices: tuple[int, ...]
    sulfur_reference_positions_nm: tuple[Vector3, ...]
    anchor_pairs: tuple[tuple[int, int], ...]
    anchor_scaling: Any
    solvent_count: int
    reactant_count: int
    sam_count: int
    platform_dimensions_nm: Vector3
    component_chain_ranges: dict[str, dict[str, object]]
    ensemble: str
    pressure_bar: float
    temperature_k: float


def build_openmm_smoke_system(
    modules: Any,
    plan: Any,
    sam_template: _MoleculeTemplate,
    reactant_template: _MoleculeTemplate,
    solvent_template: _MoleculeTemplate,
    *,
    solvent_count: int,
    reactant_count: int,
    sulfur_height_nm: float,
    solvent_padding_nm: float,
    packmol_working_dir: Path,
    pressure_bar: float,
    temperature_k: float,
    pd_s_sigma_nm: float,
    pd_s_epsilon_kcal_mol: float,
) -> _SmokeBuild:
    """Build the direct OpenMM topology/system/positions for the smoke run."""

    openmm = modules.openmm
    app = modules.app
    unit = modules.unit
    topology = app.Topology()
    system = openmm.System()
    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.PME)
    nonbonded.setCutoffDistance(plan.config.simulation.nonbonded_cutoff_nm * unit.nanometer)
    nonbonded.setUseSwitchingFunction(True)
    nonbonded.setSwitchingDistance(
        0.9 * plan.config.simulation.nonbonded_cutoff_nm * unit.nanometer
    )
    nonbonded.setUseDispersionCorrection(True)
    bond_force = openmm.HarmonicBondForce()
    angle_force = openmm.HarmonicAngleForce()
    torsion_force = openmm.PeriodicTorsionForce()

    atom_handles: list[Any] = []
    positions_nm: list[Vector3] = []
    all_bonds: list[tuple[int, int]] = []
    pd_indices: list[int] = []
    sulfur_indices: list[int] = []
    sulfur_references: list[Vector3] = []
    anchor_pairs: list[tuple[int, int]] = []
    residue_assigner = ComponentResidueAssigner()
    chain_cache: dict[str, Any] = {}

    box_dimensions_nm = derive_box_dimensions(plan, solvent_padding_nm)
    shift_nm = tuple(dimension / 2.0 for dimension in box_dimensions_nm)
    set_periodic_box(modules, topology, system, box_dimensions_nm)
    pd_identities = residue_assigner.allocate(
        "palladium_slab",
        "PD",
        len(plan.slab.positions_nm),
    )
    sam_identities = residue_assigner.allocate(
        "propanethiolate_sam",
        "PTL",
        len(plan.sam_placements.placements),
    )

    add_pd_slab(
        modules,
        topology,
        system,
        nonbonded,
        atom_handles,
        chain_cache,
        positions_nm,
        pd_indices,
        plan,
        shift_nm,
        pd_identities,
    )
    add_sam_layer(
        modules,
        topology,
        system,
        nonbonded,
        bond_force,
        angle_force,
        torsion_force,
        atom_handles,
        chain_cache,
        positions_nm,
        all_bonds,
        sulfur_indices,
        sulfur_references,
        anchor_pairs,
        plan,
        sam_template,
        shift_nm,
        sulfur_height_nm,
        sam_identities,
    )
    reactant_identities = residue_assigner.allocate(
        "cinnamaldehyde",
        "CIN",
        reactant_count,
    )
    reactant_positions = place_reactants_above_surface(
        plan,
        reactant_template,
        reactant_count,
        shift_nm,
        box_dimensions_nm,
    )
    add_reactants(
        modules,
        topology,
        system,
        nonbonded,
        bond_force,
        angle_force,
        torsion_force,
        atom_handles,
        chain_cache,
        positions_nm,
        all_bonds,
        reactant_template,
        reactant_positions,
        reactant_identities,
    )
    packed_solution = pack_solution_with_packmol(
        topology=topology,
        solute_positions_nm=tuple(positions_nm),
        solvent_template=solvent_template,
        solvent_name=SOLVENT_NAME,
        solvent_residue_name=SOLVENT_RESIDUE_NAME,
        solvent_count=solvent_count,
        box_dimensions_nm=box_dimensions_nm,
        working_dir=packmol_working_dir,
    )
    solvent_identities = residue_assigner.allocate(
        SOLVENT_NAME,
        SOLVENT_RESIDUE_NAME,
        solvent_count,
    )
    placed_solvent = add_solvent_molecules(
        modules,
        topology,
        system,
        nonbonded,
        bond_force,
        angle_force,
        torsion_force,
        atom_handles,
        chain_cache,
        positions_nm,
        all_bonds,
        solvent_template,
        packed_solution.solvent_positions_nm,
        solvent_identities,
    )
    system.addForce(nonbonded)
    system.addForce(bond_force)
    system.addForce(angle_force)
    system.addForce(torsion_force)
    system.addForce(openmm.CMMotionRemover())

    anchor_scaling = add_sulfur_metal_lj_exceptions(
        nonbonded,
        tuple(anchor_pairs),
        sigma_nm=pd_s_sigma_nm,
        epsilon_kcal_mol=pd_s_epsilon_kcal_mol,
        unit=unit,
    )
    positions_quantity = unit.Quantity(
        [openmm.Vec3(*position) for position in positions_nm],
        unit.nanometer,
    )

    return _SmokeBuild(
        topology=topology,
        system=system,
        positions_nm=tuple(positions_nm),
        positions_quantity=positions_quantity,
        pd_indices=tuple(pd_indices),
        sulfur_indices=tuple(sulfur_indices),
        sulfur_reference_positions_nm=tuple(sulfur_references),
        anchor_pairs=tuple(anchor_pairs),
        anchor_scaling=anchor_scaling,
        solvent_count=placed_solvent,
        reactant_count=reactant_count,
        sam_count=len(plan.sam_placements.placements),
        platform_dimensions_nm=box_dimensions_nm,
        component_chain_ranges=residue_assigner.component_ranges,
        ensemble="NVT",
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
    )


def derive_box_dimensions(plan: Any, solvent_padding_nm: float) -> Vector3:
    """Return periodic box lengths for slab, SAM, and two solvent regions."""

    return (
        plan.slab.lateral_size_nm[0],
        plan.slab.lateral_size_nm[1],
        plan.slab.slab_extent_nm[2] + 2.0 * (SAM_TAIL_CLEARANCE_NM + solvent_padding_nm),
    )


def set_periodic_box(modules: Any, topology: Any, system: Any, dimensions_nm: Vector3) -> None:
    """Apply orthorhombic periodic box vectors to topology and system."""

    openmm = modules.openmm
    unit = modules.unit
    vectors = (
        openmm.Vec3(dimensions_nm[0], 0.0, 0.0),
        openmm.Vec3(0.0, dimensions_nm[1], 0.0),
        openmm.Vec3(0.0, 0.0, dimensions_nm[2]),
    )
    topology.setPeriodicBoxVectors(vectors)
    system.setDefaultPeriodicBoxVectors(*(vector * unit.nanometer for vector in vectors))


def add_sulfur_metal_lj_exceptions(
    nonbonded: Any,
    pairs: tuple[tuple[int, int], ...],
    *,
    sigma_nm: float,
    epsilon_kcal_mol: float,
    unit: Any,
) -> AnchorScalingMetadata:
    """Override selected Pd-S pairs with literature-style LJ chemisorption mimic."""

    epsilon_kj_mol = epsilon_kcal_mol * KCAL_TO_KJ
    sigmas = []
    epsilons = []
    for sulfur_index, metal_index in pairs:
        nonbonded.addException(
            sulfur_index,
            metal_index,
            0.0 * unit.elementary_charge**2,
            sigma_nm * unit.nanometer,
            epsilon_kj_mol * unit.kilojoule_per_mole,
            replace=True,
        )
        sigmas.append(sigma_nm)
        epsilons.append(epsilon_kj_mol)
    return AnchorScalingMetadata(
        pairs_requested=len(pairs),
        pairs_added=len(pairs),
        force_added=False,
        scale_factor=1.0,
        force_index=None,
        sigma_nm=tuple(sigmas),
        epsilon_delta_kj_mol=tuple(epsilons),
    )


def add_pd_slab(
    modules: Any,
    topology: Any,
    system: Any,
    nonbonded: Any,
    atom_handles: list[Any],
    chain_cache: dict[str, Any],
    positions_nm: list[Vector3],
    pd_indices: list[int],
    plan: Any,
    shift_nm: Vector3,
    residue_identities: tuple[ResidueIdentity, ...],
) -> None:
    """Add Pd atoms with CHARMM-INTERFACE LJ parameters."""

    unit = modules.unit
    pd_element = element_by_symbol(modules, "Pd")
    pd_parameters = get_fcc_metal_parameters("Pd")
    sigma_nm = pd_parameters.sigma_angstrom * 0.1
    epsilon_kj_mol = pd_parameters.openff_epsilon_kcal_mol * KCAL_TO_KJ
    for position, residue_identity in zip(plan.slab.positions_nm, residue_identities, strict=True):
        chain = get_or_add_chain(topology, chain_cache, residue_identity.chain_id)
        residue = topology.addResidue(
            residue_identity.residue_name,
            chain,
            id=str(residue_identity.residue_id),
        )
        atom = topology.addAtom("Pd", pd_element, residue)
        atom_handles.append(atom)
        system.addParticle(pd_element.mass)
        nonbonded.addParticle(
            0.0 * unit.elementary_charge,
            sigma_nm * unit.nanometer,
            epsilon_kj_mol * unit.kilojoule_per_mole,
        )
        pd_indices.append(len(positions_nm))
        positions_nm.append(add_vectors(position, shift_nm))


def add_sam_layer(
    modules: Any,
    topology: Any,
    system: Any,
    nonbonded: Any,
    bond_force: Any,
    angle_force: Any,
    torsion_force: Any,
    atom_handles: list[Any],
    chain_cache: dict[str, Any],
    positions_nm: list[Vector3],
    all_bonds: list[tuple[int, int]],
    sulfur_indices: list[int],
    sulfur_references: list[Vector3],
    anchor_pairs: list[tuple[int, int]],
    plan: Any,
    template: _MoleculeTemplate,
    shift_nm: Vector3,
    sulfur_height_nm: float,
    residue_identities: tuple[ResidueIdentity, ...],
) -> None:
    """Add all planned propanethiol SAM molecules."""

    sulfur_index = sulfur_atom_index(template)
    axis_index = terminal_heavy_axis_index(template, sulfur_index)
    placed_sam_atoms: list[tuple[Vector3, str, str]] = []
    for placement_index, (placement, residue_identity) in enumerate(
        zip(
            plan.sam_placements.placements,
            residue_identities,
            strict=True,
        )
    ):
        target_sulfur = add_vectors(
            add_vectors(placement.position_nm, scale_vector(placement.normal, sulfur_height_nm)),
            shift_nm,
        )
        transformed = orient_template_by_anchor(
            template,
            anchor_index=sulfur_index,
            axis_index=axis_index,
            target_anchor_nm=target_sulfur,
            target_direction=placement.normal,
            azimuth_rad=select_sam_azimuth_rad(
                template,
                target_sulfur,
                placement.normal,
                placement.side,
                placed_sam_atoms,
                placement_index,
            ),
        )
        placed_sam_atoms.extend(
            (position, atom.element, placement.side)
            for position, atom in zip(transformed, template.atoms, strict=True)
        )
        global_indices = add_template_molecule(
            modules,
            topology,
            system,
            nonbonded,
            bond_force,
            angle_force,
            torsion_force,
            atom_handles,
            chain_cache,
            positions_nm,
            all_bonds,
            template,
            transformed,
            residue_identity,
        )
        global_sulfur = global_indices[sulfur_index]
        sulfur_indices.append(global_sulfur)
        sulfur_references.append(target_sulfur)
        nearest_metals = placement.anchor_metadata["nearest_metal_atom_indices"]
        anchor_pairs.extend((global_sulfur, int(metal_index)) for metal_index in nearest_metals)


def add_reactants(
    modules: Any,
    topology: Any,
    system: Any,
    nonbonded: Any,
    bond_force: Any,
    angle_force: Any,
    torsion_force: Any,
    atom_handles: list[Any],
    chain_cache: dict[str, Any],
    positions_nm: list[Vector3],
    all_bonds: list[tuple[int, int]],
    template: _MoleculeTemplate,
    molecule_positions_nm: tuple[tuple[Vector3, ...], ...],
    residue_identities: tuple[ResidueIdentity, ...],
) -> None:
    """Add Packmol-placed cinnamaldehyde molecule(s)."""

    for transformed, residue_identity in zip(
        molecule_positions_nm,
        residue_identities,
        strict=True,
    ):
        add_template_molecule(
            modules,
            topology,
            system,
            nonbonded,
            bond_force,
            angle_force,
            torsion_force,
            atom_handles,
            chain_cache,
            positions_nm,
            all_bonds,
            template,
            transformed,
            residue_identity,
        )


def place_reactants_above_surface(
    plan: Any,
    template: _MoleculeTemplate,
    reactant_count: int,
    shift_nm: Vector3,
    box_dimensions_nm: Vector3,
) -> tuple[tuple[Vector3, ...], ...]:
    """Place reactant molecule(s) in the upper solvent before Packmol solvent packing."""

    top_z = plan.slab.top_z_nm + SAM_TAIL_CLEARANCE_NM + 0.75
    centers = solvent_centers(reactant_count, plan.slab.lateral_size_nm, top_z)
    return tuple(
        center_template(template, clamp_to_box(add_vectors(center, shift_nm), box_dimensions_nm))
        for center in centers
    )


def solvent_centers(
    count: int,
    lateral_size_nm: tuple[float, float],
    z_nm: float,
) -> tuple[Vector3, ...]:
    """Return deterministic reactant centers in the upper solvent region."""

    centers = []
    for index in range(count):
        offset = (index - (count - 1) / 2.0) * 0.35
        x = max(-0.35 * lateral_size_nm[0], min(0.35 * lateral_size_nm[0], offset))
        y = 0.20 * lateral_size_nm[1] * ((-1) ** index)
        centers.append((x, y, z_nm + 0.20 * index))
    return tuple(centers)


def add_solvent_molecules(
    modules: Any,
    topology: Any,
    system: Any,
    nonbonded: Any,
    bond_force: Any,
    angle_force: Any,
    torsion_force: Any,
    atom_handles: list[Any],
    chain_cache: dict[str, Any],
    positions_nm: list[Vector3],
    all_bonds: list[tuple[int, int]],
    template: _MoleculeTemplate,
    molecule_positions_nm: tuple[tuple[Vector3, ...], ...],
    residue_identities: tuple[ResidueIdentity, ...],
) -> int:
    """Add Packmol-placed OpenFF solvent molecules."""

    for solvent_positions, residue_identity in zip(
        molecule_positions_nm,
        residue_identities,
        strict=True,
    ):
        add_template_molecule(
            modules,
            topology,
            system,
            nonbonded,
            bond_force,
            angle_force,
            torsion_force,
            atom_handles,
            chain_cache,
            positions_nm,
            all_bonds,
            template,
            solvent_positions,
            residue_identity,
        )
    return len(molecule_positions_nm)


def add_template_molecule(
    modules: Any,
    topology: Any,
    system: Any,
    nonbonded: Any,
    bond_force: Any,
    angle_force: Any,
    torsion_force: Any,
    atom_handles: list[Any],
    chain_cache: dict[str, Any],
    positions_nm: list[Vector3],
    all_bonds: list[tuple[int, int]],
    template: _MoleculeTemplate,
    transformed_positions_nm: tuple[Vector3, ...],
    residue_identity: ResidueIdentity,
) -> tuple[int, ...]:
    """Add one RDKit-derived molecule template to topology and system."""

    unit = modules.unit
    chain = get_or_add_chain(topology, chain_cache, residue_identity.chain_id)
    residue = topology.addResidue(
        residue_identity.residue_name,
        chain,
        id=str(residue_identity.residue_id),
    )
    global_indices = []
    local_atoms = []
    for atom, position in zip(template.atoms, transformed_positions_nm, strict=True):
        element = element_by_symbol(modules, atom.element)
        atom_handle = topology.addAtom(atom.name, element, residue)
        atom_handles.append(atom_handle)
        local_atoms.append(atom_handle)
        global_indices.append(len(positions_nm))
        system.addParticle(element.mass)
        nonbonded.addParticle(
            atom.charge_e * unit.elementary_charge,
            atom.sigma_nm * unit.nanometer,
            atom.epsilon_kj_mol * unit.kilojoule_per_mole,
        )
        positions_nm.append(position)

    for exception in template.exception_parameters:
        nonbonded.addException(
            global_indices[exception.atom1],
            global_indices[exception.atom2],
            exception.chargeprod_e2 * unit.elementary_charge**2,
            exception.sigma_nm * unit.nanometer,
            exception.epsilon_kj_mol * unit.kilojoule_per_mole,
        )
    for local_i, local_j in template.bonds:
        global_i = global_indices[local_i]
        global_j = global_indices[local_j]
        topology.addBond(local_atoms[local_i], local_atoms[local_j])
        all_bonds.append((global_i, global_j))
    for constraint in template.constraints:
        system.addConstraint(
            global_indices[constraint.atom1],
            global_indices[constraint.atom2],
            constraint.distance_nm * unit.nanometer,
        )
    for bond in template.bond_parameters:
        bond_force.addBond(
            global_indices[bond.atom1],
            global_indices[bond.atom2],
            bond.length_nm * unit.nanometer,
            bond.k_kj_mol_nm2 * unit.kilojoule_per_mole / unit.nanometer**2,
        )
    for angle in template.angle_parameters:
        angle_force.addAngle(
            global_indices[angle.atom1],
            global_indices[angle.atom2],
            global_indices[angle.atom3],
            angle.angle_rad * unit.radian,
            angle.k_kj_mol_rad2 * unit.kilojoule_per_mole / unit.radian**2,
        )
    for torsion in template.torsion_parameters:
        torsion_force.addTorsion(
            global_indices[torsion.atom1],
            global_indices[torsion.atom2],
            global_indices[torsion.atom3],
            global_indices[torsion.atom4],
            torsion.periodicity,
            torsion.phase_rad * unit.radian,
            torsion.k_kj_mol * unit.kilojoule_per_mole,
        )
    return tuple(global_indices)


def element_by_symbol(modules: Any, symbol: str) -> Any:
    """Return an OpenMM app Element by symbol."""

    return modules.app.Element.getBySymbol(symbol)


def sulfur_atom_index(template: _MoleculeTemplate) -> int:
    """Return the sulfur atom index for a SAM template."""

    matches = [index for index, atom in enumerate(template.atoms) if atom.element == "S"]
    if len(matches) != 1:
        raise ValueError("SAM template must contain exactly one sulfur atom")
    return matches[0]


def terminal_heavy_axis_index(template: _MoleculeTemplate, anchor_index: int) -> int:
    """Return the heavy atom farthest from the SAM anchor for molecular-axis alignment."""

    heavy_indices = [
        index
        for index, atom in enumerate(template.atoms)
        if index != anchor_index and atom.element != "H"
    ]
    if not heavy_indices:
        raise ValueError("SAM template must contain a heavy atom beyond the anchor")
    return max(
        heavy_indices,
        key=lambda index: distance(
            template.positions_nm[anchor_index], template.positions_nm[index]
        ),
    )


def sam_azimuth_rad(placement_index: int) -> float:
    """Return a deterministic per-SAM rotation around the sulfur-anchor axis."""

    golden_angle_rad = math.pi * (3.0 - math.sqrt(5.0))
    return placement_index * golden_angle_rad


def select_sam_azimuth_rad(
    template: _MoleculeTemplate,
    target_anchor_nm: Vector3,
    target_direction: Vector3,
    side: str,
    placed_sam_atoms: list[tuple[Vector3, str, str]],
    placement_index: int,
) -> float:
    """Choose an around-axis SAM rotation that avoids prior same-side H clashes."""

    same_side_atoms = [
        (position, element)
        for position, element, placed_side in placed_sam_atoms
        if placed_side == side
    ]
    if not same_side_atoms:
        return sam_azimuth_rad(placement_index)

    candidate_count = 24
    candidates = [
        sam_azimuth_rad(placement_index) + 2.0 * math.pi * index / candidate_count
        for index in range(candidate_count)
    ]
    return max(
        candidates,
        key=lambda angle: score_sam_azimuth(
            template,
            target_anchor_nm,
            target_direction,
            angle,
            same_side_atoms,
        ),
    )


def score_sam_azimuth(
    template: _MoleculeTemplate,
    target_anchor_nm: Vector3,
    target_direction: Vector3,
    azimuth_rad: float,
    same_side_atoms: list[tuple[Vector3, str]],
) -> tuple[float, float]:
    """Score an azimuth by its closest H-involving inter-SAM contact."""

    candidate_atoms = [
        (
            add_vectors(
                target_anchor_nm,
                rotate_about_axis(position, target_direction, azimuth_rad),
            ),
            atom.element,
        )
        for position, atom in zip(
            anchor_relative_positions(template, target_direction), template.atoms, strict=True
        )
    ]
    closest_h_contact = math.inf
    closest_any_contact = math.inf
    for candidate_position, candidate_element in candidate_atoms:
        for placed_position, placed_element in same_side_atoms:
            contact = distance(candidate_position, placed_position)
            closest_any_contact = min(closest_any_contact, contact)
            if "H" in (candidate_element, placed_element):
                closest_h_contact = min(closest_h_contact, contact)
    return closest_h_contact, closest_any_contact


def anchor_relative_positions(
    template: _MoleculeTemplate,
    target_direction: Vector3,
) -> tuple[Vector3, ...]:
    """Return template coordinates relative to its sulfur anchor."""

    sulfur_index = sulfur_atom_index(template)
    sulfur_position = template.positions_nm[sulfur_index]
    axis_index = terminal_heavy_axis_index(template, sulfur_index)
    source_vector = subtract_vectors(template.positions_nm[axis_index], sulfur_position)
    rotation = rotation_matrix(source_vector, target_direction)
    return tuple(
        matvec(rotation, subtract_vectors(position, sulfur_position))
        for position in template.positions_nm
    )


def orient_template_by_anchor(
    template: _MoleculeTemplate,
    *,
    anchor_index: int,
    axis_index: int,
    target_anchor_nm: Vector3,
    target_direction: Vector3,
    azimuth_rad: float = 0.0,
) -> tuple[Vector3, ...]:
    """Rotate and translate a template so its anchor-to-axis vector points outward."""

    anchor = template.positions_nm[anchor_index]
    source_vector = subtract_vectors(template.positions_nm[axis_index], anchor)
    rotation = rotation_matrix(source_vector, target_direction)
    return tuple(
        add_vectors(
            target_anchor_nm,
            rotate_about_axis(
                matvec(rotation, subtract_vectors(position, anchor)),
                target_direction,
                azimuth_rad,
            ),
        )
        for position in template.positions_nm
    )


def center_template(template: _MoleculeTemplate, center_nm: Vector3) -> tuple[Vector3, ...]:
    """Translate a template so its coordinate centroid is at ``center_nm``."""

    current_center = centroid(template.positions_nm)
    return tuple(
        add_vectors(center_nm, subtract_vectors(position, current_center))
        for position in template.positions_nm
    )


def clamp_to_box(position: Vector3, box_dimensions_nm: Vector3) -> Vector3:
    """Clamp a coordinate just inside the orthorhombic box."""

    margin = 0.05
    return tuple(
        max(margin, min(length - margin, coordinate))
        for coordinate, length in zip(position, box_dimensions_nm, strict=True)
    )
