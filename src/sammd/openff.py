"""Lazy OpenFF Toolkit adapters for optional SAMMD parameterization steps."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib import import_module, resources
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

OPENFF_INSTALL_GUIDANCE = (
    "OpenFF Toolkit is an optional SAMMD backend dependency. Install and use the "
    "SAMMD science/pixi environment with OpenFF Toolkit available before calling "
    "OpenFF adapter utilities."
)

OPENFF_INTERCHANGE_INSTALL_GUIDANCE = (
    "OpenFF Interchange is an optional SAMMD backend dependency. Install and use the "
    "SAMMD science/pixi environment with OpenFF Interchange available before calling "
    "OpenFF backend construction utilities."
)


@dataclass(frozen=True)
class OpenFFBackendAvailability:
    """Structured optional-backend availability report without importing backends."""

    toolkit_available: bool
    interchange_available: bool
    guidance: str = OPENFF_INSTALL_GUIDANCE
    messages: tuple[str, ...] = ()

    @property
    def backend_available(self) -> bool:
        """Return whether the OpenFF Toolkit and Interchange modules are discoverable."""

        return self.toolkit_available and self.interchange_available


@dataclass(frozen=True)
class OpenFFParameterizationPlan:
    """Lightweight record of future OpenFF parameterization choices and targets."""

    small_molecule_force_field: str
    charge_model: str
    metal_force_field_type: str
    metal_force_field_resource: str
    force_field_inputs: tuple[Any, ...]
    nonbonded_cutoff: float
    output_targets: dict[str, str | None] = field(default_factory=dict)
    component_counts: dict[str, int] = field(default_factory=dict)
    molecule_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class OpenFFMoleculePreparationIssue:
    """Structured report for a config entry not converted to an OpenFF molecule."""

    section: str
    name: str
    reason: str


@dataclass(frozen=True)
class OpenFFMoleculePreparationResult:
    """OpenFF molecule preparation result with unsupported and skipped entries."""

    molecules: dict[str, list[Any]] = field(
        default_factory=lambda: {"sam": [], "solvent": [], "reactants": [], "salts": []}
    )
    skipped: list[OpenFFMoleculePreparationIssue] = field(default_factory=list)
    unsupported: list[OpenFFMoleculePreparationIssue] = field(default_factory=list)


def require_openff_toolkit() -> Any:
    """Import and return the optional OpenFF Toolkit module.

    Returns
    -------
    Any
        Imported ``openff.toolkit`` module.

    Raises
    ------
    ImportError
        If OpenFF Toolkit is not installed in the active environment.
    """

    try:
        return import_module("openff.toolkit")
    except ImportError as error:
        raise ImportError(OPENFF_INSTALL_GUIDANCE) from error


def require_openff_interchange() -> Any:
    """Import and return the optional OpenFF Interchange module."""

    try:
        return import_module("openff.interchange")
    except ImportError as error:
        raise ImportError(OPENFF_INTERCHANGE_INSTALL_GUIDANCE) from error


def check_openff_backend_availability() -> OpenFFBackendAvailability:
    """Check optional OpenFF backend module availability without importing it."""

    toolkit_available = importlib_util.find_spec("openff.toolkit") is not None
    interchange_available = importlib_util.find_spec("openff.interchange") is not None
    messages: list[str] = []
    if not toolkit_available:
        messages.append("OpenFF Toolkit is not installed or not discoverable.")
    if not interchange_available:
        messages.append("OpenFF Interchange is not installed or not discoverable.")
    if messages:
        messages.append(OPENFF_INSTALL_GUIDANCE)
    return OpenFFBackendAvailability(
        toolkit_available=toolkit_available,
        interchange_available=interchange_available,
        messages=tuple(messages),
    )


def interface_fcc_metal_offxml_resource() -> resources.abc.Traversable:
    """Return the packaged INTERFACE Fcc metal OFFXML resource.

    Returns
    -------
    importlib.resources.abc.Traversable
        Traversable resource for ``interface_fcc_metals.offxml``.
    """

    return resources.files("sammd.data").joinpath("interface_fcc_metals.offxml")


def force_field_inputs_from_config(config: Any) -> tuple[Any, ...]:
    """Return inspectable OpenFF force-field inputs from config without OpenFF imports."""

    parameterization = config.parameterization
    inputs: list[Any] = [parameterization.small_molecule_force_field]
    metal_force_field = parameterization.metal_force_field
    resource_name = metal_force_field.resource
    if metal_force_field.type == "interface" and resource_name == "interface_fcc_metals.offxml":
        inputs.append(interface_fcc_metal_offxml_resource())
    else:
        inputs.append(resource_name)
    return tuple(inputs)


def parameterization_plan_from_config(config: Any) -> OpenFFParameterizationPlan:
    """Create a lightweight OpenFF parameterization plan from validated config."""

    from sammd.io import plan_output_paths

    output_paths = plan_output_paths(config, base_dir=config.outputs.directory)
    return _parameterization_plan(
        config,
        output_paths=output_paths,
        component_counts={
            "sam": len(tuple(_iter_config_items(config.sam, "components"))),
            "solvent": len(tuple(_iter_config_items(config.solvent, "components"))),
            "reactants": len(getattr(config, "reactants", ())),
            "salts": len(getattr(config, "salts", ())),
        },
    )


def parameterization_plan_from_build_plan(plan: Any) -> OpenFFParameterizationPlan:
    """Create a lightweight OpenFF parameterization plan from a SAMMD build plan."""

    component_counts = {
        "sam_components": len(tuple(_iter_config_items(plan.config.sam, "components"))),
        "sam_placements": len(plan.sam_placements.placements),
        "solvent_components": len(plan.solution.solvent_components),
        "reactants": len(plan.solution.reactants),
        "salts": len(plan.solution.salts),
    }
    return _parameterization_plan(
        plan.config,
        output_paths=plan.output_paths,
        component_counts=component_counts,
        molecule_counts=dict(plan.solution.molecule_counts),
    )


def molecule_from_smiles(
    smiles: str,
    name: str | None = None,
    n_conformers: int = 1,
    allow_undefined_stereo: bool = False,
    **from_smiles_kwargs: Any,
) -> Any:
    """Create an OpenFF molecule from SMILES with optional conformer generation.

    Parameters
    ----------
    smiles
        Molecular SMILES string accepted by OpenFF Toolkit.
    name
        Optional molecule name assigned to the returned OpenFF molecule.
    n_conformers
        Number of conformers to request. Use zero to skip conformer generation.
    allow_undefined_stereo
        Whether OpenFF should allow undefined stereochemistry in the input SMILES.
    **from_smiles_kwargs
        Additional keyword arguments forwarded to ``Molecule.from_smiles``.

    Returns
    -------
    Any
        OpenFF ``Molecule`` instance.
    """

    if n_conformers < 0:
        msg = "n_conformers must be non-negative"
        raise ValueError(msg)

    toolkit = require_openff_toolkit()
    molecule = toolkit.Molecule.from_smiles(
        smiles,
        allow_undefined_stereo=allow_undefined_stereo,
        **from_smiles_kwargs,
    )
    if name is not None:
        molecule.name = name
    if n_conformers:
        molecule.generate_conformers(n_conformers=n_conformers)
    return molecule


def molecules_from_config(
    config: Any,
    n_conformers: int = 1,
    allow_undefined_stereo: bool = False,
) -> OpenFFMoleculePreparationResult:
    """Prepare OpenFF molecules for supported molecule-bearing config sections.

    SAM components, solvent components with explicit SMILES, reactants, and salt ions are
    converted. Solvent components without SMILES are reported as unsupported unless they are
    named water, which is treated as a skipped built-in solvent model.

    Parameters
    ----------
    config
        SAMMD configuration-like object with ``sam``, ``solvent``, and ``reactants`` attributes.
    n_conformers
        Number of conformers to request for each generated molecule.
    allow_undefined_stereo
        Whether OpenFF should allow undefined stereochemistry in each input SMILES.

    Returns
    -------
    OpenFFMoleculePreparationResult
        Generated molecules plus structured skipped and unsupported entries.
    """

    result = OpenFFMoleculePreparationResult()

    for component in _iter_config_items(config.sam, "components"):
        result.molecules["sam"].append(
            molecule_from_smiles(
                component.smiles,
                name=component.name,
                n_conformers=n_conformers,
                allow_undefined_stereo=allow_undefined_stereo,
            )
        )

    for component in _iter_config_items(config.solvent, "components"):
        smiles = getattr(component, "smiles", None)
        if smiles is None:
            reason = "water is handled by the configured water model"
            if component.name.lower() != "water":
                reason = "solvent component has no SMILES for OpenFF molecule creation"
                result.unsupported.append(
                    OpenFFMoleculePreparationIssue(
                        section="solvent",
                        name=component.name,
                        reason=reason,
                    )
                )
                continue
            result.skipped.append(
                OpenFFMoleculePreparationIssue(
                    section="solvent",
                    name=component.name,
                    reason=reason,
                )
            )
            continue
        result.molecules["solvent"].append(
            molecule_from_smiles(
                smiles,
                name=component.name,
                n_conformers=n_conformers,
                allow_undefined_stereo=allow_undefined_stereo,
            )
        )

    for reactant in getattr(config, "reactants", []):
        result.molecules["reactants"].append(
            molecule_from_smiles(
                reactant.smiles,
                name=reactant.name,
                n_conformers=n_conformers,
                allow_undefined_stereo=allow_undefined_stereo,
            )
        )

    for salt in getattr(config, "salts", []):
        for ion_role in ("cation", "anion"):
            ion = getattr(salt, ion_role)
            result.molecules["salts"].append(
                molecule_from_smiles(
                    ion.smiles,
                    name=ion.name,
                    n_conformers=n_conformers,
                    allow_undefined_stereo=allow_undefined_stereo,
                )
            )

    return result


def load_force_field(
    base_force_fields: Iterable[str | Path] = ("openff-2.2.1.offxml",),
    include_interface_metals: bool = True,
    interface_offxml_path: str | Path | None = None,
) -> Any:
    """Load an OpenFF ``ForceField`` with optional SAMMD INTERFACE metal parameters.

    Parameters
    ----------
    base_force_fields
        Base OpenFF force field names or paths passed to ``ForceField`` first.
    include_interface_metals
        Whether to append the SAMMD INTERFACE Fcc metal OFFXML file.
    interface_offxml_path
        Optional explicit path to an INTERFACE metal OFFXML file. When omitted, the packaged
        resource is loaded with ``importlib.resources``.

    Returns
    -------
    Any
        OpenFF ``ForceField`` instance.
    """

    toolkit = require_openff_toolkit()
    force_field_inputs = [str(force_field) for force_field in base_force_fields]

    if include_interface_metals and interface_offxml_path is not None:
        force_field_inputs.append(str(interface_offxml_path))
        return toolkit.ForceField(*force_field_inputs)

    if include_interface_metals:
        resource = interface_fcc_metal_offxml_resource()
        with resources.as_file(resource) as offxml_path:
            force_field_inputs.append(str(offxml_path))
            return toolkit.ForceField(*force_field_inputs)

    return toolkit.ForceField(*force_field_inputs)


def _iter_config_items(section: Any, attribute: str) -> Iterable[Any]:
    """Yield config items from a section attribute when present."""

    return getattr(section, attribute, [])


def _parameterization_plan(
    config: Any,
    *,
    output_paths: Any,
    component_counts: dict[str, int],
    molecule_counts: dict[str, int] | None = None,
) -> OpenFFParameterizationPlan:
    """Build the shared inspectable parameterization plan record."""

    parameterization = config.parameterization
    metal_force_field = parameterization.metal_force_field
    return OpenFFParameterizationPlan(
        small_molecule_force_field=parameterization.small_molecule_force_field,
        charge_model=parameterization.charge_model,
        metal_force_field_type=metal_force_field.type,
        metal_force_field_resource=metal_force_field.resource,
        force_field_inputs=force_field_inputs_from_config(config),
        nonbonded_cutoff=parameterization.nonbonded_cutoff,
        output_targets={
            key: str(value) if value is not None else None
            for key, value in output_paths.__dict__.items()
        },
        component_counts=component_counts,
        molecule_counts={} if molecule_counts is None else molecule_counts,
    )
