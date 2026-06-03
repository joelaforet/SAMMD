"""Lazy OpenFF Toolkit adapters for optional SAMMD parameterization steps."""

from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module, resources
from pathlib import Path
from typing import Any

OPENFF_INSTALL_GUIDANCE = (
    "OpenFF Toolkit is an optional SAMMD backend dependency. Install and use the "
    "SAMMD science/pixi environment with OpenFF Toolkit available before calling "
    "OpenFF adapter utilities."
)


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


def interface_fcc_metal_offxml_resource() -> resources.abc.Traversable:
    """Return the packaged INTERFACE Fcc metal OFFXML resource.

    Returns
    -------
    importlib.resources.abc.Traversable
        Traversable resource for ``interface_fcc_metals.offxml``.
    """

    return resources.files("sammd.data").joinpath("interface_fcc_metals.offxml")


def molecule_from_smiles(
    smiles: str,
    name: str | None = None,
    n_conformers: int = 1,
    allow_undefined_stereo: bool = True,
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
    allow_undefined_stereo: bool = True,
) -> dict[str, list[Any]]:
    """Prepare OpenFF molecules for supported molecule-bearing config sections.

    SAM components, non-water solvent components with explicit SMILES, and reactants are
    converted. Water is represented by the configured water model rather than an OpenFF small
    molecule here, and salt ions are intentionally skipped because the MVP salt config records ion
    labels rather than chemically complete SMILES.

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
    dict[str, list[Any]]
        Molecules grouped as ``sam``, ``solvent``, ``reactants``, and ``salts``.
    """

    molecules: dict[str, list[Any]] = {"sam": [], "solvent": [], "reactants": [], "salts": []}

    for component in _iter_config_items(config.sam, "components"):
        molecules["sam"].append(
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
            continue
        molecules["solvent"].append(
            molecule_from_smiles(
                smiles,
                name=component.name,
                n_conformers=n_conformers,
                allow_undefined_stereo=allow_undefined_stereo,
            )
        )

    for reactant in getattr(config, "reactants", []):
        molecules["reactants"].append(
            molecule_from_smiles(
                reactant.smiles,
                name=reactant.name,
                n_conformers=n_conformers,
                allow_undefined_stereo=allow_undefined_stereo,
            )
        )

    return molecules


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
