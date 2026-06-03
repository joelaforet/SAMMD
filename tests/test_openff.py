"""Tests for lazy optional OpenFF adapter utilities."""

from __future__ import annotations

import importlib
import sys

import pytest

from sammd.config import SAMMDConfig


def test_adapter_import_does_not_import_openff_eagerly() -> None:
    """Keep importing the adapter independent of optional OpenFF dependencies."""

    openff_modules_before = {
        name for name in sys.modules if name == "openff" or name.startswith("openff.")
    }

    importlib.import_module("sammd.openff")

    openff_modules_after = {
        name for name in sys.modules if name == "openff" or name.startswith("openff.")
    }

    assert openff_modules_after == openff_modules_before


def test_require_openff_toolkit_fails_with_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explain that the optional backend requires the science environment."""

    openff_adapter = importlib.import_module("sammd.openff")
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "openff.toolkit":
            raise ImportError("missing OpenFF")
        return real_import_module(name, package)

    monkeypatch.setattr(openff_adapter, "import_module", fake_import_module)

    with pytest.raises(ImportError, match="science/pixi environment"):
        openff_adapter.require_openff_toolkit()


def test_interface_fcc_metal_offxml_resource_exists() -> None:
    """Expose the packaged metal OFFXML resource without importing OpenFF."""

    openff_adapter = importlib.import_module("sammd.openff")
    resource = openff_adapter.interface_fcc_metal_offxml_resource()

    assert resource.name == "interface_fcc_metals.offxml"
    assert resource.is_file()


def test_molecule_from_smiles_requires_nonnegative_conformer_count() -> None:
    """Validate conformer count before importing optional backends."""

    openff_adapter = importlib.import_module("sammd.openff")

    with pytest.raises(ValueError, match="n_conformers must be non-negative"):
        openff_adapter.molecule_from_smiles("CCCS", n_conformers=-1)


def test_molecules_from_config_groups_supported_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    """Create molecules only for SAMs, explicit solvent SMILES, and reactants."""

    openff_adapter = importlib.import_module("sammd.openff")
    calls: list[tuple[str, str | None, int, bool]] = []

    def fake_molecule_from_smiles(
        smiles: str,
        name: str | None = None,
        n_conformers: int = 1,
        allow_undefined_stereo: bool = True,
    ) -> dict[str, str | None]:
        calls.append((smiles, name, n_conformers, allow_undefined_stereo))
        return {"smiles": smiles, "name": name}

    monkeypatch.setattr(openff_adapter, "molecule_from_smiles", fake_molecule_from_smiles)

    molecules = openff_adapter.molecules_from_config(SAMMDConfig(), n_conformers=0)

    assert molecules["sam"] == [{"smiles": "CCCS", "name": "propanethiol"}]
    assert molecules["solvent"] == []
    assert molecules["reactants"] == [
        {"smiles": "C1=CC=C(C=C1)/C=C/C=O", "name": "cinnamaldehyde"}
    ]
    assert molecules["salts"] == []
    assert calls == [
        ("CCCS", "propanethiol", 0, True),
        ("C1=CC=C(C=C1)/C=C/C=O", "cinnamaldehyde", 0, True),
    ]


def test_molecule_from_propanethiol_smiles_when_openff_available() -> None:
    """Optionally create a thiol OpenFF molecule."""

    pytest.importorskip("openff.toolkit")
    openff_adapter = importlib.import_module("sammd.openff")

    molecule = openff_adapter.molecule_from_smiles("CCCS", name="propanethiol")

    assert molecule.name == "propanethiol"
    assert molecule.n_atoms > 0


def test_molecule_from_cinnamaldehyde_smiles_when_openff_available() -> None:
    """Optionally create the default reactant OpenFF molecule."""

    pytest.importorskip("openff.toolkit")
    openff_adapter = importlib.import_module("sammd.openff")

    molecule = openff_adapter.molecule_from_smiles(
        "C1=CC=C(C=C1)/C=C/C=O",
        name="cinnamaldehyde",
        n_conformers=0,
    )

    assert molecule.name == "cinnamaldehyde"
    assert molecule.n_atoms > 0


def test_load_force_field_with_interface_metals_when_openff_available() -> None:
    """Optionally instantiate an OpenFF force field with packaged metals."""

    pytest.importorskip("openff.toolkit")
    openff_adapter = importlib.import_module("sammd.openff")

    force_field = openff_adapter.load_force_field(include_interface_metals=True)

    assert force_field is not None
