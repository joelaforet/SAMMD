"""Tests for lazy optional OpenFF adapter utilities."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest

from sammd.config import SAMMDConfig, load_config_dict


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


def test_molecule_from_smiles_disallows_undefined_stereo_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the safer OpenFF stereochemistry default unless callers opt in."""

    openff_adapter = importlib.import_module("sammd.openff")
    captured: dict[str, bool] = {}

    molecule = SimpleNamespace(name="", generate_conformers=lambda n_conformers: None)

    def fake_from_smiles(smiles: str, allow_undefined_stereo: bool, **kwargs):
        captured["allow_undefined_stereo"] = allow_undefined_stereo
        return molecule

    fake_molecule_type = type("FakeMolecule", (), {"from_smiles": staticmethod(fake_from_smiles)})
    monkeypatch.setattr(
        openff_adapter,
        "require_openff_toolkit",
        lambda: SimpleNamespace(Molecule=fake_molecule_type),
    )

    openff_adapter.molecule_from_smiles("CCCS", n_conformers=0)

    assert captured == {"allow_undefined_stereo": False}


def test_molecules_from_config_groups_supported_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    """Create molecules and report entries that are not OpenFF molecules."""

    openff_adapter = importlib.import_module("sammd.openff")
    calls: list[tuple[str, str | None, int, bool]] = []

    def fake_molecule_from_smiles(
        smiles: str,
        name: str | None = None,
        n_conformers: int = 1,
        allow_undefined_stereo: bool = False,
    ) -> dict[str, str | None]:
        calls.append((smiles, name, n_conformers, allow_undefined_stereo))
        return {"smiles": smiles, "name": name}

    monkeypatch.setattr(openff_adapter, "molecule_from_smiles", fake_molecule_from_smiles)

    result = openff_adapter.molecules_from_config(SAMMDConfig(), n_conformers=0)

    assert result.molecules["sam"] == [{"smiles": "CCCS", "name": "propanethiol"}]
    assert result.molecules["solvent"] == [{"smiles": "CCO", "name": "ethanol"}]
    assert result.molecules["reactants"] == [
        {"smiles": "C1=CC=C(C=C1)/C=C/C=O", "name": "cinnamaldehyde"}
    ]
    assert result.molecules["salts"] == []
    assert result.skipped == []
    assert result.unsupported == []
    assert calls == [
        ("CCCS", "propanethiol", 0, False),
        ("CCO", "ethanol", 0, False),
        ("C1=CC=C(C=C1)/C=C/C=O", "cinnamaldehyde", 0, False),
    ]


def test_molecules_from_config_reports_solvent_without_smiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report non-water solvent entries that lack SMILES instead of dropping them."""

    openff_adapter = importlib.import_module("sammd.openff")
    monkeypatch.setattr(openff_adapter, "molecule_from_smiles", lambda *args, **kwargs: args[0])
    config = load_config_dict(
        {
            "solvent": {
                "components": [
                    {
                        "name": "ethanol",
                        "residue_name": "EOH",
                        "mole_fraction": 1.0,
                        "density": 0.789,
                    }
                ]
            }
        }
    )

    result = openff_adapter.molecules_from_config(config, n_conformers=0)

    assert result.molecules["solvent"] == []
    assert [(entry.section, entry.name) for entry in result.unsupported] == [
        ("solvent", "ethanol")
    ]
    assert "no SMILES" in result.unsupported[0].reason


def test_molecules_from_config_prepares_salt_ions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prepare separate OpenFF molecules for cation and anion species."""

    openff_adapter = importlib.import_module("sammd.openff")
    calls: list[tuple[str, str | None]] = []

    def fake_molecule_from_smiles(smiles: str, name: str | None = None, **kwargs):
        calls.append((smiles, name))
        return {"smiles": smiles, "name": name}

    monkeypatch.setattr(openff_adapter, "molecule_from_smiles", fake_molecule_from_smiles)
    config = load_config_dict(
        {
            "salts": [
                {
                    "name": "sodium_chloride",
                    "concentration": 0.15,
                    "cation": {
                        "name": "sodium",
                        "residue_name": "SOD",
                        "smiles": "[Na+]",
                        "count_per_formula_unit": 1,
                    },
                    "anion": {
                        "name": "chloride",
                        "residue_name": "CLA",
                        "smiles": "[Cl-]",
                        "count_per_formula_unit": 1,
                    },
                }
            ]
        }
    )

    result = openff_adapter.molecules_from_config(config, n_conformers=0)

    assert result.molecules["salts"] == [
        {"smiles": "[Na+]", "name": "sodium"},
        {"smiles": "[Cl-]", "name": "chloride"},
    ]
    assert ("[Na+]", "sodium") in calls
    assert ("[Cl-]", "chloride") in calls
    assert result.unsupported == []


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
        allow_undefined_stereo=True,
    )

    assert molecule.name == "cinnamaldehyde"
    assert molecule.n_atoms > 0


def test_load_force_field_with_interface_metals_when_openff_available() -> None:
    """Optionally verify packaged Pd vdW parameters are loaded."""

    pytest.importorskip("openff.toolkit")
    openff_adapter = importlib.import_module("sammd.openff")

    force_field = openff_adapter.load_force_field(include_interface_metals=True)
    vdw_handler = force_field.get_parameter_handler("vdW")
    pd_parameters = [
        parameter
        for parameter in vdw_handler.parameters
        if getattr(parameter, "smirks", None) == "[#46:1]"
    ]

    assert pd_parameters
    assert any(getattr(parameter, "id", None) == "Pd" for parameter in pd_parameters)
    assert any(
        "6.15" in _quantity_text(getattr(parameter, "epsilon", ""))
        for parameter in pd_parameters
    )
    assert any(
        "1.4095" in _quantity_text(getattr(parameter, "rmin_half", ""))
        for parameter in pd_parameters
    )


def _quantity_text(value: object) -> str:
    """Return robust text for OpenFF quantity-like parameter values."""

    return str(value).replace(" ", "")
