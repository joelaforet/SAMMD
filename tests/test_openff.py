"""Tests for lazy optional OpenFF adapter utilities."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest

from sammd.core.builders import build_system
from sammd.core.config import SAMMDConfig, load_config_dict


def test_adapter_import_does_not_import_openff_eagerly() -> None:
    """Keep importing the adapter independent of optional OpenFF dependencies."""

    optional_prefixes = ("openff", "openmm", "rdkit")
    optional_modules_before = _loaded_optional_modules(optional_prefixes)

    importlib.import_module("sammd.backends.openff")

    optional_modules_after = _loaded_optional_modules(optional_prefixes)

    assert optional_modules_after == optional_modules_before


def test_openff_backend_availability_reports_missing_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return a structured unavailable report without importing optional backends."""

    openff_adapter = importlib.import_module("sammd.backends.openff")
    optional_modules_before = _loaded_optional_modules(("openff", "openmm", "rdkit"))

    monkeypatch.setattr(openff_adapter.importlib_util, "find_spec", lambda name: None)

    availability = openff_adapter.check_openff_backend_availability()

    assert not availability.toolkit_available
    assert not availability.interchange_available
    assert not availability.backend_available
    assert any("OpenFF Toolkit" in message for message in availability.messages)
    assert any("OpenFF Interchange" in message for message in availability.messages)
    assert "OpenFF Toolkit" in availability.guidance
    assert "OpenFF Interchange" in availability.guidance
    assert _loaded_optional_modules(("openff", "openmm", "rdkit")) == optional_modules_before


def test_openff_backend_availability_handles_absent_parent_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat find_spec absent-parent failures as unavailable optional backends."""

    openff_adapter = importlib.import_module("sammd.backends.openff")

    def fake_find_spec(name: str):
        raise ModuleNotFoundError("No module named 'openff'", name=name)

    monkeypatch.setattr(openff_adapter.importlib_util, "find_spec", fake_find_spec)

    availability = openff_adapter.check_openff_backend_availability()

    assert not availability.toolkit_available
    assert not availability.interchange_available
    assert not availability.backend_available
    assert any("OpenFF Toolkit" in message for message in availability.messages)
    assert any("OpenFF Interchange" in message for message in availability.messages)


def test_openff_backend_availability_reports_partial_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report Interchange guidance when Toolkit is discoverable but Interchange is not."""

    openff_adapter = importlib.import_module("sammd.backends.openff")

    def fake_find_spec(name: str):
        if name == "openff.toolkit":
            return object()
        if name == "openff.interchange":
            return None
        raise AssertionError(name)

    monkeypatch.setattr(openff_adapter.importlib_util, "find_spec", fake_find_spec)

    availability = openff_adapter.check_openff_backend_availability()

    assert availability.toolkit_available
    assert not availability.interchange_available
    assert not availability.backend_available
    assert not any("Toolkit is not installed" in message for message in availability.messages)
    assert any("OpenFF Interchange" in message for message in availability.messages)
    assert "OpenFF Interchange" in availability.guidance
    assert "OpenFF Toolkit" not in availability.guidance


def test_force_field_inputs_from_config_are_inspectable_without_openff_imports() -> None:
    """Expose configured base force field and packaged INTERFACE resource marker."""

    openff_adapter = importlib.import_module("sammd.backends.openff")
    config = SAMMDConfig(parameterization={"small_molecule_force_field": "openff-2.1.0.offxml"})

    inputs = openff_adapter.force_field_inputs_from_config(config)

    assert inputs[0] == "openff-2.1.0.offxml"
    assert inputs[1].name == "interface_fcc_metals.offxml"
    assert inputs[1].is_file()


def test_parameterization_plan_from_config_records_choices_and_targets() -> None:
    """Plan future backend parameterization without constructing backend objects."""

    openff_adapter = importlib.import_module("sammd.backends.openff")
    config = SAMMDConfig(
        parameterization={
            "small_molecule_force_field": "openff-2.1.0.offxml",
            "charge_model": "am1bcc",
            "nonbonded_cutoff": 1.2,
        },
        outputs={"directory": "planned", "files": {"openff_interchange": "target.json"}},
    )

    plan = openff_adapter.parameterization_plan_from_config(config)

    assert plan.small_molecule_force_field == "openff-2.1.0.offxml"
    assert plan.charge_model == "am1bcc"
    assert plan.metal_force_field_resource == "interface_fcc_metals.offxml"
    assert plan.nonbonded_cutoff == 1.2
    assert plan.output_targets["openff_interchange"] == "planned/target.json"
    assert plan.output_targets["openmm_system"] == "planned/system.xml"
    assert plan.component_counts == {"sam": 1, "solvent": 1, "reactants": 1, "salts": 0}


def test_parameterization_plan_from_build_plan_records_counts_and_keeps_lightweight(
    tmp_path,
) -> None:
    """Summarize build-plan molecule counts while leaving construction disabled."""

    openff_adapter = importlib.import_module("sammd.backends.openff")
    build_plan = build_system(SAMMDConfig(), output_dir=tmp_path)

    plan = openff_adapter.parameterization_plan_from_build_plan(build_plan)

    assert not build_plan.full_construction_available
    assert plan.output_targets["openff_interchange"] == str(tmp_path / "interchange.json")
    assert plan.output_targets["openmm_system"] == str(tmp_path / "system.xml")
    assert plan.component_counts["sam_placements"] == len(build_plan.sam_placements.placements)
    assert plan.molecule_counts == build_plan.solution.molecule_counts


def test_require_openff_toolkit_fails_with_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explain that the optional backend requires a CUDA pixi environment."""

    openff_adapter = importlib.import_module("sammd.backends.openff")
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "openff.toolkit":
            raise ImportError("missing OpenFF")
        return real_import_module(name, package)

    monkeypatch.setattr(openff_adapter, "import_module", fake_import_module)

    with pytest.raises(ImportError, match="CUDA pixi environment"):
        openff_adapter.require_openff_toolkit()


def test_interface_fcc_metal_offxml_resource_exists() -> None:
    """Expose the packaged metal OFFXML resource without importing OpenFF."""

    openff_adapter = importlib.import_module("sammd.backends.openff")
    resource = openff_adapter.interface_fcc_metal_offxml_resource()

    assert resource.name == "interface_fcc_metals.offxml"
    assert resource.is_file()


def test_molecule_from_smiles_requires_nonnegative_conformer_count() -> None:
    """Validate conformer count before importing optional backends."""

    openff_adapter = importlib.import_module("sammd.backends.openff")

    with pytest.raises(ValueError, match="n_conformers must be non-negative"):
        openff_adapter.molecule_from_smiles("CCCS", n_conformers=-1)


def test_molecule_from_smiles_disallows_undefined_stereo_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the safer OpenFF stereochemistry default unless callers opt in."""

    openff_adapter = importlib.import_module("sammd.backends.openff")
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

    openff_adapter = importlib.import_module("sammd.backends.openff")
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

    openff_adapter = importlib.import_module("sammd.backends.openff")
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

    openff_adapter = importlib.import_module("sammd.backends.openff")
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


def test_prepare_molecule_template_assigns_charges_and_extracts_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prepare one charged molecule template without importing real OpenFF in tests."""

    openff_adapter = importlib.import_module("sammd.backends.openff")
    calls: list[tuple[str, object]] = []

    class FakeQuantity(float):
        def m_as(self, unit: str) -> float:
            calls.append(("unit", unit))
            return float(self)

    class FakeMolecule:
        def __init__(self) -> None:
            self.name = ""
            self.n_atoms = 2
            self.atoms = [SimpleNamespace(symbol="C"), SimpleNamespace(symbol="O")]
            self.conformers = [
                ((FakeQuantity(0.1), FakeQuantity(0.2), FakeQuantity(0.3)),
                 (FakeQuantity(0.4), FakeQuantity(0.5), FakeQuantity(0.6)))
            ]

        def generate_conformers(self, n_conformers: int) -> None:
            calls.append(("conformers", n_conformers))

        def assign_partial_charges(self, charge_model: str) -> None:
            calls.append(("charges", charge_model))

    fake_molecule = FakeMolecule()

    def fake_from_smiles(smiles: str, allow_undefined_stereo: bool):
        calls.append(("smiles", (smiles, allow_undefined_stereo)))
        return fake_molecule

    toolkit = SimpleNamespace(
        Molecule=SimpleNamespace(from_smiles=fake_from_smiles),
    )

    template = openff_adapter.prepare_molecule_template(
        "CCO",
        "ethanol",
        "am1bcc",
        toolkit=toolkit,
    )

    assert template.molecule is fake_molecule
    assert template.molecule.name == "ethanol"
    assert template.positions_nm == ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))
    assert template.atom_symbols == ("C", "O")
    assert ("smiles", ("CCO", True)) in calls
    assert ("conformers", 1) in calls
    assert ("charges", "am1bcc") in calls


def test_molecule_from_propanethiol_smiles_when_openff_available() -> None:
    """Optionally create a thiol OpenFF molecule."""

    pytest.importorskip("openff.toolkit")
    openff_adapter = importlib.import_module("sammd.backends.openff")

    molecule = openff_adapter.molecule_from_smiles("CCCS", name="propanethiol")

    assert molecule.name == "propanethiol"
    assert molecule.n_atoms > 0


def test_molecule_from_cinnamaldehyde_smiles_when_openff_available() -> None:
    """Optionally create the default reactant OpenFF molecule."""

    pytest.importorskip("openff.toolkit")
    openff_adapter = importlib.import_module("sammd.backends.openff")

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
    openff_adapter = importlib.import_module("sammd.backends.openff")

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


def _loaded_optional_modules(prefixes: tuple[str, ...]) -> set[str]:
    """Return loaded modules matching optional backend package roots."""

    return {
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
    }
