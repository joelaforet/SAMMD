"""Tests for SAMMD configuration defaults and validation."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from sammd.core.config import CONFIG_TEMPLATE, SAMMDConfig, load_config, load_config_dict


def test_project_scope_documents_total_solvent_padding_semantics() -> None:
    """Prevent docs from regressing to per-face solvent padding wording."""

    text = Path("docs/project-scope.md").read_text(encoding="utf-8")

    assert "3 nm total solvent padding" in text
    old_wording = (
        "3 nm solvent padding from the fully extended SAM tips to the box boundary on each side"
    )
    assert old_wording not in text


def _template_data() -> dict:
    """Return parsed YAML template data."""

    data = yaml.safe_load(CONFIG_TEMPLATE)
    assert isinstance(data, dict)
    return data


def test_yaml_template_loads(tmp_path: Path) -> None:
    """Load the commented YAML template from disk."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG_TEMPLATE, encoding="utf-8")

    config = load_config(config_path)

    assert isinstance(config, SAMMDConfig)


def test_resolved_defaults_match_approved_student_schema() -> None:
    """Validate the approved YAML-first defaults."""

    config = load_config_dict(_template_data())

    assert config.experiment.name == "propanethiol_cinnamaldehyde_pd111"
    assert config.experiment.seed == 2026
    assert config.surface.metal == "Pd"
    assert config.surface.facet == "111"
    assert config.surface.lateral_size == (2.0, 2.0)
    assert config.sam.grafting_density == 0.25
    assert config.sam.components[0].residue_name == "PTL"
    assert config.sam.components[0].extended_length_nm is None
    assert config.reactants[0].residue_name == "CIN"
    assert config.reactants[0].count == 1
    assert config.reactants[0].concentration is None
    assert config.solvent.padding == 3.0
    assert config.solvent.components[0].residue_name == "EOH"
    assert config.solvent.components[0].density == 0.789
    assert config.parameterization.nonbonded_cutoff == 1.0
    assert config.outputs.files.pymol_system == "solvated_system_pymol.pdb"
    assert config.outputs.files.openff_interchange == "interchange.json"
    assert config.outputs.files.anchor_metadata == "anchor_metadata.json"


def test_surface_validation_uses_fcc_surface_registry() -> None:
    """Validate surface metal/facet combinations through the metadata registry."""

    data = _template_data()
    data["surface"]["metal"] = "Pd"
    data["surface"]["facet"] = "111"
    assert load_config_dict(data).surface.metal == "Pd"

    data["surface"]["metal"] = "Pt"
    assert load_config_dict(data).surface.metal == "Pt"

    data["surface"]["metal"] = "Fe"
    with pytest.raises(ValidationError, match="unsupported Fcc surface 'Fe\\(111\\)'"):
        load_config_dict(data)


def test_yaml_template_describes_current_build_contract() -> None:
    """Prevent template comments from overstating current export outputs."""

    assert "Build and parameterize your molecular system" not in CONFIG_TEMPLATE
    assert "Use the exported OpenMM/OpenFF files" not in CONFIG_TEMPLATE
    assert "Validate and build the full system artifacts" in CONFIG_TEMPLATE
    assert "Use --full when you need solvated_system.cif" not in CONFIG_TEMPLATE
    assert "Files written by sammd build" in CONFIG_TEMPLATE
    assert "OpenFF Interchange artifact names written by --full" not in CONFIG_TEMPLATE
    assert "sam_grafting_density.cif" in CONFIG_TEMPLATE
    assert "build_summary.json" in CONFIG_TEMPLATE
    assert "resolved_config.yaml" in CONFIG_TEMPLATE
    assert "solvated_system.cif" in CONFIG_TEMPLATE
    assert "solvated_system_pymol.pdb" in CONFIG_TEMPLATE
    assert "interchange.json" in CONFIG_TEMPLATE
    assert "anchor_metadata.json" in CONFIG_TEMPLATE


def test_sam_extended_length_override_is_optional_and_positive() -> None:
    """Allow advanced users to override approximate SAM length for box planning."""

    data = _template_data()
    data["sam"]["components"][0]["extended_length_nm"] = 1.25

    assert load_config_dict(data).sam.components[0].extended_length_nm == 1.25

    data["sam"]["components"][0]["extended_length_nm"] = 0.0
    with pytest.raises(ValidationError, match="greater than 0"):
        load_config_dict(data)


def test_sam_null_disables_sam_while_omission_keeps_default() -> None:
    """Use explicit YAML null for bare-surface controls without changing defaults."""

    default_config = load_config_dict({})
    bare_data = _template_data()
    bare_data["sam"] = None

    bare_config = load_config_dict(bare_data)

    assert default_config.sam is not None
    assert default_config.sam.components[0].name == "propanethiol"
    assert bare_config.sam is None


def test_sam_string_none_is_rejected() -> None:
    """Require YAML null, not the string 'None', for no-SAM controls."""

    data = _template_data()
    data["sam"] = "None"

    with pytest.raises(ValidationError, match="Input should be a valid dictionary"):
        load_config_dict(data)


def test_yaml_template_describes_neutral_thiols_and_internal_nonbonded_attachment() -> None:
    """Keep beginner SAM wording aligned with the current validation contract."""

    normalized = " ".join(CONFIG_TEMPLATE.split())

    assert "neutral thiol with an HS/implicit-H sulfur" in normalized
    assert "not a pre-deprotonated thiolate" in normalized
    assert "strengthened" in normalized
    assert "nonbonded interaction" in normalized
    assert "not as covalent, quantum, or reactive chemistry" in normalized
    assert "not exposed as a beginner YAML knob yet" in normalized
    assert "optional advanced override" in normalized
    assert "fully extended SAM" in normalized
    assert "length from sulfur anchor to tail tip" in normalized
    assert "total z reservoir thickness across both exposed SAM faces" in normalized
    assert "split equally across both faces" in normalized


def test_yaml_template_describes_bare_slab_controls() -> None:
    """Teach users the explicit null syntax for no-SAM control systems."""

    normalized = " ".join(CONFIG_TEMPLATE.split())

    assert "sam: null" in normalized
    assert "bare metal-slab control" in normalized
    assert "default propanethiol SAM" in normalized
    assert "string \"None\"" in normalized
    assert "above the exposed metal surface" in normalized


def test_beginner_template_defers_sam_attachment_knobs() -> None:
    """Do not expose advanced metal-S controls in the beginner YAML template."""

    data = _template_data()

    assert "anchor" not in data["sam"]
    assert "metal_sulfur_interaction" not in data["sam"]
    assert "sam.anchor" not in CONFIG_TEMPLATE
    assert "metal_sulfur_interaction" not in CONFIG_TEMPLATE


def test_mixed_sam_fraction_validation() -> None:
    """Accept fraction-only mixed SAMs when fractions sum to one."""

    data = _template_data()
    data["sam"]["components"] = [
        {"name": "a", "residue_name": "AAA", "smiles": "CCCS", "fraction": 0.25},
        {"name": "b", "residue_name": "BBB", "smiles": "CCCCS", "fraction": 0.75},
    ]

    assert len(load_config_dict(data).sam.components) == 2

    data["sam"]["components"][1]["fraction"] = 0.5
    with pytest.raises(ValidationError, match="fractions must sum"):
        load_config_dict(data)


def test_mixed_sam_count_validation() -> None:
    """Accept count-only mixed SAMs and reject mixed composition modes."""

    data = _template_data()
    data["sam"]["components"] = [
        {"name": "a", "residue_name": "AAA", "smiles": "CCCS", "count": 4},
        {"name": "b", "residue_name": "BBB", "smiles": "CCCCS", "count": 6},
    ]
    assert [component.count for component in load_config_dict(data).sam.components] == [4, 6]

    data["sam"]["components"] = [
        {"name": "a", "residue_name": "AAA", "smiles": "CCCS", "fraction": 0.5},
        {"name": "b", "residue_name": "BBB", "smiles": "CCCCS", "count": 6},
    ]
    with pytest.raises(ValidationError, match="must not mix fraction and count"):
        load_config_dict(data)


def test_reactant_requires_count_or_concentration() -> None:
    """Require exactly one reactant amount mode."""

    data = _template_data()
    data["reactants"][0].pop("count")
    data["reactants"][0]["concentration"] = 50.0
    assert load_config_dict(data).reactants[0].concentration == 50.0

    data["reactants"][0]["count"] = 1
    with pytest.raises(ValidationError, match="exactly one of count or concentration"):
        load_config_dict(data)


def test_salt_schema_uses_explicit_ion_stoichiometry() -> None:
    """Accept separate cation/anion definitions with residue names."""

    data = _template_data()
    data["salts"] = [
        {
            "name": "sodium_sulfate",
            "concentration": 0.05,
            "cation": {
                "name": "sodium",
                "residue_name": "SOD",
                "smiles": "[Na+]",
                "count_per_formula_unit": 2,
            },
            "anion": {
                "name": "sulfate",
                "residue_name": "SUL",
                "smiles": "O=S(=O)([O-])[O-]",
                "count_per_formula_unit": 1,
            },
        }
    ]

    salt = load_config_dict(data).salts[0]

    assert salt.cation.residue_name == "SOD"
    assert salt.cation.count_per_formula_unit == 2
    assert salt.anion.residue_name == "SUL"


@pytest.mark.parametrize("residue_name", ["ptl", "PT", "PTL1", "P-L"])
def test_residue_names_are_strict_three_character_codes(residue_name: str) -> None:
    """Reject residue names that will make topology selection ambiguous."""

    data = _template_data()
    data["sam"]["components"][0]["residue_name"] = residue_name

    with pytest.raises(ValidationError, match="residue_name must be exactly 3"):
        load_config_dict(data)


def test_unknown_nested_keys_are_forbidden() -> None:
    """Reject unknown YAML keys in nested configuration sections."""

    data = _template_data()
    data["surface"]["extra"] = "bad"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_config_dict(data)

    data = _template_data()
    data["parameterization"]["extra"] = "bad"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_config_dict(data)


def test_solvent_density_and_molar_mass_validation() -> None:
    """Require physical metadata for non-water solvent count planning."""

    data = _template_data()
    data["solvent"]["components"] = [
        {"name": "ethanol", "residue_name": "EOH", "smiles": "CCO", "mole_fraction": 1.0}
    ]
    with pytest.raises(ValidationError, match="must define density"):
        load_config_dict(data)

    data["solvent"]["components"][0]["density"] = 0.789
    assert load_config_dict(data).solvent.components[0].density == 0.789

    data["solvent"]["components"] = [
        {
            "name": "custom-solvent",
            "residue_name": "CUS",
            "smiles": "CO",
            "mole_fraction": 1.0,
            "density": 0.79,
        }
    ]
    with pytest.raises(ValidationError, match="must define molar_mass"):
        load_config_dict(data)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("surface", "metal"), "Fe", "unsupported Fcc surface 'Fe\\(111\\)'"),
        (("surface", "lateral_size"), [-1.0, 2.0], "lateral_size"),
        (("sam", "grafting_density"), -0.1, "greater than 0"),
        (("solvent", "components", 0, "mole_fraction"), 0.0, "greater than 0"),
        (("reactants", 0, "initial_height_above_sam"), -1.0, "greater than 0"),
        (("parameterization", "nonbonded_cutoff"), 0.0, "greater than 0"),
    ],
)
def test_invalid_config_choices_fail_clearly(
    path: tuple[str | int, ...], value: object, message: str
) -> None:
    """Reject unsupported choices and non-positive physical values."""

    data = _template_data()
    cursor = data
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value

    with pytest.raises(ValidationError, match=message):
        load_config_dict(data)
