"""Tests for SAMMD configuration defaults and validation."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from sammd.config import CONFIG_TEMPLATE, SAMMDConfig, load_config, load_config_dict


def _template_data() -> dict:
    """Return parsed YAML template data."""

    data = yaml.safe_load(CONFIG_TEMPLATE)
    assert isinstance(data, dict)
    return data


def test_yaml_template_loads(tmp_path: Path) -> None:
    """Load the commented YAML template from disk."""

    config_path = tmp_path / "sammd.yaml"
    config_path.write_text(CONFIG_TEMPLATE, encoding="utf-8")

    config = load_config(config_path)

    assert isinstance(config, SAMMDConfig)


def test_resolved_defaults() -> None:
    """Validate canonical MVP defaults from the project scope."""

    config = load_config_dict(_template_data())

    assert config.surface.metal == "Pd"
    assert config.surface.facet == "111"
    assert config.sam.anchor.site == "fcc_hollow"
    assert config.solvent.water_model == "TIP3P"
    assert config.surface.slab.positional_restraint.value == 10000.0
    assert config.surface.slab.positional_restraint.unit == "kJ mol^-1 nm^-2"
    assert config.sam.grafting_density.value == 0.25
    assert config.sam.grafting_density.unit == "nm^2 / molecule"
    assert config.output.topology == "topology.cif"
    assert config.output.trajectory == "trajectory.dcd"
    assert config.output.thermodynamics == "thermodynamics.csv"
    assert config.sam.anchor.nonbonded.scale_factor == 4.0
    assert config.reactants[0].smiles == "C1=CC=C(C=C1)/C=C/C=O"


def test_mixed_sam_fraction_validation() -> None:
    """Accept fraction-only mixed SAMs when fractions sum to one."""

    data = _template_data()
    data["sam"]["components"] = [
        {"name": "a", "smiles": "CCCS", "fraction": 0.25},
        {"name": "b", "smiles": "CCCCS", "fraction": 0.75},
    ]

    assert len(load_config_dict(data).sam.components) == 2

    data["sam"]["components"][1]["fraction"] = 0.5
    with pytest.raises(ValidationError, match="fractions must sum"):
        load_config_dict(data)


def test_mixed_sam_count_validation() -> None:
    """Accept count-only mixed SAMs and reject mixed composition modes."""

    data = _template_data()
    data["sam"]["components"] = [
        {"name": "a", "smiles": "CCCS", "count": 4},
        {"name": "b", "smiles": "CCCCS", "count": 6},
    ]
    assert [component.count for component in load_config_dict(data).sam.components] == [4, 6]

    data["sam"]["components"] = [
        {"name": "a", "smiles": "CCCS", "fraction": 0.5},
        {"name": "b", "smiles": "CCCCS", "count": 6},
    ]
    with pytest.raises(ValidationError, match="must not mix fraction and count"):
        load_config_dict(data)


def test_sam_count_sum_validation_helper() -> None:
    """Validate explicit component counts against a supplied site count."""

    data = _template_data()
    data["sam"]["components"] = [
        {"name": "a", "smiles": "CCCS", "count": 4},
        {"name": "b", "smiles": "CCCCS", "count": 6},
    ]
    sam = load_config_dict(data).sam

    sam.validate_component_counts(total_sites=10)
    with pytest.raises(ValueError, match="counts sum to 10, but total_sites is 9"):
        sam.validate_component_counts(total_sites=9)


def test_unknown_nested_keys_are_forbidden() -> None:
    """Reject unknown YAML keys in nested configuration sections."""

    data = _template_data()
    data["surface"]["extra"] = "bad"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_config_dict(data)

    data = _template_data()
    data["reporters"]["extra"] = "bad"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_config_dict(data)


def test_cosolvent_density_validation() -> None:
    """Require density for volume-fraction co-solvents."""

    data = _template_data()
    data["solvent"]["components"] = [
        {"name": "water", "volume_fraction": 0.5},
        {"name": "ethanol", "smiles": "CCO", "volume_fraction": 0.5},
    ]
    with pytest.raises(ValidationError, match="co-solvent 'ethanol' must define density_g_ml"):
        load_config_dict(data)

    data["solvent"]["components"][1]["density_g_ml"] = 0.789
    assert load_config_dict(data).solvent.components[1].density_g_ml == 0.789


def test_slab_thickness_validation() -> None:
    """Require Pd(111) slab thickness to exceed cutoff plus buffer."""

    config = load_config_dict(_template_data())
    assert config.surface.slab.layers == 8

    data = _template_data()
    data["surface"]["slab"]["layers"] = 4
    with pytest.raises(ValidationError, match="slab thickness must exceed"):
        load_config_dict(data)


def test_expected_unit_validation() -> None:
    """Reject arbitrary unit strings for known physical defaults."""

    data = _template_data()
    data["sam"]["grafting_density"]["unit"] = "angstrom^2 / molecule"
    with pytest.raises(ValidationError, match="grafting_density unit must be"):
        load_config_dict(data)

    data = _template_data()
    data["surface"]["slab"]["positional_restraint"]["unit"] = "kcal mol^-1 angstrom^-2"
    with pytest.raises(ValidationError, match="positional_restraint unit must be"):
        load_config_dict(data)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("surface", "metal"), "Fe", "Input should be 'Pd'"),
        (("sam", "anchor", "site"), "unknown", "Input should be"),
        (("sam", "grafting_density", "value"), -0.1, "greater than 0"),
        (("solvent", "components", 0, "volume_fraction"), 0.0, "greater than 0"),
        (("reactants", 0, "concentration_molar"), -1.0, "greater than 0"),
        (("reporters", "fields"), ["step", "bad_field"], "unsupported reporter fields"),
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
