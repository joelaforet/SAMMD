"""Tests for INTERFACE Fcc metal parameters."""

from importlib.resources import as_file, files
from xml.etree import ElementTree

import pytest

from sammd.forcefields import (
    FCC_METAL_LJ_REGISTRY,
    generate_interface_metal_offxml,
    get_fcc_metal_parameters,
    sigma_from_rmin_half,
    write_interface_metal_offxml,
)


def test_registry_contains_all_fcc_metals() -> None:
    """Reproduce the CHARMM-INTERFACE Fcc metal set."""

    assert set(FCC_METAL_LJ_REGISTRY) == {"Ag", "Al", "Au", "Cu", "Ni", "Pb", "Pd", "Pt"}


def test_registry_reproduces_pd_values() -> None:
    """Validate Pd parameters from the project scope table."""

    pd = get_fcc_metal_parameters("Pd")
    assert pd.source_epsilon_kcal_mol == -6.15
    assert pd.openff_epsilon_kcal_mol == 6.15
    assert pd.rmin_half_angstrom == 1.4095


def test_epsilon_sign_conversion_and_sigma_helper() -> None:
    """Store CHARMM epsilon as negative and OpenFF epsilon as positive."""

    for parameters in FCC_METAL_LJ_REGISTRY.values():
        assert parameters.source_epsilon_kcal_mol < 0
        assert parameters.openff_epsilon_kcal_mol == abs(parameters.source_epsilon_kcal_mol)
        assert parameters.sigma_angstrom == pytest.approx(
            2 * parameters.rmin_half_angstrom / 2 ** (1 / 6)
        )
    assert sigma_from_rmin_half(1.4095) == pytest.approx(2.511443, rel=1e-6)


def test_unsupported_metal_fails_clearly() -> None:
    """Reject metals outside the lightweight Fcc registry."""

    with pytest.raises(ValueError, match="unsupported Fcc metal"):
        get_fcc_metal_parameters("Fe")


def test_generated_offxml_is_well_formed() -> None:
    """Generate loadable XML without requiring OpenFF imports."""

    root = ElementTree.fromstring(generate_interface_metal_offxml())

    assert root.tag == "SMIRNOFF"


def test_generated_offxml_contains_one_atom_per_fcc_metal() -> None:
    """Emit one vdW atom entry for each registered Fcc metal."""

    root = ElementTree.fromstring(generate_interface_metal_offxml())
    atoms = root.find("vdW").findall("Atom")

    assert len(atoms) == len(FCC_METAL_LJ_REGISTRY)
    assert {atom.attrib["id"] for atom in atoms} == set(FCC_METAL_LJ_REGISTRY)


def test_generated_offxml_reproduces_pd_parameters_with_units() -> None:
    """Use positive OpenFF epsilon and CHARMM Rmin/2 for Pd."""

    root = ElementTree.fromstring(generate_interface_metal_offxml())
    pd_atom = next(atom for atom in root.find("vdW").findall("Atom") if atom.attrib["id"] == "Pd")

    assert pd_atom.attrib["smirks"] == "[#46:1]"
    assert pd_atom.attrib["epsilon"] == "6.15 * kilocalorie_per_mole"
    assert pd_atom.attrib["rmin_half"] == "1.4095 * angstrom"


def test_generated_offxml_includes_precise_source_url() -> None:
    """Preserve the exact CHARMM-INTERFACE parameter file URL."""

    assert (
        "https://github.com/hendrikheinz/INTERFACE-force-field-and-surface-models/blob/master/"
        "charmm27_interface_v1_5.prm"
    ) in generate_interface_metal_offxml()


def test_generated_offxml_matches_packaged_resource() -> None:
    """Keep the packaged resource synchronized with registry output."""

    resource_text = files("sammd.data").joinpath("interface_fcc_metals.offxml").read_text(
        encoding="utf-8"
    )

    assert resource_text == generate_interface_metal_offxml()


def test_write_helper_writes_deterministic_loadable_xml(tmp_path) -> None:
    """Write the generated OFFXML exactly once to a user path."""

    output_path = tmp_path / "metals.offxml"

    write_interface_metal_offxml(output_path)

    written_text = output_path.read_text(encoding="utf-8")
    assert written_text == generate_interface_metal_offxml()
    assert ElementTree.fromstring(written_text).tag == "SMIRNOFF"


def test_write_helper_refuses_existing_file_without_overwrite(tmp_path) -> None:
    """Avoid silently clobbering an existing user file."""

    output_path = tmp_path / "metals.offxml"
    output_path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_interface_metal_offxml(output_path)

    assert output_path.read_text(encoding="utf-8") == "existing"


def test_write_helper_overwrites_existing_file_when_requested(tmp_path) -> None:
    """Allow explicit atomic replacement of an existing user file."""

    output_path = tmp_path / "metals.offxml"
    output_path.write_text("existing", encoding="utf-8")

    write_interface_metal_offxml(output_path, overwrite=True)

    assert output_path.read_text(encoding="utf-8") == generate_interface_metal_offxml()


def test_generated_offxml_excludes_pcff_96_parameters() -> None:
    """Avoid mixing unsupported PCFF 9-6 parameters into the export."""

    offxml = generate_interface_metal_offxml()

    assert "PCFF" not in offxml
    assert "9-6" not in offxml


def test_packaged_offxml_loads_with_openff_when_available() -> None:
    """Optionally validate packaged OFFXML with the OpenFF Toolkit."""

    openff_toolkit = pytest.importorskip("openff.toolkit")
    resource = files("sammd.data").joinpath("interface_fcc_metals.offxml")

    with as_file(resource) as resource_path:
        openff_toolkit.ForceField(str(resource_path))
