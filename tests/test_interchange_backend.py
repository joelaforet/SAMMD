"""Tests for OpenFF Interchange backend export scaffolding."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sammd.core.builders import build_system
from sammd.core.config import SAMMDConfig, load_config_dict


def test_backend_module_import_does_not_import_optional_science_modules() -> None:
    """Keep backend helpers lazy until export functions are called."""

    for name in list(sys.modules):
        if name.startswith(("openff", "openmm", "rdkit")):
            sys.modules.pop(name, None)

    importlib.import_module("sammd.backends.interchange")

    assert not any(name.startswith(("openff", "openmm", "rdkit")) for name in sys.modules)


def test_backend_build_summary_marks_completed_exports(tmp_path: Path) -> None:
    """Completed backend export metadata updates reserved artifact summaries."""

    backend = importlib.import_module("sammd.backends.interchange")
    plan = build_system(SAMMDConfig(), output_dir=tmp_path)
    result = SimpleNamespace(
        openff_toolkit_version="0.18.0",
        openff_interchange_version="0.5.3",
        positions_nm=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)),
        metal_indices=(0,),
        sulfur_indices=(1,),
        anchor_pairs=((1, 0),),
    )

    summary = backend.backend_build_summary(plan, result)

    assert summary["full_construction_available"] is True
    assert summary["artifacts"]["openff_interchange"]["available"] is True
    assert summary["artifacts"]["openff_interchange"]["constructed"] is True
    assert summary["artifacts"]["openff_interchange"]["status"] == "current"
    assert summary["artifacts"]["openmm_system"]["available"] is True
    assert summary["engine_exports"]["openmm"]["available"] is True
    assert summary["backend_export"]["openff_interchange_version"] == "0.5.3"
    assert summary["backend_export"]["sulfur_metal_pair_count"] == 1


def test_backend_export_rejects_salts_before_optional_imports(tmp_path: Path) -> None:
    """Avoid silently omitting schema-supported salts from backend artifacts."""

    backend = importlib.import_module("sammd.backends.interchange")
    config = load_config_dict(
        {
            "salts": [
                {
                    "name": "sodium_chloride",
                    "concentration": 0.1,
                    "cation": {
                        "name": "sodium",
                        "residue_name": "NAI",
                        "smiles": "[Na+]",
                        "count_per_formula_unit": 1,
                    },
                    "anion": {
                        "name": "chloride",
                        "residue_name": "CLI",
                        "smiles": "[Cl-]",
                        "count_per_formula_unit": 1,
                    },
                }
            ]
        }
    )
    plan = build_system(config, output_dir=tmp_path)

    with pytest.raises(NotImplementedError, match="does not yet support salts"):
        backend.build_interchange_backend(plan)


def test_pdbx_writer_terminates_final_loop_for_pymol(tmp_path: Path, monkeypatch) -> None:
    """PyMOL rejects CIF files that end inside the final loop without '#'."""

    backend = importlib.import_module("sammd.backends.interchange")

    class FakePDBxFile:
        @staticmethod
        def writeFile(topology, positions, handle, *, keepIds):  # noqa: N802, N803
            handle.write("loop_\n_atom_site.id\n1\n")

    fake_openmm = SimpleNamespace(app=SimpleNamespace(PDBxFile=FakePDBxFile))
    monkeypatch.setattr(backend, "require_openmm", lambda: fake_openmm)

    path = tmp_path / "solvated_system.cif"
    backend._write_pdbx(path, topology=object(), positions=object(), overwrite=False)

    assert path.read_text(encoding="utf-8").endswith("\n#\n")


def test_openmm_atom_name_sanitizer_fills_blank_metal_names() -> None:
    """OpenMM PDBx writer omits blank atom-name fields, which breaks PyMOL."""

    backend = importlib.import_module("sammd.backends.interchange")
    atom = SimpleNamespace(name="", element=SimpleNamespace(symbol="Pd"))
    topology = SimpleNamespace(atoms=lambda: iter([atom]))

    backend._ensure_openmm_atom_names(topology)

    assert atom.name == "Pd"


def test_openmm_metal_labeler_sets_atom_and_residue_names() -> None:
    """Metal particles should export as named atoms in three-character residues."""

    backend = importlib.import_module("sammd.backends.interchange")
    residue = SimpleNamespace(name="UNK", id="0", chain=SimpleNamespace(id="X"))
    atom = SimpleNamespace(index=7, name="", residue=residue)
    topology = SimpleNamespace(atoms=lambda: iter([atom]))

    backend._label_openmm_metal_atoms(topology, (7,), "Pd")

    assert atom.name == "Pd"
    assert residue.name == "Pdx"
    assert residue.id == "8"
    assert residue.chain.id == "M"


def test_metal_residue_name_pads_to_three_characters() -> None:
    """Use PDB-style three-character residue labels while preserving symbols."""

    backend = importlib.import_module("sammd.backends.interchange")

    assert backend._metal_residue_name("Pd") == "Pdx"
    assert backend._metal_residue_name("Au") == "Aux"
    assert backend._metal_residue_name("Zn") == "Znx"


def test_interface_metal_offxml_loads_with_current_openff() -> None:
    """The packaged INTERFACE OFFXML stays compatible with the CUDA env toolkit."""

    pytest.importorskip("openff.toolkit")
    from sammd.backends.openff import interface_fcc_metal_offxml_resource

    force_field_type = importlib.import_module("openff.toolkit").ForceField

    force_field_type(str(interface_fcc_metal_offxml_resource()))
