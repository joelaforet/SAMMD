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


def test_interface_metal_offxml_loads_with_current_openff() -> None:
    """The packaged INTERFACE OFFXML stays compatible with the CUDA env toolkit."""

    pytest.importorskip("openff.toolkit")
    from sammd.backends.openff import interface_fcc_metal_offxml_resource

    force_field_type = importlib.import_module("openff.toolkit").ForceField

    force_field_type(str(interface_fcc_metal_offxml_resource()))
