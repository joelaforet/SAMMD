"""Tests for the optional real-system OpenMM smoke runner."""

from __future__ import annotations

import importlib.util
import math
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from sammd.core.builders import build_system
from sammd.core.config import SAMMDConfig
from sammd.utils.geometry import norm


def load_smoke_tool():
    """Load the tools/openmm_smoke.py module without requiring tools as a package."""

    path = Path(__file__).resolve().parents[1] / "tools" / "openmm_smoke.py"
    spec = importlib.util.spec_from_file_location("sammd_openmm_smoke_tool", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_smoke_tool_import_does_not_import_heavy_science_modules() -> None:
    """Keep the smoke tool importable before the science environment is active."""

    sys.modules.pop("openmm", None)
    sys.modules.pop("rdkit", None)

    load_smoke_tool()

    assert "openmm" not in sys.modules
    assert "rdkit" not in sys.modules


def test_auto_solvent_count_uses_sammd_solution_plan() -> None:
    """Auto solvent count should use the SAMMD solution composition plan."""

    smoke = load_smoke_tool()
    config = SAMMDConfig.model_validate(
        {
            "surface": {"lateral_size": [2.0, 2.0]},
            "solvent": {
                "components": [
                    {
                        "name": "ethanol",
                        "residue_name": "EOH",
                        "smiles": "CCO",
                        "mole_fraction": 1.0,
                        "density": smoke.ETHANOL_DENSITY_G_ML,
                        "molar_mass": smoke.ETHANOL_MASS_G_MOL,
                    }
                ]
            },
        }
    )
    plan = build_system(config)

    count = smoke.resolve_solvent_count("auto", plan, "ethanol")
    expected = plan.solution.solvent_components[0].count

    assert count == expected


def test_rotation_matrix_maps_source_vector_to_target_vector() -> None:
    """SAM orientation helper should rotate the anchor bond onto the surface normal."""

    smoke = load_smoke_tool()
    matrix = smoke.rotation_matrix((1.0, 0.0, 0.0), (0.0, 0.0, -1.0))
    rotated = smoke.matvec(matrix, (1.0, 0.0, 0.0))

    assert rotated[0] == pytest.approx(0.0, abs=1.0e-12)
    assert rotated[1] == pytest.approx(0.0, abs=1.0e-12)
    assert rotated[2] == pytest.approx(-1.0, abs=1.0e-12)
    assert math.isclose(norm(rotated), 1.0)


def test_component_residue_assigner_wraps_after_9999_residues() -> None:
    """Follow PolyzyMD's one-repeat-unit-per-residue chain wrapping convention."""

    smoke = load_smoke_tool()
    assigner = smoke.ComponentResidueAssigner()

    identities = assigner.allocate("ethanol", "EOH", 10000)
    next_identity = assigner.allocate("reactant", "CIN", 1)[0]

    assert identities[0] == smoke.ResidueIdentity("A", 1, "EOH")
    assert identities[9998] == smoke.ResidueIdentity("A", 9999, "EOH")
    assert identities[9999] == smoke.ResidueIdentity("B", 1, "EOH")
    assert next_identity == smoke.ResidueIdentity("C", 1, "CIN")
    assert assigner.component_ranges["ethanol"] == {
        "residue_name": "EOH",
        "residue_count": 10000,
        "first_chain_id": "A",
        "last_chain_id": "B",
        "max_residues_per_chain": 9999,
    }


def test_default_run_schedule_records_300_frames_with_2fs_timestep() -> None:
    """Default smoke schedule should prioritize the requested trajectory frame count."""

    smoke = load_smoke_tool()

    schedule = smoke.resolve_run_schedule(
        duration_ns=5.0,
        timestep_fs=2.0,
        steps=None,
        frames=300,
        report_interval=None,
    )

    assert schedule.total_steps == 2500200
    assert schedule.report_interval == 8334
    assert schedule.frames == 300
    assert schedule.simulated_duration_ns == pytest.approx(5.0004)


def test_packmol_input_packs_solvent_around_fixed_solute() -> None:
    """Packmol input should pack solvent around a fixed solute instead of using a lattice."""

    smoke = load_smoke_tool()
    config = SAMMDConfig.model_validate(
        {"surface": {"lateral_size": [2.0, 2.0]}}
    )
    plan = build_system(config)
    box = smoke.derive_box_dimensions(plan, 3.0)

    assert box == plan.box_plan.dimensions_nm
    text = smoke.build_packmol_input(
        solute_path=Path("fixed_pd_sam.pdb"),
        solvent_path=Path("ethanol.pdb"),
        output_path=Path("packmol_output.pdb"),
        solvent_count=25,
        box_dimensions_nm=box,
    )

    assert "structure ethanol.pdb" in text
    assert "  number 25" in text
    assert "structure fixed_pd_sam.pdb" in text
    assert "fixed 0. 0. 0. 0. 0. 0." in text
    assert "inside box 0 0 0" in text
    assert "nloop 200" in text


def test_validate_args_rejects_invalid_solvent_count() -> None:
    """Catch invalid smoke-only CLI values before optional imports."""

    smoke = load_smoke_tool()
    args = Namespace(
        lateral_size_nm=2.0,
        solvent_padding_nm=3.0,
        timestep_fs=0.5,
        friction_per_ps=1.0,
        pd_s_sigma_angstrom=2.2,
        pd_s_epsilon_kcal_mol=1.0,
        duration_ns=5.0,
        sulfur_height_nm=0.0,
        seed=1,
        steps=1,
        frames=300,
        minimize_iterations=1,
        report_interval=1,
        reactant_count=None,
        solvent_count="zero",
        water_count=None,
    )

    with pytest.raises(SystemExit, match="--solvent-count"):
        smoke.validate_args(args)


def test_smoke_tool_defaults_match_canonical_metal_sulfur_strategy() -> None:
    """Keep the smoke pair override aligned with dependency-free SAM metadata."""

    smoke = load_smoke_tool()

    assert pytest.approx(2.2) == smoke.DEFAULT_PD_S_SIGMA_ANGSTROM
    assert pytest.approx(2.0) == smoke.DEFAULT_PD_S_EPSILON_KCAL_MOL
