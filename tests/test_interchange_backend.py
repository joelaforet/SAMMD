"""Tests for OpenFF Interchange export scaffolding."""

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sammd.core.builders import build_system
from sammd.core.config import SAMMDConfig, load_config_dict
from sammd.core.io import AtomRecord
from sammd.model.metal_sulfur import METAL_SULFUR_EPSILON_KCAL_MOL, METAL_SULFUR_SIGMA_NM


def test_interchange_module_import_does_not_import_unused_science_modules() -> None:
    """Keep unused science modules lazy until export functions are called."""

    code = """
import importlib
import sys

unused_prefixes = ("mbuild", "MDAnalysis", "parmed", "pdbfixer")
for name in list(sys.modules):
    if name.startswith(unused_prefixes):
        sys.modules.pop(name, None)

importlib.import_module("sammd.backends.interchange")

loaded = [name for name in sys.modules if name.startswith(unused_prefixes)]
if loaded:
    raise SystemExit(f"unused science modules imported: {loaded!r}")
"""

    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_progress_logs_and_preserves_callback_compatibility(caplog) -> None:
    """Interchange export progress should log while retaining the callback contract."""

    backend = importlib.import_module("sammd.backends.interchange")
    messages: list[str] = []

    with caplog.at_level(logging.INFO, logger="sammd.backends.interchange"):
        backend._progress(messages.append, "Writing interchange.json")

    assert messages == ["Writing interchange.json"]
    assert [record.name for record in caplog.records] == ["sammd.backends.interchange"]
    assert [record.getMessage() for record in caplog.records] == ["Writing interchange.json"]


def test_interchange_build_summary_marks_completed_exports(tmp_path: Path) -> None:
    """Completed Interchange export metadata updates reserved artifact summaries."""

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

    assert summary["artifacts"]["openff_interchange"]["available"] is True
    assert summary["artifacts"]["openff_interchange"]["constructed"] is True
    assert summary["artifacts"]["openff_interchange"]["status"] == "current"
    assert summary["artifacts"]["pymol_system"]["status"] == "current"
    assert "openmm_system" not in summary["artifacts"]
    assert summary["backend_export"]["openff_interchange_version"] == "0.5.3"
    assert summary["backend_export"]["sulfur_metal_pair_count"] == 1
    override = summary["backend_export"]["metal_sulfur_override"]
    assert override["mode"] == "openff_interchange_plugin_collection"
    assert override["sigma_nm"] == METAL_SULFUR_SIGMA_NM
    assert override["epsilon_kcal_mol"] == METAL_SULFUR_EPSILON_KCAL_MOL


def test_interchange_export_preserves_runtime_geometry_for_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Export wrapper should return runtime solvent geometry for emitted summaries."""

    backend = importlib.import_module("sammd.backends.interchange")
    plan = build_system(SAMMDConfig(), output_dir=tmp_path)
    geometry = backend.RuntimeSolventGeometry(
        solvent_boundary_z_bounds_nm=(1.0, 2.5),
        fixed_solute_z_bounds_nm=(1.0, 3.0),
        solvent_regions_nm=(((0.0, 2.0), (0.0, 2.0), (0.0, 0.8)),),
        solvent_count_planning_volume_nm3=3.2,
        solvent_padding_nm=3.0,
        solvent_padding_per_face_nm=1.5,
        solvent_clearance_nm=0.18,
        dimensions_nm=(2.0, 2.0, 4.0),
        z_shift_nm=1.2,
        molecule_counts={"ethanol": 17},
    )
    result = backend.BackendExportResult(
        interchange=SimpleNamespace(model_dump_json=lambda indent: "{}"),
        openmm_topology=object(),
        metal_sulfur_collection=object(),
        positions=object(),
        positions_nm=((0.0, 0.0, 1.0), (0.1, 0.0, 1.1)),
        sulfur_indices=(1,),
        metal_indices=(0,),
        anchor_pairs=((1, 0),),
        component_ranges={},
        files={},
        openff_toolkit_version="0.18.0",
        openff_interchange_version="0.5.3",
        runtime_solvent_geometry=geometry,
    )
    monkeypatch.setattr(backend, "build_interchange_backend", lambda plan, progress=None: result)
    monkeypatch.setattr(backend, "_write_pdbx", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend, "_write_pdb", lambda *args, **kwargs: None)

    exported = backend.export_interchange_backend(plan, overwrite=True)
    summary = backend.backend_build_summary(plan, exported)

    assert exported.runtime_solvent_geometry == geometry
    assert summary["box"]["dimensions_nm"] == [2.0, 2.0, 4.0]
    assert summary["box"]["actual_solvent_boundary_z_bounds_nm"] == [1.0, 2.5]
    assert summary["box"]["actual_fixed_solute_z_bounds_nm"] == [1.0, 3.0]
    assert summary["box"]["solvent_packing_regions_nm"] == [
        [[0.0, 2.0], [0.0, 2.0], [0.0, 0.8]]
    ]
    assert summary["solution"]["molecule_counts"] == {"ethanol": 17}


def test_interchange_export_rejects_salts_before_optional_imports(tmp_path: Path) -> None:
    """Avoid silently omitting schema-supported salts from export artifacts."""

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


def test_runtime_solvent_region_stays_at_boundary_padding_when_reactant_expands_box() -> None:
    """Reactant containment should not enlarge solvent counting reservoirs."""

    backend = importlib.import_module("sammd.backends.interchange")
    plan = SimpleNamespace(
        config=SimpleNamespace(
            solvent=SimpleNamespace(padding=3.0),
            packing=SimpleNamespace(packmol=SimpleNamespace(tolerance=1.8)),
        ),
        box_plan=SimpleNamespace(dimensions_nm=(2.0, 3.0, 4.0)),
    )
    boundary_records = (
        AtomRecord(1, "Pd", "Pd", "PD", 1, "M", "metal", (0.0, 0.0, 0.0)),
        AtomRecord(2, "S", "S", "SAM", 1, "A", "sam", (0.0, 0.0, 1.0)),
    )
    fixed_records = (
        *boundary_records,
        AtomRecord(3, "C", "C", "RCT", 1, "B", "reactant", (0.0, 0.0, 6.0)),
    )

    geometry = backend._runtime_solvent_geometry(
        plan,
        fixed_records,
        solvent_boundary_records=boundary_records,
    )

    shifted_boundary_top = 1.0 + geometry.z_shift_nm
    shifted_reactant_top = 6.0 + geometry.z_shift_nm
    assert geometry.dimensions_nm[2] == pytest.approx(shifted_reactant_top + 0.18)
    assert geometry.solvent_regions_nm[1][2][0] == pytest.approx(shifted_boundary_top + 0.18)
    assert geometry.solvent_regions_nm[1][2][1] == pytest.approx(shifted_boundary_top + 1.5)
    assert geometry.solvent_regions_nm[1][2][1] < geometry.dimensions_nm[2]


def test_pdbx_writer_terminates_final_loop_for_pymol(tmp_path: Path, monkeypatch) -> None:
    """PyMOL rejects CIF files that end inside the final loop without '#'."""

    backend = importlib.import_module("sammd.backends.interchange")

    class FakePDBxFile:
        @staticmethod
        def writeFile(topology, positions, handle, *, keepIds):  # noqa: N802, N803
            handle.write("loop_\n_atom_site.id\n1\n")

    fake_openmm = SimpleNamespace(app=SimpleNamespace(PDBxFile=FakePDBxFile))
    monkeypatch.setattr(backend, "_require_openmm", lambda: fake_openmm)

    path = tmp_path / "solvated_system.cif"
    backend._write_pdbx(path, topology=object(), positions=object(), overwrite=False)

    assert path.read_text(encoding="utf-8").endswith("\n#\n")


def test_pdb_writer_uses_pdbfile_with_kept_ids(tmp_path: Path, monkeypatch) -> None:
    """PyMOL visualization PDBs should preserve explicit OpenMM connectivity."""

    backend = importlib.import_module("sammd.backends.interchange")
    calls = []

    class FakePDBFile:
        @staticmethod
        def writeFile(topology, positions, handle, *, keepIds):  # noqa: N802, N803
            calls.append((topology, positions, keepIds))
            handle.write("HEADER    PYMOL\nCONECT    1    2\nEND\n")

    fake_openmm = SimpleNamespace(app=SimpleNamespace(PDBFile=FakePDBFile))
    monkeypatch.setattr(backend, "_require_openmm", lambda: fake_openmm)

    path = tmp_path / "solvated_system_pymol.pdb"
    topology = object()
    positions = object()
    backend._write_pdb(path, topology=topology, positions=positions, overwrite=False)

    assert calls == [(topology, positions, True)]
    assert "CONECT" in path.read_text(encoding="utf-8")


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


def test_component_residue_assigner_tracks_semantic_component_chains() -> None:
    """Components should receive chain IDs from semantic roles, not append order."""

    backend = importlib.import_module("sammd.backends.interchange")
    assigner = backend._ComponentResidueAssigner()

    first_ptl = assigner.allocate("sam:propanethiol", "PTL")
    first_mce = assigner.allocate("sam:mercaptoethanol", "MCE")
    first_reactant = assigner.allocate("reactant:cinnamaldehyde", "CIN")
    first_solvent = assigner.allocate("solvent:ethanol", "EOH")
    second_ptl = assigner.allocate("sam:propanethiol", "PTL")

    assert first_ptl.chain_id == "C"
    assert first_ptl.residue_id == 1
    assert first_mce.chain_id == "C"
    assert first_mce.residue_id == 2
    assert first_reactant.chain_id == "B"
    assert first_reactant.residue_id == 1
    assert first_solvent.chain_id == "D"
    assert first_solvent.residue_id == 1
    assert second_ptl.chain_id == "C"
    assert second_ptl.residue_id == 3
    assert assigner.component_ranges["sam:propanethiol"]["residue_count"] == 2
    assert assigner.component_ranges["reactant:cinnamaldehyde"]["chain_ids"] == ("B",)
    assert assigner.component_ranges["sam:mercaptoethanol"]["chain_ids"] == ("C",)
    assert assigner.component_ranges["solvent:ethanol"]["chain_ids"] == ("D",)


def test_openmm_identity_repair_labels_nonmetal_residues() -> None:
    """Nonmetal molecules should not remain UNK/X/0 after Interchange export."""

    backend = importlib.import_module("sammd.backends.interchange")
    from sammd.core.io import AtomRecord

    residue = SimpleNamespace(name="UNK", id="0", chain=SimpleNamespace(id="X"))
    atoms = [SimpleNamespace(name="", residue=residue), SimpleNamespace(name="", residue=residue)]
    topology = SimpleNamespace(atoms=lambda: iter(atoms))
    records = (
        AtomRecord(1, "C1", "C", "EOH", 12, "A", "solvent:ethanol", (0.0, 0.0, 0.0)),
        AtomRecord(2, "O2", "O", "EOH", 12, "A", "solvent:ethanol", (0.1, 0.0, 0.0)),
    )

    backend._apply_openmm_atom_identities(topology, records)

    assert [atom.name for atom in atoms] == ["C1", "O2"]
    assert residue.name == "EOH"
    assert residue.id == "12"
    assert residue.chain.id == "A"


def test_component_residue_assigner_wraps_solvent_from_semantic_chain() -> None:
    """Solvent chain wrapping should begin from the solvent role chain."""

    backend = importlib.import_module("sammd.backends.interchange")
    assigner = backend._ComponentResidueAssigner()

    last_first_chain = None
    for _ in range(backend.MAX_RESIDUES_PER_CHAIN):
        last_first_chain = assigner.allocate("solvent:ethanol", "EOH")
    first_second_chain = assigner.allocate("solvent:ethanol", "EOH")

    assert last_first_chain == backend._ResidueIdentity("D", 9999, "EOH")
    assert first_second_chain == backend._ResidueIdentity("E", 1, "EOH")
    assert assigner.component_ranges["solvent:ethanol"]["chain_ids"] == ("D", "E")


def test_molecule_centers_above_solute_avoid_slab_z_range() -> None:
    """Fallback placement should not put solution atoms inside the slab region."""

    backend = importlib.import_module("sammd.backends.interchange")
    from sammd.backends.openff import PreparedMoleculeTemplate

    template = PreparedMoleculeTemplate(
        molecule=object(),
        positions_nm=((0.0, 0.0, -0.05), (0.0, 0.0, 0.05)),
        atom_symbols=("C", "O"),
    )
    slab_positions = ((0.0, 0.0, 3.9), (1.0, 1.0, 5.5))

    placed = backend._molecule_centers_above_solute(
        template,
        4,
        (4.0, 4.0, 8.0),
        slab_positions,
        clearance_nm=0.25,
    )

    assert len(placed) == 4
    assert min(position[2] for molecule in placed for position in molecule) > 5.5


def test_runtime_solvent_regions_preserve_planned_count_volume() -> None:
    """Runtime PACKMOL placement should preserve planned reservoir volumes."""

    backend = importlib.import_module("sammd.backends.interchange")
    regions = (
        ((-1.0, 1.0), (-1.0, 1.0), (-2.0, -1.5)),
        ((-1.0, 1.0), (-1.0, 1.0), (1.5, 2.0)),
    )
    plan = SimpleNamespace(
        box_plan=SimpleNamespace(
            bounds_nm=((-1.0, 1.0), (-1.0, 1.0), (-2.0, 2.0)),
            solvent_packing_regions_nm=regions,
        )
    )

    runtime_regions = backend._runtime_solvent_regions(plan)

    assert runtime_regions == (
        ((0.0, 2.0), (0.0, 2.0), (0.0, 0.5)),
        ((0.0, 2.0), (0.0, 2.0), (3.5, 4.0)),
    )
    assert sum(_region_volume(region) for region in runtime_regions) == pytest.approx(
        sum(_region_volume(region) for region in regions)
    )


def test_runtime_solvent_regions_match_packmol_input_after_shift(tmp_path, monkeypatch) -> None:
    """Planned solvent regions should be the exact shifted bounds sent to PACKMOL."""

    backend = importlib.import_module("sammd.backends.interchange")
    import sammd.backends.packmol as packmol_backend
    from sammd.backends.packmol import (
        PackmolMoleculeTemplate,
        PackmolSolventComponent,
        pack_fixed_solute_with_solvent_components,
    )
    from sammd.core.io import AtomRecord

    planned_regions = (
        ((-1.0, 1.0), (-1.0, 1.0), (-2.0, -1.5)),
        ((-1.0, 1.0), (-1.0, 1.0), (1.5, 2.0)),
    )
    plan = SimpleNamespace(
        box_plan=SimpleNamespace(
            bounds_nm=((-1.0, 1.0), (-1.0, 1.0), (-2.0, 2.0)),
            solvent_packing_regions_nm=planned_regions,
        )
    )

    monkeypatch.setattr(
        packmol_backend,
        "run_packmol",
        lambda job, input_path, working_dir, stdout_path: SimpleNamespace(
            returncode=0,
            stdout="Success!",
        ),
    )
    monkeypatch.setattr(
        packmol_backend,
        "read_pdb_positions_nm",
        lambda path: ((0.0, 0.0, 0.0), (0.1, 0.1, 0.1), (0.2, 0.2, 3.8)),
    )

    pack_fixed_solute_with_solvent_components(
        solute_records=(
            AtomRecord(1, "Pd", "Pd", "Pdx", 1, "M", "metal_slab", (0.0, 0.0, 0.0)),
        ),
        solvent_components=(
            PackmolSolventComponent(
                "water",
                PackmolMoleculeTemplate("HOH", ((0.0, 0.0, 0.0),), ("O",), ("O1",)),
                2,
            ),
        ),
        dimensions_nm=(2.0, 2.0, 4.0),
        working_dir=tmp_path,
        solvent_regions_nm=backend._runtime_solvent_regions(plan),
    )

    input_text = (tmp_path / "packmol_input.inp").read_text(encoding="utf-8")
    assert "inside box 0 0 0 20 20 5" in input_text
    assert "inside box 0 0 35 20 20 40" in input_text


def test_actual_fixed_solute_geometry_defines_solvent_regions_for_short_sam() -> None:
    """Place solvent next to actual fixed-solute bounds without a clamped SAM gap."""

    backend = importlib.import_module("sammd.backends.interchange")
    from sammd.core.io import AtomRecord

    plan = SimpleNamespace(
        config=SimpleNamespace(
            solvent=SimpleNamespace(padding=3.0),
            packing=SimpleNamespace(packmol=SimpleNamespace(tolerance=1.8)),
        ),
        box_plan=SimpleNamespace(dimensions_nm=(2.0, 2.0, 5.0)),
    )
    records = (
        AtomRecord(1, "Pd", "Pd", "Pdx", 1, "M", "metal_slab", (1.0, 1.0, 1.0)),
        AtomRecord(2, "O", "O", "TGL", 2, "A", "sam:thioglycerol", (1.0, 1.0, 1.62)),
    )

    geometry = backend._runtime_solvent_geometry(plan, records)

    assert geometry.solvent_boundary_z_bounds_nm == pytest.approx((1.5, 2.12))
    assert geometry.fixed_solute_z_bounds_nm == pytest.approx((1.5, 2.12))
    assert geometry.solvent_regions_nm[0][2] == pytest.approx((0.0, 1.32))
    assert geometry.solvent_regions_nm[1][2] == pytest.approx((2.3, 3.62))
    assert geometry.solvent_count_planning_volume_nm3 == pytest.approx(10.56)


def test_reactant_geometry_does_not_define_global_solvent_reservoir() -> None:
    """Keep global solvent regions next to the SAM while reactants stay fixed."""

    backend = importlib.import_module("sammd.backends.interchange")
    from sammd.core.io import AtomRecord

    plan = SimpleNamespace(
        config=SimpleNamespace(
            solvent=SimpleNamespace(padding=3.0),
            packing=SimpleNamespace(packmol=SimpleNamespace(tolerance=1.8)),
        ),
        box_plan=SimpleNamespace(dimensions_nm=(2.0, 2.0, 8.0)),
    )
    slab_sam_records = (
        AtomRecord(1, "Pd", "Pd", "Pdx", 1, "M", "metal_slab", (1.0, 1.0, 1.0)),
        AtomRecord(2, "O", "O", "TGL", 2, "A", "sam:thioglycerol", (1.0, 1.0, 1.62)),
    )
    reactant_records = (
        AtomRecord(3, "C1", "C", "CIN", 3, "B", "reactant:cinnamaldehyde", (1.0, 1.0, 3.0)),
    )

    geometry = backend._runtime_solvent_geometry(
        plan,
        (*slab_sam_records, *reactant_records),
        solvent_boundary_records=slab_sam_records,
    )

    assert geometry.solvent_boundary_z_bounds_nm == pytest.approx((1.5, 2.12))
    assert geometry.fixed_solute_z_bounds_nm == pytest.approx((1.5, 3.5))
    assert geometry.solvent_regions_nm[1][2] == pytest.approx((2.3, 3.62))
    top_gap_nm = geometry.solvent_regions_nm[1][2][0] - geometry.solvent_boundary_z_bounds_nm[1]
    assert top_gap_nm == pytest.approx(0.18)


def test_high_reactant_extends_runtime_box_without_lifting_solvent_region() -> None:
    """Contain a high fixed reactant while keeping solvent anchored to the SAM."""

    backend = importlib.import_module("sammd.backends.interchange")
    from sammd.core.io import AtomRecord

    plan = SimpleNamespace(
        config=SimpleNamespace(
            solvent=SimpleNamespace(padding=3.0),
            packing=SimpleNamespace(packmol=SimpleNamespace(tolerance=1.8)),
        ),
        box_plan=SimpleNamespace(dimensions_nm=(2.0, 2.0, 8.0)),
    )
    slab_sam_records = (
        AtomRecord(1, "Pd", "Pd", "Pdx", 1, "M", "metal_slab", (1.0, 1.0, 1.0)),
        AtomRecord(2, "O", "O", "TGL", 2, "A", "sam:thioglycerol", (1.0, 1.0, 1.62)),
    )
    reactant_records = (
        AtomRecord(3, "C1", "C", "CIN", 3, "B", "reactant:cinnamaldehyde", (1.0, 1.0, 4.5)),
    )

    geometry = backend._runtime_solvent_geometry(
        plan,
        (*slab_sam_records, *reactant_records),
        solvent_boundary_records=slab_sam_records,
    )

    assert geometry.solvent_boundary_z_bounds_nm == pytest.approx((1.5, 2.12))
    assert geometry.fixed_solute_z_bounds_nm == pytest.approx((1.5, 5.0))
    assert geometry.solvent_regions_nm[1][2][0] == pytest.approx(2.3)
    assert geometry.solvent_regions_nm[1][2][1] == pytest.approx(3.62)
    assert geometry.dimensions_nm[2] == pytest.approx(5.18)
    assert geometry.fixed_solute_z_bounds_nm[1] < geometry.dimensions_nm[2]


def test_packmol_fixed_solute_coordinates_are_inside_runtime_box() -> None:
    """Shifted fixed-solute coordinates should fit within Packmol dimensions."""

    backend = importlib.import_module("sammd.backends.interchange")
    from sammd.core.io import AtomRecord

    plan = SimpleNamespace(
        config=SimpleNamespace(
            solvent=SimpleNamespace(padding=3.0),
            packing=SimpleNamespace(packmol=SimpleNamespace(tolerance=1.8)),
        ),
        box_plan=SimpleNamespace(dimensions_nm=(2.0, 2.0, 8.0)),
    )
    boundary_records = (
        AtomRecord(1, "Pd", "Pd", "Pdx", 1, "M", "metal_slab", (1.0, 1.0, 1.0)),
        AtomRecord(2, "O", "O", "TGL", 2, "A", "sam:thioglycerol", (1.0, 1.0, 1.62)),
    )
    records = (
        *boundary_records,
        AtomRecord(3, "C1", "C", "CIN", 3, "B", "reactant", (1.0, 1.0, 4.5)),
    )

    geometry = backend._runtime_solvent_geometry(
        plan,
        records,
        solvent_boundary_records=boundary_records,
    )
    shifted_positions = tuple(
        backend._shift_position_z(record.coordinates_nm, geometry.z_shift_nm) for record in records
    )

    backend._ensure_positions_inside_box(
        shifted_positions,
        geometry.dimensions_nm,
        context="test fixed solute",
    )


def _region_volume(region) -> float:
    """Return simple orthorhombic region volume for test assertions."""

    return (region[0][1] - region[0][0]) * (region[1][1] - region[1][0]) * (
        region[2][1] - region[2][0]
    )


def test_interface_metal_offxml_loads_with_current_openff() -> None:
    """The packaged INTERFACE OFFXML stays compatible with the CUDA env toolkit."""

    pytest.importorskip("openff.toolkit")
    from sammd.backends.openff import interface_fcc_metal_offxml_resource

    force_field_type = importlib.import_module("openff.toolkit").ForceField

    force_field_type(str(interface_fcc_metal_offxml_resource()))
