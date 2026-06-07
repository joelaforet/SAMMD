"""Documentation and notebook scaffold checks."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from sammd import build_system, load_config
from sammd.config import CONFIG_TEMPLATE

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_docs_scaffold_files_exist() -> None:
    """Check that the Sphinx and ReadTheDocs scaffold files exist."""

    expected_paths = [
        ".readthedocs.yaml",
        "docs/requirements.txt",
        "docs/source/conf.py",
        "docs/source/index.rst",
        "docs/source/reference/build-contract.rst",
        "docs/source/tutorials/canonical-workflow.rst",
        "docs/source/tutorials/yaml-configuration.rst",
        "docs/source/contributor/developer-guide.rst",
    ]
    for relative_path in expected_paths:
        assert (PROJECT_ROOT / relative_path).is_file()


def test_canonical_notebook_has_expected_sections() -> None:
    """Validate notebook JSON and expected tutorial section headings."""

    notebook_path = PROJECT_ROOT / "notebooks" / "canonical_workflow.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    headings = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    expected_headings = [
        "# SAMMD canonical lightweight workflow",
        "## Create a template configuration",
        "## Validate, load, and build a plan",
        "## Write current artifacts",
        "## Inspect current outputs",
        "## Reserved future backend artifacts",
        "## Future OpenMM handoff",
    ]
    for heading in expected_headings:
        assert heading in headings
    assert notebook["metadata"]["kernelspec"]["language"] == "python"


def test_sphinx_docs_build_without_warnings(tmp_path: Path) -> None:
    """Build Sphinx docs with warnings treated as errors when Sphinx is available."""

    pytest.importorskip("sphinx")
    source_dir = PROJECT_ROOT / "docs" / "source"
    build_dir = tmp_path / "html"
    result = subprocess.run(
        [sys.executable, "-m", "sphinx", "-W", "-b", "html", str(source_dir), str(build_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (build_dir / "index.html").is_file()


def test_project_scope_page_uses_published_safe_link() -> None:
    """Ensure project-scope docs avoid source-relative links that break in HTML."""

    page = PROJECT_ROOT / "docs" / "source" / "explanation" / "project-scope.rst"
    content = page.read_text(encoding="utf-8")
    assert "../../project-scope.md" not in content
    assert "https://github.com/joelaforet/SAMMD/blob/main/docs/project-scope.md" in content


def test_scientific_assumptions_page_exists_and_is_linked() -> None:
    """Keep the beginner-facing assumptions page in the Explanation section."""

    page = PROJECT_ROOT / "docs" / "source" / "explanation" / "scientific-assumptions.rst"
    index = PROJECT_ROOT / "docs" / "source" / "index.rst"

    assert page.is_file()
    assert "explanation/scientific-assumptions" in index.read_text(encoding="utf-8")


def test_readme_includes_approved_scientific_assumptions_wording() -> None:
    """Lock the exact approved README science-boundary wording."""

    page = PROJECT_ROOT / "README.md"
    content = page.read_text(encoding="utf-8")
    approved_wording = (
        "SAMMD builds a physically reasonable starting structure with reproducible "
        "force-field assignments for running MD simulations. The metal-S interaction "
        "is modeled with a tunable, strengthened nonbonded interaction; it is not a "
        "quantum or reactive description of chemisorption."
    )

    assert approved_wording in content


def test_scientific_assumptions_document_current_model_boundaries() -> None:
    """Document placement, force-field, backend-export, and simulation boundaries."""

    page = PROJECT_ROOT / "docs" / "source" / "explanation" / "scientific-assumptions.rst"
    content = page.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    required_phrases = [
        "registered Fcc(111) metal surfaces, defaulting to Pd(111)",
        "slab is centered at the origin",
        "SAMs placed normal to the surface: +z for the top face and -z for the bottom face",
        "internal ``fcc_hollow`` default",
        "selects sulfur pairs to the three nearest hollow-site metal atoms",
        "neutral thiols with an HS/implicit-H thiol sulfur",
        "not pre-deprotonated thiolates",
        "Padding is measured from the fully extended SAM tips",
        "Base metal Lennard-Jones parameters come from the INTERFACE Fcc metal data",
        "target route for organic molecules is OpenFF",
        "records and validates force-field choices",
        "does not yet export a full OpenFF/OpenMM backend system",
        "internal, post-export proxy",
        "not currently a beginner YAML knob",
        "does not own minimization, equilibration, production, trajectory writing, or "
        "reporter setup",
    ]
    for phrase in required_phrases:
        assert phrase in normalized


def test_scientific_assumptions_do_not_overclaim_current_backend_or_simulations() -> None:
    """Prevent assumptions docs from teaching unavailable exports or MD ownership."""

    page = PROJECT_ROOT / "docs" / "source" / "explanation" / "scientific-assumptions.rst"
    content = page.read_text(encoding="utf-8")
    stale_patterns = [
        r"SAMMD exports? a full OpenFF/OpenMM backend system",
        r"SAMMD constructs? complete OpenMM systems",
        r"SAMMD owns? minimization",
        r"SAMMD runs? equilibration",
        r"SAMMD runs? production",
        r"SAMMD writes? trajectories",
        r"SAMMD configures? reporters",
        r"covalent metal-S bond",
        r"is a quantum description",
        r"is a reactive description",
        r"is currently a beginner YAML knob",
    ]

    for pattern in stale_patterns:
        assert re.search(pattern, content, flags=re.IGNORECASE) is None


def test_beginner_docs_do_not_use_stale_fcc_or_hcp_wording() -> None:
    """Prevent beginner docs from exposing stale hollow-site selection wording."""

    beginner_doc_paths = [
        PROJECT_ROOT / "docs" / "source" / "explanation" / "scientific-assumptions.rst",
        PROJECT_ROOT / "docs" / "source" / "tutorials" / "canonical-workflow.rst",
    ]

    for path in beginner_doc_paths:
        assert "fcc or hcp" not in path.read_text(encoding="utf-8")


def test_project_scope_source_matches_first_release_contract() -> None:
    """Keep the source-of-truth scope doc aligned with v0.1.0 outputs."""

    page = PROJECT_ROOT / "docs" / "project-scope.md"
    content = page.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    assert "planned_slab.cif" not in content
    assert "topology.cif remains reserved" not in normalized
    assert "sammd build` writes" in content
    assert "`topology.cif`: lightweight topology-inspection CIF" in content
    assert "`build_summary.json`: machine-readable summary" in content
    assert "`resolved_config.yaml`: validated YAML configuration" in content
    assert (
        "`positions.cif`, `interchange.json`, `system.xml`, and `anchor_metadata.json`"
        in content
    )
    assert "`Interchange.model_dump_json`" in content
    assert "`Interchange.model_validate_json`" in content
    assert "pre-1.0 Interchange JSON compatibility is not guaranteed across versions" in content
    assert "Simulation wrappers are post-v0.1.0 target work" in content
    assert "excluded from the v0.1.0 first-release contract" in content
    assert "Lightweight/internal OpenMM utilities may exist" in content
    assert "does not expose student-facing SAMMD ownership" in content


def test_project_scope_keeps_simulation_work_out_of_current_scope() -> None:
    """Prevent stale MVP/current-scope ownership of OpenMM simulation outputs."""

    page = PROJECT_ROOT / "docs" / "project-scope.md"
    content = page.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    stale_phrases = [
        "Recommended MVP deliverables",
        "OpenMM setup, and basic inspection",
        "OpenMM thermodynamic reporting during simulation, configurable by the user",
        "DCD should be the canonical trajectory format for the MVP",
        "DCD is the default trajectory output format.",
        "YAML template includes configurable output/reporting sections for mmCIF, DCD, "
        "and thermodynamic state data",
        "The current release does not provide `create_openmm_simulation`, "
        "minimization, equilibration",
    ]
    for phrase in stale_phrases:
        assert phrase not in content

    first_release_section = re.search(
        r"Recommended v0\.1\.0 first-release deliverables:(.*?)Defer until after v0\.1\.0:",
        content,
        flags=re.DOTALL,
    )
    assert first_release_section is not None
    first_release_deliverables = first_release_section.group(1)
    assert "OpenMM setup" not in first_release_deliverables
    assert "DCD" not in first_release_deliverables
    assert "thermodynamic reporting" not in first_release_deliverables

    guarded_patterns = [
        r"post-v0\.1\.0[^.]*OpenMM setup",
        r"DCD[^.]*post-v0\.1\.0/tutorial[^.]*not a v0\.1\.0 build artifact",
        r"OpenMM thermodynamic state data[^.]*post-v0\.1\.0/tutorial-only",
    ]
    for pattern in guarded_patterns:
        assert re.search(pattern, normalized) is not None


def test_project_scope_keeps_parameterization_backend_out_of_first_release() -> None:
    """Prevent stale v0.1.0 claims of full parameterization/backends."""

    page = PROJECT_ROOT / "docs" / "project-scope.md"
    content = page.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    first_release_section = re.search(
        r"Recommended v0\.1\.0 first-release deliverables:(.*?)Defer until after v0\.1\.0:",
        content,
        flags=re.DOTALL,
    )
    assert first_release_section is not None
    first_release_deliverables = first_release_section.group(1)

    stale_patterns = [
        r"OpenFF/SMIRNOFF parameterization",
        r"full OpenFF",
        r"backend construction",
        r"OpenFF Interchange",
        r"OpenMM backend",
    ]
    for pattern in stale_patterns:
        assert re.search(pattern, first_release_deliverables, flags=re.IGNORECASE) is None

    assert "record the selected OpenFF small-molecule force field" in content
    assert "without constructing a parameterized backend system" in content
    assert "OpenFF-compatible OFFXML resource support" in content
    assert re.search(
        r"Full OpenFF/SMIRNOFF parameterization[^.]*backend construction/export",
        normalized,
    ) is not None


def test_readme_demo_uses_neutral_thiol_sam_wording() -> None:
    """Prevent stale demo wording from teaching thiolate as configured SAM input."""

    page = PROJECT_ROOT / "README.md"
    content = page.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    assert "Pd(111)/propanethiolate/" not in content
    assert "one propanethiolate residue per SAM molecule" not in content
    assert "Pd(111)/propanethiol SAM input with cinnamaldehyde and ethanol" in normalized
    assert "propanethiol-derived thiol SAM residue" in normalized


def test_project_scope_keeps_metal_s_attachment_internal_for_mvp() -> None:
    """Prevent stale user-configurable metal-S MVP wording from returning."""

    page = PROJECT_ROOT / "docs" / "project-scope.md"
    content = page.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    stale_phrases = [
        "sulfur-metal interaction scaling factor must be user-configurable",
        "Keep the scale factor in user configuration",
        "users should only need to set the interaction magnitude",
        "Increase the relevant sulfur-metal interaction strength by a configurable factor",
    ]
    for phrase in stale_phrases:
        assert phrase not in content

    assert "beginner users should not tune the interaction magnitude" in normalized
    assert "backend/internal representation, not a beginner YAML knob" in normalized
    assert (
        "user-configurable scale factor belongs in a future advanced attachment API"
        in normalized
    )


def test_project_scope_clarifies_neutral_thiol_beginner_inputs() -> None:
    """Keep beginner scope guidance from teaching configured thiolate inputs."""

    page = PROJECT_ROOT / "docs" / "project-scope.md"
    content = page.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    assert "neutral thiols with an HS/implicit-H thiol sulfur" in normalized
    assert "should not provide pre-deprotonated thiolate inputs" in normalized
    assert "SAMMD uses the sulfur atom for placement" in normalized


def test_project_scope_keeps_anchor_site_out_of_current_template_scope() -> None:
    """Prevent stale current YAML/template anchor-site configurability wording."""

    page = PROJECT_ROOT / "docs" / "project-scope.md"
    content = page.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    stale_phrases = [
        "General metal facet support beyond the first Pd(111) path.",
        '`anchor.site = "fcc_hollow"` should be the internal Pd(111) default strategy',
        "YAML template defaults include `fcc_hollow`",
        "The default sulfur site for Pd(111) should be `fcc_hollow`, "
        "but site type must be configurable.",
        "keep the adsorption site configurable",
    ]
    for phrase in stale_phrases:
        assert phrase not in content

    assert "General surface support beyond registered Fcc(111) metals" in normalized
    assert "including non-111 facets" in normalized
    assert "internal modeling hypothesis that defaults to Pd(111)" in normalized
    assert "internal registered Fcc(111) hollow-placement strategy" in normalized
    assert "Pd(111) as the canonical/default surface" in normalized
    assert "internal builder default rather than a beginner template field" in normalized
    assert "user-configurable site type belongs in a future advanced attachment API" in normalized


def test_build_contract_documents_first_release_boundary() -> None:
    """Lock the docs page that defines current and reserved build outputs."""

    page = PROJECT_ROOT / "docs" / "source" / "reference" / "build-contract.rst"
    content = page.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    assert "sammd init" in content
    assert "sammd validate CONFIG" in content
    assert "sammd build CONFIG --output-dir DIR --overwrite" in content
    assert "topology.cif" in content
    assert "positions.cif" in content
    assert "interchange.json" in content
    assert "system.xml" in content
    assert "anchor_metadata.json" in content
    assert "Full OpenFF/OpenMM construction" in normalized
    assert "does not own" in content.lower()
    assert "equilibration" in content.lower()
    assert "production simulation" in content.lower()
    assert "``interchange.json`` as the primary portable system artifact" in normalized
    assert "primary portable OpenFF Interchange export" in normalized
    assert "``Interchange.model_dump_json``" in content
    assert "``Interchange.model_validate_json``" in content
    assert "pre-1.0 Interchange JSON compatibility as not guaranteed" in normalized
    assert "OpenMM convenience export" in normalized
    assert "not the primary portable SAMMD artifact" in normalized
    assert "reserved engine export planning metadata" in normalized
    assert "OpenMM is the student teaching path" in normalized
    assert "``system.xml`` is only a convenience export" in normalized
    assert "GROMACS, LAMMPS, and Amber are reserved only as future downstream exports" in normalized
    assert "not taught in the beginner workflow" in normalized
    assert "human-inspectable/OpenMM-loadable structure file" in normalized
    assert (
        "does not write ``positions.cif``, ``interchange.json``, ``system.xml``, "
        "or ``anchor_metadata.json``" in normalized
    )
    assert "* - ``SAMMDBuildPlan``" not in content
    assert "not a top-level public import" in " ".join(content.split())


def test_build_contract_documents_deferred_backend_validation_gates() -> None:
    """Lock the planned backend validation gates without requiring them now."""

    page = PROJECT_ROOT / "docs" / "source" / "reference" / "build-contract.rst"
    content = page.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    required_phrases = [
        "skipped/not required when optional dependencies or backend artifacts are absent",
        "``interchange.json`` reloads with ``Interchange.model_validate_json``",
        "reloaded ``Interchange`` exports to an OpenMM ``System``",
        "Topology atom count, positions atom count, and OpenMM ``System`` particle count agree",
        "``system.xml`` deserializes if written and its particle count agrees",
        "Minimization produces finite energies and the final energy is not increased",
    ]

    for phrase in required_phrases:
        assert phrase in normalized


def test_yaml_configuration_docs_keep_backend_exports_reserved() -> None:
    """Keep YAML tutorial aligned with current lightweight build behavior."""

    page = PROJECT_ROOT / "docs" / "source" / "tutorials" / "yaml-configuration.rst"
    content = page.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    assert "used while building the system" not in content
    assert "defines system construction and parameterization only" not in content
    assert "future backend export" in content
    assert "validates and records these choices" in content
    assert "current build artifacts such as ``topology.cif``" in normalized
    assert "future backend artifact names such as ``positions.cif``" in normalized
    assert "``anchor_metadata.json``" in normalized
    assert "``Interchange.model_dump_json``" in content
    assert "``Interchange.model_validate_json``" in content
    assert "pre-1.0 Interchange JSON compatibility is not guaranteed" in normalized
    assert "OpenMM is the student teaching path" in normalized
    assert "``system.xml`` is only a convenience export" in normalized
    assert "GROMACS, LAMMPS, and Amber are future downstream exports" in normalized
    assert "not beginner workflow commands" in normalized


def test_yaml_configuration_docs_clarify_beginner_schema_boundary() -> None:
    """Keep YAML docs clear about SAM chemistry and deferred student knobs."""

    page = PROJECT_ROOT / "docs" / "source" / "tutorials" / "yaml-configuration.rst"
    content = page.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    assert "registered Fcc(111) INTERFACE surface" in normalized
    assert "neutral thiol SAM components" in normalized
    assert "HS/implicit-H thiol sulfur" in normalized
    assert "not a pre-deprotonated thiolate" in normalized
    assert "strengthened nonbonded interaction" in normalized
    assert "not as covalent, quantum," in normalized
    assert "not yet a student-facing YAML knob" in normalized
    assert "optional ``extended_length_nm``" in normalized
    assert "fully extended SAM length used for box planning" in normalized
    assert "requested z distance from fully extended SAM tips to the box boundary" in normalized
    assert "same planned box volume is used for solvent, reactant, and salt counts" in normalized


def test_developer_guide_cli_map_includes_build() -> None:
    """Keep contributor package map aligned with available CLI commands."""

    page = PROJECT_ROOT / "docs" / "source" / "contributor" / "developer-guide.rst"
    content = page.read_text(encoding="utf-8")

    assert "``sammd init``, ``sammd validate``, and ``sammd build``" in content


def test_canonical_workflow_separates_current_and_reserved_artifacts() -> None:
    """Ensure beginner docs separate current outputs from reserved artifacts."""

    page = PROJECT_ROOT / "docs" / "source" / "tutorials" / "canonical-workflow.rst"
    content = page.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    expected_sections = [
        "1. Create config",
        "2. Validate config",
        "3. Build system/plan",
        "4. Inspect current outputs",
        "5. Reserved future backend artifacts",
        "6. Future OpenMM handoff",
        "7. Other engines",
    ]
    for section in expected_sections:
        assert section in content
    assert "SAMMD builds; OpenMM runs" in content
    assert "``topology.cif`` for a full system" not in content
    assert "topology inspection of the deterministic plan" in content
    assert "Today this command writes exactly three build artifacts" in content
    assert "``resolved_config.yaml`` for the exact validated input used for the build" in content
    assert "reserved target artifacts, not current outputs" in normalized
    assert "``interchange.json`` for the primary portable OpenFF Interchange export" in normalized
    assert (
        "``system.xml`` for an OpenMM convenience export, not the primary portable artifact"
        in normalized
    )
    assert "``anchor_metadata.json`` for SAM anchor metadata export" in normalized
    assert "``Interchange.model_dump_json``" in content
    assert "``Interchange.model_validate_json``" in content
    assert "pre-1.0 Interchange JSON compatibility is not guaranteed" in normalized
    assert "OpenMM is the student teaching path" in normalized
    assert "optionally use ``system.xml`` only as a convenience OpenMM system export" in normalized
    assert "GROMACS, LAMMPS, and Amber are future downstream exports" in normalized
    assert "not beginner workflow commands" in normalized
    assert "students will hand those build artifacts to their own OpenMM Python API script" in normalized
    assert "That handoff is not runnable in this lightweight release" in normalized

    current_outputs_section = re.search(
        r"Today this command writes exactly three build artifacts:(.*?)The returned build plan",
        content,
        flags=re.DOTALL,
    )
    assert current_outputs_section is not None
    current_outputs = current_outputs_section.group(1)
    for current_output in ["topology.cif", "build_summary.json", "resolved_config.yaml"]:
        assert current_output in current_outputs
    for reserved_output in [
        "positions.cif",
        "interchange.json",
        "system.xml",
        "anchor_metadata.json",
    ]:
        assert reserved_output not in current_outputs


def test_tutorial_docs_do_not_teach_current_md_outputs_or_openmm_code() -> None:
    """Keep beginner tutorials aligned with the lightweight build contract."""

    tutorial_paths = [
        PROJECT_ROOT / "docs" / "source" / "tutorials" / "canonical-workflow.rst",
        PROJECT_ROOT / "docs" / "source" / "tutorials" / "yaml-configuration.rst",
    ]
    stale_patterns = [
        r"trajectory\.dcd",
        r"thermodynamics\.csv",
        r"plan\.output_paths\.trajectory",
        r"plan\.output_paths\.thermodynamics",
    ]
    executable_openmm_patterns = [
        r"^\s*(from\s+openmm\b|import\s+openmm\b)",
        r"\bSimulation\s*\(",
    ]
    executable_non_openmm_engine_patterns = [
        r"^\s*(gmx|gmx_mpi|mdrun|lmp|lmp_mpi|sander|pmemd)\b",
        r"^\s*(from\s+parmed\b|import\s+parmed\b)",
    ]

    for path in tutorial_paths:
        content = path.read_text(encoding="utf-8")
        blocked_patterns = (
            stale_patterns + executable_openmm_patterns + executable_non_openmm_engine_patterns
        )
        for pattern in blocked_patterns:
            assert re.search(pattern, content, flags=re.MULTILINE) is None


def test_canonical_notebook_outputs_match_current_contract() -> None:
    """Keep notebook cells from showing stale MD outputs or OpenMM scripts."""

    notebook_path = PROJECT_ROOT / "notebooks" / "canonical_workflow.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    stale_patterns = [
        r"trajectory\.dcd",
        r"thermodynamics\.csv",
        r"plan\.output_paths\.trajectory",
        r"plan\.output_paths\.thermodynamics",
        r"^\s*(from\s+openmm\b|import\s+openmm\b)",
        r"\bSimulation\s*\(",
    ]
    for pattern in stale_patterns:
        assert re.search(pattern, sources, flags=re.MULTILINE) is None

    for current_output in ["topology.cif", "build_summary.json", "resolved_config.yaml"]:
        assert current_output in sources
    for reserved_output in [
        "positions.cif",
        "interchange.json",
        "system.xml",
        "anchor_metadata.json",
    ]:
        assert reserved_output in sources


def test_canonical_notebook_workflow_smoke(tmp_path: Path) -> None:
    """Reproduce the notebook workflow using current lightweight package APIs."""

    config_path = tmp_path / "sammd.yaml"
    output_dir = tmp_path / "outputs"
    config_path.write_text(CONFIG_TEMPLATE, encoding="utf-8")

    config = load_config(config_path)
    plan = build_system(config, output_dir=output_dir)
    topology_path = plan.write_topology_cif(overwrite=True)
    build_summary_path = plan.write_build_summary(overwrite=True)
    resolved_config_path = plan.write_resolved_config(overwrite=True)

    assert topology_path.is_file()
    assert build_summary_path.is_file()
    assert resolved_config_path.is_file()
    assert not plan.output_paths.positions.exists()
    assert not plan.output_paths.openff_interchange.exists()
    assert not plan.output_paths.openmm_system.exists()
    assert plan.slab.metal == "Pd"
    assert plan.slab.facet == "111"
    assert len(plan.binding_sites) > 0
    assert len(plan.sam_placements.placements) > 0
    assert plan.solution.molecule_counts["ethanol"] > 0
    assert plan.output_paths.topology == output_dir / "topology.cif"
    assert plan.output_paths.build_summary == output_dir / "build_summary.json"
    assert plan.output_paths.resolved_config == output_dir / "resolved_config.yaml"
    assert not plan.full_construction_available
