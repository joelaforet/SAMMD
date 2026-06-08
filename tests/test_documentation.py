"""Documentation and notebook scaffold checks."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from sammd import build_system, load_config
from sammd.core.config import CONFIG_TEMPLATE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_BEGINNER_DOC_PATHS = [
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "docs" / "source" / "index.rst",
    PROJECT_ROOT / "docs" / "source" / "explanation" / "scientific-assumptions.rst",
    PROJECT_ROOT / "docs" / "source" / "reference" / "build-contract.rst",
    PROJECT_ROOT / "docs" / "source" / "tutorials" / "canonical-workflow.rst",
    PROJECT_ROOT / "docs" / "source" / "tutorials" / "yaml-configuration.rst",
]


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

    notebook_path = PROJECT_ROOT / "notebooks" / "building_systems_with_sammd.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    headings = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    expected_headings = [
        "# Recommended SAMMD workflow",
        "## Create a template configuration",
        "## Validate, load, and build a plan",
        "## Write default output files",
        "## Inspect default outputs",
        "## Optional OpenMM/OpenFF files",
        "## Use these files with OpenMM",
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
        "SAMMD builds a physically reasonable starting-structure plan with reproducible "
        "force-field assignments for future MD simulations. The metal-S interaction "
        "is modeled with a tunable, strengthened nonbonded interaction; it is not a "
        "quantum or reactive description of chemisorption."
    )

    assert approved_wording in content


def test_main_docs_preserve_build_export_vs_openmm_run_model() -> None:
    """Keep project-level docs aligned on SAMMD artifact export vs OpenMM runs."""

    required_by_path = {
        PROJECT_ROOT / "README.md": [
            "SAMMD builds and exports chemistry, structure, and parameter artifacts",
            "OpenMM runs minimization, equilibration, production, trajectories, and reporters",
        ],
        PROJECT_ROOT / "docs" / "source" / "index.rst": [
            (
                "building and exporting self-assembled monolayer chemistry, "
                "structure, and parameters"
            ),
            "OpenMM owns minimization, equilibration, production runs, trajectories, and reporters",
        ],
        PROJECT_ROOT / "docs" / "source" / "reference" / "build-contract.rst": [
            "SAMMD builds/exports artifacts; OpenMM owns minimization",
            "Downstream OpenMM simulation scripts are taught separately",
        ],
        PROJECT_ROOT / "docs" / "project-scope.md": [
            "SAMMD builds and exports chemistry, structure, and parameter artifacts",
            (
                "OpenMM runs minimization, equilibration, production MD, "
                "trajectory writing, and reporter setup"
            ),
            "without making SAMMD-owned run wrappers the canonical API",
        ],
    }

    for path, phrases in required_by_path.items():
        normalized = " ".join(path.read_text(encoding="utf-8").split())
        for phrase in phrases:
            assert phrase in normalized, path


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
    assert "sammd build --export-backend" in content
    assert (
        "`positions.cif`, `interchange.json`, `system.xml`, and `anchor_metadata.json`"
        in content
    )
    assert "`Interchange.model_dump_json`" in content
    assert "`Interchange.model_validate_json`" in content
    assert "pre-1.0 Interchange JSON compatibility is not guaranteed across versions" in content
    assert "SAMMD builds and exports chemistry, structure, and parameter artifacts" in content
    assert "OpenMM runs minimization, equilibration, production MD" in content
    assert "not a student-facing SAMMD run-wrapper API" in content
    assert "Lightweight/internal OpenMM utilities may exist" in content
    assert "do not establish student-facing SAMMD ownership" in content


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
        "Future SAMMD releases should provide a user-facing simulation interface",
        "Expose simple methods for minimization, equilibration, production",
        "once SAMMD-owned simulation workflows are added",
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
        r"OpenMM minimization, equilibration, production[^.]*thermodynamic reporting protocols",
        r"those runs are not part of the SAMMD build/export contract",
        r"user-owned OpenMM scripts[^.]*SAMMD-exported artifacts",
        r"DCD[^.]*post-v0\.1\.0/tutorial[^.]*not a v0\.1\.0 build artifact",
        r"OpenMM thermodynamic state data[^.]*post-v0\.1\.0/tutorial-only",
    ]
    for pattern in guarded_patterns:
        assert re.search(pattern, normalized) is not None


def test_project_scope_documents_explicit_backend_export_boundary() -> None:
    """Keep backend export explicit and salt-limited in project scope."""

    page = PROJECT_ROOT / "docs" / "project-scope.md"
    content = page.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    first_release_section = re.search(
        r"Recommended v0\.1\.0 first-release deliverables:(.*?)Defer until after v0\.1\.0:",
        content,
        flags=re.DOTALL,
    )
    assert first_release_section is not None

    assert "record the selected OpenFF small-molecule force field" in content
    assert "The default build remains lightweight" in content
    assert "sammd build --export-backend" in content
    assert "supported non-salt configs" in content
    assert (
        "Salt-containing configs are rejected until salt backend export is implemented"
        in normalized
    )
    assert "OpenFF-compatible OFFXML resource support" in content
    assert "Salt ion backend export" in content
    assert "reserved backend exports until implemented" not in content


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
    assert "--export-backend" in content
    assert "optional science environment" in normalized
    assert "Salt-containing configs are rejected until salt export is implemented" in normalized
    assert "chemistry, structure, and parameter-planning artifacts" in normalized
    assert "SAMMD builds/exports artifacts; OpenMM owns minimization" in normalized
    assert "equilibration" in content.lower()
    assert "production simulation" in content.lower()
    assert "``interchange.json`` as the primary portable system artifact" in normalized
    assert "primary portable OpenFF Interchange export" in normalized
    assert "``Interchange.model_dump_json``" in content
    assert "``Interchange.model_validate_json``" in content
    assert "pre-1.0 Interchange JSON compatibility as not guaranteed" in normalized
    assert "OpenMM convenience export" in normalized
    assert "not the primary portable SAMMD artifact" in normalized
    assert "engine export planning metadata" in normalized
    assert "OpenMM is the student teaching path" in normalized
    assert "``system.xml`` is only a convenience export" in normalized
    assert "GROMACS, LAMMPS, and Amber are reserved only as future downstream exports" in normalized
    assert "not taught in the beginner workflow" in normalized
    assert "human-inspectable/OpenMM-loadable structure file" in normalized
    assert "By default, ``sammd build`` writes only ``topology.cif``" in normalized
    assert "With ``--export-backend`` in the science environment" in normalized
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


def test_yaml_configuration_docs_describe_openmm_export_files() -> None:
    """Keep YAML tutorial aligned with default and OpenMM export files."""

    page = PROJECT_ROOT / "docs" / "source" / "tutorials" / "yaml-configuration.rst"
    content = page.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    assert "used while building the system" not in content
    assert "defines system construction and parameterization only" not in content
    assert "complete OpenMM simulation system" in content
    assert "checks and saves these choices" in normalized
    assert "Names build output files such as ``topology.cif``" in normalized
    assert "OpenMM export files written by ``--export-backend``" in normalized
    assert "``anchor_metadata.json``" in normalized
    assert "``interchange.json`` stores OpenFF Interchange data" in normalized
    assert "this JSON format may change between versions" in normalized
    assert "For this tutorial, use OpenMM" in normalized
    assert "``system.xml`` is an OpenMM file only" in normalized
    assert "GROMACS, LAMMPS, or Amber exports" in normalized
    assert "This version does not include GROMACS, LAMMPS, or Amber exports" in normalized


def test_yaml_configuration_docs_clarify_beginner_schema_limits() -> None:
    """Keep YAML docs clear about SAM chemistry and beginner settings."""

    page = PROJECT_ROOT / "docs" / "source" / "tutorials" / "yaml-configuration.rst"
    content = page.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    assert "Fcc(111) metal surface from the INTERFACE force field" in normalized
    assert "neutral thiol SAM components" in normalized
    assert "HS/implicit-H thiol sulfur" in normalized
    assert "not a pre-deprotonated thiolate" in normalized
    assert "stronger nonbonded interaction" in normalized
    assert "not as a covalent bond or chemical reaction" in normalized
    assert "You cannot change this interaction in this beginner YAML file" in normalized
    assert "``extended_length_nm`` to change the estimated fully extended SAM length" in normalized
    assert "used to size the box" in normalized
    assert "distance in ``z`` from the estimated SAM tips to the box edge" in normalized
    assert "planned box volume to count solvent, reactant, and salt molecules" in normalized


def test_developer_guide_cli_map_includes_build() -> None:
    """Keep contributor package map aligned with available CLI commands."""

    page = PROJECT_ROOT / "docs" / "source" / "contributor" / "developer-guide.rst"
    content = page.read_text(encoding="utf-8")

    assert "``sammd init``, ``sammd validate``, and ``sammd build``" in content


def test_building_systems_notebook_separates_default_and_backend_outputs() -> None:
    """Check tutorial output sections."""

    page = PROJECT_ROOT / "docs" / "source" / "tutorials" / "canonical-workflow.rst"
    content = page.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    expected_sections = [
        "1. Create config",
        "2. Validate config",
        "3. Build the starting model",
        "4. Inspect outputs",
        "5. Optional backend output files",
        "6. Use these files with OpenMM",
        "7. Other engines",
    ]
    for section in expected_sections:
        assert section in content
    assert "SAMMD builds; OpenMM runs" in content
    assert "``topology.cif`` for a full system" not in content
    assert "``topology.cif`` so you can inspect the planned topology" in content
    assert "this command writes exactly three output files" in content
    assert "``resolved_config.yaml`` for the exact validated input used for the build" in content
    assert "default lightweight command does not write" in normalized
    assert "pixi run -e science sammd build" in content
    assert "--export-backend" in content
    assert "``interchange.json`` for the primary OpenFF Interchange export" in normalized
    assert (
        "``system.xml`` for an OpenMM file, not the primary OpenFF Interchange output"
        in normalized
    )
    assert "``anchor_metadata.json`` for SAM anchor metadata" in normalized
    assert "``Interchange.model_dump_json``" in content
    assert "``Interchange.model_validate_json``" in content
    assert "``interchange.to_openmm()``" in content
    assert "pre-1.0 interchange json compatibility" in normalized.lower()
    assert "not guaranteed" in normalized.lower()
    assert "OpenMM is the recommended path for students" in normalized
    assert "optionally use ``system.xml`` only as an OpenMM file" in normalized
    assert "create and run a raw OpenMM ``Simulation``" in normalized
    assert "SAMMD does not include helper wrappers for OpenMM simulations" in normalized
    assert "Interchange may support GROMACS, LAMMPS, and Amber later" in normalized
    assert "they do not have beginner command-line workflows in this version" in normalized
    assert (
        "students use them in their own OpenMM Python API script"
        in normalized
    )
    assert "Configs that include salt are rejected until backend export supports salt" in normalized

    current_outputs_section = re.search(
        r"In this version, this command writes exactly three output files:(.*?)The result includes",
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


def test_building_systems_notebook_other_engines_section_stays_concise() -> None:
    """Keep alternate-engine notes as brief export context, not engine docs."""

    page = PROJECT_ROOT / "docs" / "source" / "tutorials" / "canonical-workflow.rst"
    content = page.read_text(encoding="utf-8")
    section_match = re.search(
        r"^7\. Other engines\n[-]+\n\n(?P<section>.*?)(?=\n\n\S.*\n[-=]+\n|\Z)",
        content,
        flags=re.DOTALL | re.MULTILINE,
    )
    assert section_match is not None

    section = section_match.group("section").strip()
    normalized = " ".join(section.split())
    for term in ["GROMACS", "LAMMPS", "Amber", "Interchange"]:
        assert term in section

    assert len(normalized.split()) <= 35
    assert ".. code-block::" not in section
    assert re.search(r"\b(gmx|mdrun|lmp|sander|pmemd)\b", section) is None


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
    sammd_openmm_wrapper_patterns = [
        r"\bcreate_openmm_simulation\b",
        r"\bsammd\.openmm_runtime\b",
        r"\bOpenMMRuntime\b",
        r"\bSAMMD OpenMM simulation wrapper",
    ]
    executable_non_openmm_engine_patterns = [
        r"^\s*(gmx|gmx_mpi|mdrun|lmp|lmp_mpi|sander|pmemd)\b",
        r"^\s*(from\s+parmed\b|import\s+parmed\b)",
    ]

    for path in tutorial_paths:
        content = path.read_text(encoding="utf-8")
        blocked_patterns = (
            stale_patterns
            + executable_openmm_patterns
            + sammd_openmm_wrapper_patterns
            + executable_non_openmm_engine_patterns
        )
        for pattern in blocked_patterns:
            assert re.search(pattern, content, flags=re.MULTILINE) is None


def test_current_beginner_docs_do_not_teach_unavailable_md_outputs_or_wrappers() -> None:
    """Guard current docs without scanning source/tests or future scope prose."""

    blocked_patterns = [
        r"\btrajectory\.dcd\b",
        r"\bthermodynamics\.csv\b",
        r"^\s*(from\s+openmm\b|import\s+openmm\b)",
        r"\bSimulation\s*\(",
        r"\bcreate_openmm_simulation\b",
        r"\bsammd\.openmm_runtime\b",
        r"\bOpenMMRuntime\b",
    ]
    reporter_names = ["DCDReporter", "StateDataReporter"]
    future_qualifiers = (
        "future",
        "planned",
        "post-v0.1.0",
        "post-0.1.0",
        "after system construction artifacts exist",
        "after full construction",
        "not current",
        "not yet",
    )

    for path in CURRENT_BEGINNER_DOC_PATHS:
        content = path.read_text(encoding="utf-8")
        for pattern in blocked_patterns:
            assert re.search(pattern, content, flags=re.MULTILINE) is None, path

        for paragraph in re.split(r"\n\s*\n", content):
            normalized_paragraph = " ".join(paragraph.split())
            lowered = normalized_paragraph.lower()
            for reporter_name in reporter_names:
                if reporter_name in normalized_paragraph:
                    assert any(qualifier in lowered for qualifier in future_qualifiers), path


def test_canonical_notebook_outputs_match_current_contract() -> None:
    """Keep notebook cells aligned with default and backend export contracts."""

    notebook_path = PROJECT_ROOT / "notebooks" / "building_systems_with_sammd.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    stale_patterns = [
        r"trajectory\.dcd",
        r"thermodynamics\.csv",
        r"plan\.output_paths\.trajectory",
        r"plan\.output_paths\.thermodynamics",
        r"^\s*(from\s+openmm\b|import\s+openmm\b)",
        r"\bSimulation\s*\(",
        r"\bcreate_openmm_simulation\b",
        r"\bsammd\.openmm_runtime\b",
        r"\bOpenMMRuntime\b",
        r"\bSAMMD OpenMM simulation wrapper",
    ]
    for pattern in stale_patterns:
        assert re.search(pattern, sources, flags=re.MULTILINE) is None

    assert "RUN_BACKEND_EXPORT = False" in sources
    assert "--export-backend" in sources
    assert "pixi run -e science" in sources
    assert "`Interchange.model_validate_json`" in sources
    assert "`interchange.to_openmm()`" in sources
    assert "Interchange.model_validate_json(interchange_path.read_text" in sources
    assert "openmm_system = interchange.to_openmm()" in sources
    assert "run an OpenMM `Simulation`" in sources
    assert "This notebook does not create or run an OpenMM simulation" in sources
    assert "does not work with configurations that include salt" in sources
    assert "By default, this notebook does not write" in sources

    for current_output in ["topology.cif", "build_summary.json", "resolved_config.yaml"]:
        assert current_output in sources
    for backend_output in [
        "positions.cif",
        "interchange.json",
        "system.xml",
        "anchor_metadata.json",
    ]:
        assert backend_output in sources


def test_openmm_simulation_tutorial_and_notebook_exist_and_are_linked() -> None:
    """Keep the OpenMM-from-SAMMD teaching files visible."""

    docs_page = PROJECT_ROOT / "docs" / "source" / "tutorials" / "openmm-simulation.rst"
    notebook_path = PROJECT_ROOT / "notebooks" / "openmm_from_sammd.ipynb"
    index = PROJECT_ROOT / "docs" / "source" / "index.rst"

    assert docs_page.is_file()
    assert notebook_path.is_file()
    assert "tutorials/openmm-simulation" in index.read_text(encoding="utf-8")

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["metadata"]["kernelspec"]["language"] == "python"


def test_openmm_simulation_tutorial_teaches_required_raw_openmm_route() -> None:
    """Guard the required Interchange-to-raw-OpenMM workflow."""

    docs_page = PROJECT_ROOT / "docs" / "source" / "tutorials" / "openmm-simulation.rst"
    notebook_path = PROJECT_ROOT / "notebooks" / "openmm_from_sammd.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook_sources = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    docs_content = docs_page.read_text(encoding="utf-8")
    contents = [docs_content, notebook_sources]
    combined = "\n".join(contents)
    normalized = " ".join(combined.split())

    required_strings = [
        "from openff.interchange import Interchange",
        'Interchange.model_validate_json(interchange_path.read_text(encoding="utf-8"))',
        "system = interchange.to_openmm()",
        "topology = interchange.to_openmm_topology()",
        "positions = interchange.get_positions(include_virtual_sites=True).to_openmm()",
        "LangevinMiddleIntegrator",
        "Simulation(topology, system, integrator)",
        "SAMMD builds/exports files",
        "OpenMM runs minimization, equilibration, production, trajectories, and reporters",
        "Use the raw OpenMM Python API",
        "not a SAMMD OpenMM wrapper",
    ]
    for content in contents:
        content_normalized = " ".join(content.split())
        for required_string in required_strings:
            assert required_string in content_normalized

    assert "``system.xml`` is an OpenMM convenience output" in normalized
    assert "post-Interchange metal-S pair edits" in normalized
    assert "present only in ``system.xml`` unless you apply the same changes again" in normalized
    assert "Do not assume ``interchange.json`` contains every later OpenMM-only change" in normalized

    blocked_wrapper_patterns = [
        r"\bcreate_openmm_simulation\b",
        r"\bsammd\.openmm_runtime\b",
        r"\bOpenMMRuntime\b",
    ]
    for pattern in blocked_wrapper_patterns:
        assert re.search(pattern, combined) is None


def test_openmm_simulation_tutorial_covers_minimization_steps_reporters_and_plots() -> None:
    """Guard the student workflow details in the new tutorial and notebook."""

    docs_page = PROJECT_ROOT / "docs" / "source" / "tutorials" / "openmm-simulation.rst"
    notebook_path = PROJECT_ROOT / "notebooks" / "openmm_from_sammd.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook_sources = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    docs_content = docs_page.read_text(encoding="utf-8")
    contents = [docs_content, notebook_sources]
    combined = "\n".join(contents)
    normalized = " ".join(combined.split())

    for content in contents:
        assert "production_time_ns = 10.0" in content
        assert "desired_trajectory_frames = 300" in content
        assert "desired_thermo_points = 1000" in content
        assert "def steps_from_time" in content
        assert "def interval_from_count" in content
        assert "equilibration_steps = steps_from_time" in content
        assert "production_steps = steps_from_time" in content
        assert "trajectory_interval = interval_from_count" in content
        assert "thermo_interval = interval_from_count" in content
    for content in contents:
        assert "unit math" in content

    for content in contents:
        assert "initial_state = simulation.context.getState(getEnergy=True)" in content
        assert "math.isfinite" in content
        assert "simulation.minimizeEnergy()" in content
        assert "simulation.context.setVelocitiesToTemperature(temperature)" in content
        assert "simulation.step(equilibration_steps)" in content
        assert "simulation.step(production_steps)" in content
        assert "DCDReporter" in content
        assert "StateDataReporter" in content
        assert "import pandas as pd" in content
        assert "import matplotlib.pyplot as plt" in content
        assert "pd.read_csv" in content
        assert "plt.plot" in content
    for content in contents:
        content_normalized = " ".join(content.split())
        assert "large positive number" in content_normalized
        assert "energy is finite" in content_normalized
    assert "maxIterations" not in combined


def test_openmm_simulation_tutorial_includes_pymol_and_optional_npt_details() -> None:
    """Keep viewer and optional pressure-control guidance present."""

    docs_page = PROJECT_ROOT / "docs" / "source" / "tutorials" / "openmm-simulation.rst"
    notebook_path = PROJECT_ROOT / "notebooks" / "openmm_from_sammd.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook_sources = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    docs_content = docs_page.read_text(encoding="utf-8")
    contents = [docs_content, notebook_sources]
    combined = "\n".join(contents)

    for content in contents:
        content_normalized = " ".join(content.split())
        assert "load_traj outputs/trajectory.dcd, sammd_system" in content
        assert "load outputs/positions.cif, sammd_system" in content
        assert "NVT" in content
        assert "MonteCarloBarostat" in content
        assert "MonteCarloAnisotropicBarostat" in content
        assert "Choose one of these examples, not both" in content_normalized
        assert "before creating" in content_normalized
        assert "fixed" in content_normalized
        assert "allowed" in content_normalized or "allow only" in content_normalized
        assert "A barostat does not replace the thermostat" in content
    assert "system.addForce(MonteCarloBarostat" in docs_content
    assert re.search(r"system\.addForce\(\s+MonteCarloAnisotropicBarostat", docs_content)
    assert "system.addForce(MonteCarloBarostat" in notebook_sources
    assert re.search(r"system\.addForce\(\s+MonteCarloAnisotropicBarostat", notebook_sources)


def test_docs_index_prose_avoids_openmm_student_blocked_terms() -> None:
    """Keep the landing-page prose aligned with current student wording."""

    index = (PROJECT_ROOT / "docs" / "source" / "index.rst").read_text(encoding="utf-8")
    prose = "\n".join(line for line in index.splitlines() if not line.startswith("   "))

    blocked_terms = [r"\bartifacts?\b", r"\bcontract\b"]
    for term in blocked_terms:
        assert re.search(term, prose, flags=re.IGNORECASE) is None


def test_openmm_simulation_student_prose_avoids_blocked_terms() -> None:
    """Avoid confusing terms in only the new student-facing OpenMM material."""

    docs_page = PROJECT_ROOT / "docs" / "source" / "tutorials" / "openmm-simulation.rst"
    notebook_path = PROJECT_ROOT / "notebooks" / "openmm_from_sammd.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook_sources = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    combined = docs_page.read_text(encoding="utf-8") + "\n" + notebook_sources

    blocked_terms = [
        r"\bcanonical\b",
        r"\bartifacts?\b",
        r"\bhandoff\b",
        r"\bboundary\b",
        r"\bcontract\b",
        r"\bportable\b",
        r"\bknob\b",
    ]
    for term in blocked_terms:
        assert re.search(term, combined, flags=re.IGNORECASE) is None


def test_building_systems_notebook_workflow_smoke(tmp_path: Path) -> None:
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
