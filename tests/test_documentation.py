"""Documentation and notebook scaffold checks."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from sammd import build_system, load_config
from sammd.analysis import analyze_orientation
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
        "## Inspect the plan summary",
        "## Write topology.cif",
        "## Toy orientation analysis",
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
    assert "`positions.cif`, `interchange.json`, and `system.xml`" in content
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


def test_build_contract_documents_first_release_boundary() -> None:
    """Lock the docs page that defines current and reserved build outputs."""

    page = PROJECT_ROOT / "docs" / "source" / "reference" / "build-contract.rst"
    content = page.read_text(encoding="utf-8")

    assert "sammd init" in content
    assert "sammd validate CONFIG" in content
    assert "sammd build CONFIG --output-dir DIR --overwrite" in content
    assert "topology.cif" in content
    assert "positions.cif" in content
    assert "interchange.json" in content
    assert "system.xml" in content
    assert "Full OpenFF/OpenMM construction" in content
    assert "does not own" in content.lower()
    assert "equilibration" in content.lower()
    assert "production simulation" in content.lower()
    assert "* - ``SAMMDBuildPlan``" not in content
    assert "not a top-level public import" in " ".join(content.split())


def test_canonical_workflow_separates_current_and_reserved_artifacts() -> None:
    """Ensure beginner docs do not overstate current topology.cif output."""

    page = PROJECT_ROOT / "docs" / "source" / "tutorials" / "canonical-workflow.rst"
    content = page.read_text(encoding="utf-8")

    assert "``topology.cif`` for a full system" not in content
    assert "topology inspection of the deterministic plan" in content
    assert "future backend construction artifacts" in content


def test_canonical_notebook_workflow_smoke(tmp_path: Path) -> None:
    """Reproduce the notebook workflow using current lightweight package APIs."""

    config_path = tmp_path / "sammd.yaml"
    output_dir = tmp_path / "outputs"
    config_path.write_text(CONFIG_TEMPLATE, encoding="utf-8")

    config = load_config(config_path)
    plan = build_system(config, output_dir=output_dir)
    topology_path = plan.write_topology_cif(overwrite=True)

    toy_coordinates = [
        (-0.30, 0.00, 0.00),
        (-0.10, 0.15, 0.02),
        (0.10, 0.10, 0.05),
        (0.30, 0.00, 0.12),
        (0.45, -0.08, 0.24),
    ]
    orientation = analyze_orientation(
        toy_coordinates,
        atom_index=4,
        masses=[12.0, 12.0, 12.0, 12.0, 16.0],
        side="top",
        reactant_label="toy cinnamaldehyde",
    )

    assert topology_path.is_file()
    assert plan.slab.metal == "Pd"
    assert plan.slab.facet == "111"
    assert len(plan.binding_sites) > 0
    assert len(plan.sam_placements.placements) > 0
    assert plan.solution.molecule_counts["ethanol"] > 0
    assert plan.output_paths.topology == output_dir / "topology.cif"
    assert not plan.full_construction_available
    assert 0.0 <= orientation.angle_degrees <= 180.0
    assert orientation.reactant_label == "toy cinnamaldehyde"
