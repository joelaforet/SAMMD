"""Documentation and notebook scaffold checks."""

import json
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
