"""Documentation and notebook scaffold checks."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_docs_scaffold_files_exist() -> None:
    """Check that the Sphinx and ReadTheDocs scaffold files exist."""

    expected_paths = [
        ".readthedocs.yaml",
        "docs/requirements.txt",
        "docs/source/conf.py",
        "docs/source/index.rst",
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
        "## Write planned_slab.cif",
        "## Toy orientation analysis",
    ]
    for heading in expected_headings:
        assert heading in headings
    assert notebook["metadata"]["kernelspec"]["language"] == "python"
