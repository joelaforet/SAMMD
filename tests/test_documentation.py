"""High-signal documentation and notebook checks."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from sammd import build_system, load_config
from sammd.core.config import CONFIG_TEMPLATE

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _notebook_sources(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_docs_scaffold_files_exist_and_tutorials_are_linked() -> None:
    """Keep docs entry points and major tutorials discoverable."""

    expected_paths = [
        ".readthedocs.yaml",
        "docs/requirements.txt",
        "docs/source/conf.py",
        "docs/source/index.rst",
        "docs/source/explanation/scientific-assumptions.rst",
        "docs/source/reference/build-contract.rst",
        "docs/source/tutorials/installation.rst",
        "docs/source/tutorials/canonical-workflow.rst",
        "docs/source/tutorials/openmm-simulation.rst",
        "docs/source/tutorials/yaml-configuration.rst",
        "docs/source/contributor/developer-guide.rst",
        "notebooks/building_systems_with_sammd.ipynb",
        "notebooks/openmm_from_sammd.ipynb",
    ]
    for relative_path in expected_paths:
        assert (PROJECT_ROOT / relative_path).is_file()

    index = (PROJECT_ROOT / "docs" / "source" / "index.rst").read_text(encoding="utf-8")
    for docname in [
        "tutorials/installation",
        "tutorials/canonical-workflow",
        "tutorials/openmm-simulation",
        "tutorials/yaml-configuration",
        "explanation/scientific-assumptions",
        "reference/build-contract",
    ]:
        assert docname in index


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


def test_build_export_and_openmm_ownership_boundary_is_documented() -> None:
    """Guard the central split: SAMMD exports files; OpenMM runs MD."""

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    build_contract = (
        PROJECT_ROOT / "docs" / "source" / "reference" / "build-contract.rst"
    ).read_text(encoding="utf-8")
    project_scope = (PROJECT_ROOT / "docs" / "project-scope.md").read_text(encoding="utf-8")
    combined = "\n".join([readme, build_contract, project_scope])

    assert "SAMMD builds and exports chemistry, structure, and parameter artifacts" in combined
    assert "OpenMM runs minimization, equilibration, production" in combined
    assert "sammd build" in combined
    assert "sammd build --full" not in combined
    assert "not a student-facing SAMMD run-wrapper API" in combined


def test_structure_artifact_terminology_is_documented() -> None:
    """Keep PDBx/mmCIF wording aligned with stable .cif artifact names."""

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    build_contract = (
        PROJECT_ROOT / "docs" / "source" / "reference" / "build-contract.rst"
    ).read_text(encoding="utf-8")
    project_scope = (PROJECT_ROOT / "docs" / "project-scope.md").read_text(encoding="utf-8")
    combined = "\n".join([readme, build_contract, project_scope])

    assert "topology.cif" not in readme
    assert "sam_grafting_density.cif" in readme
    assert "PDBx/mmCIF" in combined
    assert ".cif" in combined
    assert ".mmcif" in combined


def test_pixi_environment_guidance_is_documented() -> None:
    """Keep OpenFF available in common envs and CUDA guidance GPU-specific."""

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    openmm_page = (
        PROJECT_ROOT / "docs" / "source" / "tutorials" / "openmm-simulation.rst"
    ).read_text(encoding="utf-8")
    pixi = (PROJECT_ROOT / "pixi.toml").read_text(encoding="utf-8")

    for content in [readme, openmm_page]:
        assert "nvidia-smi" in content
        assert "cuda-12-4" in content
        assert "cuda-12-6" in content
        assert "pixi run sammd build" in content

    installation = (
        PROJECT_ROOT / "docs" / "source" / "tutorials" / "installation.rst"
    ).read_text(encoding="utf-8")
    assert "pixi shell -e default" in readme
    assert "pixi run" in readme
    assert "pixi shell -e default" in installation
    assert "pixi run" in installation

    for env_name in ["cuda-12-4", "cuda-12-6", "cuda-13-0"]:
        assert env_name in pixi
    assert 'default = ["dev", "interchange"]' in pixi
    assert 'dev = ["dev", "interchange"]' in pixi
    assert 'docs = ["dev", "docs", "interchange"]' in pixi
    assert "science =" not in pixi
    assert "[feature.science" not in pixi


def test_build_tutorial_documents_normal_build_outputs() -> None:
    """Keep the first tutorial clear that build writes normal output files."""

    tutorial = (
        PROJECT_ROOT / "docs" / "source" / "tutorials" / "canonical-workflow.rst"
    ).read_text(encoding="utf-8")
    notebook = _notebook_sources(PROJECT_ROOT / "notebooks" / "building_systems_with_sammd.ipynb")
    combined = tutorial + "\n" + notebook

    for default_output in [
        "sam_grafting_density.cif",
        "build_summary.json",
        "resolved_config.yaml",
    ]:
        assert default_output in combined
    export_outputs = [
        "solvated_system.cif",
        "solvated_system_pymol.pdb",
        "interchange.json",
        "anchor_metadata.json",
    ]
    for export_output in export_outputs:
        assert export_output in combined
    assert "--full" not in combined
    assert "pixi run sammd build" in notebook


def test_openmm_tutorial_teaches_raw_openmm_route() -> None:
    """Guard the core OpenFF Interchange to raw OpenMM teaching path."""

    docs_page = (
        PROJECT_ROOT / "docs" / "source" / "tutorials" / "openmm-simulation.rst"
    ).read_text(encoding="utf-8")
    notebook = _notebook_sources(PROJECT_ROOT / "notebooks" / "openmm_from_sammd.ipynb")

    for content in [docs_page, notebook]:
        assert "Interchange.model_validate_json" in content
        assert "system = interchange.to_openmm()" in content
        assert "topology = interchange.to_openmm_topology()" in content
        assert (
            "positions = interchange.get_positions(include_virtual_sites=True).to_openmm()"
            in content
        )
        assert "LangevinMiddleIntegrator" in content
        assert "simulation.minimizeEnergy()" in content
        assert "DCDReporter" in content
        assert "StateDataReporter" in content
        assert "pd.read_csv" in content
        assert "matplotlib.pyplot" in content
        assert "load_traj" in content
        assert "maxIterations" not in content

    blocked_wrapper_patterns = ["create_openmm_simulation", "sammd.openmm_runtime"]
    for pattern in blocked_wrapper_patterns:
        assert pattern not in docs_page
        assert pattern not in notebook


def test_building_systems_notebook_workflow_smoke(tmp_path: Path) -> None:
    """Reproduce the notebook planning workflow with package APIs."""

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
    assert not plan.output_paths.solvated_system.exists()
    assert not plan.output_paths.openff_interchange.exists()
    assert not plan.output_paths.anchor_metadata.exists()
    assert plan.slab.metal == "Pd"
    assert plan.slab.facet == "111"
    assert len(plan.binding_sites) > 0
    assert len(plan.sam_placements.placements) > 0
    assert plan.solution.molecule_counts["ethanol"] > 0
    assert plan.output_paths.sam_grafting_density == output_dir / "sam_grafting_density.cif"
    assert plan.output_paths.build_summary == output_dir / "build_summary.json"
    assert plan.output_paths.resolved_config == output_dir / "resolved_config.yaml"
    assert not plan.full_construction_available
