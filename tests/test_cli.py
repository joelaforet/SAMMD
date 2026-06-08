"""Tests for the minimal SAMMD CLI."""

from pathlib import Path

from click.testing import CliRunner

from sammd.cli import main
from sammd.core.config import load_config


def test_init_cli_writes_loadable_template() -> None:
    """Write a template YAML file and load it through the public loader."""

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["init", "-o", "sammd.yaml"])
        assert result.exit_code == 0
        assert "INIT" in result.output
        assert "Created a SAMMD configuration template at sammd.yaml" in result.output
        config = load_config("sammd.yaml")
        assert config.surface.metal == "Pd"


def test_init_cli_respects_no_overwrite_unless_force() -> None:
    """Refuse to overwrite existing templates unless --force is supplied."""

    runner = CliRunner()
    with runner.isolated_filesystem():
        runner.invoke(main, ["init", "-o", "sammd.yaml"])
        result = runner.invoke(main, ["init", "-o", "sammd.yaml"])
        assert result.exit_code != 0
        assert "already exists" in result.output

        forced = runner.invoke(main, ["init", "-o", "sammd.yaml", "--force"])
        assert forced.exit_code == 0


def test_validate_cli_accepts_template() -> None:
    """Validate a generated template from the CLI."""

    runner = CliRunner()
    with runner.isolated_filesystem():
        runner.invoke(main, ["init", "-o", "sammd.yaml"])
        result = runner.invoke(main, ["validate", "sammd.yaml"])
        assert result.exit_code == 0
        assert "OK" in result.output
        assert "Configuration valid" in result.output


def test_cli_contract_exposes_only_config_builder_commands() -> None:
    """Keep simulation wrappers out of the first-release CLI surface."""

    assert set(main.commands) == {"build", "init", "validate"}
    for command_name in ("simulate", "equilibrate", "production", "run"):
        assert command_name not in main.commands


def test_build_cli_writes_topology_and_summary() -> None:
    """Build a lightweight plan from the CLI without requiring user Python code."""

    runner = CliRunner()
    with runner.isolated_filesystem():
        runner.invoke(main, ["init", "-o", "sammd.yaml"])

        result = runner.invoke(main, ["build", "sammd.yaml", "--output-dir", "outputs"])

        assert result.exit_code == 0
        assert "SAMMD Build" in result.output
        assert "Reading config and constructing deterministic build plan" in result.output
        assert "OK" in result.output
        assert "Lightweight validation gates passed" in result.output
        assert "Surface: Pd(111)" in result.output
        assert "SAM grafting-density CIF" in result.output
        assert "Build summary" in result.output
        assert "Resolved config" in result.output
        assert "Inspect sam_grafting_density.cif" in result.output
        assert load_config("sammd.yaml").surface.metal == "Pd"

        assert Path("outputs/sam_grafting_density.cif").is_file()
        assert Path("outputs/build_summary.json").is_file()
        assert Path("outputs/resolved_config.yaml").is_file()
        assert not Path("outputs/solvated_system.cif").exists()
        assert not Path("outputs/interchange.json").exists()
        assert not Path("outputs/system.xml").exists()
        assert not Path("outputs/anchor_metadata.json").exists()


def test_build_cli_respects_no_overwrite_unless_requested() -> None:
    """Protect existing build artifacts unless --overwrite is supplied."""

    runner = CliRunner()
    with runner.isolated_filesystem():
        runner.invoke(main, ["init", "-o", "sammd.yaml"])
        runner.invoke(main, ["build", "sammd.yaml", "--output-dir", "outputs"])

        blocked = runner.invoke(main, ["build", "sammd.yaml", "--output-dir", "outputs"])
        assert blocked.exit_code != 0
        assert "refusing to overwrite existing file" in blocked.output

        forced = runner.invoke(
            main, ["build", "sammd.yaml", "--output-dir", "outputs", "--overwrite"]
        )
        assert forced.exit_code == 0


def test_build_help_exposes_full_not_backend_alias() -> None:
    """Show the intuitive MD export flag while hiding the old backend wording."""

    result = CliRunner().invoke(main, ["build", "--help"])

    assert result.exit_code == 0
    assert "--full" in result.output
    assert "--export-backend" not in result.output
