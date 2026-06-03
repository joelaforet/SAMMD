"""Tests for the minimal SAMMD CLI."""

from click.testing import CliRunner

from sammd.cli import main
from sammd.config import load_config


def test_init_cli_writes_loadable_template() -> None:
    """Write a template YAML file and load it through the public loader."""

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["init", "-o", "sammd.yaml"])
        assert result.exit_code == 0
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
        assert "Configuration valid" in result.output
