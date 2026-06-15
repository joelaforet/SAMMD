"""Tests for the minimal SAMMD CLI."""

import re
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from sammd.cli import main
from sammd.core.config import load_config

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _patch_interchange_export(monkeypatch) -> None:
    """Keep CLI tests focused without constructing real OpenFF objects."""

    def fake_export_interchange_backend(plan, overwrite: bool = False):
        files = {
            "solvated_system": plan.output_paths.solvated_system,
            "pymol_system": plan.output_paths.pymol_system,
            "openff_interchange": plan.output_paths.openff_interchange,
            "anchor_metadata": plan.output_paths.anchor_metadata,
        }
        for path in files.values():
            path.write_text("mock export\n", encoding="utf-8")
        return SimpleNamespace(
            files=files,
            runtime_solvent_geometry=SimpleNamespace(molecule_counts={"ethanol": 7}),
        )

    monkeypatch.setattr("sammd.cli.export_interchange_backend", fake_export_interchange_backend)
    monkeypatch.setattr(
        "sammd.cli.backend_build_summary",
        lambda plan, export_result: {"backend_export": {"mode": "mock"}},
    )


def test_init_cli_writes_loadable_template() -> None:
    """Write a default project template and load it through the public loader."""

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["init"])
        assert result.exit_code == 0
        assert "INIT" in result.output
        assert (
            "Created a SAMMD configuration template at sammd-project/sammd.yaml"
            in result.output
        )
        config = load_config("sammd-project/sammd.yaml")
        assert config.surface.metal == "Pd"


def test_init_cli_writes_template_to_output_directory() -> None:
    """Treat -o/--output as a project directory, not a YAML file."""

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["init", "-o", "demo"])

        assert result.exit_code == 0
        assert "Created a SAMMD configuration template at demo/sammd.yaml" in result.output
        assert load_config("demo/sammd.yaml").surface.metal == "Pd"


def test_root_help_exposes_logging_options() -> None:
    """Expose root logging controls without changing subcommands."""

    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "--verbose" in result.output
    assert "--no-color" in result.output


def test_no_color_init_succeeds_without_ansi_escapes() -> None:
    """Disable color from the root option while preserving init output."""

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["--no-color", "init"])

        assert result.exit_code == 0
        assert "INIT" in result.output
        assert (
            "Created a SAMMD configuration template at sammd-project/sammd.yaml"
            in result.output
        )
        assert ANSI_ESCAPE_RE.search(result.output) is None


def test_verbose_init_succeeds() -> None:
    """Accept verbose root logging option for init."""

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["--verbose", "init"])

        assert result.exit_code == 0


def test_init_cli_respects_no_overwrite_unless_force() -> None:
    """Refuse to overwrite existing templates unless --force is supplied."""

    runner = CliRunner()
    with runner.isolated_filesystem():
        runner.invoke(main, ["init", "-o", "demo"])
        Path("demo/notes.txt").write_text("keep me\n", encoding="utf-8")
        Path("demo/sammd.yaml").write_text("custom\n", encoding="utf-8")

        result = runner.invoke(main, ["init", "-o", "demo"])
        assert result.exit_code != 0
        assert "demo/sammd.yaml already exists" in result.output
        assert Path("demo/sammd.yaml").read_text(encoding="utf-8") == "custom\n"

        forced = runner.invoke(main, ["init", "-o", "demo", "--force"])
        assert forced.exit_code == 0
        assert Path("demo/notes.txt").read_text(encoding="utf-8") == "keep me\n"
        assert load_config("demo/sammd.yaml").surface.metal == "Pd"


def test_init_cli_writes_into_existing_directory_without_template() -> None:
    """Use an existing output directory when sammd.yaml is absent."""

    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("demo").mkdir()
        Path("demo/notes.txt").write_text("keep me\n", encoding="utf-8")

        result = runner.invoke(main, ["init", "-o", "demo"])

        assert result.exit_code == 0
        assert load_config("demo/sammd.yaml").surface.metal == "Pd"
        assert Path("demo/notes.txt").read_text(encoding="utf-8") == "keep me\n"


def test_init_cli_rejects_output_file() -> None:
    """Fail clearly when -o/--output names an existing file."""

    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("demo").write_text("not a directory\n", encoding="utf-8")

        result = runner.invoke(main, ["init", "-o", "demo"])

        assert result.exit_code != 0
        assert "demo exists and is not a directory" in result.output


def test_init_cli_rejects_yaml_output_path() -> None:
    """Avoid creating a directory named like a YAML file."""

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["init", "-o", "sammd.yaml"])

        assert result.exit_code != 0
        assert "--output expects a directory" in result.output
        assert "<directory>/sammd.yaml" in result.output


def test_validate_cli_accepts_template() -> None:
    """Validate a generated template from the CLI."""

    runner = CliRunner()
    with runner.isolated_filesystem():
        runner.invoke(main, ["init"])
        result = runner.invoke(main, ["validate", "sammd-project/sammd.yaml"])
        assert result.exit_code == 0
        assert "OK" in result.output
        assert "Configuration valid" in result.output


def test_cli_contract_exposes_only_config_builder_commands() -> None:
    """Keep simulation wrappers out of the first-release CLI surface."""

    assert set(main.commands) == {"build", "init", "validate"}
    for command_name in ("simulate", "equilibrate", "production", "run"):
        assert command_name not in main.commands


def test_build_cli_writes_full_system_artifacts(monkeypatch) -> None:
    """Build a full system from the CLI without requiring user Python code."""

    _patch_interchange_export(monkeypatch)
    runner = CliRunner()
    with runner.isolated_filesystem():
        runner.invoke(main, ["init"])

        result = runner.invoke(
            main, ["build", "sammd-project/sammd.yaml", "--output-dir", "outputs"]
        )

        assert result.exit_code == 0
        assert "sammd.cli" in result.output
        assert "INFO" in result.output
        assert "SAMMD Build" in result.output
        assert "Reading config and constructing deterministic build plan" in result.output
        assert "OK" in result.output
        assert "Dependency-free validation gates passed" in result.output
        assert "Surface: Pd(111)" in result.output
        assert "Preliminary planned solution counts" in result.output
        assert "Final runtime solution counts" in result.output
        assert "SYSTEM Solution counts" not in result.output
        assert "SAM grafting-density CIF" in result.output
        assert "Build summary" in result.output
        assert "Resolved config" in result.output
        assert "OpenFF Interchange Export" in result.output
        assert "Load interchange.json" in result.output
        assert load_config("sammd-project/sammd.yaml").surface.metal == "Pd"

        assert Path("outputs/sam_grafting_density.cif").is_file()
        assert Path("outputs/build_summary.json").is_file()
        assert Path("outputs/resolved_config.yaml").is_file()
        assert Path("outputs/solvated_system.cif").is_file()
        assert Path("outputs/solvated_system_pymol.pdb").is_file()
        assert Path("outputs/interchange.json").is_file()
        assert Path("outputs/anchor_metadata.json").is_file()


def test_build_cli_respects_no_overwrite_unless_requested(monkeypatch) -> None:
    """Protect existing build artifacts unless --overwrite is supplied."""

    _patch_interchange_export(monkeypatch)
    runner = CliRunner()
    with runner.isolated_filesystem():
        runner.invoke(main, ["init"])
        runner.invoke(main, ["build", "sammd-project/sammd.yaml", "--output-dir", "outputs"])

        blocked = runner.invoke(
            main, ["build", "sammd-project/sammd.yaml", "--output-dir", "outputs"]
        )
        assert blocked.exit_code != 0
        assert "refusing to overwrite existing file" in blocked.output

        forced = runner.invoke(
            main,
            ["build", "sammd-project/sammd.yaml", "--output-dir", "outputs", "--overwrite"],
        )
        assert forced.exit_code == 0


def test_build_help_does_not_expose_removed_full_flags() -> None:
    """Build always exports the full system without an opt-in flag."""

    result = CliRunner().invoke(main, ["build", "--help"])

    assert result.exit_code == 0
    assert "--full" not in result.output
    assert "--export-backend" not in result.output
