"""Command-line interface for SAMMD scaffolding."""

from pathlib import Path
from typing import Any

import click

from sammd.builders import build_system
from sammd.config import CONFIG_TEMPLATE, load_config
from sammd.validation import validate_build_plan, validate_output_paths


def _step(label: str, message: str, fg: str) -> None:
    """Print a colored CLI step label followed by plain message text."""

    click.secho(f"{label} ", fg=fg, bold=True, nl=False)
    click.echo(message)


def _failed_error_messages(*reports: Any) -> list[str]:
    messages: list[str] = []
    for report in reports:
        for gate in report.gates:
            if not gate.passed and gate.severity == "error":
                messages.append(f"{gate.gate_id}: {gate.message}")
    return messages


@click.group()
def main() -> None:
    """Manage lightweight SAMMD configuration files."""


@main.command()
@click.option("-o", "--output", type=click.Path(path_type=Path), default=Path("sammd.yaml"))
@click.option("--force", is_flag=True, help="Overwrite an existing output file.")
def init(output: Path, force: bool) -> None:
    """Write a commented SAMMD YAML template."""

    if output.exists() and not force:
        raise click.ClickException(f"{output} already exists; use --force to overwrite")
    output.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    _step("INIT", f"Created a SAMMD configuration template at {output}", "cyan")


@main.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def validate(config: Path) -> None:
    """Validate a SAMMD YAML configuration."""

    loaded = load_config(config)
    component_count = len(loaded.sam.components)
    _step(
        "OK",
        f"Configuration valid: {loaded.surface.metal}({loaded.surface.facet}) "
        f"with {component_count} SAM component(s)",
        "green",
    )


@main.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override the outputs.directory value from the YAML config.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Replace existing build artifacts.",
)
def build(config: Path, output_dir: Path | None, overwrite: bool) -> None:
    """Build the current plan and write topology.cif plus reports."""

    plan = build_system(config, output_dir=output_dir)
    build_report = validate_build_plan(plan)
    output_report = validate_output_paths(plan.output_paths)
    failed_messages = _failed_error_messages(build_report, output_report)
    if failed_messages:
        raise click.ClickException("Validation failed: " + "; ".join(failed_messages))

    _step("PLAN", "SAMMD build plan ready", "yellow")
    _step("OK", "Lightweight validation gates passed", "green")
    _step("PLAN", f"Surface: {plan.slab.metal}({plan.slab.facet})", "yellow")
    _step("PLAN", f"Binding sites: {len(plan.binding_sites)}", "yellow")
    _step("PLAN", f"SAM molecules: {len(plan.sam_placements.placements)}", "yellow")
    _step("PLAN", f"Solution counts: {dict(plan.solution.molecule_counts)}", "yellow")

    try:
        topology = plan.write_topology_cif(overwrite=overwrite)
        build_summary = plan.write_build_summary(overwrite=overwrite)
        resolved_config = plan.write_resolved_config(overwrite=overwrite)
    except FileExistsError as exc:
        raise click.ClickException(str(exc)) from exc

    _step("WRITE", f"Wrote topology CIF: {topology}", "blue")
    _step("WRITE", f"Wrote build summary: {build_summary}", "blue")
    _step("WRITE", f"Wrote resolved config: {resolved_config}", "blue")
    _step("NEXT", "Inspect topology.cif before moving on to OpenMM simulation setup.", "magenta")
