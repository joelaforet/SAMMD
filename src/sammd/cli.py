"""Command-line interface for SAMMD scaffolding."""

from pathlib import Path

import click

from sammd.builders import build_system
from sammd.config import CONFIG_TEMPLATE, load_config


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
    click.echo(f"Wrote SAMMD configuration template to {output}")


@main.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def validate(config: Path) -> None:
    """Validate a SAMMD YAML configuration."""

    loaded = load_config(config)
    component_count = len(loaded.sam.components)
    click.echo(
        f"Configuration valid: {loaded.surface.metal}({loaded.surface.facet}) "
        f"with {component_count} SAM component(s)"
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
    try:
        topology = plan.write_topology_cif(overwrite=overwrite)
        build_summary = plan.write_build_summary(overwrite=overwrite)
        resolved_config = plan.write_resolved_config(overwrite=overwrite)
    except FileExistsError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("SAMMD build plan ready")
    click.echo(f"Surface: {plan.slab.metal}({plan.slab.facet})")
    click.echo(f"Binding sites: {len(plan.binding_sites)}")
    click.echo(f"SAM molecules: {len(plan.sam_placements.placements)}")
    click.echo(f"Solution counts: {dict(plan.solution.molecule_counts)}")
    click.echo(f"Wrote topology CIF: {topology}")
    click.echo(f"Wrote build summary: {build_summary}")
    click.echo(f"Wrote resolved config: {resolved_config}")
    click.echo("Inspect topology.cif before moving on to OpenMM simulation setup.")
