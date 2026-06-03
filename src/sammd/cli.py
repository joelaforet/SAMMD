"""Command-line interface for SAMMD scaffolding."""

from pathlib import Path

import click

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
