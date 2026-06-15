"""Command-line interface for SAMMD scaffolding."""

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

import click

from sammd.backends.interchange import backend_build_summary, export_interchange_backend
from sammd.colors import setup_colored_logging
from sammd.core.builders import build_system
from sammd.core.config import CONFIG_TEMPLATE, load_config
from sammd.core.validation import validate_build_plan, validate_output_paths

LOGGER = logging.getLogger(__name__)


@contextmanager
def _timed(message: str) -> Iterator[None]:
    """Log start/end messages with elapsed wall time."""

    start = perf_counter()
    LOGGER.info("%s", message)
    try:
        yield
    finally:
        LOGGER.info("DONE %s %.2fs", message, perf_counter() - start)


def _failed_error_messages(*reports: Any) -> list[str]:
    messages: list[str] = []
    for report in reports:
        for gate in report.gates:
            if not gate.passed and gate.severity == "error":
                messages.append(f"{gate.gate_id}: {gate.message}")
    return messages


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
@click.option("--no-color", is_flag=True, help="Disable colored logging output.")
def main(verbose: bool, no_color: bool) -> None:
    """Manage SAMMD configuration files."""

    setup_colored_logging(verbose=verbose, no_color=no_color)


@main.command()
@click.option("-o", "--output", type=click.Path(path_type=Path), default=Path("sammd-project"))
@click.option("--force", is_flag=True, help="Overwrite an existing sammd.yaml file.")
def init(output: Path, force: bool) -> None:
    """Write a commented SAMMD YAML template."""

    if output.suffix.lower() in {".yaml", ".yml"}:
        raise click.ClickException(
            "sammd init --output expects a directory and writes "
            "<directory>/sammd.yaml; pass a directory path instead"
        )
    if output.exists() and not output.is_dir():
        raise click.ClickException(f"{output} exists and is not a directory")

    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "sammd.yaml"
    if config_path.exists() and not force:
        raise click.ClickException(f"{config_path} already exists; use --force to overwrite")

    config_path.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    LOGGER.info("INIT Created a SAMMD configuration template at %s", config_path)


@main.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def validate(config: Path) -> None:
    """Validate a SAMMD YAML configuration."""

    loaded = load_config(config)
    component_count = len(loaded.sam.components)
    LOGGER.info(
        "OK Configuration valid: %s(%s) with %s SAM component(s)",
        loaded.surface.metal,
        loaded.surface.facet,
        component_count,
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
    """Build the current plan and write Interchange export files."""

    LOGGER.info("SAMMD Build")
    with _timed("Reading config and constructing deterministic build plan"):
        plan = build_system(config, output_dir=output_dir)
    with _timed("Running dependency-free validation gates"):
        build_report = validate_build_plan(plan)
        output_report = validate_output_paths(plan.output_paths)
    failed_messages = _failed_error_messages(build_report, output_report)
    if failed_messages:
        raise click.ClickException("Validation failed: " + "; ".join(failed_messages))

    LOGGER.info("OK Dependency-free validation gates passed")
    LOGGER.info("SYSTEM Surface: %s(%s)", plan.slab.metal, plan.slab.facet)
    LOGGER.info("SYSTEM Binding sites: %s", len(plan.binding_sites))
    LOGGER.info("SYSTEM SAM molecules: %s", len(plan.sam_placements.placements))
    LOGGER.info(
        "SYSTEM Preliminary planned solution counts: %s",
        dict(plan.solution.molecule_counts),
    )

    try:
        with _timed("Writing SAM grafting-density visual check"):
            topology = plan.write_topology_cif(overwrite=overwrite)
        with _timed("Writing resolved configuration"):
            resolved_config = plan.write_resolved_config(overwrite=overwrite)
        LOGGER.info("OpenFF Interchange Export")
        export_result = export_interchange_backend(
            plan,
            overwrite=overwrite,
        )
        runtime_geometry = export_result.runtime_solvent_geometry
        if runtime_geometry is not None:
            LOGGER.info(
                "SYSTEM Final runtime solution counts: %s",
                runtime_geometry.molecule_counts,
            )
        export_files = export_result.files
        with _timed("Writing Interchange export build summary"):
            build_summary = plan.write_build_summary(overwrite=overwrite)
            build_summary.write_text(
                json.dumps(
                    backend_build_summary(plan, export_result),
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
    except FileExistsError as exc:
        raise click.ClickException(str(exc)) from exc
    except (ImportError, NotImplementedError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    LOGGER.info("Outputs")
    LOGGER.info("FILE SAM grafting-density CIF: %s", topology)
    LOGGER.info("FILE Build summary: %s", build_summary)
    LOGGER.info("FILE Resolved config: %s", resolved_config)
    LOGGER.info("FILE Wrote solvated system CIF: %s", export_files["solvated_system"])
    LOGGER.info("FILE PyMOL PDB with CONECT records: %s", export_files["pymol_system"])
    LOGGER.info("FILE Interchange JSON: %s", export_files["openff_interchange"])
    LOGGER.info("FILE Anchor metadata: %s", export_files["anchor_metadata"])
    LOGGER.info(
        "NEXT Load interchange.json and call Interchange.to_openmm() downstream when needed."
    )
