"""Output path planning and lightweight mmCIF writing helpers."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sammd.surfaces import SurfaceSlab

Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class OutputPaths:
    """Resolved output artifact paths for visualization and runtime reporting."""

    topology: Path
    trajectory: Path
    thermodynamics: Path
    checkpoint: Path | None = None
    state: Path | None = None


@dataclass(frozen=True)
class AtomRecord:
    """Lightweight atom record for mmCIF visualization scaffolds."""

    serial: int
    atom_name: str
    element: str
    residue_name: str
    residue_id: int
    chain_id: str
    component_label: str
    coordinates_nm: Vector3


def plan_output_paths(config: Any, base_dir: str | Path = ".") -> OutputPaths:
    """Resolve deterministic output paths from a SAMMD config or output section.

    Parameters
    ----------
    config
        Top-level config with an ``output`` attribute, or an output config object.
    base_dir
        Base directory used for relative artifact paths.

    Returns
    -------
    OutputPaths
        Resolved paths with validated visualization and reporter suffixes.
    """

    output_config = getattr(config, "output", config)
    root = Path(base_dir)
    topology = _resolve_output_path(root, output_config.topology)
    trajectory = _resolve_output_path(root, output_config.trajectory)
    thermodynamics = _resolve_output_path(root, output_config.thermodynamics)
    checkpoint = _resolve_optional_output_path(root, getattr(output_config, "checkpoint", None))
    state = _resolve_optional_output_path(root, getattr(output_config, "state", None))

    _validate_suffix(topology, ".cif", "topology")
    _validate_suffix(trajectory, ".dcd", "trajectory")
    _validate_suffix(thermodynamics, ".csv", "thermodynamics")
    return OutputPaths(
        topology=topology,
        trajectory=trajectory,
        thermodynamics=thermodynamics,
        checkpoint=checkpoint,
        state=state,
    )


def safe_write_text(path: str | Path, text: str, *, overwrite: bool = False) -> Path:
    """Write text atomically without silent overwrite by default.

    Parameters
    ----------
    path
        Destination path.
    text
        Text content to write.
    overwrite
        Whether an existing destination may be replaced.

    Returns
    -------
    Path
        Destination path.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        msg = f"refusing to overwrite existing file: {destination}"
        raise FileExistsError(msg)

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=destination.parent, delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(text)
        if overwrite:
            os.replace(temporary_name, destination)
        else:
            os.link(temporary_name, destination)
            os.unlink(temporary_name)
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()
    return destination


def format_mmcif(
    atom_records: tuple[AtomRecord, ...] | list[AtomRecord],
    *,
    data_name: str = "sammd_scaffold",
    cell_lengths_nm: Vector3 | None = None,
) -> str:
    """Format lightweight atom records as mmCIF/PDBx text.

    Parameters
    ----------
    atom_records
        Atom records with coordinates in nanometers.
    data_name
        mmCIF data block name.
    cell_lengths_nm
        Optional periodic cell lengths in nanometers.

    Returns
    -------
    str
        mmCIF text with ``_atom_site`` rows and Angstrom coordinates.
    """

    if not atom_records:
        msg = "at least one atom record is required"
        raise ValueError(msg)
    lines = [f"data_{data_name}", "#"]
    if cell_lengths_nm is not None:
        lines.extend(
            [
                f"_cell.length_a {_format_float(cell_lengths_nm[0] * 10.0)}",
                f"_cell.length_b {_format_float(cell_lengths_nm[1] * 10.0)}",
                f"_cell.length_c {_format_float(cell_lengths_nm[2] * 10.0)}",
                "_cell.angle_alpha 90.000",
                "_cell.angle_beta 90.000",
                "_cell.angle_gamma 90.000",
                "#",
            ]
        )
    lines.extend(
        [
            "loop_",
            "_atom_site.group_PDB",
            "_atom_site.id",
            "_atom_site.type_symbol",
            "_atom_site.label_atom_id",
            "_atom_site.label_comp_id",
            "_atom_site.label_asym_id",
            "_atom_site.label_entity_id",
            "_atom_site.label_seq_id",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
            "_atom_site.pdbx_PDB_model_num",
        ]
    )
    for record in atom_records:
        x_angstrom, y_angstrom, z_angstrom = (
            coordinate * 10.0 for coordinate in record.coordinates_nm
        )
        lines.append(
            " ".join(
                [
                    "HETATM",
                    str(record.serial),
                    _quote_cif_value(record.element),
                    _quote_cif_value(record.atom_name),
                    _quote_cif_value(record.residue_name),
                    _quote_cif_value(record.chain_id),
                    _quote_cif_value(record.component_label),
                    str(record.residue_id),
                    _format_float(x_angstrom),
                    _format_float(y_angstrom),
                    _format_float(z_angstrom),
                    "1",
                ]
            )
        )
    lines.append("#")
    return "\n".join(lines) + "\n"


def write_mmcif(
    path: str | Path,
    atom_records: tuple[AtomRecord, ...] | list[AtomRecord],
    *,
    data_name: str = "sammd_scaffold",
    cell_lengths_nm: Vector3 | None = None,
    overwrite: bool = False,
) -> Path:
    """Write lightweight mmCIF/PDBx atom records to disk.

    Parameters
    ----------
    path
        Destination ``.cif`` path.
    atom_records
        Atom records with nanometer coordinates.
    data_name
        mmCIF data block name.
    cell_lengths_nm
        Optional periodic cell lengths in nanometers.
    overwrite
        Whether an existing destination may be replaced.

    Returns
    -------
    Path
        Written path.
    """

    destination = Path(path)
    _validate_suffix(destination, ".cif", "topology")
    text = format_mmcif(atom_records, data_name=data_name, cell_lengths_nm=cell_lengths_nm)
    return safe_write_text(destination, text, overwrite=overwrite)


def slab_to_atom_records(slab: SurfaceSlab, *, chain_id: str = "M") -> tuple[AtomRecord, ...]:
    """Convert a planned metal slab into PyMOL-friendly atom records.

    Parameters
    ----------
    slab
        Planned surface slab.
    chain_id
        Chain identifier used for visualization.

    Returns
    -------
    tuple[AtomRecord, ...]
        Deterministic metal atom records labeled by slab layer role.
    """

    records: list[AtomRecord] = []
    for index, (position_nm, layer_index) in enumerate(
        zip(slab.positions_nm, slab.layer_indices, strict=True)
    ):
        layer_role, residue_name = _slab_layer_label(layer_index, slab.layers)
        atom_label = slab.labels[index] if index < len(slab.labels) else f"{slab.metal}{index + 1}"
        records.append(
            AtomRecord(
                serial=index + 1,
                atom_name=atom_label,
                element=slab.metal,
                residue_name=residue_name,
                residue_id=layer_index + 1,
                chain_id=chain_id,
                component_label=f"metal_{layer_role}",
                coordinates_nm=position_nm,
            )
        )
    return tuple(records)


def _resolve_output_path(base_dir: Path, value: str | Path) -> Path:
    """Resolve one configured output path."""

    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _resolve_optional_output_path(base_dir: Path, value: str | Path | None) -> Path | None:
    """Resolve one optional configured output path."""

    if value is None:
        return None
    return _resolve_output_path(base_dir, value)


def _validate_suffix(path: Path, expected_suffix: str, artifact_name: str) -> None:
    """Validate a practical artifact suffix."""

    if path.suffix.lower() != expected_suffix:
        msg = f"{artifact_name} output must use a '{expected_suffix}' suffix"
        raise ValueError(msg)


def _format_float(value: float) -> str:
    """Format a coordinate deterministically."""

    return f"{value:.6f}"


def _quote_cif_value(value: str) -> str:
    """Quote mmCIF values only when required."""

    if value and all(character not in value for character in " \t\n'\""):
        return value
    return "'" + value.replace("'", "''") + "'"


def _slab_layer_label(layer_index: int, layers: int) -> tuple[str, str]:
    """Return component and residue labels for a slab layer."""

    if layer_index == 0:
        return "bottom_layer", "PDB"
    if layer_index == layers - 1:
        return "top_layer", "PDT"
    return "slab", "PDM"
