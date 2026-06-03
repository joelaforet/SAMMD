"""Lightweight CHARMM-INTERFACE metal parameter registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

INTERFACE_SOURCE_URL = "https://github.com/CHARMM-INTERFACE/"


_FCC_METAL_ATOMIC_NUMBERS: dict[str, int] = {
    "Ag": 47,
    "Al": 13,
    "Au": 79,
    "Cu": 29,
    "Ni": 28,
    "Pb": 82,
    "Pd": 46,
    "Pt": 78,
}


@dataclass(frozen=True)
class FccMetalLJParameters:
    """CHARMM-INTERFACE Lennard-Jones parameters for an Fcc metal."""

    symbol: str
    source_epsilon_kcal_mol: float
    openff_epsilon_kcal_mol: float
    rmin_half_angstrom: float

    @property
    def sigma_angstrom(self) -> float:
        """Return Lennard-Jones sigma in angstrom."""

        return sigma_from_rmin_half(self.rmin_half_angstrom)


def sigma_from_rmin_half(rmin_half: float) -> float:
    """Convert CHARMM Rmin/2 to Lennard-Jones sigma.

    Parameters
    ----------
    rmin_half
        CHARMM Rmin/2 value.

    Returns
    -------
    float
        Lennard-Jones sigma in the same length unit as ``rmin_half``.
    """

    if rmin_half <= 0:
        msg = "rmin_half must be positive"
        raise ValueError(msg)
    return 2 * rmin_half / 2 ** (1 / 6)


FCC_METAL_LJ_REGISTRY: dict[str, FccMetalLJParameters] = {
    symbol: FccMetalLJParameters(
        symbol=symbol,
        source_epsilon_kcal_mol=-epsilon,
        openff_epsilon_kcal_mol=epsilon,
        rmin_half_angstrom=rmin_half,
    )
    for symbol, epsilon, rmin_half in [
        ("Ag", 4.56, 1.4775),
        ("Al", 4.02, 1.4625),
        ("Au", 5.29, 1.4755),
        ("Cu", 4.72, 1.3080),
        ("Ni", 5.65, 1.2760),
        ("Pb", 2.93, 1.7825),
        ("Pd", 6.15, 1.4095),
        ("Pt", 7.80, 1.4225),
    ]
}


def get_fcc_metal_parameters(symbol: str) -> FccMetalLJParameters:
    """Return registered Fcc metal Lennard-Jones parameters.

    Parameters
    ----------
    symbol
        Element symbol such as ``Pd``.

    Returns
    -------
    FccMetalLJParameters
        Registered metal parameter data.
    """

    try:
        return FCC_METAL_LJ_REGISTRY[symbol]
    except KeyError as error:
        supported = ", ".join(sorted(FCC_METAL_LJ_REGISTRY))
        msg = f"unsupported Fcc metal '{symbol}'; supported metals: {supported}"
        raise ValueError(msg) from error


def generate_interface_metal_offxml() -> str:
    """Generate SMIRNOFF OFFXML for registered INTERFACE Fcc metals.

    Returns
    -------
    str
        Deterministic XML text containing one ``vdW`` ``Atom`` entry per registered Fcc metal.
    """

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<SMIRNOFF version="0.3" aromaticity_model="OEAroModel_MDL">',
        "  <!-- Generated from CHARMM-INTERFACE Fcc metal Lennard-Jones parameters. -->",
        "  <!-- Provenance: Heinz et al.; CHARMM-INTERFACE; "
        f"{INTERFACE_SOURCE_URL} -->",
        "  <Author>SAMMD contributors</Author>",
        "  <Date>2026-06-03</Date>",
        '  <vdW version="0.4" potential="Lennard-Jones-12-6" '
        'combining_rules="Lorentz-Berthelot" scale12="0.0" scale13="0.0" '
        'scale14="0.5" scale15="1.0" cutoff="9.0 * angstrom" '
        'switch_width="1.0 * angstrom" method="cutoff">',
    ]
    for symbol in sorted(FCC_METAL_LJ_REGISTRY):
        parameters = FCC_METAL_LJ_REGISTRY[symbol]
        atomic_number = _FCC_METAL_ATOMIC_NUMBERS[symbol]
        lines.append(
            f'    <Atom smirks="[#{atomic_number}:1]" id="{symbol}" '
            f'epsilon="{parameters.openff_epsilon_kcal_mol:.2f} * kilocalorie_per_mole" '
            f'rmin_half="{parameters.rmin_half_angstrom:.4f} * angstrom"/>'
        )
    lines.extend(["  </vdW>", "</SMIRNOFF>", ""])
    return "\n".join(lines)


def write_interface_metal_offxml(path: str | Path) -> None:
    """Write generated INTERFACE Fcc metal OFFXML to a path.

    Parameters
    ----------
    path
        Destination path for the generated OFFXML text.
    """

    Path(path).write_text(generate_interface_metal_offxml(), encoding="utf-8")
