"""Lightweight CHARMM-INTERFACE metal parameter registry."""

from __future__ import annotations

from dataclasses import dataclass


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
