"""Lightweight surface metadata for SAMMD validation."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class FccSurfaceMetadata:
    """Crystallographic metadata for an Fcc metal surface."""

    metal: str
    facet: str
    lattice_constant_nm: float

    @property
    def interlayer_spacing_nm(self) -> float:
        """Return the interlayer spacing for the configured Fcc facet."""

        if self.facet != "111":
            msg = f"unsupported Fcc facet '{self.facet}'"
            raise ValueError(msg)
        return self.lattice_constant_nm / sqrt(3)

    def slab_thickness_nm(self, layers: int) -> float:
        """Return first-to-last layer separation for an Fcc slab.

        Parameters
        ----------
        layers
            Number of atomic layers in the slab.

        Returns
        -------
        float
            Physical separation from the first to last atomic layer.
        """

        if layers <= 0:
            msg = "layers must be positive"
            raise ValueError(msg)
        return (layers - 1) * self.interlayer_spacing_nm


FCC_SURFACE_REGISTRY: dict[tuple[str, str], FccSurfaceMetadata] = {
    ("Pd", "111"): FccSurfaceMetadata(
        metal="Pd",
        facet="111",
        lattice_constant_nm=0.389,
    )
}


def get_fcc_surface_metadata(metal: str, facet: str) -> FccSurfaceMetadata:
    """Return registered Fcc surface metadata.

    Parameters
    ----------
    metal
        Metal element symbol.
    facet
        Miller-index facet label.

    Returns
    -------
    FccSurfaceMetadata
        Registered surface metadata.
    """

    try:
        return FCC_SURFACE_REGISTRY[(metal, facet)]
    except KeyError as error:
        supported = ", ".join(f"{item[0]}({item[1]})" for item in sorted(FCC_SURFACE_REGISTRY))
        msg = f"unsupported Fcc surface '{metal}({facet})'; supported surfaces: {supported}"
        raise ValueError(msg) from error
