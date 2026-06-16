"""Dependency-free metal-sulfur interaction metadata for SAM planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass

METAL_SULFUR_INTERACTION_MODE = "nonbonded_lj_override"
METAL_SULFUR_SITE_KIND = "fcc_hollow"
METAL_SULFUR_PAIRS_PER_ANCHOR = 3
METAL_SULFUR_SIGMA_NM = 0.22
METAL_SULFUR_EPSILON_KCAL_MOL = 2.0
KCAL_TO_KJ = 4.184
METAL_SULFUR_EPSILON_KJ_MOL = METAL_SULFUR_EPSILON_KCAL_MOL * KCAL_TO_KJ


@dataclass(frozen=True)
class MetalSulfurLJOverrideSpec:
    """Serializable first-release metadata for selected metal-S LJ overrides."""

    mode: str = METAL_SULFUR_INTERACTION_MODE
    site_kind: str = METAL_SULFUR_SITE_KIND
    pairs_per_anchor: int = METAL_SULFUR_PAIRS_PER_ANCHOR
    sigma_nm: float = METAL_SULFUR_SIGMA_NM
    epsilon_kcal_mol: float = METAL_SULFUR_EPSILON_KCAL_MOL
    epsilon_kj_mol: float = METAL_SULFUR_EPSILON_KJ_MOL
    interpretation: str = (
        "strengthened nonbonded LJ attraction for neutral thiol sulfur-metal pairs; "
        "not covalent, quantum, or reactive chemisorption"
    )

    def to_summary(self) -> dict[str, object]:
        """Return a JSON-serializable summary for build metadata."""

        return asdict(self)


def default_metal_sulfur_interaction() -> MetalSulfurLJOverrideSpec:
    """Return the canonical v0.1.0 metal-S interaction strategy metadata."""

    return MetalSulfurLJOverrideSpec()
