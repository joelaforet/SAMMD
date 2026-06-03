"""Tests for lightweight surface metadata."""

import pytest

from sammd.surfaces import FCC_SURFACE_REGISTRY, get_fcc_surface_metadata


def test_pd111_surface_metadata() -> None:
    """Expose Pd(111) lattice and interlayer spacing metadata."""

    metadata = get_fcc_surface_metadata("Pd", "111")

    assert FCC_SURFACE_REGISTRY[("Pd", "111")] == metadata
    assert metadata.lattice_constant_nm == 0.389
    assert metadata.interlayer_spacing_nm == pytest.approx(0.2245892547, rel=1e-6)
    assert metadata.slab_thickness_nm(8) == pytest.approx(7 * 0.389 / 3**0.5)


def test_unsupported_surface_fails_clearly() -> None:
    """Reject surfaces outside the lightweight metadata registry."""

    with pytest.raises(ValueError, match="unsupported Fcc surface"):
        get_fcc_surface_metadata("Au", "111")
