"""Smoke tests for the public SAMMD API."""


def test_public_imports() -> None:
    """Import the stable public API names."""

    import sammd
    from sammd import SAMMDConfig, build_system, load_config, load_config_dict

    assert sammd.__version__
    assert SAMMDConfig is not None
    assert build_system is not None
    assert load_config is not None
    assert load_config_dict is not None
    assert "plan_pd111_slab" not in sammd.__all__
    assert "plan_sam_placements" not in sammd.__all__


def test_build_system_stub_fails_clearly() -> None:
    """Keep docs workflow importable while backend construction is deferred."""

    from sammd import build_system
    from sammd.config import SAMMDConfig

    config = SAMMDConfig()
    try:
        build_system(config)
    except NotImplementedError as error:
        assert "Full OpenFF/OpenMM system construction is not implemented" in str(error)
        assert "scaffolding milestone" in str(error)
    else:  # pragma: no cover - defensive failure path
        raise AssertionError("build_system should raise NotImplementedError")
