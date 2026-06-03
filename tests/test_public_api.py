"""Smoke tests for the public SAMMD API."""


def test_public_imports() -> None:
    """Import the stable public API names."""

    import sammd
    from sammd import SAMMDConfig, load_config, load_config_dict

    assert sammd.__version__
    assert SAMMDConfig is not None
    assert load_config is not None
    assert load_config_dict is not None
