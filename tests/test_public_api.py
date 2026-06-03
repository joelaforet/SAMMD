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
    assert "add_position_restraints" not in sammd.__all__
    assert "add_sulfur_metal_lj_scaling" not in sammd.__all__
    assert "create_langevin_integrator" not in sammd.__all__
    assert "create_openmm_simulation" not in sammd.__all__
    assert "require_openmm" not in sammd.__all__
    assert "SolutionPlan" not in sammd.__all__
    assert "plan_solution_composition" not in sammd.__all__
    assert "plan_pd111_slab" not in sammd.__all__
    assert "plan_sam_placements" not in sammd.__all__


def test_build_system_returns_lightweight_plan() -> None:
    """Keep docs workflow importable while backend construction is deferred."""

    from sammd import build_system
    from sammd.config import SAMMDConfig

    config = SAMMDConfig()
    plan = build_system(config)

    assert plan.config is config
    assert not plan.full_construction_available
    assert plan.openmm_construction_implemented is False
