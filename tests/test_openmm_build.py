"""Tests for the staged OpenMM smoke builder facade."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sammd.config import EXPECTED_GRAFTING_DENSITY_UNIT
from sammd.openmm_build import OpenMMSmokeBuilder, OpenMMSmokeBuildOptions


@dataclass(frozen=True)
class DummyAtom:
    """Minimal atom template used by builder ordering tests."""

    element: str


@dataclass(frozen=True)
class DummyTemplate:
    """Minimal molecule template used by builder ordering tests."""

    atoms: tuple[DummyAtom, ...]


def test_openmm_build_import_does_not_import_heavy_science_modules() -> None:
    """Importing the staged builder must not activate OpenMM or OpenFF."""

    sys.modules.pop("openmm", None)
    sys.modules.pop("openff", None)

    __import__("sammd.openmm_build")

    assert "openmm" not in sys.modules
    assert "openff" not in sys.modules


def test_builder_rejects_sam_before_surface() -> None:
    """SAM declaration should require the surface stage first."""

    builder = make_builder()

    with pytest.raises(RuntimeError, match="before surface stage"):
        builder.add_sam_layer(sulfur_template())


def test_builder_rejects_reactants_before_sam() -> None:
    """Reactant declaration should require the SAM layer stage first."""

    builder = make_builder().add_surface()

    with pytest.raises(RuntimeError, match="before sam layer stage"):
        builder.add_reactants(carbon_template(), count=1)


def test_builder_rejects_solvent_before_reactants() -> None:
    """Solvent declaration should require reactants after the SAM layer."""

    builder = make_builder().add_surface().add_sam_layer(sulfur_template())

    with pytest.raises(RuntimeError, match="before reactants stage"):
        builder.add_solvent(carbon_template(), count=1)


def test_builder_rejects_finalize_before_required_stages() -> None:
    """Finalization should fail with a clear missing-stage error."""

    builder = make_builder().add_surface()

    with pytest.raises(RuntimeError, match="cannot finalize before sam layer stage"):
        builder.finalize(default_options())


def test_builder_rejects_duplicate_stage_calls() -> None:
    """Duplicate stage calls should be rejected rather than silently replacing inputs."""

    builder = make_builder().add_surface()
    with pytest.raises(RuntimeError, match="surface stage has already been added"):
        builder.add_surface()


def test_builder_finalize_delegates_existing_constructor() -> None:
    """Finalization should pass staged inputs to the injected backend callable."""

    calls: list[dict[str, Any]] = []

    def construction_fn(*args: Any, **kwargs: Any) -> str:
        """Record construction inputs and return a sentinel build object."""

        calls.append({"args": args, "kwargs": kwargs})
        return "built"

    plan = make_plan()
    sam_template = sulfur_template()
    reactant_template = carbon_template()
    solvent_template = carbon_template()
    result = (
        OpenMMSmokeBuilder.from_plan(
            modules="modules",
            plan=plan,
            construction_fn=construction_fn,
        )
        .add_surface()
        .add_sam_layer(sam_template)
        .add_reactants(reactant_template, count=2)
        .add_solvent(solvent_template, count=3)
        .finalize(default_options())
    )

    assert result == "built"
    assert calls[0]["args"] == ("modules", plan, sam_template, reactant_template, solvent_template)
    assert calls[0]["kwargs"]["reactant_count"] == 2
    assert calls[0]["kwargs"]["solvent_count"] == 3


def test_builder_rejects_sam_template_without_one_sulfur() -> None:
    """Existing single-sulfur SAM constraint should be preserved at the SAM stage."""

    builder = make_builder().add_surface()
    with pytest.raises(ValueError, match="exactly one sulfur"):
        builder.add_sam_layer(carbon_template())


def make_builder() -> OpenMMSmokeBuilder:
    """Return a staged builder with no-op construction for ordering tests."""

    return OpenMMSmokeBuilder.from_plan(
        modules=SimpleNamespace(),
        plan=make_plan(),
        construction_fn=lambda *args, **kwargs: (args, kwargs),
    )


def make_plan() -> Any:
    """Return the minimal plan shape required by the builder facade."""

    return SimpleNamespace(
        config=SimpleNamespace(
            surface=SimpleNamespace(facet="111", metal="Pd"),
            sam=SimpleNamespace(
                grafting_density=SimpleNamespace(unit=EXPECTED_GRAFTING_DENSITY_UNIT),
                components=[SimpleNamespace(name="propanethiol")],
            ),
        )
    )


def sulfur_template() -> DummyTemplate:
    """Return a template with exactly one sulfur atom."""

    return DummyTemplate(atoms=(DummyAtom("S"), DummyAtom("C")))


def carbon_template() -> DummyTemplate:
    """Return a template without sulfur atoms."""

    return DummyTemplate(atoms=(DummyAtom("C"),))


def default_options() -> OpenMMSmokeBuildOptions:
    """Return valid smoke build options for facade tests."""

    return OpenMMSmokeBuildOptions(
        sulfur_height_nm=0.18,
        solvent_padding_nm=3.0,
        packmol_working_dir=Path("packmol"),
        pressure_bar=1.0,
        temperature_k=300.0,
        pd_s_sigma_nm=0.22,
        pd_s_epsilon_kcal_mol=2.0,
    )
