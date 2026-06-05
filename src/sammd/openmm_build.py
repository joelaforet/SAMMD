"""Staged OpenMM smoke-system builder facade.

This module keeps the teachable build sequence importable without importing OpenMM or
OpenFF. Backend-specific object construction is injected as a callable so the smoke
tool can preserve current thermodynamic behavior while exposing an explicit staged
workflow.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sammd.config import EXPECTED_GRAFTING_DENSITY_UNIT

OpenMMConstructionFn = Callable[..., Any]


@dataclass(frozen=True)
class OpenMMSmokeBuildOptions:
    """Runtime options for direct OpenMM smoke construction.

    Parameters
    ----------
    sulfur_height_nm
        Height added along each placement normal for sulfur anchor coordinates.
    solvent_padding_nm
        Solvent padding used to derive the final smoke box height.
    packmol_working_dir
        Directory used for Packmol intermediate files.
    pressure_bar
        Planned pressure recorded in smoke provenance.
    temperature_k
        Planned temperature recorded in smoke provenance.
    pd_s_sigma_nm
        Lennard-Jones sigma for selected Pd-S anchor exceptions.
    pd_s_epsilon_kcal_mol
        Lennard-Jones epsilon for selected Pd-S anchor exceptions.
    """

    sulfur_height_nm: float
    solvent_padding_nm: float
    packmol_working_dir: Path
    pressure_bar: float
    temperature_k: float
    pd_s_sigma_nm: float
    pd_s_epsilon_kcal_mol: float


class OpenMMSmokeBuilder:
    """Order-enforcing builder for the direct OpenMM smoke workflow.

    The builder is intentionally a staged facade in this refactor loop. It records the
    conceptual smoke-system construction stages and delegates the existing low-level
    backend construction to an injected callable during :meth:`finalize`.
    """

    _SURFACE_STAGE = "surface"
    _SAM_STAGE = "sam layer"
    _REACTANTS_STAGE = "reactants"
    _SOLVENT_STAGE = "solvent"

    def __init__(self, *, modules: Any, plan: Any, construction_fn: OpenMMConstructionFn) -> None:
        """Create a staged builder from validated dependencies.

        Parameters
        ----------
        modules
            Lazily imported OpenMM module bundle used by the backend callable.
        plan
            Deterministic SAMMD build plan.
        construction_fn
            Callable that performs concrete OpenMM object construction.
        """

        self._modules = modules
        self._plan = plan
        self._construction_fn = construction_fn
        self._surface_added = False
        self._sam_template: Any | None = None
        self._reactant_template: Any | None = None
        self._reactant_count: int | None = None
        self._solvent_template: Any | None = None
        self._solvent_count: int | None = None
        self._finalized = False

    @classmethod
    def from_plan(
        cls,
        *,
        modules: Any,
        plan: Any,
        construction_fn: OpenMMConstructionFn,
    ) -> OpenMMSmokeBuilder:
        """Create a builder for a v0.1.0-compatible SAMMD build plan.

        Parameters
        ----------
        modules
            Lazily imported OpenMM module bundle.
        plan
            Deterministic SAMMD build plan.
        construction_fn
            Callable used by :meth:`finalize` for direct OpenMM construction.

        Returns
        -------
        OpenMMSmokeBuilder
            Builder ready for the surface stage.
        """

        _validate_v010_plan(plan)
        return cls(modules=modules, plan=plan, construction_fn=construction_fn)

    def add_surface(self) -> OpenMMSmokeBuilder:
        """Mark the Pd(111) surface stage as ready for construction.

        Returns
        -------
        OpenMMSmokeBuilder
            This builder for fluent stage chaining.
        """

        self._require_stage_absent(self._surface_added, self._SURFACE_STAGE)
        self._surface_added = True
        return self

    def add_sam_layer(self, template: Any) -> OpenMMSmokeBuilder:
        """Add the SAM layer stage after the surface stage.

        Parameters
        ----------
        template
            Internal molecule template for the SAM component.

        Returns
        -------
        OpenMMSmokeBuilder
            This builder for fluent stage chaining.
        """

        self._require_stage_present(self._surface_added, self._SURFACE_STAGE, self._SAM_STAGE)
        self._require_stage_absent(self._sam_template is not None, self._SAM_STAGE)
        _validate_single_sulfur_template(template, "SAM template")
        self._sam_template = template
        return self

    def add_reactants(self, template: Any, *, count: int) -> OpenMMSmokeBuilder:
        """Add reactants after the SAM layer has been declared.

        Parameters
        ----------
        template
            Internal molecule template for the reactant component.
        count
            Number of reactant molecules to construct.

        Returns
        -------
        OpenMMSmokeBuilder
            This builder for fluent stage chaining.
        """

        self._require_stage_present(
            self._sam_template is not None,
            self._SAM_STAGE,
            self._REACTANTS_STAGE,
        )
        self._require_stage_absent(self._reactant_template is not None, self._REACTANTS_STAGE)
        _validate_positive_count(count, "reactant count")
        self._reactant_template = template
        self._reactant_count = count
        return self

    def add_solvent(self, template: Any, *, count: int) -> OpenMMSmokeBuilder:
        """Add solvent after SAM and reactants have been declared.

        Parameters
        ----------
        template
            Internal molecule template for the solvent component.
        count
            Number of solvent molecules to pack and construct.

        Returns
        -------
        OpenMMSmokeBuilder
            This builder for fluent stage chaining.
        """

        self._require_stage_present(
            self._sam_template is not None,
            self._SAM_STAGE,
            self._SOLVENT_STAGE,
        )
        self._require_stage_present(
            self._reactant_template is not None,
            self._REACTANTS_STAGE,
            self._SOLVENT_STAGE,
        )
        self._require_stage_absent(self._solvent_template is not None, self._SOLVENT_STAGE)
        _validate_positive_count(count, "solvent count")
        self._solvent_template = template
        self._solvent_count = count
        return self

    def finalize(self, options: OpenMMSmokeBuildOptions) -> Any:
        """Construct and return concrete OpenMM smoke objects.

        Parameters
        ----------
        options
            Runtime construction options that do not belong to the immutable plan.

        Returns
        -------
        Any
            Concrete smoke build object returned by the injected backend callable.
        """

        if self._finalized:
            raise RuntimeError("OpenMM smoke builder has already been finalized")
        self._require_stage_present(self._surface_added, self._SURFACE_STAGE, "finalize")
        self._require_stage_present(self._sam_template is not None, self._SAM_STAGE, "finalize")
        self._require_stage_present(
            self._reactant_template is not None,
            self._REACTANTS_STAGE,
            "finalize",
        )
        self._require_stage_present(
            self._solvent_template is not None,
            self._SOLVENT_STAGE,
            "finalize",
        )

        build = self._construction_fn(
            self._modules,
            self._plan,
            self._sam_template,
            self._reactant_template,
            self._solvent_template,
            solvent_count=self._solvent_count,
            reactant_count=self._reactant_count,
            sulfur_height_nm=options.sulfur_height_nm,
            solvent_padding_nm=options.solvent_padding_nm,
            packmol_working_dir=options.packmol_working_dir,
            pressure_bar=options.pressure_bar,
            temperature_k=options.temperature_k,
            pd_s_sigma_nm=options.pd_s_sigma_nm,
            pd_s_epsilon_kcal_mol=options.pd_s_epsilon_kcal_mol,
        )
        self._finalized = True
        return build

    @staticmethod
    def _require_stage_absent(already_added: bool, stage: str) -> None:
        """Raise if a stage has already been declared."""

        if already_added:
            raise RuntimeError(f"{stage} stage has already been added")

    @staticmethod
    def _require_stage_present(is_present: bool, required_stage: str, requested_stage: str) -> None:
        """Raise if a requested stage is missing a prerequisite stage."""

        if not is_present:
            msg = f"cannot add {requested_stage} before {required_stage} stage"
            if requested_stage == "finalize":
                msg = f"cannot finalize before {required_stage} stage"
            raise RuntimeError(msg)


def _validate_v010_plan(plan: Any) -> None:
    """Validate smoke builder assumptions shared with the v0.1.0 planner."""

    config = plan.config
    if config.surface.facet != "111":
        raise ValueError("only facet '111' is supported by the OpenMM smoke builder")
    if config.surface.metal != "Pd":
        raise ValueError("only Pd surfaces are supported by the OpenMM smoke builder")
    if config.sam.grafting_density.unit != EXPECTED_GRAFTING_DENSITY_UNIT:
        msg = f"grafting density unit must be '{EXPECTED_GRAFTING_DENSITY_UNIT}'"
        raise ValueError(msg)
    if len(config.sam.components) != 1:
        raise NotImplementedError("alloy or mixed SAM components are not supported in v0.1.0")


def _validate_single_sulfur_template(template: Any, label: str) -> None:
    """Validate that an internal molecule template contains one sulfur atom."""

    atoms = getattr(template, "atoms", ())
    sulfur_count = sum(1 for atom in atoms if getattr(atom, "element", None) == "S")
    if sulfur_count != 1:
        raise ValueError(f"{label} must contain exactly one sulfur atom")


def _validate_positive_count(count: int, label: str) -> None:
    """Validate a positive molecule count for a staged component."""

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(f"{label} must be a positive integer")
