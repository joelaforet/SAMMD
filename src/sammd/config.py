"""Validated YAML configuration for SAMMD MVP workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AnchorMode = Literal["nonbonded", "bonded"]
AnchorSite = Literal["fcc_hollow", "hcp_hollow", "bridge", "atop"]
Facet = Literal["111"]
Metal = Literal["Pd"]
WaterModel = Literal["TIP3P"]
EXPECTED_GRAFTING_DENSITY_UNIT = "nm^2 / molecule"
EXPECTED_PD_RESTRAINT_UNIT = "kJ mol^-1 nm^-2"
KNOWN_COSOLVENT_MOLAR_MASSES_G_MOL = {"ethanol": 46.06844}
MOLE_FRACTION_TOLERANCE = 1.0e-6


class SAMMDBaseModel(BaseModel):
    """Strict base model for all SAMMD configuration sections."""

    model_config = ConfigDict(extra="forbid")


class UnitValue(SAMMDBaseModel):
    """Numeric value paired with a human-readable unit string."""

    value: float = Field(gt=0)
    unit: str = Field(min_length=1)


class SlabConfig(SAMMDBaseModel):
    """Metal slab geometry and restraint defaults."""

    layers: int = Field(default=8, gt=0)
    lateral_size_nm: tuple[float, float] = (5.0, 5.0)
    centered: bool = True
    double_sided: bool = True
    positional_restraint: UnitValue = Field(
        default_factory=lambda: UnitValue(value=10000.0, unit=EXPECTED_PD_RESTRAINT_UNIT)
    )

    @model_validator(mode="after")
    def _validate_restraint_unit(self) -> SlabConfig:
        """Validate the expected Pd positional restraint unit."""

        if self.positional_restraint.unit != EXPECTED_PD_RESTRAINT_UNIT:
            msg = f"positional_restraint unit must be '{EXPECTED_PD_RESTRAINT_UNIT}'"
            raise ValueError(msg)
        return self

    @field_validator("lateral_size_nm")
    @classmethod
    def _validate_lateral_size(cls, value: tuple[float, float]) -> tuple[float, float]:
        """Validate positive lateral box dimensions."""

        if len(value) != 2 or any(dimension <= 0 for dimension in value):
            msg = "lateral_size_nm must contain two positive dimensions"
            raise ValueError(msg)
        return value


class SurfaceConfig(SAMMDBaseModel):
    """Metal surface configuration."""

    metal: Metal = "Pd"
    facet: Facet = "111"
    slab: SlabConfig = Field(default_factory=SlabConfig)


class AnchorNonbondedConfig(SAMMDBaseModel):
    """Nonbonded sulfur-metal anchor proxy configuration."""

    scale_factor: float = Field(default=4.0, gt=0)


class AnchorConfig(SAMMDBaseModel):
    """SAM anchor strategy and adsorption site configuration."""

    mode: AnchorMode = "nonbonded"
    site: AnchorSite = "fcc_hollow"
    nonbonded: AnchorNonbondedConfig = Field(default_factory=AnchorNonbondedConfig)


class SAMComponentConfig(SAMMDBaseModel):
    """Single SAM molecular component definition."""

    name: str = Field(min_length=1)
    smiles: str = Field(min_length=1)
    fraction: float | None = Field(default=None, gt=0, le=1)
    count: int | None = Field(default=None, gt=0)
    anchor: AnchorConfig | None = None

    @model_validator(mode="after")
    def _validate_composition_mode(self) -> SAMComponentConfig:
        """Require exactly one composition control per SAM component."""

        if (self.fraction is None) == (self.count is None):
            msg = "each SAM component must define exactly one of fraction or count"
            raise ValueError(msg)
        return self


class SAMConfig(SAMMDBaseModel):
    """SAM composition and grafting-density configuration."""

    grafting_density: UnitValue = Field(
        default_factory=lambda: UnitValue(value=0.25, unit=EXPECTED_GRAFTING_DENSITY_UNIT)
    )
    anchor: AnchorConfig = Field(default_factory=AnchorConfig)
    components: list[SAMComponentConfig] = Field(
        default_factory=lambda: [
            SAMComponentConfig(name="propanethiol", smiles="CCCS", fraction=1.0)
        ],
        min_length=1,
    )

    @model_validator(mode="after")
    def _validate_mixed_composition(self) -> SAMConfig:
        """Validate that mixed SAMs use fractions or counts consistently."""

        if self.grafting_density.unit != EXPECTED_GRAFTING_DENSITY_UNIT:
            msg = f"grafting_density unit must be '{EXPECTED_GRAFTING_DENSITY_UNIT}'"
            raise ValueError(msg)
        fraction_count = sum(component.fraction is not None for component in self.components)
        explicit_count = sum(component.count is not None for component in self.components)
        if fraction_count and explicit_count:
            msg = "SAM components must not mix fraction and count composition modes"
            raise ValueError(msg)
        total_fraction = sum(component.fraction or 0.0 for component in self.components)
        if fraction_count and abs(total_fraction - 1.0) > 1e-6:
            msg = "SAM component fractions must sum to 1.0"
            raise ValueError(msg)
        return self

    def validate_component_counts(self, total_sites: int) -> None:
        """Validate explicit SAM component counts against selected grafting sites.

        Parameters
        ----------
        total_sites
            Number of selected grafting sites available for one decorated composition.
        """

        if total_sites <= 0:
            msg = "total_sites must be positive"
            raise ValueError(msg)
        counts = [component.count for component in self.components]
        if any(count is None for count in counts):
            msg = "SAM component count validation requires explicit count mode"
            raise ValueError(msg)
        total_count = sum(count or 0 for count in counts)
        if total_count != total_sites:
            msg = f"SAM component counts sum to {total_count}, but total_sites is {total_sites}"
            raise ValueError(msg)


class SolventComponentConfig(SAMMDBaseModel):
    """Solvent component specified by solvent-only mole fraction."""

    name: str = Field(min_length=1)
    smiles: str | None = None
    mole_fraction: float = Field(gt=0, le=1)
    density_g_ml: float | None = Field(default=None, gt=0)
    molar_mass_g_mol: float | None = Field(default=None, gt=0)


class SolventConfig(SAMMDBaseModel):
    """Solvent model and composition configuration."""

    water_model: WaterModel = "TIP3P"
    padding_nm: float = Field(default=3.0, gt=0)
    components: list[SolventComponentConfig] = Field(
        default_factory=lambda: [SolventComponentConfig(name="water", mole_fraction=1.0)],
        min_length=1,
    )

    @model_validator(mode="after")
    def _validate_mole_fractions(self) -> SolventConfig:
        """Require solvent-only mole fractions to form one bulk phase."""

        total = sum(component.mole_fraction for component in self.components)
        if abs(total - 1.0) > MOLE_FRACTION_TOLERANCE:
            msg = "solvent component mole fractions must sum to 1.0"
            raise ValueError(msg)
        for component in self.components:
            if component.name.lower() != "water" and component.density_g_ml is None:
                msg = f"co-solvent '{component.name}' must define density_g_ml"
                raise ValueError(msg)
            if (
                component.name.lower() != "water"
                and component.molar_mass_g_mol is None
                and component.name.lower() not in KNOWN_COSOLVENT_MOLAR_MASSES_G_MOL
            ):
                msg = (
                    f"co-solvent '{component.name}' must define molar_mass_g_mol or use a "
                    "supported built-in name"
                )
                raise ValueError(msg)
        return self


class SaltConfig(SAMMDBaseModel):
    """Salt composition specified by molar concentration."""

    cation: str = Field(default="Na+", min_length=1)
    anion: str = Field(default="Cl-", min_length=1)
    concentration_molar: float = Field(default=0.0, ge=0)
    neutralize: bool = True


class ReactantConfig(SAMMDBaseModel):
    """Reactant molecule specified by SMILES and target concentration."""

    name: str = Field(min_length=1)
    smiles: str = Field(min_length=1)
    concentration_millimolar: float = Field(gt=0)


class OutputConfig(SAMMDBaseModel):
    """Default visualization and reporter output paths."""

    topology: str = "topology.cif"
    trajectory: str = "trajectory.dcd"
    thermodynamics: str = "thermodynamics.csv"
    checkpoint: str | None = None
    state: str | None = None


class ReporterConfig(SAMMDBaseModel):
    """Thermodynamic reporter field selection."""

    interval_steps: int = Field(default=1000, gt=0)
    test_all_fields: bool = False
    fields: list[str] = Field(
        default_factory=lambda: [
            "step",
            "time",
            "potential_energy",
            "kinetic_energy",
            "total_energy",
            "temperature",
            "volume",
            "density",
            "speed",
            "elapsed_time",
        ],
        min_length=1,
    )

    @field_validator("fields")
    @classmethod
    def _validate_supported_fields(cls, value: list[str]) -> list[str]:
        """Validate reporter field names against the lightweight registry."""

        from sammd.reporting import SUPPORTED_THERMODYNAMIC_FIELDS

        unknown = sorted(set(value) - set(SUPPORTED_THERMODYNAMIC_FIELDS))
        if unknown:
            msg = f"unsupported reporter fields: {', '.join(unknown)}"
            raise ValueError(msg)
        if len(set(value)) != len(value):
            msg = "reporter fields must not contain duplicates"
            raise ValueError(msg)
        return value


class SimulationConfig(SAMMDBaseModel):
    """Lightweight simulation defaults used before backend construction."""

    timestep_fs: float = Field(default=2.0, gt=0)
    temperature_k: float = Field(default=300.0, gt=0)
    pressure_bar: float = Field(default=1.0, gt=0)
    nonbonded_cutoff_nm: float = Field(default=1.0, gt=0)
    slab_cutoff_buffer_nm: float = Field(default=0.5, ge=0)
    seed: int = Field(default=2026, ge=0)


class SAMMDConfig(SAMMDBaseModel):
    """Top-level SAMMD configuration model."""

    surface: SurfaceConfig = Field(default_factory=SurfaceConfig)
    sam: SAMConfig = Field(default_factory=SAMConfig)
    solvent: SolventConfig = Field(default_factory=SolventConfig)
    salts: list[SaltConfig] = Field(default_factory=list)
    reactants: list[ReactantConfig] = Field(
        default_factory=lambda: [
            ReactantConfig(
                name="cinnamaldehyde",
                smiles="C1=CC=C(C=C1)/C=C/C=O",
                concentration_millimolar=50.0,
            )
        ]
    )
    output: OutputConfig = Field(default_factory=OutputConfig)
    reporters: ReporterConfig = Field(default_factory=ReporterConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)

    @model_validator(mode="after")
    def _validate_slab_thickness(self) -> SAMMDConfig:
        """Require the Pd(111) slab to exceed the cutoff plus buffer."""

        from sammd.surfaces import get_fcc_surface_metadata

        surface_metadata = get_fcc_surface_metadata(self.surface.metal, self.surface.facet)
        thickness_nm = surface_metadata.slab_thickness_nm(self.surface.slab.layers)
        minimum_thickness_nm = (
            self.simulation.nonbonded_cutoff_nm + self.simulation.slab_cutoff_buffer_nm
        )
        if thickness_nm <= minimum_thickness_nm:
            msg = (
                "Pd(111) slab thickness must exceed nonbonded_cutoff_nm plus "
                f"slab_cutoff_buffer_nm; got {thickness_nm:.3f} nm and require more than "
                f"{minimum_thickness_nm:.3f} nm"
            )
            raise ValueError(msg)
        return self


CONFIG_TEMPLATE = """# SAMMD MVP configuration template
# Defaults follow docs/project-scope.md and avoid backend-specific build settings.
surface:
  metal: Pd
  facet: "111"
  slab:
    layers: 8
    lateral_size_nm: [5.0, 5.0]
    centered: true
    double_sided: true
    positional_restraint:
      value: 10000.0
      unit: kJ mol^-1 nm^-2

sam:
  grafting_density:
    value: 0.25
    unit: nm^2 / molecule
  anchor:
    mode: nonbonded
    site: fcc_hollow
    nonbonded:
      scale_factor: 4.0
  components:
    - name: propanethiol
      smiles: CCCS
      fraction: 1.0

solvent:
  water_model: TIP3P
  padding_nm: 3.0
  components:
    - name: water
      mole_fraction: 1.0

salts: []

reactants:
  - name: cinnamaldehyde
    smiles: C1=CC=C(C=C1)/C=C/C=O
    concentration_millimolar: 50.0

output:
  topology: topology.cif
  trajectory: trajectory.dcd
  thermodynamics: thermodynamics.csv

reporters:
  interval_steps: 1000
  test_all_fields: false
  fields:
    - step
    - time
    - potential_energy
    - kinetic_energy
    - total_energy
    - temperature
    - volume
    - density
    - speed
    - elapsed_time

simulation:
  timestep_fs: 2.0
  temperature_k: 300.0
  pressure_bar: 1.0
  nonbonded_cutoff_nm: 1.0
  slab_cutoff_buffer_nm: 0.5
  seed: 2026
"""


def load_config_dict(data: dict[str, Any]) -> SAMMDConfig:
    """Load and validate a SAMMD configuration from a mapping.

    Parameters
    ----------
    data
        Parsed configuration data.

    Returns
    -------
    SAMMDConfig
        Validated configuration object.
    """

    return SAMMDConfig.model_validate(data)


def load_config(path: str | Path) -> SAMMDConfig:
    """Load and validate a SAMMD YAML configuration file.

    Parameters
    ----------
    path
        Path to a YAML configuration file.

    Returns
    -------
    SAMMDConfig
        Validated configuration object.
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        msg = "configuration root must be a mapping"
        raise ValueError(msg)
    return load_config_dict(data)
