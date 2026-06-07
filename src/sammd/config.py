"""Validated YAML configuration for SAMMD system-building workflows."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MOLE_FRACTION_TOLERANCE = 1.0e-6
KNOWN_COSOLVENT_MOLAR_MASSES_G_MOL = {"ethanol": 46.06844}
RESIDUE_NAME_PATTERN = re.compile(r"^[A-Z0-9]{3}$")


class SAMMDBaseModel(BaseModel):
    """Strict base model for all SAMMD configuration sections."""

    model_config = ConfigDict(extra="forbid")


class ExperimentConfig(SAMMDBaseModel):
    """Human-readable metadata for one buildable system."""

    name: str = Field(default="propanethiol_cinnamaldehyde_pd111", min_length=1)
    description: str = "Cinnamaldehyde near a propanethiol SAM on Pd(111)"
    seed: int = Field(default=2026, ge=0)


class SurfaceConfig(SAMMDBaseModel):
    """Metal surface exposed to the SAM."""

    metal: str = "Pd"
    facet: str = "111"
    lateral_size: tuple[float, float] = (2.0, 2.0)

    @model_validator(mode="after")
    def _validate_registered_fcc_surface(self) -> SurfaceConfig:
        """Require a surface present in the Fcc metadata registry."""

        from sammd.surfaces import get_fcc_surface_metadata

        get_fcc_surface_metadata(self.metal, self.facet)
        return self

    @field_validator("lateral_size")
    @classmethod
    def _validate_lateral_size(cls, value: tuple[float, float]) -> tuple[float, float]:
        """Validate positive lateral surface dimensions."""

        if len(value) != 2 or any(dimension <= 0 for dimension in value):
            msg = "lateral_size must contain two positive dimensions"
            raise ValueError(msg)
        return value


class SAMComponentConfig(SAMMDBaseModel):
    """Single SAM molecular component definition."""

    name: str = Field(min_length=1)
    residue_name: str
    smiles: str = Field(min_length=1)
    fraction: float | None = Field(default=None, gt=0, le=1)
    count: int | None = Field(default=None, gt=0)
    extended_length_nm: float | None = Field(default=None, gt=0)

    @field_validator("residue_name")
    @classmethod
    def _validate_residue_name(cls, value: str) -> str:
        """Require a PDB-style three-character residue code."""

        return validate_residue_name(value)

    @model_validator(mode="after")
    def _validate_composition_mode(self) -> SAMComponentConfig:
        """Require exactly one composition control per SAM component."""

        if (self.fraction is None) == (self.count is None):
            msg = "each SAM component must define exactly one of fraction or count"
            raise ValueError(msg)
        return self


class SAMConfig(SAMMDBaseModel):
    """SAM composition and grafting-density configuration."""

    grafting_density: float = Field(default=0.25, gt=0)
    components: list[SAMComponentConfig] = Field(
        default_factory=lambda: [
            SAMComponentConfig(
                name="propanethiol",
                residue_name="PTL",
                smiles="CCCS",
                fraction=1.0,
            )
        ],
        min_length=1,
    )

    @model_validator(mode="after")
    def _validate_mixed_composition(self) -> SAMConfig:
        """Validate that mixed SAMs use fractions or counts consistently."""

        fraction_count = sum(component.fraction is not None for component in self.components)
        explicit_count = sum(component.count is not None for component in self.components)
        if fraction_count and explicit_count:
            msg = "SAM components must not mix fraction and count composition modes"
            raise ValueError(msg)
        total_fraction = sum(component.fraction or 0.0 for component in self.components)
        if fraction_count and abs(total_fraction - 1.0) > MOLE_FRACTION_TOLERANCE:
            msg = "SAM component fractions must sum to 1.0"
            raise ValueError(msg)
        return self

    def validate_component_counts(self, total_sites: int) -> None:
        """Validate explicit SAM component counts against a supplied site count."""

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


class ReactantConfig(SAMMDBaseModel):
    """Reactant molecule placed near the SAM surface."""

    name: str = Field(min_length=1)
    residue_name: str
    smiles: str = Field(min_length=1)
    count: int | None = Field(default=None, gt=0)
    concentration: float | None = Field(default=None, gt=0)
    initial_height_above_sam: float = Field(default=0.3, gt=0)

    @field_validator("residue_name")
    @classmethod
    def _validate_residue_name(cls, value: str) -> str:
        """Require a PDB-style three-character residue code."""

        return validate_residue_name(value)

    @model_validator(mode="after")
    def _validate_amount_mode(self) -> ReactantConfig:
        """Require exactly one reactant amount control."""

        if (self.count is None) == (self.concentration is None):
            msg = "each reactant must define exactly one of count or concentration"
            raise ValueError(msg)
        return self


class SolventComponentConfig(SAMMDBaseModel):
    """Solvent component specified by solvent-only mole fraction."""

    name: str = Field(min_length=1)
    residue_name: str
    smiles: str | None = None
    mole_fraction: float = Field(gt=0, le=1)
    density: float | None = Field(default=None, gt=0)
    molar_mass: float | None = Field(default=None, gt=0)

    @field_validator("residue_name")
    @classmethod
    def _validate_residue_name(cls, value: str) -> str:
        """Require a PDB-style three-character residue code."""

        return validate_residue_name(value)


class SolventConfig(SAMMDBaseModel):
    """Solvent composition and z-direction padding."""

    padding: float = Field(default=3.0, gt=0)
    components: list[SolventComponentConfig] = Field(
        default_factory=lambda: [
            SolventComponentConfig(
                name="ethanol",
                residue_name="EOH",
                smiles="CCO",
                mole_fraction=1.0,
                density=0.789,
                molar_mass=46.06844,
            )
        ],
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
            if component.name.lower() == "water":
                continue
            if component.density is None:
                msg = f"solvent component '{component.name}' must define density"
                raise ValueError(msg)
            if (
                component.molar_mass is None
                and component.name.lower() not in KNOWN_COSOLVENT_MOLAR_MASSES_G_MOL
            ):
                msg = (
                    f"solvent component '{component.name}' must define molar_mass or use a "
                    "supported built-in name"
                )
                raise ValueError(msg)
        return self


class IonConfig(SAMMDBaseModel):
    """One ionic species in an explicitly stoichiometric salt."""

    name: str = Field(min_length=1)
    residue_name: str
    smiles: str = Field(min_length=1)
    count_per_formula_unit: int = Field(gt=0)

    @field_validator("residue_name")
    @classmethod
    def _validate_residue_name(cls, value: str) -> str:
        """Require a PDB-style three-character residue code."""

        return validate_residue_name(value)


class SaltConfig(SAMMDBaseModel):
    """Dissolved salt with explicit cation/anion stoichiometry."""

    name: str = Field(min_length=1)
    concentration: float = Field(gt=0)
    cation: IonConfig
    anion: IonConfig


class PackmolConfig(SAMMDBaseModel):
    """PACKMOL settings used during molecule packing."""

    tolerance: float = Field(default=1.8, gt=0)
    nloop: int = Field(default=200, gt=0)


class PackingConfig(SAMMDBaseModel):
    """Packing backend configuration."""

    packmol: PackmolConfig = Field(default_factory=PackmolConfig)


class MetalForceFieldConfig(SAMMDBaseModel):
    """Metal force-field selection."""

    type: Literal["interface"] = "interface"
    resource: str = "interface_fcc_metals.offxml"


class ParameterizationConfig(SAMMDBaseModel):
    """Force-field choices for system building and parameterization."""

    small_molecule_force_field: str = "openff-2.2.1.offxml"
    charge_model: str = "openff-gnn-am1bcc-1.0.0.pt"
    metal_force_field: MetalForceFieldConfig = Field(default_factory=MetalForceFieldConfig)
    nonbonded_cutoff: float = Field(default=1.0, gt=0)


class OutputFilesConfig(SAMMDBaseModel):
    """File names written by the system builder."""

    topology: str = "topology.cif"
    positions: str = "positions.cif"
    openff_interchange: str = "interchange.json"
    openmm_system: str = "system.xml"
    build_summary: str = "build_summary.json"
    resolved_config: str = "resolved_config.yaml"


class OutputsConfig(SAMMDBaseModel):
    """Output directory and system-build artifact file names."""

    directory: str = "outputs/propanethiol_cinnamaldehyde_pd111"
    files: OutputFilesConfig = Field(default_factory=OutputFilesConfig)


class ReporterConfig(SAMMDBaseModel):
    """OpenMM StateDataReporter field selection helper.

    This model is intentionally not part of the top-level YAML system-building
    schema. It remains available for OpenMM teaching utilities and runtime helper
    tests that configure reporters from Python.
    """

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


class SAMMDConfig(SAMMDBaseModel):
    """Top-level SAMMD system-building configuration model."""

    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    surface: SurfaceConfig = Field(default_factory=SurfaceConfig)
    sam: SAMConfig = Field(default_factory=SAMConfig)
    reactants: list[ReactantConfig] = Field(
        default_factory=lambda: [
            ReactantConfig(
                name="cinnamaldehyde",
                residue_name="CIN",
                smiles="C1=CC=C(C=C1)/C=C/C=O",
                count=1,
                initial_height_above_sam=0.3,
            )
        ]
    )
    solvent: SolventConfig = Field(default_factory=SolventConfig)
    salts: list[SaltConfig] = Field(default_factory=list)
    packing: PackingConfig = Field(default_factory=PackingConfig)
    parameterization: ParameterizationConfig = Field(default_factory=ParameterizationConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)


CONFIG_TEMPLATE = """# ============================================================================
# SAMMD: Created by Joseph R. Laforet Jr.
# Self-Assembled Monolayers studied with Molecular Dynamics
# ============================================================================

# ============================================================================
# SAMMD System Configuration Template
# ============================================================================
#
# INSTRUCTIONS:
# 1. Save this file as config.yaml in your project directory.
# 2. Edit the values for your surface, SAM, solvent, salts, and reactants.
# 3. Validate your configuration:
#
#      sammd validate config.yaml
#
# 4. Validate and build the current inspection artifacts:
#
#      sammd build config.yaml --output-dir outputs/my_system --overwrite
#
# 5. Inspect topology.cif, build_summary.json, and resolved_config.yaml.
#    OpenMM/OpenFF backend exports are reserved target work.
#
# This file defines the molecular system only.
# It does NOT define equilibration, production MD, thermostats, barostats,
# trajectory saving, or OpenMM simulation phases.
#
# ============================================================================

# ============================================================================
# Experiment Metadata
# ============================================================================
experiment:
  name: propanethiol_cinnamaldehyde_pd111
  description: Cinnamaldehyde near a propanethiol SAM on Pd(111)
  seed: 2026


# ============================================================================
# Metal Surface
# ============================================================================
# SAMMD currently supports registered Fcc(111) INTERFACE metals and defaults to
# Pd(111). Registered metals: Ag, Al, Au, Cu, Ni, Pb, Pd, and Pt.
#
# The slab thickness is chosen automatically from the metal geometry and
# nonbonded cutoff so that periodic metal surfaces do not interact through
# the slab in the z direction.
#
surface:
  metal: Pd
  facet: "111"
  lateral_size: [2.0, 2.0]  # nm, x and y size of the surface


# ============================================================================
# Self-Assembled Monolayer
# ============================================================================
# Each SAM molecule must be a neutral thiol with an HS/implicit-H sulfur
# terminus, not a pre-deprotonated thiolate. SAMMD uses the sulfur atom for
# placement on the metal surface.
#
# Metal-S attachment is represented/planned internally as a strengthened
# nonbonded interaction, not as covalent, quantum, or reactive chemistry. This
# is not exposed as a beginner YAML knob yet.
#
# residue_name must be a 3-character PDB residue code.
#
# For mixed SAMs, list multiple components and make fractions sum to 1.0.
#
# extended_length_nm is an optional advanced override for the fully extended SAM
# length from sulfur anchor to tail tip. If omitted, SAMMD uses a lightweight,
# conservative SMILES heuristic with a 0.95 nm minimum default.
#
sam:
  grafting_density: 0.25  # nm^2 / molecule

  components:
    - name: propanethiol
      residue_name: PTL
      smiles: CCCS
      fraction: 1.0

# Example mixed SAM:
#
# sam:
#   grafting_density: 0.25  # nm^2 / molecule
#   components:
#     - name: propanethiol
#       residue_name: PTL
#       smiles: CCCS
#       fraction: 0.75
#     - name: mercaptoethanol
#       residue_name: MCE
#       smiles: OCCS
#       fraction: 0.25


# ============================================================================
# Reactants
# ============================================================================
# Molecules that will be placed near the SAM surface.
#
# residue_name must be a 3-character PDB residue code.
#
# Use exactly ONE of:
#   - count
#   - concentration
#
# concentration is interpreted as millimolar.
#
# initial_height_above_sam controls the starting distance above the SAM.
# This is only an initial placement choice. It is not a restraint.
#
reactants:
  - name: cinnamaldehyde
    residue_name: CIN
    smiles: C1=CC=C(C=C1)/C=C/C=O
    count: 1
    initial_height_above_sam: 0.3  # nm

# Example concentration-based reactant:
#
# reactants:
#   - name: cinnamaldehyde
#     residue_name: CIN
#     smiles: C1=CC=C(C=C1)/C=C/C=O
#     concentration: 50.0  # mM
#     initial_height_above_sam: 0.3  # nm


# ============================================================================
# Solvent
# ============================================================================
# Solvent is packed above and around the slab/SAM/reactant system.
#
# padding is the requested z distance from fully extended SAM tips to the box
# boundary. The same planned box volume is used for solvent/reactant/salt counts.
#
# residue_name must be a 3-character PDB residue code.
#
solvent:
  padding: 3.0  # nm, distance from fully extended SAM tips to box boundary

  components:
    - name: ethanol
      residue_name: EOH
      smiles: CCO
      mole_fraction: 1.0
      density: 0.789  # g/mL
      molar_mass: 46.06844  # g/mol

# Example water solvent:
#
# solvent:
#   padding: 3.0  # nm, distance from fully extended SAM tips to box boundary
#   components:
#     - name: water
#       residue_name: HOH
#       smiles: O
#       mole_fraction: 1.0
#       density: 0.997  # g/mL
#       molar_mass: 18.01528  # g/mol


# ============================================================================
# Salts
# ============================================================================
# Optional dissolved salts.
# Leave as [] if no salt is needed.
#
# concentration is interpreted as molar.
#
# Each ion gets its own residue_name so it can be selected separately
# during visualization and analysis.
#
# residue_name must be a 3-character PDB residue code.
#
# count_per_formula_unit defines explicit salt stoichiometry.
#
salts: []

# Example NaCl:
#
# salts:
#   - name: sodium_chloride
#     concentration: 0.15  # M
#
#     cation:
#       name: sodium
#       residue_name: SOD
#       smiles: "[Na+]"
#       count_per_formula_unit: 1
#
#     anion:
#       name: chloride
#       residue_name: CLA
#       smiles: "[Cl-]"
#       count_per_formula_unit: 1
#
# Example sodium sulfate:
#
# salts:
#   - name: sodium_sulfate
#     concentration: 0.05  # M
#
#     cation:
#       name: sodium
#       residue_name: SOD
#       smiles: "[Na+]"
#       count_per_formula_unit: 2
#
#     anion:
#       name: sulfate
#       residue_name: SUL
#       smiles: "O=S(=O)([O-])[O-]"
#       count_per_formula_unit: 1


# ============================================================================
# Packing
# ============================================================================
# PACKMOL options used when placing solvent molecules around the fixed
# slab/SAM/reactant structure.
#
packing:
  packmol:
    tolerance: 1.8  # Angstrom
    nloop: 200


# ============================================================================
# Parameterization
# ============================================================================
# Force-field choices recorded and validated by the current lightweight builder.
#
# The small-molecule force field is used for SAM molecules, reactants,
# solvent molecules, and salts.
#
# Metal atoms use the SAMMD INTERFACE force-field port.
#
parameterization:
  small_molecule_force_field: openff-2.2.1.offxml
  charge_model: openff-gnn-am1bcc-1.0.0.pt

  metal_force_field:
    type: interface
    resource: interface_fcc_metals.offxml

  nonbonded_cutoff: 1.0  # nm


# ============================================================================
# Outputs
# ============================================================================
# Current files written by the lightweight system builder:
#   - topology.cif
#   - build_summary.json
#   - resolved_config.yaml
#
# Reserved future backend artifact names:
#   - positions.cif
#   - interchange.json
#   - system.xml
#
# None of these are trajectory outputs from an MD simulation.
#
outputs:
  directory: outputs/propanethiol_cinnamaldehyde_pd111

  files:
    topology: topology.cif
    positions: positions.cif
    openff_interchange: interchange.json
    openmm_system: system.xml
    build_summary: build_summary.json
    resolved_config: resolved_config.yaml
"""


def validate_residue_name(value: str) -> str:
    """Validate and normalize a 3-character PDB residue code."""

    normalized = value.upper()
    if normalized != value or RESIDUE_NAME_PATTERN.fullmatch(normalized) is None:
        msg = "residue_name must be exactly 3 uppercase letters or digits"
        raise ValueError(msg)
    return normalized


def load_config_dict(data: dict[str, Any]) -> SAMMDConfig:
    """Load and validate a SAMMD configuration from a mapping."""

    return SAMMDConfig.model_validate(data)


def load_config(path: str | Path) -> SAMMDConfig:
    """Load and validate a SAMMD YAML configuration file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        msg = "configuration root must be a mapping"
        raise ValueError(msg)
    return load_config_dict(data)
