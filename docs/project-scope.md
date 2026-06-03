# SAMMD Project Scope

This document captures the initial research context and scope interpretation for SAMMD: a Python package for reproducible molecular dynamics simulations of self-assembled monolayers on metal supports.

## Goal

SAMMD should let a user specify a metal surface, one or more thiol SAM components, solvent mixture, salts, and reactant molecules from a validated YAML file, then build an OpenFF/OpenMM-ready simulation system that can be inspected and extended from Python.

The first scientific target is a propanethiol SAM (`CCCS`) on Pd(111), with small molecule ligands approaching the monolayer-covered metal surface.

The default periodic geometry should use a centered, double-sided slab:

```text
z: Bulk solvent -> Bottom SAM -> Metal slab -> Top SAM -> Bulk solvent
```

The top and bottom bulk solvent regions are the same phase through periodic boundary conditions in `z`. The slab is centered in the box, SAM molecules decorate both exposed surfaces, and the metal slab must be thick enough that species on one SAM/slab interface do not interact with the other interface through the nonbonded cutoff.

## MVP Boundary

Ship a Python-first workflow before a broad CLI.

Recommended MVP deliverables:

- Pixi-installable Python package with `src/` layout.
- Pydantic configuration models loaded from YAML.
- A minimal `sammd init` command that writes a commented template YAML file.
- A canonical Jupyter notebook showing build, parameterization, OpenMM setup, and basic inspection.
- A Pd(111) slab builder and propanethiol SAM placement with tunable grafting density.
- Mixed SAM support through multiple SAM component definitions with fractions or explicit counts.
- Centered, double-sided Pd(111) slab geometry with SAMs on both faces and solvent on both sides.
- OpenFF/SMIRNOFF parameterization for organic SAM, solvent, salts, and reactants where supported by OpenFF.
- CHARMM-INTERFACE-derived Fcc metal Lennard-Jones parameters converted to an OpenFF-compatible offxml resource.
- Visualization-friendly output files, with mmCIF/PDBx for topology/starting coordinates and DCD for trajectories.
- OpenMM thermodynamic reporting during simulation, configurable by the user.
- Pytest coverage for configuration validation, parameter conversion, deterministic builders, and import smoke tests.
- ReadTheDocs/Sphinx scaffold with a canonical workflow tutorial and YAML configuration tutorial.

Defer until after MVP:

- Full production CLI for build/run/submit workflows.
- SLURM or other HPC orchestration.
- Umbrella sampling automation.
- General metal facet support beyond the first Pd(111) path.
- Explicit metal-sulfur bonded topology terms unless needed to correct SAM geometry.
- Advanced analysis workflows beyond basic orientation observables.

## Reference Findings

### PolyzyMD

[PolyzyMD](https://github.com/joelaforet/polyzymd) is a useful architecture reference, not a template to copy wholesale.

Reusable ideas:

- Pixi environments for reproducible conda-forge installs.
- Pydantic YAML schema plus loader functions for user configuration.
- Small CLI entry points for initialization and validation.
- `pytest` plus `pytest-cov` in the developer environment.
- Sphinx/ReadTheDocs documentation organized by user need.

SAMMD should stay smaller than PolyzyMD for the prototype. The immediate value is a reliable system builder and exposed Python API.

### INTERFACE Force Field

The [INTERFACE force field repository](https://github.com/hendrikheinz/INTERFACE-force-field-and-surface-models) includes Fcc metals: Ag, Al, Au, Cu, Ni, Pb, Pd, and Pt.

The best MVP source is [CHARMM-INTERFACE](https://github.com/hendrikheinz/INTERFACE-force-field-and-surface-models/blob/master/charmm27_interface_v1_5.prm), because it uses Lennard-Jones 12-6 nonbonded parameters with `epsilon` and `Rmin/2`, which maps cleanly onto OpenFF SMIRNOFF `vdW` parameters.

Fcc metal entries from `charmm27_interface_v1_5.prm`:

| Metal | CHARMM epsilon (kcal/mol) | OpenFF epsilon (kcal/mol) | Rmin/2 (angstrom) |
| --- | ---: | ---: | ---: |
| Ag | -4.56 | 4.56 | 1.4775 |
| Al | -4.02 | 4.02 | 1.4625 |
| Au | -5.29 | 5.29 | 1.4755 |
| Cu | -4.72 | 4.72 | 1.3080 |
| Ni | -5.65 | 5.65 | 1.2760 |
| Pb | -2.93 | 2.93 | 1.7825 |
| Pd | -6.15 | 6.15 | 1.4095 |
| Pt | -7.80 | 7.80 | 1.4225 |

Conversion notes:

- CHARMM stores epsilon as negative in the parameter file; OpenFF expects a positive well depth.
- SMIRNOFF `vdW` can use `rmin_half`, matching CHARMM `Rmin/2` directly.
- If a downstream API requires sigma, use `sigma = 2 * rmin_half / 2**(1/6)`.
- PCFF-INTERFACE uses a 9-6 nonbonded form, so it is not the preferred MVP route for standard OpenFF/OpenMM 12-6 `vdW` handling.

### OpenFF And OpenMM

[OpenFF Interchange construction](https://docs.openforcefield.org/projects/interchange/en/stable/using/construction.html) supports constructing an `Interchange` from a SMIRNOFF `ForceField` and OpenFF `Molecule` or `Topology` objects.

Relevant behavior:

- `Interchange` stores topology, parameter collections, positions, box vectors, and velocities.
- Conformers on OpenFF `Molecule` objects can seed positions.
- Box vectors can be passed during construction or assigned afterward.
- `Interchange` can export to OpenMM `System`, `Topology`, positions, and `Simulation` objects.
- [SMIRNOFF](https://docs.openforcefield.org/projects/toolkit/en/stable/users/smirnoff.html) parameters are defined by SMIRKS patterns and explicit units.

The local notebook [`metal_example.ipynb`](file:///home/joelaforet/Desktop/SAMS_MD/metal_example.ipynb) demonstrates the exact OpenFF extension mechanism needed here:

- Add `LibraryCharges` by mapped SMILES.
- Add metal bond, angle, torsion, and `vdW` parameters directly to a `ForceField` object.
- Build an `Interchange` from the resulting force field.
- Minimize and convert the system to an OpenMM simulation.

### mBuild And Surface Coatings

[mBuild](https://github.com/mosdef-hub/mbuild) provides hierarchical `Compound`, `Box`, `Lattice`, `Port`, tiling, and packing concepts that fit the slab/SAM construction problem.

[surface_coatings](https://github.com/daico007/surface_coatings) shows useful mBuild patterns:

- `Monolayer` attaches chain compounds to a tiled surface through an `mb.Pattern`.
- `SolvatedMonolayer` fills a solvent box above the monolayer.
- The Au surface example uses `mb.Lattice` and `TiledCompound`-style construction for Fcc metals.

SAMMD should own its Pd(111) builder rather than depend on surface_coatings directly, because the MVP needs metal/facet-aware parameters, grafting-density controls, sulfur anchoring behavior, and OpenFF-ready topology handling.

### Solvent And Packing

[PolyzyMD solvent handling](https://github.com/joelaforet/polyzymd/blob/main/src/polyzymd/builders/solvent.py) provides a useful model for SAMMD solvent composition.

Reusable ideas:

- Compute molecule counts from box volume, target density, volume fractions, and molar concentrations.
- Require density for volume-fraction co-solvents.
- Add salts by concentration and optionally neutralize total system charge.
- Cache or reuse pre-parameterized solvent molecules so each copy has identical charges and parameters.
- Pack molecules with PACKMOL through the OpenFF ecosystem, rather than writing custom placement code.

SAMMD should keep this simpler than PolyzyMD initially: water, ethanol, simple monovalent salts, and reactants by concentration are enough for the canonical workflow.

### Sulfur Binding Sites

[Thermodynamics of Alkanethiol Self-Assembled Monolayer Assembly on Pd Surfaces](https://doi.org/10.1021/acs.langmuir.7b04351) is the most relevant Pd-specific reference identified so far. It establishes that Pd alkanethiol SAMs are a real target system and should be kept in the context bibliography.

Broader thiolate literature cautions that site preferences can be metal-, coverage-, and adsorbate-dependent. For example, Au(111) DFT/HREELS work reports methylthiolate near a bridge site displaced toward fcc hollow rather than a simple hollow-site picture ([Hayashi et al., J. Chem. Phys. 2001](https://doi.org/10.1063/1.1360245)).

For SAMMD MVP, use three-fold hollow placement on Pd(111) as the default modeling hypothesis, but keep the adsorption site configurable. The builder should not hard-code this assumption so future calibration against Pd-specific data can switch among `fcc_hollow`, `hcp_hollow`, `bridge`, and `atop` placement.

## Proposed Package Shape

Initial package modules:

- `sammd.config`: Pydantic models, YAML loading, YAML template generation.
- `sammd.surfaces`: Fcc slab builders, initially Pd(111), with lattice constants and facet metadata.
- `sammd.sam`: RDKit/OpenFF molecule creation, conformer generation, sulfur anchor detection, single- and mixed-component monolayer placement.
- `sammd.forcefields`: INTERFACE metal parameter registry, offxml generation, OpenFF force field assembly.
- `sammd.solvation`: solvent mixture, salt, and reactant count calculations from volume fractions and concentrations, followed by OpenFF/PACKMOL packing.
- `sammd.builders`: high-level builder. Current MVP code returns a lightweight deterministic build plan; future backend integration should add full OpenFF `Interchange` construction.
- `sammd.simulation`: thin, user-facing OpenMM setup and run helpers for the canonical notebook, not a large production run manager in the MVP.
- `sammd.io`: mmCIF/PDBx topology writing, DCD trajectory naming conventions, and visualization-oriented metadata helpers.
- `sammd.reporting`: OpenMM reporter configuration for trajectories and thermodynamic state data.
- `sammd.analysis`: orientation metrics and later umbrella-sampling support.
- `sammd.cli`: minimal `init` and maybe `validate` commands.

The current public Python API exposes a lightweight planning workflow that avoids heavy
OpenFF/OpenMM construction:

```python
from sammd import load_config, build_system

config = load_config("sammd.yaml")
plan = build_system(config, output_dir="outputs")
plan.write_planned_slab_mmcif()  # Writes outputs/planned_slab.cif by default
```

The returned plan contains validated configuration, a commensurate Pd(111) slab,
binding sites, seeded SAM placement choices, approximate solution composition counts,
and planned output paths. It is not a final OpenFF/OpenMM system and does not contain
complete atomic coordinates for SAM, solvent, salts, or reactants.

The future full-construction workflow should preserve the small API surface while adding
backend artifacts and simulation helpers, for example:

```python
from sammd import load_config, build_system

config = load_config("sammd.yaml")
system = build_system(config)
interchange = system.interchange
simulation = system.create_openmm_simulation()
```

## Scientific Design Decisions

### Simulation API Abstraction

SAMMD should provide a user-facing simulation interface for the canonical workflow, not require users to write pure OpenMM code for routine runs.

The interface should:

- Accept validated configuration objects and produce build artifacts.
- Expose simple methods for minimization, equilibration, production, and export in the notebook workflow.
- Return or expose the underlying OpenFF `Interchange` and OpenMM `System`, `Topology`, positions, and `Simulation` objects for advanced users.
- Hide implementation details of sulfur-metal anchoring behind configuration fields such as `anchor.mode`, `anchor.site`, and `anchor.nonbonded.scale_factor`.

For the nonbonded anchor proxy, users should only need to set the interaction magnitude, for example 4x or 6x stronger than the baseline sulfur-metal interaction. The implementation may use an OpenMM custom force, pair-specific nonbonded overrides, or another correct backend, but that choice should not leak into the user-facing API.

### Visualization And Output Files

SAMMD-generated systems should be optimized for visualization in tools such as PyMOL.

Default output artifacts for a built or simulated system should include:

- `topology.cif`: mmCIF/PDBx topology and starting coordinates.
- `trajectory.dcd`: DCD trajectory from OpenMM.
- `thermodynamics.csv`: tabular OpenMM state data from a thermodynamic reporter.
- Optional OpenMM restart/checkpoint artifacts for continuing simulations.

During the current lightweight planning milestone, `topology.cif` remains reserved for
the future complete topology. The slab-only visualization helper writes
`planned_slab.cif` by default and should not be interpreted as a complete topology or
final simulation cell.

mmCIF/PDBx should be preferred over legacy PDB because SAMMD systems may have many atoms, many solvent/reactant molecules, nonstandard residues, and metal particles. Atom names, residue names, chain IDs, molecule labels, and component metadata should be chosen so PyMOL sessions are easy to inspect: metal slab, top SAM, bottom SAM, solvent, salts, and reactants should be distinguishable by selection.

DCD should be the canonical trajectory format for the MVP. The topology/trajectory pair should load cleanly as `topology.cif` plus `trajectory.dcd` in PyMOL or common Python analysis tools.

### Thermodynamic Reporting

OpenMM simulations should report key thermodynamic quantities while they run. The user-facing configuration should allow users to choose output path, reporting interval, and reported fields.

Default report fields for normal runs should include at least:

- Step.
- Simulation time.
- Potential energy.
- Kinetic energy.
- Total energy.
- Temperature.
- Volume and density for periodic systems when available.
- Simulation speed and elapsed time when available.

For tests, reporter configuration should request every supported field so schema validation and reporter construction cover the full surface area. Users should be able to override this with a smaller field list for production runs.

### Mixed SAMs

Mixed SAMs are in scope, even if the canonical demo uses a single propanethiol component.

The configuration should allow a list of SAM components:

- `name`: human-readable component label.
- `smiles`: thiol molecule SMILES.
- `fraction` or `count`: surface composition control.
- Optional per-component anchor settings for future calibration.

The total grafting density applies to all SAM components combined. The default grafting density is `0.25 nm^2 / molecule`, equivalent to 4 molecules per nm^2. Placement should select binding sites deterministically from a seed, then assign component identities from fractions or counts. The same composition should decorate both slab faces by default, with a future extension point for side-specific compositions.

### Slab Geometry And PBC

The MVP box should be periodic in `x`, `y`, and `z`, with the Pd slab centered in `z`. The intended ordering is:

```text
Bulk solvent -> Bottom SAM -> Pd(111) slab -> Top SAM -> Bulk solvent
```

The bottom SAM should point toward negative `z`; the top SAM should point toward positive `z`. Through PBC, the two bulk solvent regions are one continuous bulk phase.

The Pd slab should be restrained by default. We are not simulating SAM assembly or metal reconstruction; the surface mainly provides geometry and INTERFACE-derived nonbonded parameters. Use positional restraints to keep Pd atoms near ideal lattice coordinates. The default positional restraint force constant is `10000 kJ mol^-1 nm^-2`, with the exact units and restraint form documented in the generated YAML template.

The slab must be thick enough that the two decorated interfaces do not see each other through the nonbonded cutoff. With a typical 0.9-1.2 nm cutoff and Pd(111) layer spacing of roughly `a / sqrt(3)`, a safe starting point is 8-10 Pd(111) layers. This should be configurable, validated against the chosen cutoff, and documented with a warning if the requested slab is too thin.

For the first notebook/demo, a practical starting system is roughly 5 x 5 nm in the lateral plane, 8 Pd layers, SAMs on both faces, and at least 3 nm solvent padding on each side before PBC wrapping. This is large enough for a full SAM coating and multiple cinnamaldehyde reactants (`C1=CC=C(C=C1)/C=C/C=O`) while still being plausible for prototype runs.

Default metal-sulfur anchoring should begin as a nonbonded proxy:

- Keep the SAM sulfur and surface metal atoms non-covalent in topology.
- Increase the relevant sulfur-metal interaction strength by a configurable factor, initially `4.0`.
- Keep the scale factor in user configuration, not source code, because users may tune values such as 4x, 5x, or 6x.
- Preserve a configuration/API path for future explicit bonded anchors.

Future explicit anchor mode should be designed as a replaceable strategy:

- `anchor.mode = "nonbonded"` for MVP.
- `anchor.mode = "bonded"` later for metal-sulfur bond, angle, and torsion parameters.
- A future angle target, such as 23 degrees relative to the surface, should not require rewriting the builder API.
- `anchor.site = "fcc_hollow"` should be the Pd(111) default, with other configurable site labels available later.

Solvent and reactant composition should be converted from user-facing units:

- Volume fractions for co-solvents, for example 50 percent v/v water/ethanol.
- Molarity for salts and reactants.
- Counts derived from simulation box volume, with deterministic rounding and validation warnings.
- OpenFF should parameterize solvent and reactant molecules where supported, and OpenFF/PACKMOL should place molecules according to target composition.
- TIP3P should be the default water model.

### Orientation Analysis

The canonical ligand-orientation observable is not finalized. A reasonable first proxy is:

- Define the surface normal as the `z` axis, with sign chosen by which SAM face the reactant approaches.
- Define a reactant orientation vector from the reactant center of mass to a user-selected atom, midpoint, or SMARTS atom selection.
- Track the angle between the reactant vector and the surface normal.

An angle near 90 degrees means the selected reactant vector lies roughly parallel to the surface, which may correspond to a flatter approach depending on the atom selection. This should stay configurable because cinnamaldehyde and future ligands may need different chemically meaningful vectors.

## Testing Strategy

High-value unit tests for the first implementation:

- YAML template loads into the Pydantic model.
- YAML template defaults include `fcc_hollow`, TIP3P, `0.25 nm^2 / molecule`, and `10000 kJ mol^-1 nm^-2` Pd restraints.
- YAML template includes configurable output/reporting sections for mmCIF, DCD, and thermodynamic state data.
- Invalid metal, facet, grafting density, solvent fraction, or concentration fails clearly.
- Mixed SAM fractions must sum to one, or explicit counts must match selected grafting sites.
- INTERFACE Fcc metal registry reproduces the table above.
- CHARMM epsilon sign conversion is tested.
- Generated offxml can be loaded by OpenFF `ForceField`.
- Pd(111) builder returns deterministic atom counts and box dimensions.
- Pd(111) slab thickness validation warns or fails when the slab is thinner than the configured nonbonded cutoff plus buffer.
- Propanethiol sulfur anchor detection is deterministic for `CCCS`.
- Cinnamaldehyde reactant parsing is deterministic for `C1=CC=C(C=C1)/C=C/C=O`.
- Reporter configuration maps user field names to OpenMM reporter arguments.
- Test reporter defaults request all supported thermodynamic fields.
- mmCIF and DCD output paths are deterministic and can be overridden.
- Python public imports are stable.

Integration tests should be marked separately because OpenFF/OpenMM/packmol can be slow or environment-sensitive.

## Documentation Plan

Use a small ReadTheDocs/Sphinx site from the start.

Initial pages:

- Install with pixi.
- Canonical SAMMD workflow tutorial.
- YAML configuration tutorial.
- Configuration reference generated or mirrored from Pydantic models.
- Contributor guide explaining package layout and where to add new metals/facets.

The docs should assume undergraduate contributors and make success states explicit.

## Resolved Decisions

- Pd(111) atoms should be restrained by default, not freely mobile.
- The MVP slab should be centered and decorated on both faces to use PBC in `z` cleanly.
- The slab should be thick enough that the two SAM/slab interfaces are separated by more than the nonbonded cutoff plus a buffer.
- The lateral box dimensions should be user-tunable; the first demo can start near 5 x 5 nm.
- The default sulfur site for Pd(111) should be `fcc_hollow`, but site type must be configurable.
- The sulfur-metal interaction scaling factor must be user-configurable and backend details should be hidden behind SAMMD's simulation interface.
- TIP3P is the default water model.
- The default Pd positional restraint force constant is `10000 kJ mol^-1 nm^-2`.
- The default grafting density is `0.25 nm^2 / molecule`.
- Solvent and reactant placement should use OpenFF/PACKMOL-style packing, with counts derived from target composition.
- Reactant orientation analysis can begin with a configurable COM-to-selection vector relative to the surface normal.
- mmCIF/PDBx is the default topology/starting-coordinate output format.
- DCD is the default trajectory output format.
- OpenMM thermodynamic state data should be written during runs, with tests requesting all supported fields and users allowed to override fields/intervals.

## Remaining Open Questions

- None at the current scope-planning level. Backend implementation choices remain, but the user-facing behavior is now specified.

## Implementation Readiness

The scope is ready for MVP scaffolding. The first implementation pass should create the package skeleton, pixi/pyproject configuration, Pydantic YAML schema, template generation, INTERFACE metal registry, and tests for the resolved defaults before implementing expensive OpenFF/OpenMM build steps.

## Links

- [PolyzyMD](https://github.com/joelaforet/polyzymd)
- [INTERFACE force field and surface models](https://github.com/hendrikheinz/INTERFACE-force-field-and-surface-models)
- [CHARMM-INTERFACE parameter file](https://github.com/hendrikheinz/INTERFACE-force-field-and-surface-models/blob/master/charmm27_interface_v1_5.prm)
- [OpenFF Interchange construction](https://docs.openforcefield.org/projects/interchange/en/stable/using/construction.html)
- [OpenFF SMIRNOFF documentation](https://docs.openforcefield.org/projects/toolkit/en/stable/users/smirnoff.html)
- [OpenFF Interchange OpenMM export](https://docs.openforcefield.org/projects/interchange/en/stable/using/output.html#openmm)
- [mBuild](https://github.com/mosdef-hub/mbuild)
- [mBuild data structures](https://mbuild.mosdef.org/en/stable/topic_guides/data_structures.html)
- [surface_coatings](https://github.com/daico007/surface_coatings)
- [Local metal OpenFF notebook](file:///home/joelaforet/Desktop/SAMS_MD/metal_example.ipynb)
- [PolyzyMD solvent builder](https://github.com/joelaforet/polyzymd/blob/main/src/polyzymd/builders/solvent.py)
- [Alkanethiol SAM assembly on Pd surfaces](https://doi.org/10.1021/acs.langmuir.7b04351)
- [Biomolecular force fields for alkanethiol SAM simulations](https://doi.org/10.1021/acs.jpcc.7b08092)
- [Methylthiolate adsorption on Au(111)](https://doi.org/10.1063/1.1360245)
