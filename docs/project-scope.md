# SAMMD Project Scope

This document captures the initial research context and scope interpretation for SAMMD: a Python package for reproducible molecular dynamics simulations of self-assembled monolayers on metal supports.

## Goal

SAMMD should let a user specify a metal surface, thiol SAM, solvent mixture, salts, and reactant molecules from a validated YAML file, then build an OpenFF/OpenMM-ready simulation system that can be inspected and extended from Python.

The first scientific target is a propanethiol SAM (`CCCS`) on Pd(111), with small molecule ligands approaching the monolayer-covered metal surface.

## MVP Boundary

Ship a Python-first workflow before a broad CLI.

Recommended MVP deliverables:

- Pixi-installable Python package with `src/` layout.
- Pydantic configuration models loaded from YAML.
- A minimal `sammd init` command that writes a commented template YAML file.
- A canonical Jupyter notebook showing build, parameterization, OpenMM setup, and basic inspection.
- A Pd(111) slab builder and propanethiol SAM placement with tunable grafting density.
- OpenFF/SMIRNOFF parameterization for organic SAM, solvent, salts, and reactants where supported by OpenFF.
- CHARMM-INTERFACE-derived Fcc metal Lennard-Jones parameters converted to an OpenFF-compatible offxml resource.
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

## Proposed Package Shape

Initial package modules:

- `sammd.config`: Pydantic models, YAML loading, YAML template generation.
- `sammd.surfaces`: Fcc slab builders, initially Pd(111), with lattice constants and facet metadata.
- `sammd.sam`: RDKit/OpenFF molecule creation, conformer generation, sulfur anchor detection, monolayer placement.
- `sammd.forcefields`: INTERFACE metal parameter registry, offxml generation, OpenFF force field assembly.
- `sammd.solvation`: solvent mixture, salt, and reactant count calculations from volume fractions and concentrations.
- `sammd.builders`: high-level system builder that returns structured build artifacts and an OpenFF `Interchange`.
- `sammd.simulation`: thin OpenMM setup helpers, not a large run manager in the MVP.
- `sammd.analysis`: orientation metrics and later umbrella-sampling support.
- `sammd.cli`: minimal `init` and maybe `validate` commands.

The public Python API should expose a small workflow surface:

```python
from sammd import load_config, build_system

config = load_config("sammd.yaml")
system = build_system(config)
interchange = system.interchange
simulation = system.create_openmm_simulation()
```

## Scientific Design Decisions

Default metal-sulfur anchoring should begin as a nonbonded proxy:

- Keep the SAM sulfur and surface metal atoms non-covalent in topology.
- Increase the relevant sulfur-metal interaction strength by a configurable factor, initially `4.0`.
- Preserve a configuration/API path for future explicit bonded anchors.

Future explicit anchor mode should be designed as a replaceable strategy:

- `anchor.mode = "nonbonded"` for MVP.
- `anchor.mode = "bonded"` later for metal-sulfur bond, angle, and torsion parameters.
- A future angle target, such as 23 degrees relative to the surface, should not require rewriting the builder API.

The metal slab treatment needs an early decision:

- Fixed atom positions, zero-mass metal atoms, or strong positional restraints are all plausible.
- The default should be explicit and documented because it changes physical interpretation and OpenMM integration.

Solvent and reactant composition should be converted from user-facing units:

- Volume fractions for co-solvents, for example 50 percent v/v water/ethanol.
- Molarity for salts and reactants.
- Counts derived from simulation box volume, with deterministic rounding and validation warnings.

## Testing Strategy

High-value unit tests for the first implementation:

- YAML template loads into the Pydantic model.
- Invalid metal, facet, grafting density, solvent fraction, or concentration fails clearly.
- INTERFACE Fcc metal registry reproduces the table above.
- CHARMM epsilon sign conversion is tested.
- Generated offxml can be loaded by OpenFF `ForceField`.
- Pd(111) builder returns deterministic atom counts and box dimensions.
- Propanethiol sulfur anchor detection is deterministic for `CCCS`.
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

## Open Questions

- Should Pd(111) atoms be frozen, position-restrained, or mobile by default?
- What default Pd lattice constant and slab thickness should the MVP use?
- Should the initial monolayer place sulfur over atop, bridge, or hollow sites?
- Should the 4x sulfur-metal interaction be implemented with an OpenMM custom nonbonded force or by assigning a distinct sulfur/metal parameter combination through SMIRNOFF?
- Which solvent model should be the default for mixed water/ethanol systems in the OpenFF/OpenMM stack?
- What orientation observable should be considered canonical for the first ligand-approach analysis?

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
