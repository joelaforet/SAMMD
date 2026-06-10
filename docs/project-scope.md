# SAMMD Project Scope

This document captures the initial research context and scope interpretation for SAMMD: a Python package that builds and exports chemistry, structure, and parameter artifacts for self-assembled monolayers on metal supports. OpenMM owns minimization, equilibration, production runs, trajectories, and reporters after those artifacts exist.

## Goal

SAMMD should let a user specify a metal surface, one or more thiol SAM components, solvent mixture, salts, and reactant molecules from a validated YAML file, then build/export OpenFF/OpenMM-ready chemistry, structure, and parameter artifacts that can be inspected and handed to user-controlled OpenMM Python scripts.

The first scientific target is a propanethiol SAM (`CCCS`) on Pd(111), with small molecule ligands approaching the monolayer-covered metal surface.

The default periodic geometry should use a centered, double-sided slab:

```text
z: Bulk solvent -> Bottom SAM -> Metal slab -> Top SAM -> Bulk solvent
```

The top and bottom bulk solvent regions are the same phase through periodic boundary conditions in `z`. The slab is centered in the box, SAM molecules decorate both exposed surfaces, and the metal slab must be thick enough that species on one SAM/slab interface do not interact with the other interface through the nonbonded cutoff.

## v0.1.0 First-Release Boundary

Ship a Python-first workflow before a broad CLI.

Recommended v0.1.0 first-release deliverables:

- Pixi-installable Python package with `src/` layout.
- Pydantic configuration models loaded from YAML.
- A minimal `sammd init` command that writes a commented template YAML file.
- A canonical Jupyter notebook showing configuration, deterministic build planning, export, and basic inspection.
- A registered Fcc(111) slab builder, defaulting to Pd(111), and propanethiol SAM placement with tunable grafting density.
- Mixed SAM support through multiple SAM component definitions with fractions or explicit counts.
- Centered, double-sided registered Fcc(111) slab geometry with SAMs on both faces and solvent on both sides.
- Configuration fields and validation that record the selected OpenFF small-molecule force field, charge model, water model, and INTERFACE metal resource choices. The default build remains lightweight; `sammd build --full` performs parameterized MD-file export in a CUDA-labeled pixi environment for supported non-salt configs.
- Static CHARMM-INTERFACE-derived Fcc metal Lennard-Jones parameters packaged or identified as OpenFF-compatible OFFXML resource support.
- Visualization-friendly build artifacts, centered on PDBx/mmCIF topology-inspection output plus machine-readable build summaries.
- Pytest coverage for configuration validation, parameter conversion, deterministic builders, and import smoke tests.
- ReadTheDocs/Sphinx scaffold with a canonical workflow tutorial and YAML configuration tutorial.

Defer until after v0.1.0:

- Full production CLI for build/run/submit workflows.
- SLURM or other HPC orchestration.
- Umbrella sampling automation.
- General surface support beyond registered Fcc(111) metals, including non-111 facets.
- Explicit metal-sulfur bonded topology terms unless needed to correct SAM geometry.
- OpenMM minimization, equilibration, production, DCD trajectory output, and thermodynamic reporting protocols. Tutorial-only user code may demonstrate OpenMM API usage after SAMMD exports backend artifacts, but those runs are not part of the SAMMD build/export contract.
- Salt ion backend export and broader noncanonical chemistry coverage beyond the first supported Interchange/OpenMM export path.
- Advanced analysis workflows beyond basic orientation observables.

## Reference Findings

### PolyzyMD

[PolyzyMD](https://github.com/joelaforet/polyzymd) is a useful architecture reference, not a template to copy wholesale.

Reusable ideas:

- Pixi environments for reproducible conda-forge installs.
- CUDA-labeled pixi environments so OpenMM matches the NVIDIA driver available
  on the machine.
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

The SAMMD backend pipeline uses OpenFF Toolkit molecule preparation and SMIRNOFF `ForceField` assembly, then OpenFF Interchange construction/export. The strengthened metal-S Lennard-Jones anchor proxy is encoded in a SAMMD Interchange plugin collection rather than exposed as a beginner-facing YAML option. The explicit `sammd build --full` path writes `solvated_system.cif`, `interchange.json`, and `anchor_metadata.json` for supported non-salt configs in a CUDA-labeled pixi environment.

OpenMM GPU support is tied to the NVIDIA driver and CUDA line available on the
machine. Students should run `nvidia-smi` on the GPU node or workstation, then
choose the SAMMD pixi environment whose CUDA version is not newer than the CUDA
version shown there. Current teaching environments are:

| Environment | CUDA line | OpenMM pin | Example location |
| --- | --- | --- | --- |
| `cuda-12-4` | 12.4 | `openmm=8.1.2` | CU Boulder Blanca older-GPU nodes |
| `cuda-12-6` | 12.6 | `openmm=8.4.0` | PSC Bridges2 |
| `cuda-13-0` | 13.0 | `openmm=8.5.1` | newer local NVIDIA drivers |

The `interchange.json` artifact is OpenFF Interchange JSON serialization with `Interchange.model_dump_json` and reload through `Interchange.model_validate_json` after registering SAMMD's plugin collection. SAMMD records the concrete `openff-interchange` package version when a real artifact is written, because pre-1.0 Interchange JSON compatibility is not guaranteed across versions.

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

SAMMD should own its registered Fcc(111) builder, defaulting to Pd(111), rather than depend on surface_coatings directly, because the MVP needs metal/facet-aware parameters, grafting-density controls, sulfur anchoring behavior, and OpenFF-ready topology handling.

### Solvent And Packing

[PolyzyMD solvent handling](https://github.com/joelaforet/polyzymd/blob/main/src/polyzymd/builders/solvent.py) provides a useful model for SAMMD solvent composition.

Reusable ideas:

- Compute molecule counts from box volume, solvent-only mole fractions, pure-component density/molar-volume metadata, and reactant concentrations.
- Require density for mole-fraction co-solvents.
- Add salts by concentration and optionally neutralize total system charge.
- Cache or reuse pre-parameterized solvent molecules so each copy has identical charges and parameters.
- Pack molecules with PACKMOL through the OpenFF ecosystem, rather than writing custom placement code.

SAMMD should keep this simpler than PolyzyMD initially: water, ethanol, simple monovalent salts, and reactants by concentration are enough for the canonical workflow.

### Sulfur Binding Sites

[Thermodynamics of Alkanethiol Self-Assembled Monolayer Assembly on Pd Surfaces](https://doi.org/10.1021/acs.langmuir.7b04351) is the most relevant Pd-specific reference identified so far. It establishes that Pd alkanethiol SAMs are a real target system and should be kept in the context bibliography.

Broader thiolate literature cautions that site preferences can be metal-, coverage-, and adsorbate-dependent. For example, Au(111) DFT/HREELS work reports methylthiolate near a bridge site displaced toward fcc hollow rather than a simple hollow-site picture ([Hayashi et al., J. Chem. Phys. 2001](https://doi.org/10.1063/1.1360245)).

For SAMMD MVP, use three-fold hollow placement on registered Fcc(111) surfaces as an internal modeling hypothesis that defaults to Pd(111). The builder should keep this assumption localized so future advanced APIs can switch among `fcc_hollow`, `hcp_hollow`, `bridge`, and `atop` placement after calibration against metal- and adsorbate-specific data.

## Proposed Package Shape

Initial package modules:

- `sammd.core.config`: Pydantic models, YAML loading, YAML template generation.
- `sammd.model.surfaces`: registered Fcc(111) slab builders with lattice constants and facet metadata, defaulting to Pd(111).
- `sammd.model.sam`: dependency-free SAM placement and sulfur anchor pose planning today; future backend work should add OpenFF/RDKit molecule creation, conformer generation, and molecule-graph sulfur anchor detection.
- `sammd.backends.forcefields`: INTERFACE metal parameter registry, offxml generation, OpenFF force field assembly.
- `sammd.model.solvation`: solvent mixture, salt, and reactant count calculations from solvent-only mole fractions and concentrations, followed by OpenFF/PACKMOL packing.
- `sammd.core.builders`: high-level lightweight build planner.
- `sammd.backends.interchange`: optional CUDA-environment OpenFF `Interchange` construction/export path for supported non-salt configs.
- `sammd.backends.openmm_runtime`: optional/internal OpenMM utilities for development and backend validation, not a student-facing SAMMD run-wrapper API.
- `sammd.core.io`: v0.1.0 PDBx/mmCIF topology-inspection writing; post-v0.1.0 DCD trajectory naming conventions and visualization-oriented metadata helpers.
- `sammd.runtime.reporting`: post-v0.1.0 OpenMM reporter configuration for trajectories and thermodynamic state data.
- `sammd.analysis`: orientation metrics and later umbrella-sampling support.
- `sammd.cli`: minimal `init` and maybe `validate` commands.

The current public Python API exposes a lightweight planning workflow that avoids heavy
OpenFF/OpenMM construction:

```python
from sammd import load_config, build_system

config = load_config("sammd.yaml")
plan = build_system(config, output_dir="outputs")
plan.write_topology_cif()  # Writes outputs/sam_grafting_density.cif by default
```

The returned plan contains validated configuration, a commensurate registered Fcc(111) slab,
binding sites, seeded SAM placement choices with sulfur anchor poses, approximate
solution composition counts, and planned output paths. It is not a final
OpenFF/OpenMM system and does not contain complete atomic coordinates for SAM,
solvent, salts, or reactants.

The current `sammd build` command writes the v0.1.0 first-release artifacts:

- `sam_grafting_density.cif`: lightweight visual smoke-test CIF for the deterministic plan, with the Pd slab and SAM sulfur atoms at planned three-fold hollow-site anchors.
- `build_summary.json`: machine-readable summary of the validated plan and output paths.
- `resolved_config.yaml`: validated YAML configuration used for the build.

The optional CUDA-environment backend export preserves the small public API surface while
adding artifact exports for OpenMM handoff:

```bash
pixi run -e cuda-12-4 sammd build sammd.yaml --output-dir outputs --overwrite --full
```

That command writes `interchange.json`, `solvated_system.cif`, and
`anchor_metadata.json` in addition to the lightweight artifacts. Salt-containing
configs are rejected until salt backend export is implemented.

## Scientific Design Decisions

### OpenMM Handoff Boundary

SAMMD builds and exports chemistry, structure, and parameter artifacts; OpenMM runs minimization, equilibration, production MD, trajectory writing, and reporter setup. Lightweight/internal OpenMM utilities may exist for development and backend validation, but they do not establish student-facing SAMMD ownership of `create_openmm_simulation`, minimization, equilibration, production MD, trajectory writing, or reporter setup in the top-level build workflow.

Future SAMMD releases should improve the build/export handoff and examples without making SAMMD-owned run wrappers the canonical API. Users should be able to inspect SAMMD artifacts and pass them to the OpenMM Python API for routine runs.

The handoff should:

- Accept validated configuration objects and produce build/export artifacts.
- Return or expose the underlying OpenFF `Interchange` and OpenMM `System`, `Topology`, and positions for advanced users.
- Keep minimization, equilibration, production, trajectory writing, and reporter setup in user-owned OpenMM scripts.
- Keep sulfur-metal anchoring details internal to planning/backend representation until an explicit advanced attachment API is designed.

For the nonbonded anchor proxy, beginner users should not tune the interaction magnitude in the current MVP. Future advanced APIs may expose attachment strategy and strength choices once the backend behavior is validated, but v0.1.0 should treat these as internal planning/representation details.

### Visualization And Output Files

SAMMD-generated structures and OpenMM-run outputs should be optimized for visualization in tools such as PyMOL.

Default post-v0.1.0 backend exports plus tutorial-only OpenMM run outputs should include:

- `solvated_system.cif`: PDBx/mmCIF topology and starting coordinates for the full slab + SAMs + reactants + solvent system.
- `trajectory.dcd`: DCD trajectory from OpenMM.
- `thermodynamics.csv`: tabular OpenMM state data from a thermodynamic reporter.
- Optional OpenMM restart/checkpoint artifacts for continuing simulations.

By default, `sammd build` writes `sam_grafting_density.cif`,
`build_summary.json`, and `resolved_config.yaml`. The default
`sam_grafting_density.cif` is a lightweight
visual smoke-test CIF for the deterministic plan, showing the Pd slab and SAM
sulfur atoms at planned three-fold hollow-site anchors. It is useful for
checking slab geometry and SAM grafting density before full backend export; it
does not include full SAM molecule, solvent, or reactant coordinates.
With `--full` in a CUDA-labeled pixi environment, SAMMD also writes
`solvated_system.cif`, `interchange.json`, and `anchor_metadata.json`.
The `interchange.json` artifact is JSON from `Interchange.model_dump_json` with
reload through `Interchange.model_validate_json` after registering SAMMD's
plugin collection; the concrete
`openff-interchange` package version is recorded when SAMMD writes the artifact,
and pre-1.0 JSON compatibility is not guaranteed across versions.

PDBx/mmCIF should be preferred over legacy PDB because SAMMD systems may have many atoms, many solvent/reactant molecules, nonstandard residues, and metal particles. SAMMD uses stable `.cif` artifact names for these PDBx/mmCIF files; `.mmcif` is also used elsewhere in the ecosystem. Atom names, residue names, chain IDs, molecule labels, and component metadata should be chosen so PyMOL sessions are easy to inspect: metal slab, top SAM, bottom SAM, solvent, salts, and reactants should be distinguishable by selection.

DCD should be the canonical post-v0.1.0/tutorial OpenMM trajectory convention, not a v0.1.0 build artifact. The topology/trajectory pair should load cleanly as `solvated_system.cif` plus `trajectory.dcd` in PyMOL or common Python analysis tools after user-owned OpenMM scripts run from SAMMD-exported artifacts.

### Thermodynamic Reporting

Post-v0.1.0 or tutorial-only OpenMM simulations should report key thermodynamic quantities while they run. Future user-facing configuration should allow users to choose output path, reporting interval, and reported fields.

Default report fields for normal runs should include at least:

- Step.
- Simulation time.
- Potential energy.
- Kinetic energy.
- Total energy.
- Temperature.
- Volume and density for periodic systems when available.
- Simulation speed and elapsed time when available.

For post-v0.1.0 reporting tests, reporter configuration should request every supported field so schema validation and reporter construction cover the full surface area. Users should be able to override this with a smaller field list for production runs.

### Mixed SAMs

Mixed SAMs are in scope, even if the canonical demo uses a single propanethiol component.

The configuration should allow a list of SAM components:

- `name`: human-readable component label.
- `smiles`: neutral thiol molecule SMILES.
- `fraction` or `count`: surface composition control.
- Optional per-component anchor settings for future calibration.

Beginner YAML configuration should teach SAM components as neutral thiols with an HS/implicit-H thiol sulfur, such as propanethiol `CCCS`. Users should not provide pre-deprotonated thiolate inputs; SAMMD uses the sulfur atom for placement and represents metal-sulfur attachment internally during planning/backend construction.

The total grafting density applies to all SAM components combined. The default grafting density is `0.25 nm^2 / molecule`, equivalent to 4 molecules per nm^2. Placement should select binding sites deterministically from a seed, then assign component identities from fractions or counts. The same composition should decorate both slab faces by default, with a future extension point for side-specific compositions.

### Slab Geometry And PBC

The MVP box should be periodic in `x`, `y`, and `z`, with the registered Fcc(111) slab centered in `z`. Pd(111) remains the default and canonical scientific target. The intended ordering is:

```text
Bulk solvent -> Bottom SAM -> Fcc(111) slab -> Top SAM -> Bulk solvent
```

The bottom SAM should point toward negative `z`; the top SAM should point toward positive `z`. Through PBC, the two bulk solvent regions are one continuous bulk phase.

The registered Fcc(111) slab should be restrained by default. We are not simulating SAM assembly or metal reconstruction; the surface mainly provides geometry and INTERFACE-derived nonbonded parameters. Use positional restraints to keep metal atoms near ideal lattice coordinates. The default positional restraint force constant is `10000 kJ mol^-1 nm^-2`, with the exact units and restraint form documented in the generated YAML template.

The slab must be thick enough that the two decorated interfaces do not see each other through the nonbonded cutoff. With a typical 0.9-1.2 nm cutoff and registered Fcc(111) layer spacing of roughly `a / sqrt(3)`, a safe starting point is 8-10 layers for the default Pd(111) target. This should be configurable, validated against the chosen cutoff, and documented with a warning if the requested slab is too thin.

For the first notebook/demo, a practical starting system is roughly 5 x 5 nm in the lateral plane, 8 layers of the default Pd(111) slab, SAMs on both faces, and at least 3 nm solvent padding from the fully extended SAM tips to the box boundary on each side before PBC wrapping. This is large enough for a full SAM coating and multiple cinnamaldehyde reactants (`C1=CC=C(C=C1)/C=C/C=O`) while still being plausible for prototype runs.

Default metal-sulfur anchoring should begin as an internal nonbonded proxy:

- Keep the SAM sulfur and surface metal atoms non-covalent in topology.
- Plan the relevant sulfur-metal interaction strength as backend/internal representation, not a beginner YAML knob.
- Use a selected post-export OpenMM pair-specific LJ override as the first-release strategy, not an OFFXML/Interchange-only strategy.
- Pair each planned SAM sulfur with the three nearest metal atoms at the registered Fcc(111) hollow site.
- Record the canonical first-release override as `mode = "nonbonded_lj_override"`, `site_kind = "fcc_hollow"`, `pairs_per_anchor = 3`, `sigma = 0.22 nm`, and `epsilon = 2.0 kcal/mol` (`8.368 kJ/mol`).
- Treat the override as strengthened nonbonded LJ attraction for neutral thiols, not covalent, quantum, or reactive chemisorption.
- Keep the base INTERFACE Fcc metal LJ parameters as the slab nonbonded model; the selected metal-S override is additional backend metadata for specific sulfur-metal pairs.
- Defer any user-configurable scale factor, such as 4x, 5x, or 6x, until a later advanced attachment API.
- Preserve a configuration/API path for future explicit bonded anchors.

Future explicit anchor mode should be designed as a replaceable strategy:

- `anchor.mode = "nonbonded_lj_override"` for MVP backend metadata.
- `anchor.mode = "bonded"` later for metal-sulfur bond, angle, and torsion parameters.
- A future angle target, such as 23 degrees relative to the surface, should not require rewriting the builder API.
- `anchor.site = "fcc_hollow"` should be the internal registered Fcc(111) hollow-placement strategy, with Pd(111) as the canonical/default surface and other site labels reserved for a later advanced attachment API.

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
- YAML template defaults include TIP3P, `0.25 nm^2 / molecule`, and `10000 kJ mol^-1 nm^-2` Pd restraints; the Pd(111) sulfur site remains an internal builder default rather than a beginner template field.
- YAML template includes v0.1.0 configurable output sections for PDBx/mmCIF `.cif` build artifacts; DCD and thermodynamic state data remain post-v0.1.0/tutorial-only simulation conventions.
- Invalid metal, facet, grafting density, solvent fraction, or concentration fails clearly.
- Mixed SAM fractions must sum to one, or explicit counts must match selected grafting sites.
- INTERFACE Fcc metal registry reproduces the table above.
- CHARMM epsilon sign conversion is tested.
- Generated offxml can be loaded by OpenFF `ForceField`.
- Registered Fcc(111) builder returns deterministic atom counts and box dimensions.
- Registered Fcc(111) slab thickness validation warns or fails when the slab is thinner than the configured nonbonded cutoff plus buffer.
- Propanethiol sulfur anchor detection is deterministic for `CCCS`.
- Cinnamaldehyde reactant parsing is deterministic for `C1=CC=C(C=C1)/C=C/C=O`.
- Post-v0.1.0 reporter configuration maps user field names to OpenMM reporter arguments.
- Post-v0.1.0 test reporter defaults request all supported thermodynamic fields.
- v0.1.0 PDBx/mmCIF `.cif` output paths are deterministic and can be overridden; DCD output paths are post-v0.1.0/tutorial-only simulation conventions.
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
- The default sulfur site for Pd(111) should be internal `fcc_hollow` builder behavior; user-configurable site type belongs in a future advanced attachment API.
- Sulfur-metal interaction metadata remains an internal planning/backend detail for v0.1.0; the default selected-pair strategy is a post-export OpenMM pair-specific LJ override, and any user-configurable scale factor belongs in a future advanced attachment API.
- TIP3P is the default water model.
- The default Pd positional restraint force constant is `10000 kJ mol^-1 nm^-2`.
- The default grafting density is `0.25 nm^2 / molecule`.
- Solvent and reactant placement should use OpenFF/PACKMOL-style packing, with counts derived from target composition.
- Reactant orientation analysis can begin with a configurable COM-to-selection vector relative to the surface normal.
- PDBx/mmCIF is the default topology/starting-coordinate output format, written with stable `.cif` artifact names.
- DCD is the default post-v0.1.0/tutorial-only OpenMM trajectory output convention, not a v0.1.0 build artifact.
- OpenMM thermodynamic state data should be written during post-v0.1.0/tutorial-only OpenMM runs, with future tests requesting all supported fields and users allowed to override fields/intervals.

## Remaining Open Questions

- None at the current scope-planning level. Backend implementation choices remain, but the user-facing behavior is now specified.

## Implementation Readiness

The scope is ready for v0.1.0 scaffolding. The first implementation pass should create the package skeleton, pixi/pyproject configuration, Pydantic YAML schema, template generation, INTERFACE metal registry, lightweight build/export artifacts, and tests for the resolved defaults before implementing expensive OpenFF/OpenMM build or simulation steps.

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
