# SAMMD

[![CI](https://github.com/joelaforet/SAMMD/actions/workflows/ci.yml/badge.svg)](https://github.com/joelaforet/SAMMD/actions/workflows/ci.yml)

SAMMD is a Python package for reproducible molecular dynamics setup of
self-assembled monolayers on metal supports. The student-facing workflow uses a
version-controllable YAML file to describe the system contents, packing, and
parameterization choices before any OpenMM simulation protocol is written.

SAMMD builds a physically reasonable starting structure with reproducible force-field assignments for running MD simulations. The metal-S interaction is modeled with a tunable, strengthened nonbonded interaction; it is not a quantum or reactive description of chemisorption.

The package includes a generated CHARMM-INTERFACE Fcc metal OFFXML resource for
OpenFF force-field assembly without making OpenFF a test dependency. Solution
composition planning converts solvent mole fractions, explicit salt
stoichiometry, and reactant counts or concentrations into deterministic molecule
counts.

Build output planning resolves deterministic current and reserved artifact paths.
The build command writes an inspectable `topology.cif` for molecule-viewer checks;
SAM sulfur anchor placeholders are shown at planned sulfur positions above or
below the selected surface sites. Future backend work reserves `interchange.json`
as the primary portable export, with `system.xml` only as an OpenMM convenience
export.

Lightweight orientation analysis primitives are available in `sammd.analysis` for
future trajectory observables. They calculate reactant centers of mass, target
atoms or midpoint/centroid selections, COM-to-target orientation vectors, and
angles relative to top (+z), bottom (-z), or explicit surface normals without
requiring MDAnalysis, OpenMM, OpenFF, or RDKit.

Optional OpenFF adapter helpers live in `sammd.openff` for future backend
parameterization work. They lazily create OpenFF molecules from configured SAM
components, SMILES-bearing solvents, reactants, and separate salt ions, and can
record or load base OpenFF force fields together with the packaged INTERFACE Fcc
metal OFFXML resource. The planned backend route is OpenFF Toolkit molecule and
ForceField preparation, OpenFF Interchange construction/export, then a selected
post-export metal-S Lennard-Jones override in OpenMM representation.
Undefined stereochemistry follows the safer OpenFF default unless explicitly
allowed. These helpers require the SAMMD science/pixi environment and do not
perform full system construction.

Optional OpenMM runtime helpers live in `sammd.openmm_runtime`. They lazily
create Langevin integrators, create/configure a new OpenMM `Simulation` from
existing topology, system, positions, and reporter settings, and include an
experimental sulfur-metal LJ scaling helper for explicit pair lists.
Users must still supply existing OpenMM topology, system, and positions from
future construction code or their own backend workflow; SAMMD does not yet
construct complete OpenMM systems.

A development smoke runner is available at `tools/openmm_smoke.py` for testing
the science environment against a compact real Pd(111)/propanethiol SAM input
with cinnamaldehyde and ethanol. It is not part of the public package API and
uses pragmatic direct OpenMM construction for stability testing while full
backend construction remains a future milestone. Ethanol placement is generated
by PACKMOL while holding the prebuilt Pd+SAM+cinnamaldehyde coordinates fixed for
packing only; the OpenMM smoke run uses mobile, unrestrained Pd atoms. Topology
output uses a PolyzyMD-style repeat-unit residue convention: chain A is the Pd
slab, chain B contains one propanethiol-derived thiol SAM residue per SAM
molecule, chain C contains cinnamaldehyde, and chain D+ contains one ethanol
residue per molecule with wrapping every 9999 residues.

`build_system()` currently returns a deterministic build plan. The plan contains
the validated config, an automatically thickened centered registered Fcc(111)
slab defaulting to Pd(111), internal fcc hollow thiol binding sites, seeded
top/bottom SAM placement choices with dependency-free sulfur anchor poses,
solution molecule counts from an approximate composition-planning volume, and
build output paths. The YAML intentionally does not define OpenMM simulation
phases, thermostats, barostats, trajectory writing, or production protocols.
The first-release metal-S strategy is recorded as dependency-free metadata: each
neutral thiol sulfur is paired with the three nearest hollow-site metal atoms for
a future post-export OpenMM pair-specific LJ override (`sigma = 0.22 nm`,
`epsilon = 2.0 kcal/mol`). The base INTERFACE metal parameters remain the slab
nonbonded model; the selected metal-S override is an internal strengthened
nonbonded attraction, not covalent or reactive chemisorption and not a beginner
YAML knob.

See [docs/project-scope.md](docs/project-scope.md) for the source-of-truth scope
and scientific defaults.

The documentation scaffold lives in [docs/source](docs/source/index.rst) for
future ReadTheDocs builds. The interactive MVP walkthrough is available at
[notebooks/canonical_workflow.ipynb](notebooks/canonical_workflow.ipynb).

## Quick start

```bash
sammd init -o sammd.yaml
sammd validate sammd.yaml
sammd build sammd.yaml --output-dir outputs --overwrite
```

The build command writes `outputs/topology.cif`, `outputs/build_summary.json`, and
`outputs/resolved_config.yaml`. Open `outputs/topology.cif` in a molecule viewer
to inspect the configured surface and SAM anchor placements before moving on.
Full SAM molecule coordinates remain future backend work.

```python
from sammd import build_system, load_config

config = load_config("sammd.yaml")
plan = build_system(config, output_dir="outputs")

print(plan.slab.metal, plan.slab.facet)
print(plan.solution.molecule_counts)
plan.write_topology_cif()  # Writes outputs/topology.cif by default
```

The config also names future backend artifacts: `interchange.json` as the primary
portable export, `positions.cif` as the companion coordinate structure file, and
`system.xml` as an OpenMM convenience export. The reserved Interchange plan is a
future JSON write/reload path using `Interchange.model_dump_json` and
`Interchange.model_validate_json`; SAMMD does not write that artifact in the
current release, and pre-1.0 Interchange JSON compatibility is not guaranteed
across OpenFF Interchange versions.

## Developer checks

Use the lightweight development environment for routine checks; optional
OpenFF/OpenMM science tests skip unless those packages are available.

```bash
python -m pytest --cov=sammd --cov-report=term-missing
ruff check src/sammd tests
```

With pixi, the same checks are available as:

```bash
pixi run test
pixi run lint
```

The optional science environment includes OpenMM, OpenFF, RDKit, mBuild,
PACKMOL, and `ipykernel`. To run the real-system smoke test from that
environment:

```bash
pixi install -e science
pixi run -e science real-system-smoke
```

To register the same environment as a Jupyter kernel:

```bash
pixi run -e science install-science-kernel
```

The docs/notebook smoke coverage is included in pytest. To force a Sphinx docs
build locally, install the docs extras and run:

```bash
python -m pip install -e ".[dev,docs]"
python -m sphinx -W -b html docs/source docs/_build/html
```
