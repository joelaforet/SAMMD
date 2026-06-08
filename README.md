# SAMMD

[![CI](https://github.com/joelaforet/SAMMD/actions/workflows/ci.yml/badge.svg)](https://github.com/joelaforet/SAMMD/actions/workflows/ci.yml)

SAMMD is a Python package for reproducible molecular dynamics setup of
self-assembled monolayers on metal supports. The student-facing workflow uses a
version-controllable YAML file to describe the system contents, packing, and
parameterization choices before any OpenMM simulation protocol is written.
SAMMD builds and exports chemistry, structure, and parameter artifacts; OpenMM
runs minimization, equilibration, production, trajectories, and reporters.

SAMMD builds a physically reasonable starting-structure plan with reproducible force-field assignments for future MD simulations. The metal-S interaction is modeled with a tunable, strengthened nonbonded interaction; it is not a quantum or reactive description of chemisorption.

The package includes a generated CHARMM-INTERFACE Fcc metal OFFXML resource for
OpenFF force-field assembly without making OpenFF a test dependency. Solution
composition planning converts solvent mole fractions, explicit salt
stoichiometry, and reactant counts or concentrations into deterministic molecule
counts.

Build output planning resolves deterministic artifact paths. By default, the
build command writes an inspectable `topology.cif` for molecule-viewer checks;
SAM sulfur anchor placeholders are shown at planned sulfur positions above or
below the selected surface sites. In a CUDA-labeled pixi environment,
`sammd build --export-backend` writes `interchange.json` as the primary portable
OpenFF Interchange export, with `system.xml` only as an OpenMM convenience export.

Lightweight orientation analysis primitives are available in `sammd.analysis` for
future trajectory observables. They calculate reactant centers of mass, target
atoms or midpoint/centroid selections, COM-to-target orientation vectors, and
angles relative to top (+z), bottom (-z), or explicit surface normals without
requiring MDAnalysis, OpenMM, OpenFF, or RDKit.

Optional OpenFF adapter helpers live in `sammd.backends.openff` for future backend
parameterization work. They lazily create OpenFF molecules from configured SAM
components, SMILES-bearing solvents, reactants, and separate salt ions, and can
record or load base OpenFF force fields together with the packaged INTERFACE Fcc
metal OFFXML resource. The planned backend route is OpenFF Toolkit molecule and
ForceField preparation, OpenFF Interchange construction/export, then a selected
post-export metal-S Lennard-Jones override in OpenMM representation.
Undefined stereochemistry follows the safer OpenFF default unless explicitly
allowed. These helpers require a SAMMD CUDA pixi environment with OpenFF
available. Salt ions are
planned in the YAML schema but are not yet supported by `--export-backend`; salt
configs fail loudly rather than producing partial backend artifacts.

Internal optional OpenMM runtime utilities exist for development and backend
integration. They can lazily create Langevin integrators, configure a
caller-supplied OpenMM simulation context from existing topology, system,
positions, and reporter settings, and include an experimental sulfur-metal LJ
scaling helper for explicit pair lists. These helpers are not a current
student-facing SAMMD simulation wrapper and are not used by the quick start.
Users must still run dynamics through OpenMM itself; SAMMD does not own
minimization, production, trajectory writing, or reporter setup.

A development smoke runner is available at `tools/openmm_smoke.py` for testing
a CUDA pixi environment against a compact real Pd(111)/propanethiol SAM input
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
[notebooks/building_systems_with_sammd.ipynb](notebooks/building_systems_with_sammd.ipynb).

## Quick start

```bash
sammd init -o sammd.yaml
sammd validate sammd.yaml
sammd build sammd.yaml --output-dir outputs --overwrite
```

The default build command writes `outputs/topology.cif`,
`outputs/build_summary.json`, and `outputs/resolved_config.yaml`. Open
`outputs/topology.cif` in a molecule viewer to inspect the configured surface and
SAM anchor placements before moving on.

To write parameterized backend artifacts, use a CUDA-labeled pixi environment.
On a GPU node or workstation, start by checking the NVIDIA driver:

```bash
nvidia-smi
```

Choose an environment whose CUDA version is not newer than the CUDA version shown
by `nvidia-smi`:

| Environment | CUDA line | OpenMM pin | Example location |
| --- | --- | --- | --- |
| `cuda-12-4` | 12.4 | `openmm=8.1.2` | CU Boulder Blanca older-GPU nodes |
| `cuda-12-6` | 12.6 | `openmm=8.4.0` | PSC Bridges2 |
| `cuda-13-0` | 13.0 | `openmm=8.5.1` | newer local NVIDIA drivers |

For example, on Bridges2:

```bash
pixi run -e cuda-12-6 sammd build sammd.yaml --output-dir outputs --overwrite --export-backend
```

That backend mode writes `interchange.json`, `positions.cif`, `system.xml`, and
`anchor_metadata.json` in addition to the default artifacts. `interchange.json`
is the primary portable artifact; `system.xml` is an OpenMM convenience export
after SAMMD applies the selected metal-S pair overrides to the OpenMM system.

```python
from sammd import build_system, load_config

config = load_config("sammd.yaml")
plan = build_system(config, output_dir="outputs")

print(plan.slab.metal, plan.slab.facet)
print(plan.solution.molecule_counts)
plan.write_topology_cif()  # Writes outputs/topology.cif by default
```

The Interchange JSON write/reload path uses `Interchange.model_dump_json` and
`Interchange.model_validate_json`; pre-1.0 Interchange JSON compatibility is not
guaranteed across OpenFF Interchange versions.

## Developer checks

Use the lightweight development environment for routine checks; optional
OpenFF/OpenMM CUDA tests skip unless those packages are available.

```bash
python -m pytest --cov=sammd --cov-report=term-missing
ruff check src/sammd tests
```

With pixi, the same checks are available as:

```bash
pixi run test
pixi run lint
```

The CUDA pixi environments include OpenMM, OpenFF, RDKit, mBuild, PACKMOL, and
`ipykernel`. To run the real-system smoke test from the Bridges2-style
environment:

```bash
pixi install -e cuda-12-6
pixi run -e cuda-12-6 real-system-smoke
```

To register the selected environment as a Jupyter kernel:

```bash
pixi run -e cuda-12-6 install-cuda-kernel
```

The docs/notebook smoke coverage is included in pytest. To force a Sphinx docs
build locally, install the docs extras and run:

```bash
python -m pip install -e ".[dev,docs]"
python -m sphinx -W -b html docs/source docs/_build/html
```
