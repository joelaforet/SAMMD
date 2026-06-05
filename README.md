# SAMMD

[![CI](https://github.com/joelaforet/SAMMD/actions/workflows/ci.yml/badge.svg)](https://github.com/joelaforet/SAMMD/actions/workflows/ci.yml)

SAMMD is a Python package for reproducible molecular dynamics setup of
self-assembled monolayers on metal supports. The MVP workflow now provides a
validated YAML configuration schema, a minimal CLI, lightweight INTERFACE metal
parameter metadata, reporter configuration helpers, and deterministic build-plan
composition without requiring OpenMM/OpenFF build steps during unit tests.

The package also includes a generated CHARMM-INTERFACE Fcc metal OFFXML resource
for future OpenFF force field assembly without making OpenFF a test dependency.
Solution composition planning is available for converting solvent volume
fractions plus salt/reactant molarities into deterministic molecule counts while
remaining independent of PACKMOL and OpenFF during MVP scaffolding.

Visualization output planning now resolves deterministic mmCIF/PDBx topology,
DCD trajectory, and OpenMM thermodynamics CSV artifact paths. A lightweight
mmCIF writer can emit planned scaffold atoms for PyMOL inspection, while full
trajectory production still awaits OpenMM builder integration.

Lightweight orientation analysis primitives are available in `sammd.analysis` for
future trajectory observables. They calculate reactant centers of mass, target
atoms or midpoint/centroid selections, COM-to-target orientation vectors, and
angles relative to top (+z), bottom (-z), or explicit surface normals without
requiring MDAnalysis, OpenMM, OpenFF, or RDKit.

Optional OpenFF adapter helpers live in `sammd.openff` for future backend
parameterization work. They lazily create OpenFF molecules from configured SAM
components, simple SMILES-bearing solvents, and reactants, report unsupported
solvent and salt entries that cannot be converted, and can load base OpenFF
force fields together with the packaged INTERFACE Fcc metal OFFXML resource.
Undefined stereochemistry follows the safer OpenFF default unless explicitly
allowed. These helpers require the SAMMD science/pixi environment and do not
perform full system construction.

Optional OpenMM runtime helpers live in `sammd.openmm_runtime`. They lazily
create Langevin integrators, create/configure a new OpenMM `Simulation` from
existing topology, system, positions, and reporter settings, and apply a
pair-specific sulfur-metal LJ scaling proxy.
Users must still supply existing OpenMM topology, system, and positions from
future construction code or their own backend workflow; SAMMD does not yet
construct complete OpenMM systems.

A development smoke runner is available at `tools/openmm_smoke.py` for testing
the science environment against a compact real Pd(111)/propanethiolate/
cinnamaldehyde/ethanol OpenMM system. It is not part of the public package API and
uses pragmatic direct OpenMM construction for stability testing while full
backend construction remains a future milestone. Ethanol placement is generated
by PACKMOL while holding the prebuilt Pd+SAM+cinnamaldehyde coordinates fixed for
packing only; the OpenMM smoke run uses mobile, unrestrained Pd atoms. Topology
output uses a PolyzyMD-style repeat-unit residue convention: chain A is the Pd
slab, chain B contains one propanethiolate residue per SAM molecule, chain C
contains cinnamaldehyde, and chain D+ contains one ethanol residue per molecule
with wrapping every 9999 residues.

The smoke runner's default `--seed 2026` is the canonical validation contract.
Changing `--seed` changes finite-system construction, including SAM/site placement,
and the OpenMM stochastic velocity initialization. Alternate seeds are therefore
distinct systems for seed-sensitivity exploration and should not be used to replace
the canonical smoke validation result.

`build_system()` currently returns a lightweight plan rather than OpenFF/OpenMM
objects. The plan contains the validated config, a centered double-sided
commensurate Pd(111) slab, fcc/hcp hollow binding sites, seeded top/bottom SAM
placement choices, solution molecule counts from an approximate composition
planning volume, and output paths. Bridge/atop sites and one-sided or off-center
slabs remain future build-planner work. Full backend construction is the next
milestone.

See [docs/project-scope.md](docs/project-scope.md) for the source-of-truth scope
and scientific defaults.

The documentation scaffold lives in [docs/source](docs/source/index.rst) for
future ReadTheDocs builds. The interactive MVP walkthrough is available at
[notebooks/canonical_workflow.ipynb](notebooks/canonical_workflow.ipynb).

## Quick start

```bash
sammd init -o sammd.yaml
sammd validate sammd.yaml
```

```python
from sammd import build_system, load_config

config = load_config("sammd.yaml")
plan = build_system(config, output_dir="outputs")

print(plan.slab.metal, plan.slab.facet)
print(plan.solution.molecule_counts)
plan.write_planned_slab_mmcif()  # Writes outputs/planned_slab.cif by default
```

The emitted mmCIF is a slab-only visualization scaffold, not a complete system
topology or final simulation cell. The configured `topology.cif` path remains
reserved for the future full topology artifact.

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
