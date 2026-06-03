# SAMMD

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

`build_system()` currently returns a lightweight plan rather than OpenFF/OpenMM
objects. The plan contains the validated config, a commensurate Pd(111) slab,
fcc binding sites, seeded top/bottom SAM placement choices, solution molecule
counts, and output paths. Full backend construction is the next milestone.

See [docs/project-scope.md](docs/project-scope.md) for the source-of-truth scope
and scientific defaults.

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
plan.write_planned_slab_mmcif()
```
