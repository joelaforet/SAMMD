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

Lightweight orientation analysis primitives are available in `sammd.analysis` for
future trajectory observables. They calculate reactant centers of mass, target
atoms or midpoint/centroid selections, COM-to-target orientation vectors, and
angles relative to top (+z), bottom (-z), or explicit surface normals without
requiring MDAnalysis, OpenMM, OpenFF, or RDKit.

Optional OpenFF adapter helpers live in `sammd.openff` for future backend
parameterization work. They lazily create OpenFF molecules from configured SAM
components, simple SMILES-bearing solvents, and reactants, and can load base
OpenFF force fields together with the packaged INTERFACE Fcc metal OFFXML
resource. These helpers require the SAMMD science/pixi environment and do not
perform full system construction.

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
