# SAMMD

SAMMD is a Python package for reproducible molecular dynamics setup of
self-assembled monolayers on metal supports. The first MVP milestone provides a
validated YAML configuration schema, a minimal CLI, lightweight INTERFACE metal
parameter metadata, and reporter configuration helpers without requiring
OpenMM/OpenFF build steps during unit tests.

The package also includes a generated CHARMM-INTERFACE Fcc metal OFFXML resource
for future OpenFF force field assembly without making OpenFF a test dependency.
Solution composition planning is available for converting solvent volume
fractions plus salt/reactant molarities into deterministic molecule counts while
remaining independent of PACKMOL and OpenFF during MVP scaffolding.

See [docs/project-scope.md](docs/project-scope.md) for the source-of-truth scope
and scientific defaults.

## Quick start

```bash
sammd init -o sammd.yaml
sammd validate sammd.yaml
```

```python
from sammd import load_config

config = load_config("sammd.yaml")
print(config.surface.metal, config.surface.facet)
```
