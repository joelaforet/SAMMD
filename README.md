# SAMMD

SAMMD is a Python package for reproducible molecular dynamics setup of
self-assembled monolayers on metal supports. The first MVP milestone provides a
validated YAML configuration schema, a minimal CLI, lightweight INTERFACE metal
parameter metadata, and reporter configuration helpers without requiring
OpenMM/OpenFF build steps during unit tests.

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
