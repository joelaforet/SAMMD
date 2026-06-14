<h1 align="center">SAMMD</h1>

<p align="center">
  <a href="https://github.com/joelaforet/SAMMD/actions/workflows/ci.yml"><img src="https://github.com/joelaforet/SAMMD/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-%3E%3D3.12-blue.svg" alt="Python >=3.12"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

<p align="center">
  <strong>Build and export self-assembled monolayer systems for OpenMM simulations.</strong>
</p>

<p align="center">
  <a href="docs/source/index.rst">Documentation</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#pixi-environments">Pixi Environments</a>
</p>

---

## Overview

SAMMD is a Python package for reproducible molecular dynamics setup of
self-assembled monolayers on metal supports. The student-facing workflow uses a
version-controllable YAML file to describe the system contents, packing, and
parameterization choices before any OpenMM simulation protocol is written.

SAMMD handles:

- **Configuration**: YAML input with Pydantic validation.
- **Surface and SAM planning**: registered Fcc(111) metal slabs and thiol SAM
  placement.
- **Composition planning**: solvent, reactant, and salt counts from the YAML
  settings.
- **Interchange export**: optional OpenFF/OpenMM files for supported non-salt
  systems.
- **Inspection files**: `sam_grafting_density.cif`, `build_summary.json`, and
  `resolved_config.yaml` for the default dependency-light build.

SAMMD builds and exports chemistry, structure, and parameter artifacts. OpenMM
runs minimization, equilibration, production, trajectories, and reporters.

SAMMD builds a physically reasonable starting-structure plan with reproducible
force-field assignments for future MD simulations. The metal-S interaction is
modeled with a tunable, strengthened nonbonded interaction; it is not a quantum
or reactive description of chemisorption.

## Installation

SAMMD uses [pixi](https://pixi.sh) for environment management. Pixi installs
Python and conda-forge dependencies from `pixi.toml` and `pixi.lock`.

### 1. Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Restart your shell if the installer asks you to. Then clone SAMMD:

```bash
git clone https://github.com/joelaforet/SAMMD.git
cd SAMMD
```

### 2. Install the dependency-light environment

Use this environment for configuration, validation, tests, and docs. It does
not require OpenMM, OpenFF, RDKit, PACKMOL, or a GPU.

```bash
pixi install
pixi shell -e default
```

After `pixi shell -e default`, the `sammd` command is on `PATH`, so validation
commands work normally:

```bash
sammd init -o sammd.yaml
sammd validate sammd.yaml
```

Leave the active pixi shell with:

```bash
exit
```

### 3. Run one command without entering a shell

Pixi also supports one-shot commands. Prefix the command with `pixi run`:

```bash
pixi run sammd validate sammd.yaml
```

For named environments, add `-e <env>`:

```bash
pixi run -e cuda-12-4 sammd build sammd.yaml --output-dir outputs --overwrite
```

### 4. Switch environments

Pixi does not use `conda activate`. Use `pixi shell -e <env>` instead.

```bash
# Enter the default dependency-light environment
pixi shell -e default

# Leave it
exit

# Enter a CUDA export environment
pixi shell -e cuda-12-4

# Leave it before switching again
exit

# Enter a different CUDA export environment
pixi shell -e cuda-12-6
```

## Pixi Environments

| Environment | Use case | CUDA line | OpenMM pin |
| --- | --- | --- | --- |
| `default` | dependency-light config/validate/test workflow | none | none |
| `dev` | same as default, explicit dev environment | none | none |
| `docs` | Sphinx documentation build | none | none |
| `cuda-12-4` | OpenFF Interchange export and GPU OpenMM work | 12.4 | `openmm=8.1.2` |
| `cuda-12-6` | OpenFF Interchange export and GPU OpenMM work | 12.6 | `openmm=8.4.0` |
| `cuda-13-0` | OpenFF Interchange export and GPU OpenMM work | 13.0 | `openmm=8.5.1` |

OpenMM GPU support depends on the NVIDIA driver and CUDA version available on
your machine. On a GPU node or workstation, run:

```bash
nvidia-smi
```

Use an environment whose CUDA version is not newer than the CUDA version shown by
`nvidia-smi`.

Known examples:

| Machine or cluster | Environment |
| --- | --- |
| CU Boulder Blanca older-GPU nodes | `cuda-12-4` |
| PSC Bridges2 | `cuda-12-6` |
| Newer local NVIDIA drivers | `cuda-13-0`, if the driver supports CUDA 13.0 |

When unsure, choose the older compatible environment. For Interchange export examples in
this README, the default is `cuda-12-4`.

## Quick Start

### Option A: use pixi shells

```bash
pixi shell -e default
sammd init -o sammd.yaml
sammd validate sammd.yaml
exit
pixi shell -e cuda-12-4
sammd build sammd.yaml --output-dir outputs --overwrite
```

### Option B: use `pixi run`

```bash
pixi run sammd init -o sammd.yaml
pixi run sammd validate sammd.yaml
pixi run -e cuda-12-4 sammd build sammd.yaml --output-dir outputs --overwrite
```

The build command writes:

- `outputs/sam_grafting_density.cif`
- `outputs/build_summary.json`
- `outputs/resolved_config.yaml`
- `outputs/interchange.json`
- `outputs/solvated_system.cif`
- `outputs/solvated_system_pymol.pdb`
- `outputs/anchor_metadata.json`

Open `outputs/sam_grafting_density.cif` in a molecule viewer such as PyMOL to
inspect the configured surface and SAM anchor placements before moving on.

SAMMD writes PDBx/mmCIF structure files with the standard `.cif` extension.
The `.mmcif` extension is also common in the broader ecosystem; SAMMD keeps
stable `.cif` artifact names in its CLI and docs.

## OpenFF Interchange Export For OpenMM

`sammd build` writes parameterized OpenFF/OpenMM files. Use a CUDA-labeled
environment for builds. First choose the environment with `nvidia-smi`, then run
the build command.

Example for the default CUDA 12.4 export environment:

```bash
pixi run -e cuda-12-4 sammd build sammd.yaml --output-dir outputs --overwrite
```

Or enter the environment once:

```bash
pixi shell -e cuda-12-4
sammd build sammd.yaml --output-dir outputs --overwrite
```

The Interchange export writes these files:

- `interchange.json`: OpenFF Interchange JSON
- `solvated_system.cif`: PDBx/mmCIF topology and coordinates for the
  constructed system
- `anchor_metadata.json`: SAM anchor and sulfur-metal pair metadata

Salt-containing configs are rejected until salt Interchange export is implemented.

The Interchange JSON write/reload path uses `Interchange.model_dump_json` and
`Interchange.model_validate_json` after registering SAMMD's plugin collection;
pre-1.0 Interchange JSON compatibility is not guaranteed across OpenFF
Interchange versions.

## OpenMM Simulation

SAMMD prepares files. OpenMM runs minimization, equilibration, production,
trajectory writing, and thermodynamic reporters.

Start with:

- `docs/source/tutorials/openmm-simulation.rst`
- `notebooks/openmm_from_sammd.ipynb`

The OpenMM tutorial teaches the raw OpenMM Python API with
`LangevinMiddleIntegrator`, NVT equilibration, NVT production, `DCDReporter`,
`StateDataReporter`, pandas, matplotlib, and PyMOL `load_traj`.

## CLI Commands

| Command | Description |
| --- | --- |
| `sammd init -o sammd.yaml` | Write a starter YAML file |
| `sammd validate sammd.yaml` | Check the YAML before building |
| `sammd build sammd.yaml --output-dir outputs --overwrite` | Write inspection and OpenFF Interchange export files from a CUDA pixi environment |

Use `pixi run ...` outside a pixi shell. For shell-based workflows, use
`pixi shell -e default` for `init` and `validate`, then switch to a CUDA-labeled
environment for `build`.

CLI logs include a timestamp, full module path, level, and message. Add the root
option `--verbose` before the subcommand to show DEBUG logs:

```bash
sammd --verbose build sammd.yaml --output-dir outputs --overwrite
```

Colored ANSI log output is enabled automatically on terminals that support it.
Disable color with `sammd --no-color <command>` or `NO_COLOR=1 sammd <command>`.
Root options such as `--verbose` and `--no-color` must appear before the
subcommand.

## Documentation And Notebooks

- Install and pixi tutorial: `docs/source/tutorials/installation.rst`
- Build workflow tutorial: `docs/source/tutorials/canonical-workflow.rst`
- YAML tutorial: `docs/source/tutorials/yaml-configuration.rst`
- OpenMM tutorial: `docs/source/tutorials/openmm-simulation.rst`
- Build workflow notebook: `notebooks/building_systems_with_sammd.ipynb`
- OpenMM notebook: `notebooks/openmm_from_sammd.ipynb`
- Project scope and scientific defaults: `docs/project-scope.md`

## Developer Checks

Use the dependency-light development environment for routine checks:

```bash
pixi run lint
pixi run pytest -q
pixi run -e docs python -m sphinx -W -b html docs/source docs/build/html
```

To register the selected CUDA environment as a Jupyter kernel:

```bash
pixi run -e cuda-12-4 install-cuda-kernel
```

## License

MIT License. See `LICENSE` for details.
