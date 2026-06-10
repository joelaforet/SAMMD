First-release build contract
============================

SAMMD's first release is a configuration-first builder/exporter contract. A user
edits a YAML file, validates it, and runs a CLI build that writes deterministic
chemistry, structure, and parameter-planning artifacts. SAMMD builds/exports
artifacts; OpenMM owns minimization, equilibration, production runs,
trajectories, and reporters. Downstream OpenMM simulation scripts are taught
separately and should use the OpenMM Python API directly.

CLI contract
------------

The supported first-release command-line surface is:

.. list-table::
   :header-rows: 1

   * - Command
     - Contract
   * - ``sammd init [-o PATH] [--force]``
     - Write a commented starter YAML configuration. Existing files are protected
       unless ``--force`` is supplied.
   * - ``sammd validate CONFIG``
     - Load and validate a YAML configuration without writing build artifacts.
   * - ``sammd build CONFIG --output-dir DIR --overwrite``
     - Build the current deterministic plan and write the currently implemented
       artifacts into ``DIR``. Existing artifacts are protected unless
       ``--overwrite`` is supplied.
   * - ``sammd build CONFIG --output-dir DIR --overwrite --full``
     - In a CUDA-labeled pixi environment, write OpenFF Interchange and OpenMM
       export artifacts in addition to the default artifacts. Salt-containing
       configs are rejected until salt export is implemented.

The ``build`` command does not run minimization, equilibration, production MD,
trajectory writing, or reporter setup.

Python API contract
-------------------

The supported first-release Python surface is:

.. list-table::
   :header-rows: 1

   * - Name
     - Contract
   * - ``load_config(path)``
     - Load and validate a YAML file into ``SAMMDConfig``.
   * - ``load_config_dict(data)``
     - Validate an already parsed mapping into ``SAMMDConfig``.
   * - ``build_system(config, output_dir=None, seed=None)``
     - Return a dependency-light ``SAMMDBuildPlan`` from a ``SAMMDConfig``, YAML path,
       or parsed mapping.

The object returned by ``build_system`` is documented as ``SAMMDBuildPlan``. It
exposes deterministic slab, SAM placement, solution composition, output paths,
``build_summary()``, and artifact writers for the current plan, but it is not a
top-level public import in ``sammd.__all__``. ``SAMMDBuildPlan`` is not an OpenMM
``System``, an OpenFF ``Interchange``, or a simulation wrapper. Code that needs
full Interchange construction should use the explicit CLI ``--full`` export path or the
internal Interchange module from a CUDA-labeled pixi environment.

The ``build_summary()`` SAM section records the first-release metal-S interaction
strategy as dependency-free metadata. The canonical mode is
``nonbonded_lj_override``: for each neutral thiol anchor, the export should use
the three nearest registered Fcc(111) hollow-site metal atoms as selected pairs
for a post-export OpenMM pair-specific LJ override with ``sigma = 0.22 nm`` and
``epsilon = 2.0 kcal/mol``. This is a strengthened nonbonded attraction layered
on top of the base INTERFACE metal LJ model, not covalent, quantum, or reactive
chemisorption. The first release records the strategy and selected pair indices
in dependency-light mode. The explicit Interchange export applies those selected pairs as
post-Interchange OpenMM ``NonbondedForce`` exceptions and records them in
``anchor_metadata.json``.

Validation gates
----------------

The internal ``sammd.core.validation`` module provides dependency-free gates for the
current build plan and topology CIF text. These gates check surface
atom metadata lengths, non-empty top and bottom binding-site labels, SAM counts,
solution-volume/box-volume agreement, finite positive box dimensions/bounds and
volume consistency, slab/box lateral-size agreement, SAM anchor metadata,
metal-S pair counts and slab-local indices, canonical metal-S strategy metadata,
current/reserved output suffixes, and inspection topology CIF atom counts and
cell lengths.

These gates intentionally do not require OpenMM, OpenFF, or full Interchange export
artifacts. Missing export artifacts such as ``solvated_system.cif``,
``interchange.json`` and ``anchor_metadata.json`` are not failures unless
``--full`` is requested.

Interchange export validation gates should stay skipped/not required when optional
dependencies or export artifacts are absent. Once ``--full`` writes
concrete artifacts, those gates should check that:

* ``interchange.json`` reloads with ``Interchange.model_validate_json``.
* The reloaded ``Interchange`` exports to an OpenMM ``System``.
* Topology atom count, positions atom count, and OpenMM ``System`` particle
  count agree.
* Minimization produces finite energies and the final energy is not increased.

Artifact contract
-----------------

The output names are stable so user scripts and documentation can refer to one
set of paths. Interchange exports treat ``interchange.json`` as the primary portable
system artifact.

SAMMD writes PDBx/mmCIF structure artifacts using the standard ``.cif``
extension. The ``.mmcif`` extension is also used elsewhere in the ecosystem, but
SAMMD keeps stable ``.cif`` names for the artifacts below.

The build summary also records engine export planning metadata. OpenMM is the
student teaching path through the OpenMM Python API, while OpenFF Interchange
remains the primary handoff. GROMACS, LAMMPS, Amber, and OpenMM XML are reserved
only as future downstream exports from Interchange and are not taught in the
beginner workflow.

.. list-table::
   :header-rows: 1

   * - Artifact
     - Status
     - Contract
   * - ``sam_grafting_density.cif``
     - Current
     - Inspection visual smoke-test PDBx/mmCIF ``.cif`` file for the
       deterministic plan, including the Pd slab and SAM sulfur atoms at planned
       three-fold hollow-site anchor positions. In default builds, this file is
       meant for checking slab geometry and grafting density; it does not include
       full SAM, solvent, or reactant coordinates and is not a parameterized
       Interchange export system. Interchange export leaves this smoke-test file
       separate from the full solvated-system structure.
   * - ``build_summary.json``
     - Current
     - Machine-readable summary of the validated plan, output paths, and
       Interchange-ready metal-S LJ override metadata.
   * - ``resolved_config.yaml``
     - Current
     - Validated YAML configuration used for the build.
   * - ``solvated_system.cif``
     - Interchange Export
     - Written by ``--full`` for fully constructed SAM, solvent, and
       reactant coordinates. This is a human-inspectable/OpenMM-loadable
       PDBx/mmCIF ``.cif`` structure file paired with the Interchange export
       artifact.
   * - ``interchange.json``
     - Interchange Export
     - Written by ``--full`` as the primary portable OpenFF
       Interchange export. The JSON path is ``Interchange.model_dump_json`` for
       saving and ``Interchange.model_validate_json`` for reload. SAMMD records
       the concrete ``openff-interchange`` package version when the artifact is
       written and treats pre-1.0 Interchange JSON compatibility as not
       guaranteed across versions.
   * - ``anchor_metadata.json``
     - Interchange Export
     - Written by ``--full`` for selected sulfur-metal pair metadata.

Current limitation
------------------

By default, ``sammd build`` writes only ``sam_grafting_density.cif``,
``build_summary.json``, and ``resolved_config.yaml``. With ``--full``
in a CUDA-labeled pixi environment, it also writes ``solvated_system.cif``,
``interchange.json``, and ``anchor_metadata.json``. Public SAMMD APIs should not
add equilibration, production simulation helpers, or direct GROMACS/LAMMPS/Amber
command workflows as part of this contract.
