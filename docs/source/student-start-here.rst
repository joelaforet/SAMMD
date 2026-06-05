Student start here
==================

This page is the recommended entry point for a new undergraduate user. It is
organized using the Diataxis documentation pattern: start with a tutorial, then
use how-to guidance for specific tasks, explanation pages for background, and
reference pages when you need exact API details.

What SAMMD does today
---------------------

SAMMD is currently a configuration-first package for planning self-assembled
monolayer molecular dynamics systems on metal supports. The stable public API can
load and validate YAML, build deterministic Pd(111)/SAM/solution plans, and write
a slab visualization file for inspection.

The repository also contains an OpenMM workflow prototype for a small
Pd(111)/propanethiol/cinnamaldehyde/ethanol system. That prototype is useful for
short smoke tests and teaching, but it still uses private backend modules and is
not yet the public production MD API.

There are two configuration paths:

* Use ``examples/student_config.yaml`` for the OpenMM notebook.
* Use a generated ``sammd.yaml`` for the lightweight public planning tutorial.

Recommended learning path
-------------------------

1. Install the pixi environments from the repository root:

   .. code-block:: bash

      pixi install
      pixi install -e science

2. If your goal is the OpenMM notebook, validate the student configuration:

   .. code-block:: bash

      pixi run sammd validate examples/student_config.yaml

3. If your goal is the lightweight public tutorial, generate and validate
   ``sammd.yaml``:

   .. code-block:: bash

      pixi run sammd init -o sammd.yaml
      pixi run sammd validate sammd.yaml

4. Follow the first tutorial:

   :doc:`tutorials/canonical-workflow`

5. Learn what each YAML section means:

   :doc:`tutorials/yaml-configuration`

6. Read the project scope before interpreting results as science:

   :doc:`explanation/project-scope`

Tutorials: learning by doing
----------------------------

Use tutorials when you are new and want a guided path with commands to run.

Start with :doc:`tutorials/canonical-workflow`. It shows the lightweight workflow
that is safe for new users:

* create ``sammd.yaml``
* validate the configuration
* build a deterministic plan from Python
* inspect the planned Pd(111) slab
* write ``planned_slab.cif`` for visualization

Then use :doc:`tutorials/yaml-configuration` to understand the editable parts of
the configuration file.

How-to: common student tasks
----------------------------

Use these task recipes when you know what you want to change.

Change the SAM molecule
~~~~~~~~~~~~~~~~~~~~~~~

Edit the ``sam.components`` section in ``sammd.yaml``. For the current milestone,
SAMMD expects thiol SAM components with exactly one sulfur anchor. Validate after
editing:

.. code-block:: bash

   pixi run sammd validate sammd.yaml

For the OpenMM notebook, make the same kind of edit in
``examples/student_config.yaml`` and validate with:

.. code-block:: bash

   pixi run sammd validate examples/student_config.yaml

Change the solvent or reactant concentration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit ``solvent.components`` for solvent mole fractions and ``reactants`` for
millimolar reactant concentrations. SAMMD converts those inputs into approximate
finite-system molecule counts during planning.

After editing, rebuild the plan:

.. code-block:: python

   from sammd import build_system, load_config

   config = load_config("sammd.yaml")
   plan = build_system(config, output_dir="outputs")
   print(plan.solution.molecule_counts)

For the OpenMM notebook, use ``examples/student_config.yaml`` instead of
``sammd.yaml``.

Understand examples/student_config.yaml
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The OpenMM notebook starts from ``examples/student_config.yaml``. The most common
student-editable fields are:

``surface.metal`` and ``surface.facet``
   The metal support. SAMMD v0.1.0 supports ``Pd`` and facet ``"111"``.

``surface.slab.lateral_size_nm``
   Requested x/y slab size in nanometers. SAMMD adjusts this to a commensurate
   Pd(111) lattice, so the printed plan size may differ slightly.

``sam.grafting_density``
   Area per SAM molecule. The supported unit is ``nm^2 / molecule``.

``sam.components[].name`` and ``sam.components[].smiles``
   The SAM molecule identity. Current OpenMM smoke runs expect one thiol/S anchor
   per SAM molecule.

``solvent.padding_nm``
   Approximate solvent height used for finite-box molecule count planning.

``solvent.components[].mole_fraction``
   Solvent-only mole fractions. These must sum to ``1.0`` over solvent
   components.

``reactants[].concentration_millimolar``
   Target reactant concentration. SAMMD converts this to a finite molecule count
   and enforces at least one molecule when the concentration is nonzero.

``simulation.timestep_fs`` and ``simulation.temperature_k``
   The OpenMM timestep and target temperature used by the notebook run.

``simulation.seed``
   Controls deterministic finite-system construction and velocity initialization.

Inspect the planned slab
~~~~~~~~~~~~~~~~~~~~~~~~

Build a plan and write the slab visualization file:

.. code-block:: python

   path = plan.write_planned_slab_mmcif(overwrite=True)
   print(path)

Open ``planned_slab.cif`` in a molecular viewer such as PyMOL. This file is a
visualization scaffold, not a complete simulation topology.

Run the instructor-facing OpenMM prototype
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your instructor has set up the optional science environment, the repository
contains a student notebook and a notebook-shaped script prototype:

.. code-block:: bash

   pixi run -e science install-science-kernel

Then open ``notebooks/student_openmm_workflow.ipynb`` with the
``Python (SAMMD science)`` kernel. The notebook starts from
``examples/student_config.yaml``.

The same workflow is also available as a script:

.. code-block:: bash

   pixi run -e science python examples/canonical_openmm_workflow.py --overwrite

This is useful for demonstration and smoke validation. Do not treat the output as
production MD or as evidence of scientific convergence.

Smoke-test checklist
~~~~~~~~~~~~~~~~~~~~

A successful notebook smoke run should:

* validate the YAML without errors
* build a SAMMD plan and print finite molecule counts
* build the prototype OpenMM system without missing-parameter errors
* minimize without NaNs
* complete the requested number of steps
* report a final temperature near the target temperature
* write ``topology.cif``, ``trajectory.dcd``, ``thermodynamics.csv``,
  ``system.xml``, and ``smoke_summary.json``

Passing this checklist means the workflow runs. It does not prove equilibration,
sampling quality, or scientific convergence.

Explanation: concepts worth understanding
-----------------------------------------

Use explanation pages when you need background rather than commands.

Read :doc:`explanation/project-scope` to understand what SAMMD currently supports
and what is intentionally out of scope for this milestone.

Important current limitations:

* Pd(111) is the supported MVP surface facet.
* The public API produces a build plan, not a complete production simulation.
* The slab visualization file does not include SAM, reactant, or solvent atoms.
* The OpenMM notebook is still a prototype because it imports private backend
  modules.

Reference: exact API details
----------------------------

Use reference pages when you need exact names, signatures, or model fields.

The main public entry points are documented in :doc:`reference/api`:

* ``sammd.load_config``
* ``sammd.build_system``
* ``sammd.SAMMDConfig``
* orientation analysis helpers in ``sammd.analysis``

What to ask before changing a science input
-------------------------------------------

Before changing a SAM, solvent, reactant, or run condition, write down:

* What experimental system am I trying to represent?
* Which input in ``examples/student_config.yaml`` or ``sammd.yaml`` maps to that
  experimental condition?
* Is this supported by the current SAMMD milestone?
* Am I looking at a planning artifact, a smoke-test artifact, or a production MD
  result?

If you are unsure, ask your instructor before interpreting the output.
