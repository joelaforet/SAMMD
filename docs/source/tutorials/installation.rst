Installation and pixi basics
============================

This page gets SAMMD installed and explains the pixi commands used in the other
tutorials.

SAMMD uses `pixi <https://pixi.sh>`_ instead of conda or mamba environment YAML
files. Pixi reads ``pixi.toml`` and ``pixi.lock`` from this repository and
creates the requested environment from conda-forge packages.

1. Install pixi
---------------

Install pixi once on your machine:

.. code-block:: bash

   curl -fsSL https://pixi.sh/install.sh | sh

Restart your shell if the installer asks you to.

2. Clone SAMMD
--------------

.. code-block:: bash

   git clone https://github.com/joelaforet/SAMMD.git
   cd SAMMD

3. Use the lightweight environment
----------------------------------

The default environment is for YAML files, validation, lightweight builds,
testing, and docs. It does not need OpenMM, OpenFF, RDKit, PACKMOL, or a GPU.

.. code-block:: bash

   pixi install
   pixi shell -e default

After ``pixi shell -e default``, the ``sammd`` command is available directly:

.. code-block:: bash

   sammd init -o sammd.yaml
   sammd validate sammd.yaml
   sammd build sammd.yaml --output-dir outputs --overwrite

Leave the pixi shell with:

.. code-block:: bash

   exit

4. Run one command without entering a shell
-------------------------------------------

If you are not inside a pixi shell, prefix commands with ``pixi run``:

.. code-block:: bash

   pixi run sammd validate sammd.yaml

For a named environment, add ``-e``:

.. code-block:: bash

   pixi run -e cuda-12-4 sammd build sammd.yaml --output-dir outputs --overwrite --export-backend

5. Switch environments
----------------------

Pixi does not use ``conda activate``. Use ``pixi shell -e ENV_NAME`` instead.

.. code-block:: bash

   pixi shell -e default
   exit

   pixi shell -e cuda-12-4
   exit

   pixi shell -e cuda-12-6

6. Choose a CUDA environment
----------------------------

OpenMM GPU support depends on the NVIDIA driver and CUDA version available on
the machine. On a GPU node or workstation, run:

.. code-block:: bash

   nvidia-smi

Use an environment whose CUDA version is not newer than the CUDA version shown by
``nvidia-smi``.

.. list-table:: SAMMD pixi environments
   :header-rows: 1

   * - Environment
     - Use case
     - CUDA line
     - OpenMM pin
   * - ``default``
     - lightweight config, validation, builds, tests
     - none
     - none
   * - ``docs``
     - Sphinx documentation builds
     - none
     - none
   * - ``cuda-12-4``
     - OpenFF/OpenMM backend export and GPU OpenMM work
     - 12.4
     - ``openmm=8.1.2``
   * - ``cuda-12-6``
     - OpenFF/OpenMM backend export and GPU OpenMM work
     - 12.6
     - ``openmm=8.4.0``
   * - ``cuda-13-0``
     - OpenFF/OpenMM backend export and GPU OpenMM work
     - 13.0
     - ``openmm=8.5.1``

Known examples:

* CU Boulder Blanca older-GPU nodes: ``cuda-12-4``
* PSC Bridges2: ``cuda-12-6``

When unsure, choose the older compatible environment. The SAMMD tutorials use
``cuda-12-4`` as the default backend example.

7. Next step
------------

After installation, continue to :doc:`canonical-workflow` to build your first
SAMMD system.
