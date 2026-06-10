# Temporary Developer Utilities

This folder is for developer-only utilities that are not part of the SAMMD
v0.1.0 release, public API, or long-term support contract.

`openmm_smoke.py` is kept only for manual validation while backend export work is
in flight. It may be changed or deleted without deprecation.

SAMMD should own system setup and export code. OpenMM simulation workflows belong
in documentation and notebook examples, not in package source.
