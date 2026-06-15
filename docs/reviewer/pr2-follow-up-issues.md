# PR2 Follow-Up Issues

Scope: non-blocking design and policy work from PR #2 review that should not block `v0.1.0`. These items preserve reviewer context from `docs/reviewer/pr2-drop-in-replacement-audit.md` and track post-release decisions.

## Issue Candidates

### Evaluate optional `openff-packmol` adapter

- GitHub issue: <https://github.com/joelaforet/SAMMD/issues/6>
- Label: `enhancement`
- Context: `openff-packmol` may be useful after release, but it is not a drop-in replacement for SAMMD's current PACKMOL wrapper because SAMMD owns AtomRecord/template PDB I/O, exact input rendering, PACKMOL option handling, stdout/error behavior, and grouped solvent coordinate return values.
- Acceptance criteria: decide whether to keep the custom wrapper, add an optional adapter, or reject the dependency; document behavior gaps and dependency impact; add tests if an adapter is implemented.

### Evaluate NumPy/SciPy geometry refactor

- GitHub issue: <https://github.com/joelaforet/SAMMD/issues/4>
- Label: `enhancement`
- Context: NumPy could replace many low-level vector operations, but the release contract includes tuple-shaped returns, custom zero-vector errors, deterministic anti-parallel rotation behavior, and finite/tolerance semantics. SciPy should not be added unless it is directly needed.
- Acceptance criteria: decide whether to refactor, keep current helpers, or narrow the change to selected internals; preserve tuple/error semantics; add compatibility tests for public/internal geometry behavior if code changes.

### Evaluate mBuild surface coordinate generation

- GitHub issue: <https://github.com/joelaforet/SAMMD/issues/5>
- Label: `enhancement`
- Context: mBuild may be useful as an optional coordinate generator, but SAMMD must continue to own registered metal/facet metadata, `SurfaceSlab`/`BindingSite` records, top/bottom site semantics, nearest metal atom indices, and deterministic ordering.
- Acceptance criteria: decide whether mBuild should be adopted, rejected, or prototyped behind an adapter; preserve current metadata and binding-site contracts if implemented; add regression tests for ordering, site metadata, and nearest atom indices if code changes.

### Consider Jinja config template rendering

- GitHub issue: <https://github.com/joelaforet/SAMMD/issues/8>
- Label: `enhancement`
- Context: Jinja is unnecessary for the current static beginner config template, but it may become useful if SAMMD grows multiple generated templates, conditional examples, or versioned variants.
- Acceptance criteria: revisit only when multiple template variants exist; decide whether static templates remain sufficient or Jinja is warranted; if Jinja is adopted, add byte-for-byte or schema-equivalent output tests and document the new dependency.

### Decide analysis and notebook dependencies

- GitHub issue: <https://github.com/joelaforet/SAMMD/issues/7>
- Label: `enhancement`
- Context: Analysis and notebook workflows need a separate dependency decision for packages such as `pymbar`, `MDAnalysis`, `MDTraj`, `numpy`, `pandas`, `scipy`, `matplotlib`, and `seaborn`.
- Acceptance criteria: record which packages belong in core runtime dependencies, optional features, examples, notebooks, or docs-only guidance; update project metadata if package ownership changes; update docs or examples so users know which environment to use.

### Document SAM placement and slab commensuration behavior

- GitHub issue: <https://github.com/joelaforet/SAMMD/issues/9>
- Label: `documentation`
- Context: SAM placement, binding-site selection, deterministic ordering, and requested-versus-effective slab size commensuration need a public design/documentation decision after release.
- Acceptance criteria: document the SAM placement algorithm and slab commensuration behavior; decide which details are public contract versus implementation detail; add or update tests if documented guarantees expose missing coverage.

### Decide documentation-test and NumPy-style docstring policy

- GitHub issue: <https://github.com/joelaforet/SAMMD/issues/3>
- Label: `documentation`
- Context: PR #2 deferred whether documentation examples should be tested and whether NumPy-style docstrings are required everywhere, only for public APIs, or optional.
- Acceptance criteria: document the chosen documentation-test and docstring policy; update contributor or developer docs if present; add tooling or tests only if the policy requires enforcement.
