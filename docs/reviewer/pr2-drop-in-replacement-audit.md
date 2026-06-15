# PR2 Drop-In Replacement Audit

Scope: reviewer-driven audit of whether candidate dependencies can replace custom SAMMD code before `v0.1.0` without broad behavior changes.

## Verdict Table

| Component | Candidate replacement | Drop-in? | Reason | Recommended v0.1.0 action |
| --- | --- | --- | --- | --- |
| `src/sammd/backends/packmol.py` | `openff-packmol` | Partial | `openff-packmol.pack_box` can run PACKMOL with a fixed solute, but it takes/returns OpenFF `Topology` objects and owns input generation, PBC handling, solute wrapping, and output parsing. SAMMD currently needs AtomRecord/template PDB I/O, exact input rendering, `nloop`, `movebadrandom`, stdout files, clear validation errors, and solvent coordinate tuples in append order. | Keep custom wrapper for release. Consider a post-release adapter only around solvent packing if it preserves SAMMD metadata and tested errors. |
| `src/sammd/core/io.py` | OpenMM `PDBxFile.writeFile` | No | OpenMM is already used for the full `solvated_system.cif` export, but the custom writer produces the dependency-light inspection `sam_grafting_density.cif` from `AtomRecord` rows before an OpenMM topology exists. It also emits SAMMD component metadata, deterministic atom/residue labels, cell lengths, quoting, and loop terminators expected by tests and PyMOL inspection workflows. | Keep custom inspection writer. Keep OpenMM writer isolated to the Interchange-backed full-system export. |
| `src/sammd/utils/geometry.py` and SAM/analysis usage | `numpy` / `scipy` | Partial | Basic vector arithmetic, dot/cross products, norms, centroids, and matrix products are trivial NumPy equivalents. The nontrivial release contract is tuple-shaped return values, zero-vector errors, anti-parallel rotation handling, finite/tolerance behavior, and dependency-light import behavior used by builders, Interchange placement, and analysis tests. SciPy is not directly needed for the current functions. | Do not refactor before release. Defer any NumPy migration to a narrow performance/readability change with compatibility tests. |
| `src/sammd/model/surfaces.py` | mBuild slab construction | No | mBuild can build crystal/slab-like structures, but SAMMD's current planner is also the source of registered Fcc(111) metadata, deterministic centered coordinates, effective commensurate cell reporting, ABC layer offsets, top/bottom `fcc_hollow` and `hcp_hollow` sites, minimum-image nearest metal atom indices, and deterministic tests used by SAM anchor metadata. | Keep custom surface planner for release. Treat mBuild as a future optional coordinate generator only if metadata/site contracts remain SAMMD-owned. |
| `CONFIG_TEMPLATE` in `src/sammd/core/config.py` | Jinja templating | No for drop-in; possible refactor | Jinja could render the same YAML text, but it would add an undeclared dependency and a template/rendering layer without changing release behavior. Current tests assert exact beginner-schema wording, default YAML loadability, and absence of advanced knobs. | Defer. Only use Jinja if multiple generated templates become necessary, and require byte-for-byte or schema-equivalent output tests. |

## Detailed Notes

### PACKMOL Wrapper

`pixi.toml` declares the `packmol` executable in the `interchange` feature. The lock file also includes `openff-packmol` transitively through `openff-interchange`, but SAMMD does not import or declare it directly.

Current SAMMD behavior that is not a drop-in match for `openff-packmol`:

- `build_packmol_input` has deterministic text output tested for `tolerance`, `filetype`, `output`, `nloop`, `movebadrandom`, fixed structures, free structures, and nanometer-to-Angstrom bounds.
- `pack_fixed_solute_with_solvent` accepts SAMMD `AtomRecord` solute records plus a `PackmolMoleculeTemplate`, writes PACKMOL-readable PDB files with SAMMD atom/residue labels, runs a configured executable, writes stdout, checks both return code and `Success!`, parses fixed-width PDB coordinates back to nanometers, and returns only solvent molecule coordinate tuples.
- Tests assert custom validation errors for bad counts, paths, bounds, numeric values, missing executable, overwrite refusal, PDB parsing, fixed-solute input, and output atom-count checks.
- `openff-packmol.pack_box` is useful but higher level: it expects OpenFF molecules/topologies, generates its own input files, may wrap solute coordinates into a brick representation, returns an OpenFF `Topology`, and does not expose SAMMD's `nloop`/`movebadrandom` contract or raw grouped solvent positions.

### Inspection mmCIF Writer

OpenMM `PDBxFile.writeFile` is already the right writer for `solvated_system.cif` once Interchange has produced an OpenMM topology and positions. The Interchange wrapper still adds SAMMD behavior: overwrite protection, `keepIds=True`, atom/residue/chain identity repair, metal labels, missing atom-name repair, and a final `#` because PyMOL treats EOF inside the final atom-site loop as truncated.

The `core/io.py` writer serves a different release path: dependency-light inspection output for `sam_grafting_density.cif`. Replacing it with OpenMM would require constructing an OpenMM topology solely for inspection, moving an optional/heavy dependency into the normal planning path, and losing current custom rows unless reimplemented elsewhere. The custom writer emits `_entity`, `_sammd_entity`, `_atom_site`, optional cell lengths, deterministic Angstrom coordinates, numeric entity IDs, quoted CIF tokens, and explicit validation for duplicate serials and invalid atom records.

### Geometry Helpers

Trivial replacement candidates:

- `add_vectors`, `subtract_vectors`, `scale_vector`, `dot_product`, `cross_product`, `norm`, `distance`, `centroid`, `matvec`, `matrix_add`, `matrix_scale`, `matrix_multiply`, `outer_product`, and `skew_matrix` map directly to NumPy operations.

Not true drop-in behavior:

- `normalize` raises `ValueError("cannot normalize a zero vector")` below the current tolerance instead of returning `nan`/`inf` arrays.
- `rotation_matrix` includes project-specific source-to-target behavior, including identity for aligned vectors and a deterministic anti-parallel axis choice.
- `rotate_about_axis` returns plain `tuple[float, float, float]` and normalizes the axis with the same custom error behavior.
- `interchange.py` depends on tuple outputs during SAM orientation, reactant placement, and solvent centering. `analysis/orientation.py` separately keeps dependency-free validated geometry primitives with named result metadata and error messages.

NumPy is already present transitively in the locked interchange environment, and `openff-packmol` depends on SciPy, but neither is a top-level SAMMD dependency in `pixi.toml`. A release refactor would add risk without changing behavior.

### Surface Slab Planning

The surface planner is not just coordinate construction. It owns release-critical metadata and deterministic selection inputs:

- Registered Fcc(111) metals and lattice constants for Ag, Al, Au, Cu, Ni, Pb, Pd, and Pt.
- Clear rejection for unsupported metal/facet combinations.
- Deterministic centered slab positions, labels, layer indices, effective lateral size, supercell counts, top/bottom z positions, and slab thickness.
- Top and bottom `fcc_hollow` and `hcp_hollow` site generation with side-specific normals.
- Minimum-image nearest metal atom indices for each hollow site, which feed SAM anchor metadata and metal-sulfur pair overrides.
- Deterministic tests for non-Pd registered metals, commensurate cell expansion, seam neighbors, hollow-site distinctions, and failure messages.

mBuild may remain useful for future optional slab coordinate generation, but using it as the owner of this release contract would not preserve SAMMD's registered metadata, site records, nearest-neighbor index semantics, or deterministic tests without substantial custom code around it.

### Config Templating

`CONFIG_TEMPLATE` is a static beginner-facing YAML template with comments that tests use as a release contract. Jinja is not declared in `pixi.toml`, and introducing it would not change the default YAML schema, output files, or user workflow. A Jinja refactor could be worthwhile later if SAMMD needs multiple generated templates, conditional examples, or versioned variants, but it is not a drop-in release simplification.

## Follow-Up Items

- Prototype an optional `openff-packmol` adapter after `v0.1.0` only for solvent packing, and compare generated positions, retained labels, working files, stdout/error behavior, and failure messages against `tests/test_packmol.py`.
- Keep `core/io.py` dependency-light unless OpenMM becomes mandatory for ordinary `sammd build`; if that changes, add tests proving `sam_grafting_density.cif` still carries SAMMD component labels and remains PyMOL-readable.
- If moving geometry helpers to NumPy, preserve tuple return types at public/internal boundaries and keep explicit zero-vector, finite-number, and anti-parallel rotation tests.
- If evaluating mBuild slabs, define an adapter that returns `SurfaceSlab` and `BindingSite` unchanged, including registered metadata, nearest atom indices, and deterministic ordering.
- Defer Jinja until there is more than one template variant; require output stability tests before replacing `CONFIG_TEMPLATE`.
