# Changelog

All notable changes to this project can be documented in this file.

The format follows the spirit of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses semantic versioning through the package version in
`pyproject.toml`.

## [Unreleased]

### Added

- GitHub issue form templates (bug / science / notebook / docs / enhancement) and pastel label palette, synced by Actions from `.github/labels.yml` (#12).
- GitHub Actions CI: `ruff` + `pytest` on Python 3.11/3.12 (`.[dev]` only; optional DMRG/NQS extras skipped) (#2).
- Neural Quantum State comparison via NetKet (`spinq_vqe.nqs`, NB07, `data/method_comparison.csv`) (#9).
- Optional dependency group `[nqs]` with `netket`.
- Figures: `nqs_vmc_convergence.png`, `nqs_method_comparison.png`, `nqs_error_vs_ed.png`.
- Script `scripts/run_nqs_benchmark.py` to regenerate NB07 outputs (append-only; does not modify DMRG figures).
- Convergence histories as CSV (`data/nqs_*_history_n*.csv`) for notebook figure regeneration.

## [0.1.5] - 2026-08-06

### Added

- DMRG comparison via TeNPy (`spinq_vqe.dmrg`, NB06, `data/dmrg_reference_energies.csv`) (#8, #30).
- Optional dependency group `[dmrg]` with `physics-tenpy`.
- Figures: `dmrg_chi_convergence.png`, `dmrg_vqe_comparison.png`, `dmrg_entanglement_profile.png`.
- Script `scripts/run_dmrg_benchmark.py` to regenerate NB06 outputs and update `scaling_energy.png`.
- θ_SH provenance contract (`data/theta_sh_sources.md`) and row-level ledger
  (`data/theta_sh_provenance.csv`) for NB04 (#22, #29).
- Contract test requiring every committed oracle row to be explicitly sourced or
  illustrative.

### Changed

- NB05 scaling figure and VQE error metric now reference DMRG (not ED-only) where DMRG data exists.
- README key results: VQE error relative to DMRG; hero figures use χ-convergence and entanglement profile.
- Reclassified all 12 committed `theta_sh` targets as `illustrative_oracle`
  (not unqualified `literature`); aligned README, OVERVIEW, NB04, `surrogate.py`,
  and `docs/notebooks.md` with the same scientific boundary (#22, #29).
- Package version bumped to 0.1.5.

## [0.1.4] - 2026-07-27

### Added

- Zenodo concept DOI `10.5281/zenodo.21628505` in `README.md`, `CITATION.cff`,
  and `OVERVIEW.md`.

### Changed

- `CITATION.cff` notes/message clarified: cite the software concept DOI for the
  code artifact; cite any related paper DOI separately.
- Package version bumped to 0.1.4.

## [0.1.3] - 2026-07-27

### Added

- QAOA p=1 landscape utilities (`qaoa_landscape_grid`, `find_landscape_minima`,
  `record_param_history` on `run_qaoa`) and `figures/qaoa_landscape.png` (#7, #28).
- NB04: landscape heatmap + θ_SH-vs-depth panel; `scripts/generate_qaoa_landscape.py`.
- `seed_statistics`, `run_vqe_cobyla_multi_seed`, `SeedStatistics`, and
  `VQEMultiSeedResult` in `vqe.py` for seed-level VQE reporting (#6).
- NB02: mean ± std table, per-seed CSV (`data/vqe_seeds_n9.csv`), box plot
  (`figures/vqe_seed_distribution.png`).
- NB05: multi-seed N=12 via `run_vqe_cobyla_multi_seed`, cross-size box plot
  (`figures/vqe_seed_scaling_boxplot.png`).
- Zenodo DOI badge placeholder in `README.md` (concept DOI to be filled after
  first deposit).

### Changed

- Clarified the NB04 data boundary in `OVERVIEW.md`: Materials Project provides
  structure metadata, θ_SH targets remain curated literature labels, and the
  notebook demonstrates the method rather than claiming material discovery.
- Landscape plot uses `pcolormesh` with wrapped COBYLA trajectories; depth panel
  reports total θ_SH (with greedy reference) rather than ⟨H_C⟩ (#7).
- `data/vqe_results.csv` and `data/vqe_scaling.csv` schemas extended with
  `mean_energy`, `std_energy`, `n_seeds`, `min_energy`, `max_energy` (#6).
- README, OVERVIEW, and docs updated to report best seed and mean ± std.
- `CITATION.cff` and package metadata: author Vladimiros Peilivanidis with
  ARPA Quantum Logical Systems (QONDRA); Zenodo-ready CFF fields (`type: software`).

## [0.1.2] - 2026-07-03

### Changed

- Rewrote `OVERVIEW.md` as a public-facing research narrative with key results,
  scientific context, roadmap, and cross-repo dependencies (#3).
- Linked `OVERVIEW.md` from `README.md` and `docs/README.md`.
- Removed internal planning language from `docs/notebooks.md` (status badges),
  `docs/physics.md`, `src/spinq_vqe` module docstrings, and notebook headers
  in NB04/NB05 (#3).
- Updated `src/spinq_vqe/__init__.py` module descriptions for `surrogate` and
  `qaoa` to behaviour-focused wording with no internal cluster codes (#4).
- NB04 uses `load_theta_sh_data()` instead of hardcoded mock data (#5).
- `fetch_curated_mp_dataset()` replaces stub `load_mp_data()` — correct MP IDs
  by formula, literature θ_SH (#5).

### Fixed

- NB04: `ax.axhline` placeholder and `greedy["total"]` dict access after
  `classical_greedy` return-type change; regenerated figures and
  `data/qaoa_results.csv` with MP-grounded CSV inputs.

### Added

- Committed `data/mp_theta_sh.csv` — 12 spintronic materials with MP structure
  descriptors and literature θ_SH labels (#5).
- `load_theta_sh_data()` CSV-first loader; `scripts/fetch_mp_theta_sh.py` to
  refresh data via Materials Project API when `MP_API_KEY` is set (#5).
- `.env.example` for optional MP API key; `.env` added to `.gitignore` (#5).

## [0.1.1] - 2026-07-03

### Added

- Citation metadata in `CITATION.cff`.
- Conda-style environment specification in `environment.yml`.
- Initial changelog for release and publication-readiness tracking.
- Pytest unit test suite (`tests/`) covering `kagome`, `ansatz`, `vqe`,
  `entanglement`, `surrogate`, and `qaoa`, with shared session fixtures in
  `conftest.py`.
- Testing guide in `docs/testing.md` (how to run, coverage map, fixtures).
- Notebook 04 (`04_soc_qaoa.ipynb`): SOC QAOA workstream — MLP surrogate on a
  12-material mock dataset, QAOA at depths p = 1/2/3, greedy and simulated-
  annealing baselines; results in `data/qaoa_results.csv` and
  `figures/qaoa_*.png`.

### Changed

- `classical_greedy` in `qaoa.py` now returns `list[int]` (top-k material
  indices) instead of a dict.

### Fixed

- Notebook 04: updated `classical_greedy` call sites after the `list[int]`
  return-type refactor.

## [0.1.0] - 2026-06-01

### Added

- Initial Python package for variational quantum simulation of
  antiferromagnetic Hamiltonians.
- Kagome lattice Hamiltonian utilities, variational ansatz modules, VQE
  runners, entanglement analysis helpers, surrogate modeling, and QAOA tools.
- Research notebooks, generated figures, reference data, documentation, and
  bibliography files.
