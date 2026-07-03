# Changelog

All notable changes to this project can be documented in this file.

The format follows the spirit of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses semantic versioning through the package version in
`pyproject.toml`.

## [Unreleased]

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
