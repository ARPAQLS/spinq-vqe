# Testing Guide

> **`spinq-vqe` test suite** — how to run, what's covered, and how to extend.

---

## Quick start

```bash
# From the repo root, activate your venv
source .venv/bin/activate       # Linux / macOS
.venv\Scripts\activate          # Windows

# Install with dev extras (includes pytest and ruff)
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v
```

On Windows using the shared workspace venv at `d:\ARPA\OpenSource\Spintronics\.venv`:

```bash
d:\ARPA\OpenSource\Spintronics\.venv\Scripts\python.exe -m pytest tests/ -v
```

---

## Test structure

```
tests/
├── conftest.py             # Shared fixtures (G1, G3, H1, edges)
├── test_kagome.py          # Lattice graph + Hamiltonian builder
├── test_ansatz.py          # HEA, HVA, MERA param counts + circuit execution
├── test_vqe.py             # COBYLA and Adam VQE runners
├── test_entanglement.py    # Reduced density matrix + Von Neumann entropy
├── test_surrogate.py       # CSV + mock data, feature extraction, MLP training, prediction
├── test_qaoa.py            # Cost/mixer Hamiltonians, QAOA run, classical greedy
└── test_dmrg.py            # TeNPy Hamiltonian match + N=9 DMRG energy (optional dep)
```

All tests use **N=3** (one Kagome unit cell, 3 sites) or **N=4** (QAOA) with minimal optimizer steps. The full suite runs in **under 90 seconds** on CPU.

---

## Coverage by module

| Module | File | Key checks |
|--------|------|------------|
| `kagome.py` | `test_kagome.py` | Node/edge counts per `n_cells`, sublattice labels, boundary conditions, Hamiltonian term counts, normalize flag, utility functions |
| `ansatz.py` | `test_ansatz.py` | HEA/HVA/MERA parameter counts (exact), `init_params` reproducibility and scale, circuit execution → normalized statevector |
| `vqe.py` | `test_vqe.py` | `VQEResult` fields, COBYLA: energy finiteness, history length, gradient variance empty, statevector shape/normalization, metadata; Adam: grad variance non-empty |
| `entanglement.py` | `test_entanglement.py` | RDM shape/trace/hermiticity/PSD, entropy = 0 for product state, entropy = 1 for Bell pair, upper bound, base conversion |
| `surrogate.py` | `test_surrogate.py` | CSV load (`mp_theta_sh.csv`), mock fallback, Mn₃Sn + real MP ID, feature matrix, training, prediction |
| `qaoa.py` | `test_qaoa.py` | Cost/mixer Hamiltonian types, QAOA result structure, landscape grid, param history, classical greedy top-k |
| `dmrg.py` | `test_dmrg.py` | TeNPy/PennyLane Hamiltonian match, N=9 energy vs ED (< 0.01%), CSV round-trip (skipped if `physics-tenpy` not installed) |

---

## Fixtures (`conftest.py`)

All fixtures use `scope="session"` — built once per test run, not per test:

| Fixture | Type | Description |
|---------|------|-------------|
| `G1` | `nx.Graph` | 3-node Kagome graph (1 unit cell) |
| `G3` | `nx.Graph` | 9-node Kagome graph (3 unit cells) |
| `H1` | `qp.Hamiltonian` | Heisenberg Hamiltonian on G1, default params, normalize=True |
| `H1_bare` | `qp.Hamiltonian` | J only (D=0, B=0), normalize=False — used in VQE tests |
| `edges1` | `list[tuple]` | Edge list for G1 |
| `edges3` | `list[tuple]` | Edge list for G3 |

---

## Running subsets

```bash
# Single file
pytest tests/test_kagome.py -v

# Single class
pytest tests/test_vqe.py::TestRunVQECobyla -v

# Single test
pytest tests/test_entanglement.py::TestVonNeumannEntropy::test_maximal_for_bell_pair -v

# With coverage (requires pytest-cov)
pip install pytest-cov
pytest tests/ --cov=spinq_vqe --cov-report=term-missing
```

---

## Lint

```bash
ruff check src/
```

`ruff` is configured in `pyproject.toml` under `[tool.ruff]`. Run before committing.

---

## Known limitations

- **No convergence test for VQE on real physics.** N=3 with 30 function evaluations is enough to verify the runner works, not to verify it reaches the true ground state. The notebooks (`NB02`, `NB05`, `NB06`) carry the physics-level validation.
- **DMRG tests require `physics-tenpy`.** `test_dmrg.py` uses `pytest.importorskip("tenpy")` and is skipped when the optional `[dmrg]` extra is not installed.
- **No GPU tests.** All tests use `default.qubit` (CPU). `lightning.qubit` and JAX backends are exercised in notebooks only.
- **QAOA k-constraint test is probabilistic.** `run_qaoa` selects k materials by sampling from the optimized circuit. With very few optimizer steps the selection is not guaranteed to satisfy the constraint. The test uses a tolerance-free `len(selected_indices) == k` check — if the QAOA rarely fails this, the implementation correctly decodes the argmax bitstring.

---

## Adding new tests

When adding a new module or extending an existing one:

1. Add a `tests/test_<module>.py` file.
2. Use `conftest.py` fixtures where applicable; add new fixtures to `conftest.py` if they'd be reused.
3. Keep all tests CPU-only (`default.qubit`) and parameterized with small system sizes.
4. Tests should complete in `< 5 seconds each`. Use `pytest.mark.slow` (and skip by default) for anything longer.

---

*Last updated: 2026-08-06 · spinq-vqe v0.1.4 (NB06 DMRG)*
