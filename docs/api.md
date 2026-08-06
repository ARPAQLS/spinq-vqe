# API Reference

> [← index](README.md)

---

## `spinq_vqe.kagome`

Kagome lattice graph construction and Heisenberg Hamiltonian builder.

```python
from spinq_vqe import kagome
```

| Function | Signature | Returns |
|----------|-----------|---------|
| `kagome_graph` | `(n_cells=1, boundary='open')` | `nx.Graph` with `sublattice` node attr |
| `heisenberg_kagome_hamiltonian` | `(G, J=4.0, D=0.3, B=0.0, normalize=True)` | `qp.Hamiltonian` |
| `sublattice_partition` | `(G)` | `{0: [sites_A], 1: [sites_B], 2: [sites_C]}` |
| `n_sites` | `(G)` | `int` |
| `n_bonds` | `(G)` | `int` |

```python
G = kagome.kagome_graph(n_cells=3)   # 9 sites
H = kagome.heisenberg_kagome_hamiltonian(G)
parts = kagome.sublattice_partition(G)  # {0: [...], 1: [...], 2: [...]}
```

**Physical constants** (module-level): `J_MN3SN_MEV = 4.0`, `D_MN3SN_MEV = 0.3`

---

## `spinq_vqe.ansatz`

Variational ansatze. See [ansatze.md](ansatze.md) for the full guide.

```python
from spinq_vqe import ansatz
```

| Function | Description |
|----------|-------------|
| `hva_ansatz(params, n_sites, depth, edges)` | Trotterized Heisenberg (physics-motivated) |
| `hva_n_params(depth)` | `depth * 3` |
| `hea_ansatz(params, n_sites, depth, edges)` | **Used in NB02.** Hardware-efficient RY+CNOT |
| `hea_n_params(n_sites, depth)` | `depth * n_sites` |
| `mera_ansatz(params, n_sites, G)` | 2-scale MERA-inspired |
| `mera_n_params(G)` | `4*n_bonds + 4*min(\|A\|,\|B\|) + n_sites` |
| `init_params(ansatz, n_sites, G, depth, seed, scale)` | Random parameter init |

---

## `spinq_vqe.vqe`

Two VQE runners. **Use `run_vqe_cobyla` for actual optimization** on this system.

```python
from spinq_vqe import vqe
```

### `run_vqe_cobyla_multi_seed` ← multi-seed statistics

Runs COBYLA over a list of integer seeds (via ``init_params_fn``), returns the best
``VQEResult`` plus ``SeedStatistics`` (mean, std, min, max).

```python
def init_fn(seed: int) -> np.ndarray:
    return ansatz.init_params('hea', n_sites=9, depth=3, seed=seed, scale=1.0)

multi = vqe.run_vqe_cobyla_multi_seed(
    hamiltonian, ansatz.hea_ansatz, init_fn, n_sites=9,
    seeds=[42, 7, 123, 99, 17],
    n_evals=5000, depth=3, edges=edges,
)
print(multi.statistics.mean_energy, multi.statistics.std_energy)
best = multi.best  # lowest energy; statevector attached if return_statevector=True
```

### `seed_statistics`

```python
stats = vqe.seed_statistics([r.energy for r in multi.runs])
```

### `run_vqe_cobyla` ← primary

COBYLA gradient-free optimizer. Immune to the zero-gradient problem at Z-basis eigenstates.

```python
result = vqe.run_vqe_cobyla(
    hamiltonian,          # qp.Hamiltonian
    ansatz_fn,            # callable: fn(params, n_sites, **kwargs)
    init_params,          # np.ndarray  (use scale=1.0 for broad exploration)
    n_sites,              # int
    ansatz_name='hea',
    n_evals=5000,         # max energy evaluations
    rhobeg=0.5,           # initial COBYLA trust-region radius
    return_statevector=True,
    verbose=True,
    **ansatz_kwargs,
)
```

### `run_vqe` — Adam (diagnostic)

For gradient variance analysis and barren plateau diagnostics only.  
Will stall on the Heisenberg AFM due to zero gradients at `|0⟩⊗N`.

```python
result = vqe.run_vqe(
    hamiltonian, ansatz_fn, init_params, n_sites,
    n_steps=2000, step_size=0.05,
    conv_tol=1e-8, conv_window=500,
    **ansatz_kwargs,
)
```

### `VQEResult` dataclass

| Field | Type | Description |
|-------|------|-------------|
| `energy` | `float` | Best energy found |
| `params` | `np.ndarray` | Optimal parameters |
| `energy_history` | `list[float]` | Energy at each step / evaluation |
| `gradient_variance_history` | `list[float]` | Barren plateau diagnostic (empty for COBYLA) |
| `n_steps` | `int` | Steps / evaluations taken |
| `converged` | `bool` | Whether optimizer reported convergence |
| `statevector` | `np.ndarray \| None` | Final `\|ψ⟩`, shape `(2^N,)` |
| `optimizer` | `str` | `'cobyla'` or `'adam'` |

### `SeedStatistics` / `VQEMultiSeedResult`

| Field | Type | Description |
|-------|------|-------------|
| `mean_energy`, `std_energy`, `min_energy`, `max_energy` | `float` | Aggregate over seeds |
| `n_seeds` | `int` | Number of runs |
| `best` | `VQEResult` | Lowest-energy run (statevector on best only) |
| `runs` | `list[VQEResult]` | Per-seed results |
| `seeds` | `list[int]` | Seeds used |

---

## `spinq_vqe.entanglement`

Von Neumann entropy and mutual information from VQE statevectors.

```python
from spinq_vqe import entanglement
```

| Function | Signature | Returns |
|----------|-----------|---------|
| `reduced_density_matrix` | `(statevector, subsystem, n_sites)` | `np.ndarray` (density matrix) |
| `von_neumann_entropy` | `(rho, base=2.0)` | `float` (bits) |
| `mutual_information` | `(statevector, subsystem_A, subsystem_B, n_sites)` | `float` |
| `mutual_information_matrix` | `(statevector, n_sites)` | `np.ndarray` shape `(N, N)` |
| `entanglement_profile` | `(statevector, n_sites)` | `{'subsystem_sizes': [...], 'entropies': [...]}` |
| `sublattice_mutual_info_matrix` | `(statevector, sublattices, n_sites)` | `np.ndarray` shape `(3, 3)` |

> **Note:** `sublattice_mutual_info_matrix` takes the `sublattices` dict from
> `kagome.sublattice_partition(G)` — **not** the graph `G` directly.

```python
sv    = np.load('data/statevector_hea_best.npy')
parts = kagome.sublattice_partition(G)

rho_A  = entanglement.reduced_density_matrix(sv, subsystem=[0,1,2], n_sites=9)
S_A    = entanglement.von_neumann_entropy(rho_A)
MI     = entanglement.mutual_information_matrix(sv, n_sites=9)
sub_MI = entanglement.sublattice_mutual_info_matrix(sv, parts, n_sites=9)
```

---

## `spinq_vqe.surrogate`

MLP surrogate for spin Hall angle (θ_SH) prediction. Used as oracle for `qaoa`.

```python
from spinq_vqe import surrogate
```

| Function | Description |
|----------|-------------|
| `load_theta_sh_data()` | **Primary** — load `data/mp_theta_sh.csv` (committed, no API key) |
| `load_theta_sh_csv(path)` | Load CSV directly |
| `load_mock_data()` | Offline fallback for unit tests only |
| `fetch_curated_mp_dataset(api_key)` | Fetch MP descriptors + illustrative oracle targets (refresh) |
| `load_mp_data(api_key)` | Same as fetch, returns dataset only |
| `save_theta_sh_csv(dataset, path, extra)` | Write CSV after a refresh |
| `build_features(dataset)` | Extract feature matrix `(N, 6)` |
| `train_surrogate(dataset, ...)` | Fit sklearn MLP (or numpy ridge fallback) |
| `predict(surrogate, records)` | Predict θ_SH for new records |
| `surrogate_summary(surrogate)` | Print model info + CV R² |

```python
ds = surrogate.load_theta_sh_data()    # uses data/mp_theta_sh.csv
sr = surrogate.train_surrogate(ds)
surrogate.surrogate_summary(sr)
theta_sh = surrogate.predict(sr, ds.records)
```

Refresh CSV (optional, requires `MP_API_KEY` in `.env`):

```bash
python scripts/fetch_mp_theta_sh.py
```

**Optional deps:** `scikit-learn` (MLP), `mp-api` (MP refresh only). Install: `pip install -e ".[data]"`.

---

## `spinq_vqe.qaoa`

QAOA circuit and optimizer for k-from-N material selection.

```python
from spinq_vqe import qaoa
```

| Function | Description |
|----------|-------------|
| `build_cost_hamiltonian(theta_sh, k, lam)` | QUBO cost Hamiltonian |
| `build_mixer_hamiltonian(n_materials)` | Transverse-field X mixer |
| `run_qaoa(theta_sh, k, p, ...)` | Full QAOA optimization (COBYLA) |
| `evaluate_qaoa_cost(theta_sh, params, k, p, lam)` | Single cost evaluation |
| `qaoa_landscape_grid(theta_sh, k, lam, n_gamma, n_beta)` | Sample p=1 (γ, β) landscape |
| `find_landscape_minima(landscape)` | Coarse local minima on a landscape grid |
| `classical_greedy(theta_sh, k)` | Top-k greedy baseline → `list[int]` of selected indices |
| `classical_simulated_annealing(theta_sh, k, ...)` | SA baseline |
| `qaoa_summary(result, formulas)` | Print result summary |

```python
theta_sh = surrogate.predict(sr, ds.records)
result = qaoa.run_qaoa(theta_sh, k=3, p=2, n_seeds=5, verbose=True)
# result.selected_indices → best 3 materials
# result.selected_theta_sh → total θ_SH
```

---

## `spinq_vqe.dmrg`

TeNPy DMRG reference energies for 1D Kagome strips (NB06). Requires `pip install -e ".[dmrg]"`.

```python
from spinq_vqe import dmrg
```

| Function | Signature | Returns |
|----------|-----------|---------|
| `build_kagome_dmrg_model` | `(n_cells, J=..., D=...)` | `KagomeStripDMRG` TeNPy model |
| `validate_hamiltonian_against_pennylane` | `(n_cells, ...)` | `float` max matrix difference |
| `run_dmrg` | `(n_cells, chi_max=400, ...)` | `DMRGResult` |
| `run_dmrg_chi_sweep` | `(n_cells, chi_values, ...)` | `list[DMRGResult]` |
| `save_dmrg_reference_csv` | `(results, path)` | `Path` |
| `load_dmrg_reference_csv` | `(path)` | `dict[int, float]` keyed by `n_sites` |

```python
res = dmrg.run_dmrg(n_cells=3, chi_max=64)
print(res.e0, res.chi)  # normalized E0/site, bond dimension used
```

---

## `spinq_vqe.utils`

Publication-quality plot helpers.

```python
from spinq_vqe import utils
```

| Function | Description |
|----------|-------------|
| `plot_kagome_graph(G, ...)` | Lattice with sublattice colors |
| `plot_energy_convergence(results, ed_energy, ...)` | VQE energy history |
| `plot_entanglement_profile(profile, ...)` | S_vN vs subsystem size |
| `plot_mutual_info_matrix(matrix, ...)` | Sublattice MI heatmap |
| `plot_gradient_variance(results, ...)` | Barren plateau diagnostic |
| `plot_qaoa_landscape(gamma, beta, energies, ...)` | NB04 landscape + θ_SH depth panel |

All plots use a consistent soft pastel palette (`SUBLATTICE_COLORS`, `ANSATZ_COLORS`).
