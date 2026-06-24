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
| `load_mock_data()` | 12-material curated dataset (offline, no API key) |
| `load_mp_data(api_key, elements, n_results)` | Fetch from Materials Project API |
| `build_features(dataset)` | Extract feature matrix `(N, 6)` |
| `train_surrogate(dataset, ...)` | Fit sklearn MLP (or numpy ridge fallback) |
| `predict(surrogate, records)` | Predict θ_SH for new records |
| `surrogate_summary(surrogate)` | Print model info + CV R² |

```python
ds = surrogate.load_mock_data()        # or load_mp_data(api_key)
sr = surrogate.train_surrogate(ds)
surrogate.surrogate_summary(sr)        # sklearn MLP | features: 6 | CV R²: ...
theta_sh = surrogate.predict(sr, ds.records)
```

**Optional deps:** `scikit-learn` (MLP), `mp-api` (MP fetch). Falls back to numpy ridge if sklearn missing. Install: `pip install -e ".[data]"`.

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
| `classical_greedy(theta_sh, k)` | Top-k greedy baseline |
| `classical_simulated_annealing(theta_sh, k, ...)` | SA baseline |
| `qaoa_summary(result, formulas)` | Print result summary |

```python
theta_sh = surrogate.predict(sr, ds.records)
result = qaoa.run_qaoa(theta_sh, k=3, p=2, n_seeds=5, verbose=True)
# result.selected_indices → best 3 materials
# result.selected_theta_sh → total θ_SH
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

All plots use a consistent soft pastel palette (`SUBLATTICE_COLORS`, `ANSATZ_COLORS`).
