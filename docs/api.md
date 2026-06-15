# API Reference

> [← index](index.md)

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

**Usage:**
```python
G = kagome.kagome_graph(n_cells=3)   # 9 sites
H = kagome.heisenberg_kagome_hamiltonian(G)
```

**Physical constants** (module-level): `J_MN3SN_MEV = 4.0`, `D_MN3SN_MEV = 0.3`

---

## `spinq_vqe.ansatz`

Variational ansatze. See [ansatze.md](ansatze.md) for full guide.

```python
from spinq_vqe import ansatz
```

| Function | Description |
|----------|-------------|
| `hva_ansatz(params, n_sites, depth, edges)` | **Primary.** Trotterized Heisenberg ansatz |
| `hva_n_params(depth)` | `depth * 3` |
| `hea_ansatz(params, n_sites, depth, edges)` | Hardware-efficient RY+CNOT |
| `hea_n_params(n_sites, depth)` | `depth * n_sites` |
| `mera_ansatz(params, n_sites, G)` | 2-scale MERA-inspired |
| `mera_n_params(G)` | `4*n_bonds + 4*min(|A|,|B|) + n_sites` |
| `init_params(ansatz, n_sites, G, depth, seed, scale)` | Random parameter init |

---

## `spinq_vqe.vqe`

VQE runner using PennyLane's Adam optimizer.

```python
from spinq_vqe import vqe
```

### `run_vqe`

```python
result = vqe.run_vqe(
    hamiltonian,        # qp.Hamiltonian
    ansatz_fn,          # callable: fn(params, n_sites, **kwargs)
    init_params,        # np.ndarray
    n_sites,            # int
    ansatz_name='hva',
    n_steps=2000,
    step_size=0.05,
    conv_tol=1e-8,      # variance threshold
    conv_window=500,    # steps to check; only triggers if energy < 0
    return_statevector=True,
    verbose=True,
    **ansatz_kwargs,    # passed through to ansatz_fn
)
```

**Returns:** `VQEResult` dataclass with fields:

| Field | Type | Description |
|-------|------|-------------|
| `energy` | `float` | Best energy found |
| `params` | `np.ndarray` | Optimal parameters |
| `energy_history` | `list[float]` | Energy at each step |
| `gradient_variance_history` | `list[float]` | Barren plateau diagnostic |
| `n_steps` | `int` | Steps taken |
| `converged` | `bool` | Whether convergence criterion triggered |
| `statevector` | `np.ndarray | None` | Final $|\psi\rangle$ (shape: `(2^N,)`) |

**Convergence note:** Early stopping requires `energy < 0` AND `np.var(last conv_window steps) < conv_tol`. This prevents false convergence at the ferromagnetic saddle point.

### `compare_ansatze`

```python
results = vqe.compare_ansatze(H, hea_fn, mera_fn, p_hea, p_mera, n_sites, ...)
# → {'hea': VQEResult, 'mera': VQEResult}
```

---

## `spinq_vqe.entanglement`

Von Neumann entropy and mutual information from VQE wavefunctions.

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
| `sublattice_mutual_info_matrix` | `(statevector, G, n_sites)` | `np.ndarray` shape `(3, 3)` |

**Typical usage:**
```python
sv = np.load('data/statevector_hva_best.npy')
rho_A = entanglement.reduced_density_matrix(sv, subsystem=[0,1,2], n_sites=9)
S_A   = entanglement.von_neumann_entropy(rho_A)
MI    = entanglement.mutual_information_matrix(sv, n_sites=9)
```

---

## `spinq_vqe.utils`

Publication-quality plot helpers. All functions return `plt.Figure` and optionally save to disk.

```python
from spinq_vqe import utils
```

| Function | Description |
|----------|-------------|
| `plot_kagome_graph(G, ...)` | Lattice with sublattice colors |
| `plot_energy_convergence(results, ed_energy, ...)` | VQE energy history |
| `plot_entanglement_profile(profile, ...)` | $S_{vN}$ vs subsystem size |
| `plot_mutual_info_matrix(matrix, ...)` | 3×3 sublattice MI heatmap |
| `plot_gradient_variance(results, ...)` | Barren plateau diagnostic |

All plots use a consistent soft pastel palette (`SUBLATTICE_COLORS`, `ANSATZ_COLORS`).
