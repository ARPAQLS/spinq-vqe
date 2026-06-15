# Ansatz Guide

> [← index](index.md)

Three variational ansatze are implemented. **HVA is the primary choice for this system.**

---

## Comparison at a glance

| Ansatz | Params (N=9) | Init scale | Barren plateau | Use for |
|--------|-------------|-----------|----------------|---------|
| **HVA** (depth=6) | 18 | 0.05 | ✅ None | Heisenberg models — primary |
| **HEA** (depth=3) | 27 | 1.0 | ⚠️ Moderate (N=9) | Baseline comparison |
| **MERA** | 65 | 1.0 | ⚠️ Moderate | Architecture research |

---

## HVA — Hamiltonian Variational Ansatz

**Why this one?** The HVA gates are built directly from the Hamiltonian terms. At small angles the gradients are $O(1)$ regardless of system size — provably no barren plateau for translation-invariant Heisenberg models (Wiersema et al. 2020).

### Circuit structure

Each layer applies Trotterized evolution under each bond term:

$$\text{Layer } l: \quad \prod_{(i,j) \in \text{bonds}} e^{-i\theta_{XX}^{(l)} X_iX_j} \cdot e^{-i\theta_{YY}^{(l)} Y_iY_j} \cdot e^{-i\theta_{ZZ}^{(l)} Z_iZ_j}$$

In PennyLane (using $\text{IsingXX}(\phi) \equiv e^{-i\phi/2 \cdot XX}$):

```python
for i, j in edges:
    qp.IsingXX(2.0 * theta_xx, wires=[i, j])
    qp.IsingYY(2.0 * theta_yy, wires=[i, j])
    qp.IsingZZ(2.0 * theta_zz, wires=[i, j])
```

One $(\theta_{XX}, \theta_{YY}, \theta_{ZZ})$ per layer, **shared across all bonds**. Valid because the Kagome Hamiltonian has uniform exchange $J$ on all bonds.

### Parameters

- `depth * 3` total (e.g. depth=6 → 18 params)
- **Init:** `scale=0.05` (small-angle, near identity circuit)
- **Steps:** 2000 sufficient; convergence criterion requires `energy < 0` (past ferromagnetic saddle) before early stopping

### Why not small angles for HEA?

Near $|0\rangle^{\otimes N}$ (all spins up = ferromagnetic state), $\langle XX \rangle = \langle YY \rangle = 0$ exactly. The gradient of the exchange terms vanishes. For the AFM ground state the optimizer needs to cross the ferromagnetic saddle point first — HEA has no gradient signal to do this at small angles. HVA's Ising gates have non-zero gradients at $\theta = 0$ because they act on the off-diagonal elements of $H$.

---

## HEA — Hardware-Efficient Ansatz

General-purpose ansatz. Good as a comparison baseline and for hardware experiments where native 2-qubit gates are constrained to a specific topology.

### Circuit structure

```
Layer l:
  RY(θ_{l,0}), ..., RY(θ_{l,N-1})   ← single-qubit rotations
  CNOT on all Kagome bonds            ← entanglers
```

- `depth * n_sites` parameters
- **Init:** `scale=1.0` (full random angles — at least breaks symmetry)
- Known limitation: gradient variance decays as $O(4^{-N/2})$ at random init (McClean et al. 2018)

---

## MERA — Multi-scale Entanglement Renormalization Ansatz (simplified)

A 2-scale architecture inspired by Vidal (2007). Uses parameterized two-qubit unitary blocks at two coarsening scales matching the Kagome geometry. Included as an architectural comparison — more parameters than needed for N=9 and harder to optimize.

### Circuit structure

```
Scale 1 (fine):   2-qubit blocks on all Kagome bonds      (4 params/bond)
Scale 2 (coarse): 2-qubit blocks on sublattice A-B pairs  (4 params/pair)
Final:            RY on all sites                          (1 param/site)
```

- 65 parameters for N=9
- **Init:** `scale=1.0`

---

## API

```python
from spinq_vqe import ansatz

# HVA — preferred
p0 = ansatz.init_params('hva', n_sites=9, depth=6, seed=42, scale=0.05)
ansatz.hva_ansatz(p0, n_sites=9, depth=6, edges=edges)
ansatz.hva_n_params(depth=6)   # → 18

# HEA
p0 = ansatz.init_params('hea', n_sites=9, depth=3, seed=42, scale=1.0)
ansatz.hea_ansatz(p0, n_sites=9, depth=3, edges=edges)
ansatz.hea_n_params(n_sites=9, depth=3)  # → 27

# MERA
p0 = ansatz.init_params('mera', n_sites=9, G=G9, seed=42, scale=1.0)
ansatz.mera_ansatz(p0, n_sites=9, G=G9)
ansatz.mera_n_params(G=G9)     # → 65
```

---

## References

- Kandala et al. (2017) Nature 549, 242 — HEA
- Wiersema et al. (2020) PRX Quantum 1, 020319 — HVA, barren plateau analysis
- Vidal (2007) PRL 99, 220405 — MERA
- McClean et al. (2018) Nat. Commun. 9, 4812 — barren plateaus
