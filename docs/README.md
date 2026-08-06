# spinq-vqe — Documentation

> ARPA Quantum Logical Systems (QONDRA) &nbsp;·&nbsp; [qondra@arpacorp.net](mailto:qondra@arpacorp.net)

---

## Contents

| Document | What it covers |
|----------|---------------|
| [Research overview](../OVERVIEW.md) | Narrative, key results, scientific context, roadmap |
| [Physics background](physics.md) | Kagome AFM, Mn₃Sn, frustration, the Hamiltonian |
| [Ansatz guide](ansatze.md) | HVA, HEA, MERA — what they are, which to use, why |
| [API reference](api.md) | Module-level reference for `spinq_vqe` |
| [Notebook guide](notebooks.md) | What each notebook does, how to run, expected outputs |
| [Testing guide](testing.md) | Running tests, coverage map, fixtures, extending |

---

## Quick orientation

**The physics:** Mn₃Sn is a Kagome antiferromagnet. Its ground state is geometrically frustrated — no classical solution exists. VQE uses a parameterized quantum circuit to approximate the ground state variationally.

**The stack:** PennyLane handles circuits, operators, and optimization. NetworkX builds the Kagome graph. NumPy/SciPy handle numerics. All simulation runs on CPU via `default.qubit`. scikit-learn provides the SOC surrogate MLP.

**The optimizer:** Use **COBYLA** (`run_vqe_cobyla`) for this system — not Adam. The `|0⟩⊗N` initial state is a Z-basis eigenstate where all Ising-gate gradients cancel by SU(2) symmetry. COBYLA samples the energy directly and is immune to this zero-gradient problem.

**The ansatz:** HEA depth=3 (27 parameters) with COBYLA achieves **9.66% error vs DMRG** (best seed)
on the 9-site Kagome strip, with mean E₀ = −1.23572 ± 0.02853 across
5 random initializations. HVA is available for physics-motivated experiments but showed zero
gradient from `|0⟩⊗N`.

**Current results (NB01–NB06):**
- DMRG reference (TeNPy): E₀ = −1.4219 (N=9), −1.4804 (N=12), −1.4996 (N=18), −1.5094 (N=24)
- COBYLA/HEA (N=9): best E₀ = −1.28456 (9.66% vs DMRG); mean ± std = −1.23572 ± 0.02853 (5 seeds)
- Mean single-site entropy: 0.9066 bits (near-maximal spin liquid signature)
- Sublattice MI I(A:B): 3.689 bits
- SOC QAOA: greedy/SA reach oracle optimum (θ_SH ≈ 4.26); best QAOA is p=1/2 at 3.05
- Scaling (NB05): N=12 VQE best error 16.33% vs DMRG; gradient variance barren plateau confirmed

---

*Part of the ARPA Spintronics QML Research Program*
