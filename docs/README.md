# spinq-vqe — Documentation

> ARPA Quantum Logical Systems (QONDRA) &nbsp;·&nbsp; [qondra@arpacorp.net](mailto:qondra@arpacorp.net)

---

## Contents

| Document | What it covers |
|----------|---------------|
| [Physics background](physics.md) | Kagome AFM, Mn₃Sn, frustration, the Hamiltonian |
| [Ansatz guide](ansatze.md) | HVA, HEA, MERA — what they are, which to use, why |
| [API reference](api.md) | Module-level reference for `spinq_vqe` |
| [Notebook guide](notebooks.md) | What each notebook does, how to run, expected outputs |

---

## Quick orientation

**The physics:** Mn₃Sn is a Kagome antiferromagnet. Its ground state is geometrically frustrated — no classical solution exists. VQE uses a parameterized quantum circuit to approximate the ground state variationally.

**The stack:** PennyLane handles circuits, operators, and optimization. NetworkX builds the Kagome graph. NumPy/SciPy handle numerics. All simulation runs on CPU via `default.qubit`. scikit-learn provides the B2 surrogate MLP.

**The optimizer:** Use **COBYLA** (`run_vqe_cobyla`) for this system — not Adam. The `|0⟩⊗N` initial state is a Z-basis eigenstate where all Ising-gate gradients cancel by SU(2) symmetry. COBYLA samples the energy directly and is immune to this zero-gradient problem.

**The ansatz:** HEA depth=3 (27 parameters) with COBYLA achieves **9.66% error** from exact diagonalization on the 9-site Kagome. HVA is available for physics-motivated experiments but showed zero gradient from `|0⟩⊗N`.

**Current results (NB01–03 complete):**
- ED ground state: E₀ = −1.42190399 (N=9)
- COBYLA/HEA: E₀ = −1.28456 (9.66% error, 801 evaluations)
- Mean single-site entropy: 0.9066 bits (near-maximal spin liquid signature)
- Sublattice MI I(A:B): 3.689 bits

---

*Part of the ARPA Spintronics QML Research Program*
