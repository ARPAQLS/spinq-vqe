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

**The stack:** PennyLane handles the circuits, operators, and optimizer. NetworkX builds the Kagome graph. NumPy/SciPy handle numerics. All simulation runs on CPU via `default.qubit`.

**The ansatz choice:** Use **HVA** for this system. It is built from the Hamiltonian terms directly, has $O(1)$ gradients at small angles, and is the correct choice for uniform Heisenberg models. HEA is included as a comparison baseline.

---

*Part of the ARPA Spintronics QML Research Program*
