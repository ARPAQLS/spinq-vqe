"""
ansatz.py
---------
Variational ansatze for VQE on the Kagome antiferromagnet.

Three ansatze are implemented:

1. **HEA** (Hardware-Efficient Ansatz)
   Alternating layers of single-qubit RY rotations and CNOT entanglers
   on Kagome nearest-neighbor bonds. General-purpose; suffers barren
   plateau gradients at random initialisation for large N.

2. **HVA** (Hamiltonian Variational Ansatz)  ← preferred for Heisenberg
   Trotterised evolution under the Kagome Hamiltonian terms (IsingXX,
   IsingYY, IsingZZ per bond, per layer). One (θ_XX, θ_YY, θ_ZZ) per
   layer, shared across all bonds. Provably avoids barren plateaus at
   small-angle initialisation for translation-invariant spin Hamiltonians.

3. **MERA** (Multi-scale Entanglement Renormalization Ansatz — simplified)
   Two-scale disentangler + isometry structure inspired by Vidal (2007).
   More parameters than needed; included as a comparison baseline.

References
----------
- Kandala et al. (2017) Nature 549, 242          — hardware-efficient ansatz
- Wiersema et al. (2020) PRX Quantum 1, 020319   — HVA, no barren plateau
- Vidal (2007) PRL 99, 220405                    — MERA
"""

from __future__ import annotations

from typing import Literal

import networkx as nx
import numpy as np
import pennylane as qp

# ---------------------------------------------------------------------------
# Hardware-Efficient Ansatz (HEA)
# ---------------------------------------------------------------------------


def hea_ansatz(
    params: list | np.ndarray,
    n_sites: int,
    depth: int,
    edges: list[tuple[int, int]],
) -> None:
    """
    Hardware-Efficient Ansatz circuit (applied in-place as a PennyLane circuit).

    Architecture per layer:
        1. Single-qubit RY(θ) on all sites
        2. CNOT entanglers on Kagome nearest-neighbor pairs

    Parameters
    ----------
    params : array-like, shape (depth * n_sites,)
        Flat array of rotation angles.
    n_sites : int
        Total number of qubits.
    depth : int
        Number of RY + CNOT layers.
    edges : list of (int, int)
        Kagome neighbor pairs for entanglers.

    Notes
    -----
    ``params`` is indexed as params[l * n_sites + i] → RY angle for
    qubit i in layer l.
    """
    params = np.asarray(params).reshape(depth, n_sites)

    for layer in range(depth):
        # Single-qubit rotations
        for i in range(n_sites):
            qp.RY(params[layer, i], wires=i)
        # Entanglers on Kagome bonds
        for i, j in edges:
            qp.CNOT(wires=[i, j])


def hea_n_params(n_sites: int, depth: int) -> int:
    """Return the number of trainable parameters for HEA."""
    return depth * n_sites


# ---------------------------------------------------------------------------
# MERA-Inspired Ansatz (simplified 2-scale)
# ---------------------------------------------------------------------------


def _two_qubit_unitary(params: np.ndarray, wires: list[int]) -> None:
    """
    Parameterized two-qubit unitary block.

    Implements: RY(a)⊗RY(b) → CNOT → RY(c)⊗RY(d) → CNOT(reversed)
    This is a universal 2-qubit gate approximation with 4 parameters.

    Parameters
    ----------
    params : array, shape (4,)
        [a, b, c, d] rotation angles.
    wires : list of int, length 2
        Target qubits [control, target].
    """
    a, b, c, d = params
    qp.RY(a, wires=wires[0])
    qp.RY(b, wires=wires[1])
    qp.CNOT(wires=wires)
    qp.RY(c, wires=wires[0])
    qp.RY(d, wires=wires[1])
    qp.CNOT(wires=[wires[1], wires[0]])


def mera_ansatz(
    params: list | np.ndarray,
    n_sites: int,
    G: nx.Graph,
) -> None:
    """
    Simplified 2-scale MERA-inspired ansatz.

    Structure:
        Scale 1 (fine): two-qubit disentanglers on all Kagome bonds
        Scale 2 (coarse): two-qubit isometries on sublattice pairs
        Final layer: single-qubit RY on all sites

    Each two-qubit block uses 4 parameters.

    Parameters
    ----------
    params : array-like
        Flat parameter array. Total count: :func:`mera_n_params`.
    n_sites : int
        Total number of qubits.
    G : nx.Graph
        Kagome lattice graph (provides bond structure for scale 1).

    Notes
    -----
    This is a *simplified* MERA — not a full hierarchical MERA with
    causal cone structure. It uses the same two-qubit block primitives
    but at two coarsening scales matching the Kagome geometry.
    """
    params = np.asarray(params)
    edges = list(G.edges())

    # --- Scale 1: fine-grained disentanglers on all Kagome bonds ---
    ptr = 0
    for i, j in edges:
        _two_qubit_unitary(params[ptr : ptr + 4], wires=[i, j])
        ptr += 4

    # --- Scale 2: coarse-grained isometries on sublattice A-B pairs ---
    # Group sites by sublattice
    sublattices: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for node, data in G.nodes(data=True):
        sublattices[data["sublattice"]].append(node)

    # Pair up A and B sublattice sites for scale-2 blocks
    min_len = min(len(sublattices[0]), len(sublattices[1]))
    for k in range(min_len):
        i = sublattices[0][k]
        j = sublattices[1][k]
        _two_qubit_unitary(params[ptr : ptr + 4], wires=[i, j])
        ptr += 4

    # --- Final layer: single-qubit rotations on all sites ---
    for i in range(n_sites):
        qp.RY(params[ptr], wires=i)
        ptr += 1


def mera_n_params(G: nx.Graph) -> int:
    """
    Return the number of trainable parameters for the MERA-inspired ansatz.

    Scale 1: 4 params × n_bonds
    Scale 2: 4 params × min(|sublattice_A|, |sublattice_B|)
    Final:   1 param  × n_sites
    """
    n_sites = G.number_of_nodes()
    n_bonds = G.number_of_edges()
    sublattice_sizes = {}
    for node, data in G.nodes(data=True):
        sl = data["sublattice"]
        sublattice_sizes[sl] = sublattice_sizes.get(sl, 0) + 1
    n_scale2 = min(sublattice_sizes.get(0, 0), sublattice_sizes.get(1, 0))
    return 4 * n_bonds + 4 * n_scale2 + n_sites


# ---------------------------------------------------------------------------
# Hamiltonian Variational Ansatz (HVA)  — preferred for Heisenberg models
# ---------------------------------------------------------------------------


def hva_ansatz(
    params: list | np.ndarray,
    n_sites: int,
    depth: int,
    edges: list[tuple[int, int]],
) -> None:
    """
    Hamiltonian Variational Ansatz for the Heisenberg model.

    Structure per layer::

        for each bond (i, j):
            IsingXX(2θ_XX, [i, j])   ≡  exp(-iθ_XX · X⊗X)
            IsingYY(2θ_YY, [i, j])   ≡  exp(-iθ_YY · Y⊗Y)
            IsingZZ(2θ_ZZ, [i, j])   ≡  exp(-iθ_ZZ · Z⊗Z)

    One (θ_XX, θ_YY, θ_ZZ) per layer, shared across all bonds.
    This is valid because the Kagome Hamiltonian has uniform exchange
    coupling J on all bonds.

    Parameters
    ----------
    params : array-like, shape (depth * 3,)
        Flat array of (theta_XX, theta_YY, theta_ZZ) per layer.
        Use small-angle initialisation (scale=0.05) — gradients are
        O(1) at θ≋0 for HVA, so no barren plateau.
    n_sites : int
        Number of qubits (unused in circuit but kept for API consistency).
    depth : int
        Number of Trotter layers.
    edges : list of (int, int)
        Kagome bond list.

    References
    ----------
    Wiersema et al. (2020) PRX Quantum 1, 020319
    """
    params = np.asarray(params).reshape(depth, 3)
    for layer in range(depth):
        t_xx, t_yy, t_zz = params[layer]
        for i, j in edges:
            qp.IsingXX(2.0 * t_xx, wires=[i, j])
            qp.IsingYY(2.0 * t_yy, wires=[i, j])
            qp.IsingZZ(2.0 * t_zz, wires=[i, j])


def hva_n_params(depth: int) -> int:
    """HVA: 3 parameters per layer (θ_XX, θ_YY, θ_ZZ), shared across all bonds."""
    return depth * 3


# ---------------------------------------------------------------------------
# Initial parameter factory
# ---------------------------------------------------------------------------


def init_params(
    ansatz: Literal["hea", "hva", "mera"],
    n_sites: int,
    G: nx.Graph | None = None,
    depth: int = 3,
    seed: int = 42,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Generate random initial parameters for a given ansatz.

    Parameters
    ----------
    ansatz : {"hea", "hva", "mera"}
    n_sites : int
    G : nx.Graph, required for "mera"
    depth : int
    seed : int
    scale : float
        Multiplier on the uniform range [-π, π].
        For HVA use scale=0.05 (small-angle init — no barren plateau).
        For HEA use scale=1.0 (random).

    Returns
    -------
    np.ndarray
        Flat array of initial parameters in [-π·scale, π·scale].
    """
    rng = np.random.default_rng(seed)
    if ansatz == "hea":
        n = hea_n_params(n_sites, depth)
    elif ansatz == "hva":
        n = hva_n_params(depth)
    elif ansatz == "mera":
        if G is None:
            raise ValueError("G (Kagome graph) is required for MERA ansatz.")
        n = mera_n_params(G)
    else:
        raise ValueError(f"Unknown ansatz: {ansatz!r}. Choose 'hea', 'hva', or 'mera'.")
    return rng.uniform(-np.pi * scale, np.pi * scale, size=n)

