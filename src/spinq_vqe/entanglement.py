"""
entanglement.py
---------------
Entanglement entropy analysis from VQE ground-state wavefunctions.

Computes:
1. Von Neumann entropy S(ρ_A) = -Tr(ρ_A log ρ_A) for bipartitions of the Kagome lattice
2. Mutual information I(A:B) = S(A) + S(B) - S(AB) between sublattices

Both are computed directly from the VQE statevector — no additional circuit
measurements needed. This is exact in simulation.

References
----------
- Nielsen & Chuang (2010) — Quantum Computation and Quantum Information, Ch. 11
- Consiglio et al. (2022) PRR 4, 033257 — Entanglement in Kagome VQE
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

# ---------------------------------------------------------------------------
# Core: reduced density matrix and Von Neumann entropy
# ---------------------------------------------------------------------------


def reduced_density_matrix(
    statevector: np.ndarray,
    subsystem: list[int],
    n_sites: int,
) -> np.ndarray:
    """
    Compute the reduced density matrix ρ_A by tracing out the complement.

    Parameters
    ----------
    statevector : np.ndarray, shape (2**n_sites,)
        Full statevector |ψ⟩ from VQE.
    subsystem : list of int
        Qubit indices in subsystem A.
    n_sites : int
        Total number of qubits.

    Returns
    -------
    np.ndarray, shape (2**|A|, 2**|A|)
        Reduced density matrix ρ_A = Tr_B(|ψ⟩⟨ψ|).

    Notes
    -----
    Uses tensor reshaping: reshape |ψ⟩ as a tensor of shape (2, 2, ..., 2),
    permute subsystem A to the front, then trace over B indices.
    """
    n_A = len(subsystem)
    n_B = n_sites - n_A
    complement = [i for i in range(n_sites) if i not in subsystem]

    # Reshape into (2, 2, ..., 2) tensor — qubit ordering: site 0 = MSB
    psi = statevector.reshape([2] * n_sites)

    # Permute: subsystem A first, then B
    perm = list(subsystem) + complement
    psi = np.transpose(psi, perm)

    # Reshape to (dim_A, dim_B)
    dim_A = 2**n_A
    dim_B = 2**n_B
    psi = psi.reshape(dim_A, dim_B)

    # ρ_A = Tr_B(|ψ⟩⟨ψ|) = ψ @ ψ†
    rho_A = psi @ psi.conj().T
    return rho_A


def von_neumann_entropy(rho: np.ndarray, base: float = 2.0) -> float:
    """
    Compute the von Neumann entropy S(ρ) = -Tr(ρ log ρ).

    Parameters
    ----------
    rho : np.ndarray
        Density matrix (must be Hermitian, positive semidefinite).
    base : float
        Logarithm base. Default: 2 (entropy in bits). Use np.e for nats.

    Returns
    -------
    float
        Von Neumann entropy S(ρ) ≥ 0.

    Notes
    -----
    Uses eigendecomposition. Eigenvalues λ < threshold are treated as 0
    to avoid log(0) issues from numerical noise.
    """
    eigenvalues = np.linalg.eigvalsh(rho)
    # Clip negative numerical noise
    eigenvalues = np.clip(eigenvalues.real, 0, None)
    # Filter near-zero eigenvalues
    mask = eigenvalues > 1e-12
    if not np.any(mask):
        return 0.0
    lam = eigenvalues[mask]
    entropy = -np.sum(lam * np.log(lam) / np.log(base))
    return float(entropy)


# ---------------------------------------------------------------------------
# Mutual information
# ---------------------------------------------------------------------------


def mutual_information(
    statevector: np.ndarray,
    subsystem_A: list[int],
    subsystem_B: list[int],
    n_sites: int,
    base: float = 2.0,
) -> float:
    """
    Compute mutual information I(A:B) = S(A) + S(B) - S(AB).

    Parameters
    ----------
    statevector : np.ndarray
        Full VQE statevector.
    subsystem_A, subsystem_B : list of int
        Qubit indices for subsystems A and B. Must be disjoint.
    n_sites : int
        Total number of qubits.
    base : float
        Logarithm base.

    Returns
    -------
    float
        Mutual information I(A:B) ≥ 0.

    Raises
    ------
    ValueError
        If subsystem_A and subsystem_B overlap.
    """
    if set(subsystem_A) & set(subsystem_B):
        raise ValueError("subsystem_A and subsystem_B must be disjoint.")

    subsystem_AB = list(subsystem_A) + list(subsystem_B)

    rho_A = reduced_density_matrix(statevector, subsystem_A, n_sites)
    rho_B = reduced_density_matrix(statevector, subsystem_B, n_sites)
    rho_AB = reduced_density_matrix(statevector, subsystem_AB, n_sites)

    S_A = von_neumann_entropy(rho_A, base=base)
    S_B = von_neumann_entropy(rho_B, base=base)
    S_AB = von_neumann_entropy(rho_AB, base=base)

    return float(S_A + S_B - S_AB)


# ---------------------------------------------------------------------------
# Kagome-specific: bipartition scan and sublattice mutual info matrix
# ---------------------------------------------------------------------------


def entanglement_profile(
    statevector: np.ndarray,
    n_sites: int,
    max_subsystem_size: int | None = None,
    base: float = 2.0,
) -> dict:
    """
    Scan all contiguous bipartitions of the Kagome chain and compute S_vN.

    For a chain of n_sites, considers bipartitions A = {0, 1, ..., k-1}
    for k = 1 to n_sites-1.

    Parameters
    ----------
    statevector : np.ndarray
    n_sites : int
    max_subsystem_size : int or None
        Maximum subsystem size to scan. Defaults to n_sites // 2.
    base : float

    Returns
    -------
    dict with keys:
        "subsystem_sizes" : list of int
        "entropies"       : list of float
        "max_entropy"     : float
        "max_at_size"     : int
    """
    if max_subsystem_size is None:
        max_subsystem_size = n_sites // 2

    sizes = list(range(1, max_subsystem_size + 1))
    entropies = []

    for k in sizes:
        subsystem = list(range(k))
        rho = reduced_density_matrix(statevector, subsystem, n_sites)
        S = von_neumann_entropy(rho, base=base)
        entropies.append(S)

    max_idx = int(np.argmax(entropies))
    return {
        "subsystem_sizes": sizes,
        "entropies": entropies,
        "max_entropy": entropies[max_idx],
        "max_at_size": sizes[max_idx],
    }


def sublattice_mutual_info_matrix(
    statevector: np.ndarray,
    sublattices: dict[int, list[int]],
    n_sites: int,
    base: float = 2.0,
) -> np.ndarray:
    """
    Compute the 3×3 mutual information matrix between Kagome sublattices.

    Parameters
    ----------
    statevector : np.ndarray
    sublattices : dict {0: [sites_A], 1: [sites_B], 2: [sites_C]}
        From :func:`spinq_vqe.kagome.sublattice_partition`.
    n_sites : int
    base : float

    Returns
    -------
    np.ndarray, shape (3, 3)
        Symmetric matrix where entry [i, j] = I(sublattice_i : sublattice_j).
        Diagonal entries are 0 by convention.
    """
    n_sl = len(sublattices)
    matrix = np.zeros((n_sl, n_sl))

    for i, j in combinations(range(n_sl), 2):
        I_ij = mutual_information(
            statevector,
            sublattice_A=sublattices[i],
            subsystem_B=sublattices[j],
            n_sites=n_sites,
            base=base,
        )
        matrix[i, j] = I_ij
        matrix[j, i] = I_ij  # symmetric

    return matrix
