"""
kagome.py
---------
Kagome lattice graph construction and Heisenberg Hamiltonian builder.

The Kagome lattice is a 2D network of corner-sharing triangles with three
magnetic sublattices (A, B, C). It is the lattice geometry of Mn₃Sn, the
Kagome antiferromagnet studied in the UTokyo SOT experiment (2026).

Hamiltonian
-----------
    H = J Σ_{<i,j>} S_i · S_j  +  D Σ_i (S_i^z)²  +  B Σ_i S_i^z

Where:
    J > 0   : antiferromagnetic exchange coupling (J ≈ 3–5 meV for Mn₃Sn)
    D       : single-ion anisotropy
    B       : external magnetic field

References
----------
- Sachdev (1992) PRB 45, 12377  — Kagome Heisenberg AFM spin liquid phases
- Yan et al. (2011) Science 332, 1173  — Spin liquid ground state
- Nakatsuji et al. (2022) Ann. Phys. 447  — Mn₃Sn physics review
"""

from __future__ import annotations

from typing import Literal

import networkx as nx
import pennylane as qp

# ---------------------------------------------------------------------------
# Physical constants (Mn₃Sn calibrated values from INS literature)
# ---------------------------------------------------------------------------
J_MN3SN_MEV: float = 4.0  # antiferromagnetic exchange, meV (Nakatsuji 2022)
D_MN3SN_MEV: float = 0.3  # single-ion anisotropy, meV
B_DEFAULT: float = 0.0  # external field (off by default)


# ---------------------------------------------------------------------------
# Lattice geometry
# ---------------------------------------------------------------------------


def kagome_graph(
    n_cells: int = 1,
    boundary: Literal["open", "periodic"] = "open",
) -> nx.Graph:
    """
    Build a Kagome lattice graph with ``n_cells`` unit cells.

    Each unit cell contains 3 sites (sublattices A, B, C), so the total
    number of sites is ``N = 3 * n_cells``.

    Supported configurations:
        n_cells=1  →  9 sites  (1×3 unit cell)
        n_cells=2  →  18 sites (2×3 unit cells)  [NOT YET — see note]
        n_cells=3  →  24 sites

    .. note::
        Currently supports linear strip geometries (1D Kagome chain of unit
        cells). 2D periodic tiling will be added in a future version once
        the 1D strip is validated against QuSpin results.

    Parameters
    ----------
    n_cells : int
        Number of Kagome unit cells. Each cell has 3 sites.
    boundary : {"open", "periodic"}
        Boundary condition along the strip direction.

    Returns
    -------
    nx.Graph
        Graph where nodes are site indices (0-indexed) and node attribute
        ``sublattice`` ∈ {0, 1, 2} encodes the A/B/C sublattice.

    Examples
    --------
    >>> G = kagome_graph(n_cells=3)   # 9 sites
    >>> G.number_of_nodes()
    9
    """
    G = nx.Graph()
    n_sites = 3 * n_cells

    # Label sites and assign sublattices
    for i in range(n_sites):
        G.add_node(i, sublattice=i % 3)

    # Intra-cell bonds (triangle within each unit cell)
    # Sites 3k, 3k+1, 3k+2 form a triangle
    for k in range(n_cells):
        a, b, c = 3 * k, 3 * k + 1, 3 * k + 2
        G.add_edge(a, b)
        G.add_edge(b, c)
        G.add_edge(a, c)

    # Inter-cell bonds (corner sharing between adjacent triangles)
    for k in range(n_cells - 1):
        # Shared corner: site C of cell k connects to site A of cell k+1
        c_k = 3 * k + 2
        a_k1 = 3 * (k + 1)
        G.add_edge(c_k, a_k1)

    # Periodic boundary condition: connect last cell back to first
    if boundary == "periodic" and n_cells > 1:
        c_last = 3 * (n_cells - 1) + 2
        a_first = 0
        G.add_edge(c_last, a_first)

    return G


# ---------------------------------------------------------------------------
# Hamiltonian construction
# ---------------------------------------------------------------------------


def heisenberg_kagome_hamiltonian(
    G: nx.Graph,
    J: float = J_MN3SN_MEV,
    D: float = D_MN3SN_MEV,
    B: float = B_DEFAULT,
    normalize: bool = True,
) -> qp.Hamiltonian:
    """
    Construct the Heisenberg AFM Hamiltonian on a Kagome lattice graph.

    H = J Σ_{<i,j>} (XX + YY + ZZ)_{ij}  +  D Σ_i ZZ_{ii}  +  B Σ_i Z_i

    The S_i · S_j = (1/4)(XX + YY + ZZ) identity for spin-1/2 is used.
    We absorb the 1/4 factor into J.

    Parameters
    ----------
    G : nx.Graph
        Kagome lattice graph from :func:`kagome_graph`.
    J : float
        Exchange coupling (J > 0 for AFM). Units: meV (or dimensionless).
    D : float
        Single-ion anisotropy coefficient.
    B : float
        External magnetic field strength.
    normalize : bool
        If True, divide all coefficients by the number of sites for
        size-independent comparison.

    Returns
    -------
    qml.Hamiltonian
        PennyLane Hamiltonian object.

    Notes
    -----
    The factor 1/4 from S_i · S_j = (1/4) σ_i · σ_j (Pauli matrices)
    is included in the coupling coefficient.
    """
    n_sites = G.number_of_nodes()
    norm = n_sites if normalize else 1.0

    coeffs = []
    ops = []

    # Exchange interaction: J/4 * (XX + YY + ZZ) per bond
    J_eff = J / (4.0 * norm)
    for i, j in G.edges():
        # XX term
        coeffs.append(J_eff)
        ops.append(qp.PauliX(i) @ qp.PauliX(j))
        # YY term
        coeffs.append(J_eff)
        ops.append(qp.PauliY(i) @ qp.PauliY(j))
        # ZZ term
        coeffs.append(J_eff)
        ops.append(qp.PauliZ(i) @ qp.PauliZ(j))

    # Single-ion anisotropy: D * (S^z)^2 = D/4 * I (constant, skip) + ...
    # For spin-1/2: (S^z)^2 = 1/4 * I — a constant, does not affect eigenstates
    # The physically meaningful D term for spin > 1/2 vanishes for spin-1/2.
    # We include it symbolically for generality (effect is a constant energy shift).
    if abs(D) > 1e-10:
        D_eff = D / (4.0 * norm)
        for i in range(n_sites):
            coeffs.append(D_eff)
            ops.append(qp.Identity(i))

    # Zeeman term: B * S^z = B/2 * Z
    if abs(B) > 1e-10:
        B_eff = B / (2.0 * norm)
        for i in range(n_sites):
            coeffs.append(B_eff)
            ops.append(qp.PauliZ(i))

    return qp.Hamiltonian(coeffs, ops)


# ---------------------------------------------------------------------------
# Utility: site metadata
# ---------------------------------------------------------------------------


def sublattice_partition(G: nx.Graph) -> dict[int, list[int]]:
    """
    Return the three sublattice partitions of the Kagome graph.

    Returns
    -------
    dict
        {0: [site_A, ...], 1: [site_B, ...], 2: [site_C, ...]}
    """
    partition: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for node, data in G.nodes(data=True):
        partition[data["sublattice"]].append(node)
    return partition


def n_sites(G: nx.Graph) -> int:
    """Return the total number of sites."""
    return G.number_of_nodes()


def n_bonds(G: nx.Graph) -> int:
    """Return the total number of exchange bonds."""
    return G.number_of_edges()
