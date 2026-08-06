"""
dmrg.py
-------
Density Matrix Renormalization Group (DMRG) reference energies for Kagome strips.

Uses TeNPy to build the same normalized Heisenberg Hamiltonian as
:func:`kagome.heisenberg_kagome_hamiltonian`, with one ``add_coupling_term`` per
graph edge (required for the non-nearest-neighbour Kagome strip).

References
----------
- Yan, Huse, White (2011) Science 332 — DMRG spin liquid on Kagome
- Depenbrock, McCulloch, Schollwöck (2012) PRL 109 — Kagome spin liquid
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from spinq_vqe.kagome import (
    B_DEFAULT,
    D_MN3SN_MEV,
    J_MN3SN_MEV,
    heisenberg_kagome_hamiltonian,
    kagome_graph,
)

try:
    from tenpy.algorithms import dmrg as tenpy_dmrg
    from tenpy.algorithms.exact_diag import get_numpy_Hamiltonian
    from tenpy.models.lattice import Chain
    from tenpy.models.model import CouplingMPOModel
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite

    TENPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    TENPY_AVAILABLE = False

InitialState = Literal["alternating", "up"]


@dataclass
class DMRGResult:
    """Ground-state DMRG result for one Kagome strip size."""

    n_cells: int
    n_sites: int
    e0: float
    chi_max: int
    chi: int
    sweeps: int
    truncation_error: float
    entropies: list[float] = field(default_factory=list)
    # Bond-cut von Neumann entropies in bits (log2), if computed.


class KagomeStripDMRG(CouplingMPOModel):
    """TeNPy MPO matching ``heisenberg_kagome_hamiltonian`` on a 1D strip."""

    def init_lattice(self, model_params):
        n_cells = int(model_params["n_cells"])
        site = SpinHalfSite(conserve=None)
        return Chain(L=3 * n_cells, site=site, bc="open", bc_MPS="finite")

    def init_sites(self, model_params):
        return SpinHalfSite(conserve=None)

    def init_terms(self, model_params):
        n_cells = int(model_params["n_cells"])
        j_val = float(model_params.get("J", J_MN3SN_MEV))
        d_val = float(model_params.get("D", D_MN3SN_MEV))
        graph = kagome_graph(n_cells)
        n_sites = graph.number_of_nodes()
        j_coupling = j_val / (4.0 * n_sites)

        for i, j in graph.edges():
            for op in ("Sigmax", "Sigmay", "Sigmaz"):
                self.add_coupling_term(j_coupling, i, j, op, op)

        if abs(d_val) > 1e-12:
            d_onsite = d_val / (4.0 * n_sites)
            for site in range(n_sites):
                self.add_onsite_term(d_onsite, site, "Id")


def require_tenpy() -> None:
    if not TENPY_AVAILABLE:
        raise ImportError(
            "physics-tenpy is required for DMRG. Install with: pip install -e '.[dmrg]'"
        )


def _initial_product_state(n_sites: int, mode: InitialState) -> list[list[str]]:
    if mode == "up":
        return [["up"] for _ in range(n_sites)]
    return [["up" if i % 2 == 0 else "down"] for i in range(n_sites)]


def build_kagome_dmrg_model(
    n_cells: int,
    *,
    J: float = J_MN3SN_MEV,
    D: float = D_MN3SN_MEV,
) -> KagomeStripDMRG:
    """Construct the TeNPy model for ``n_cells`` Kagome unit cells."""
    require_tenpy()
    return KagomeStripDMRG({"n_cells": n_cells, "J": J, "D": D})


def validate_hamiltonian_against_pennylane(
    n_cells: int,
    *,
    J: float = J_MN3SN_MEV,
    D: float = D_MN3SN_MEV,
    B: float = B_DEFAULT,
    atol: float = 1e-10,
) -> float:
    """
    Return max |H_tenpy - H_pennylane| for sanity-checking the MPO builder.

    Raises ``AssertionError`` when the matrices differ beyond ``atol``.
    """
    require_tenpy()
    import pennylane as qp

    graph = kagome_graph(n_cells)
    n_sites = graph.number_of_nodes()
    h_pl = np.array(
        qp.matrix(
            heisenberg_kagome_hamiltonian(graph, J=J, D=D, B=B),
            wire_order=list(range(n_sites)),
        )
    )
    h_ten = get_numpy_Hamiltonian(build_kagome_dmrg_model(n_cells, J=J, D=D))
    max_diff = float(np.max(np.abs(h_pl - h_ten)))
    if max_diff > atol:
        raise AssertionError(
            f"TeNPy/PennyLane Hamiltonian mismatch at N={n_sites}: max diff {max_diff:.3e}"
        )
    return max_diff


def run_dmrg(
    n_cells: int,
    *,
    J: float = J_MN3SN_MEV,
    D: float = D_MN3SN_MEV,
    chi_max: int = 400,
    max_sweeps: int = 80,
    initial_state: InitialState = "alternating",
    compute_entropies: bool = False,
) -> DMRGResult:
    """
    Run finite-chain DMRG for a Kagome strip with ``n_cells`` unit cells.

    Returns the normalized ground-state energy matching ``E0_normalized`` in
    ``data/ed_reference_energies.csv``.
    """
    require_tenpy()
    model = build_kagome_dmrg_model(n_cells, J=J, D=D)
    n_sites = 3 * n_cells
    psi = MPS.from_lat_product_state(
        model.lat, _initial_product_state(n_sites, initial_state)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = tenpy_dmrg.run(
            psi,
            model,
            {
                "trunc_params": {"chi_max": chi_max},
                "max_sweeps": max_sweeps,
                "combine": True,
            },
        )
    e0 = float(model.H_MPO.expectation_value(psi))
    entropies: list[float] = []
    if compute_entropies and n_sites > 1:
        # TeNPy returns von Neumann entropy in nats; convert to bits (log2)
        # to match spinq_vqe.entanglement / NB03.
        bond_cuts = list(range(1, n_sites))
        ln2 = float(np.log(2.0))
        entropies = [
            float(s) / ln2
            for s in psi.entanglement_entropy(bonds=bond_cuts)
        ]
    return DMRGResult(
        n_cells=n_cells,
        n_sites=n_sites,
        e0=e0,
        chi_max=chi_max,
        chi=int(max(psi.chi)),
        sweeps=int(info.get("sweeps", max_sweeps)),
        truncation_error=float(info.get("max_trunc_err", 0.0)),
        entropies=entropies,
    )


def run_dmrg_chi_sweep(
    n_cells: int,
    chi_values: list[int],
    **kwargs,
) -> list[DMRGResult]:
    """Run DMRG at increasing ``chi_max`` to certify convergence."""
    return [run_dmrg(n_cells, chi_max=chi, **kwargs) for chi in chi_values]


def save_dmrg_reference_csv(
    results: list[DMRGResult],
    path: Path | str,
) -> Path:
    """Write ``data/dmrg_reference_energies.csv``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "n_sites",
                "n_cells",
                "E0_normalized",
                "chi",
                "chi_max",
                "sweeps",
                "truncation_error",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "n_sites": row.n_sites,
                    "n_cells": row.n_cells,
                    "E0_normalized": f"{row.e0:.10f}",
                    "chi": row.chi,
                    "chi_max": row.chi_max,
                    "sweeps": row.sweeps,
                    "truncation_error": f"{row.truncation_error:.3e}",
                }
            )
    return path


def load_dmrg_reference_csv(path: Path | str) -> dict[int, float]:
    """Load DMRG reference energies keyed by ``n_sites``."""
    path = Path(path)
    out: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[int(row["n_sites"])] = float(row["E0_normalized"])
    return out
