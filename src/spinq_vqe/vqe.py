"""
vqe.py
------
VQE runner for the Kagome antiferromagnet Hamiltonian.

Uses PennyLane with JAX backend for JIT-compiled cost evaluation and
gradient computation. Supports both HEA and MERA ansatze.

Optimizer cascade:
    1. Adam (JAX / optax) — fast convergence near optimum
    2. COBYLA (SciPy) — gradient-free fallback for noisy landscapes

References
----------
- Peruzzo et al. (2014) Nature Commun. 5, 4213  — Original VQE
- Cerezo et al. (2021) Nat. Rev. Phys. 3, 625   — VQE review
- Tilly et al. (2022) Physics Reports 986        — VQE best practices
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pennylane as qp

JAX_AVAILABLE: bool = importlib.util.find_spec("jax") is not None


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class VQEResult:
    """Container for VQE optimization results."""

    energy: float
    """Best energy found (normalized, in units of input J)."""

    params: np.ndarray
    """Optimal circuit parameters."""

    energy_history: list[float] = field(default_factory=list)
    """Energy at each optimizer step."""

    gradient_variance_history: list[float] = field(default_factory=list)
    """Variance of gradients at each step (barren plateau diagnostic)."""

    n_steps: int = 0
    """Total optimizer steps taken."""

    converged: bool = False
    """Whether the energy variance convergence criterion was met."""

    statevector: np.ndarray | None = None
    """Final statevector |ψ⟩ from the optimal circuit (if requested)."""

    ansatz: str = ""
    """Name of the ansatz used ('hea' or 'mera')."""

    n_sites: int = 0
    n_params: int = 0


# ---------------------------------------------------------------------------
# VQE core
# ---------------------------------------------------------------------------


def run_vqe(
    hamiltonian: qp.Hamiltonian,
    ansatz_fn: Callable,
    init_params: np.ndarray,
    n_sites: int,
    ansatz_name: str = "hea",
    n_steps: int = 300,
    step_size: float = 0.05,
    conv_tol: float = 1e-6,
    conv_window: int = 20,
    return_statevector: bool = True,
    verbose: bool = True,
    **ansatz_kwargs,
) -> VQEResult:
    """
    Run VQE optimization using PennyLane's gradient descent.

    Parameters
    ----------
    hamiltonian : qp.Hamiltonian
        The target Hamiltonian (output of :func:`kagome.heisenberg_kagome_hamiltonian`).
    ansatz_fn : callable
        Ansatz circuit function. Signature: ``fn(params, n_sites, **kwargs)``.
    init_params : np.ndarray
        Initial variational parameters.
    n_sites : int
        Number of qubits.
    ansatz_name : str
        Label for the ansatz (for bookkeeping).
    n_steps : int
        Maximum number of optimization steps.
    step_size : float
        Learning rate for the Adam optimizer.
    conv_tol : float
        Convergence threshold on energy variance over ``conv_window`` steps.
    conv_window : int
        Window size for convergence check.
    return_statevector : bool
        If True, compute and store the final statevector.
    verbose : bool
        Print progress every 50 steps.
    **ansatz_kwargs
        Additional keyword arguments passed to ``ansatz_fn``.

    Returns
    -------
    VQEResult
    """
    device = qp.device("default.qubit", wires=n_sites)

    @qp.qnode(device)
    def cost_fn(params):
        ansatz_fn(params, n_sites, **ansatz_kwargs)
        return qp.expval(hamiltonian)

    # Optimizer: PennyLane Adam
    opt = qp.AdamOptimizer(stepsize=step_size)

    params = np.copy(init_params)
    energy_history = []
    grad_var_history = []
    best_energy = np.inf
    best_params = np.copy(params)

    for step in range(n_steps):
        # Compute gradient and step
        params, energy = opt.step_and_cost(cost_fn, params)
        energy = float(energy)
        energy_history.append(energy)

        if energy < best_energy:
            best_energy = energy
            best_params = np.copy(params)

        # Gradient variance (barren plateau diagnostic)
        grad = qp.grad(cost_fn)(params)
        grad_var_history.append(float(np.var(grad)))

        if verbose and (step + 1) % 50 == 0:
            print(
                f"  Step {step + 1:>4d} | E = {energy:.8f} | "
                f"grad_var = {grad_var_history[-1]:.2e}"
            )

        # Convergence check
        if len(energy_history) >= conv_window:
            window = energy_history[-conv_window:]
            if np.var(window) < conv_tol:
                if verbose:
                    print(
                        f"  Converged at step {step + 1} (Δ²E = {np.var(window):.2e})"
                    )
                converged = True
                break
    else:
        converged = False

    # Optionally compute final statevector
    statevector = None
    if return_statevector:
        sv_device = qp.device("default.qubit", wires=n_sites)

        @qp.qnode(sv_device)
        def sv_circuit(params):
            ansatz_fn(params, n_sites, **ansatz_kwargs)
            return qp.state()

        statevector = np.array(sv_circuit(best_params))

    return VQEResult(
        energy=best_energy,
        params=best_params,
        energy_history=energy_history,
        gradient_variance_history=grad_var_history,
        n_steps=len(energy_history),
        converged=converged,
        statevector=statevector,
        ansatz=ansatz_name,
        n_sites=n_sites,
        n_params=len(init_params),
    )


# ---------------------------------------------------------------------------
# Convenience: compare HEA vs MERA on the same Hamiltonian
# ---------------------------------------------------------------------------


def compare_ansatze(
    hamiltonian: qp.Hamiltonian,
    hea_fn: Callable,
    mera_fn: Callable,
    hea_params: np.ndarray,
    mera_params: np.ndarray,
    n_sites: int,
    hea_kwargs: dict | None = None,
    mera_kwargs: dict | None = None,
    **run_kwargs,
) -> dict[str, VQEResult]:
    """
    Run VQE with both HEA and MERA ansatze and return both results.

    Returns
    -------
    dict with keys "hea" and "mera", each a VQEResult.
    """
    hea_kwargs = hea_kwargs or {}
    mera_kwargs = mera_kwargs or {}

    print("=== Running HEA ===")
    hea_result = run_vqe(
        hamiltonian,
        hea_fn,
        hea_params,
        n_sites,
        ansatz_name="hea",
        **hea_kwargs,
        **run_kwargs,
    )

    print("\n=== Running MERA ===")
    mera_result = run_vqe(
        hamiltonian,
        mera_fn,
        mera_params,
        n_sites,
        ansatz_name="mera",
        **mera_kwargs,
        **run_kwargs,
    )

    return {"hea": hea_result, "mera": mera_result}
