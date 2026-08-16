"""
vqe.py
------
VQE runner for the Kagome antiferromagnet Hamiltonian.

Two optimizers are provided:

1. ``run_vqe``        — PennyLane Adam (gradient-based).
   Use for: gradient variance diagnostics, convergence curves, warm-starting.
   Known limitation: Adam stalls on the Heisenberg AFM because the |0⟩⊗N
   initial state is a Z-basis eigenstate — all IsingXX/YY/ZZ gradients cancel
   to zero by SU(2) symmetry, leaving the optimizer without a usable signal.

2. ``run_vqe_cobyla`` — SciPy COBYLA (gradient-free).  ← primary for this system
   Use for: actual ground-state search on the Kagome AFM.
   COBYLA builds a local linear model from function evaluations; it does not
   need gradients and naturally escapes the local minima where Adam stalls.

References
----------
- Peruzzo et al. (2014) Nature Commun. 5, 4213  — original VQE
- Cerezo et al. (2021) Nat. Rev. Phys. 3, 625   — VQE review
- Tilly et al. (2022) Physics Reports 986        — VQE best practices
- Powell (1994) — COBYLA algorithm
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pennylane as qp
from scipy.optimize import minimize

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
    """Energy at each optimizer step / function evaluation."""

    gradient_variance_history: list[float] = field(default_factory=list)
    """Variance of gradients at each step (barren plateau diagnostic).
    Empty for gradient-free optimizers."""

    n_steps: int = 0
    """Total optimizer steps / function evaluations."""

    converged: bool = False
    """Whether the optimizer reported convergence."""

    statevector: np.ndarray | None = None
    """Final statevector |ψ⟩ from the optimal circuit (if requested)."""

    ansatz: str = ""
    """Name of the ansatz used."""

    optimizer: str = ""
    """Optimizer used: 'adam' or 'cobyla'."""

    n_sites: int = 0
    n_params: int = 0


@dataclass
class SeedStatistics:
    """Aggregate statistics over multiple random VQE initializations."""

    mean_energy: float
    std_energy: float
    min_energy: float
    max_energy: float
    n_seeds: int


@dataclass
class VQEMultiSeedResult:
    """Best COBYLA run plus seed-level statistics and per-seed histories."""

    best: VQEResult
    statistics: SeedStatistics
    runs: list[VQEResult]
    seeds: list[int]


def seed_statistics(energies: Sequence[float]) -> SeedStatistics:
    """
    Compute mean, sample std, min, and max over final energies from multiple seeds.

    Parameters
    ----------
    energies : sequence of float
        Final best energies, one per random initialization.

    Returns
    -------
    SeedStatistics
    """
    arr = np.asarray(energies, dtype=float)
    if arr.size == 0:
        raise ValueError("seed_statistics requires at least one energy value.")
    n = int(arr.size)
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    return SeedStatistics(
        mean_energy=float(np.mean(arr)),
        std_energy=std,
        min_energy=float(np.min(arr)),
        max_energy=float(np.max(arr)),
        n_seeds=n,
    )


# ---------------------------------------------------------------------------
# Helper: statevector from best params
# ---------------------------------------------------------------------------


def _get_statevector(
    ansatz_fn: Callable, best_params: np.ndarray, n_sites: int, **ansatz_kwargs
) -> np.ndarray:
    sv_device = qp.device("default.qubit", wires=n_sites)

    @qp.qnode(sv_device)
    def sv_circuit(params):
        ansatz_fn(params, n_sites, **ansatz_kwargs)
        return qp.state()

    return np.array(sv_circuit(best_params))


# ---------------------------------------------------------------------------
# Optimizer 1: COBYLA  (gradient-free — primary for Kagome AFM)
# ---------------------------------------------------------------------------


def run_vqe_cobyla(
    hamiltonian: qp.Hamiltonian,
    ansatz_fn: Callable,
    init_params: np.ndarray,
    n_sites: int,
    ansatz_name: str = "hea",
    n_evals: int = 5000,
    rhobeg: float = 0.5,
    return_statevector: bool = True,
    verbose: bool = True,
    **ansatz_kwargs,
) -> VQEResult:
    """
    Run VQE with the COBYLA gradient-free optimizer (SciPy).

    COBYLA (Constrained Optimization BY Linear Approximations) builds a local
    linear model of the objective from function evaluations without computing
    gradients. It is robust to the zero-gradient problem that affects Adam
    on the Heisenberg AFM near Z-basis eigenstates.

    Parameters
    ----------
    hamiltonian : qp.Hamiltonian
    ansatz_fn : callable
        Signature: ``fn(params, n_sites, **ansatz_kwargs)``
    init_params : np.ndarray
        Initial variational parameters (use scale=1.0 for broad exploration).
    n_sites : int
    ansatz_name : str
    n_evals : int
        Maximum number of energy evaluations (COBYLA ``maxiter``).
    rhobeg : float
        Initial trust-region radius for COBYLA. 0.5 works well for angles in
        [-π, π]; decrease to 0.1 for fine-tuning near a known minimum.
    return_statevector : bool
    verbose : bool
        Print progress every 500 evaluations.
    **ansatz_kwargs
        Passed through to ``ansatz_fn``.

    Returns
    -------
    VQEResult
        gradient_variance_history is empty (gradient-free optimizer).
    """
    device = qp.device("default.qubit", wires=n_sites)

    @qp.qnode(device)
    def cost_fn(params):
        ansatz_fn(params, n_sites, **ansatz_kwargs)
        return qp.expval(hamiltonian)

    energy_history: list[float] = []
    best_energy = np.inf
    best_params = np.copy(init_params)
    eval_count = [0]  # mutable counter for closure

    def objective(params: np.ndarray) -> float:
        nonlocal best_energy, best_params
        e = float(cost_fn(params))
        energy_history.append(e)
        eval_count[0] += 1
        if e < best_energy:
            best_energy = e
            best_params = np.copy(params)
        if verbose and eval_count[0] % 500 == 0:
            print(
                f"  Eval {eval_count[0]:>5d} | E = {e:.8f} | "
                f"best = {best_energy:.8f}"
            )
        return e

    result = minimize(
        objective,
        init_params.copy(),
        method="COBYLA",
        options={"maxiter": n_evals, "rhobeg": rhobeg, "disp": False},
    )

    statevector = None
    if return_statevector:
        statevector = _get_statevector(ansatz_fn, best_params, n_sites, **ansatz_kwargs)

    return VQEResult(
        energy=best_energy,
        params=best_params,
        energy_history=energy_history,
        gradient_variance_history=[],
        n_steps=eval_count[0],
        converged=result.success,
        statevector=statevector,
        ansatz=ansatz_name,
        optimizer="cobyla",
        n_sites=n_sites,
        n_params=len(init_params),
    )


def run_vqe_cobyla_multi_seed(
    hamiltonian: qp.Hamiltonian,
    ansatz_fn: Callable,
    init_params_fn: Callable[[int], np.ndarray],
    n_sites: int,
    seeds: Sequence[int],
    ansatz_name: str = "hea",
    n_evals: int = 5000,
    rhobeg: float = 0.5,
    return_statevector: bool = True,
    verbose: bool = True,
    **ansatz_kwargs,
) -> VQEMultiSeedResult:
    """
    Run COBYLA VQE over multiple random seeds; keep the best run and seed statistics.

    Parameters
    ----------
    hamiltonian : qp.Hamiltonian
    ansatz_fn : callable
    init_params_fn : callable
        ``init_params_fn(seed) -> np.ndarray`` for each integer seed.
    n_sites : int
    seeds : sequence of int
        Random seeds passed to ``init_params_fn``.
    ansatz_name, n_evals, rhobeg, return_statevector, verbose
        Forwarded to ``run_vqe_cobyla`` (statevector computed for the best run only).
    **ansatz_kwargs
        Passed through to ``ansatz_fn``.

    Returns
    -------
    VQEMultiSeedResult
    """
    seed_list = [int(s) for s in seeds]
    if not seed_list:
        raise ValueError("seeds must contain at least one integer.")

    runs: list[VQEResult] = []
    for i, seed in enumerate(seed_list):
        p0 = init_params_fn(seed)
        result = run_vqe_cobyla(
            hamiltonian=hamiltonian,
            ansatz_fn=ansatz_fn,
            init_params=p0,
            n_sites=n_sites,
            ansatz_name=ansatz_name,
            n_evals=n_evals,
            rhobeg=rhobeg,
            return_statevector=False,
            verbose=verbose,
            **ansatz_kwargs,
        )
        runs.append(result)

    best = min(runs, key=lambda r: r.energy)
    if return_statevector:
        best = VQEResult(
            energy=best.energy,
            params=best.params,
            energy_history=best.energy_history,
            gradient_variance_history=best.gradient_variance_history,
            n_steps=best.n_steps,
            converged=best.converged,
            statevector=_get_statevector(
                ansatz_fn, best.params, n_sites, **ansatz_kwargs
            ),
            ansatz=best.ansatz,
            optimizer=best.optimizer,
            n_sites=best.n_sites,
            n_params=best.n_params,
        )

    stats = seed_statistics([r.energy for r in runs])
    return VQEMultiSeedResult(
        best=best,
        statistics=stats,
        runs=runs,
        seeds=seed_list,
    )


# ---------------------------------------------------------------------------
# Optimizer 2: Adam  (gradient-based — use for diagnostics / warm-start)
# ---------------------------------------------------------------------------


def run_vqe(
    hamiltonian: qp.Hamiltonian,
    ansatz_fn: Callable,
    init_params: np.ndarray,
    n_sites: int,
    ansatz_name: str = "hea",
    n_steps: int = 2000,
    step_size: float = 0.05,
    conv_tol: float = 1e-8,
    conv_window: int = 500,
    return_statevector: bool = True,
    verbose: bool = True,
    **ansatz_kwargs,
) -> VQEResult:
    """
    Run VQE with PennyLane's Adam optimizer.

    Note: Adam stalls on the Heisenberg AFM near Z-basis eigenstates because
    the Ising-gate gradients cancel by SU(2) symmetry. Use ``run_vqe_cobyla``
    for the actual ground-state search. Use this function for:
    - Gradient variance (barren plateau) diagnostics
    - Fine-tuning from a COBYLA-found minimum
    - Convergence curve visualization

    Convergence guard: early stopping only triggers if ``energy < 0``
    (optimizer has crossed the ferromagnetic saddle) AND the energy variance
    over the last ``conv_window`` steps is below ``conv_tol``.

    Parameters
    ----------
    hamiltonian : qp.Hamiltonian
    ansatz_fn : callable
    init_params : np.ndarray
    n_sites : int
    ansatz_name : str
    n_steps : int
    step_size : float
    conv_tol : float
    conv_window : int
    return_statevector : bool
    verbose : bool
    **ansatz_kwargs

    Returns
    -------
    VQEResult
    """
    device = qp.device("default.qubit", wires=n_sites)

    @qp.qnode(device)
    def cost_fn(params):
        ansatz_fn(params, n_sites, **ansatz_kwargs)
        return qp.expval(hamiltonian)

    opt = qp.AdamOptimizer(stepsize=step_size)

    params = np.copy(init_params)
    energy_history: list[float] = []
    grad_var_history: list[float] = []
    best_energy = np.inf
    best_params = np.copy(params)
    converged = False

    for step in range(n_steps):
        params, energy = opt.step_and_cost(cost_fn, params)
        energy = float(energy)
        energy_history.append(energy)

        if energy < best_energy:
            best_energy = energy
            best_params = np.copy(params)

        grad = qp.grad(cost_fn)(params)
        grad_var_history.append(float(np.var(grad)))

        if verbose and (step + 1) % 200 == 0:
            print(
                f"  Step {step + 1:>4d} | E = {energy:.8f} | "
                f"grad_var = {grad_var_history[-1]:.2e}"
            )

        if len(energy_history) >= conv_window and energy < 0:
            window = energy_history[-conv_window:]
            if np.var(window) < conv_tol:
                if verbose:
                    print(
                        f"  Converged at step {step + 1} (Δ²E = {np.var(window):.2e})"
                    )
                converged = True
                break

    statevector = None
    if return_statevector:
        statevector = _get_statevector(ansatz_fn, best_params, n_sites, **ansatz_kwargs)

    return VQEResult(
        energy=best_energy,
        params=best_params,
        energy_history=energy_history,
        gradient_variance_history=grad_var_history,
        n_steps=len(energy_history),
        converged=converged,
        statevector=statevector,
        ansatz=ansatz_name,
        optimizer="adam",
        n_sites=n_sites,
        n_params=len(init_params),
    )
