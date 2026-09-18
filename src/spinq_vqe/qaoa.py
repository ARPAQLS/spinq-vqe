"""
qaoa.py
-------
QAOA circuit and optimizer for SOC material composition selection.

Problem
-------
Given N candidate spintronic materials with predicted spin Hall angles θ_SH(i)
(from the ``surrogate`` module), find the k-layer heterostructure composition
that maximizes the total θ_SH:

    Maximize:  Σᵢ xᵢ · θ_SH(i)
    Subject to: Σᵢ xᵢ = k,  xᵢ ∈ {0, 1}

This is formulated as a QUBO and solved with QAOA (Farhi et al. 2014).

QUBO encoding
-------------
Map binary variables xᵢ ∈ {0,1} to Pauli-Z: xᵢ = (1 − Zᵢ) / 2.

Cost Hamiltonian:
    H_C = −(1/2) Σᵢ wᵢ Zᵢ + λ (Σᵢ Zᵢ − (N − 2k))²

where wᵢ = θ_SH(i) (objective weights) and λ controls the selection
constraint penalty.

Mixer Hamiltonian (standard transverse field):
    H_M = Σᵢ Xᵢ

Pipeline
--------
1. ``build_cost_hamiltonian(theta_sh, k, lam)``  — build H_C from θ_SH values
2. ``build_mixer_hamiltonian(n_materials)``       — build H_M
3. ``run_qaoa(theta_sh, k, p, ...)``              — full QAOA optimization
4. ``qaoa_landscape_grid(...)``                     — (γ, β) cost landscape at p=1
5. ``sample_bitstrings(result, n_shots)``          — sample from optimized circuit
6. ``classical_greedy(theta_sh, k)``              — greedy baseline comparison → ``list[int]``
7. ``classical_simulated_annealing(theta_sh, k)`` — SA baseline comparison

References
----------
- Farhi et al. (2014) arXiv:1411.4028 — original QAOA
- Hadfield et al. (2019) Algorithms 12, 34 — QAOA variants / mixers
- Lucas (2014) Front. Phys. 2, 5 — QUBO encoding of combinatorial problems
- Blekos et al. (2024) Physics Reports 1068 — QAOA review
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pennylane as qp
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class QAOAResult:
    """Container for a single QAOA optimization run."""

    energy: float
    """Best QAOA cost value (lower = better heterostructure)."""

    gamma: np.ndarray
    """Optimal cost layer angles, shape (p,)."""

    beta: np.ndarray
    """Optimal mixer layer angles, shape (p,)."""

    energy_history: list[float] = field(default_factory=list)
    """Cost value at each optimizer step."""

    selected_indices: list[int] = field(default_factory=list)
    """Indices of the selected k materials (from most likely bitstring)."""

    selected_theta_sh: float = 0.0
    """Sum of θ_SH for the selected materials."""

    p: int = 1
    """QAOA depth (number of alternating layers)."""

    n_materials: int = 0
    k: int = 0

    param_history: list[np.ndarray] = field(default_factory=list)
    """(γ, β, …) at each COBYLA evaluation for the best seed (if recorded)."""


@dataclass
class QAOALandscapeGrid:
    """Cost landscape samples for QAOA at fixed depth ``p`` (typically p=1)."""

    gamma: np.ndarray
    """γ grid, shape ``(n_gamma,)``."""

    beta: np.ndarray
    """β grid, shape ``(n_beta,)``."""

    energies: np.ndarray
    """Cost values, shape ``(n_gamma, n_beta)``."""

    p: int = 1
    k: int = 0
    lam: float = 5.0


# ---------------------------------------------------------------------------
# Hamiltonian construction
# ---------------------------------------------------------------------------


def build_cost_hamiltonian(
    theta_sh: np.ndarray,
    k: int,
    lam: float = 5.0,
) -> qp.Hamiltonian:
    """
    Build the QAOA cost Hamiltonian for k-from-N material selection.

    H_C = −(1/2) Σᵢ wᵢ Zᵢ + λ (Σᵢ Zᵢ − (N − 2k))²

    The first term encodes the objective (maximize θ_SH).
    The second term penalizes deviation from exactly k selected materials.

    Parameters
    ----------
    theta_sh : np.ndarray, shape (N,)
        Predicted spin Hall angles for the N candidate materials.
    k : int
        Number of materials to select (heterostructure layers).
    lam : float
        Constraint penalty strength. Increase if the optimizer frequently
        selects ≠ k materials.

    Returns
    -------
    qp.Hamiltonian
    """
    N = len(theta_sh)
    coeffs: list[float] = []
    ops: list = []

    # Objective: -(1/2) Σᵢ wᵢ Zᵢ  (we minimize, so negate the objective)
    for i, w in enumerate(theta_sh):
        coeffs.append(-0.5 * float(w))
        ops.append(qp.PauliZ(i))

    # Constraint: λ (Σᵢ Zᵢ - target)²  where target = N - 2k
    # Expanding: λ [Σᵢ Zᵢ² + 2Σᵢ<ⱼ ZᵢZⱼ - 2·target·Σᵢ Zᵢ + target²]
    # Zᵢ² = I (dropped — constant energy shift)
    target = float(N - 2 * k)

    # Cross terms: 2λ ZᵢZⱼ
    for i in range(N):
        for j in range(i + 1, N):
            coeffs.append(2.0 * lam)
            ops.append(qp.PauliZ(i) @ qp.PauliZ(j))

    # Linear terms: -2λ·target·Zᵢ
    for i in range(N):
        coeffs.append(-2.0 * lam * target)
        ops.append(qp.PauliZ(i))

    return qp.Hamiltonian(coeffs, ops)


def build_mixer_hamiltonian(n_materials: int) -> qp.Hamiltonian:
    """
    Build the standard transverse-field mixer Hamiltonian.

    H_M = Σᵢ Xᵢ

    Parameters
    ----------
    n_materials : int

    Returns
    -------
    qp.Hamiltonian
    """
    coeffs = [1.0] * n_materials
    ops = [qp.PauliX(i) for i in range(n_materials)]
    return qp.Hamiltonian(coeffs, ops)


# ---------------------------------------------------------------------------
# QAOA circuit
# ---------------------------------------------------------------------------


def qaoa_circuit(
    params: np.ndarray,
    cost_h: qp.Hamiltonian,
    mixer_h: qp.Hamiltonian,
    n_materials: int,
    p: int,
) -> None:
    """
    Apply p layers of QAOA to the uniform superposition state.

    Each layer:
        cost layer:  exp(−i γ_l H_C)   — phase separation
        mixer layer: exp(−i β_l H_M)   — mixing

    Starting state: |+⟩^⊗N (uniform superposition, prepared by Hadamard on all).

    Parameters
    ----------
    params : np.ndarray, shape (2*p,)
        Interleaved [γ₁, β₁, γ₂, β₂, ..., γ_p, β_p].
    cost_h : qp.Hamiltonian
    mixer_h : qp.Hamiltonian
    n_materials : int
    p : int
    """
    gamma = params[:p]
    beta = params[p:]

    # Initial state: uniform superposition
    for i in range(n_materials):
        qp.Hadamard(wires=i)

    # p alternating layers
    for layer in range(p):
        qp.ApproxTimeEvolution(cost_h,  gamma[layer], 1)
        qp.ApproxTimeEvolution(mixer_h, beta[layer],  1)


# ---------------------------------------------------------------------------
# Cost evaluation and landscape sampling
# ---------------------------------------------------------------------------


def make_qaoa_cost_fn(
    theta_sh: np.ndarray,
    k: int,
    p: int = 1,
    lam: float = 5.0,
):
    """
    Build a PennyLane QAOA cost evaluator ⟨H_C⟩ for fixed depth ``p``.

    Parameters
    ----------
    theta_sh : np.ndarray
        Oracle θ_SH values (length N).
    k : int
        Selection size.
    p : int
        QAOA depth.
    lam : float
        Constraint penalty.

    Returns
    -------
    evaluate : callable
        ``evaluate(params) -> float`` where ``params`` has shape ``(2*p,)``.
    cost_h, mixer_h : qp.Hamiltonian
        Hamiltonians used by the circuit.
    n_materials : int
    """
    N = len(theta_sh)
    if k >= N:
        raise ValueError(f"k={k} must be < N={N}.")
    if p < 1:
        raise ValueError("p must be >= 1")

    cost_h = build_cost_hamiltonian(theta_sh, k, lam)
    mixer_h = build_mixer_hamiltonian(N)
    device = qp.device("default.qubit", wires=N)

    @qp.qnode(device)
    def cost_fn(params):
        qaoa_circuit(params, cost_h, mixer_h, N, p)
        return qp.expval(cost_h)

    def evaluate(params: np.ndarray) -> float:
        return float(cost_fn(np.asarray(params, dtype=float)))

    return evaluate, cost_h, mixer_h, N


def evaluate_qaoa_cost(
    theta_sh: np.ndarray,
    params: np.ndarray,
    k: int,
    p: int = 1,
    lam: float = 5.0,
) -> float:
    """Evaluate the QAOA cost at a single parameter vector."""
    evaluate, _, _, _ = make_qaoa_cost_fn(theta_sh, k, p=p, lam=lam)
    return evaluate(params)


def qaoa_landscape_grid(
    theta_sh: np.ndarray,
    k: int,
    lam: float = 5.0,
    p: int = 1,
    n_gamma: int = 40,
    n_beta: int = 40,
    gamma_bounds: tuple[float, float] = (0.0, 2 * np.pi),
    beta_bounds: tuple[float, float] = (0.0, np.pi),
) -> QAOALandscapeGrid:
    """
    Sample the QAOA cost over a (γ, β) grid at fixed depth ``p``.

    For ``p > 1`` only the first layer angles are swept; remaining layers are
    held at zero (use ``p=1`` for the standard landscape diagnostic).

    Parameters
    ----------
    theta_sh, k, lam, p
        Problem definition (same as ``run_qaoa``).
    n_gamma, n_beta : int
        Grid resolution.
    gamma_bounds, beta_bounds : tuple of float
        Angle ranges in radians.

    Returns
    -------
    QAOALandscapeGrid
    """
    if p != 1:
        raise ValueError("qaoa_landscape_grid currently supports p=1 only.")

    evaluate, _, _, _ = make_qaoa_cost_fn(theta_sh, k, p=p, lam=lam)
    gammas = np.linspace(gamma_bounds[0], gamma_bounds[1], n_gamma)
    betas = np.linspace(beta_bounds[0], beta_bounds[1], n_beta)
    energies = np.zeros((n_gamma, n_beta), dtype=float)

    for i, gamma in enumerate(gammas):
        for j, beta in enumerate(betas):
            energies[i, j] = evaluate(np.array([gamma, beta], dtype=float))

    return QAOALandscapeGrid(
        gamma=gammas,
        beta=betas,
        energies=energies,
        p=p,
        k=k,
        lam=lam,
    )


def find_landscape_minima(
    landscape: QAOALandscapeGrid,
    *,
    neighborhood: int = 3,
    max_minima: int = 5,
) -> list[tuple[float, float, float]]:
    """
    Find coarse local minima on a sampled QAOA landscape.

    Returns
    -------
    list of (gamma, beta, energy), sorted by increasing energy.
    """
    from scipy.ndimage import minimum_filter

    if neighborhood < 3:
        raise ValueError("neighborhood must be >= 3")
    if neighborhood % 2 == 0:
        neighborhood += 1

    filtered = minimum_filter(landscape.energies, size=neighborhood, mode="nearest")
    mask = np.isclose(landscape.energies, filtered, rtol=0.0, atol=1e-8)
    coords = np.argwhere(mask)

    minima: list[tuple[float, float, float]] = []
    for i, j in coords:
        minima.append(
            (
                float(landscape.gamma[i]),
                float(landscape.beta[j]),
                float(landscape.energies[i, j]),
            )
        )

    minima.sort(key=lambda item: item[2])
    deduped: list[tuple[float, float, float]] = []
    for gamma, beta, energy in minima:
        if all(
            np.hypot(gamma - g, beta - b) > 0.35
            for g, b, _ in deduped
        ):
            deduped.append((gamma, beta, energy))
        if len(deduped) >= max_minima:
            break
    return deduped


# ---------------------------------------------------------------------------
# QAOA optimizer
# ---------------------------------------------------------------------------


def run_qaoa(
    theta_sh: np.ndarray,
    k: int,
    p: int = 1,
    lam: float = 5.0,
    n_optimizer_steps: int = 500,
    n_seeds: int = 5,
    step_size: float = 0.1,
    verbose: bool = True,
    record_param_history: bool = False,
) -> QAOAResult:
    """
    Run QAOA optimization for the k-from-N material selection problem.

    Parameters
    ----------
    theta_sh : np.ndarray, shape (N,)
        Predicted spin Hall angles for N candidate materials.
    k : int
        Number of materials to select.
    p : int
        QAOA circuit depth. Higher p → better approximation, slower.
        Start with p=1, benchmark up to p=5.
    lam : float
        Constraint penalty. Rule of thumb: lam > max(theta_sh).
    n_optimizer_steps : int
        COBYLA evaluations per seed.
    n_seeds : int
        Number of random initializations. Best result is kept.
    step_size : float
        Initial step size for COBYLA (rhobeg).
    verbose : bool
    record_param_history : bool
        If True, store parameter vectors at each evaluation for the best seed
        (useful for landscape trajectory overlays).

    Returns
    -------
    QAOAResult
    """
    N = len(theta_sh)
    if k >= N:
        raise ValueError(f"k={k} must be < N={N}.")

    cost_h = build_cost_hamiltonian(theta_sh, k, lam)
    mixer_h = build_mixer_hamiltonian(N)
    device = qp.device("default.qubit", wires=N)

    @qp.qnode(device)
    def cost_fn(params):
        qaoa_circuit(params, cost_h, mixer_h, N, p)
        return qp.expval(cost_h)

    best_energy = np.inf
    best_params = None
    best_history: list[float] = []
    best_param_history: list[np.ndarray] = []

    rng = np.random.default_rng(42)
    for seed_idx in range(n_seeds):
        # Random init in [0, 2π] for gamma, [0, π] for beta
        p0_gamma = rng.uniform(0, 2 * np.pi, size=p)
        p0_beta  = rng.uniform(0, np.pi,     size=p)
        p0 = np.concatenate([p0_gamma, p0_beta])

        history: list[float] = []
        param_history: list[np.ndarray] = []

        def objective(params):
            params = np.asarray(params, dtype=float)
            e = float(cost_fn(params))
            history.append(e)
            if record_param_history:
                param_history.append(params.copy())
            return e

        result = minimize(
            objective,
            p0,
            method="COBYLA",
            options={"maxiter": n_optimizer_steps, "rhobeg": step_size},
        )

        if result.fun < best_energy:
            best_energy = result.fun
            best_params = result.x
            best_history = history
            best_param_history = param_history

        if verbose:
            print(f"  seed={seed_idx}  E={result.fun:.6f}  evals={len(history)}")

    # Sample bitstrings from optimal circuit to find selected materials
    selected = _decode_selection(best_params, cost_h, mixer_h, N, p, k, device)
    selected_theta = float(np.sum(theta_sh[selected]))

    if verbose:
        greedy_indices = classical_greedy(theta_sh, k)
        greedy_total = float(np.sum(theta_sh[greedy_indices]))
        print(f"\nSelected: {[int(i) for i in selected]}")
        print(f"Total theta_SH: {selected_theta:.4f}  (greedy: {greedy_total:.4f})")

    return QAOAResult(
        energy=best_energy,
        gamma=best_params[:p],
        beta=best_params[p:],
        energy_history=best_history,
        selected_indices=selected,
        selected_theta_sh=selected_theta,
        p=p,
        n_materials=N,
        k=k,
        param_history=best_param_history,
    )


def _decode_selection(
    params: np.ndarray,
    cost_h: qp.Hamiltonian,
    mixer_h: qp.Hamiltonian,
    n_materials: int,
    p: int,
    k: int,
    device,
) -> list[int]:
    """
    Sample the optimal QAOA circuit and decode the most likely valid selection.
    """
    @qp.qnode(device)
    def sample_circuit(params):
        qaoa_circuit(params, cost_h, mixer_h, n_materials, p)
        return qp.probs(wires=range(n_materials))

    probs = np.array(sample_circuit(params))

    # Find the highest-probability bitstring with exactly k ones
    best_prob = -1.0
    best_bits = None
    for idx in np.argsort(probs)[::-1]:
        bits = np.array(list(np.binary_repr(idx, width=n_materials)), dtype=int)
        if bits.sum() == k:
            if probs[idx] > best_prob:
                best_prob = probs[idx]
                best_bits = bits
        if best_bits is not None and probs[idx] < best_prob * 0.01:
            break  # stop when remaining probs are negligible

    if best_bits is None:
        # Fallback: greedy selection
        return list(np.argsort(np.zeros(n_materials))[:k])

    return [int(i) for i in np.where(best_bits == 1)[0]]


# ---------------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------------


def classical_greedy(theta_sh: np.ndarray, k: int) -> list[int]:
    """
    Greedy baseline: select the k materials with highest θ_SH.

    Parameters
    ----------
    theta_sh : np.ndarray
    k : int

    Returns
    -------
    list of int
        Indices of the k materials with the highest θ_SH, sorted descending.
    """
    selected = np.argsort(theta_sh)[::-1][:k].tolist()
    return [int(i) for i in selected]


def classical_simulated_annealing(
    theta_sh: np.ndarray,
    k: int,
    n_steps: int = 10_000,
    T_start: float = 1.0,
    T_end: float = 0.01,
    seed: int = 42,
) -> dict:
    """
    Simulated annealing baseline for the k-from-N selection problem.

    Parameters
    ----------
    theta_sh : np.ndarray
    k : int
    n_steps : int
    T_start, T_end : float
        Exponential cooling schedule.
    seed : int

    Returns
    -------
    dict with keys 'selected_indices', 'total', 'energy_history'
    """
    N = len(theta_sh)
    rng = np.random.default_rng(seed)

    # Initial solution: random k-selection
    selected = set(rng.choice(N, size=k, replace=False).tolist())

    def energy(sel):
        return -float(np.sum(theta_sh[list(sel)]))  # minimize negative = maximize

    current_e = energy(selected)
    best_sel   = set(selected)
    best_e     = current_e
    history    = [current_e]

    temperatures = np.exp(np.linspace(np.log(T_start), np.log(T_end), n_steps))

    for step, T in enumerate(temperatures):
        # Swap one selected for one unselected
        unselected = list(set(range(N)) - selected)
        remove = rng.choice(list(selected))
        add    = rng.choice(unselected)
        candidate = (selected - {remove}) | {add}

        delta = energy(candidate) - current_e
        if delta < 0 or rng.random() < np.exp(-delta / T):
            selected = candidate
            current_e = energy(candidate)
            if current_e < best_e:
                best_e = current_e
                best_sel = set(selected)

        if step % 1000 == 0:
            history.append(current_e)

    return {
        "selected_indices": sorted(int(i) for i in best_sel),
        "total": -best_e,
        "energy_history": history,
    }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def qaoa_summary(result: QAOAResult, formulas: list[str] | None = None) -> None:
    """Print a summary of a QAOAResult."""
    print(f"QAOA depth p={result.p}  |  N={result.n_materials}  k={result.k}")
    print(f"Best QAOA energy: {result.energy:.6f}")
    print(f"Selected indices: {result.selected_indices}")
    if formulas:
        selected_formulas = [formulas[i] for i in result.selected_indices]
        print(f"Selected formulas: {selected_formulas}")
    print(f"Total theta_SH:  {result.selected_theta_sh:.4f}")
