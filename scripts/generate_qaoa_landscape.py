"""
Generate figures/qaoa_landscape.png using the same NB04 oracle setup.

Run from repo root:
    python scripts/generate_qaoa_landscape.py

Uses committed data/mp_theta_sh.csv (no MP API key). Does not modify other figures.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from spinq_vqe import qaoa, surrogate, utils

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures" / "qaoa_landscape.png"
CSV = REPO / "data" / "qaoa_results.csv"

LAM = 6.0
K = 3
GRID = 40


def _depth_theta_from_csv() -> tuple[dict[int, float], float]:
    """Load QAOA depths and greedy baseline from committed NB04 CSV."""
    import csv

    depth: dict[int, float] = {}
    classical = 0.0
    with CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            method = row["method"]
            total = float(row["total_theta_sh"])
            if method == "Greedy":
                classical = total
            elif method.startswith("QAOA_p"):
                p = int(method.split("p", 1)[1])
                depth[p] = total
    if not depth or classical == 0.0:
        raise RuntimeError(f"Could not parse depth/greedy rows from {CSV}")
    return depth, classical


def main() -> None:
    warnings.filterwarnings("ignore")

    ds = surrogate.load_theta_sh_data()
    pool = surrogate.qaoa_pool_dataset(ds)
    # In-sample N=12 pool oracle (same policy as published NB04 QAOA).
    # random_state=42 is the historical landscape seed — do not change
    # without regenerating figures/qaoa_landscape.png.
    theta_oracle, _ = surrogate.predict_oracle(
        pool,
        mode="in_sample",
        hidden_layer_sizes=(64, 32),
        max_iter=3000,
        random_state=42,
    )

    print("Sampling p=1 (gamma, beta) landscape...")
    landscape = qaoa.qaoa_landscape_grid(
        theta_oracle,
        k=K,
        lam=LAM,
        p=1,
        n_gamma=GRID,
        n_beta=GRID,
    )
    minima = qaoa.find_landscape_minima(landscape, neighborhood=5, max_minima=4)

    print("Re-running QAOA p=1 with COBYLA trajectory recording...")
    res_p1 = qaoa.run_qaoa(
        theta_oracle,
        k=K,
        p=1,
        lam=LAM,
        n_optimizer_steps=300,
        n_seeds=5,
        step_size=0.3,
        verbose=False,
        record_param_history=True,
    )

    depth_theta, classical_theta = _depth_theta_from_csv()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    utils.plot_qaoa_landscape(
        landscape.gamma,
        landscape.beta,
        landscape.energies,
        cobyla_gamma=float(res_p1.gamma[0]),
        cobyla_beta=float(res_p1.beta[0]),
        param_trajectory=res_p1.param_history,
        local_minima=minima,
        depth_theta_sh=depth_theta,
        classical_theta_sh=classical_theta,
        save_path=str(OUT),
    )
    print(f"Saved -> {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
