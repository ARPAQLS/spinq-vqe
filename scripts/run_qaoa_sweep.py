#!/usr/bin/env python
"""Regenerate NB04 QAOA hyperparameter-sweep CSV + figure (#21).

Pins the N=12 in-sample pool oracle (same policy as published QAOA totals).
Does **not** modify ``data/qaoa_results.csv``.

Usage (repo root)::

    python scripts/run_qaoa_sweep.py           # committed grid (see below)
    python scripts/run_qaoa_sweep.py --full    # full λ × budget × p factorial
    python scripts/run_qaoa_sweep.py --quick   # tiny smoke grid

Committed grid (default)
------------------------
Around the published NB04 point (λ=6, 300 evals, 5 seeds, p=1/2/3):

* λ ∈ {2, 5, 6, 10, 20} at every depth, budget=300
* budget ∈ {100, 500} at λ=6, every depth (300 already covered above)
* p=4 probe at λ=6, budget=300

That covers every axis in issue #21 without a 45-cell factorial (p=3 × 500
evals × 5 seeds is the expensive corner). Use ``--full`` for the Cartesian
product.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from spinq_vqe import qaoa, surrogate, utils

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "qaoa_sweep.png"
SUMMARY = ROOT / "data" / "qaoa_sweep.csv"
SEEDS = ROOT / "data" / "qaoa_sweep_seeds.csv"


def _frozen_oracle():
    ds = surrogate.load_theta_sh_csv()
    pool = surrogate.qaoa_pool_dataset(ds)
    theta, metrics = surrogate.predict_oracle(
        pool,
        mode="in_sample",
        hidden_layer_sizes=(64, 32),
        max_iter=3000,
        random_state=0,
    )
    return pool, theta, metrics


def _cell_key(row: dict) -> tuple:
    return (int(row["p"]), float(row["lam"]), int(row["n_optimizer_steps"]))


def _seed_key(row: dict) -> tuple:
    return (*_cell_key(row), int(row["seed_idx"]))


def _merge(base: qaoa.QAOASweepResult, extra: qaoa.QAOASweepResult) -> None:
    seen = {_cell_key(r) for r in base.summary}
    for row in extra.summary:
        key = _cell_key(row)
        if key in seen:
            continue
        base.summary.append(row)
        seen.add(key)
    oid = base.oracle_id
    seed_seen = {_seed_key(r) for r in base.seeds}
    for row in extra.seeds:
        if row["oracle_id"] != oid:
            continue
        key = _seed_key(row)
        if key in seed_seen:
            continue
        base.seeds.append(row)
        seed_seen.add(key)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    quick = "--quick" in argv
    full = "--full" in argv

    pool, theta, metrics = _frozen_oracle()
    print(
        f"Frozen oracle: N={pool.n_samples}  mode=in_sample  "
        f"train RMSE={metrics.train_rmse:.4f}  id={qaoa.oracle_id(theta)}",
        flush=True,
    )
    greedy = qaoa.classical_greedy(theta, qaoa.NB04_K)
    print(
        "Greedy: "
        + ", ".join(pool.formulas[i] for i in greedy)
        + f"  total={float(theta[greedy].sum()):.4f}",
        flush=True,
    )

    kwargs = dict(
        k=qaoa.NB04_K,
        formulas=pool.formulas,
        verbose=True,
        oracle_mode="in_sample",
    )

    def checkpoint(s: qaoa.QAOASweepResult) -> None:
        qaoa.save_qaoa_sweep(s, SUMMARY, SEEDS)

    if quick:
        sweep = qaoa.run_qaoa_sweep(
            theta,
            p_values=(1,),
            lam_values=(qaoa.NB04_LAM,),
            step_values=(20,),
            n_seeds=2,
            on_cell=checkpoint,
            **kwargs,
        )
    elif full:
        sweep = qaoa.run_qaoa_sweep(theta, on_cell=checkpoint, **kwargs)
        print("p=4 probe at published lam / budget ...", flush=True)
        extra = qaoa.run_qaoa_sweep(
            theta,
            p_values=(4,),
            lam_values=(qaoa.NB04_LAM,),
            step_values=(qaoa.NB04_OPTIMIZER_STEPS,),
            **kwargs,
        )
        _merge(sweep, extra)
        checkpoint(sweep)
    else:
        print("lam x depth at published budget (300 evals) ...", flush=True)
        sweep = qaoa.run_qaoa_sweep(
            theta,
            step_values=(qaoa.NB04_OPTIMIZER_STEPS,),
            on_cell=checkpoint,
            **kwargs,
        )
        print("Budget slice at published lam=6 ...", flush=True)
        budget = qaoa.run_qaoa_sweep(
            theta,
            lam_values=(qaoa.NB04_LAM,),
            step_values=(100, 500),
            **kwargs,
        )
        _merge(sweep, budget)
        checkpoint(sweep)
        print("p=4 probe at published lam / budget ...", flush=True)
        extra = qaoa.run_qaoa_sweep(
            theta,
            p_values=(4,),
            lam_values=(qaoa.NB04_LAM,),
            step_values=(qaoa.NB04_OPTIMIZER_STEPS,),
            **kwargs,
        )
        _merge(sweep, extra)
        checkpoint(sweep)

    qaoa.save_qaoa_sweep(sweep, SUMMARY, SEEDS)
    print(f"Saved -> {SUMMARY.relative_to(ROOT)}", flush=True)
    print(f"Saved -> {SEEDS.relative_to(ROOT)}", flush=True)

    best = qaoa.sweep_best_row(sweep.summary)
    print(
        f"Best QAOA in sweep: p={best['p']}  lam={best['lam']:g}  "
        f"steps={best['n_optimizer_steps']}  theta={best['best_theta_sh']:.4f}  "
        f"gap_to_greedy={best['gap_to_greedy']:.4f}  "
        f"formulas={best['selected_formulas']}",
        flush=True,
    )

    fig = utils.plot_qaoa_sweep(sweep.summary, save_path=str(FIG))
    plt.close(fig)
    print(f"Saved -> {FIG.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
