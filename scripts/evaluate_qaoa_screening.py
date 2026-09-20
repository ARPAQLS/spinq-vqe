#!/usr/bin/env python
"""Regenerate NB04 screening-evaluation CSV + figure (#23).

Trains the MLP on the #20 hold-out complement (25 rows), predicts the
historical N=12 QAOA pool (4 of those 12 never seen in the fit), then runs
greedy / SA / QAOA on the *predicted* weights.

Does **not** modify ``data/qaoa_results.csv``.

Usage (repo root)::

    python scripts/evaluate_qaoa_screening.py          # greedy/SA + QAOA p=1,2,3
    python scripts/evaluate_qaoa_screening.py --quick  # greedy/SA only
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from spinq_vqe import qaoa, surrogate, utils

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "qaoa_screening.png"
SUMMARY = ROOT / "data" / "qaoa_screening.csv"


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    quick = "--quick" in argv

    ds = surrogate.load_theta_sh_csv()
    split = surrogate.screening_split(ds)
    theta_pred, sr = surrogate.predict_screening_oracle(
        split,
        hidden_layer_sizes=(64, 32),
        max_iter=3000,
        random_state=0,
    )
    theta_label = split.pool.theta_sh_values
    formulas = split.pool.formulas
    print(
        f"Screening split: n_train={split.n_train}  n_pool={split.n_pool}  "
        f"n_unseen_pool={split.n_unseen_pool}  "
        f"unseen={list(split.unseen_pool_formulas)}",
        flush=True,
    )
    print(f"Train RMSE (screening fit): {sr.train_rmse:.4f}", flush=True)

    kw = dict(
        n_train=split.n_train,
        n_pool=split.n_pool,
        held_out_formulas=split.hold_out.formulas,
        unseen_pool_formulas=split.unseen_pool_formulas,
    )

    greedy_pred = qaoa.classical_greedy(theta_pred, qaoa.NB04_K)
    greedy_label = qaoa.classical_greedy(theta_label, qaoa.NB04_K)
    sa = qaoa.classical_simulated_annealing(theta_pred, qaoa.NB04_K, seed=42)

    rows: list[dict] = []
    if not quick:
        for p in qaoa.NB04_DEPTHS:
            print(f"QAOA p={p} on screening oracle ...", flush=True)
            res = qaoa.run_qaoa(
                theta_pred,
                k=qaoa.NB04_K,
                p=int(p),
                lam=qaoa.NB04_LAM,
                n_optimizer_steps=qaoa.NB04_OPTIMIZER_STEPS,
                n_seeds=qaoa.NB04_N_SEEDS,
                step_size=qaoa.NB04_STEP_SIZE,
                rng_seed=qaoa.NB04_RNG_SEED,
                verbose=False,
            )
            rows.append(
                qaoa.screening_selection_row(
                    f"QAOA_p{p}",
                    res.selected_indices,
                    formulas,
                    theta_pred,
                    theta_label,
                    extra={"evals": res.n_evals},
                    **kw,
                )
            )

    rows.extend(
        [
            qaoa.screening_selection_row(
                "Greedy_pred", greedy_pred, formulas, theta_pred, theta_label, **kw
            ),
            qaoa.screening_selection_row(
                "SA",
                sa["selected_indices"],
                formulas,
                theta_pred,
                theta_label,
                **kw,
            ),
            qaoa.screening_selection_row(
                "Greedy_label",
                greedy_label,
                formulas,
                theta_pred,
                theta_label,
                **kw,
            ),
        ]
    )

    # Stable published-adjacent order
    order = ["QAOA_p1", "QAOA_p2", "QAOA_p3", "Greedy_pred", "SA", "Greedy_label"]
    by_method = {r["method"]: r for r in rows}
    rows = [by_method[m] for m in order if m in by_method]

    qaoa.save_qaoa_screening(rows, SUMMARY)
    print(f"Saved -> {SUMMARY.relative_to(ROOT)}", flush=True)
    for r in rows:
        print(
            f"  {r['method']:<14}  pred={r['total_pred']:.4f}  "
            f"label={r['total_label']:.4f}  "
            f"sel={r['selected_formulas']}  "
            f"unseen_in_sel={r['n_unseen_selected']}",
            flush=True,
        )

    fig = utils.plot_qaoa_screening(rows, save_path=str(FIG))
    plt.close(fig)
    print(f"Saved -> {FIG.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
