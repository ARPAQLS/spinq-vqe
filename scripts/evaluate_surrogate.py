#!/usr/bin/env python
"""Regenerate NB04 surrogate metrics CSV and train/hold-out figure (#20)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from spinq_vqe import surrogate, utils

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "surrogate_predictions.png"
METRICS = ROOT / "data" / "surrogate_metrics.csv"


def main() -> None:
    ds = surrogate.load_theta_sh_csv()
    sr_full = surrogate.train_surrogate(
        ds,
        hidden_layer_sizes=(64, 32),
        max_iter=3000,
        random_state=0,
        cv_strategy="kfold",
        cv_folds=5,
        hold_out_frac=0.2,
    )
    surrogate.surrogate_summary(sr_full)

    train_ds, hold_ds = surrogate.split_hold_out(ds, hold_out_frac=0.2, random_state=0)
    theta_train_pred = surrogate.predict(sr_full, train_ds.records)
    theta_hold_pred = surrogate.predict(sr_full, hold_ds.records)

    pool = surrogate.qaoa_pool_dataset(ds)
    _, pool_metrics = surrogate.predict_oracle(
        pool, mode="in_sample", hidden_layer_sizes=(64, 32), max_iter=3000, random_state=0
    )
    _, pool_loo = surrogate.predict_oracle(
        pool, mode="loocv", hidden_layer_sizes=(64, 32), max_iter=3000, random_state=0
    )

    fig = utils.plot_surrogate_holdout(
        train_ds.theta_sh_values,
        theta_train_pred,
        hold_ds.theta_sh_values,
        theta_hold_pred,
        hold_ds.formulas,
        save_path=str(FIG),
    )
    plt.close(fig)
    print(f"Saved -> {FIG.relative_to(ROOT)}")

    assert sr_full.metrics is not None
    surrogate.save_surrogate_metrics(
        sr_full.metrics,
        METRICS,
        extra={
            "pool_n": pool.n_samples,
            "pool_train_rmse": f"{pool_metrics.train_rmse:.6f}",
            "pool_loocv_rmse": f"{pool_loo.cv_rmse:.6f}",
            "pool_oracle_mode": "in_sample",
        },
    )
    print(f"Saved -> {METRICS.relative_to(ROOT)}")
    print(
        f"pool in-sample RMSE={pool_metrics.train_rmse:.4f}  "
        f"LOOCV RMSE={pool_loo.cv_rmse:.4f}"
    )


if __name__ == "__main__":
    main()
