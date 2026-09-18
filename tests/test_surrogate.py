"""
test_surrogate.py
-----------------
Unit tests for spinq_vqe.surrogate — mock data loading, feature extraction,
surrogate training, and prediction.

Does not require a Materials Project API key or internet access.
"""

import numpy as np
import pytest

from spinq_vqe.surrogate import (
    DEFAULT_THETA_SH_CSV,
    FEATURE_NAMES,
    MIN_CURATED_N,
    QAOA_POOL_SIZE,
    MaterialRecord,
    SurrogateDataset,
    SurrogateMetrics,
    TrainedSurrogate,
    build_features,
    cross_validate_surrogate,
    filter_by_formulas,
    load_mock_data,
    load_theta_sh_csv,
    load_theta_sh_data,
    predict,
    predict_oracle,
    qaoa_pool_dataset,
    save_surrogate_metrics,
    split_hold_out,
    train_surrogate,
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


class TestLoadMockData:
    def test_returns_surrogate_dataset(self):
        ds = load_mock_data()
        assert isinstance(ds, SurrogateDataset)

    def test_has_phase_a_records(self):
        ds = load_mock_data()
        assert ds.n_samples >= MIN_CURATED_N
        assert ds.n_samples >= QAOA_POOL_SIZE

    def test_records_are_material_records(self):
        ds = load_mock_data()
        for r in ds.records:
            assert isinstance(r, MaterialRecord)

    def test_mn3sn_present(self):
        ds = load_mock_data()
        assert "Mn3Sn" in ds.formulas

    def test_theta_sh_are_finite(self):
        ds = load_mock_data()
        for r in ds.records:
            assert np.isfinite(r.theta_sh)

    def test_theta_sh_values_property(self):
        ds = load_mock_data()
        vals = ds.theta_sh_values
        assert isinstance(vals, np.ndarray)
        assert len(vals) == ds.n_samples

    def test_source_is_mock(self):
        ds = load_mock_data()
        for r in ds.records:
            assert r.source == "mock"


class TestLoadThetaShCsv:
    def test_csv_exists(self):
        assert DEFAULT_THETA_SH_CSV.is_file()

    def test_loads_phase_a_records(self):
        ds = load_theta_sh_csv()
        assert ds.n_samples >= MIN_CURATED_N

    def test_historical_pool_formulas_preserved(self):
        ds = load_theta_sh_csv()
        pool = qaoa_pool_dataset(ds)
        assert pool.n_samples == QAOA_POOL_SIZE
        assert "Mn3Sn" in pool.formulas
        assert "Bi2Se3" in pool.formulas

    def test_mn3sn_present_with_real_mp_id(self):
        ds = load_theta_sh_csv()
        mn = next(r for r in ds.records if r.formula == "Mn3Sn")
        assert mn.mp_id == "mp-22389"
        assert mn.theta_sh == pytest.approx(0.35)

    def test_load_theta_sh_data_uses_csv(self):
        ds = load_theta_sh_data()
        assert ds.n_samples >= MIN_CURATED_N
        assert ds.records[0].source in ("csv", "mp_api")


class TestQaoaPool:
    def test_pool_size(self):
        ds = load_theta_sh_csv()
        pool = qaoa_pool_dataset(ds)
        assert pool.n_samples == QAOA_POOL_SIZE

    def test_filter_aliases(self):
        ds = load_theta_sh_csv()
        subset = filter_by_formulas(ds, ["Co2MnGa", "IrMn3", "MoTe2"])
        assert len(subset.records) == 3
        assert {r.formula for r in subset.records} == {"MnGaCo2", "Mn3Ir", "Te2Mo"}

    def test_pool_only_surrogate_keeps_greedy_leaders(self):
        """N=12 MLP in-sample fit should still rank Bi2Se3 / CrTe2 / Mn3Sn highly."""
        pytest.importorskip("sklearn")
        from spinq_vqe import qaoa

        pool = qaoa_pool_dataset(load_theta_sh_csv())
        sr = train_surrogate(
            pool, hidden_layer_sizes=(64, 32), max_iter=3000, random_state=42
        )
        assert sr.sklearn
        pred = predict(sr, pool.records)
        chosen = [pool.records[i].formula for i in qaoa.classical_greedy(pred, k=3)]
        assert set(chosen) == {"Bi2Se3", "CrTe2", "Mn3Sn"}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


class TestBuildFeatures:
    def test_returns_2d_array(self):
        ds = load_mock_data()
        X = build_features(ds)
        assert isinstance(X, np.ndarray)
        assert X.ndim == 2

    def test_n_rows_matches_n_samples(self):
        ds = load_mock_data()
        X = build_features(ds)
        assert X.shape[0] == ds.n_samples

    def test_n_cols_matches_feature_names(self):
        ds = load_mock_data()
        X = build_features(ds)
        assert X.shape[1] == len(FEATURE_NAMES)

    def test_all_finite(self):
        ds = load_mock_data()
        X = build_features(ds)
        assert np.all(np.isfinite(X))


# ---------------------------------------------------------------------------
# Surrogate training
# ---------------------------------------------------------------------------


class TestTrainSurrogate:
    def test_returns_trained_surrogate(self):
        ds = load_mock_data()
        sr = train_surrogate(ds)
        assert isinstance(sr, TrainedSurrogate)

    def test_has_model(self):
        ds = load_mock_data()
        sr = train_surrogate(ds)
        assert sr.model is not None

    def test_has_scaler(self):
        ds = load_mock_data()
        sr = train_surrogate(ds)
        assert sr.scaler is not None

    def test_feature_names_match_constant(self):
        ds = load_mock_data()
        sr = train_surrogate(ds)
        assert sr.feature_names == FEATURE_NAMES

    def test_too_few_samples_raises(self):
        ds = load_mock_data()
        small_ds = SurrogateDataset(records=ds.records[:3])
        with pytest.raises(ValueError, match="samples"):
            train_surrogate(small_ds)

    def test_cv_metrics_finite_with_sklearn(self):
        pytest.importorskip("sklearn")
        ds = load_theta_sh_csv()
        sr = train_surrogate(ds, cv_strategy="kfold", cv_folds=5, random_state=0)
        assert sr.metrics is not None
        assert np.isfinite(sr.cv_r2)
        assert np.isfinite(sr.cv_rmse)
        assert np.isfinite(sr.train_rmse)
        assert sr.metrics.cv_strategy.startswith("kfold")
        # Honest CV should be worse than (or equal to) in-sample train fit
        assert sr.cv_rmse >= sr.train_rmse - 1e-9

    def test_hold_out_metrics(self):
        pytest.importorskip("sklearn")
        ds = load_theta_sh_csv()
        sr = train_surrogate(
            ds, hold_out_frac=0.2, random_state=0, cv_strategy="kfold"
        )
        assert sr.metrics is not None
        assert sr.metrics.n_hold_out >= 1
        assert np.isfinite(sr.metrics.hold_out_rmse)
        assert len(sr.metrics.hold_out_formulas) == sr.metrics.n_hold_out


class TestHoldOutAndOracle:
    def test_split_hold_out_sizes(self):
        ds = load_theta_sh_csv()
        train, hold = split_hold_out(ds, hold_out_frac=0.2, random_state=0)
        assert train.n_samples + hold.n_samples == ds.n_samples
        assert hold.n_samples == 7
        assert train.n_samples == 25
        assert set(train.formulas).isdisjoint(set(hold.formulas))
        # Seed 0 on the committed 32-row CSV (order = original CSV order)
        assert list(hold.formulas) == [
            "W", "Pd", "MnPt", "Bi2Se3", "Ag", "Sb2Te3", "Mn3Ga",
        ]

    def test_split_hold_out_explicit_formulas(self):
        ds = load_theta_sh_csv()
        train, hold = split_hold_out(ds, hold_out_formulas=["Bi2Te3", "Os"])
        assert set(hold.formulas) == {"Bi2Te3", "Os"}
        assert "Bi2Te3" not in train.formulas

    def test_predict_oracle_in_sample_matches_predict(self):
        pytest.importorskip("sklearn")
        pool = qaoa_pool_dataset(load_theta_sh_csv())
        oof, metrics = predict_oracle(
            pool, mode="in_sample", max_iter=3000, random_state=42
        )
        sr = train_surrogate(pool, max_iter=3000, random_state=42, compute_cv=False)
        pred = predict(sr, pool.records)
        assert metrics.oracle_mode == "in_sample"
        np.testing.assert_allclose(oof, pred, rtol=1e-5, atol=1e-5)

    def test_predict_oracle_loocv_finite(self):
        pytest.importorskip("sklearn")
        pool = qaoa_pool_dataset(load_theta_sh_csv())
        oof, metrics = predict_oracle(
            pool, mode="loocv", max_iter=1500, random_state=0
        )
        assert len(oof) == pool.n_samples
        assert np.all(np.isfinite(oof))
        assert metrics.oracle_mode == "loocv"
        assert metrics.cv_strategy == "loocv"
        assert np.isfinite(metrics.cv_rmse)

    def test_predict_oracle_rejects_unknown_mode(self):
        pool = qaoa_pool_dataset(load_theta_sh_csv())
        with pytest.raises(ValueError, match="mode"):
            predict_oracle(pool, mode="train_rmse")

    def test_linear_relationship_cv_sanity(self):
        """Synthetic linear target → CV R² should be strongly positive."""
        pytest.importorskip("sklearn")
        rng = np.random.default_rng(0)
        records = []
        for i in range(24):
            z = 20 + i
            # Plant a near-linear θ_SH in z_max so MLP/ridge can recover it
            theta = 0.01 * z + 0.001 * rng.normal()
            records.append(
                MaterialRecord(
                    mp_id=f"mp-lin-{i}",
                    formula=f"X{i}",
                    crystal_system="cubic",
                    z_max=z,
                    n_elements=1,
                    space_group=225,
                    ahc=0.0,
                    theta_sh=theta,
                    source="mock",
                )
            )
        ds = SurrogateDataset(records=records)
        _, metrics = cross_validate_surrogate(
            ds, cv_strategy="kfold", cv_folds=5, max_iter=2000, random_state=0
        )
        assert metrics.cv_r2 > 0.5

    def test_save_surrogate_metrics_roundtrip(self, tmp_path):
        m = SurrogateMetrics(
            n_samples=32,
            train_rmse=0.1,
            train_r2=0.9,
            cv_rmse=0.5,
            cv_r2=-1.0,
            cv_strategy="kfold-5",
            n_hold_out=6,
            hold_out_rmse=0.6,
            hold_out_r2=-2.0,
            hold_out_formulas=("Os", "Bi"),
            oracle_mode="in_sample",
        )
        path = save_surrogate_metrics(m, tmp_path / "m.csv", extra={"scope": "full"})
        text = path.read_text(encoding="utf-8")
        assert "kfold-5" in text
        assert "Os;Bi" in text
        assert "full" in text


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class TestPredict:
    def test_returns_array(self):
        ds = load_mock_data()
        sr = train_surrogate(ds)
        preds = predict(sr, ds.records)
        assert isinstance(preds, np.ndarray)

    def test_length_matches_input(self):
        ds = load_mock_data()
        sr = train_surrogate(ds)
        preds = predict(sr, ds.records)
        assert len(preds) == ds.n_samples

    def test_predictions_finite(self):
        ds = load_mock_data()
        sr = train_surrogate(ds)
        preds = predict(sr, ds.records)
        assert np.all(np.isfinite(preds))

    def test_single_record(self):
        ds = load_mock_data()
        sr = train_surrogate(ds)
        preds = predict(sr, [ds.records[0]])
        assert len(preds) == 1


def test_committed_surrogate_metrics_csv():
    """NB04 evaluation artifact exists with expected columns and hold-out set."""
    import csv
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data" / "surrogate_metrics.csv"
    assert path.is_file()
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    for col in (
        "n_samples",
        "train_rmse",
        "cv_rmse",
        "hold_out_rmse",
        "hold_out_formulas",
        "pool_oracle_mode",
        "pool_loocv_rmse",
    ):
        assert col in row
    assert int(row["n_samples"]) == 25
    assert int(row["n_hold_out"]) == 7
    assert row["hold_out_formulas"] == "W;Pd;MnPt;Bi2Se3;Ag;Sb2Te3;Mn3Ga"
    assert row["pool_oracle_mode"] == "in_sample"
    assert int(row["pool_n"]) == 12
    assert float(row["hold_out_rmse"]) > float(row["train_rmse"])
    assert float(row["pool_loocv_rmse"]) > float(row["pool_train_rmse"])


def test_committed_theta_sh_provenance_contract():
    """Every oracle row is explicitly sourced or explicitly illustrative."""
    import csv
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    with (root / "data" / "mp_theta_sh.csv").open(newline="", encoding="utf-8") as f:
        oracle_rows = list(csv.DictReader(f))
    with (root / "data" / "theta_sh_provenance.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        provenance_rows = list(csv.DictReader(f))

    allowed = {"illustrative_oracle", "sourced_primary"}
    assert oracle_rows
    assert all(row["theta_sh_source"] in allowed for row in oracle_rows)
    assert all(row["oracle_status"] in allowed for row in provenance_rows)
    assert {row["mp_id"] for row in oracle_rows} == {
        row["mp_id"] for row in provenance_rows
    }
    assert len(oracle_rows) >= 30


def test_plot_surrogate_holdout_numbers_match_table(tmp_path):
    """Each hold-out diamond number corresponds to one table row."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from spinq_vqe.utils import plot_surrogate_holdout

    rng = np.random.default_rng(0)
    train_t = rng.normal(size=12)
    train_p = train_t + 0.05 * rng.normal(size=12)
    hold_t = np.array([3.5, 1.5, 0.15])
    hold_p = np.array([1.0, 0.0, 1.4])
    formulas = ["Bi2Se3", "Sb2Te3", "MnPt"]
    fig = plot_surrogate_holdout(
        train_t, train_p, hold_t, hold_p, formulas, save_path=str(tmp_path / "p.png")
    )
    assert len(fig.axes) == 2
    table = fig.axes[1].tables[0]
    # Header + 3 data rows; first data cell is "1" (worst |err| = Bi2Se3)
    cells = table.get_celld()
    assert cells[1, 0].get_text().get_text() == "1"
    assert cells[1, 1].get_text().get_text() == "Bi2Se3"
    matplotlib.pyplot.close(fig)
