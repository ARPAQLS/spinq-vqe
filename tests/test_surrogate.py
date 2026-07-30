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
    FEATURE_NAMES,
    MaterialRecord,
    SurrogateDataset,
    TrainedSurrogate,
    DEFAULT_THETA_SH_CSV,
    build_features,
    load_mock_data,
    load_theta_sh_csv,
    load_theta_sh_data,
    predict,
    train_surrogate,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


class TestLoadMockData:
    def test_returns_surrogate_dataset(self):
        ds = load_mock_data()
        assert isinstance(ds, SurrogateDataset)

    def test_has_12_records(self):
        ds = load_mock_data()
        assert ds.n_samples == 12

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
        assert len(vals) == 12

    def test_source_is_mock(self):
        ds = load_mock_data()
        for r in ds.records:
            assert r.source == "mock"


class TestLoadThetaShCsv:
    def test_csv_exists(self):
        assert DEFAULT_THETA_SH_CSV.is_file()

    def test_loads_12_records(self):
        ds = load_theta_sh_csv()
        assert ds.n_samples == 12

    def test_mn3sn_present_with_real_mp_id(self):
        ds = load_theta_sh_csv()
        mn = next(r for r in ds.records if r.formula == "Mn3Sn")
        assert mn.mp_id == "mp-22389"
        assert mn.theta_sh == pytest.approx(0.35)

    def test_load_theta_sh_data_uses_csv(self):
        ds = load_theta_sh_data()
        assert ds.n_samples == 12
        assert ds.records[0].source in ("csv", "mp_api")


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
