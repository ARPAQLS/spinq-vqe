"""
test_qaoa.py
------------
Unit tests for spinq_vqe.qaoa — Hamiltonian construction, circuit, and optimizer.

Uses N=4 materials (4 qubits) with very few optimizer steps to stay fast.
We verify structure, types, and constraint satisfaction — not convergence quality.
"""

import numpy as np
import pytest

from spinq_vqe.qaoa import (
    QAOAResult,
    build_cost_hamiltonian,
    build_mixer_hamiltonian,
    classical_greedy,
    evaluate_qaoa_cost,
    find_landscape_minima,
    is_published_qaoa_config,
    load_qaoa_sweep,
    qaoa_landscape_grid,
    run_qaoa,
    run_qaoa_sweep,
    save_qaoa_sweep,
    sweep_best_row,
)

# Small synthetic θ_SH dataset: 4 materials
THETA_SH_4 = np.array([0.35, -0.33, 0.40, 0.08])


# ---------------------------------------------------------------------------
# Hamiltonian builders
# ---------------------------------------------------------------------------


class TestBuildCostHamiltonian:
    def test_returns_hamiltonian(self):
        import pennylane as qp

        H = build_cost_hamiltonian(THETA_SH_4, k=2)
        assert isinstance(H, qp.Hamiltonian)

    def test_coeffs_finite(self):
        H = build_cost_hamiltonian(THETA_SH_4, k=2)
        assert all(np.isfinite(c) for c in H.coeffs)

    def test_larger_lam_larger_constraint_coeffs(self):
        H_low = build_cost_hamiltonian(THETA_SH_4, k=2, lam=1.0)
        H_high = build_cost_hamiltonian(THETA_SH_4, k=2, lam=10.0)
        # ZZ cross terms scale with lam — the maximum abs coeff should be larger
        assert max(abs(c) for c in H_high.coeffs) > max(abs(c) for c in H_low.coeffs)


class TestBuildMixerHamiltonian:
    def test_returns_hamiltonian(self):
        import pennylane as qp

        H = build_mixer_hamiltonian(4)
        assert isinstance(H, qp.Hamiltonian)

    def test_n_terms_equals_n_materials(self):
        H = build_mixer_hamiltonian(4)
        assert len(H.coeffs) == 4

    def test_all_unit_coeffs(self):
        H = build_mixer_hamiltonian(4)
        assert all(abs(c - 1.0) < 1e-10 for c in H.coeffs)


# ---------------------------------------------------------------------------
# QAOA run
# ---------------------------------------------------------------------------


class TestRunQAOA:
    @pytest.fixture(scope="class")
    @classmethod
    def result(cls):
        return run_qaoa(
            THETA_SH_4,
            k=2,
            p=1,
            n_optimizer_steps=15,
            n_seeds=1,
            verbose=False,
        )

    def test_returns_qaoa_result(self, result):
        assert isinstance(result, QAOAResult)

    def test_energy_finite(self, result):
        assert np.isfinite(result.energy)

    def test_gamma_shape(self, result):
        assert result.gamma.shape == (1,)  # p=1

    def test_beta_shape(self, result):
        assert result.beta.shape == (1,)  # p=1

    def test_selected_indices_count(self, result):
        # Should select exactly k=2 materials
        assert len(result.selected_indices) == 2

    def test_selected_indices_valid(self, result):
        N = len(THETA_SH_4)
        for idx in result.selected_indices:
            assert 0 <= idx < N

    def test_selected_theta_sh_finite(self, result):
        assert np.isfinite(result.selected_theta_sh)

    def test_metadata_fields(self, result):
        assert result.p == 1
        assert result.k == 2
        assert result.n_materials == 4

    def test_k_ge_n_raises(self):
        with pytest.raises(ValueError, match="k="):
            run_qaoa(
                THETA_SH_4, k=4, p=1, n_optimizer_steps=5, n_seeds=1, verbose=False
            )


# ---------------------------------------------------------------------------
# Classical baseline
# ---------------------------------------------------------------------------


class TestClassicalGreedy:
    def test_returns_indices(self):
        indices = classical_greedy(THETA_SH_4, k=2)
        assert len(indices) == 2

    def test_selects_top_k(self):
        # Greedy should select the two largest θ_SH: indices 2 (0.40) and 0 (0.35)
        indices = set(classical_greedy(THETA_SH_4, k=2))
        assert indices == {0, 2}

    def test_k_1_selects_best(self):
        idx = classical_greedy(THETA_SH_4, k=1)
        assert idx == [2]  # CrTe2 has max θ_SH = 0.40


# ---------------------------------------------------------------------------
# Landscape utilities
# ---------------------------------------------------------------------------


class TestQAOALandscape:
    def test_evaluate_cost_matches_run_shape(self):
        params = np.array([0.5, 1.0])
        e = evaluate_qaoa_cost(THETA_SH_4, params, k=2, p=1, lam=5.0)
        assert np.isfinite(e)

    def test_landscape_grid_shape(self):
        grid = qaoa_landscape_grid(
            THETA_SH_4,
            k=2,
            lam=5.0,
            n_gamma=5,
            n_beta=4,
        )
        assert grid.energies.shape == (5, 4)
        assert np.all(np.isfinite(grid.energies))

    def test_find_landscape_minima(self):
        grid = qaoa_landscape_grid(
            THETA_SH_4,
            k=2,
            lam=5.0,
            n_gamma=7,
            n_beta=7,
        )
        minima = find_landscape_minima(grid, neighborhood=3, max_minima=3)
        assert len(minima) >= 1
        assert all(len(m) == 3 for m in minima)

    def test_param_history_recorded(self):
        result = run_qaoa(
            THETA_SH_4,
            k=2,
            p=1,
            n_optimizer_steps=10,
            n_seeds=1,
            verbose=False,
            record_param_history=True,
        )
        assert len(result.param_history) > 0
        assert result.param_history[0].shape == (2,)


# ---------------------------------------------------------------------------
# Hyperparameter sweep (#21)
# ---------------------------------------------------------------------------


class TestQAOASweep:
    def test_empty_grid_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            run_qaoa_sweep(
                THETA_SH_4, k=2, p_values=(), lam_values=(5.0,), step_values=(5,)
            )

    def test_tiny_grid_rows_and_seeds(self):
        sweep = run_qaoa_sweep(
            THETA_SH_4,
            k=2,
            formulas=["A", "B", "C", "D"],
            p_values=(1,),
            lam_values=(1.0, 5.0),
            step_values=(8,),
            n_seeds=2,
            step_size=0.3,
            rng_seed=0,
            verbose=False,
        )
        assert len(sweep.summary) == 2
        assert len(sweep.seeds) == 4
        greedy = float(np.sum(THETA_SH_4[classical_greedy(THETA_SH_4, 2)]))
        assert sweep.greedy_theta_sh == pytest.approx(greedy)
        for row in sweep.summary:
            assert row["n_seeds"] == 2
            assert row["k"] == 2
            assert np.isfinite(row["best_theta_sh"])
            assert row["gap_to_greedy"] == pytest.approx(greedy - row["best_theta_sh"])
            assert row["published_config"] == 0
            assert len(row["oracle_id"]) == 12

    def test_csv_roundtrip(self, tmp_path):
        sweep = run_qaoa_sweep(
            THETA_SH_4,
            k=2,
            p_values=(1,),
            lam_values=(5.0,),
            step_values=(6,),
            n_seeds=1,
            verbose=False,
        )
        path, seeds_path = save_qaoa_sweep(
            sweep, tmp_path / "s.csv", tmp_path / "seeds.csv"
        )
        loaded = load_qaoa_sweep(path)
        assert len(loaded) == 1
        assert float(loaded[0]["best_theta_sh"]) == pytest.approx(
            sweep.summary[0]["best_theta_sh"]
        )
        assert seeds_path is not None
        assert seeds_path.is_file()
        best = sweep_best_row(loaded)
        assert best["p"] == "1"

    def test_published_config_helper(self):
        assert is_published_qaoa_config(
            p=1, lam=6.0, n_optimizer_steps=300, n_seeds=5, step_size=0.3
        )
        assert not is_published_qaoa_config(
            p=1, lam=5.0, n_optimizer_steps=300, n_seeds=5, step_size=0.3
        )

    def test_run_qaoa_records_seed_stats(self):
        res = run_qaoa(
            THETA_SH_4, k=2, p=1, n_optimizer_steps=8, n_seeds=3, verbose=False
        )
        assert len(res.seed_theta_sh) == 3
        assert len(res.seed_energies) == 3
        assert np.isfinite(res.mean_theta_sh)
        assert res.n_evals >= 1

    def test_plot_qaoa_sweep_two_panels(self, tmp_path):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        from spinq_vqe.utils import plot_qaoa_sweep

        rows = [
            {
                "p": 1,
                "lam": 2.0,
                "n_optimizer_steps": 300,
                "best_theta_sh": 1.0,
                "greedy_theta_sh": 4.0,
            },
            {
                "p": 1,
                "lam": 6.0,
                "n_optimizer_steps": 300,
                "best_theta_sh": 3.0,
                "greedy_theta_sh": 4.0,
            },
            {
                "p": 1,
                "lam": 6.0,
                "n_optimizer_steps": 100,
                "best_theta_sh": 2.0,
                "greedy_theta_sh": 4.0,
            },
        ]
        fig = plot_qaoa_sweep(rows, save_path=str(tmp_path / "s.png"))
        assert len(fig.axes) == 2
        matplotlib.pyplot.close(fig)


def test_committed_qaoa_sweep_csv_if_present():
    """When the artifact is committed, it must include the published config."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data" / "qaoa_sweep.csv"
    if not path.is_file():
        pytest.skip("qaoa_sweep.csv not generated yet")
    rows = load_qaoa_sweep(path)
    assert rows
    published = [
        r
        for r in rows
        if int(r["p"]) in (1, 2, 3)
        and abs(float(r["lam"]) - 6.0) < 1e-12
        and int(r["n_optimizer_steps"]) == 300
        and int(r["n_seeds"]) == 5
    ]
    assert len(rows) == 22
    assert len(published) == 3
    greedy = float(published[0]["greedy_theta_sh"])
    assert greedy == pytest.approx(4.258934, rel=1e-4)
    p_map = {int(r["p"]): float(r["best_theta_sh"]) for r in published}
    assert p_map[1] == pytest.approx(3.049394, rel=1e-4)
    assert p_map[2] == pytest.approx(3.049394, rel=1e-4)
    assert p_map[3] == pytest.approx(-0.451042, rel=1e-4)
    best = sweep_best_row(rows)
    assert int(best["p"]) == 1
    assert float(best["lam"]) == pytest.approx(5.0)
    assert int(best["n_optimizer_steps"]) == 300
    assert float(best["best_theta_sh"]) == pytest.approx(3.570124, rel=1e-4)
    assert float(best["gap_to_greedy"]) == pytest.approx(0.688810, rel=1e-4)
    assert {r["oracle_id"] for r in rows} == {"c732a02c1736"}
    seeds_path = path.parent / "qaoa_sweep_seeds.csv"
    assert seeds_path.is_file()
    seed_rows = seeds_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(seed_rows) == 1 + 22 * 5
