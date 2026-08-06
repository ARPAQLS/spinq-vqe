"""
test_dmrg.py
------------
Tests for spinq_vqe.dmrg (requires physics-tenpy).
"""

from __future__ import annotations

import pytest

tenpy = pytest.importorskip("tenpy")

from spinq_vqe import dmrg  # noqa: E402


class TestDMRGHamiltonian:
    def test_hamiltonian_matches_pennylane_n3(self):
        diff = dmrg.validate_hamiltonian_against_pennylane(3)
        assert diff < 1e-10

    def test_hamiltonian_matches_pennylane_n4(self):
        diff = dmrg.validate_hamiltonian_against_pennylane(4)
        assert diff < 1e-10


class TestDMRGEnergy:
    def test_n9_reproduces_ed_within_0_01_percent(self):
        result = dmrg.run_dmrg(3, chi_max=64, max_sweeps=40)
        ed_e0 = -1.4219039949999581
        rel_err = abs(result.e0 - ed_e0) / abs(ed_e0) * 100
        assert rel_err < 0.01
        assert result.n_sites == 9

    def test_save_and_load_csv(self, tmp_path):
        results = [dmrg.run_dmrg(3, chi_max=32, max_sweeps=20)]
        path = dmrg.save_dmrg_reference_csv(results, tmp_path / "dmrg.csv")
        loaded = dmrg.load_dmrg_reference_csv(path)
        assert loaded[9] == pytest.approx(results[0].e0, rel=1e-8)
