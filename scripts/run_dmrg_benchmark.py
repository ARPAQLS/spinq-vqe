"""
Generate DMRG reference CSV and figures for NB06.

Run from repo root:
    pip install -e ".[dmrg]"
    python scripts/run_dmrg_benchmark.py
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spinq_vqe import dmrg, entanglement

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
FIG = REPO / "figures"

N_CELLS = [3, 4, 6, 8]
# N=9 is exact at χ≈4; use N=18 where truncation is nontrivial.
CHI_SWEEP_N_CELLS = 6
CHI_SWEEP = [8, 16, 32, 64, 128, 200, 300, 400]
CHI_MAX = 400


def _load_vqe_energies() -> dict[int, float]:
    out: dict[int, float] = {}
    with (DATA / "vqe_scaling.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["E_VQE"] in ("N/A", ""):
                continue
            out[int(row["N"])] = float(row["E_VQE"])
    return out


def _load_ed_energies() -> dict[int, float]:
    out: dict[int, float] = {}
    with (DATA / "ed_reference_energies.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[int(row["n_sites"])] = float(row["E0_normalized"])
    with (DATA / "vqe_scaling.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["E_ED"] not in ("N/A", ""):
                out[int(row["N"])] = float(row["E_ED"])
    return out


def main() -> None:
    warnings.filterwarnings("ignore")

    print("Validating TeNPy Hamiltonian against PennyLane ED at N=9...")
    diff = dmrg.validate_hamiltonian_against_pennylane(3)
    print(f"  max |H_tenpy - H_pl| = {diff:.3e}")

    print("Running DMRG benchmarks...")
    results = []
    n9_entropy: dmrg.DMRGResult | None = None
    for n_cells in N_CELLS:
        res = dmrg.run_dmrg(
            n_cells,
            chi_max=CHI_MAX,
            compute_entropies=(n_cells == 3),
        )
        results.append(res)
        if n_cells == 3:
            n9_entropy = res
        print(
            f"  N={res.n_sites:>2}  E0={res.e0:.8f}  chi={res.chi}  "
            f"trunc={res.truncation_error:.2e}"
        )

    csv_path = dmrg.save_dmrg_reference_csv(results, DATA / "dmrg_reference_energies.csv")
    print(f"Saved -> {csv_path.relative_to(REPO)}")

    ed = _load_ed_energies()
    vqe = _load_vqe_energies()
    dmrg_e = {r.n_sites: r.e0 for r in results}

    # Comparison bar/table figure
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sizes = sorted(set(ed) | set(vqe) | set(dmrg_e))
    x = np.arange(len(sizes))
    width = 0.25
    ed_vals = [ed.get(n, np.nan) for n in sizes]
    dmrg_vals = [dmrg_e.get(n, np.nan) for n in sizes]
    vqe_vals = [vqe.get(n, np.nan) for n in sizes]
    ax.bar(x - width, ed_vals, width, label="ED", color="#EBD8DC")
    ax.bar(x, dmrg_vals, width, label="DMRG", color="#C97B7B")
    ax.bar(x + width, vqe_vals, width, label="VQE best", color="#7EB8D4")
    ax.set_xticks(x, [str(n) for n in sizes])
    ax.set_xlabel("System size N (sites)")
    ax.set_ylabel("E₀ (normalized per site)")
    ax.set_title("ED vs DMRG vs VQE", fontweight="semibold", color="#333")
    ax.legend()
    plt.tight_layout()
    out_cmp = FIG / "dmrg_vqe_comparison.png"
    plt.savefig(out_cmp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_cmp.relative_to(REPO)}")

    # Chi convergence at N=18 (N=9 is exact at χ≈4 — flat residual)
    n_chi = 3 * CHI_SWEEP_N_CELLS
    print(f"Chi convergence sweep at N={n_chi}...")
    chi_results = dmrg.run_dmrg_chi_sweep(
        CHI_SWEEP_N_CELLS, CHI_SWEEP, max_sweeps=60
    )
    e_ref = chi_results[-1].e0
    chis = [r.chi_max for r in chi_results]
    residuals = [max(abs(r.e0 - e_ref), 1e-16) for r in chi_results]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(chis, [r.e0 for r in chi_results], "o-", color="#8B6FA8", lw=2, ms=6)
    ax.axhline(e_ref, color="#C97B7B", ls="--", lw=1.2, label=f"E₀(χ={CHI_MAX})")
    ax.set_xlabel("Bond dimension χ_max")
    ax.set_ylabel("E₀ (normalized per site)")
    ax.set_title(f"DMRG energy vs χ (N={n_chi})", fontweight="semibold", color="#333")
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.semilogy(chis, residuals, "o-", color="#8B6FA8", lw=2, ms=6)
    ax.set_xlabel("Bond dimension χ_max")
    ax.set_ylabel("|E₀(χ) − E₀(χ_max)|")
    ax.set_title(f"Convergence residual (N={n_chi})", fontweight="semibold", color="#333")
    ax.set_xscale("log", base=2)
    plt.tight_layout()
    out_chi = FIG / "dmrg_chi_convergence.png"
    plt.savefig(out_chi, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_chi.relative_to(REPO)}")

    # Entanglement profile at N=9 (bits). VQE RDM is reliable for |A|<=N/2 (NB03).
    if n9_entropy and n9_entropy.entropies:
        cuts = list(range(1, n9_entropy.n_sites))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(
            cuts,
            n9_entropy.entropies,
            "o-",
            color="#8B6FA8",
            lw=2,
            ms=5,
            label="DMRG MPS",
        )
        sv_path = DATA / "statevector_hea_best.npy"
        if sv_path.exists():
            sv = np.load(sv_path)
            vqe_profile = entanglement.entanglement_profile(sv, n9_entropy.n_sites)
            ax.plot(
                vqe_profile["subsystem_sizes"],
                vqe_profile["entropies"],
                "s--",
                color="#7EB8D4",
                lw=2,
                ms=5,
                label="VQE (NB03, |A|≤N/2)",
            )
        ax.set_xlabel("Bond cut index")
        ax.set_ylabel("Entanglement entropy (bits)")
        ax.set_title("Bipartite entropy profile (N=9)", fontweight="semibold", color="#333")
        ax.legend()
        plt.tight_layout()
        out_ent = FIG / "dmrg_entanglement_profile.png"
        plt.savefig(out_ent, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {out_ent.relative_to(REPO)}")

    # Regenerate scaling_energy.png with DMRG reference line
    print("Updating figures/scaling_energy.png with DMRG reference...")
    sizes = [9, 12]
    err9 = abs(vqe[9] - dmrg_e[9]) / abs(dmrg_e[9]) * 100
    err12 = abs(vqe[12] - dmrg_e[12]) / abs(dmrg_e[12]) * 100
    errors = [err9, err12]
    ed_sizes = sorted(ed.keys())
    ed_e0s = [ed[n] for n in ed_sizes]
    dmrg_sizes = sorted(dmrg_e.keys())
    dmrg_e0s = [dmrg_e[n] for n in dmrg_sizes]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.plot(sizes, errors, "o-", color="#7EB8D4", lw=2, ms=8, label="COBYLA/HEA")
    ax.set_xlabel("System size N (sites)")
    ax.set_ylabel("VQE energy error vs DMRG (%)")
    ax.set_title("VQE Error vs System Size", fontweight="semibold", color="#333")
    ax.set_xticks(sizes)
    ax.set_ylim(0, max(errors) * 1.4)
    for s, e in zip(sizes, errors):
        ax.annotate(
            f"{e:.2f}%",
            (s, e),
            textcoords="offset points",
            xytext=(5, 8),
            fontsize=9,
            color="#555",
        )

    ax = axes[1]
    ax.plot(ed_sizes, ed_e0s, "s--", color="#EBD8DC", lw=2, ms=8, label="ED E₀/site")
    ax.plot(
        dmrg_sizes,
        dmrg_e0s,
        "^--",
        color="#C97B7B",
        lw=2,
        ms=8,
        label="DMRG E₀/site",
    )
    ax.plot(sizes, [vqe[9], vqe[12]], "o", color="#7EB8D4", ms=8, label="VQE best")
    ax.set_xlabel("System size N (sites)")
    ax.set_ylabel("E₀ (normalized per site)")
    ax.set_title("Ground State Energy vs System Size", fontweight="semibold", color="#333")
    ax.legend(fontsize=9)
    ax.set_xticks(sorted(set(ed_sizes) | set(sizes)))
    plt.tight_layout()
    out_scale = FIG / "scaling_energy.png"
    plt.savefig(out_scale, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_scale.relative_to(REPO)}")

    print("\nVQE error vs DMRG:")
    for n in sizes:
        print(f"  N={n:>2}  err={abs(vqe[n]-dmrg_e[n])/abs(dmrg_e[n])*100:.2f}%")


if __name__ == "__main__":
    main()
