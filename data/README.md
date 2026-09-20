# Data artifacts (`spinq-vqe/data`)

## `mp_theta_sh.csv` (NB04 oracle)

Phase-A curated spintronic set (**32** materials). Materials Project supplies
structure descriptors (`mp_id`, crystal system, space group, …). The `theta_sh`
column is an **illustrative oracle** for workflow reproducibility — see
[`theta_sh_sources.md`](theta_sh_sources.md) and
[`theta_sh_provenance.csv`](theta_sh_provenance.csv).

| Role | Size | Notes |
|------|------|-------|
| Full training corpus | 32 | Surrogate fit + k-fold CV / hold-out diagnostics |
| QAOA / greedy / SA pool | 12 | First 12 historical formulas; Hilbert space `2^12` |

Refresh MP descriptors (requires `MP_API_KEY`):

```bash
pip install -e ".[data]"
python scripts/fetch_mp_theta_sh.py
```

Regenerate surrogate evaluation artifacts (no API key):

```bash
python scripts/evaluate_surrogate.py
```

Notebook runs use the committed CSV — **no API key**.

### Family coverage (illustrative)

- Heavy metals / 5d–4d: Pt, W, Ta, Pd, Au, Ir, Rh, Ru, Mo, Nb, Hf, Re, Os, Cu, Ag, Bi, Sb
- AFM / Kagome-related: Mn3Sn, Fe3Sn, Mn3Ir, MnPt, Mn3Ge, Mn3Ga, FeRh
- Heusler: MnGaCo2 (Co2MnGa), MnCo2Si (Co2MnSi)
- TI / TMD: Bi2Se3, Bi2Te3, Sb2Te3, Te2Mo (MoTe2), Te2W (WTe2), CrTe2

## Other committed CSVs

| File | Notebook |
|------|----------|
| `ed_reference_energies.csv` | NB01 / NB05 |
| `vqe_results.csv`, `vqe_seeds_n9.csv`, `vqe_scaling.csv` | NB02 / NB05 |
| `qaoa_results.csv` | NB04 (published single-config totals) |
| `qaoa_sweep.csv`, `qaoa_sweep_seeds.csv` | NB04 (#21 λ / budget / depth sweep) |
| `qaoa_screening.csv` | NB04 (#23 train/pool screening evaluation) |
| `surrogate_metrics.csv` | NB04 (train / CV / hold-out + pool LOOCV) |
| `dmrg_reference_energies.csv` | NB06 |
| `method_comparison.csv`, `nqs_*_history_*.csv` | NB07 |

`n_samples` in `surrogate_metrics.csv` is the **training split** (25 rows after
the 20% hold-out), not the 32-row corpus. Regenerate the CSV and
`figures/surrogate_predictions.png` with `python scripts/evaluate_surrogate.py`.

Regenerate the QAOA sweep (does not touch `qaoa_results.csv`):

```bash
python scripts/run_qaoa_sweep.py
```

Committed default grid: 22 cells (110 seed rows). Best QAOA is 3.570
(p=1, λ=5) vs greedy 4.259. `best_theta_sh` is the best-cost seed.

Regenerate the screening evaluation (does not touch `qaoa_results.csv`):

```bash
python scripts/evaluate_qaoa_screening.py
```

Train = #20 25-row complement; pool = historical 12; unseen in pool =
W, Pd, MnPt, Bi2Se3. On the screening oracle, greedy/SA reach 2.900 vs
best QAOA 2.505 (p=3). `total_label` is the post-hoc sum of committed
CSV labels for the selected triple (Greedy_label = 4.250, not the
in-sample 4.259).
