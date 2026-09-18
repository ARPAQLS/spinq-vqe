# Data artifacts (`spinq-vqe/data`)

## `mp_theta_sh.csv` (NB04 oracle)

Phase-A curated spintronic set (**32** materials). Materials Project supplies
structure descriptors (`mp_id`, crystal system, space group, …). The `theta_sh`
column is an **illustrative oracle** for workflow reproducibility — see
[`theta_sh_sources.md`](theta_sh_sources.md) and
[`theta_sh_provenance.csv`](theta_sh_provenance.csv).

| Role | Size | Notes |
|------|------|-------|
| Full training corpus | 32 | Surrogate fit + CV-ready (`n ≥ 20`) |
| QAOA / greedy / SA pool | 12 | First 12 historical formulas; Hilbert space `2^12` |

Refresh MP descriptors (requires `MP_API_KEY`):

```bash
pip install -e ".[data]"
python scripts/fetch_mp_theta_sh.py
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
| `qaoa_results.csv` | NB04 |
| `dmrg_reference_energies.csv` | NB06 |
| `method_comparison.csv`, `nqs_*_history_*.csv` | NB07 |
