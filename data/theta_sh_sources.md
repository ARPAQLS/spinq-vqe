# θ_SH provenance and NB04 scope

## Status contract

`data/mp_theta_sh.csv` is the committed input used to reproduce NB04. Its
`theta_sh` column is an **oracle target**, not automatically a measured material
property. Every row must have one of these `theta_sh_source` values:

- `sourced_primary`: the exact value, sign, sample, method, and conditions are
  traceable to a primary reference.
- `illustrative_oracle`: the value is retained only to reproduce the workflow
  and must not be presented as a literature measurement.

The row-level ledger is `data/theta_sh_provenance.csv`. A broad review citation
does not upgrade a row to `sourced_primary`.

## Current audit

All 12 committed targets are currently `illustrative_oracle`. Three spot checks
already demonstrate why:

| CSV row | Oracle target | Primary result | Primary reference | Audit result |
|---|---:|---:|---|---|
| Mn3Sn | 0.35 | 0.053 ± 0.024 | `10.1103/PhysRevB.99.184425` | Different magnitude |
| MnGaCo2 / Co2MnGa | +0.20 | -0.19 ± 0.04 | `10.1103/PhysRevB.103.L041114` | Opposite sign |
| Bi2Se3 | 3.50 | 0.0093 ± 0.0013 | `10.1103/PhysRevB.90.094403` | Method- and stack-specific; not interchangeable |

The remaining rows have no verified row-level primary source in this audit.
They remain illustrative rather than being covered by generic review citations.

## Materials Project boundary

Materials Project supplies the structural descriptors and selected computed
properties in `mp_theta_sh.csv`. It does not supply the committed θ_SH targets.
Matching a chemical formula to an MP structure does not establish that a
thin-film or device measurement applies to that polymorph.

## What NB04 demonstrates

NB04 demonstrates the pipeline

`12 fixed oracle targets -> in-sample surrogate -> k=3 QUBO -> QAOA/classical comparison`.

The reported totals and selected triples are results on that precise committed
oracle. They do not establish DFT-computed θ_SH, production materials discovery,
a transferable ML model, a global physical optimum, or quantum advantage.

If a target is replaced with a sourced value, the surrogate fit, classical
baselines, QAOA runs, tables, and figures must be regenerated together.
