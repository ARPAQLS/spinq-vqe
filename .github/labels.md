# Issue Labels

Pastel label palette for spinq-vqe, per issue #12.

**Source of truth:** [`.github/labels.yml`](labels.yml).
On every push to `master` that touches that file, [`.github/workflows/sync-labels.yml`](workflows/sync-labels.yml) creates or updates the labels automatically. Default GitHub labels not in the palette (`duplicate`, `help wanted`, …) are left in place (`skip-delete: true`).

## Palette

| Label | Color | Description |
|-------|-------|-------------|
| `bug` | `#EBD8DC` | Something is broken |
| `science` | `#C7E4CA` | New scientific analysis or result |
| `enhancement` | `#DBD3DC` | Improvement to existing functionality |
| `documentation` | `#F4ECC8` | Docs fixes or additions |
| `notebook` | `#F0D9CC` | Notebook-specific issue |
| `data` | `#D4E8F4` | Data pipeline or reproducibility |
| `test` | `#DCE8D4` | Test coverage |
| `ci` | `#E4DCF0` | CI/CD workflows |
| `chore` | `#EBEBEB` | Housekeeping, no functional change |
| `needs-triage` | `#F5E6D0` | Needs initial assessment |
| `needs-discussion` | `#F0ECD8` | Design decision required |
| `good first issue` | `#C7E4CA` | Good entry point for new contributors |
| `wontfix` | `#F0F0F0` | Out of scope or declined |
| `blocked` | `#EBD8DC` | Waiting on another issue |
| `kagome` | `#DBE8DC` | Kagome lattice / Hamiltonian specific |
| `vqe` | `#D8E4EB` | VQE algorithm specific |
| `qaoa` | `#E4D8EB` | QAOA / material selection specific |
| `entanglement` | `#EBE8D8` | Entanglement analysis specific |
| `dmrg` | `#D8EBE4` | DMRG comparison (TeNPy) |
| `nqs` | `#E8EBD8` | Neural Quantum States (NetKet) |
| `barren-plateau` | `#EBD8D8` | Barren plateau / gradient analysis |
| `materials-project` | `#D8E8EB` | Materials Project API / data |
| `publication` | `#F0D9CC` | Paper-related, pre-submission |

Manual `gh label create --force` is no longer required after this workflow is on `master`. To sync immediately, run **Actions → Sync labels → Run workflow**.
