# Issue Labels

Pastel label palette for spinq-vqe, per issue #12. Create them once via
GitHub -> Settings -> Labels, or with the `gh` commands below (`--force` makes them
idempotent).

## Palette

| Label | Color | Description |
|-------|-------|-------------|
| `bug` | `##EBD8DC` | Something is broken |
| `science` | `##C7E4CA` | New scientific analysis or result |
| `enhancement` | `##DBD3DC` | Improvement to existing functionality |
| `documentation` | `##F4ECC8` | Docs fixes or additions |
| `notebook` | `##F0D9CC` | Notebook-specific issue |
| `data` | `##D4E8F4` | Data pipeline or reproducibility |
| `test` | `##DCE8D4` | Test coverage |
| `ci` | `##E4DCF0` | CI/CD workflows |
| `chore` | `##EBEBEB` | Housekeeping, no functional change |
| `needs-triage` | `##F5E6D0` | Needs initial assessment |
| `needs-discussion` | `##F0ECD8` | Design decision required |
| `good first issue` | `##C7E4CA` | Good entry point for new contributors |
| `wontfix` | `##F0F0F0` | Out of scope or declined |
| `blocked` | `##EBD8DC` | Waiting on another issue |
| `kagome` | `##DBE8DC` | Kagome lattice / Hamiltonian specific |
| `vqe` | `##D8E4EB` | VQE algorithm specific |
| `qaoa` | `##E4D8EB` | QAOA / material selection specific |
| `entanglement` | `##EBE8D8` | Entanglement analysis specific |
| `dmrg` | `##D8EBE4` | DMRG comparison (TeNPy) |
| `nqs` | `##E8EBD8` | Neural Quantum States (NetKet) |
| `barren-plateau` | `##EBD8D8` | Barren plateau / gradient analysis |
| `materials-project` | `##D8E8EB` | Materials Project API / data |
| `publication` | `##F0D9CC` | Paper-related, pre-submission |

## Create all

```bash
gh label create "bug" --color "#EBD8DC" --description "Something is broken" --force
gh label create "science" --color "#C7E4CA" --description "New scientific analysis or result" --force
gh label create "enhancement" --color "#DBD3DC" --description "Improvement to existing functionality" --force
gh label create "documentation" --color "#F4ECC8" --description "Docs fixes or additions" --force
gh label create "notebook" --color "#F0D9CC" --description "Notebook-specific issue" --force
gh label create "data" --color "#D4E8F4" --description "Data pipeline or reproducibility" --force
gh label create "test" --color "#DCE8D4" --description "Test coverage" --force
gh label create "ci" --color "#E4DCF0" --description "CI/CD workflows" --force
gh label create "chore" --color "#EBEBEB" --description "Housekeeping, no functional change" --force
gh label create "needs-triage" --color "#F5E6D0" --description "Needs initial assessment" --force
gh label create "needs-discussion" --color "#F0ECD8" --description "Design decision required" --force
gh label create "good first issue" --color "#C7E4CA" --description "Good entry point for new contributors" --force
gh label create "wontfix" --color "#F0F0F0" --description "Out of scope or declined" --force
gh label create "blocked" --color "#EBD8DC" --description "Waiting on another issue" --force
gh label create "kagome" --color "#DBE8DC" --description "Kagome lattice / Hamiltonian specific" --force
gh label create "vqe" --color "#D8E4EB" --description "VQE algorithm specific" --force
gh label create "qaoa" --color "#E4D8EB" --description "QAOA / material selection specific" --force
gh label create "entanglement" --color "#EBE8D8" --description "Entanglement analysis specific" --force
gh label create "dmrg" --color "#D8EBE4" --description "DMRG comparison (TeNPy)" --force
gh label create "nqs" --color "#E8EBD8" --description "Neural Quantum States (NetKet)" --force
gh label create "barren-plateau" --color "#EBD8D8" --description "Barren plateau / gradient analysis" --force
gh label create "materials-project" --color "#D8E8EB" --description "Materials Project API / data" --force
gh label create "publication" --color "#F0D9CC" --description "Paper-related, pre-submission" --force
```
