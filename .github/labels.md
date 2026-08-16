# Label configuration for spinq-vqe
# Pastel color scheme matching the project aesthetic

## Labels

| Name | Color | Description |
|------|-------|-------------|
| bug | E8A598 | Something isn't working |
| enhancement | B8B8E8 | New feature or improvement |
| good first issue | C7E4CA | Friendly for newcomers |
| help wanted | F0D9CC | Extra attention needed |
| science | D4C5F9 | Scientific content, methods, or results |
| discussion | C5DFF9 | Open discussion topic |
| documentation | F9E8C5 | Docs, README, comments |
| testing | C5F9E8 | Tests, CI, validation |
| performance | F9C5D4 | Speed, memory, optimization |
| qaoa | E8C5F9 | QAOA-related work |
| vqe | C5E8F9 | VQE-related work |
| kagome | F9D4C5 | Kagome lattice physics |
| entanglement | D4F9C5 | Entanglement metrics, entropy |
| dmrg | F9C5E8 | DMRG / TeNPy methods |
| nqs | C5F9D4 | Neural Quantum States |

## Syncing labels to GitHub

These labels are not auto-created from this file. To sync them to the repo, run:

```bash
# Option 1: Using gh CLI (one-liner per label)
gh label create bug --color E8A598 --description "Something isn't working"
gh label create enhancement --color B8B8E8 --description "New feature or improvement"
gh label create "good first issue" --color C7E4CA --description "Friendly for newcomers"
gh label create "help wanted" --color F0D9CC --description "Extra attention needed"
gh label create science --color D4C5F9 --description "Scientific content, methods, or results"
gh label create discussion --color C5DFF9 --description "Open discussion topic"
gh label create documentation --color F9E8C5 --description "Docs, README, comments"
gh label create testing --color C5F9E8 --description "Tests, CI, validation"
gh label create performance --color F9C5D4 --description "Speed, memory, optimization"
gh label create qaoa --color E8C5F9 --description "QAOA-related work"
gh label create vqe --color C5E8F9 --description "VQE-related work"
gh label create kagome --color F9D4C5 --description "Kagome lattice physics"
gh label create entanglement --color D4F9C5 --description "Entanglement metrics, entropy"
gh label create dmrg --color F9C5E8 --description "DMRG / TeNPy methods"
gh label create nqs --color C5F9D4 --description "Neural Quantum States"
```

```bash
# Option 2: Using the sync script (if added as a workflow)
# See .github/workflows/sync-labels.yml for automated sync on push
```

> **Note:** Labels referenced in issue templates (bug, enhancement, science, documentation, testing, qaoa, vqe, kagome, entanglement, dmrg, nqs) must exist before templates can auto-apply them.
