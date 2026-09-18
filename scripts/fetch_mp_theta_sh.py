#!/usr/bin/env python3
"""
Refresh data/mp_theta_sh.csv from the Materials Project API.

MP provides structure descriptors; theta_SH labels come from the illustrative
CURATED_ORACLE table in spinq_vqe.surrogate (see data/theta_sh_sources.md).

Requires:
  - pip install -e ".[data]"
  - MP_API_KEY in environment or .env in repo root

Usage (from repo root, shared venv):
  python scripts/fetch_mp_theta_sh.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

_ENV = _REPO / ".env"
if _ENV.is_file():
    for line in _ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

from spinq_vqe.surrogate import (  # noqa: E402
    DEFAULT_THETA_SH_CSV,
    fetch_curated_mp_dataset,
    save_theta_sh_csv,
)


def main() -> None:
    if not os.environ.get("MP_API_KEY"):
        print("Error: MP_API_KEY not set. Add it to .env or export it.", file=sys.stderr)
        sys.exit(1)

    dataset, extra = fetch_curated_mp_dataset()
    out = save_theta_sh_csv(dataset, DEFAULT_THETA_SH_CSV, extra=extra)
    print(f"Wrote {dataset.n_samples} rows to {out}")


if __name__ == "__main__":
    main()
