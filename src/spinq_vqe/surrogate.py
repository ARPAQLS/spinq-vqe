"""
surrogate.py
------------
Classical MLP surrogate for spin Hall angle (θ_SH) prediction.

Used as the oracle for the SOC QAOA material-selection optimizer.

The surrogate is trained on Materials Project structure descriptors plus a
committed, illustrative θ_SH target vector. The target values reproduce the
NB04 workflow; they are not a row-wise set of verified measurements. See
``data/theta_sh_sources.md`` for the provenance contract and audit.

Pipeline
--------
1. ``load_theta_sh_data()``  — load committed CSV (default for NB04)
2. ``load_mp_data()``        — fetch from Materials Project API (refresh only)
   or ``load_mock_data()``   — offline fallback for unit tests
2. ``build_features()``     — extract numerical descriptors from raw MP records
3. ``train_surrogate()``    — fit sklearn MLPRegressor + StandardScaler
4. ``predict()``            — predict θ_SH for new compositions

Dependencies
------------
- Core: numpy, scipy (always available)
- Optional: mp-api (Materials Project API client)
- Optional: scikit-learn (MLP surrogate). Falls back to a simple linear
  ridge regression (numpy-only) if sklearn is not installed.

References
----------
- Materials Project: https://materialsproject.org
- Sinova et al. (2015) Rev. Mod. Phys. 87, 1213 — spin Hall effects
- Blöchl et al. (1994) PRB 50, 17953 — PAW method (MP DFT basis)
"""

from __future__ import annotations

import csv
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports (graceful fallback if not installed)
# ---------------------------------------------------------------------------

try:
    from mp_api.client import MPRester  # type: ignore
    MP_API_AVAILABLE = True
except ImportError:
    MPRester = None  # type: ignore
    MP_API_AVAILABLE = False

try:
    from sklearn.neural_network import MLPRegressor  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    from sklearn.model_selection import cross_val_score  # type: ignore
    SKLEARN_AVAILABLE = True
except ImportError:
    MLPRegressor = None  # type: ignore
    StandardScaler = None  # type: ignore
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class MaterialRecord:
    """Single material entry with SOC-relevant descriptors."""

    mp_id: str
    """Materials Project ID (e.g. 'mp-989807' for Mn₃Sn)."""

    formula: str
    """Reduced chemical formula."""

    crystal_system: str
    """Crystal system: cubic, hexagonal, trigonal, etc."""

    z_max: int
    """Atomic number of the heaviest element (proxy for SOC strength)."""

    n_elements: int
    """Number of distinct elements."""

    space_group: int
    """International space group number (1–230)."""

    ahc: float
    """Anomalous Hall conductivity σ_AH (S/cm). From MP or literature."""

    theta_sh: float
    """Dimensionless θ_SH oracle target; provenance is tracked separately."""

    source: str = "unknown"
    """Descriptor source: 'mp_api', 'csv', or 'mock'."""


@dataclass
class SurrogateDataset:
    """Dataset of material records for surrogate training."""

    records: list[MaterialRecord] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.records)

    @property
    def formulas(self) -> list[str]:
        return [r.formula for r in self.records]

    @property
    def theta_sh_values(self) -> np.ndarray:
        return np.array([r.theta_sh for r in self.records])


@dataclass
class TrainedSurrogate:
    """Container for a fitted surrogate model."""

    model: Any
    """Fitted sklearn MLPRegressor or numpy ridge model."""

    scaler: Any
    """Fitted feature scaler (StandardScaler or identity)."""

    feature_names: list[str]
    """Feature column names (for inspection)."""

    cv_r2: float = float("nan")
    """5-fold cross-validation R² score on training data."""

    sklearn: bool = False
    """True if sklearn MLPRegressor; False if numpy ridge fallback."""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Repo-root path: src/spinq_vqe/surrogate.py → parents[2] == repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THETA_SH_CSV = _REPO_ROOT / "data" / "mp_theta_sh.csv"

CSV_COLUMNS = [
    "mp_id", "formula", "crystal_system", "space_group", "z_max", "n_elements",
    "ahc", "theta_sh", "theta_sh_source", "band_gap", "is_magnetic", "source",
]

# 12 reproducibility targets for NB04. These are illustrative oracle inputs,
# not verified row-wise literature measurements (MP has no θ_SH field).
CURATED_ORACLE: list[dict[str, Any]] = [
    {"formula": "Mn3Sn",   "theta_sh":  0.35, "ahc": 200.0},
    {"formula": "Pt",      "theta_sh":  0.08, "ahc":   0.0},
    {"formula": "W",       "theta_sh": -0.33, "ahc":   0.0},
    {"formula": "Ta",      "theta_sh": -0.12, "ahc":   0.0},
    {"formula": "Pd",      "theta_sh":  0.01, "ahc":   0.0},
    {"formula": "Au",      "theta_sh":  0.11, "ahc":   0.0},
    {"formula": "Co2MnGa", "theta_sh":  0.20, "ahc": 1600.0},
    {"formula": "Fe3Sn",   "theta_sh":  0.25, "ahc": 450.0},
    {"formula": "IrMn3",   "theta_sh":  0.18, "ahc":   0.0},
    {"formula": "CrTe2",   "theta_sh":  0.40, "ahc": 320.0},
    {"formula": "MnPt",    "theta_sh":  0.15, "ahc": 150.0},
    {"formula": "Bi2Se3",  "theta_sh":  3.50, "ahc":   0.0},
]

# Offline test fallback (no CSV, no API).
_MOCK_DATA: list[dict] = [
    {"mp_id": "mp-mock", "formula": e["formula"], "crystal_system": "unknown",
     "z_max": 50, "n_elements": 2, "space_group": 1,
     "ahc": e["ahc"], "theta_sh": e["theta_sh"]}
    for e in CURATED_ORACLE
]

_CRYSTAL_SYSTEM_MAP = {
    "cubic": 0, "hexagonal": 1, "trigonal": 2, "tetragonal": 3,
    "orthorhombic": 4, "monoclinic": 5, "triclinic": 6,
}


def _crystal_system_str(symmetry: Any) -> str:
    if symmetry is None:
        return "unknown"
    cs = symmetry.crystal_system
    if hasattr(cs, "value"):
        return str(cs.value).lower()
    return str(cs).lower()


def _record_to_csv_row(
    r: MaterialRecord,
    *,
    theta_sh_source: str = "illustrative_oracle",
    band_gap: float | None = None,
    is_magnetic: bool | None = None,
) -> dict:
    return {
        "mp_id": r.mp_id,
        "formula": r.formula,
        "crystal_system": r.crystal_system,
        "space_group": r.space_group,
        "z_max": r.z_max,
        "n_elements": r.n_elements,
        "ahc": r.ahc,
        "theta_sh": r.theta_sh,
        "theta_sh_source": theta_sh_source,
        "band_gap": "" if band_gap is None else band_gap,
        "is_magnetic": "" if is_magnetic is None else is_magnetic,
        "source": r.source,
    }


def _csv_row_to_record(row: dict[str, str]) -> MaterialRecord:
    return MaterialRecord(
        mp_id=row["mp_id"],
        formula=row["formula"],
        crystal_system=row["crystal_system"],
        z_max=int(float(row["z_max"])),
        n_elements=int(float(row["n_elements"])),
        space_group=int(float(row["space_group"])),
        ahc=float(row["ahc"]),
        theta_sh=float(row["theta_sh"]),
        source=row.get("source", "csv"),
    )


def save_theta_sh_csv(
    dataset: SurrogateDataset,
    path: Path | str = DEFAULT_THETA_SH_CSV,
    *,
    extra: list[dict] | None = None,
) -> Path:
    """Write a SurrogateDataset to ``data/mp_theta_sh.csv``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extras = extra or [{}] * len(dataset.records)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r, meta in zip(dataset.records, extras):
            writer.writerow(_record_to_csv_row(
                r,
                theta_sh_source=meta.get("theta_sh_source", "illustrative_oracle"),
                band_gap=meta.get("band_gap"),
                is_magnetic=meta.get("is_magnetic"),
            ))
    return path


def load_theta_sh_csv(path: Path | str = DEFAULT_THETA_SH_CSV) -> SurrogateDataset:
    """Load the committed θ_SH dataset from CSV."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"theta_SH CSV not found: {path}")
    records: list[MaterialRecord] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append(_csv_row_to_record(row))
    return SurrogateDataset(records=records)


def load_theta_sh_data(path: Path | str | None = None) -> SurrogateDataset:
    """
    Load the committed illustrative θ_SH oracle (primary entry point for NB04).

    Reads ``data/mp_theta_sh.csv``. Falls back to ``load_mock_data()`` with a
    warning if the CSV is missing (unit tests only).
    """
    csv_path = Path(path) if path is not None else DEFAULT_THETA_SH_CSV
    try:
        ds = load_theta_sh_csv(csv_path)
        print(f"Loaded {ds.n_samples} materials from {csv_path.name}.")
        return ds
    except FileNotFoundError:
        warnings.warn(
            f"{csv_path} not found — using offline mock data. "
            "Run: python scripts/fetch_mp_theta_sh.py to generate the CSV.",
            stacklevel=2,
        )
        return load_mock_data()


def load_mock_data() -> SurrogateDataset:
    """
    Return the illustrative NB04 oracle fallback (offline, no API key needed).

    Includes 12 representative spintronic materials with fixed target values.
    Useful for tests and workflow reproduction, not as a measurement table.

    Returns
    -------
    SurrogateDataset
    """
    records = [
        MaterialRecord(source="mock", **{k: v for k, v in d.items()})
        for d in _MOCK_DATA
    ]
    return SurrogateDataset(records=records)


def fetch_curated_mp_dataset(api_key: str | None = None) -> tuple[SurrogateDataset, list[dict]]:
    """
    Fetch MP structure descriptors for the 12 curated spintronic materials.

    θ_SH and AHC come from ``CURATED_ORACLE`` (MP does not expose θ_SH).
    For each formula, picks the lowest-energy-above-hull MP entry.

    Parameters
    ----------
    api_key : str, optional
        Materials Project API key. Defaults to ``MP_API_KEY`` env var.

    Returns
    -------
    dataset : SurrogateDataset
    extra : list of dict
        Per-row metadata (band_gap, is_magnetic, theta_sh_source) for CSV export.
    """
    if not MP_API_AVAILABLE:
        raise ImportError(
            "mp-api is not installed. Run: pip install mp-api"
        )

    key = api_key or os.environ.get("MP_API_KEY")
    if not key:
        raise ValueError(
            "Materials Project API key required. Set MP_API_KEY in .env or pass api_key=."
        )

    from pymatgen.core import Element

    records: list[MaterialRecord] = []
    extra_rows: list[dict] = []

    with MPRester(key) as mpr:
        for entry in CURATED_ORACLE:
            formula = entry["formula"]
            docs = mpr.materials.summary.search(
                formula=formula,
                fields=[
                    "material_id", "formula_pretty", "elements", "nelements",
                    "symmetry", "band_gap", "is_magnetic", "energy_above_hull",
                ],
                num_chunks=1,
                chunk_size=10,
            )
            if not docs:
                warnings.warn(f"No MP entry found for formula {formula!r}")
                continue

            best = min(docs, key=lambda d: (d.energy_above_hull or 999.0))
            z_max = max(Element(e).Z for e in best.elements)
            sg = best.symmetry.number if best.symmetry else 1

            records.append(MaterialRecord(
                mp_id=str(best.material_id),
                formula=best.formula_pretty,
                crystal_system=_crystal_system_str(best.symmetry),
                z_max=z_max,
                n_elements=best.nelements,
                space_group=sg,
                ahc=float(entry["ahc"]),
                theta_sh=float(entry["theta_sh"]),
                source="mp_api",
            ))
            extra_rows.append({
                "theta_sh_source": "illustrative_oracle",
                "band_gap": best.band_gap,
                "is_magnetic": best.is_magnetic,
            })

    if not records:
        raise RuntimeError("No materials fetched from Materials Project.")

    print(f"Fetched {len(records)} curated materials from Materials Project.")
    return SurrogateDataset(records=records), extra_rows


def load_mp_data(api_key: str | None = None) -> SurrogateDataset:
    """
    Fetch the curated spintronic dataset from the Materials Project API.

    Prefer ``load_theta_sh_data()`` for notebook runs (uses committed CSV).
    Use this only to refresh data or when building the CSV for the first time.

    Parameters
    ----------
    api_key : str, optional
        Materials Project API key. Defaults to ``MP_API_KEY`` environment variable.

    Returns
    -------
    SurrogateDataset
        Records with MP descriptors and illustrative θ_SH oracle targets.

    Raises
    ------
    ImportError
        If mp-api is not installed.
    ValueError
        If no API key is available.
    """
    dataset, _ = fetch_curated_mp_dataset(api_key)
    return dataset


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "z_max",           # heaviest atomic number → SOC strength
    "n_elements",      # compound complexity
    "crystal_encoded", # crystal system (ordinal)
    "space_group",     # space group number
    "ahc",             # anomalous Hall conductivity (S/cm)
    "z_max_sq",        # z_max² — SOC scales as Z⁴
]


def build_features(dataset: SurrogateDataset) -> np.ndarray:
    """
    Extract numerical feature matrix from a SurrogateDataset.

    Parameters
    ----------
    dataset : SurrogateDataset

    Returns
    -------
    np.ndarray, shape (n_samples, n_features)
        Feature matrix. Column order matches ``FEATURE_NAMES``.
    """
    rows = []
    for r in dataset.records:
        crystal_enc = _CRYSTAL_SYSTEM_MAP.get(r.crystal_system.lower(), 6)
        rows.append([
            r.z_max,
            r.n_elements,
            crystal_enc,
            r.space_group,
            r.ahc,
            r.z_max ** 2,
        ])
    return np.array(rows, dtype=float)


# ---------------------------------------------------------------------------
# Surrogate model
# ---------------------------------------------------------------------------


def train_surrogate(
    dataset: SurrogateDataset,
    hidden_layer_sizes: tuple[int, ...] = (64, 32),
    max_iter: int = 2000,
    random_state: int = 42,
    cv_folds: int = 5,
) -> TrainedSurrogate:
    """
    Train an MLP surrogate on θ_SH from the dataset.

    If scikit-learn is available, uses ``MLPRegressor``.
    Falls back to numpy ridge regression otherwise.

    Parameters
    ----------
    dataset : SurrogateDataset
    hidden_layer_sizes : tuple of int
        MLP hidden layer sizes. Default (64, 32) is sufficient for this problem.
    max_iter : int
        Maximum MLP training iterations.
    random_state : int
    cv_folds : int
        Number of cross-validation folds for R² reporting.

    Returns
    -------
    TrainedSurrogate
    """
    X = build_features(dataset)
    y = dataset.theta_sh_values

    if len(dataset.records) < 4:
        raise ValueError(
            f"Need at least 4 samples to train a surrogate, got {len(dataset.records)}."
        )

    if SKLEARN_AVAILABLE:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # early_stopping needs an internal validation split; disable for small datasets
        use_early_stopping = len(dataset.records) >= 30

        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=use_early_stopping,
        )
        model.fit(X_scaled, y)

        cv_r2 = float("nan")
        min_cv_samples = cv_folds * 4  # need enough per fold for reliable CV
        if len(dataset.records) >= min_cv_samples:
            scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring="r2")
            cv_r2 = float(scores.mean())

        return TrainedSurrogate(
            model=model, scaler=scaler, feature_names=FEATURE_NAMES,
            cv_r2=cv_r2, sklearn=True,
        )

    else:
        warnings.warn(
            "scikit-learn not installed. Falling back to numpy ridge regression. "
            "Install scikit-learn for better surrogate quality: pip install scikit-learn"
        )
        # Numpy ridge regression fallback
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        X_scaled = (X - mu) / sigma

        # Ridge: (XᵀX + λI)⁻¹ Xᵀy
        lam = 1e-3
        XtX = X_scaled.T @ X_scaled
        w = np.linalg.solve(XtX + lam * np.eye(X_scaled.shape[1]), X_scaled.T @ y)

        class _RidgeModel:
            def __init__(self, w, mu, sigma):
                self.w, self.mu, self.sigma = w, mu, sigma
            def predict(self, X_new):
                return ((X_new - self.mu) / self.sigma) @ self.w

        model = _RidgeModel(w, mu, sigma)

        class _IdentityScaler:
            def transform(self, X): return X

        return TrainedSurrogate(
            model=model, scaler=_IdentityScaler(),
            feature_names=FEATURE_NAMES, cv_r2=float("nan"), sklearn=False,
        )


def predict(
    surrogate: TrainedSurrogate,
    records: list[MaterialRecord],
) -> np.ndarray:
    """
    Predict θ_SH for a list of MaterialRecord instances.

    Parameters
    ----------
    surrogate : TrainedSurrogate
    records : list of MaterialRecord

    Returns
    -------
    np.ndarray, shape (n_records,)
        Predicted θ_SH values.
    """
    dataset = SurrogateDataset(records=records)
    X = build_features(dataset)
    if surrogate.sklearn:
        X_scaled = surrogate.scaler.transform(X)
        return surrogate.model.predict(X_scaled)
    else:
        return surrogate.model.predict(X)


def surrogate_summary(surrogate: TrainedSurrogate) -> None:
    """Print a one-line summary of the trained surrogate."""
    kind = "sklearn MLP" if surrogate.sklearn else "numpy ridge"
    r2_str = f"{surrogate.cv_r2:.3f}" if not np.isnan(surrogate.cv_r2) else "n/a"
    print(f"Surrogate: {kind}  |  features: {len(surrogate.feature_names)}  |  CV R²: {r2_str}")
