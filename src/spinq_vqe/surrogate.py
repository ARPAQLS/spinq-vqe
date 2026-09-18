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
3. ``build_features()``      — extract numerical descriptors from raw MP records
4. ``train_surrogate()``     — fit sklearn Pipeline (scaler+MLP) + CV / hold-out metrics
5. ``predict()`` / ``predict_oracle()`` — in-sample or out-of-fold θ_SH oracles

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
from typing import Any, Sequence

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
    from sklearn.metrics import r2_score  # type: ignore
    from sklearn.model_selection import (  # type: ignore
        KFold,
        LeaveOneOut,
        cross_val_predict,
    )
    from sklearn.neural_network import MLPRegressor  # type: ignore
    from sklearn.pipeline import Pipeline  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    SKLEARN_AVAILABLE = True
except ImportError:
    MLPRegressor = None  # type: ignore
    StandardScaler = None  # type: ignore
    Pipeline = None  # type: ignore
    SKLEARN_AVAILABLE = False
    r2_score = None  # type: ignore


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
class SurrogateMetrics:
    """Train / CV / hold-out diagnostics for a surrogate fit."""

    n_samples: int
    train_rmse: float = float("nan")
    train_r2: float = float("nan")
    cv_rmse: float = float("nan")
    cv_r2: float = float("nan")
    cv_strategy: str = "none"
    """e.g. ``kfold-5``, ``loocv``, or ``none``."""
    n_hold_out: int = 0
    hold_out_rmse: float = float("nan")
    hold_out_r2: float = float("nan")
    hold_out_formulas: tuple[str, ...] = ()
    oracle_mode: str = "in_sample"
    """Oracle policy used for QAOA weights: ``in_sample`` or ``loocv`` / ``kfold``."""


@dataclass
class TrainedSurrogate:
    """Container for a fitted surrogate model."""

    model: Any
    """Fitted sklearn MLPRegressor (or Pipeline) or numpy ridge model."""

    scaler: Any
    """Fitted feature scaler (StandardScaler or identity). Unused if model is a Pipeline."""

    feature_names: list[str]
    """Feature column names (for inspection)."""

    cv_r2: float = float("nan")
    """Cross-validated R² (also mirrored on ``metrics.cv_r2``)."""

    sklearn: bool = False
    """True if sklearn MLPRegressor; False if numpy ridge fallback."""

    train_rmse: float = float("nan")
    """In-sample training RMSE."""

    cv_rmse: float = float("nan")
    """Cross-validated RMSE (out-of-fold)."""

    metrics: SurrogateMetrics | None = None
    """Full train / CV / hold-out metric bundle when available."""

    is_pipeline: bool = False
    """True when ``model`` is an sklearn ``Pipeline`` (scaler + estimator)."""


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

# Curated spintronic Phase-A oracle (≥30). θ_SH/AHC are illustrative workflow
# targets (MP has no θ_SH field) — see data/theta_sh_sources.md. The first
# QAOA_POOL_SIZE entries are the historical NB04 candidate pool (k-from-N=12).
CURATED_ORACLE: list[dict[str, Any]] = [
    # --- historical QAOA pool (N=12); values frozen for continuity ---
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
    # --- Phase A expansion: heavy metals / TMD / Heusler / TI family ---
    {"formula": "Ir",      "theta_sh":  0.05, "ahc":   0.0},
    {"formula": "Rh",      "theta_sh":  0.03, "ahc":   0.0},
    {"formula": "Ru",      "theta_sh": -0.05, "ahc":   0.0},
    {"formula": "Mo",      "theta_sh": -0.08, "ahc":   0.0},
    {"formula": "Nb",      "theta_sh": -0.04, "ahc":   0.0},
    {"formula": "Hf",      "theta_sh": -0.15, "ahc":   0.0},
    {"formula": "Re",      "theta_sh": -0.20, "ahc":   0.0},
    {"formula": "Os",      "theta_sh":  0.06, "ahc":   0.0},
    {"formula": "Cu",      "theta_sh":  0.02, "ahc":   0.0},
    {"formula": "Ag",      "theta_sh":  0.04, "ahc":   0.0},
    {"formula": "Bi",      "theta_sh":  0.50, "ahc":   0.0},
    {"formula": "Sb",      "theta_sh":  0.25, "ahc":   0.0},
    {"formula": "Bi2Te3",  "theta_sh":  2.00, "ahc":   0.0},
    {"formula": "Sb2Te3",  "theta_sh":  1.50, "ahc":   0.0},
    {"formula": "MoTe2",   "theta_sh":  0.80, "ahc":   0.0},
    {"formula": "WTe2",    "theta_sh":  1.00, "ahc":   0.0},
    {"formula": "Mn3Ge",   "theta_sh":  0.30, "ahc": 180.0},
    {"formula": "Mn3Ga",   "theta_sh":  0.28, "ahc": 220.0},
    {"formula": "Co2MnSi", "theta_sh":  0.12, "ahc": 900.0},
    {"formula": "FeRh",    "theta_sh":  0.10, "ahc":  80.0},
]

QAOA_POOL_SIZE = 12
QAOA_POOL_FORMULAS: tuple[str, ...] = tuple(
    e["formula"] for e in CURATED_ORACLE[:QAOA_POOL_SIZE]
)
MIN_CURATED_N = 30

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

    Mirrors ``CURATED_ORACLE`` (≥30 Phase-A materials) with fixed target values.
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


def filter_by_formulas(
    dataset: SurrogateDataset,
    formulas: Sequence[str],
) -> SurrogateDataset:
    """
    Return a dataset restricted to ``formulas`` (order preserved).

    Formula matching is exact on ``MaterialRecord.formula`` as stored in the
    CSV / MP fetch (e.g. ``MnGaCo2`` may appear for oracle key ``Co2MnGa``).
    Pass the formulas as they appear in ``dataset.formulas``.
    """
    wanted = list(formulas)
    by_formula = {r.formula: r for r in dataset.records}
    missing = [f for f in wanted if f not in by_formula]
    if missing:
        # Allow oracle-key aliases used in CURATED_ORACLE vs MP pretty formulas.
        alias = {
            "Co2MnGa": "MnGaCo2",
            "IrMn3": "Mn3Ir",
            "MoTe2": "Te2Mo",
            "WTe2": "Te2W",
            "Co2MnSi": "MnCo2Si",
        }
        resolved: list[MaterialRecord] = []
        still_missing: list[str] = []
        for f in wanted:
            if f in by_formula:
                resolved.append(by_formula[f])
            elif alias.get(f) in by_formula:
                resolved.append(by_formula[alias[f]])
            else:
                still_missing.append(f)
        if still_missing:
            raise KeyError(
                f"Formulas not in dataset: {still_missing}. "
                f"Available: {sorted(by_formula)}"
            )
        return SurrogateDataset(records=resolved)

    return SurrogateDataset(records=[by_formula[f] for f in wanted])


def qaoa_pool_dataset(dataset: SurrogateDataset | None = None) -> SurrogateDataset:
    """
    Historical NB04 QAOA candidate pool (N=12).

    Surrogate training may use the full Phase-A CSV (≥30 rows); QAOA / greedy /
    SA stay on this pool so the Hilbert space remains 2^12.
    """
    ds = dataset if dataset is not None else load_theta_sh_data()
    return filter_by_formulas(ds, QAOA_POOL_FORMULAS)


def fetch_curated_mp_dataset(api_key: str | None = None) -> tuple[SurrogateDataset, list[dict]]:
    """
    Fetch MP structure descriptors for all ``CURATED_ORACLE`` materials.

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
# Surrogate model + evaluation
# ---------------------------------------------------------------------------

DEFAULT_SURROGATE_METRICS_CSV = _REPO_ROOT / "data" / "surrogate_metrics.csv"
ORACLE_MODES = ("in_sample", "loocv", "kfold")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-15:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _sklearn_pipeline(
    *,
    hidden_layer_sizes: tuple[int, ...],
    max_iter: int,
    random_state: int,
    early_stopping: bool,
) -> Any:
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "mlp",
            MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation="relu",
                solver="adam",
                max_iter=max_iter,
                random_state=random_state,
                early_stopping=early_stopping,
            ),
        ),
    ])


def _resolve_cv(
    n_samples: int, cv_folds: int, strategy: str, random_state: int = 0
):
    """Return ``(cv_splitter_or_int, strategy_label)``."""
    strat = strategy.lower().strip()
    if strat == "auto":
        strat = "loocv" if n_samples < 15 else "kfold"
    if strat == "loocv":
        if not SKLEARN_AVAILABLE:
            return max(2, min(n_samples, cv_folds)), "loocv-approx"
        return LeaveOneOut(), "loocv"
    if strat == "kfold":
        folds = max(2, min(cv_folds, n_samples))
        if n_samples < folds:
            folds = max(2, n_samples)
        if SKLEARN_AVAILABLE:
            return (
                KFold(n_splits=folds, shuffle=True, random_state=random_state),
                f"kfold-{folds}",
            )
        return folds, f"kfold-{folds}"
    raise ValueError(f"Unknown CV strategy {strategy!r}; use 'auto', 'kfold', or 'loocv'.")


def split_hold_out(
    dataset: SurrogateDataset,
    hold_out_frac: float = 0.2,
    *,
    hold_out_formulas: Sequence[str] | None = None,
    random_state: int = 0,
    min_train: int = 5,
) -> tuple[SurrogateDataset, SurrogateDataset]:
    """
    Split a dataset into train and hold-out sets.

    Parameters
    ----------
    dataset : SurrogateDataset
    hold_out_frac : float
        Fraction held out when ``hold_out_formulas`` is not given (rounded up,
        at least 1 when ``n > min_train``).
    hold_out_formulas : sequence of str, optional
        Explicit hold-out formulas (aliases resolved like ``filter_by_formulas``).
    random_state : int
        RNG seed for fractional splits.
    min_train : int
        Minimum training rows required.

    Returns
    -------
    train, hold_out : SurrogateDataset
    """
    n = dataset.n_samples
    if n < min_train + 1:
        raise ValueError(
            f"Need at least {min_train + 1} samples for a hold-out split, got {n}."
        )

    if hold_out_formulas is not None:
        hold = filter_by_formulas(dataset, hold_out_formulas)
        hold_forms = set(hold.formulas)
        train_recs = [r for r in dataset.records if r.formula not in hold_forms]
        if len(train_recs) < min_train:
            raise ValueError(
                f"Hold-out left only {len(train_recs)} train rows "
                f"(need ≥{min_train})."
            )
        if not hold.records:
            raise ValueError("Hold-out formula list matched no records.")
        return SurrogateDataset(records=train_recs), hold

    rng = np.random.default_rng(random_state)
    n_hold = max(1, int(np.ceil(n * hold_out_frac)))
    if n - n_hold < min_train:
        n_hold = n - min_train
    if n_hold < 1:
        raise ValueError(
            f"hold_out_frac={hold_out_frac} leaves fewer than {min_train} train rows."
        )
    idx = rng.permutation(n)
    hold_idx = set(int(i) for i in idx[:n_hold])
    hold_recs = [dataset.records[i] for i in range(n) if i in hold_idx]
    train_recs = [dataset.records[i] for i in range(n) if i not in hold_idx]
    return SurrogateDataset(records=train_recs), SurrogateDataset(records=hold_recs)


def _ridge_fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    lam: float = 1e-3,
) -> np.ndarray:
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8
    Xs = (X_train - mu) / sigma
    w = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ y_train)
    return ((X_test - mu) / sigma) @ w


def _cross_val_predict_ridge(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
) -> np.ndarray:
    n = len(y)
    folds = max(2, min(n_splits, n))
    # Deterministic contiguous folds for the numpy fallback.
    indices = np.arange(n)
    fold_ids = np.array_split(indices, folds)
    preds = np.empty(n, dtype=float)
    for fold in fold_ids:
        test = np.asarray(fold, dtype=int)
        train = np.setdiff1d(indices, test, assume_unique=False)
        preds[test] = _ridge_fit_predict(X[train], y[train], X[test])
    return preds


def cross_validate_surrogate(
    dataset: SurrogateDataset,
    *,
    hidden_layer_sizes: tuple[int, ...] = (64, 32),
    max_iter: int = 2000,
    random_state: int = 42,
    cv_folds: int = 5,
    cv_strategy: str = "auto",
) -> tuple[np.ndarray, SurrogateMetrics]:
    """
    Out-of-fold predictions and CV metrics (no train/test leakage).

    Uses an sklearn ``Pipeline(StandardScaler, MLPRegressor)`` so scaling is
    fit inside each fold. Falls back to numpy ridge with manual folds.
    """
    X = build_features(dataset)
    y = dataset.theta_sh_values
    n = len(y)
    if n < 4:
        raise ValueError(f"Need at least 4 samples for CV, got {n}.")

    cv, label = _resolve_cv(n, cv_folds, cv_strategy, random_state)

    if SKLEARN_AVAILABLE:
        pipe = _sklearn_pipeline(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=False,  # avoid nested validation inside CV
        )
        oof = cross_val_predict(pipe, X, y, cv=cv)
        # Fresh clone metrics via scoring helpers on OOF predictions
        cv_rmse = _rmse(y, oof)
        cv_r2 = float(r2_score(y, oof)) if SKLEARN_AVAILABLE else _r2(y, oof)
    else:
        n_splits = int(cv) if isinstance(cv, int) else cv_folds
        oof = _cross_val_predict_ridge(X, y, n_splits)
        cv_rmse = _rmse(y, oof)
        cv_r2 = _r2(y, oof)

    metrics = SurrogateMetrics(
        n_samples=n,
        cv_rmse=cv_rmse,
        cv_r2=cv_r2,
        cv_strategy=label,
    )
    return oof, metrics


def train_surrogate(
    dataset: SurrogateDataset,
    hidden_layer_sizes: tuple[int, ...] = (64, 32),
    max_iter: int = 2000,
    random_state: int = 42,
    cv_folds: int = 5,
    cv_strategy: str = "auto",
    hold_out_frac: float | None = None,
    hold_out_formulas: Sequence[str] | None = None,
    compute_cv: bool = True,
) -> TrainedSurrogate:
    """
    Train an MLP surrogate on θ_SH from the dataset.

    If scikit-learn is available, uses a ``Pipeline(StandardScaler, MLPRegressor)``
    so inference applies the same scaling as training. Falls back to numpy ridge
    otherwise.

    Cross-validation uses out-of-fold predictions (no scaler leakage). Optional
    hold-out evaluation trains only on the train split and scores the held-out
    rows separately; the returned model is the **train-split** fit when hold-out
    is requested.

    Parameters
    ----------
    dataset : SurrogateDataset
    hidden_layer_sizes, max_iter, random_state
        MLP hyperparameters.
    cv_folds : int
        K for k-fold when ``cv_strategy`` is ``kfold`` or ``auto`` with n≥15.
    cv_strategy : {'auto', 'kfold', 'loocv'}
        ``auto`` → LOOCV for n<15, else 5-fold (or ``cv_folds``).
    hold_out_frac, hold_out_formulas
        Optional hold-out split for generalization metrics.
    compute_cv : bool
        If True, attach CV metrics (slightly more compute).

    Returns
    -------
    TrainedSurrogate
    """
    hold_metrics_extra: dict[str, Any] = {}
    train_ds = dataset
    if hold_out_frac is not None or hold_out_formulas is not None:
        frac = 0.2 if hold_out_frac is None else hold_out_frac
        train_ds, hold_ds = split_hold_out(
            dataset,
            hold_out_frac=frac,
            hold_out_formulas=hold_out_formulas,
            random_state=random_state,
        )
        hold_metrics_extra = {
            "n_hold_out": hold_ds.n_samples,
            "hold_out_formulas": tuple(hold_ds.formulas),
            "_hold_ds": hold_ds,
        }

    if train_ds.n_samples < 4:
        raise ValueError(
            f"Need at least 4 samples to train a surrogate, got {train_ds.n_samples}."
        )

    X = build_features(train_ds)
    y = train_ds.theta_sh_values

    cv_bundle: SurrogateMetrics | None = None
    if compute_cv:
        _, cv_bundle = cross_validate_surrogate(
            train_ds,
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            cv_folds=cv_folds,
            cv_strategy=cv_strategy,
        )

    if SKLEARN_AVAILABLE:
        # Final fit: early stopping only when enough data and no hold-out
        # (hold-out already provides an external check).
        use_early_stopping = (
            train_ds.n_samples >= 30 and hold_out_frac is None and hold_out_formulas is None
        )
        pipe = _sklearn_pipeline(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=use_early_stopping,
        )
        pipe.fit(X, y)
        train_pred = pipe.predict(X)
        train_rmse = _rmse(y, train_pred)
        train_r2 = float(r2_score(y, train_pred))

        hold_rmse = float("nan")
        hold_r2 = float("nan")
        if "_hold_ds" in hold_metrics_extra:
            hold_ds = hold_metrics_extra.pop("_hold_ds")
            hp = pipe.predict(build_features(hold_ds))
            hold_rmse = _rmse(hold_ds.theta_sh_values, hp)
            hold_r2 = float(r2_score(hold_ds.theta_sh_values, hp))

        metrics = SurrogateMetrics(
            n_samples=train_ds.n_samples,
            train_rmse=train_rmse,
            train_r2=train_r2,
            cv_rmse=cv_bundle.cv_rmse if cv_bundle else float("nan"),
            cv_r2=cv_bundle.cv_r2 if cv_bundle else float("nan"),
            cv_strategy=cv_bundle.cv_strategy if cv_bundle else "none",
            n_hold_out=hold_metrics_extra.get("n_hold_out", 0),
            hold_out_rmse=hold_rmse,
            hold_out_r2=hold_r2,
            hold_out_formulas=hold_metrics_extra.get("hold_out_formulas", ()),
            oracle_mode="in_sample",
        )
        return TrainedSurrogate(
            model=pipe,
            scaler=pipe.named_steps["scaler"],
            feature_names=FEATURE_NAMES,
            cv_r2=metrics.cv_r2,
            sklearn=True,
            train_rmse=train_rmse,
            cv_rmse=metrics.cv_rmse,
            metrics=metrics,
            is_pipeline=True,
        )

    warnings.warn(
        "scikit-learn not installed. Falling back to numpy ridge regression. "
        "Install scikit-learn for better surrogate quality: pip install scikit-learn",
        stacklevel=2,
    )
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    X_scaled = (X - mu) / sigma
    lam = 1e-3
    w = np.linalg.solve(X_scaled.T @ X_scaled + lam * np.eye(X_scaled.shape[1]), X_scaled.T @ y)

    class _RidgeModel:
        def __init__(self, w, mu, sigma):
            self.w, self.mu, self.sigma = w, mu, sigma

        def predict(self, X_new):
            return ((X_new - self.mu) / self.sigma) @ self.w

    model = _RidgeModel(w, mu, sigma)

    class _IdentityScaler:
        def transform(self, X):
            return X

    train_pred = model.predict(X)
    train_rmse = _rmse(y, train_pred)
    train_r2 = _r2(y, train_pred)

    hold_rmse = float("nan")
    hold_r2 = float("nan")
    if "_hold_ds" in hold_metrics_extra:
        hold_ds = hold_metrics_extra.pop("_hold_ds")
        hp = model.predict(build_features(hold_ds))
        hold_rmse = _rmse(hold_ds.theta_sh_values, hp)
        hold_r2 = _r2(hold_ds.theta_sh_values, hp)

    metrics = SurrogateMetrics(
        n_samples=train_ds.n_samples,
        train_rmse=train_rmse,
        train_r2=train_r2,
        cv_rmse=cv_bundle.cv_rmse if cv_bundle else float("nan"),
        cv_r2=cv_bundle.cv_r2 if cv_bundle else float("nan"),
        cv_strategy=cv_bundle.cv_strategy if cv_bundle else "none",
        n_hold_out=hold_metrics_extra.get("n_hold_out", 0),
        hold_out_rmse=hold_rmse,
        hold_out_r2=hold_r2,
        hold_out_formulas=hold_metrics_extra.get("hold_out_formulas", ()),
        oracle_mode="in_sample",
    )
    return TrainedSurrogate(
        model=model,
        scaler=_IdentityScaler(),
        feature_names=FEATURE_NAMES,
        cv_r2=metrics.cv_r2,
        sklearn=False,
        train_rmse=train_rmse,
        cv_rmse=metrics.cv_rmse,
        metrics=metrics,
        is_pipeline=False,
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
    if surrogate.is_pipeline:
        return np.asarray(surrogate.model.predict(X), dtype=float)
    if surrogate.sklearn:
        X_scaled = surrogate.scaler.transform(X)
        return np.asarray(surrogate.model.predict(X_scaled), dtype=float)
    return np.asarray(surrogate.model.predict(X), dtype=float)


def predict_oracle(
    dataset: SurrogateDataset,
    mode: str = "in_sample",
    *,
    hidden_layer_sizes: tuple[int, ...] = (64, 32),
    max_iter: int = 2000,
    random_state: int = 42,
    cv_folds: int = 5,
) -> tuple[np.ndarray, SurrogateMetrics]:
    """
    Build θ_SH oracle weights for QAOA / greedy / SA.

    Modes
    -----
    ``in_sample``
        Fit on all rows, predict the same rows (pipeline debugging; default for
        published NB04 QAOA continuity).
    ``loocv`` / ``kfold``
        Out-of-fold predictions — a more honest oracle when generalization matters.
    """
    mode_norm = mode.lower().strip()
    if mode_norm not in ORACLE_MODES:
        raise ValueError(f"mode must be one of {ORACLE_MODES}, got {mode!r}")

    if mode_norm == "in_sample":
        sr = train_surrogate(
            dataset,
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            cv_folds=cv_folds,
            cv_strategy="auto",
            compute_cv=True,
        )
        preds = predict(sr, dataset.records)
        metrics = sr.metrics or SurrogateMetrics(n_samples=dataset.n_samples)
        metrics.oracle_mode = "in_sample"
        return preds, metrics

    strategy = "loocv" if mode_norm == "loocv" else "kfold"
    oof, metrics = cross_validate_surrogate(
        dataset,
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
        cv_folds=cv_folds,
        cv_strategy=strategy,
    )
    # Attach train metrics from a full in-sample fit for comparison
    sr = train_surrogate(
        dataset,
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
        compute_cv=False,
    )
    metrics.train_rmse = sr.train_rmse
    metrics.train_r2 = sr.metrics.train_r2 if sr.metrics else float("nan")
    metrics.oracle_mode = mode_norm
    return oof, metrics


def save_surrogate_metrics(
    metrics: SurrogateMetrics,
    path: Path | str = DEFAULT_SURROGATE_METRICS_CSV,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a one-row (plus optional extras) metrics CSV for NB04 reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "n_samples": metrics.n_samples,
        "train_rmse": f"{metrics.train_rmse:.6f}" if np.isfinite(metrics.train_rmse) else "",
        "train_r2": f"{metrics.train_r2:.6f}" if np.isfinite(metrics.train_r2) else "",
        "cv_rmse": f"{metrics.cv_rmse:.6f}" if np.isfinite(metrics.cv_rmse) else "",
        "cv_r2": f"{metrics.cv_r2:.6f}" if np.isfinite(metrics.cv_r2) else "",
        "cv_strategy": metrics.cv_strategy,
        "n_hold_out": metrics.n_hold_out,
        "hold_out_rmse": (
            f"{metrics.hold_out_rmse:.6f}" if np.isfinite(metrics.hold_out_rmse) else ""
        ),
        "hold_out_r2": (
            f"{metrics.hold_out_r2:.6f}" if np.isfinite(metrics.hold_out_r2) else ""
        ),
        "hold_out_formulas": ";".join(metrics.hold_out_formulas),
        "oracle_mode": metrics.oracle_mode,
    }
    if extra:
        for k, v in extra.items():
            row[k] = v
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return path


def surrogate_summary(surrogate: TrainedSurrogate) -> None:
    """Print train / CV / hold-out metrics for a trained surrogate."""
    kind = "sklearn MLP" if surrogate.sklearn else "numpy ridge"
    m = surrogate.metrics
    cv_r2 = m.cv_r2 if m else surrogate.cv_r2
    cv_rmse = m.cv_rmse if m else surrogate.cv_rmse
    train_rmse = m.train_rmse if m else surrogate.train_rmse
    strat = m.cv_strategy if m else "n/a"
    r2_str = f"{cv_r2:.3f}" if np.isfinite(cv_r2) else "n/a"
    cv_rmse_str = f"{cv_rmse:.4f}" if np.isfinite(cv_rmse) else "n/a"
    train_str = f"{train_rmse:.4f}" if np.isfinite(train_rmse) else "n/a"
    print(
        f"Surrogate: {kind}  |  features: {len(surrogate.feature_names)}  |  "
        f"train RMSE: {train_str}  |  CV RMSE: {cv_rmse_str} ({strat})  |  CV R²: {r2_str}"
    )
    if m and m.n_hold_out:
        ho = f"{m.hold_out_rmse:.4f}" if np.isfinite(m.hold_out_rmse) else "n/a"
        print(
            f"  hold-out: n={m.n_hold_out}  RMSE={ho}  "
            f"formulas={list(m.hold_out_formulas)}"
        )
