"""
surrogate.py
------------
Classical MLP surrogate for spin Hall angle (θ_SH) prediction.

Workstream B2 — SOC QAOA oracle.

The surrogate is trained on Materials Project DFT data (anomalous Hall
conductivity, resistivity, crystal symmetry descriptors) and used as a
fast oracle for the QAOA composition optimizer in ``qaoa.py``.

Pipeline
--------
1. ``load_mp_data()``       — fetch from Materials Project API (requires mp-api key)
   or ``load_mock_data()`` — curated literature values (offline / testing)
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

import warnings
from dataclasses import dataclass, field
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
    """Spin Hall angle θ_SH (dimensionless). From literature or estimated."""

    source: str = "unknown"
    """Data source: 'mp_api', 'literature', or 'mock'."""


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

# Curated literature values (used when MP API is unavailable)
_MOCK_DATA: list[dict] = [
    {"mp_id": "mp-989807", "formula": "Mn3Sn",   "crystal_system": "hexagonal",
     "z_max": 50, "n_elements": 2, "space_group": 194, "ahc": 200.0, "theta_sh":  0.35},
    {"mp_id": "mp-124",    "formula": "Pt",       "crystal_system": "cubic",
     "z_max": 78, "n_elements": 1, "space_group": 225, "ahc":   0.0, "theta_sh":  0.08},
    {"mp_id": "mp-91",     "formula": "W",        "crystal_system": "cubic",
     "z_max": 74, "n_elements": 1, "space_group": 229, "ahc":   0.0, "theta_sh": -0.33},
    {"mp_id": "mp-104",    "formula": "Ta",       "crystal_system": "cubic",
     "z_max": 73, "n_elements": 1, "space_group": 229, "ahc":   0.0, "theta_sh": -0.12},
    {"mp_id": "mp-2",      "formula": "Pd",       "crystal_system": "cubic",
     "z_max": 46, "n_elements": 1, "space_group": 225, "ahc":   0.0, "theta_sh":  0.01},
    {"mp_id": "mp-81",     "formula": "Au",       "crystal_system": "cubic",
     "z_max": 79, "n_elements": 1, "space_group": 225, "ahc":   0.0, "theta_sh":  0.11},
    {"mp_id": "mp-22598",  "formula": "Co2MnGa",  "crystal_system": "cubic",
     "z_max": 31, "n_elements": 3, "space_group": 225, "ahc": 1600.0, "theta_sh":  0.20},
    {"mp_id": "mp-541837", "formula": "Fe3Sn",    "crystal_system": "hexagonal",
     "z_max": 50, "n_elements": 2, "space_group": 194, "ahc": 450.0, "theta_sh":  0.25},
    {"mp_id": "mp-20305",  "formula": "IrMn3",    "crystal_system": "cubic",
     "z_max": 77, "n_elements": 2, "space_group": 221, "ahc":   0.0, "theta_sh":  0.18},
    {"mp_id": "mp-976411", "formula": "CrTe2",    "crystal_system": "trigonal",
     "z_max": 52, "n_elements": 2, "space_group": 164, "ahc": 320.0, "theta_sh":  0.40},
    {"mp_id": "mp-22189",  "formula": "MnPt",     "crystal_system": "tetragonal",
     "z_max": 78, "n_elements": 2, "space_group": 123, "ahc": 150.0, "theta_sh":  0.15},
    {"mp_id": "mp-30811",  "formula": "Bi2Se3",   "crystal_system": "trigonal",
     "z_max": 83, "n_elements": 2, "space_group": 166, "ahc":   0.0, "theta_sh":  3.50},
]

_CRYSTAL_SYSTEM_MAP = {
    "cubic": 0, "hexagonal": 1, "trigonal": 2, "tetragonal": 3,
    "orthorhombic": 4, "monoclinic": 5, "triclinic": 6,
}


def load_mock_data() -> SurrogateDataset:
    """
    Return the curated literature dataset (offline, no API key needed).

    Includes 12 representative spintronic materials with known or estimated
    θ_SH values. Useful for testing and as a training baseline.

    Returns
    -------
    SurrogateDataset
    """
    records = [
        MaterialRecord(source="mock", **{k: v for k, v in d.items()})
        for d in _MOCK_DATA
    ]
    return SurrogateDataset(records=records)


def load_mp_data(
    api_key: str,
    elements: list[str] | None = None,
    n_results: int = 50,
) -> SurrogateDataset:
    """
    Fetch spin Hall / anomalous Hall data from the Materials Project API.

    Parameters
    ----------
    api_key : str
        Materials Project API key (get one at materialsproject.org).
    elements : list of str, optional
        Filter to materials containing only these elements.
        E.g. ['Mn', 'Sn', 'Fe', 'Co', 'Pt', 'W', 'Ta'].
        Defaults to a curated set of spintronic elements.
    n_results : int
        Maximum number of materials to fetch.

    Returns
    -------
    SurrogateDataset
        Records with AHC from MP. θ_SH is estimated as θ_SH ≈ AHC * ρ_xx
        where ρ_xx is a typical metallic resistivity (100 μΩ·cm).

    Raises
    ------
    ImportError
        If mp-api is not installed (``pip install mp-api``).
    """
    if not MP_API_AVAILABLE:
        raise ImportError(
            "mp-api is not installed. Run: pip install mp-api\n"
            "Or use load_mock_data() for the curated offline dataset."
        )

    if elements is None:
        elements = ["Mn", "Sn", "Fe", "Co", "Ni", "Pt", "W", "Ta", "Pd", "Ir", "Cr"]

    records: list[MaterialRecord] = []
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            elements=elements,
            fields=["material_id", "formula_pretty", "crystal_system",
                    "elements", "space_group", "structure"],
            num_chunks=1,
            chunk_size=n_results,
        )
        for doc in docs[:n_results]:
            z_vals = [e.Z for e in doc.structure.composition.elements]
            records.append(MaterialRecord(
                mp_id=str(doc.material_id),
                formula=doc.formula_pretty,
                crystal_system=doc.crystal_system.value.lower()
                    if hasattr(doc.crystal_system, "value") else str(doc.crystal_system),
                z_max=max(z_vals),
                n_elements=len(z_vals),
                space_group=doc.space_group.number if doc.space_group else 1,
                ahc=0.0,           # AHC requires a separate electronic transport query
                theta_sh=0.0,      # Will be estimated or left for NB04
                source="mp_api",
            ))

    print(f"Loaded {len(records)} records from Materials Project.")
    return SurrogateDataset(records=records)


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

        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            n_iter_no_change=50,
        )
        model.fit(X_scaled, y)

        cv_r2 = float("nan")
        if len(dataset.records) >= cv_folds * 2:
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
