"""Shared utilities for the KKBox subscription-trajectory classification work.

Loaded by every Classify_*.ipynb notebook.  Centralises:
  * paths + the feature_manifest.json contract
  * data loading (memory-aware, column-pruned)
  * feature-matrix construction (datetime -> numeric, bd sanitising,
    categorical dtype handling) with a strict leakage guard
  * the user-grouped CV / holdout / walk-forward split helpers
  * metric + plotting helpers used across experiments

Nothing here trains a model; model-specific code lives in the notebooks so it
stays visible and auditable.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths + manifest contract
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "KKBoxData"
PARQUET = DATA / "events_engineered.parquet"
MANIFEST = DATA / "feature_manifest.json"
PLOTS = ROOT / "FinalPlots"
PLOTS.mkdir(exist_ok=True)

with open(MANIFEST) as _f:
    MANIFEST_D = json.load(_f)

CLASS_ORDER = MANIFEST_D["class_order"]                       # 5 incl. INSUFFICIENT_DATA
MODEL_CLASSES = [c for c in CLASS_ORDER if c != "INSUFFICIENT_DATA"]  # 4-class target
MOVEMENT_CLASSES = ["CONTRACTION", "STABLE", "EXPANSION"]     # renewer movement (stage-2)

ID_COLS = MANIFEST_D["id_cols"]
SPLIT_COLS = MANIFEST_D["split_cols"]
TARGETS = MANIFEST_D["targets"]
LEAKAGE = set(MANIFEST_D["leakage_cols"])
BASE_FEATURES = list(MANIFEST_D["base_features"])
HORIZONS = list(MANIFEST_D["horizons"])
HORIZON_FEATURES = MANIFEST_D["horizon_features"]

# Columns that are categorical in nature (low-ish cardinality codes / strings).
CAT_FEATURES = ["city", "gender", "registered_via", "payment_method_id", "mode_pay_method"]

# Reproducible class -> int code maps (stable ordering for every model/plot).
CLASS_TO_CODE = {c: i for i, c in enumerate(MODEL_CLASSES)}
CODE_TO_CLASS = {i: c for c, i in CLASS_TO_CODE.items()}
MOVE_TO_CODE = {c: i for i, c in enumerate(MOVEMENT_CLASSES)}
CODE_TO_MOVE = {i: c for c, i in MOVE_TO_CODE.items()}

RANDOM_STATE = 42

# Seeds for the repeated-holdout protocol. Every classification notebook loops over
# these; because the split is a pure function of (seed, msno), run r uses the SAME
# held-out users in notebooks 01 / 02 / 02b, so cross-model comparisons stay paired.
RUN_SEEDS = [42, 43, 44, 45, 46]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _needed_columns(feature_set: str = "base", extra=None) -> list[str]:
    """Minimal column set to load for a given experiment (keeps RAM down)."""
    cols = list(dict.fromkeys(
        ID_COLS + SPLIT_COLS
        + ["label", "renewed", "vol_ratio"]
        + BASE_FEATURES
    ))
    if feature_set in ("horizon", "base+horizon"):
        for h in HORIZONS:
            cols += HORIZON_FEATURES[str(h)]
    if extra:
        cols += [c for c in extra if c not in cols]
    return list(dict.fromkeys(cols))


def load_events(feature_set: str = "base", modellable_only: bool = False,
                extra_cols=None) -> pd.DataFrame:
    """Load the engineered event table with only the columns we need."""
    df = pd.read_parquet(PARQUET, columns=_needed_columns(feature_set, extra_cols))
    if modellable_only:
        df = df.loc[df["is_modellable"] == 1].reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Feature-matrix construction
# --------------------------------------------------------------------------- #
def feature_list(feature_set: str = "base") -> list[str]:
    if feature_set == "base":
        return list(BASE_FEATURES)
    if feature_set == "horizon":
        return [f for h in HORIZONS for f in HORIZON_FEATURES[str(h)]]
    if feature_set == "base+horizon":
        return list(BASE_FEATURES) + [f for h in HORIZONS for f in HORIZON_FEATURES[str(h)]]
    raise ValueError(f"unknown feature_set {feature_set!r}")


def build_X(df: pd.DataFrame, feature_set: str = "base",
            cat_as_category: bool = True) -> pd.DataFrame:
    """Assemble a model-ready feature matrix.

    * datetime ``registration_init_time`` -> days since epoch (float)
    * ``bd`` clipped to a plausible [0, 100] age range (else NaN)
    * categorical codes optionally cast to pandas ``category`` dtype
      (native handling for LightGBM / XGBoost / HistGB; one-hot for linear)
    * hard assertion that no leakage column slips into X
    """
    feats = feature_list(feature_set)
    leaked = LEAKAGE.intersection(feats)
    assert not leaked, f"LEAKAGE columns present in feature set: {sorted(leaked)}"

    X = df[feats].copy()

    if "registration_init_time" in X.columns:
        rit = pd.to_datetime(X["registration_init_time"])
        X["registration_init_time"] = (rit - pd.Timestamp("1970-01-01")).dt.days.astype("float64")

    if "bd" in X.columns:
        bd = X["bd"]
        X["bd"] = bd.where((bd >= 0) & (bd <= 100), np.nan)

    if cat_as_category:
        for c in CAT_FEATURES:
            if c in X.columns:
                X[c] = X[c].astype("category")
    return X


def categorical_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c in CAT_FEATURES]


def numeric_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in CAT_FEATURES]


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
def y_multiclass(df: pd.DataFrame) -> np.ndarray:
    """4-class integer target in MODEL_CLASSES order."""
    return df["label"].map(CLASS_TO_CODE).to_numpy()


def y_movement(df: pd.DataFrame) -> np.ndarray:
    """3-class movement target (only valid on renewer rows)."""
    return df["label"].map(MOVE_TO_CODE).to_numpy()


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def dev_mask(df: pd.DataFrame) -> np.ndarray:
    """Development rows: not in the current user-grouped holdout set."""
    return (df["is_holdout"] == 0).to_numpy()


def holdout_mask(df: pd.DataFrame) -> np.ndarray:
    return (df["is_holdout"] == 1).to_numpy()


def cv_folds(df: pd.DataFrame, dev_only: bool = True):
    """Yield (train_idx, val_idx) positional arrays for the user-grouped 5-fold CV.

    Uses the pre-baked ``user_fold`` (>=0) so no user spans a train/val boundary.
    Indices are positional into ``df`` (assumes a clean RangeIndex).
    """
    fold = df["user_fold"].to_numpy()
    in_dev = dev_mask(df) if dev_only else np.ones(len(df), bool)
    valid_folds = sorted(f for f in np.unique(fold) if f >= 0)
    pos = np.arange(len(df))
    for k in valid_folds:
        val = (fold == k) & in_dev
        trn = (fold != k) & (fold >= 0) & in_dev
        yield pos[trn], pos[val]


def walk_forward_splits(df: pd.DataFrame, min_train_months: int = 1):
    """Yield (train_idx, test_idx, t) expanding-window splits by ``month_idx``.

    train = month_idx <= t, test = month_idx == t + 1 (dev rows only).
    """
    m = df["month_idx"].to_numpy()
    in_dev = dev_mask(df)
    months = sorted(int(x) for x in np.unique(m))
    pos = np.arange(len(df))
    for t in months[:-1]:
        if (t - months[0] + 1) < min_train_months:
            continue
        trn = (m <= t) & in_dev
        tst = (m == t + 1) & in_dev
        if tst.sum() == 0 or trn.sum() == 0:
            continue
        yield pos[trn], pos[tst], t


def stratified_subsample(df: pd.DataFrame, y: np.ndarray, n: int,
                         random_state: int = RANDOM_STATE) -> np.ndarray:
    """Return positional indices of a label-stratified subsample of size ~n."""
    rng = np.random.default_rng(random_state)
    pos = np.arange(len(df))
    if n >= len(df):
        return pos
    frac = n / len(df)
    keep = []
    for cls in np.unique(y):
        idx = pos[y == cls]
        take = max(1, int(round(len(idx) * frac)))
        keep.append(rng.choice(idx, size=min(take, len(idx)), replace=False))
    out = np.concatenate(keep)
    rng.shuffle(out)
    return np.sort(out)


# --------------------------------------------------------------------------- #
# Repeated-holdout splits (re-derived per run, user-grouped, leakage-safe)
# --------------------------------------------------------------------------- #
def _user_uniform(users, salt: str) -> np.ndarray:
    """Deterministic ~uniform value in [0, 1) per user id, salted by `salt`.

    Pure function of (salt, user) via md5, so the same user always maps to the
    same value for a given salt -> a user's rows never straddle dev/holdout or a
    CV-fold boundary within a run.
    """
    out = np.empty(len(users), dtype="float64")
    for i, u in enumerate(users):
        h = int(hashlib.md5(f"{salt}:{u}".encode()).hexdigest(), 16)
        out[i] = (h % 1_000_000) / 1_000_000
    return out


def assign_run_split(df: pd.DataFrame, seed: int, holdout_frac: float = 0.15,
                     n_folds: int = 5) -> pd.DataFrame:
    """Overwrite ``is_holdout`` and ``user_fold`` in place with a fresh,
    user-grouped random split keyed by ``seed`` (replaces the baked-in columns).

    Leakage-safe by construction: the holdout flag and fold index are pure
    functions of (seed, msno), so every row of a user shares one holdout flag and
    one fold. Held-out users get ``user_fold == -1``. Returns ``df`` for chaining.
    """
    users = pd.Index(df["msno"].unique())
    h_ho = _user_uniform(users, f"ho:{seed}")
    h_fold = _user_uniform(users, f"fold:{seed}")
    is_ho = (h_ho < holdout_frac).astype("int8")
    fold = np.where(is_ho == 1, -1, (h_fold * n_folds).astype(int)).astype("int8")
    ho_map = pd.Series(is_ho, index=users)
    fold_map = pd.Series(fold, index=users)
    df["is_holdout"] = df["msno"].map(ho_map).astype("int8").to_numpy()
    df["user_fold"] = df["msno"].map(fold_map).astype("int8").to_numpy()
    return df


def agg_mean_se(long_df: pd.DataFrame, group_cols, value_cols) -> pd.DataFrame:
    """Aggregate repeated-run metrics into mean / std / se (ddof=1) / n_runs.

    ``long_df`` has one row per (group, run). Standard error = std / sqrt(n_runs).
    Returns a flat DataFrame (one row per group) with ``{v}_mean``, ``{v}_std``,
    ``{v}_se`` columns for each value column.
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    if isinstance(value_cols, str):
        value_cols = [value_cols]
    g = long_df.groupby(group_cols, sort=False)
    out = g.size().rename("n_runs").to_frame()
    for v in value_cols:
        out[f"{v}_mean"] = g[v].mean()
        out[f"{v}_std"] = g[v].std(ddof=1)
        out[f"{v}_se"] = out[f"{v}_std"] / np.sqrt(out["n_runs"])
    return out.reset_index()


# --------------------------------------------------------------------------- #
# Metrics + plotting
# --------------------------------------------------------------------------- #
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, accuracy_score,
    classification_report, confusion_matrix,
)


def score_block(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def per_class_f1(y_true, y_pred, class_names) -> pd.Series:
    f1 = f1_score(y_true, y_pred, average=None, labels=list(range(len(class_names))))
    return pd.Series(f1, index=class_names, name="f1")


def report_df(y_true, y_pred, class_names) -> pd.DataFrame:
    rep = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0,
    )
    return pd.DataFrame(rep).T


def plot_confusion(y_true, y_pred, class_names, ax=None, title="", normalize="true"):
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))),
                          normalize=normalize)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thr = cm.max() / 2 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j]):,}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=8)
    return ax


# --------------------------------------------------------------------------- #
# Hyper-parameter sweep result caching
# --------------------------------------------------------------------------- #
SWEEP_DIR = DATA / "sweeps"
SWEEP_DIR.mkdir(exist_ok=True)


def _json_default(o):
    """JSON encoder fallback for numpy scalars/arrays."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def sweep_path(name: str) -> Path:
    return SWEEP_DIR / f"{name}.json"


def has_sweep(name: str) -> bool:
    return sweep_path(name).exists()


def save_sweep(name: str, best_params: dict, best_score: float,
               results=None, meta: dict | None = None) -> Path:
    """Persist a sweep's best config (+ optional full trial table) to disk."""
    payload = {
        "name": name,
        "best_params": best_params,
        "best_score": float(best_score),
        "meta": meta or {},
    }
    with open(sweep_path(name), "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    if results is not None:
        pd.DataFrame(results).to_csv(SWEEP_DIR / f"{name}_trials.csv", index=False)
    return sweep_path(name)


def load_sweep(name: str):
    """Return the cached sweep payload dict, or None if not present."""
    p = sweep_path(name)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)
