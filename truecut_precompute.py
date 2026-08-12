"""Genuine time-before-renewal cutoff — feature rebuild.

Recomputes EVERY base feature at the decision date ``E_dec = E - c`` as if all
data in the window ``[E_dec, E)`` never existed. This is the honest "true cutoff":
not column masking, but a from-raw rebuild of the transaction event fields, the
history aggregates, the prior-renewal counts and the observed-listening baseline
at the shifted decision date.

Per cutoff ``c`` in {0, 7, 14, 30, 60}  (``off = (E - transaction_date).days``):
  * transaction event group  -> most recent txn with ``transaction_date <= E_dec`` (off >= c)
  * history_txn aggregates    -> txns with ``transaction_date <  E_dec`` (off > c)
  * prior_* counts            -> txns with ``transaction_date <  E_dec`` (off > c)
  * pre_* listening window     -> 30-day window ending at E_dec  (== horizon h == c)
  * obs_secs_per_active_day    -> all log days with ``date < E_dec``
  * recent_vs_life_ratio       -> pre_rate(E_dec) / obs baseline(E_dec)
  * demographics               -> static (unchanged)
``c == 0`` reproduces the engineered base features exactly (decision at E).

Outputs (all in KKBoxData/, resumable — a phase is skipped if its cache exists):
  * truecut_obs_full.parquet          per-event lifetime listening secs / active days (< E)
  * truecut_recent.parquet            per-event recent secs / active days for offsets [-c, -1]
  * truecut_base_c{c}.parquet         full base-feature matrix at E_dec (one per cutoff)
"""
from __future__ import annotations

from pathlib import Path
import gc
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA = Path("KKBoxData")
RAW = DATA / "RawData"

CUTOFFS = [0, 7, 14, 30, 60]
HORIZ = [7, 14, 30, 60]          # cutoffs with a matching listening horizon h == c
WINDOW_DAYS = 30                 # pre-window length (matches ClassEngineering OBS_LEN)

ID_COLS = ["msno", "E", "expiry_month"]
SPLIT_COLS = ["user_fold", "is_holdout", "month_idx", "is_modellable"]
TARGET_COLS = ["label", "renewed", "vol_ratio"]
DEMOG = ["city", "bd", "gender", "registered_via", "registration_init_time"]

# base_features order from feature_manifest.json (must match clf_utils.build_X)
TXN_EVENT = ["pre_list_price", "pre_amount_paid", "payment_plan_days",
             "payment_method_id", "is_auto_renew", "is_cancel"]
HIST_TXN = ["n_hist_txns", "mean_hist_price", "median_hist_paid", "cancel_rate",
            "auto_renew_rate", "mode_pay_method", "tenure_days", "recency_days",
            "has_hist_txn", "n_prior_renewals", "n_prior_cancels",
            "n_prior_autorenew", "prior_renewal_rate"]
BEHAV_PRE = ["pre_total_secs", "pre_n_days", "pre_rate"]
ENGAGE = ["obs_secs_per_active_day", "recent_vs_life_ratio"]
BASE_FEATURES = BEHAV_PRE + TXN_EVENT + HIST_TXN + ENGAGE + DEMOG

TXN_RAW_COLS = ["msno", "transaction_date", "membership_expire_date",
                "actual_amount_paid", "plan_list_price", "is_cancel",
                "is_auto_renew", "payment_plan_days", "payment_method_id"]


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Phase 0 — modellable events (the row universe every cutoff shares)
# --------------------------------------------------------------------------- #
def load_events():
    cols = (ID_COLS + SPLIT_COLS + TARGET_COLS + BASE_FEATURES
            + [f"h{h}_{s}" for h in HORIZ for s in ("secs", "ndays", "rate")])
    cols = list(dict.fromkeys(cols))
    ev = pd.read_parquet(DATA / "events_engineered.parquet", columns=cols)
    ev = ev.loc[ev["is_modellable"] == 1].reset_index(drop=True)
    ev["E"] = pd.to_datetime(ev["E"])
    log(f"modellable events: {len(ev):,}")
    return ev


# --------------------------------------------------------------------------- #
# Phase 1 — lifetime observed-listening baseline (secs / active days, date < E)
# --------------------------------------------------------------------------- #
def build_obs_full(core_events):
    out = DATA / "truecut_obs_full.parquet"
    if out.exists():
        log(f"[phase1] cache hit {out.name}")
        return pd.read_parquet(out)

    core_users = set(core_events["msno"])
    obs_parts, nb = [], 0
    for src in ["user_logs.parquet", "user_logs_v2.parquet"]:
        p = RAW / src
        if not p.exists():
            continue
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=1_000_000,
                                     columns=["msno", "date", "total_secs"]):
            ch = batch.to_pandas()
            ch = ch[ch["msno"].isin(core_users)]
            if ch.empty:
                continue
            ch["total_secs"] = ch["total_secs"].clip(lower=0, upper=86_400)
            ch["date"] = pd.to_datetime(ch["date"], format="%Y%m%d")
            ch = ch.merge(core_events[["msno", "E"]], on="msno", how="inner")
            ch = ch[ch["date"] < ch["E"]]
            if ch.empty:
                del ch
                continue
            obs_parts.append(
                ch.groupby(["msno", "E"], as_index=False)
                  .agg(obs_secs=("total_secs", "sum"), obs_ndays=("total_secs", "size")))
            del ch
            nb += 1
            if nb % 20 == 0:
                obs_parts = [pd.concat(obs_parts, ignore_index=True)
                             .groupby(["msno", "E"], as_index=False)
                             .agg(obs_secs=("obs_secs", "sum"),
                                  obs_ndays=("obs_ndays", "sum"))]
                log(f"[phase1] consolidated at batch {nb}")
        gc.collect()

    obs = (pd.concat(obs_parts, ignore_index=True)
           .groupby(["msno", "E"], as_index=False)
           .agg(obs_secs_full=("obs_secs", "sum"), obs_ndays_full=("obs_ndays", "sum")))
    del obs_parts
    gc.collect()
    pq.write_table(pa.Table.from_pandas(obs, preserve_index=False), out, compression="snappy")
    log(f"[phase1] wrote {out.name}  ({len(obs):,} events)")

    # sanity: obs_secs_full / obs_ndays_full must match the stored ratio
    chk = core_events[["msno", "E", "obs_secs_per_active_day"]].merge(obs, on=["msno", "E"], how="left")
    ratio = np.where(chk["obs_ndays_full"] > 0, chk["obs_secs_full"] / chk["obs_ndays_full"], np.nan)
    both = chk["obs_secs_per_active_day"].notna() & pd.notna(ratio)
    if both.any():
        diff = np.abs(ratio[both] - chk["obs_secs_per_active_day"][both].to_numpy())
        log(f"[phase1] obs ratio reconstruction max|Δ|={np.nanmax(diff):.4g}  "
            f"mean|Δ|={np.nanmean(diff):.4g}")
    return obs


# --------------------------------------------------------------------------- #
# Phase 2 — recent listening (offsets [-c, -1]) from the daily panel, per cutoff
# --------------------------------------------------------------------------- #
def build_recent(core_events):
    out = DATA / "truecut_recent.parquet"
    if out.exists():
        log(f"[phase2] cache hit {out.name}")
        return pd.read_parquet(out)

    keys = core_events[["msno", "E"]].copy()
    # accumulators per cutoff: (msno,E) -> secs sum / active-day count over offset >= -c
    acc = {c: [] for c in HORIZ}
    pf = pq.ParquetFile(DATA / "events_daily_panel.parquet")
    nb = 0
    for batch in pf.iter_batches(batch_size=5_000_000,
                                 columns=["msno", "E", "offset", "total_secs"]):
        ch = batch.to_pandas()
        ch["total_secs"] = ch["total_secs"].clip(lower=0, upper=86_400)
        for c in HORIZ:
            w = ch[ch["offset"] >= -c]
            if w.empty:
                continue
            acc[c].append(w.groupby(["msno", "E"], as_index=False)
                           .agg(secs=("total_secs", "sum"), ndays=("total_secs", "size")))
        nb += 1
        if nb % 8 == 0:
            for c in HORIZ:
                if acc[c]:
                    acc[c] = [pd.concat(acc[c], ignore_index=True)
                              .groupby(["msno", "E"], as_index=False)
                              .agg(secs=("secs", "sum"), ndays=("ndays", "sum"))]
            log(f"[phase2] consolidated at batch {nb}")
        del ch
    gc.collect()

    recent = keys
    for c in HORIZ:
        g = (pd.concat(acc[c], ignore_index=True)
             .groupby(["msno", "E"], as_index=False)
             .agg(**{f"recent_secs_{c}": ("secs", "sum"),
                     f"recent_ndays_{c}": ("ndays", "sum")})) if acc[c] else \
            pd.DataFrame(columns=["msno", "E", f"recent_secs_{c}", f"recent_ndays_{c}"])
        recent = recent.merge(g, on=["msno", "E"], how="left")
    for c in HORIZ:
        recent[f"recent_secs_{c}"] = recent[f"recent_secs_{c}"].fillna(0.0)
        recent[f"recent_ndays_{c}"] = recent[f"recent_ndays_{c}"].fillna(0.0)
    del acc
    gc.collect()
    pq.write_table(pa.Table.from_pandas(recent, preserve_index=False), out, compression="snappy")
    log(f"[phase2] wrote {out.name}")
    return recent


# --------------------------------------------------------------------------- #
# Phase 3 — transaction rebuild at each cutoff (event fields + history + prior)
# --------------------------------------------------------------------------- #
def load_txn(core_users):
    txn = pd.concat(
        [pq.read_table(RAW / f, columns=TXN_RAW_COLS).to_pandas()
         for f in ["transactions.parquet", "transactions_v2.parquet"]],
        ignore_index=True).drop_duplicates()
    txn = txn[txn["msno"].isin(core_users)]
    for c in ["transaction_date", "membership_expire_date"]:
        txn[c] = pd.to_datetime(txn[c], format="%Y%m%d")
    log(f"[phase3] cohort txns: {len(txn):,}")
    return txn


def txn_features_at_cutoff(tx_ev, c):
    """Rebuild the transaction-derived features at decision date E - c.

    tx_ev must carry an integer ``off = (E - transaction_date).days`` column and be
    pre-sorted by ['msno','E','transaction_date','membership_expire_date'].
    """
    # ---- event-defining transaction: most recent txn with transaction_date <= E-c ----
    ev_txn = (tx_ev[tx_ev["off"] >= c]
              .groupby(["msno", "E"], as_index=False).tail(1)
              [["msno", "E", "plan_list_price", "actual_amount_paid", "payment_plan_days",
                "payment_method_id", "is_auto_renew", "is_cancel"]]
              .rename(columns={"plan_list_price": "pre_list_price",
                               "actual_amount_paid": "pre_amount_paid"}))

    # ---- history aggregates over txns strictly before E-c ----
    hist_src = tx_ev[tx_ev["off"] > c]
    hist = hist_src.groupby(["msno", "E"]).agg(
        n_hist_txns=("transaction_date", "size"),
        first_txn_date=("transaction_date", "min"),
        last_hist_txn_date=("transaction_date", "max"),
        mean_hist_price=("plan_list_price", "mean"),
        median_hist_paid=("actual_amount_paid", "median"),
        cancel_rate=("is_cancel", "mean"),
        auto_renew_rate=("is_auto_renew", "mean"),
        n_prior_cancels=("is_cancel", "sum"),
        n_prior_autorenew=("is_auto_renew", "sum"),
        n_prior_renewals=("is_paid_renewal", "sum"),
    ).reset_index()
    mode_pay = (hist_src.groupby(["msno", "E", "payment_method_id"]).size().reset_index(name="_cnt")
                .sort_values(["msno", "E", "_cnt", "payment_method_id"],
                             ascending=[True, True, True, False])
                .groupby(["msno", "E"], as_index=False).tail(1)
                [["msno", "E", "payment_method_id"]]
                .rename(columns={"payment_method_id": "mode_pay_method"}))
    hist = hist.merge(mode_pay, on=["msno", "E"], how="left")

    E_dec = hist["E"] - pd.Timedelta(days=c)
    hist["tenure_days"] = (E_dec - hist["first_txn_date"]).dt.days
    hist["recency_days"] = (E_dec - hist["last_hist_txn_date"]).dt.days
    hist = hist.drop(columns=["first_txn_date", "last_hist_txn_date"])

    out = ev_txn.merge(hist, on=["msno", "E"], how="outer")
    out["has_hist_txn"] = out["n_hist_txns"].notna().astype("int64")
    for col in ["n_hist_txns", "n_prior_renewals", "n_prior_cancels", "n_prior_autorenew"]:
        out[col] = out[col].fillna(0)
    for col in ["n_prior_renewals", "n_prior_cancels", "n_prior_autorenew"]:
        out[col] = out[col].astype("int32")
    denom = out["n_hist_txns"].replace(0, np.nan)
    out["prior_renewal_rate"] = (out["n_prior_renewals"] / denom).fillna(0.0)
    return out


def build_cutoffs(ev, obs, recent):
    todo = [c for c in CUTOFFS if not (DATA / f"truecut_base_c{c}.parquet").exists()]
    if not todo:
        log("[phase3] all cutoff bases cached")
        return

    keep_meta = ID_COLS + SPLIT_COLS + TARGET_COLS

    # c == 0 is exactly the engineered base features.
    if 0 in todo:
        base0 = ev[keep_meta + BASE_FEATURES].copy()
        pq.write_table(pa.Table.from_pandas(base0, preserve_index=False),
                       DATA / "truecut_base_c0.parquet", compression="snappy")
        log("[phase3] wrote truecut_base_c0.parquet (copy of engineered base)")
        todo = [c for c in todo if c != 0]
    if not todo:
        return

    txn = load_txn(set(ev["msno"]))
    tx_ev = txn.merge(ev[["msno", "E"]], on="msno", how="inner")
    del txn
    gc.collect()
    tx_ev["off"] = (tx_ev["E"] - tx_ev["transaction_date"]).dt.days
    tx_ev = tx_ev[tx_ev["off"] >= 0]                       # only pre-E transactions
    tx_ev["is_paid_renewal"] = ((tx_ev["actual_amount_paid"] > 0) & (tx_ev["is_cancel"] == 0)).astype(int)
    tx_ev = tx_ev.sort_values(["msno", "E", "transaction_date", "membership_expire_date"])
    log(f"[phase3] tx_ev rows (fan-out, off>=0): {len(tx_ev):,}")

    obs_m = obs.set_index(["msno", "E"])
    for c in todo:
        txf = txn_features_at_cutoff(tx_ev, c)

        base = ev[keep_meta + DEMOG].merge(txf, on=["msno", "E"], how="left")

        # listening pre-window at E-c == horizon h == c
        base["pre_total_secs"] = ev[f"h{c}_secs"].to_numpy()
        base["pre_n_days"] = ev[f"h{c}_ndays"].to_numpy()
        base["pre_rate"] = ev[f"h{c}_rate"].to_numpy()

        # observed-listening baseline at E-c = lifetime(<E) minus recent [-c,-1]
        m = ev[["msno", "E"]].merge(obs, on=["msno", "E"], how="left") \
                              .merge(recent[["msno", "E", f"recent_secs_{c}", f"recent_ndays_{c}"]],
                                     on=["msno", "E"], how="left")
        obs_secs = m["obs_secs_full"].to_numpy() - m[f"recent_secs_{c}"].fillna(0.0).to_numpy()
        obs_nd = m["obs_ndays_full"].to_numpy() - m[f"recent_ndays_{c}"].fillna(0.0).to_numpy()
        obs_secs = np.where(np.isnan(obs_secs), np.nan, np.clip(obs_secs, 0, None))
        obs_nd = np.where(np.isnan(obs_nd), np.nan, np.clip(obs_nd, 0, None))
        with np.errstate(invalid="ignore", divide="ignore"):
            opad = np.where(obs_nd > 0, obs_secs / obs_nd, np.nan)
        base["obs_secs_per_active_day"] = opad
        with np.errstate(invalid="ignore", divide="ignore"):
            rvl = base["pre_rate"].to_numpy() / opad
        base["recent_vs_life_ratio"] = np.where(np.isfinite(rvl), rvl, np.nan)

        base = base[keep_meta + BASE_FEATURES]
        pq.write_table(pa.Table.from_pandas(base, preserve_index=False),
                       DATA / f"truecut_base_c{c}.parquet", compression="snappy")
        log(f"[phase3] wrote truecut_base_c{c}.parquet  "
            f"(has_hist_txn mean={base['has_hist_txn'].mean():.3f}, "
            f"recency_days median={base['recency_days'].median():.1f})")
        del txf, base, m
        gc.collect()
    del tx_ev, obs_m
    gc.collect()


def main():
    ev = load_events()
    obs = build_obs_full(ev)
    recent = build_recent(ev)
    build_cutoffs(ev, obs, recent)
    log("DONE truecut_precompute")


if __name__ == "__main__":
    main()
