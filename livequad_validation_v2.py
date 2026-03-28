from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from livequad_archive import list_snapshots, read_snapshot

DEFAULT_PROXY_MAP: Dict[str, Tuple[str, str]] = {
    "Q1": ("QQQ", "SPY"),
    "Q2": ("IWM", "SPY"),
    "Q3": ("GLD", "SPY"),
    "Q4": ("TLT", "SPY"),
}


@dataclass
class ValidationResult:
    summary: pd.DataFrame
    walkforward_rows: pd.DataFrame
    confidence_calibration: pd.DataFrame
    crash_calibration: pd.DataFrame
    regime_stability: pd.DataFrame
    confusion: pd.DataFrame
    notes: List[str]


def _forward_return(price_df: pd.DataFrame, ticker: str, horizon: int) -> pd.Series:
    if ticker not in price_df.columns:
        return pd.Series(index=price_df.index, dtype=float)
    px = price_df[ticker].astype(float)
    return px.shift(-horizon) / px - 1.0


def build_price_panel_from_archive(root: str | Path, tickers: Optional[Sequence[str]] = None) -> pd.DataFrame:
    idx = list_snapshots(root)
    if idx.empty:
        return pd.DataFrame()
    rows: List[Dict[str, float]] = []
    for asof in idx["asof_date"].astype(str).tolist():
        snap = read_snapshot(root, asof)
        market = snap.get("market", {})
        row: Dict[str, float] = {"asof_date": pd.Timestamp(asof)}
        for name, series in market.items():
            if tickers is not None and name not in tickers:
                continue
            if len(series) == 0:
                continue
            row[name] = float(pd.Series(series).dropna().iloc[-1])
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset=["asof_date"]).sort_values("asof_date").set_index("asof_date")
    return df


def build_walkforward_rows(root: str | Path, horizon_days: int = 21, proxy_map: Optional[Dict[str, Tuple[str, str]]] = None) -> pd.DataFrame:
    proxy_map = proxy_map or DEFAULT_PROXY_MAP
    idx = list_snapshots(root)
    if idx.empty:
        return pd.DataFrame()

    needed = sorted({t for pair in proxy_map.values() for t in pair})
    price_df = build_price_panel_from_archive(root, needed)
    rows: List[Dict[str, object]] = []

    for _, r in idx.sort_values("asof_date").iterrows():
        asof = str(r["asof_date"])
        regime = str(r.get("decision_regime", ""))
        next_regime = str(r.get("next_regime", ""))
        conf = float(r.get("confidence", np.nan)) if pd.notna(r.get("confidence")) else np.nan
        crash = float(r.get("crash_meter", np.nan)) if pd.notna(r.get("crash_meter")) else np.nan
        stage = r.get("stage")
        variant = r.get("variant")

        long_ticker, bench_ticker = proxy_map.get(regime, ("SPY", "SPY"))
        asof_ts = pd.Timestamp(asof)
        long_ret = float(_forward_return(price_df, long_ticker, horizon_days).get(asof_ts, np.nan))
        bench_ret = float(_forward_return(price_df, bench_ticker, horizon_days).get(asof_ts, np.nan))
        spread = long_ret - bench_ret if np.isfinite(long_ret) and np.isfinite(bench_ret) else np.nan
        worst_equity = float(_forward_return(price_df, "SPY", horizon_days).get(asof_ts, np.nan))
        is_hit = bool(np.isfinite(spread) and spread > 0)
        crash_realized = bool(np.isfinite(worst_equity) and worst_equity <= -0.08)
        realized_label = "Crash" if crash_realized else "Normal"
        rows.append({
            "asof_date": asof_ts,
            "decision_regime": regime,
            "next_regime": next_regime,
            "stage": stage,
            "variant": variant,
            "confidence": conf,
            "crash_meter": crash,
            "long_ticker": long_ticker,
            "benchmark_ticker": bench_ticker,
            "forward_long_return": long_ret,
            "forward_benchmark_return": bench_ret,
            "forward_spread": spread,
            "is_hit": is_hit,
            "realized_state": realized_label,
            "crash_realized": crash_realized,
        })
    df = pd.DataFrame(rows).sort_values("asof_date")
    return df


def confidence_calibration_table(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or "confidence" not in rows.columns:
        return pd.DataFrame(columns=["bucket", "mean_confidence", "hit_rate", "count", "gap"])
    work = rows[["confidence", "is_hit"]].dropna().copy()
    if work.empty:
        return pd.DataFrame(columns=["bucket", "mean_confidence", "hit_rate", "count", "gap"])
    work["bucket"] = pd.cut(work["confidence"], bins=[0, 0.4, 0.55, 0.7, 0.85, 1.0], include_lowest=True)
    out = work.groupby("bucket", observed=False).agg(mean_confidence=("confidence", "mean"), hit_rate=("is_hit", "mean"), count=("is_hit", "size")).reset_index()
    out["gap"] = out["hit_rate"] - out["mean_confidence"]
    return out


def crash_calibration_table(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or "crash_meter" not in rows.columns:
        return pd.DataFrame(columns=["bucket", "mean_crash_meter", "crash_rate", "count", "gap"])
    work = rows[["crash_meter", "crash_realized"]].dropna().copy()
    if work.empty:
        return pd.DataFrame(columns=["bucket", "mean_crash_meter", "crash_rate", "count", "gap"])
    work["bucket"] = pd.cut(work["crash_meter"], bins=[0, 0.2, 0.35, 0.5, 0.65, 1.0], include_lowest=True)
    out = work.groupby("bucket", observed=False).agg(mean_crash_meter=("crash_meter", "mean"), crash_rate=("crash_realized", "mean"), count=("crash_realized", "size")).reset_index()
    out["gap"] = out["crash_rate"] - out["mean_crash_meter"]
    return out


def stability_table(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["decision_regime", "count", "avg_confidence", "avg_spread", "hit_rate"])
    grp = rows.groupby("decision_regime", observed=False).agg(
        count=("decision_regime", "size"),
        avg_confidence=("confidence", "mean"),
        avg_spread=("forward_spread", "mean"),
        hit_rate=("is_hit", "mean"),
    ).reset_index()
    return grp.sort_values("count", ascending=False)


def confusion_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    actual = np.where(rows["crash_realized"], "Crash", rows["decision_regime"])
    pred = np.where(rows["crash_meter"].fillna(0) >= 0.5, "Crash", rows["decision_regime"])
    return pd.crosstab(pd.Series(actual, name="actual"), pd.Series(pred, name="predicted"))


def summary_table(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["metric", "value"])
    valid_spread = rows["forward_spread"].dropna()
    valid_conf = rows["confidence"].dropna()
    metrics = [
        ("rows", len(rows)),
        ("hit_rate", float(rows["is_hit"].mean()) if len(rows) else np.nan),
        ("avg_forward_spread", float(valid_spread.mean()) if not valid_spread.empty else np.nan),
        ("median_forward_spread", float(valid_spread.median()) if not valid_spread.empty else np.nan),
        ("avg_confidence", float(valid_conf.mean()) if not valid_conf.empty else np.nan),
        ("crash_event_rate", float(rows["crash_realized"].mean()) if len(rows) else np.nan),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value"])


def run_validation(root: str | Path, horizon_days: int = 21) -> ValidationResult:
    rows = build_walkforward_rows(root=root, horizon_days=horizon_days)
    notes: List[str] = []
    if rows.empty:
        notes.append("Belum ada snapshot archive yang cukup untuk validation numerik.")
    else:
        if rows["forward_spread"].notna().sum() < 20:
            notes.append("Forward return rows masih sedikit; baca hasil dengan hati-hati.")
        if rows["decision_regime"].nunique() < 2:
            notes.append("Coverage regime masih sempit; confusion/stability belum matang.")
    return ValidationResult(
        summary=summary_table(rows),
        walkforward_rows=rows,
        confidence_calibration=confidence_calibration_table(rows),
        crash_calibration=crash_calibration_table(rows),
        regime_stability=stability_table(rows),
        confusion=confusion_matrix(rows),
        notes=notes,
    )
