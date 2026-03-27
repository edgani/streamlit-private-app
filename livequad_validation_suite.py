from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class ValidationReadiness:
    score: float
    label: str
    notes: List[str]


def readiness_score(snapshot_count: int, has_release_lag: bool, has_data_health: bool, has_risk_budget: bool) -> ValidationReadiness:
    score = 0.20
    notes: List[str] = []
    if snapshot_count >= 20:
        score += 0.25
    elif snapshot_count >= 5:
        score += 0.15
        notes.append("Snapshot history masih pendek; WFO belum kuat.")
    else:
        notes.append("Snapshot history hampir belum ada; mulai archive harian dulu.")
    if has_release_lag:
        score += 0.20
    else:
        notes.append("Release-lag belum aktif; raw macro bisa terlalu optimistis.")
    if has_data_health:
        score += 0.15
    else:
        notes.append("Data health belum ada; missing/stale risk susah dilacak.")
    if has_risk_budget:
        score += 0.10
    else:
        notes.append("Risk budget belum aktif; playbook belum portfolio-aware.")
    score += 0.10
    label = "Low"
    if score >= 0.75:
        label = "High"
    elif score >= 0.50:
        label = "Medium"
    return ValidationReadiness(score=score, label=label, notes=notes)


def confusion_rows(df: pd.DataFrame, actual_col: str = "actual_regime", pred_col: str = "pred_regime") -> List[List[str]]:
    if df.empty or actual_col not in df.columns or pred_col not in df.columns:
        return [["No validation rows", "-", "-"]]
    ct = pd.crosstab(df[actual_col], df[pred_col])
    out: List[List[str]] = []
    for idx in ct.index:
        row = ct.loc[idx]
        best = row.idxmax() if len(row) else "-"
        acc = row.max() / row.sum() if row.sum() else 0.0
        out.append([str(idx), str(best), f"{acc*100:.1f}%"])
    return out


def calibration_rows(df: pd.DataFrame, prob_col: str = "confidence", hit_col: str = "is_correct") -> List[List[str]]:
    if df.empty or prob_col not in df.columns or hit_col not in df.columns:
        return [["No calibration rows", "-", "-"]]
    work = df[[prob_col, hit_col]].dropna().copy()
    if work.empty:
        return [["No calibration rows", "-", "-"]]
    work["bucket"] = pd.cut(work[prob_col], bins=[0, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
    out = []
    for bucket, grp in work.groupby("bucket", observed=False):
        out.append([str(bucket), f"{grp[prob_col].mean()*100:.1f}%", f"{grp[hit_col].mean()*100:.1f}%"])
    return out or [["No calibration rows", "-", "-"]]


def snapshot_paths(snapshot_root: str | Path) -> List[str]:
    base = Path(snapshot_root)
    if not base.exists():
        return []
    return [str(p) for p in sorted(base.iterdir()) if p.is_dir()]


def _load_index(snapshot_root: str | Path) -> pd.DataFrame:
    base = Path(snapshot_root)
    idx_path = base / "index.csv"
    if not idx_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(idx_path)
        if "snapshot_date" in df.columns:
            df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
            df = df.dropna(subset=["snapshot_date"]).sort_values("snapshot_date").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def _forward_row(df: pd.DataFrame, i: int, horizon_days: int) -> Tuple[int | None, pd.Timestamp | None]:
    if df.empty or i >= len(df):
        return None, None
    cur = df.loc[i, "snapshot_date"]
    target = cur + pd.Timedelta(days=horizon_days)
    future = df[df["snapshot_date"] >= target]
    if future.empty:
        return None, None
    j = int(future.index[0])
    return j, pd.Timestamp(df.loc[j, "snapshot_date"])


def run_snapshot_walkforward(snapshot_root: str | Path, horizon_days: int = 21) -> pd.DataFrame:
    df = _load_index(snapshot_root)
    if df.empty:
        return pd.DataFrame()
    required = ["current_q", "next_q", "confidence", "crash_now", "snapshot_date"]
    for c in required:
        if c not in df.columns:
            return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for i in range(len(df)):
        j, future_date = _forward_row(df, i, horizon_days)
        if j is None:
            continue
        cur = df.loc[i].to_dict()
        fut = df.loc[j].to_dict()
        row: Dict[str, object] = {
            "asof": pd.Timestamp(cur["snapshot_date"]),
            "future_asof": pd.Timestamp(future_date),
            "pred_regime": cur.get("current_q"),
            "pred_next": cur.get("next_q"),
            "actual_regime": fut.get("current_q"),
            "actual_variant": fut.get("variant_now", fut.get("sub_phase", "n/a")),
            "confidence": pd.to_numeric(cur.get("confidence"), errors="coerce"),
            "crash_now": pd.to_numeric(cur.get("crash_now"), errors="coerce"),
        }
        row["is_correct"] = int(str(row["pred_regime"]) == str(row["actual_regime"]))
        row["next_hit"] = int(str(row["pred_next"]) == str(row["actual_regime"]))
        # proxy returns if present in snapshot meta/index
        for px in ["spy_close", "iwm_close", "gld_close", "tlt_close", "xle_close", "btc_close", "eido_close"]:
            if px in df.columns:
                try:
                    p0 = float(cur.get(px))
                    p1 = float(fut.get(px))
                    row[f"{px}_fwd_ret"] = p1 / p0 - 1 if p0 else None
                except Exception:
                    row[f"{px}_fwd_ret"] = None
        if "spy_close_fwd_ret" in row and row["spy_close_fwd_ret"] is not None:
            row["drawdown_hit"] = int(float(row["spy_close_fwd_ret"]) <= -0.08)
        else:
            row["drawdown_hit"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def walkforward_summary_rows(wfo: pd.DataFrame) -> List[List[str]]:
    if wfo.empty:
        return [["Walk-forward", "No rows yet", "Need more snapshots / history"]]
    def _pct(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        return f"{s.mean()*100:.1f}%" if len(s) else "n/a"
    rows = [
        ["Rows", str(len(wfo)), "Snapshot-aware validation observations"],
        ["Current regime hit rate", _pct(wfo.get("is_correct")), "Did current call match future regime?"],
        ["Next regime hit rate", _pct(wfo.get("next_hit")), "Did next call become future regime?"],
        ["Avg confidence", _pct(wfo.get("confidence")), "Mean stated confidence at decision time"],
        ["Avg crash meter", _pct(wfo.get("crash_now")), "Mean risk state at decision time"],
    ]
    if "spy_close_fwd_ret" in wfo.columns:
        s = pd.to_numeric(wfo["spy_close_fwd_ret"], errors="coerce").dropna()
        if len(s):
            rows.append(["SPY 21d avg fwd return", f"{s.mean()*100:.2f}%", "Context only; not alpha by itself"])
    if "drawdown_hit" in wfo.columns:
        dd = pd.to_numeric(wfo["drawdown_hit"], errors="coerce").dropna()
        if len(dd):
            rows.append(["Drawdown incidence", f"{dd.mean()*100:.1f}%", "Share of rows followed by >8% SPY drop"])
    return rows


def crash_calibration_rows(wfo: pd.DataFrame) -> List[List[str]]:
    if wfo.empty or "crash_now" not in wfo.columns or "drawdown_hit" not in wfo.columns:
        return [["No crash calibration rows", "-", "-"]]
    work = wfo[["crash_now", "drawdown_hit"]].dropna().copy()
    if work.empty:
        return [["No crash calibration rows", "-", "-"]]
    work["bucket"] = pd.cut(work["crash_now"], bins=[0, 0.30, 0.45, 0.60, 1.0], include_lowest=True)
    out = []
    for bucket, grp in work.groupby("bucket", observed=False):
        out.append([str(bucket), f"{grp['crash_now'].mean()*100:.1f}%", f"{grp['drawdown_hit'].mean()*100:.1f}%"])
    return out or [["No crash calibration rows", "-", "-"]]
