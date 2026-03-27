from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

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
    score += 0.10  # basic harness exists
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
