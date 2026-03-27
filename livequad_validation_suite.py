from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


@dataclass
class ValidationReadiness:
    score: float
    label: str
    notes: List[str]


def readiness_score(snapshot_count: int, has_release_lag: bool, has_data_health: bool, has_risk_budget: bool) -> ValidationReadiness:
    score = 0.20
    notes: List[str] = []
    if snapshot_count >= 60:
        score += 0.30
    elif snapshot_count >= 20:
        score += 0.22
        notes.append("Snapshot history sudah lumayan, tapi belum panjang.")
    elif snapshot_count >= 5:
        score += 0.14
        notes.append("Snapshot history masih pendek; WFO belum kuat.")
    else:
        notes.append("Snapshot history hampir belum ada; mulai archive harian dulu.")
    if has_release_lag:
        score += 0.18
    else:
        notes.append("Release-lag belum aktif; raw macro bisa terlalu optimistis.")
    if has_data_health:
        score += 0.12
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
    return ValidationReadiness(score=min(1.0, score), label=label, notes=notes)


def snapshot_paths(snapshot_root: str | Path) -> List[str]:
    base = Path(snapshot_root)
    if not base.exists():
        return []
    return [str(p) for p in sorted(base.iterdir()) if p.is_dir()]


def build_meta_frame(snapshot_root: str | Path) -> pd.DataFrame:
    base = Path(snapshot_root)
    rows = []
    if not base.exists():
        return pd.DataFrame()
    for snap_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        meta_path = snap_dir / 'meta.json'
        if not meta_path.exists():
            continue
        try:
            rows.append(json.loads(meta_path.read_text()))
        except Exception:
            continue
    return pd.DataFrame(rows)


def walkforward_rows(meta_df: pd.DataFrame) -> List[List[str]]:
    if meta_df.empty or len(meta_df) < 2:
        return [["No WFO history yet", "-", "-"]]
    df = meta_df.copy().sort_values('asof')
    if 'current_q' not in df.columns:
        return [["No WFO history yet", "-", "-"]]
    df['next_day_current'] = df['current_q'].shift(-1)
    out = []
    for q, grp in df.groupby('current_q'):
        hit = (grp['next_q'] == grp['next_day_current']).mean() if 'next_q' in grp.columns else float('nan')
        conf = grp['confidence'].mean() if 'confidence' in grp.columns else float('nan')
        out.append([str(q), f"{len(grp)}", f"{(0 if pd.isna(hit) else hit)*100:.1f}%", f"{(0 if pd.isna(conf) else conf)*100:.1f}%"])
    return out or [["No WFO history yet", "-", "-"]]


def calibration_rows(meta_df: pd.DataFrame, prob_col: str = 'confidence', hit_col: str = 'next_hit') -> List[List[str]]:
    if meta_df.empty or len(meta_df) < 3:
        return [["No calibration rows", "-", "-"]]
    df = meta_df.copy().sort_values('asof')
    if 'next_q' in df.columns and 'current_q' in df.columns:
        df['next_day_current'] = df['current_q'].shift(-1)
        df['next_hit'] = (df['next_q'] == df['next_day_current']).astype(float)
    if prob_col not in df.columns or hit_col not in df.columns:
        return [["No calibration rows", "-", "-"]]
    work = df[[prob_col, hit_col]].dropna().copy()
    if work.empty:
        return [["No calibration rows", "-", "-"]]
    work['bucket'] = pd.cut(work[prob_col], bins=[0,0.4,0.6,0.8,1.0], include_lowest=True)
    out = []
    for bucket, grp in work.groupby('bucket', observed=False):
        out.append([str(bucket), f"{grp[prob_col].mean()*100:.1f}%", f"{grp[hit_col].mean()*100:.1f}%"])
    return out or [["No calibration rows", "-", "-"]]


def stability_rows(meta_df: pd.DataFrame) -> List[List[str]]:
    if meta_df.empty or len(meta_df) < 3 or 'current_q' not in meta_df.columns:
        return [["No stability history", "-", "-"]]
    df = meta_df.copy().sort_values('asof')
    turns = (df['current_q'] != df['current_q'].shift(1)).fillna(False)
    avg_conf = df['confidence'].mean() if 'confidence' in df.columns else float('nan')
    avg_crash = df['crash_now'].mean() if 'crash_now' in df.columns else float('nan')
    return [
        ["Snapshots", str(len(df)), "Archive depth"],
        ["Regime turns", str(int(turns.sum())), "Current-Q changes over snapshot history"],
        ["Avg confidence", f"{(0 if pd.isna(avg_conf) else avg_conf)*100:.1f}%", "Helps judge signal stability"],
        ["Avg crash meter", f"{(0 if pd.isna(avg_crash) else avg_crash)*100:.1f}%", "Average hazard over history"],
    ]
