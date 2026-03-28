from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

SERIES_RELEASE_RULES = {
    # name: cadence, lag days from period end, stale days tolerance
    "INDPRO": {"cadence": "monthly", "lag_days": 16, "stale_days": 45, "label": "Industrial Production"},
    "RSAFS": {"cadence": "monthly", "lag_days": 16, "stale_days": 45, "label": "Retail Sales"},
    "PAYEMS": {"cadence": "monthly", "lag_days": 3, "stale_days": 35, "label": "Payrolls"},
    "UNRATE": {"cadence": "monthly", "lag_days": 3, "stale_days": 35, "label": "Unemployment Rate"},
    "ICSA": {"cadence": "weekly", "lag_days": 6, "stale_days": 14, "label": "Initial Claims"},
    "CPI": {"cadence": "monthly", "lag_days": 12, "stale_days": 40, "label": "CPI"},
    "CORE_CPI": {"cadence": "monthly", "lag_days": 12, "stale_days": 40, "label": "Core CPI"},
    "PPI": {"cadence": "monthly", "lag_days": 14, "stale_days": 40, "label": "PPI"},
    "SAHM": {"cadence": "monthly", "lag_days": 5, "stale_days": 35, "label": "Sahm Rule"},
    "NFCI": {"cadence": "weekly", "lag_days": 4, "stale_days": 14, "label": "NFCI"},
    "HY": {"cadence": "daily", "lag_days": 1, "stale_days": 5, "label": "High Yield OAS"},
    "WTI": {"cadence": "daily", "lag_days": 1, "stale_days": 5, "label": "WTI"},
    "WALCL": {"cadence": "weekly", "lag_days": 3, "stale_days": 14, "label": "Fed Balance Sheet"},
    "M2": {"cadence": "monthly", "lag_days": 25, "stale_days": 60, "label": "M2"},
    "USD": {"cadence": "daily", "lag_days": 1, "stale_days": 5, "label": "Broad Dollar"},
}


def _period_end(ts: pd.Timestamp, cadence: str) -> pd.Timestamp:
    ts = pd.Timestamp(ts).normalize()
    if cadence == "monthly":
        return ts + pd.offsets.MonthEnd(0)
    if cadence == "weekly":
        # FRED weekly observations are usually week-ending levels.
        return ts + pd.Timedelta(days=max(0, 6 - ts.weekday()))
    return ts


def availability_date(ts: pd.Timestamp, cadence: str, lag_days: int) -> pd.Timestamp:
    return _period_end(ts, cadence) + pd.Timedelta(days=lag_days)


def apply_release_lag(series: pd.Series, series_name: str, asof: pd.Timestamp | None = None) -> pd.Series:
    s = pd.Series(series).dropna().copy()
    if s.empty:
        return s
    rule = SERIES_RELEASE_RULES.get(series_name, {"cadence": "daily", "lag_days": 1})
    cadence = rule.get("cadence", "daily")
    lag_days = int(rule.get("lag_days", 1))
    asof_ts = pd.Timestamp(asof).normalize() if asof is not None else pd.Timestamp.utcnow().normalize()
    keep_idx = []
    for idx in s.index:
        if availability_date(pd.Timestamp(idx), cadence, lag_days) <= asof_ts:
            keep_idx.append(idx)
    if not keep_idx:
        return pd.Series(dtype=float, name=s.name)
    return s.loc[keep_idx]


def adjust_series_dict(raw: Dict[str, pd.Series], asof: pd.Timestamp | None = None) -> Dict[str, pd.Series]:
    return {k: apply_release_lag(v, k, asof=asof) for k, v in raw.items()}


def health_rows(raw: Dict[str, pd.Series], adjusted: Dict[str, pd.Series], asof: pd.Timestamp | None = None) -> List[List[str]]:
    asof_ts = pd.Timestamp(asof).normalize() if asof is not None else pd.Timestamp.utcnow().normalize()
    rows: List[List[str]] = []
    for name, raw_series in raw.items():
        raw_s = pd.Series(raw_series).dropna()
        adj_s = pd.Series(adjusted.get(name, pd.Series(dtype=float))).dropna()
        rule = SERIES_RELEASE_RULES.get(name, {"cadence": "daily", "lag_days": 1, "stale_days": 5, "label": name})
        label = rule.get("label", name)
        raw_last = raw_s.index.max().strftime("%Y-%m-%d") if not raw_s.empty else "n/a"
        adj_last = adj_s.index.max().strftime("%Y-%m-%d") if not adj_s.empty else "n/a"
        if adj_s.empty:
            status = "Missing"
            age_days = "-"
            release_info = f"{rule['cadence']} +{rule['lag_days']}d"
        else:
            last_ts = pd.Timestamp(adj_s.index.max()).normalize()
            age = int((asof_ts - last_ts).days)
            age_days = str(age)
            stale_limit = int(rule.get("stale_days", 30))
            if age <= stale_limit:
                status = "Fresh"
            elif age <= stale_limit * 2:
                status = "Aging"
            else:
                status = "Stale"
            release_info = f"{rule['cadence']} +{rule['lag_days']}d"
        dropped = max(0, len(raw_s) - len(adj_s))
        rows.append([label, status, raw_last, adj_last, age_days, release_info, str(dropped)])
    order = {"Missing": 0, "Stale": 1, "Aging": 2, "Fresh": 3}
    rows.sort(key=lambda r: (order.get(r[1], 9), r[0]))
    return rows


def persist_snapshot(raw: Dict[str, pd.Series], adjusted: Dict[str, pd.Series], out_dir: str | Path, meta: Dict[str, object] | None = None) -> str | None:
    try:
        base = Path(out_dir)
        base.mkdir(parents=True, exist_ok=True)
        asof = pd.Timestamp(meta.get("asof") if meta else pd.Timestamp.utcnow().date()).strftime("%Y-%m-%d")
        snap_dir = base / asof
        snap_dir.mkdir(parents=True, exist_ok=True)
        for kind, data in (("raw", raw), ("adjusted", adjusted)):
            kind_dir = snap_dir / kind
            kind_dir.mkdir(parents=True, exist_ok=True)
            for name, series in data.items():
                s = pd.Series(series).dropna().rename("value")
                if s.empty:
                    continue
                df = s.reset_index()
                df.columns = ["date", "value"]
                df.to_csv(kind_dir / f"{name}.csv", index=False)
        if meta:
            with open(snap_dir / "meta.json", "w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=2, default=str)
        return str(snap_dir)
    except Exception:
        return None


def snapshot_coverage(out_dir: str | Path) -> tuple[int, str]:
    try:
        base = Path(out_dir)
        if not base.exists():
            return 0, "n/a"
        dirs = [p for p in base.iterdir() if p.is_dir()]
        if not dirs:
            return 0, "n/a"
        last = max(dirs, key=lambda p: p.name)
        return len(dirs), last.name
    except Exception:
        return 0, "n/a"
