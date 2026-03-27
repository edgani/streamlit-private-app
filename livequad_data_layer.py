from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

SERIES_RELEASE_RULES = {
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

STATUS_ORDER = {"Missing": 0, "Stale": 1, "Aging": 2, "Fresh": 3}


def _period_end(ts: pd.Timestamp, cadence: str) -> pd.Timestamp:
    ts = pd.Timestamp(ts).normalize()
    if cadence == "monthly":
        return ts + pd.offsets.MonthEnd(0)
    if cadence == "weekly":
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
        dropped = max(0, len(raw_s) - len(adj_s))
        rows.append([label, status, raw_last, adj_last, age_days, f"{rule['cadence']} +{rule['lag_days']}d", str(dropped)])
    rows.sort(key=lambda r: (STATUS_ORDER.get(r[1], 9), r[0]))
    return rows


def health_summary(rows: List[List[str]]) -> Tuple[float, List[List[str]]]:
    if not rows:
        return 0.0, [["No series", "-"]]
    score = 1.0
    counts = {"Fresh": 0, "Aging": 0, "Stale": 0, "Missing": 0}
    for _, status, *_ in rows:
        counts[status] = counts.get(status, 0) + 1
        if status == "Aging":
            score -= 0.05
        elif status == "Stale":
            score -= 0.12
        elif status == "Missing":
            score -= 0.18
    score = max(0.0, min(1.0, score))
    summary = [
        ["Fresh", str(counts['Fresh'])],
        ["Aging", str(counts['Aging'])],
        ["Stale", str(counts['Stale'])],
        ["Missing", str(counts['Missing'])],
        ["Health score", f"{score*100:.1f}%"],
    ]
    return score, summary


def _upsert_index(base: Path, snap_dir: Path, meta: Dict[str, object]) -> None:
    idx_path = base / "index.csv"
    row = {
        "asof": meta.get("asof", snap_dir.name),
        "path": str(snap_dir),
        "current_q": meta.get("current_q", ""),
        "next_q": meta.get("next_q", ""),
        "confidence": meta.get("confidence", ""),
        "crash_now": meta.get("crash_now", ""),
        "health_score": meta.get("health_score", ""),
    }
    if idx_path.exists():
        df = pd.read_csv(idx_path)
        df = df[df["asof"].astype(str) != str(row["asof"])]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.sort_values("asof", inplace=True)
    df.to_csv(idx_path, index=False)


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
            _upsert_index(base, snap_dir, meta)
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


def snapshot_index(out_dir: str | Path) -> pd.DataFrame:
    base = Path(out_dir)
    idx_path = base / "index.csv"
    if not idx_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(idx_path)
    except Exception:
        return pd.DataFrame()
