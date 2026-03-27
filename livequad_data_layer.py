from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

SERIES_RELEASE_RULES = {
    "INDPRO": {"cadence": "monthly", "lag_days": 16, "stale_days": 45, "label": "Industrial Production"},
    "RSAFS": {"cadence": "monthly", "lag_days": 17, "stale_days": 45, "label": "Retail Sales"},
    "PAYEMS": {"cadence": "monthly", "lag_days": 3, "stale_days": 35, "label": "Payrolls"},
    "UNRATE": {"cadence": "monthly", "lag_days": 3, "stale_days": 35, "label": "Unemployment Rate"},
    "ICSA": {"cadence": "weekly", "lag_days": 6, "stale_days": 14, "label": "Initial Claims"},
    "CPI": {"cadence": "monthly", "lag_days": 12, "stale_days": 45, "label": "CPI"},
    "CORE_CPI": {"cadence": "monthly", "lag_days": 12, "stale_days": 45, "label": "Core CPI"},
    "PPI": {"cadence": "monthly", "lag_days": 14, "stale_days": 45, "label": "PPI"},
    "SAHM": {"cadence": "monthly", "lag_days": 3, "stale_days": 45, "label": "Sahm Rule"},
    "NFCI": {"cadence": "weekly", "lag_days": 5, "stale_days": 14, "label": "NFCI"},
    "HY": {"cadence": "daily", "lag_days": 1, "stale_days": 5, "label": "HY Spread"},
    "WTI": {"cadence": "daily", "lag_days": 1, "stale_days": 5, "label": "WTI Oil"},
    "WALCL": {"cadence": "weekly", "lag_days": 5, "stale_days": 14, "label": "Fed Balance Sheet"},
}


def _apply_release_lag(series: pd.Series, series_name: str, asof: pd.Timestamp | None = None) -> pd.Series:
    s = pd.Series(series).dropna().copy()
    if s.empty:
        return s
    ts = pd.Timestamp(asof).normalize() if asof is not None else pd.Timestamp.utcnow().normalize()
    rule = SERIES_RELEASE_RULES.get(series_name, {"cadence": "daily", "lag_days": 1, "stale_days": 5, "label": series_name})
    lag_days = int(rule.get("lag_days", 1))
    avail_cutoff = ts - pd.Timedelta(days=lag_days)
    return s.loc[s.index <= avail_cutoff].copy()


def adjust_series_dict(raw: Dict[str, pd.Series], asof: pd.Timestamp | None = None) -> Dict[str, pd.Series]:
    return {k: _apply_release_lag(v, k, asof=asof) for k, v in raw.items()}


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
    order = {"Missing": 0, "Stale": 1, "Aging": 2, "Fresh": 3}
    rows.sort(key=lambda r: (order.get(r[1], 9), r[0]))
    return rows


def health_summary(rows: List[List[str]]) -> List[List[str]]:
    if not rows:
        return [["Coverage", "0 series", "No health rows yet"]]
    counts = {"Fresh": 0, "Aging": 0, "Stale": 0, "Missing": 0}
    dropped = 0
    for r in rows:
        counts[r[1]] = counts.get(r[1], 0) + 1
        try:
            dropped += int(r[6])
        except Exception:
            pass
    total = sum(counts.values())
    coverage = counts.get("Fresh", 0) + counts.get("Aging", 0)
    return [
        ["Coverage", f"{coverage}/{total} usable", "Fresh + Aging series available to model now"],
        ["Fresh", str(counts.get("Fresh", 0)), "No immediate staleness concern"],
        ["Aging", str(counts.get("Aging", 0)), "Use but treat as fading signal"],
        ["Stale / Missing", str(counts.get("Stale", 0) + counts.get("Missing", 0)), "Can distort regime confidence"],
        ["Rows dropped by release-lag", str(dropped), "Observations hidden to reduce look-ahead bias"],
    ]


def persist_snapshot(
    raw: Dict[str, pd.Series],
    adjusted: Dict[str, pd.Series],
    out_dir: str | Path,
    meta: Dict[str, object] | None = None,
) -> str | None:
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
        idx_path = base / "index.csv"
        row = {k: v for k, v in (meta or {}).items()}
        row["snapshot_path"] = str(snap_dir)
        row["snapshot_date"] = asof
        idx = pd.DataFrame([row])
        if idx_path.exists():
            old = pd.read_csv(idx_path)
            old = old[old.get("snapshot_date", "") != asof]
            idx = pd.concat([old, idx], ignore_index=True)
        idx.sort_values("snapshot_date", inplace=True)
        idx.to_csv(idx_path, index=False)
        return str(snap_dir)
    except Exception:
        return None


def snapshot_coverage(out_dir: str | Path) -> Tuple[int, str]:
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


def load_snapshot_index(out_dir: str | Path) -> pd.DataFrame:
    base = Path(out_dir)
    idx_path = base / "index.csv"
    if not idx_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(idx_path)
        if "snapshot_date" in df.columns:
            df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
            df = df.sort_values("snapshot_date")
        return df
    except Exception:
        return pd.DataFrame()
