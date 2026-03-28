from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd


DEFAULT_ROOT = ".livequad_archive"


@dataclass
class SnapshotManifest:
    asof_date: str
    created_at_utc: str
    app_version: str = "livequad_research_risk_v2"
    decision_regime: Optional[str] = None
    next_regime: Optional[str] = None
    stage: Optional[str] = None
    variant: Optional[str] = None
    confidence: Optional[float] = None
    crash_meter: Optional[float] = None
    risk_posture: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    macro_series_count: int = 0
    market_series_count: int = 0
    file_count: int = 0


class ArchiveError(RuntimeError):
    pass


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _normalize_series_map(data: Optional[Mapping[str, Any]]) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    if not data:
        return out
    for name, obj in data.items():
        if isinstance(obj, pd.Series):
            s = obj.copy()
        elif isinstance(obj, pd.DataFrame):
            if obj.shape[1] == 1:
                s = obj.iloc[:, 0].copy()
            elif "value" in obj.columns:
                s = obj["value"].copy()
            else:
                raise ArchiveError(f"Cannot infer value column for dataframe {name}")
        else:
            s = pd.Series(obj)
        s = pd.Series(s).dropna()
        if s.empty:
            continue
        if not isinstance(s.index, pd.DatetimeIndex):
            try:
                s.index = pd.to_datetime(s.index)
            except Exception as exc:  # pragma: no cover - defensive
                raise ArchiveError(f"Series {name} index is not datetime-like") from exc
        s = s.sort_index()
        out[str(name)] = s
    return out


def _series_to_csv(series: pd.Series, path: Path) -> None:
    df = pd.DataFrame({"date": pd.to_datetime(series.index), "value": series.astype(float).values})
    df.to_csv(path, index=False)


def _read_series_csv(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if df.empty:
        return pd.Series(dtype=float)
    idx_col = "date" if "date" in df.columns else df.columns[0]
    val_col = "value" if "value" in df.columns else df.columns[-1]
    s = pd.Series(df[val_col].values, index=pd.to_datetime(df[idx_col]), name=path.stem)
    return s.sort_index()


def _load_index(index_path: Path) -> pd.DataFrame:
    if not index_path.exists():
        return pd.DataFrame(columns=[
            "asof_date", "created_at_utc", "decision_regime", "next_regime", "stage", "variant",
            "confidence", "crash_meter", "risk_posture", "macro_series_count", "market_series_count",
            "file_count", "snapshot_path"
        ])
    return pd.read_csv(index_path)


def _write_index(index_path: Path, df: pd.DataFrame) -> None:
    df = df.sort_values("asof_date").drop_duplicates(subset=["asof_date"], keep="last")
    df.to_csv(index_path, index=False)


def save_snapshot(
    root: str | Path = DEFAULT_ROOT,
    *,
    asof_date: str,
    macro_raw: Optional[Mapping[str, Any]] = None,
    macro_adjusted: Optional[Mapping[str, Any]] = None,
    market: Optional[Mapping[str, Any]] = None,
    state: Optional[Mapping[str, Any]] = None,
    validation: Optional[Mapping[str, Any]] = None,
    portfolio: Optional[Mapping[str, Any]] = None,
    notes: Optional[Iterable[str]] = None,
    tags: Optional[Iterable[str]] = None,
    app_version: str = "livequad_research_risk_v2",
) -> Path:
    root_path = _ensure_dir(root)
    snap_path = root_path / str(asof_date)
    snap_path.mkdir(parents=True, exist_ok=True)

    macro_raw_map = _normalize_series_map(macro_raw)
    macro_adj_map = _normalize_series_map(macro_adjusted)
    market_map = _normalize_series_map(market)

    dirs = {
        "macro_raw": snap_path / "macro_raw",
        "macro_adjusted": snap_path / "macro_adjusted",
        "market": snap_path / "market",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    file_count = 0
    for name, series in macro_raw_map.items():
        _series_to_csv(series, dirs["macro_raw"] / f"{name}.csv")
        file_count += 1
    for name, series in macro_adj_map.items():
        _series_to_csv(series, dirs["macro_adjusted"] / f"{name}.csv")
        file_count += 1
    for name, series in market_map.items():
        _series_to_csv(series, dirs["market"] / f"{name}.csv")
        file_count += 1

    state_dict = dict(state or {})
    manifest = SnapshotManifest(
        asof_date=str(asof_date),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        app_version=app_version,
        decision_regime=state_dict.get("decision_regime") or state_dict.get("current_q"),
        next_regime=state_dict.get("next_regime") or state_dict.get("next_q"),
        stage=state_dict.get("stage"),
        variant=state_dict.get("variant") or state_dict.get("variant_now"),
        confidence=float(state_dict["confidence"]) if state_dict.get("confidence") is not None else None,
        crash_meter=float(state_dict["crash_meter"]) if state_dict.get("crash_meter") is not None else None,
        risk_posture=state_dict.get("risk_posture"),
        notes=list(notes or []),
        tags=list(tags or []),
        macro_series_count=len(macro_adj_map) or len(macro_raw_map),
        market_series_count=len(market_map),
        file_count=file_count,
    )
    (snap_path / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    (snap_path / "state.json").write_text(json.dumps(state_dict, indent=2, default=str), encoding="utf-8")
    (snap_path / "validation.json").write_text(json.dumps(dict(validation or {}), indent=2, default=str), encoding="utf-8")
    (snap_path / "portfolio.json").write_text(json.dumps(dict(portfolio or {}), indent=2, default=str), encoding="utf-8")

    index_path = root_path / "index.csv"
    idx = _load_index(index_path)
    row = {
        **asdict(manifest),
        "snapshot_path": str(snap_path),
    }
    idx = pd.concat([idx, pd.DataFrame([row])], ignore_index=True)
    _write_index(index_path, idx)
    return snap_path


def list_snapshots(root: str | Path = DEFAULT_ROOT) -> pd.DataFrame:
    root_path = Path(root)
    return _load_index(root_path / "index.csv")


def latest_snapshot(root: str | Path = DEFAULT_ROOT) -> Optional[Dict[str, Any]]:
    idx = list_snapshots(root)
    if idx.empty:
        return None
    row = idx.sort_values("asof_date").iloc[-1].to_dict()
    return row


def read_snapshot(root: str | Path = DEFAULT_ROOT, asof_date: Optional[str] = None) -> Dict[str, Any]:
    idx = list_snapshots(root)
    if idx.empty:
        raise ArchiveError("No snapshots found")
    if asof_date is None:
        row = idx.sort_values("asof_date").iloc[-1].to_dict()
    else:
        rows = idx[idx["asof_date"].astype(str) == str(asof_date)]
        if rows.empty:
            raise ArchiveError(f"Snapshot {asof_date} not found")
        row = rows.iloc[-1].to_dict()
    snap_path = Path(row["snapshot_path"])
    out: Dict[str, Any] = {"manifest": row, "path": str(snap_path)}
    for bucket in ["macro_raw", "macro_adjusted", "market"]:
        bucket_path = snap_path / bucket
        data: Dict[str, pd.Series] = {}
        if bucket_path.exists():
            for f in sorted(bucket_path.glob("*.csv")):
                data[f.stem] = _read_series_csv(f)
        out[bucket] = data
    for name in ["state.json", "validation.json", "portfolio.json", "manifest.json"]:
        p = snap_path / name
        if p.exists():
            out[name.replace(".json", "")] = json.loads(p.read_text(encoding="utf-8"))
    return out


def archive_health(root: str | Path = DEFAULT_ROOT) -> Dict[str, Any]:
    idx = list_snapshots(root)
    if idx.empty:
        return {
            "snapshot_count": 0,
            "latest_asof": None,
            "gaps_over_3d": 0,
            "duplicate_dates": 0,
            "regimes_present": [],
            "market_series_median": 0,
        }
    work = idx.copy()
    work["asof_date"] = pd.to_datetime(work["asof_date"])
    work = work.sort_values("asof_date")
    diffs = work["asof_date"].diff().dt.days.dropna()
    return {
        "snapshot_count": int(len(work)),
        "latest_asof": work["asof_date"].iloc[-1].strftime("%Y-%m-%d"),
        "gaps_over_3d": int((diffs > 3).sum()),
        "duplicate_dates": int(work["asof_date"].duplicated().sum()),
        "regimes_present": sorted({x for x in work["decision_regime"].dropna().astype(str).unique().tolist()}),
        "market_series_median": float(pd.to_numeric(work["market_series_count"], errors="coerce").fillna(0).median()),
    }


def build_manifest_table(root: str | Path = DEFAULT_ROOT, limit: int = 120) -> pd.DataFrame:
    idx = list_snapshots(root)
    if idx.empty:
        return idx
    cols = [
        "asof_date", "decision_regime", "next_regime", "stage", "variant", "confidence",
        "crash_meter", "risk_posture", "macro_series_count", "market_series_count", "file_count"
    ]
    cols = [c for c in cols if c in idx.columns]
    out = idx.sort_values("asof_date", ascending=False)[cols].head(limit).copy()
    return out
