
import math
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from livequad_data_layer import adjust_series_dict, health_rows, health_summary, persist_snapshot, snapshot_coverage
    from livequad_risk_engine import recommended_risk_budget, risk_controls_rows, tactical_sizing_rows, scenario_override_rows
    from livequad_validation_suite import readiness_score, run_snapshot_walkforward, walkforward_summary_rows, calibration_rows, crash_calibration_rows
except Exception:
    adjust_series_dict = None
    health_rows = None
    health_summary = None
    persist_snapshot = None
    snapshot_coverage = None
    recommended_risk_budget = None
    risk_controls_rows = None
    tactical_sizing_rows = None
    scenario_override_rows = None
    readiness_score = None
    run_snapshot_walkforward = None
    walkforward_summary_rows = None
    calibration_rows = None
    crash_calibration_rows = None

st.set_page_config(page_title="QuantFinalV6_3_Q3Anchored", layout="wide")

APP_NAME = "QuantFinalV6_3_Q3Anchored"
CORE_NAME = "Hedgeye_LiveQuad_Core_v2_5"

Q3_CONSENSUS_ANCHOR = True
Q3_CONSENSUS_FORCE_TOP = True

# --------------------
# VISUAL SHELL (attachment-2 style)
# --------------------
st.markdown("""
<style>
:root {
  --bg:#07101f;
  --card:#0c1526;
  --line:#263246;
  --muted:#9fb0c8;
  --text:#f3f6fb;
  --red:#ff4d4d;
}
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg);
  color: var(--text);
}
.block-container {padding-top: 1.6rem; padding-bottom: 2.0rem; max-width: 1650px;}
h1,h2,h3,h4,h5,h6,p,span,div,label {color: var(--text);}
.card {
  background: linear-gradient(180deg, rgba(16,24,41,0.98), rgba(10,17,30,0.98));
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 14px 16px;
  height: 100%;
}
.hero-card {
  background: linear-gradient(180deg, rgba(19,29,50,1), rgba(12,20,35,1));
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 12px 14px;
  min-height: 96px;
}
.section-title {
  font-weight: 800;
  letter-spacing: .02em;
  margin-bottom: 10px;
  font-size: 1.0rem;
}
.metric-title {
  font-size: .76rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .05em;
}
.metric-value {
  font-size: 1.85rem;
  font-weight: 800;
  line-height: 1.1;
}
.metric-sub {
  font-size: .88rem;
  color:#c4d2e6;
  margin-top:4px;
}
.pill {
  display:inline-block;
  border:1px solid #33435d;
  border-radius:999px;
  padding:3px 10px;
  font-size:.82rem;
  color:#dbe7ff;
  background: rgba(30,42,63,.82);
  margin-right:6px;
  margin-top:4px;
}
.pill-red {
  display:inline-block;
  border:1px solid #ff4d4d;
  border-radius:999px;
  padding:3px 10px;
  font-size:.82rem;
  color:#ffd7d7;
  background: rgba(120,20,20,.22);
  margin-right:6px;
  margin-top:4px;
}
.small-muted {color: var(--muted); font-size: .92rem;}
.tight-table table {
  width:100%;
  border-collapse: collapse;
  table-layout: fixed;
}
.tight-table th, .tight-table td {
  border:1px solid #243147;
  padding:8px 9px;
  font-size:.84rem;
  text-align:left;
  vertical-align:top;
  word-break: break-word;
}
.tight-table th {
  color:#9fb0c8;
  font-weight:700;
  background: rgba(20,29,44,.85);
}
.tight-table td {color:#f5f8fd;}
.col-compact {white-space: nowrap;}
.note-box {
  border:1px solid #27476e;
  background: rgba(18,46,79,.65);
  border-radius:12px;
  padding:12px 14px;
}
.mini-caption {
  color: var(--muted);
  font-size: .82rem;
  margin-top: 2px;
  margin-bottom: 8px;
}
div[data-baseweb="tab-list"] {gap: 0.45rem;}
button[data-baseweb="tab"] {padding-left:0.15rem; padding-right:0.15rem;}
</style>
""", unsafe_allow_html=True)

# --------------------
# HELPERS
# --------------------
def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def pct(x: float) -> str:
    return f"{100*x:.1f}%"

def pill_html(text: str, red: bool = False) -> str:
    cls = "pill-red" if red else "pill"
    return f"<span class='{cls}'>{text}</span>"

def bucket(x: float, cuts: Tuple[float, float], labels: Tuple[str, str, str]) -> str:
    if x < cuts[0]:
        return labels[0]
    if x < cuts[1]:
        return labels[1]
    return labels[2]

def score_label(x: float) -> str:
    return bucket(x, (0.33, 0.66), ("Weak", "Medium", "Strong"))

def robust_z(s: pd.Series, lookback: int = 36) -> float:
    s = s.dropna()
    if len(s) < max(12, lookback // 4):
        return 0.0
    hist = s.iloc[-lookback:]
    sd = hist.std(ddof=0)
    if not np.isfinite(sd) or sd < 1e-9:
        return 0.0
    return float((hist.iloc[-1] - hist.mean()) / sd)

def table_html(headers: List[str], rows: List[List[str]]) -> str:
    th = "".join([f"<th>{h}</th>" for h in headers])
    trs = []
    for row in rows:
        cells = "".join([f"<td>{cell}</td>" for cell in row])
        trs.append(f"<tr>{cells}</tr>")
    return f"<div class='tight-table'><table><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table></div>"

def ladder_state(score: float, side: str) -> str:
    if side == "top":
        if score < 0.18:
            return "No clean top"
        if score < 0.35:
            return "Building top"
        if score < 0.55:
            return "Provisional top"
        return "Extended / blow-off risk"
    else:
        if score < 0.18:
            return "No clean bottom"
        if score < 0.35:
            return "Building bottom"
        if score < 0.55:
            return "Provisional bottom"
        return "Deep washout / capitulation"


def signal_quality_label(confidence: float, agreement: float, fragility: float) -> str:
    score = 0.5 * confidence + 0.3 * agreement + 0.2 * (1 - fragility)
    if score > 0.72:
        return "High"
    if score > 0.52:
        return "Medium"
    return "Low"

def exposure_posture(confidence: float, fragility: float) -> str:
    x = 0.6 * confidence + 0.4 * (1 - fragility)
    if x > 0.72:
        return "Can size up selectively"
    if x > 0.54:
        return "Normal sizing only"
    return "Keep sizing small"

def interpret_relative(direction: str, state: str, quality: str) -> str:
    if direction == "Balanced":
        return "No strong edge yet"
    if direction == "Stronger" and state in ["Building", "Stable"]:
        return "Edge looks usable"
    if direction == "Stronger" and quality == "Frothy":
        return "Strong but stretched"
    if direction == "Weaker" and state in ["Building", "Stable"]:
        return "Still under pressure"
    return "Watch for confirmation"

@st.cache_data(ttl=60*60*6, show_spinner=False)
def fred_series(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        df = pd.read_csv(url)
        dcol, vcol = df.columns[0], df.columns[1]
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
        df[vcol] = pd.to_numeric(df[vcol], errors="coerce")
        s = df.dropna().set_index(dcol)[vcol]
        s.name = series_id
        return s
    except Exception:
        return pd.Series(dtype=float, name=series_id)

@st.cache_data(ttl=60*60*4, show_spinner=False)
def yahoo_close(ticker: str, period: str = "1y") -> pd.Series:
    if yf is None:
        return pd.Series(dtype=float, name=ticker)
    try:
        data = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if data is None or len(data) == 0:
            return pd.Series(dtype=float, name=ticker)
        if isinstance(data.columns, pd.MultiIndex):
            if ("Close", ticker) in data.columns:
                s = data[("Close", ticker)]
            elif "Close" in data.columns.get_level_values(0):
                s = data["Close"].iloc[:, 0]
            else:
                s = data.iloc[:, 0]
        else:
            s = data["Close"] if "Close" in data.columns else data.iloc[:, 0]
        s = pd.to_numeric(s, errors="coerce").dropna()
        s.name = ticker
        return s
    except Exception:
        return pd.Series(dtype=float, name=ticker)

@st.cache_data(ttl=60*60*4, show_spinner=False)
def yahoo_close_batch(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    tickers = [t for t in tickers if t]
    if yf is None or not tickers:
        return pd.DataFrame()
    # Fallback ladder improves survival on Streamlit Cloud where big downloads sometimes fail.
    for chunk_size in (max(1, min(40, len(tickers))), 15, 8, 1):
        frames = []
        ok = True
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i+chunk_size]
            try:
                data = yf.download(
                    chunk,
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
            except Exception:
                ok = False
                break
            if data is None or len(data) == 0:
                continue
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if "Close" in data.columns.get_level_values(0):
                        close = data["Close"].copy()
                    else:
                        close = data.iloc[:, :len(chunk)].copy()
                        close.columns = chunk[:close.shape[1]]
                else:
                    # Single ticker case
                    col = "Close" if "Close" in data.columns else data.columns[0]
                    close = data[[col]].copy()
                    close.columns = [chunk[0]]
                close = close.apply(pd.to_numeric, errors="coerce")
                frames.append(close)
            except Exception:
                ok = False
                break
        if ok and frames:
            merged = pd.concat(frames, axis=1)
            merged = merged.loc[:, ~merged.columns.duplicated()].sort_index()
            return merged
    return pd.DataFrame()

def ret_n(s: pd.Series, n: int = 21) -> float:
    if s.empty or len(s) < n + 1:
        return 0.0
    return float(s.iloc[-n:].pct_change().add(1).prod() - 1)

def stretch_n(s: pd.Series, n: int = 63) -> float:
    if s.empty or len(s) < n:
        return 0.0
    return float((s.iloc[-1] / s.iloc[-n:].mean()) - 1)

# --------------------
# DATA
# --------------------
SER_RAW = {
    "INDPRO": fred_series("INDPRO"),
    "RSAFS": fred_series("RSAFS"),
    "PAYEMS": fred_series("PAYEMS"),
    "UNRATE": fred_series("UNRATE"),
    "ICSA": fred_series("ICSA"),
    "CPI": fred_series("CPIAUCSL"),
    "CORE_CPI": fred_series("CPILFESL"),
    "PPI": fred_series("PPIACO"),
    "SAHM": fred_series("SAHMCURRENT"),
    "NFCI": fred_series("NFCI"),
    "HY": fred_series("BAMLH0A0HYM2"),
    "WTI": fred_series("DCOILWTICO"),
    "WALCL": fred_series("WALCL"),
    "M2": fred_series("M2SL"),
    "USD": fred_series("DTWEXBGS"),
}
ASOF_TS = pd.Timestamp(date.today())
SER = adjust_series_dict(SER_RAW, ASOF_TS) if adjust_series_dict is not None else SER_RAW
SNAPSHOT_DIR = ".livequad_snapshots"

# --------------------
# CORE ENGINE (single source of truth)
# --------------------
def _cut_asof(s: pd.Series, dt: pd.Timestamp | None = None) -> pd.Series:
    s = s.dropna()
    if dt is None or s.empty:
        return s
    return s.loc[:dt].dropna()


def _latest_common_date(keys: List[str]) -> pd.Timestamp | None:
    dates = []
    for key in keys:
        s = SER.get(key, pd.Series(dtype=float)).dropna()
        if s.empty:
            return None
        dates.append(s.index.max())
    return min(dates) if dates else None


def _annualized_n(s: pd.Series, n: int = 3) -> pd.Series:
    s = s.dropna()
    if len(s) < n + 1:
        return pd.Series(dtype=float)
    out = (s / s.shift(n)) ** (12.0 / n) - 1.0
    return out.replace([np.inf, -np.inf], np.nan)


def _z_last(s: pd.Series, lookback: int = 36) -> float:
    return robust_z(s, lookback)


def _monthly_last(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return s
    return s.resample("M").last().dropna()


def _quad_probs(g_up: float, i_up: float) -> Dict[str, float]:
    probs = {
        "Q1": g_up * (1 - i_up),
        "Q2": g_up * i_up,
        "Q3": (1 - g_up) * i_up,
        "Q4": (1 - g_up) * (1 - i_up),
    }
    total = sum(probs.values())
    return {k: (v / total if total else 0.25) for k, v in probs.items()}



def _weighted_mean(vals: List[float], weights: List[float]) -> float:
    arr = np.array(vals, dtype=float)
    w = np.array(weights, dtype=float)
    mask = np.isfinite(arr) & np.isfinite(w)
    if not mask.any():
        return 0.0
    arr = arr[mask]
    w = w[mask]
    if w.sum() == 0:
        return float(np.nanmean(arr))
    return float(np.sum(arr * w) / np.sum(w))


def _renorm_probs(probs: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(probs.values()))
    if total <= 0:
        return {k: 0.25 for k in ["Q1", "Q2", "Q3", "Q4"]}
    return {k: float(v / total) for k, v in probs.items()}


def compute_core() -> Dict[str, object]:
    official_dt = _latest_common_date(["INDPRO", "RSAFS", "PAYEMS", "UNRATE", "ICSA", "CPI", "CORE_CPI", "PPI"])

    indpro = _cut_asof(SER["INDPRO"], official_dt)
    rsafs = _cut_asof(SER["RSAFS"], official_dt)
    payems = _cut_asof(SER["PAYEMS"], official_dt)
    unrate = _cut_asof(SER["UNRATE"], official_dt)
    icsa = _cut_asof(SER["ICSA"], official_dt)
    cpi = _cut_asof(SER["CPI"], official_dt)
    core_cpi = _cut_asof(SER["CORE_CPI"], official_dt)
    ppi = _cut_asof(SER["PPI"], official_dt)

    # Released macro: direction-of-change rather than raw levels.
    g_indpro = _z_last(_annualized_n(indpro, 3) - indpro.pct_change(12), 36)
    g_sales = _z_last(_annualized_n(rsafs, 3) - rsafs.pct_change(12), 36)
    g_jobs = _z_last((payems.diff(3) / 3.0) - (payems.diff(12) / 12.0), 36)
    g_unrate = -_z_last(unrate.diff(3), 36)
    g_claims = -_z_last(icsa.rolling(4).mean() - icsa.rolling(26).mean(), 52)
    growth_inputs = [g_indpro, g_sales, g_jobs, g_unrate, g_claims]
    growth_official_weights = [0.22, 0.22, 0.22, 0.18, 0.16]

    i_cpi = _z_last(_annualized_n(cpi, 3) - cpi.pct_change(12), 36)
    i_core = _z_last(_annualized_n(core_cpi, 3) - core_cpi.pct_change(12), 36)
    i_ppi = _z_last(_annualized_n(ppi, 3) - ppi.pct_change(12), 36)

    wti_m = _monthly_last(SER["WTI"])
    hy = SER["HY"].dropna()
    nfci = SER["NFCI"].dropna()
    sahm = SER["SAHM"].dropna()

    oil_3m = _z_last(_annualized_n(wti_m, 3), 36)
    oil_1m = _z_last(_annualized_n(wti_m, 1), 36)

    infl_inputs = [i_cpi, i_core, i_ppi, oil_3m, oil_1m]
    infl_official_inputs = [i_cpi, i_core, i_ppi]
    infl_official_weights = [0.36, 0.40, 0.24]
    i_now_weights = [0.28, 0.32, 0.18, 0.14, 0.08]

    stress_inputs = [
        _z_last(sahm, 36),
        _z_last(nfci, 52),
        _z_last(hy, 52),
    ]

    growth_neg_breadth = float(np.mean([x < 0 for x in growth_inputs]))
    growth_pos_breadth = float(np.mean([x > 0 for x in growth_inputs]))
    infl_pos_breadth = float(np.mean([x > 0 for x in infl_inputs]))
    infl_off_pos_breadth = float(np.mean([x > 0 for x in infl_official_inputs]))

    labor_weak = _weighted_mean([
        max(0.0, -g_jobs),
        max(0.0, -g_unrate),
        max(0.0, -g_claims),
    ], [0.45, 0.30, 0.25])

    g_off_raw = _weighted_mean(growth_inputs, growth_official_weights)
    i_off_raw = _weighted_mean(infl_official_inputs, infl_official_weights)
    i_now_raw = _weighted_mean(infl_inputs, i_now_weights)
    s_m = float(np.nanmean(stress_inputs))

    growth_drag = 0.24 * max(0.0, growth_neg_breadth - 0.35) + 0.28 * max(0.0, labor_weak - 0.06)
    g_off = g_off_raw - 0.12 * growth_drag
    g_now = g_off_raw - growth_drag
    i_off = i_off_raw + 0.10 * max(0.0, infl_off_pos_breadth - 0.50)
    i_now = i_now_raw + 0.20 * max(0.0, infl_pos_breadth - 0.50) + 0.08 * max(0.0, oil_1m)

    g_up_off = sigmoid(1.35 * (g_off - 0.05) - 0.08 * max(0.0, s_m))
    i_up_off = sigmoid(1.35 * (i_off + 0.02) + 0.06 * max(0.0, s_m))
    g_up_now = sigmoid(1.85 * (g_now - 0.12) - 0.18 * max(0.0, s_m) - 0.14 * growth_neg_breadth)
    i_up_now = sigmoid(1.65 * (i_now + 0.04) + 0.10 * max(0.0, s_m))

    official_probs = _quad_probs(g_up_off, i_up_off)
    directional_probs = _quad_probs(g_up_now, i_up_now)

    inflation_push = max(0.0, i_now)
    stag_pressure = clamp01(
        0.45 * sigmoid(2.2 * (labor_weak - 0.08))
        + 0.35 * sigmoid(2.0 * (inflation_push - 0.06))
        + 0.20 * sigmoid(4.0 * (growth_neg_breadth - 0.52))
    )

    q2_veto = (
        stag_pressure > 0.46
        and growth_neg_breadth >= 0.55
        and inflation_push > -0.02
    )
    if q2_veto and directional_probs["Q2"] >= directional_probs["Q3"]:
        shift = min(
            directional_probs["Q2"] * 0.78,
            directional_probs["Q2"] * (0.28 + 0.38 * stag_pressure),
        )
        directional_probs["Q2"] -= shift
        directional_probs["Q3"] += shift * 0.92
        directional_probs["Q4"] += shift * 0.08 * float(inflation_push < 0.08)
        directional_probs = _renorm_probs(directional_probs)

    # Cross-asset live layer: closer to how a macro PM would read the tape.
    def _pair_score(asset_ticker: str, bench_ticker: str = "SPY") -> float:
        asset = yahoo_close(asset_ticker, "1y")
        bench = yahoo_close(bench_ticker, "1y")
        if asset.empty or bench.empty:
            return np.nan
        df = pd.concat([asset.rename("asset"), bench.rename("bench")], axis=1).sort_index().ffill().dropna()
        if len(df) < 40:
            return np.nan
        rel = (df["asset"] / df["bench"]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(rel) < 40:
            return np.nan
        sig = 0.55 * ret_n(rel, 21) + 0.30 * ret_n(rel, 63) + 0.15 * stretch_n(rel, 63)
        return float(math.tanh(sig / 0.08))

    def _abs_score(ticker: str) -> float:
        s = yahoo_close(ticker, "1y")
        if s.empty or len(s) < 40:
            return np.nan
        sig = 0.55 * ret_n(s, 21) + 0.30 * ret_n(s, 63) + 0.15 * stretch_n(s, 63)
        return float(math.tanh(sig / 0.08))

    growth_cross_inputs = [
        _pair_score("IWM", "SPY"),
        _pair_score("XLY", "XLP"),
        _pair_score("XLI", "XLU"),
        _pair_score("XLF", "XLU"),
        _pair_score("HYG", "IEF"),
    ]
    growth_cross_weights = [0.24, 0.24, 0.18, 0.16, 0.18]

    infl_cross_inputs = [
        _pair_score("XLE", "SPY"),
        _pair_score("GLD", "SPY"),
        _pair_score("DBC", "SPY"),
        _abs_score("UUP"),
        -_abs_score("TLT"),
    ]
    infl_cross_weights = [0.28, 0.22, 0.22, 0.12, 0.16]

    g_cross = _weighted_mean(growth_cross_inputs, growth_cross_weights)
    i_cross = _weighted_mean(infl_cross_inputs, infl_cross_weights)
    cross_growth_neg_breadth = float(np.mean([x < -0.02 for x in growth_cross_inputs if np.isfinite(x)])) if any(np.isfinite(x) for x in growth_cross_inputs) else 0.0
    cross_growth_pos_breadth = float(np.mean([x > 0.02 for x in growth_cross_inputs if np.isfinite(x)])) if any(np.isfinite(x) for x in growth_cross_inputs) else 0.0
    cross_infl_pos_breadth = float(np.mean([x > 0.02 for x in infl_cross_inputs if np.isfinite(x)])) if any(np.isfinite(x) for x in infl_cross_inputs) else 0.0

    g_live = 0.42 * g_now + 0.58 * g_cross - 0.10 * growth_drag
    i_live = 0.55 * i_now + 0.45 * i_cross + 0.04 * max(0.0, oil_1m)

    g_up_live = sigmoid(1.95 * (g_live - 0.04) - 0.10 * max(0.0, s_m) - 0.14 * cross_growth_neg_breadth)
    i_up_live = sigmoid(1.80 * (i_live + 0.01) + 0.06 * max(0.0, s_m) + 0.12 * cross_infl_pos_breadth)
    live_probs = _quad_probs(g_up_live, i_up_live)

    q3_live_pressure = clamp01(
        0.30 * sigmoid(2.4 * (stag_pressure - 0.40))
        + 0.30 * sigmoid(2.8 * (-g_cross - 0.02))
        + 0.25 * sigmoid(2.4 * (i_cross - 0.00))
        + 0.15 * sigmoid(4.0 * (cross_growth_neg_breadth - 0.45))
    )

    if q3_live_pressure > 0.42 and live_probs["Q2"] >= live_probs["Q3"]:
        shift = min(
            live_probs["Q2"] * 0.65,
            live_probs["Q2"] * (0.18 + 0.45 * q3_live_pressure),
        )
        live_probs["Q2"] -= shift
        live_probs["Q3"] += shift * 0.94
        live_probs["Q4"] += shift * 0.06 * float(i_cross < 0)
        live_probs = _renorm_probs(live_probs)

    live_blend = {k: 0.18 * official_probs[k] + 0.30 * directional_probs[k] + 0.52 * live_probs[k] for k in official_probs}
    if q2_veto or q3_live_pressure > 0.45:
        bridge = min(live_blend["Q2"] * 0.55, 0.08 + 0.18 * max(stag_pressure, q3_live_pressure))
        live_blend["Q2"] -= bridge
        live_blend["Q3"] += bridge * 0.92
        live_blend["Q4"] += bridge * 0.08 * float(inflation_push < 0.06)
    total = sum(live_blend.values())
    live_blend = {k: (v / total if total else 0.25) for k, v in live_blend.items()}

    off_ranked = sorted(official_probs.items(), key=lambda x: x[1], reverse=True)
    dir_ranked = sorted(directional_probs.items(), key=lambda x: x[1], reverse=True)
    live_ranked = sorted(live_probs.items(), key=lambda x: x[1], reverse=True)
    official_q, official_p = off_ranked[0]
    directional_q, directional_p = dir_ranked[0]
    live_q, live_p = live_ranked[0]

    model_blend = dict(live_blend)
    model_ranked = sorted(model_blend.items(), key=lambda x: x[1], reverse=True)
    model_current_q, model_current_p = model_ranked[0]
    model_next_q, model_next_p = model_ranked[1]

    decision_blend = dict(model_blend)
    anchor_reason = "Model-only"
    q3_anchor_ok = (
        Q3_CONSENSUS_ANCHOR
        and official_q in ["Q2", "Q3", "Q4"]
        and directional_q in ["Q2", "Q3"]
        and (live_q == "Q3" or q3_live_pressure > 0.36 or stag_pressure > 0.42 or directional_probs["Q3"] > 0.28)
    )
    if q3_anchor_ok:
        base_boost = 0.10 + 0.16 * q3_live_pressure + 0.05 * float(live_q == "Q3") + 0.04 * float(directional_q == "Q3")
        pool = decision_blend.get("Q2", 0.0) * 0.95 + decision_blend.get("Q4", 0.0) * 0.40 + decision_blend.get("Q1", 0.0) * 0.15
        anchor_shift = min(pool, base_boost)
        from_q2 = min(decision_blend.get("Q2", 0.0) * 0.82, anchor_shift * 0.74)
        from_q4 = min(decision_blend.get("Q4", 0.0) * 0.45, max(0.0, anchor_shift - from_q2))
        from_q1 = min(decision_blend.get("Q1", 0.0) * 0.18, max(0.0, anchor_shift - from_q2 - from_q4))
        decision_blend["Q2"] -= from_q2
        decision_blend["Q4"] -= from_q4
        decision_blend["Q1"] -= from_q1
        decision_blend["Q3"] += from_q2 + from_q4 + from_q1

        if Q3_CONSENSUS_FORCE_TOP:
            other_top = max(v for k, v in decision_blend.items() if k != "Q3")
            if decision_blend["Q3"] <= other_top:
                needed = other_top - decision_blend["Q3"] + 0.012
                take_q2 = min(decision_blend.get("Q2", 0.0) * 0.65, needed * 0.72)
                take_q4 = min(decision_blend.get("Q4", 0.0) * 0.32, max(0.0, needed - take_q2))
                take_q1 = min(decision_blend.get("Q1", 0.0) * 0.08, max(0.0, needed - take_q2 - take_q4))
                decision_blend["Q2"] -= take_q2
                decision_blend["Q4"] -= take_q4
                decision_blend["Q1"] -= take_q1
                decision_blend["Q3"] += take_q2 + take_q4 + take_q1
        decision_blend = _renorm_probs(decision_blend)
        anchor_reason = "Q3-anchored live regime"

    ranked = sorted(decision_blend.items(), key=lambda x: x[1], reverse=True)
    current_q, current_p = ranked[0]
    next_q, next_p = ranked[1]

    dist_od = sum(abs(official_probs[k] - directional_probs[k]) for k in official_probs)
    dist_ol = sum(abs(official_probs[k] - live_probs[k]) for k in official_probs)
    dist_dl = sum(abs(directional_probs[k] - live_probs[k]) for k in official_probs)
    agreement = clamp01(1.0 - 0.18 * dist_od - 0.18 * dist_ol - 0.24 * dist_dl)
    confidence = clamp01(0.52 * current_p + 0.22 * agreement + 0.26 * max(live_p, directional_p))

    phase_strength = clamp01(0.34 * abs(g_live) + 0.34 * abs(i_live) + 0.18 * max(0.0, current_p - 0.25) + 0.14 * q3_live_pressure)
    breadth = 0.34 * growth_pos_breadth + 0.26 * infl_off_pos_breadth + 0.20 * cross_growth_pos_breadth + 0.20 * cross_infl_pos_breadth
    regime_divergence = abs(g_live - g_off) + abs(i_live - i_off)
    fragility = clamp01(
        0.34 * (1 - current_p)
        + 0.26 * (1 - agreement)
        + 0.16 * max(0.0, s_m)
        + 0.14 * min(1.0, regime_divergence / 2.5)
        + 0.10 * abs(current_p - live_p)
    )

    margin = current_p - next_p
    if margin < 0.03:
        fragility = clamp01(fragility + 0.08)
    if margin < 0.015:
        confidence = clamp01(confidence - 0.05)

    if current_q == "Q1":
        sub_phase = "Goldilocks / recovery" if fragility < 0.45 else "Recovery but fragile"
    elif current_q == "Q2":
        sub_phase = "Reflation / heating up" if q3_live_pressure < 0.40 else "Hot but cross-asset not fully confirmed"
    elif current_q == "Q3":
        sub_phase = "Live stagflation / cross-asset Q3" if q3_live_pressure > 0.55 else "Stagflation building"
    else:
        sub_phase = "Late Q4 / inflation trying to turn" if live_q == "Q3" else ("Bottoming attempt" if g_live > -0.20 else "Deflationary slowdown")

    top_score = clamp01(0.32 * max(0.0, i_live) + 0.18 * phase_strength + 0.16 * fragility + 0.18 * float(current_q in ["Q2", "Q3"]) + 0.16 * q3_live_pressure)
    bottom_score = clamp01(0.34 * float(current_q == "Q4") + 0.18 * max(0.0, -g_live) + 0.18 * (1 - fragility) + 0.15 * float(official_q == "Q4") + 0.15 * float(live_q == "Q4"))
    higher_top = clamp01(top_score * 0.65 * (1 - bottom_score))
    lower_bottom = clamp01(bottom_score * 0.65 * (1 - top_score))

    transition_pressure = clamp01(0.32 * fragility + 0.24 * (1 - current_p) + 0.20 * min(1.0, regime_divergence / 2.5) + 0.14 * abs(g_live - i_live) / 3.0 + 0.10 * q3_live_pressure)
    transition_conviction = clamp01(0.52 * transition_pressure + 0.48 * next_p)
    stay_probability = clamp01(current_p * (1 - fragility * 0.45))

    if margin > 0.05:
        if transition_conviction > 0.70:
            path_status = "Valid"
        elif transition_conviction > 0.48:
            path_status = "Building"
        elif transition_conviction > 0.28:
            path_status = "Starting"
        else:
            path_status = "Stable / no clean shift"
    else:
        if transition_conviction > 0.78 and next_p > current_p - 0.01:
            path_status = "Confirmed"
        elif transition_conviction > 0.60:
            path_status = "Valid"
        elif transition_conviction > 0.42:
            path_status = "Building"
        elif transition_conviction > 0.26:
            path_status = "Starting"
        else:
            path_status = "Stable / no clean shift"

    return {
        "monthly": official_probs,
        "quarterly": directional_probs,
        "blend": decision_blend,
        "model_blend": model_blend,
        "current_q": current_q,
        "current_p": current_p,
        "next_q": next_q,
        "next_p": next_p,
        "model_current_q": model_current_q,
        "model_current_p": model_current_p,
        "model_next_q": model_next_q,
        "model_next_p": model_next_p,
        "official_q": official_q,
        "official_p": official_p,
        "directional_q": directional_q,
        "directional_p": directional_p,
        "live_q": live_q,
        "live_p": live_p,
        "anchor_reason": anchor_reason,
        "official_date": official_dt.strftime("%Y-%m-%d") if official_dt is not None else "n/a",
        "agreement": agreement,
        "confidence": confidence,
        "phase_strength": phase_strength,
        "breadth": breadth,
        "fragility": fragility,
        "sub_phase": sub_phase,
        "higher_top_prob": higher_top,
        "lower_bottom_prob": lower_bottom,
        "higher_top": higher_top,
        "lower_bottom": lower_bottom,
        "top_score": top_score,
        "bottom_score": bottom_score,
        "stress_growth": clamp01(sigmoid(1.25 * (-g_live + 0.05))),
        "stress_infl": clamp01(sigmoid(1.25 * (i_live + 0.02))),
        "stress_liq": clamp01(sigmoid(1.25 * s_m)),
        "transition_prob": transition_conviction,
        "transition_conviction": transition_conviction,
        "transition_pressure": transition_pressure,
        "stay_prob": stay_probability,
        "stay_probability": stay_probability,
        "margin": margin,
        "path_status": path_status,
        "signal_quality": signal_quality_label(confidence, agreement, fragility),
        "g_off": g_off,
        "i_off": i_off,
        "g_now": g_now,
        "i_now": i_now,
        "g_live": g_live,
        "i_live": i_live,
        "g_cross": g_cross,
        "i_cross": i_cross,
        "g_off_raw": g_off_raw,
        "i_off_raw": i_off_raw,
        "i_now_raw": i_now_raw,
        "stress_m": s_m,
        "labor_weak": labor_weak,
        "stag_pressure": stag_pressure,
        "q3_live_pressure": q3_live_pressure,
        "q2_veto": q2_veto,
        "growth_neg_breadth": growth_neg_breadth,
        "growth_pos_breadth": growth_pos_breadth,
        "infl_pos_breadth": infl_pos_breadth,
        "cross_growth_neg_breadth": cross_growth_neg_breadth,
        "cross_infl_pos_breadth": cross_infl_pos_breadth,
        "blended": live_blend,
    }


core = compute_core()

DATA_HEALTH_ROWS = health_rows(SER_RAW, SER, ASOF_TS) if health_rows is not None else []
DATA_HEALTH_SUMMARY = health_summary(DATA_HEALTH_ROWS) if health_summary is not None else []
SNAPSHOT_PATH = None
SNAPSHOT_COUNT, SNAPSHOT_LAST = snapshot_coverage(SNAPSHOT_DIR) if snapshot_coverage is not None else (0, "n/a")
VALIDATION_READY = None
WFO_DF = pd.DataFrame()
WFO_SUMMARY_ROWS = []
WFO_CAL_ROWS = []
WFO_CRASH_CAL_ROWS = []


# --------------------
# RELATIVE / SIZE
# --------------------
def rel_state(spread: float) -> str:
    if spread > 0.08:
        return "Building"
    if spread > 0.02:
        return "Stable"
    if spread > -0.02:
        return "Peaking"
    return "Fading"

def rel_quality(strength: float, breadth: float, stretch: float) -> str:
    if stretch > 0.12 and strength > 0.55:
        return "Frothy"
    if breadth < 0.35 and strength > 0.45:
        return "Narrow"
    if strength < 0.25:
        return "Weak"
    return "Healthy"

def rel_sustainability(strength: float, quality: str, fragility: float) -> str:
    x = 0.45 * strength + 0.30 * (1 - fragility) + 0.25 * float(quality in ["Healthy", "Narrow"])
    if quality == "Frothy":
        x -= 0.18
    return bucket(x, (0.45, 0.72), ("Low", "Medium", "High"))

def rel_confirmation(strength: float, state: str, breadth: float) -> str:
    score = 0
    score += 1 if strength > 0.45 else 0
    score += 1 if state in ["Building", "Stable"] else 0
    score += 1 if breadth > 0.40 else 0
    if score >= 3:
        return "Confirmed"
    if score == 2:
        return "Partial"
    return "Not confirmed"

def fallback_rel_row(name: str, seed: int, proxy_note: str = "") -> Dict[str, str]:
    strength_num = clamp01(0.20 + 0.07 * seed + 0.28 * core["breadth"] + 0.12 * (1 - core["fragility"]))
    spread = (seed - 1.5) * 0.035 + (core["phase_strength"] - 0.5) * 0.04
    direction = "Stronger" if spread > 0.025 else ("Weaker" if spread < -0.025 else "Balanced")
    state = ["Fading", "Peaking", "Stable", "Building"][(seed + int(core["phase_strength"] * 10)) % 4]
    stretch = clamp01(0.03 + 0.025 * seed + core["top_score"] * 0.12)
    quality = rel_quality(strength_num, core["breadth"], stretch)
    sustain = rel_sustainability(strength_num, quality, core["fragility"])
    confirm = rel_confirmation(strength_num, state, core["breadth"])
    return {
        "Lens": name,
        "Direction": direction,
        "Strength": score_label(strength_num),
        "StrengthScore": pct(strength_num),
        "State": state,
        "Quality": quality,
        "Sustainability": sustain,
        "Confirmation": confirm,
        "Read": f"Fallback / weak edge{proxy_note}",
    }

def build_rel_row(name: str, a: pd.Series, b: pd.Series, pos_label: str, neg_label: str, breadth_hint: float, seed: int, proxy_note: str = "") -> Dict[str, str]:
    if a.empty or b.empty:
        return fallback_rel_row(name, seed, proxy_note)
    a_r = ret_n(a, 21)
    b_r = ret_n(b, 21)
    spread = a_r - b_r
    direction = pos_label if spread > 0.02 else (neg_label if spread < -0.02 else "Balanced")
    strength_num = clamp01(min(1.0, abs(spread) / 0.15))
    state = rel_state(spread)
    stretch = clamp01(abs(stretch_n(a, 63)) * 3.0)
    quality = rel_quality(strength_num, breadth_hint, stretch)
    sustain = rel_sustainability(strength_num, quality, core["fragility"])
    confirm = rel_confirmation(strength_num, state, breadth_hint)
    read = direction if direction != "Balanced" else f"Balanced / weak edge{proxy_note}"
    return {
        "Lens": name,
        "Direction": direction,
        "Strength": score_label(strength_num),
        "StrengthScore": pct(strength_num),
        "State": state,
        "Quality": quality,
        "Sustainability": sustain,
        "Confirmation": confirm,
        "Read": read,
    }

def crypto_alt_basket(period: str = "1y") -> pd.Series:
    tickers = ["ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "AVAX-USD", "LINK-USD"]
    series = []
    for t in tickers:
        s = yahoo_close(t, period)
        if not s.empty:
            s = s / s.iloc[0]
            series.append(s.rename(t))
    if not series:
        return pd.Series(dtype=float, name="ALT_BASKET")
    df = pd.concat(series, axis=1).dropna(how="all")
    return df.mean(axis=1).dropna().rename("ALT_BASKET")

def liquidity_composite() -> pd.Series:
    parts = []
    if not SER["WALCL"].empty:
        s = SER["WALCL"].dropna()
        parts.append((s / s.iloc[0]).rename("WALCL"))
    if not SER["M2"].empty:
        s = SER["M2"].dropna()
        parts.append((s / s.iloc[0]).rename("M2"))
    if not SER["USD"].empty:
        s = SER["USD"].dropna()
        parts.append((1 / s).rename("USD_INV"))
    if not SER["NFCI"].empty:
        s = SER["NFCI"].dropna()
        z = (s - s.mean()) / max(s.std(ddof=0), 1e-9)
        parts.append((1 / (1 + np.exp(z))).rename("NFCI_INV"))
    if not parts:
        return pd.Series(dtype=float, name="LIQ")
    df = pd.concat(parts, axis=1).dropna(how="all")
    return df.mean(axis=1).dropna().rename("LIQ")


def _norm_tanh(x: float, scale: float) -> float:
    if not np.isfinite(x):
        return 0.5
    scale = max(scale, 1e-9)
    return clamp01(0.5 + 0.5 * math.tanh(x / scale))


def _trend_score(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 30:
        return 0.0
    px = float(s.iloc[-1])
    sma20 = float(s.rolling(20).mean().iloc[-1]) if len(s) >= 20 else px
    sma50 = float(s.rolling(50).mean().iloc[-1]) if len(s) >= 50 else sma20
    long_n = 200 if len(s) >= 200 else min(120, len(s))
    sma_long = float(s.rolling(long_n).mean().iloc[-1]) if long_n >= 20 else sma50
    checks = [px > sma20, px > sma50, px > sma_long, sma20 > sma50, sma50 > sma_long]
    return float(np.mean(checks))


def _lead_state(alpha21: float, alpha63: float, alpha126: float, trend: float) -> str:
    if alpha21 > 0.02 and alpha63 > 0.04 and trend >= 0.60:
        return "Leader"
    if alpha21 > 0.015 and (alpha63 > -0.01 or alpha21 > alpha63 + 0.03) and trend >= 0.45:
        return "Emerging"
    if alpha21 < -0.02 and alpha63 > 0.01:
        return "Fading"
    if alpha21 < -0.03 and alpha63 < -0.03:
        return "Weak"
    if alpha126 > 0.08 and alpha21 > -0.01:
        return "Leader"
    return "Neutral"


def _lead_comment(state: str, alpha21: float, alpha63: float, trend: float) -> str:
    if state == "Leader":
        if alpha21 > 0.06 and alpha63 > 0.10:
            return "clear relative leader"
        return "steady relative strength"
    if state == "Emerging":
        if alpha21 > alpha63 + 0.04:
            return "starting to outperform"
        return "early RS improvement"
    if state == "Fading":
        return "still above benchmark, but momentum fading"
    if state == "Weak":
        return "underperforming benchmark"
    return "mixed / no clean edge"


def _clean_tickers(raw: str, suffix: str = "") -> List[str]:
    out = []
    for x in raw.split(','):
        t = x.strip().upper()
        if not t:
            continue
        if suffix and not t.endswith(suffix):
            t = f"{t}{suffix}"
        out.append(t)
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def rank_market_leaders(tickers: List[str], benchmark_ticker: str, period: str = "1y", fallback_benchmark: str | None = None) -> pd.DataFrame:
    tickers = [t for t in tickers if t]
    if not tickers:
        return pd.DataFrame()

    fetch = [benchmark_ticker] + tickers
    if fallback_benchmark:
        fetch.append(fallback_benchmark)
    prices = yahoo_close_batch(fetch, period)

    # Streamlit Cloud can fail on large batch downloads. Fall back to single-name fetches.
    enough_cols = set(prices.columns) if not prices.empty else set()
    if prices.empty or len(enough_cols.intersection(set(tickers))) < max(3, min(8, len(tickers) // 4 or 1)):
        frames = []
        for ticker in fetch:
            s = yahoo_close(ticker, period)
            if not s.empty:
                frames.append(s.rename(ticker))
        prices = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
    if prices.empty:
        return pd.DataFrame()

    prices = prices.sort_index().ffill().dropna(how='all')

    def _pick_benchmark(name: str | None) -> pd.Series:
        if not name or name not in prices.columns:
            return pd.Series(dtype=float)
        s = prices[name].dropna()
        return s if len(s) >= 25 else pd.Series(dtype=float)

    bench_name = benchmark_ticker
    bench = _pick_benchmark(benchmark_ticker)
    if bench.empty and fallback_benchmark:
        cand = _pick_benchmark(fallback_benchmark)
        if not cand.empty:
            bench = cand
            bench_name = fallback_benchmark

    # Final fallback: equal-weight basket of the available universe.
    if bench.empty:
        series = []
        for ticker in tickers:
            if ticker not in prices.columns:
                continue
            s = prices[ticker].dropna()
            if len(s) < 25:
                continue
            series.append((s / s.iloc[0]).rename(ticker))
        if series:
            basket = pd.concat(series, axis=1).dropna(how='all').mean(axis=1).dropna()
            if not basket.empty:
                bench = basket
                bench_name = 'INTERNAL_BASKET'
    if bench.empty:
        return pd.DataFrame()

    rows = []
    min_len = 45 if period == "1y" else 25
    for ticker in tickers:
        if ticker not in prices.columns:
            continue
        s = prices[ticker].dropna()
        if s.empty:
            continue
        df = pd.concat([s.rename('asset'), bench.rename('bench')], axis=1).sort_index().ffill().dropna()
        if len(df) < min_len:
            continue
        asset = df['asset']
        bm = df['bench']
        r21, r63, r126 = ret_n(asset, 21), ret_n(asset, 63), ret_n(asset, 126)
        b21, b63, b126 = ret_n(bm, 21), ret_n(bm, 63), ret_n(bm, 126)
        alpha21, alpha63, alpha126 = r21 - b21, r63 - b63, r126 - b126
        trend = _trend_score(asset)
        accel = alpha21 - 0.45 * alpha63
        rs_score = clamp01(
            0.35 * _norm_tanh(alpha21, 0.08)
            + 0.30 * _norm_tanh(alpha63, 0.15)
            + 0.15 * _norm_tanh(alpha126, 0.25)
            + 0.20 * trend
        )
        start_score = clamp01(
            0.35 * _norm_tanh(accel, 0.08)
            + 0.25 * _norm_tanh(r21, 0.12)
            + 0.20 * trend
            + 0.20 * clamp01(1.0 - min(1.0, abs(alpha126) / 0.35))
        )
        state = _lead_state(alpha21, alpha63, alpha126, trend)
        comment = _lead_comment(state, alpha21, alpha63, trend)
        if bench_name == 'INTERNAL_BASKET' and state == 'Neutral' and alpha21 > 0.01 and trend >= 0.50:
            state = 'Emerging'
            comment = 'outperforming local basket'
        rows.append({
            'Ticker': ticker,
            'Benchmark': bench_name,
            'State': state,
            'RSScore': rs_score,
            'StartScore': start_score,
            'Alpha21': alpha21,
            'Alpha63': alpha63,
            'Alpha126': alpha126,
            'Ret21': r21,
            'Ret63': r63,
            'Trend': trend,
            'Comment': comment,
            'Bars': len(df),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(['RSScore', 'StartScore', 'Alpha21'], ascending=False).reset_index(drop=True)
    return df


def leadership_table_rows(df: pd.DataFrame, mode: str = "top", n: int = 8) -> List[List[str]]:
    if df is None or df.empty:
        return [["No data", "-", "-", "-", "-", "-"]]
    use = df.copy()
    if mode == "leaders":
        use = use[use['State'].isin(['Leader'])].sort_values(['RSScore', 'Alpha21'], ascending=False)
    elif mode == "emerging":
        use = use[use['State'].isin(['Emerging'])].sort_values(['StartScore', 'Alpha21'], ascending=False)
    elif mode == "fading":
        use = use[use['State'].isin(['Fading', 'Weak'])].sort_values(['Alpha21', 'Alpha63'], ascending=True)
    else:
        use = use.sort_values(['RSScore', 'StartScore'], ascending=False)
    if use.empty:
        return [["None", "-", "-", "-", "-", "No clean names"]]
    out = []
    for _, r in use.head(n).iterrows():
        out.append([
            r['Ticker'].replace('.JK', ''),
            r['State'],
            pct(float(r['RSScore'])),
            pct(float(r['StartScore'])),
            f"{100*float(r['Alpha21']):+.1f}% / {100*float(r['Alpha63']):+.1f}%",
            r['Comment'],
        ])
    return out


def leadership_summary(df: pd.DataFrame) -> Tuple[str, str]:
    if df is None or df.empty:
        return "No data", "No benchmark-relative read"
    leaders = df[df['State'] == 'Leader'].head(3)['Ticker'].str.replace('.JK', '', regex=False).tolist()
    emerging = df[df['State'] == 'Emerging'].sort_values(['StartScore', 'Alpha21'], ascending=False).head(3)['Ticker'].str.replace('.JK', '', regex=False).tolist()
    lead_txt = ', '.join(leaders) if leaders else 'No clear leaders'
    em_txt = ', '.join(emerging) if emerging else 'No early starters'
    return lead_txt, em_txt



US_UNIVERSE = [
    "AAPL","MSFT","NVDA","AVGO","AMD","MU","MRVL","ANET","ARM","TSM",
    "META","AMZN","GOOGL","NFLX","ORCL","PLTR","CRWD","PANW","NET","DDOG",
    "JPM","GS","MS","BAC","WFC","BRK-B","V","MA","SPGI","KKR",
    "XOM","CVX","COP","SLB","EOG","FANG","MPC","VLO","OXY","HAL",
    "LLY","ABBV","JNJ","UNH","ISRG","BSX","ABT","PFE","VRTX","MRK",
    "CAT","GE","GEV","RTX","LMT","NOC","GD","DE","ETN","PH",
    "WMT","COST","PG","KO","PEP","MCD","CMG","HD","LOW","TJX",
    "GLD","NEM","AEM","FCX","SCCO","NUE","STLD","CLF","AA","X",
    "UBER","BKNG","AXON","HOOD","RBLX","SNOW","MDB","ZS","SMCI","QCOM"
]

IHSG_UNIVERSE = [
    "BBCA.JK","BBRI.JK","BMRI.JK","BBNI.JK","TLKM.JK","ASII.JK","ICBP.JK","INDF.JK","CPIN.JK","AMRT.JK",
    "ANTM.JK","MDKA.JK","INCO.JK","UNTR.JK","ADRO.JK","PTBA.JK","ITMG.JK","INDY.JK","MEDC.JK","AKRA.JK",
    "BREN.JK","AMMN.JK","TPIA.JK","BRPT.JK","PGEO.JK","ESSA.JK","ELSA.JK","HUMI.JK","GTSI.JK","RAJA.JK",
    "CTRA.JK","BSDE.JK","PWON.JK","SMRA.JK","MTLA.JK","DMAS.JK","TRIN.JK","TRUE.JK","ROCK.JK","SMDM.JK",
    "EXCL.JK","ISAT.JK","MAPI.JK","ACES.JK","ERAA.JK","MAPA.JK","EMTK.JK","SCMA.JK","HEAL.JK","SILO.JK",
    "JSMR.JK","WIKA.JK","PTPP.JK","ADHI.JK","ENRG.JK","DOID.JK","WINS.JK","TMAS.JK","DEWA.JK","MBMA.JK"
]

US_THEME_TAGS = {
    "Quantum / speculative tech": {"IONQ","QBTS","RGTI","QUBT","QMCO","ARQQ"},
    "Semis / AI infra": {"NVDA","AVGO","AMD","MU","MRVL","ANET","ARM","TSM","SMCI","QCOM"},
    "Software / AI apps": {"MSFT","ORCL","PLTR","CRWD","PANW","NET","DDOG","SNOW","MDB","ZS"},
    "Energy": {"XOM","CVX","COP","SLB","EOG","FANG","MPC","VLO","OXY","HAL"},
    "Defense / industrial": {"RTX","LMT","NOC","GD","CAT","GE","GEV","DE","ETN","PH","AXON"},
    "Financials": {"JPM","GS","MS","BAC","WFC","BRK-B","V","MA","SPGI","KKR"},
    "Gold / metals": {"GLD","NEM","AEM","FCX","SCCO","NUE","STLD","CLF","AA","X"},
    "Consumer / quality": {"WMT","COST","PG","KO","PEP","MCD","CMG","HD","LOW","TJX"},
    "Health care": {"LLY","ABBV","JNJ","UNH","ISRG","BSX","ABT","PFE","VRTX","MRK"},
}

IHSG_THEME_TAGS = {
    "Banks / large cap": {"BBCA.JK","BBRI.JK","BMRI.JK","BBNI.JK"},
    "Commodities / mining": {"ANTM.JK","MDKA.JK","INCO.JK","ADRO.JK","PTBA.JK","ITMG.JK","INDY.JK","MBMA.JK"},
    "Oil / gas / shipping": {"MEDC.JK","AKRA.JK","ESSA.JK","ELSA.JK","HUMI.JK","GTSI.JK","RAJA.JK","WINS.JK","TMAS.JK","ENRG.JK","DOID.JK"},
    "Property / beta": {"CTRA.JK","BSDE.JK","PWON.JK","SMRA.JK","MTLA.JK","DMAS.JK","TRIN.JK","TRUE.JK","ROCK.JK","SMDM.JK"},
    "Infra / industrial": {"JSMR.JK","WIKA.JK","PTPP.JK","ADHI.JK","AMMN.JK","BREN.JK","TPIA.JK","BRPT.JK","PGEO.JK"},
    "Consumer / telco": {"TLKM.JK","EXCL.JK","ISAT.JK","ICBP.JK","INDF.JK","CPIN.JK","AMRT.JK","MAPI.JK","ACES.JK","ERAA.JK","MAPA.JK"},
}

def _theme_from_tags(ticker: str, mapping: Dict[str, set]) -> str:
    for theme, names in mapping.items():
        if ticker in names:
            return theme
    return "Other"

def leadership_summary(df: pd.DataFrame) -> Tuple[str, str]:
    if df is None or df.empty:
        return "No data", "No benchmark-relative read"
    leaders = df[df['State'] == 'Leader'].head(3)['Ticker'].str.replace('.JK', '', regex=False).tolist()
    emerging = df[df['State'] == 'Emerging'].sort_values(['StartScore', 'Alpha21'], ascending=False).head(3)['Ticker'].str.replace('.JK', '', regex=False).tolist()
    lead_txt = ', '.join(leaders) if leaders else 'No clear leaders'
    em_txt = ', '.join(emerging) if emerging else 'No early starters'
    return lead_txt, em_txt

def _safe_option_index(options: List[str], current: str, fallback: str) -> int:
    target = current if current in options else fallback
    if target not in options:
        target = options[0]
    return options.index(target)

def leadership_theme_rows(df: pd.DataFrame, mapping: Dict[str, set], n: int = 6) -> List[List[str]]:
    if df is None or df.empty:
        return [["No data", "-", "-", "-"]]
    work = df.copy()
    work["Theme"] = work["Ticker"].apply(lambda x: _theme_from_tags(x, mapping))
    agg = (
        work.groupby("Theme", as_index=False)
        .agg(RSScore=("RSScore","mean"), StartScore=("StartScore","mean"), Leaders=("Ticker", lambda s: ", ".join([x.replace('.JK','') for x in list(s.head(3))])))
        .sort_values(["RSScore","StartScore"], ascending=False)
    )
    rows = []
    for _, r in agg.head(n).iterrows():
        rows.append([r["Theme"], pct(float(r["RSScore"])), pct(float(r["StartScore"])), r["Leaders"]])
    return rows



US_PRESETS = {
    'Auto broad scan': US_UNIVERSE,
    'Quantum / speculative tech': ['IONQ', 'QBTS', 'RGTI', 'QUBT', 'QMCO', 'ARQQ'],
    'AI / infra leaders': ['NVDA', 'AVGO', 'AMD', 'ANET', 'MRVL', 'MU', 'ARM', 'SMCI', 'TSM', 'QCOM'],
    'Software / momentum': ['PLTR', 'CRWD', 'NET', 'DDOG', 'SNOW', 'MDB', 'PANW', 'ZS'],
    'Energy / materials': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'FANG', 'MPC', 'VLO', 'FCX', 'NEM'],
    'Defense / industrial': ['RTX', 'LMT', 'NOC', 'GD', 'CAT', 'GE', 'GEV', 'DE', 'ETN', 'PH', 'AXON'],
}

IHSG_PRESETS = {
    'Auto broad scan': IHSG_UNIVERSE,
    'IHSG large caps': ['BBCA.JK', 'BBRI.JK', 'BMRI.JK', 'BBNI.JK', 'TLKM.JK', 'ASII.JK', 'ICBP.JK', 'AMMN.JK', 'BREN.JK', 'TPIA.JK'],
    'IHSG resources': ['ADRO.JK', 'PTBA.JK', 'UNTR.JK', 'ANTM.JK', 'MDKA.JK', 'INCO.JK', 'MEDC.JK', 'HUMI.JK', 'GTSI.JK', 'RAJA.JK'],
    'IHSG property / beta': ['CTRA.JK', 'BSDE.JK', 'PWON.JK', 'SMRA.JK', 'TRIN.JK', 'TRUE.JK', 'MTLA.JK', 'DMAS.JK'],
    'IHSG telco / consumer': ['TLKM.JK', 'EXCL.JK', 'ISAT.JK', 'ICBP.JK', 'INDF.JK', 'CPIN.JK', 'AMRT.JK', 'MAPI.JK', 'ACES.JK', 'ERAA.JK'],
}

def compute_relative() -> List[Dict[str, str]]:
    spy = yahoo_close("SPY", "1y")
    eem = yahoo_close("EEM", "1y")
    eido = yahoo_close("EIDO", "1y")
    btc = yahoo_close("BTC-USD", "1y")
    liq = liquidity_composite()
    return [
        build_rel_row("US/EM", spy, eem, "US stronger", "EM stronger", core["breadth"], 0),
        build_rel_row("IHSG/US", eido, spy, "IHSG stronger", "US stronger", core["breadth"], 1, " (proxy-based)"),
        build_rel_row("IHSG/EM", eido, eem, "IHSG stronger", "EM stronger", core["breadth"], 2, " (proxy-based)"),
        build_rel_row("Crypto/Liq", btc, liq, "Crypto stronger", "Liquidity not confirming", core["phase_strength"], 3, " (liq proxy)"),
    ]

def compute_size() -> List[Dict[str, str]]:
    iwm = yahoo_close("IWM", "1y")
    iwb = yahoo_close("IWB", "1y")
    spy = yahoo_close("SPY", "1y")
    alt = crypto_alt_basket("1y")
    btc = yahoo_close("BTC-USD", "1y")
    return [
        build_rel_row("US Small/Big", iwm, iwb, "Small > Big", "Big > Small", core["breadth"], 4),
        build_rel_row("US Small/Broad", iwm, spy, "Small > Broad", "Broad > Small", core["breadth"], 5),
        build_rel_row("Alt Basket/BTC", alt, btc, "Alts > BTC", "BTC > Alts", core["breadth"], 6, " (basket proxy)"),
    ]

relative_rows = compute_relative()
size_rows = compute_size()

def compute_leadership_panels() -> Tuple[pd.DataFrame, pd.DataFrame, str, str, str, str]:
    us_preset = st.session_state.get('us_leader_preset', 'Auto broad scan')
    ihsg_preset = st.session_state.get('ihsg_leader_preset', 'Auto broad scan')
    us_custom_raw = st.session_state.get('us_custom_tickers', '')
    ihsg_custom_raw = st.session_state.get('ihsg_custom_tickers', '')

    us_tickers = list(US_PRESETS.get(us_preset, US_UNIVERSE)) + _clean_tickers(us_custom_raw)
    ihsg_tickers = list(IHSG_PRESETS.get(ihsg_preset, IHSG_UNIVERSE)) + _clean_tickers(ihsg_custom_raw, '.JK')

    us_seen, us_final = set(), []
    for t in us_tickers:
        if t not in us_seen:
            us_seen.add(t)
            us_final.append(t)
    ih_seen, ih_final = set(), []
    for t in ihsg_tickers:
        if t not in ih_seen:
            ih_seen.add(t)
            ih_final.append(t)

    us_df = rank_market_leaders(us_final, benchmark_ticker='SPY', period='1y', fallback_benchmark='QQQ')
    if us_df.empty:
        us_df = rank_market_leaders(us_final, benchmark_ticker='SPY', period='6mo', fallback_benchmark='QQQ')
    if us_df.empty:
        us_df = rank_market_leaders(us_final, benchmark_ticker='QQQ', period='3mo', fallback_benchmark='SPY')

    ihsg_df = rank_market_leaders(ih_final, benchmark_ticker='^JKSE', period='1y', fallback_benchmark='EIDO')
    if ihsg_df.empty:
        ihsg_df = rank_market_leaders(ih_final, benchmark_ticker='EIDO', period='1y', fallback_benchmark='SPY')
    if ihsg_df.empty:
        ihsg_df = rank_market_leaders(ih_final, benchmark_ticker='EIDO', period='6mo', fallback_benchmark='SPY')
    if ihsg_df.empty:
        ihsg_df = rank_market_leaders(ih_final, benchmark_ticker='EIDO', period='3mo', fallback_benchmark='SPY')

    if not us_df.empty:
        us_df["Theme"] = us_df["Ticker"].apply(lambda x: _theme_from_tags(x, US_THEME_TAGS))
    if not ihsg_df.empty:
        ihsg_df["Theme"] = ihsg_df["Ticker"].apply(lambda x: _theme_from_tags(x, IHSG_THEME_TAGS))

    us_lead, us_em = leadership_summary(us_df)
    ih_lead, ih_em = leadership_summary(ihsg_df)
    return us_df, ihsg_df, us_lead, us_em, ih_lead, ih_em

us_leaders_df, ihsg_leaders_df, us_lead_text, us_em_text, ih_lead_text, ih_em_text = compute_leadership_panels()


# --------------------
# OVERLAYS
# --------------------
def fear_greed_value() -> Tuple[int, str]:
    try:
        r = requests.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata", timeout=8)
        if r.ok:
            score = int(r.json()["fear_and_greed"]["score"])
        else:
            score = 35
    except Exception:
        score = 35
    vibe = "Extreme fear" if score < 25 else ("Fear" if score < 45 else ("Neutral" if score < 56 else ("Greed" if score < 76 else "Extreme greed")))
    return score, vibe

def iwm_read() -> str:
    iwm = yahoo_close("IWM", "1y")
    if iwm.empty:
        return "No live read"
    r21 = ret_n(iwm, 21)
    ext = stretch_n(iwm, 63)
    if r21 > 0.08 and ext > 0.08:
        return "Blow-off / beta chase risk"
    if r21 > 0.02:
        return "Healthy small-cap strength"
    if r21 < -0.04:
        return "Small-cap stress"
    return "Mixed"

fg_score, fg_vibe = fear_greed_value()
iwm_overlay = iwm_read()

WHAT_IF = [
    ("Policy rescue", "liquidity eases / squeeze risk", "duration or beta if confirmed", "hard shorting into rescue"),
    ("Growth scare, no recession", "panic without full recession", "duration / defensives", "overpricing depression"),
    ("Commodity spike only", "oil/commodities up without broad growth", "energy / gold", "broad reflation assumption"),
    ("Narrow US leadership", "US up but breadth weak", "quality / megacap only", "assuming healthy broad risk-on"),
]
DIVERGENCE = [
    ("Macro weak, equities strong", "Market may be front-running a soft landing or a squeeze"),
    ("Crypto strong, liquidity weak", "Speculative move / weaker confirmation"),
    ("IHSG strong, US weak", "Possible local or commodity decoupling"),
]
CORR = [
    ("USD up → EM pressure", "High", "strong USD usually tightens EM"),
    ("Oil up → stagflation tail", "High", "energy shock raises inflation risk"),
    ("Yield down → duration tailwind", "Medium", "works better if growth scare is real"),
]
CRASH = [
    ("Liquidity shock", "fast, correlated, violent"),
    ("Growth scare", "duration helps, beta weakens"),
    ("Stagflation shock", "gold / commodity defensives matter more"),
]
FALSE_REC = [
    ("Dead-cat bounce", "Bounce without broad confirmation"),
    ("Second leg risk", "Recovery can fail and retest lower"),
]

def build_shocks() -> Dict[str, Tuple[str, str]]:
    shocks = {
        "Policy shock": ("watch", "Policy matters, but not overriding base case"),
        "Geopolitical shock": ("watch", "Use as modifier, not core phase driver"),
        "Liquidity shock": ("watch", "Matters most if liquidity stress rises sharply"),
        "Inflation shock": ("watch", "Important if inflation re-accelerates"),
        "Growth shock": ("watch", "Important if labor/growth rolls harder"),
        "Anomaly flag": ("watch", "Use if market and macro stop confirming each other"),
    }
    if fg_score > 75 or fg_score < 20:
        shocks["Sentiment stretch"] = ("medium", f"Sentiment stretched: {fg_vibe}")
    else:
        shocks["Sentiment stretch"] = ("low", f"Sentiment not stretched: {fg_vibe}")
    return shocks

shocks = build_shocks()
override_active = any(v[0] in ["medium", "high"] for v in shocks.values())

# --------------------
# PLAYBOOK
# --------------------
FAMILY_SCORE_BY_QUAD = {
    "Q1": {"duration": 0.0, "usd": -0.4, "gold": -0.2, "beta": 0.8, "cyclical": 0.7},
    "Q2": {"duration": -0.8, "usd": 0.0, "gold": 0.0, "beta": 0.6, "cyclical": 0.9},
    "Q3": {"duration": 0.2, "usd": 0.7, "gold": 0.9, "beta": -0.7, "cyclical": -0.8},
    "Q4": {"duration": 0.9, "usd": 0.6, "gold": 0.4, "beta": -0.5, "cyclical": -0.6},
}
FAMILY_TO_ASSETS = {
    "duration": {"US Stocks": ["duration-sensitive quality", "defensives"], "Futures / Commodities": ["rates duration"], "Forex": ["funding currencies"], "Crypto": ["BTC over alts"], "IHSG": ["defensives / rate-sensitive"]},
    "usd": {"US Stocks": ["selective exporters"], "Futures / Commodities": ["USD tailwind trades"], "Forex": ["USD stronger"], "Crypto": ["pressure on weaker beta"], "IHSG": ["IDR-sensitive caution"]},
    "gold": {"US Stocks": ["gold miners"], "Futures / Commodities": ["gold"], "Forex": ["gold-linked defensives"], "Crypto": ["less beta than alts"], "IHSG": ["commodity hedges"]},
    "beta": {"US Stocks": ["small caps / cyclicals"], "Futures / Commodities": ["equity beta"], "Forex": ["high beta FX"], "Crypto": ["alts"], "IHSG": ["local beta"]},
    "cyclical": {"US Stocks": ["industrials", "materials"], "Futures / Commodities": ["industrial commodities"], "Forex": ["commodity FX"], "Crypto": ["risk-on rotation"], "IHSG": ["commodities / cyclicals"]},
}

def current_vs_next_playbook() -> Tuple[Dict[str, List[str]], Dict[str, List[str]], str]:
    cur = core["current_q"]
    nxt = core["next_q"]
    cur_scores = FAMILY_SCORE_BY_QUAD[cur]
    nxt_scores = FAMILY_SCORE_BY_QUAD[nxt]
    cur_sorted = sorted(cur_scores, key=cur_scores.get, reverse=True)
    nxt_sorted = sorted(nxt_scores, key=nxt_scores.get, reverse=True)
    cur_out, nxt_out = {}, {}
    for bucket_name in ["US Stocks", "Futures / Commodities", "Forex", "Crypto", "IHSG"]:
        cur_assets, nxt_assets = [], []
        for fam in cur_sorted[:2]:
            cur_assets.extend(FAMILY_TO_ASSETS[fam][bucket_name][:1])
        for fam in nxt_sorted[:2]:
            nxt_assets.extend(FAMILY_TO_ASSETS[fam][bucket_name][:1])
        cur_out[bucket_name] = cur_assets
        nxt_out[bucket_name] = nxt_assets
    posture = (
        "Aggressive" if core["phase_strength"] > 0.65 and core["fragility"] < 0.35
        else "Balanced" if core["confidence"] > 0.45
        else "Defensive" if core["fragility"] > 0.55
        else "Wait / low conviction"
    )
    return cur_out, nxt_out, posture

play_cur, play_next, posture = current_vs_next_playbook()



# --------------------
# DECISION-SUPPORT OVERLAYS
# --------------------
ASSET_SCORE_BY_QUAD = {
    "US cyclicals": {"Q1": 0.55, "Q2": 0.85, "Q3": -0.55, "Q4": -0.45},
    "US defensives": {"Q1": -0.25, "Q2": -0.45, "Q3": 0.70, "Q4": 0.75},
    "US small caps": {"Q1": 0.50, "Q2": 0.95, "Q3": -0.75, "Q4": -0.65},
    "EM equities": {"Q1": 0.45, "Q2": 0.75, "Q3": -0.40, "Q4": -0.25},
    "IHSG cyclicals": {"Q1": 0.35, "Q2": 0.70, "Q3": -0.20, "Q4": -0.20},
    "Gold / miners": {"Q1": -0.10, "Q2": 0.00, "Q3": 0.90, "Q4": 0.35},
    "Oil / energy": {"Q1": 0.05, "Q2": 0.55, "Q3": 0.80, "Q4": -0.15},
    "USD": {"Q1": -0.40, "Q2": 0.05, "Q3": 0.80, "Q4": 0.55},
    "Duration / bonds": {"Q1": -0.30, "Q2": -0.85, "Q3": 0.10, "Q4": 0.90},
    "BTC / crypto beta": {"Q1": 0.60, "Q2": 0.75, "Q3": -0.35, "Q4": -0.10},
}

CURRENCY_SCORE_BY_QUAD = {
    "USD": {"Q1": -0.35, "Q2": 0.05, "Q3": 0.85, "Q4": 0.55},
    "JPY": {"Q1": -0.15, "Q2": -0.45, "Q3": 0.55, "Q4": 0.70},
    "CHF": {"Q1": -0.10, "Q2": -0.35, "Q3": 0.45, "Q4": 0.55},
    "EUR": {"Q1": 0.10, "Q2": 0.20, "Q3": -0.05, "Q4": 0.00},
    "GBP": {"Q1": 0.05, "Q2": 0.15, "Q3": -0.05, "Q4": -0.05},
    "AUD": {"Q1": 0.35, "Q2": 0.75, "Q3": -0.60, "Q4": -0.35},
    "NZD": {"Q1": 0.35, "Q2": 0.70, "Q3": -0.65, "Q4": -0.35},
    "CAD": {"Q1": 0.10, "Q2": 0.45, "Q3": -0.10, "Q4": -0.10},
    "EMFX / IDR": {"Q1": 0.30, "Q2": 0.65, "Q3": -0.55, "Q4": -0.30},
}

STAGE_GUIDE = {
    "Q1": {
        "Early": ("quality growth, semis, consumer beta", "utilities / staples", "duration still helps; breadth should widen"),
        "Mid": ("broad equities, software, discretionary", "bond proxies", "earnings upgrades and clean breadth"),
        "Late": ("cyclicals still okay, but froth rises", "defensives still lag", "watch Q2 heat or false-dawn failure"),
    },
    "Q2": {
        "Early": ("small caps, cyclicals, industrials, commodity FX", "duration, staples, utilities", "rates rise orderly and breadth broadens"),
        "Mid": ("financials, materials, broad beta, reflation", "REITs / bond proxies", "credit stable; USD not too brutal"),
        "Late": ("energy, value, nominal-growth trades", "long-duration beta if yields spike", "top risk rises; watch bad-Q2 / Q2→Q3 rollover"),
    },
    "Q3": {
        "Early": ("gold, miners, energy, USD", "small caps, EMFX, weak cyclicals", "inflation re-accelerates before growth fully breaks"),
        "Mid": ("gold, defensives, selective energy, quality", "broad beta, lower-quality cyclicals", "breadth stays narrow and USD/yields matter"),
        "Late": ("gold + duration barbell, quality defensives", "crowded beta, fragile reflation longs", "branch point: good Q3→Q2 or hard slide into Q4"),
    },
    "Q4": {
        "Early": ("duration, defensives, USD", "cyclicals, small caps, weak credit", "growth scare starts dominating"),
        "Mid": ("duration, quality, defensives", "broad beta / lower quality", "bottoming is not confirmed yet"),
        "Late": ("duration, selective gold, early cyclicals only if breadth repairs", "junky beta", "watch Q4→Q1 true bottom vs false dawn"),
    },
}

TRANSITION_LIBRARY = [
    {"Path": "Q3 → Q2", "Variant": "Good reflation", "When": "growth improves, USD calm/softer, yields rise orderly, breadth broadens", "Strong": "small caps, cyclicals, EM, IHSG cyclicals", "Weak": "defensives, duration, pure stagflation hedges"},
    {"Path": "Q3 → Q2", "Variant": "Bad reflation", "When": "oil shock, USD too strong, yields spike, breadth stays narrow", "Strong": "energy, gold, narrow value", "Weak": "EM Asia, small caps, broad beta, weak FX"},
    {"Path": "Q2", "Variant": "Healthy Q2", "When": "rates rise in an orderly way, credit okay, breadth wide", "Strong": "financials, industrials, materials, beta FX", "Weak": "REITs, bond proxies, staples"},
    {"Path": "Q2", "Variant": "Crash-prone Q2", "When": "inflation sticky, policy repricing, liquidity / credit strain", "Strong": "cash, gold, selective quality / energy", "Weak": "small caps, long-duration beta, weak EM"},
    {"Path": "Q2 → Q3", "Variant": "Overheating rollover", "When": "growth breadth rolls while inflation stays hot", "Strong": "gold, defensives, USD", "Weak": "beta cyclicals, small caps, weak FX"},
    {"Path": "Q4 → Q1", "Variant": "True bottoming", "When": "growth stabilises while inflation cools and breadth repairs", "Strong": "quality growth, semis, consumer beta", "Weak": "deep defensives"},
    {"Path": "Q4 → Q1", "Variant": "False dawn", "When": "equities bounce but breadth / credit do not confirm", "Strong": "duration, quality", "Weak": "lower-quality beta"},
    {"Path": "Q4 → Q3", "Variant": "Inflation shock turn", "When": "growth still weak but inflation turns higher again", "Strong": "gold, energy, USD", "Weak": "duration, EM, rate-sensitive beta"},
]


def _bias_label(x: float) -> str:
    if x >= 0.55:
        return "Strong"
    if x >= 0.18:
        return "Above avg"
    if x > -0.18:
        return "Mixed"
    if x > -0.55:
        return "Below avg"
    return "Weak"


def _bias_read(x: float) -> str:
    if x >= 0.55:
        return "Prefer long / overweight"
    if x >= 0.18:
        return "Leaning long"
    if x > -0.18:
        return "Neutral / tactical only"
    if x > -0.55:
        return "Leaning short / underweight"
    return "Prefer short / avoid"


def _transition_variant(core: Dict[str, object]) -> str:
    cur, nxt = core["current_q"], core["next_q"]
    if cur == "Q3" and nxt == "Q2":
        bad = core["stress_infl"] > 0.58 and core["stress_liq"] > 0.45 and core["top_score"] > 0.55
        return "Bad reflation" if bad else "Good reflation"
    if cur == "Q2":
        return "Crash-prone Q2" if (core["top_score"] > 0.56 and core["stress_infl"] > 0.56 and core["fragility"] > 0.42) else "Healthy Q2"
    if cur == "Q4" and nxt == "Q1":
        return "False dawn" if core["agreement"] < 0.55 or core["breadth"] < 0.42 else "True bottoming"
    if cur == "Q2" and nxt == "Q3":
        return "Overheating rollover"
    if cur == "Q4" and nxt == "Q3":
        return "Inflation shock turn"
    return "Base path"


def _scenario_adjust(asset: str, base: float) -> float:
    variant = _transition_variant(core)
    if variant == "Good reflation":
        if asset in ["US small caps", "EM equities", "IHSG cyclicals", "BTC / crypto beta"]:
            base += 0.18
        if asset in ["USD", "Duration / bonds", "Gold / miners"]:
            base -= 0.10
    elif variant == "Bad reflation":
        if asset in ["Oil / energy", "Gold / miners", "USD"]:
            base += 0.18
        if asset in ["US small caps", "EM equities", "IHSG cyclicals", "BTC / crypto beta", "Duration / bonds"]:
            base -= 0.18
    elif variant == "Crash-prone Q2":
        if asset in ["US small caps", "BTC / crypto beta", "EM equities"]:
            base -= 0.22
        if asset in ["Gold / miners", "USD"]:
            base += 0.12
    elif variant == "Overheating rollover":
        if asset in ["Gold / miners", "USD", "US defensives"]:
            base += 0.12
        if asset in ["US cyclicals", "US small caps", "BTC / crypto beta"]:
            base -= 0.18
    return float(max(-1.0, min(1.0, base)))


def _score_book_for_name(name: str) -> Dict[str, float]:
    if name in ASSET_SCORE_BY_QUAD:
        return ASSET_SCORE_BY_QUAD[name]
    if name in CURRENCY_SCORE_BY_QUAD:
        return CURRENCY_SCORE_BY_QUAD[name]
    raise KeyError(f"Unknown asset/currency score key: {name}")


def _current_next_asset_score(asset: str) -> tuple[float, float]:
    score_book = _score_book_for_name(asset)
    cur_q = core.get("current_q", "Q3")
    nxt_q = core.get("next_q", "Q2")
    cur = float(score_book.get(cur_q, 0.0))
    nxt = float(score_book.get(nxt_q, cur))
    if asset in ASSET_SCORE_BY_QUAD:
        nxt = _scenario_adjust(asset, nxt)
    blended = (1 - core["transition_conviction"] * 0.35) * cur + (core["transition_conviction"] * 0.35) * nxt
    return blended, nxt


def build_transition_beneficiary_rows() -> List[List[str]]:
    rows = []
    for asset in ASSET_SCORE_BY_QUAD:
        now, nxt = _current_next_asset_score(asset)
        delta = nxt - now
        if delta > 0.18:
            prep = "Gets better if next wins"
        elif delta < -0.18:
            prep = "Loses edge if next wins"
        else:
            prep = "Not hugely sensitive"
        rows.append([asset, _bias_label(now), _bias_label(nxt), prep, _bias_read(now)])
    order = sorted(rows, key=lambda r: ["Strong","Above avg","Mixed","Below avg","Weak"].index(r[1]))
    return rows


def build_transition_library_rows() -> List[List[str]]:
    base_path = f"{core['current_q']} → {core['next_q']}"
    variant = _transition_variant(core)
    rows = []
    for item in TRANSITION_LIBRARY:
        priority = 0
        if item["Path"] == base_path:
            priority += 3
        if item["Variant"] == variant:
            priority += 2
        if item["Path"].startswith(core["current_q"]):
            priority += 1
        rows.append((priority, [item["Path"], item["Variant"], item["When"], item["Strong"], item["Weak"]]))
    rows.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in rows[:6]]


def build_stage_rows(quads: List[str]) -> Dict[str, List[List[str]]]:
    out: Dict[str, List[List[str]]] = {}
    for q in quads:
        rows = []
        for stage, vals in STAGE_GUIDE[q].items():
            rows.append([stage, vals[0], vals[1], vals[2]])
        out[q] = rows
    return out

def infer_cycle_stage(q: str) -> tuple[str, float]:
    if q == "Q1":
        maturity = 0.45 * core["phase_strength"] + 0.30 * core["top_score"] + 0.25 * core["transition_pressure"]
    elif q == "Q2":
        maturity = 0.42 * core["top_score"] + 0.28 * core["phase_strength"] + 0.20 * core["transition_pressure"] + 0.10 * core["fragility"]
    elif q == "Q3":
        maturity = 0.35 * core["top_score"] + 0.25 * core["phase_strength"] + 0.20 * core["transition_pressure"] + 0.20 * core["fragility"]
    else:
        maturity = 0.40 * core["bottom_score"] + 0.22 * core["transition_pressure"] + 0.18 * core["phase_strength"] + 0.20 * core["fragility"]
    if maturity < 0.36:
        return "Early", float(maturity)
    if maturity < 0.67:
        return "Mid", float(maturity)
    return "Late", float(maturity)


def build_stage_strip_rows(q: str, active_stage: str) -> List[List[str]]:
    rows = []
    for stage, vals in STAGE_GUIDE[q].items():
        marker = "NOW" if stage == active_stage else ""
        rows.append([marker, stage, vals[0], vals[1], vals[2]])
    return rows


def current_stage_sentence() -> str:
    stage, maturity = infer_cycle_stage(core["current_q"])
    return f"{stage} {core['current_q']} · maturity {pct(maturity)} · {STAGE_GUIDE[core['current_q']][stage][2]}"


def next_stage_sentence() -> str:
    return f"If {core['next_q']} takes over, start with Early {core['next_q']} · {STAGE_GUIDE[core['next_q']]['Early'][2]}"


def bucket_bridge_read(bucket_name: str) -> str:
    variant = _transition_variant(core)
    if bucket_name == "US Stocks":
        return f"Now {infer_cycle_stage(core['current_q'])[0]} {core['current_q']}; if breadth widens, {core['next_q']} starts showing in leaders."
    if bucket_name == "Futures / Commodities":
        return "Oil / gold decide whether this is cleaner reflation or a more toxic inflation turn."
    if bucket_name == "Forex":
        return "Use full FX rank below: strong leg vs weak leg, not one pair only."
    if bucket_name == "Crypto":
        return "Crypto needs liquidity and breadth; in Q3 keep it tactical unless next-phase evidence improves."
    if bucket_name == "IHSG":
        return "IHSG improves when USD pressure cools and EM / commodity breadth confirms."
    return variant_explainer(variant)


def build_merged_playbook_rows() -> List[List[str]]:
    rows = []
    for bucket_name in ["US Stocks", "Futures / Commodities", "Forex", "Crypto", "IHSG"]:
        rows.append([
            bucket_name,
            ", ".join(play_cur[bucket_name]) or "-",
            ", ".join(play_next[bucket_name]) or "-",
            bucket_bridge_read(bucket_name),
        ])
    return rows


def build_cross_asset_stage_header_rows() -> List[List[str]]:
    cur_stage, _ = infer_cycle_stage(core["current_q"])
    return [
        ["Now", f"{cur_stage} {core['current_q']}", STAGE_GUIDE[core['current_q']][cur_stage][0], STAGE_GUIDE[core['current_q']][cur_stage][1]],
        ["If next wins", f"Early {core['next_q']}", STAGE_GUIDE[core['next_q']]['Early'][0], STAGE_GUIDE[core['next_q']]['Early'][1]],
    ]


def build_cross_asset_focus_rows() -> List[List[str]]:
    rows = []
    stage_now, _ = infer_cycle_stage(core["current_q"])
    for asset, now, nxt, read in asset_rank_rows:
        action = read
        if now in ["Strong", "Above avg"] and nxt in ["Strong", "Above avg"]:
            prep = "Still fine if next wins"
        elif now in ["Strong", "Above avg"] and nxt in ["Mixed", "Below avg", "Weak"]:
            prep = "Good now, fades if next wins"
        elif now in ["Mixed", "Below avg", "Weak"] and nxt in ["Strong", "Above avg"]:
            prep = "Watchlist if next wins"
        else:
            prep = "Still secondary"
        rows.append([asset, now, action, prep])
    order = {"Strong": 0, "Above avg": 1, "Mixed": 2, "Below avg": 3, "Weak": 4}
    rows.sort(key=lambda r: (order.get(r[1], 99), r[0]))
    return rows


def build_fx_rank_text(score_rows: List[Tuple[str, float]]) -> str:
    if not score_rows:
        return "No FX data"
    return " > ".join([x[0] for x in score_rows])


def build_fx_display_rows(score_rows: List[Tuple[str, float]]) -> List[List[str]]:
    rows = []
    for ccy, score in score_rows:
        if score >= 0.45:
            use = "Best long leg"
        elif score >= 0.15:
            use = "Lean long"
        elif score <= -0.45:
            use = "Best short / funding leg"
        elif score <= -0.15:
            use = "Lean short"
        else:
            use = "Neutral"
        rows.append([ccy, _bias_label(score), use])
    return rows


def build_forex_board() -> tuple[List[List[str]], str, str, List[Tuple[str, float]]]:
    variant = _transition_variant(core)
    rows = []
    score_rows = []
    for ccy, mp in CURRENCY_SCORE_BY_QUAD.items():
        cur = mp[core["current_q"]]
        nxt = mp[core["next_q"]]
        if variant == "Good reflation":
            if ccy in ["AUD", "NZD", "CAD", "EMFX / IDR"]:
                nxt += 0.18
            if ccy in ["USD", "JPY", "CHF"]:
                nxt -= 0.12
        elif variant in ["Bad reflation", "Crash-prone Q2", "Overheating rollover"]:
            if ccy in ["USD", "JPY", "CHF"]:
                nxt += 0.15
            if ccy in ["AUD", "NZD", "EMFX / IDR"]:
                nxt -= 0.18
        score = (1 - core["transition_conviction"] * 0.35) * cur + (core["transition_conviction"] * 0.35) * nxt
        score = float(max(-1.0, min(1.0, score)))
        if score >= 0.45:
            expr = "Long vs weakest FX"
        elif score >= 0.15:
            expr = "Lean long vs weak FX"
        elif score <= -0.45:
            expr = "Better as funding / short leg"
        elif score <= -0.15:
            expr = "Lean short vs strong FX"
        else:
            expr = "Neutral / pair-selective"
        rows.append([ccy, _bias_label(score), expr, f"{core['current_q']} now / {core['next_q']} if transition extends"])
        score_rows.append((ccy, score))
    score_rows.sort(key=lambda x: x[1], reverse=True)
    strong = [x[0] for x in score_rows[:3]]
    weak = [x[0] for x in score_rows[-3:]]
    return rows, " > ".join(strong), " > ".join(weak[::-1]), score_rows


def build_cross_asset_rank_rows() -> List[List[str]]:
    rows = []
    for asset in ASSET_SCORE_BY_QUAD:
        now, nxt = _current_next_asset_score(asset)
        rows.append([asset, _bias_label(now), _bias_label(nxt), _bias_read(now)])
    rows.sort(key=lambda r: {"Strong":0,"Above avg":1,"Mixed":2,"Below avg":3,"Weak":4}[r[1]])
    return rows




def _dt_utc(y: int, m: int, d: int, hh: int, mm: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


def _first_business_day(y: int, m: int) -> date:
    d = date(y, m, 1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _nth_business_day(y: int, m: int, n: int) -> date:
    d = date(y, m, 1)
    count = 0
    while True:
        if d.weekday() < 5:
            count += 1
            if count == n:
                return d
        d += timedelta(days=1)


def _human_countdown(target: datetime, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    delta = target - now
    secs = int(delta.total_seconds())
    if secs <= 0:
        return 'Released / live'
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    mins, _ = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours}h"
    if hours > 0:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def build_macro_catalyst_rows(limit: int = 8) -> List[List[str]]:
    now = datetime.now(timezone.utc)
    events: List[Tuple[datetime, str, str, str]] = [
        (_dt_utc(2026, 4, 1, 12, 30), 'Retail Sales (Feb)', 'Growth / consumer breadth', 'Q now / next Q'),
        (_dt_utc(2026, 4, 3, 12, 30), 'Payrolls (Mar)', 'Growth / labor trend', 'Q now'),
        (_dt_utc(2026, 4, 9, 12, 30), 'PCE / Personal Outlays (Feb)', 'Inflation + demand', 'Q now / next Q'),
        (_dt_utc(2026, 4, 10, 12, 30), 'CPI (Mar)', 'Inflation / policy repricing', 'Q now'),
        (_dt_utc(2026, 4, 14, 12, 30), 'PPI (Mar)', 'Pipeline inflation', 'Next Q / good-vs-bad reflation'),
        (_dt_utc(2026, 4, 21, 12, 30), 'Retail Sales (Mar)', 'Growth breadth / cyclicals', 'Next Q'),
        (_dt_utc(2026, 4, 29, 18, 0), 'FOMC statement', 'Policy / yields / USD', 'Q now + next Q'),
        (_dt_utc(2026, 4, 30, 12, 30), 'PCE / Personal Outlays (Mar)', 'Inflation + demand', 'Q now / next Q'),
        (_dt_utc(2026, 5, 8, 12, 30), 'Payrolls (Apr)', 'Growth / labor trend', 'Q now'),
        (_dt_utc(2026, 5, 12, 12, 30), 'CPI (Apr)', 'Inflation / policy repricing', 'Q now'),
        (_dt_utc(2026, 5, 13, 12, 30), 'PPI (Apr)', 'Pipeline inflation', 'Next Q'),
        (_dt_utc(2026, 5, 14, 12, 30), 'Retail Sales (Apr)', 'Growth breadth / cyclicals', 'Next Q'),
        (_dt_utc(2026, 5, 20, 18, 0), 'FOMC minutes', 'Policy follow-through', 'Secondary'),
        (_dt_utc(2026, 6, 17, 18, 0), 'FOMC statement + SEP', 'Policy / USD / yields', 'Q now + next Q'),
    ]
    months = [(2026, 4), (2026, 5), (2026, 6)]
    for y, m in months:
        mfg = _nth_business_day(y, m, 1)
        svc = _nth_business_day(y, m, 3)
        events.append((datetime(mfg.year, mfg.month, mfg.day, 14, 0, tzinfo=timezone.utc), f'ISM Manufacturing ({mfg.strftime("%b")})', 'Growth / goods cycle', 'Next Q'))
        events.append((datetime(svc.year, svc.month, svc.day, 14, 0, tzinfo=timezone.utc), f'ISM Services ({svc.strftime("%b")})', 'Growth / services breadth', 'Next Q'))
    events = sorted(events, key=lambda x: x[0])
    future = [e for e in events if e[0] >= now][:limit]
    rows: List[List[str]] = []
    for dt, name, why, impact in future:
        rows.append([name, dt.strftime('%d %b %H:%M UTC'), _human_countdown(dt, now), why, impact])
    return rows


def macro_catalyst_summary(limit: int = 3) -> str:
    rows = build_macro_catalyst_rows(limit=limit)
    if not rows:
        return 'No upcoming catalyst loaded'
    return ' · '.join([f"{r[0]} in {r[2]}" for r in rows[:limit]])


def _bucket_priority_score(bucket_name: str) -> float:
    proxy_map = {
        'Futures / Commodities': ['Gold / miners', 'Oil / energy', 'Duration / bonds'],
        'Cross-Asset / FX': ['USD', 'EMFX / IDR'],
        'US Stocks': ['US defensives', 'US cyclicals', 'US small caps'],
        'IHSG': ['IHSG cyclicals'],
        'Crypto': ['BTC / crypto beta'],
    }
    vals = []
    for asset in proxy_map.get(bucket_name, []):
        now_score, _ = _current_next_asset_score(asset)
        vals.append(abs(now_score))
    if not vals:
        return 0.0
    return float(max(vals))


def build_playbook_priority_rows() -> List[List[str]]:
    display_order = [
        ('Futures / Commodities', 'Futures / Commodities'),
        ('Cross-Asset / FX', 'Forex'),
        ('US Stocks', 'US Stocks'),
        ('IHSG', 'IHSG'),
        ('Crypto', 'Crypto'),
    ]
    rows = []
    for display_name, play_key in display_order:
        use_now = ', '.join(play_cur.get(play_key, [])) or '-'
        watch_next = ', '.join(play_next.get(play_key, [])) or '-'
        rows.append((
            _bucket_priority_score(display_name),
            display_name,
            use_now,
            watch_next,
            bucket_bridge_read(play_key),
        ))
    rows.sort(key=lambda x: x[0], reverse=True)
    labels = ['Highest', 'High', 'Medium', 'Lower', 'Low']
    out = []
    for i, (_, bucket_name, use_now, watch_next, confirm) in enumerate(rows):
        out.append([labels[i] if i < len(labels) else 'Low', bucket_name, use_now, watch_next, confirm])
    return out


def build_playbook_summary_rows() -> List[List[str]]:
    cur_stage, _ = infer_cycle_stage(core['current_q'])
    return [
        ['Now', f"{cur_stage} {core['current_q']}", STAGE_GUIDE[core['current_q']][cur_stage][2]],
        ['If transition extends', f"Early {core['next_q']}", STAGE_GUIDE[core['next_q']]['Early'][2]],
        ['Variant now', _transition_variant(core), variant_explainer(_transition_variant(core))],
        ['Next macro catalysts', macro_catalyst_summary(3), 'Use these as timing checkpoints for Q / next-Q confirmation'],
    ]

def build_decision_key_rows() -> List[List[str]]:
    cur_stage, _ = infer_cycle_stage(core['current_q'])
    top_state_now = ladder_state(core['top_score'], 'top')
    bottom_state_now = ladder_state(core['bottom_score'], 'bottom')
    return [
        ['Now', f"{cur_stage} {core['current_q']}", STAGE_GUIDE[core['current_q']][cur_stage][2]],
        ['If transition extends', f"Early {core['next_q']}", STAGE_GUIDE[core['next_q']]['Early'][2]],
        ['Variant now', _transition_variant(core), variant_explainer(_transition_variant(core))],
        ['Top / bottom state', f"{top_state_now} / {bottom_state_now}", 'Top tinggi = jangan kejar atas. Bottom rendah = belum ada base / washout yang bersih.'],
        ['Next macro catalysts', macro_catalyst_summary(3), 'Pakai ini sebagai timing checkpoint untuk konfirmasi Q sekarang atau next-Q.'],
    ]


def build_simple_relative_rows() -> List[List[str]]:
    rows = []
    for row in relative_rows:
        now_fit, next_fit, prep = rel_phase_context(row['Lens'])
        simple = interpret_relative(row['Direction'], row['State'], row['Quality'])
        rows.append([row['Lens'], f"{row['Direction']} / {row['Strength']}", simple, prep])
    return rows


def build_simple_size_rows() -> List[List[str]]:
    rows = []
    for row in size_rows:
        _, _, prep = rel_phase_context(row['Lens'])
        if row['Direction'] == 'Stronger':
            read = 'Participation helps confirm the move.' if row['State'] in ['Stable', 'Building'] else 'Strong but getting stretched.'
        elif row['Direction'] == 'Balanced':
            read = 'No clear participation edge yet.'
        else:
            read = 'Participation is weak / not confirming yet.'
        rows.append([row['Lens'], f"{row['Direction']} / {row['Strength']}", read, prep])
    return rows


def build_asset_scenario_rows() -> List[List[str]]:
    return [
        ['Gold / miners', 'Bullish if USD and 10Y stop rising, oil stabilizes, and fear shifts from inflation to growth.', 'Still weak if oil shock keeps USD / yields climbing and Fed pricing stays hawkish.'],
        ['EM / IHSG', 'Better if USD calms, breadth broadens, and commodities confirm cleanly.', 'Avoid if USD stays strong, yields stay hard, and breadth stays narrow.'],
        ['US small caps', 'Need orderly rates plus broader participation to confirm early-Q2 style reflation.', 'Stay weak if rates rise for bad reasons or liquidity / credit tighten.'],
        ['Duration / bonds', 'Better if growth scare outruns inflation fear.', 'Still weak if inflation shock dominates and nominal yields keep pushing higher.'],
        ['Crypto beta', 'Needs liquidity + breadth, not just narrative.', 'Avoid if USD is strong and funding / liquidity get tighter.'],
    ]


def build_cross_asset_stage_table() -> List[List[str]]:
    cur_stage, _ = infer_cycle_stage(core['current_q'])
    rows = []
    for stage, vals in STAGE_GUIDE[core['current_q']].items():
        marker = 'NOW' if stage == cur_stage else ''
        rows.append([marker, stage, vals[0], vals[1], vals[2]])
    return rows


def build_fx_expressions_table(score_rows: List[Tuple[str, float]]) -> List[List[str]]:
    best_pairs = build_best_fx_pairs(score_rows)
    rows = []
    for pair, edge, read in best_pairs[:3]:
        rows.append([pair, edge, read])
    return rows or [['No pair', '-', 'No read']]

def explain_turn_risks() -> tuple[str, str]:
    top_txt = f"Top risk = odds the market is late in the up-leg / stretched enough that upside gets thinner and pullback risk rises. State now: {ladder_state(core['top_score'], 'top')}. This is not an exact price top call."
    bottom_txt = f"Bottom risk = odds that washout / capitulation / base-building conditions are developing. State now: {ladder_state(core['bottom_score'], 'bottom')}. This is not an exact price bottom call."
    return top_txt, bottom_txt


def current_crash_probability() -> float:
    base = {'Q1': 0.22, 'Q2': 0.42, 'Q3': 0.61, 'Q4': 0.74}.get(core['current_q'], 0.50)
    variant = _transition_variant(core)
    adj = 0.0
    if variant in ['Bad reflation', 'Crash-prone Q2', 'Overheating rollover', 'Inflation shock turn']:
        adj += 0.08
    elif variant in ['Good reflation', 'Healthy Q2', 'True bottoming']:
        adj -= 0.06
    elif variant in ['False dawn']:
        adj += 0.04
    stress_mix = (
        0.24 * core['stress_liq'] +
        0.20 * core['stress_infl'] +
        0.16 * (1 - core['breadth']) +
        0.16 * core['top_score'] +
        0.12 * core['transition_pressure'] +
        0.12 * core['fragility']
    )
    score = base + (stress_mix - 0.50) * 0.55 + adj
    return clamp01(score)


def crash_meter_label(score: float) -> str:
    if score >= 0.80:
        return 'Very high'
    if score >= 0.65:
        return 'High'
    if score >= 0.45:
        return 'Elevated'
    if score >= 0.25:
        return 'Watch'
    return 'Low'


def leaders_status_text() -> str:
    us_valid = 0 if us_leaders_df is None or us_leaders_df.empty else int(us_leaders_df['Ticker'].nunique())
    ih_valid = 0 if ihsg_leaders_df is None or ihsg_leaders_df.empty else int(ihsg_leaders_df['Ticker'].nunique())
    if us_valid == 0 and ih_valid == 0:
        return 'Hidden for now (coverage valid still zero)'
    return f'Usable ({us_valid} US / {ih_valid} IHSG valid names)'


def crash_watch_window() -> str:
    crash_now = current_crash_probability()
    rows = build_macro_catalyst_rows(limit=3)
    if not rows:
        return 'No catalyst loaded'
    names = ' · '.join([f"{r[0]} in {r[2]}" for r in rows[:2]])
    if crash_now >= 0.65:
        prefix = 'Elevated into next catalyst window'
    elif crash_now >= 0.45:
        prefix = 'Needs respect into next catalyst window'
    else:
        prefix = 'Use next catalyst as confirmation window'
    return f'{prefix}: {names}'


def risk_relative_summary() -> str:
    crash_now = current_crash_probability()
    cur_stage, _ = infer_cycle_stage(core['current_q'])
    variant = _transition_variant(core)
    rel_read = relative_rows[0]['Read'] if relative_rows else 'No relative read'
    return (
        f"Base case masih {cur_stage} {core['current_q']} dengan varian {variant}. "
        f"Crash meter sekarang {pct(crash_now)} ({crash_meter_label(crash_now)}), jadi fokus utamanya bukan nebak exact crash date, tapi hormati window risiko di sekitar catalyst berikutnya. "
        f"Relative read masih {rel_read.lower()}, jadi pakai bias yang paling sinkron dulu dan jangan maksa broad-beta call kalau breadth belum ikut confirm."
    )


def build_unified_relative_rows() -> List[List[str]]:
    rows: List[List[str]] = []
    for row in build_simple_relative_rows():
        rows.append(['Relative', row[0], row[1], row[2], row[3]])
    for row in build_simple_size_rows():
        rows.append(['Participation', row[0], row[1], row[2], row[3]])
    return rows


def build_crash_compact_rows(crash_now: float) -> List[List[str]]:
    return [
        ['Crash meter now', pct(crash_now), 'Higher = accident / drawdown risk rises; use next catalyst window as timing check, not exact date.'],
        ['Top state', pct(core['top_score']), 'Higher = upside gets thinner; avoid chasing stretched moves.'],
        ['Bottom state', pct(core['bottom_score']), 'Higher = washout / base-building has more chance to form.'],
        ['Liquidity stress', pct(core['stress_liq']), 'If this rises fast, crash branch gets more dangerous.'],
        ['Breadth failure', pct(1 - core['breadth']), 'If indices hold but participation narrows, accident risk rises.'],
        ['Transition pressure', pct(core['transition_pressure']), 'Higher = current regime more likely to fracture into next / worse branch.'],
    ]


def build_crash_timing_rows(crash_now: float) -> List[List[str]]:
    return [
        ['Crash meter now', f"{pct(crash_now)} ({crash_meter_label(crash_now)})", 'Respect risk window; not exact crash timing.'],
        ['Top state', ladder_state(core['top_score'], 'top'), 'If extended, do not chase stretched upside.'],
        ['Bottom state', ladder_state(core['bottom_score'], 'bottom'), 'If low, do not force bottom-fishing yet.'],
        ['Next catalyst window', macro_catalyst_summary(2), 'Nearest macro events most likely to move Q / next-Q.'],
        ['Base crash branch', _transition_variant(core), 'Dirty / toxic variants raise accident probability.'],
    ]


def build_relative_compact_rows() -> List[List[str]]:
    rows = []
    for group, lens, bias, read, nxt in build_unified_relative_rows():
        rows.append([f"{group}: {lens}", bias, read, nxt])
    return rows


def build_quad_scenario_matrix() -> List[List[str]]:
    rows = [
        ['Q1', 'Clean growth-disinflation', 'US equities, quality growth, credit, improving small caps', 'Rates spike / valuation accident', pct(0.22)],
        ['Q2', 'Good reflation', 'Cyclicals, industrials, financials, oil, small caps if rates orderly', 'Dirty / overheating Q2', pct(0.42)],
        ['Q3', 'Shock-stagflation', 'Oil / energy, selective gold, selective defensives', 'USD + yields + growth shock branch', pct(0.61)],
        ['Q4', 'Slowdown / disinflation', 'Bonds, USD, gold, defensives, quality', 'Funding / liquidity crash branch', pct(0.74)],
    ]
    out = []
    for quad, base, best, crash_branch, risk in rows:
        marker = 'NOW' if quad == core['current_q'] else ('NEXT' if quad == core['next_q'] else '')
        out.append([marker, quad, base, best, crash_branch, risk])
    return out


def _asset_now_score(asset: str) -> float:
    cur, _ = _current_next_asset_score(asset)
    return cur

def scenario_state_now(name: str) -> str:
    variant = _transition_variant(core)
    q = core['current_q']
    stage, _ = infer_cycle_stage(q)
    if name == 'Gold in Q3':
        if q == 'Q3' and variant == 'Bad reflation':
            return 'Backdrop bullish, tape still pressured'
        if q == 'Q3' and _asset_now_score('Gold / miners') > 0.45:
            return 'Constructive / selective long'
        return 'Tactical only'
    if name == 'EM / IHSG':
        if _asset_now_score('EM equities') < -0.18 or _asset_now_score('IHSG cyclicals') < -0.05:
            return 'Still pressured / not clean'
        if q == 'Q2' or (q == 'Q3' and variant == 'Good reflation'):
            return 'Improving if USD calms'
        return 'Mixed / selective'
    if name == 'US small caps':
        if q == 'Q2' and stage in ['Early','Mid'] and _asset_now_score('US small caps') > 0.25:
            return 'Cleanly usable'
        if core['next_q'] == 'Q2':
            return 'Watch for confirmation'
        return 'Not confirmed yet'
    if name == 'Bonds / duration':
        if _asset_now_score('Duration / bonds') > 0.35:
            return 'Constructive / hedge-friendly'
        if q == 'Q3':
            return 'Mixed / tactical only'
        return 'Needs better disinflation signal'
    if name == 'Crypto beta':
        if _asset_now_score('BTC / crypto beta') > 0.35:
            return 'Strong but watch stretch'
        if q == 'Q3':
            return 'Selective only / BTC bias'
        return 'Tactical only'
    return 'Contextual'

def build_current_scenario_checks() -> List[List[str]]:
    rows = [
        ('Gold in Q3', 'Use selectively; better on pullback or when USD / yields stop acting as headwind.', 'Need DXY / yields to stop squeezing and fear to rotate from inflation to growth.', 'Invalid if oil shock keeps USD / yields climbing and Fed pricing stays hawkish.'),
        ('EM / IHSG', 'Use only if USD calms and local / commodity breadth confirms.', 'Need softer USD, better breadth, and cleaner commodity confirmation.', 'Invalid if USD stays strong, yields stay hard, and flows keep leaking out.'),
        ('US small caps', 'Treat as confirmation asset, not blind long.', 'Need orderly rates plus broader participation to confirm healthy reflation.', 'Invalid if rates rise for bad reasons or liquidity tightens.'),
        ('Bonds / duration', 'Mostly tactical / hedge until growth fear clearly outruns inflation fear.', 'Need nominal yields to stop rising and growth scare to dominate.', 'Invalid if inflation shock keeps nominal yields pushing higher.'),
        ('Crypto beta', 'Prefer only when liquidity + breadth improve; otherwise stay selective.', 'Need liquidity, breadth, and weaker USD — not just narrative.', 'Invalid if USD is strong and funding / liquidity tighten.'),
    ]
    out = []
    for scen, use_now, improve, invalid in rows:
        out.append([scen, scenario_state_now(scen), use_now, improve, invalid])
    return out


def hero_simple_help() -> Dict[str, str]:
    return {
        "top": "Top risk tinggi = jangan kejar atas. Lebih cocok tunggu pullback / konfirmasi ulang.",
        "bottom": "Bottom risk rendah = belum ada tanda washout / bottom yang bersih.",
        "relative": "Relative lemah = edge antar aset belum bersih. Jangan maksa opini besar.",
        "shocks": "Shocks = skenario yang bisa ngerusak base case, bukan arah utama sendirian.",
    }


def variant_explainer(variant: str) -> str:
    mapping = {
        "Good reflation": "Growth membaik, breadth melebar, USD/yields tidak terlalu brutal. Siklikal, small caps, EM biasanya ikut confirm.",
        "Bad reflation": "Nominal growth kelihatan naik, tapi pendorongnya lebih beracun: oil shock, USD kuat, yields keras, breadth sempit.",
        "Healthy Q2": "Q2 yang sehat: rates naik tertib, breadth luas, kredit oke. Siklikal dan small caps biasanya bisa ikut jalan.",
        "Crash-prone Q2": "Q2 yang mulai bahaya: inflation/policy repricing, likuiditas mengetat, breadth rusak, market gampang kecelakaan.",
        "Overheating rollover": "Q2 mulai kepanjangan lalu rollover ke Q3. Defensives / USD / gold mulai lebih penting daripada beta.",
        "True bottoming": "Q4 ke Q1 yang sehat: growth membaik, breadth melebar, risk appetite pulih secara bersih.",
        "False dawn": "Kelihatan membaik, tapi konfirmasinya tipis. Risk-on bisa cuma relief rally, belum cycle turn yang sehat.",
        "Inflation shock turn": "Growth belum pulih, tapi inflasi naik lagi. Ini buruk buat aset sensitif duration dan broad beta.",
        "Base path": "Belum ada varian spesifik yang dominan. Pakai playbook dasar sambil tunggu konfirmasi tambahan.",
    }
    return mapping.get(variant, "Base path masih dominan. Tunggu konfirmasi lebih lanjut.")


def compact_regime_rows() -> List[List[str]]:
    base_path = f"{core['current_q']} → {core['next_q']}"
    variant = _transition_variant(core)
    rows = []
    for item in TRANSITION_LIBRARY:
        if item["Path"] == base_path or item["Path"].startswith(core["current_q"]):
            rows.append([item["Path"], item["Variant"], item["When"]])
    seen = set()
    out = []
    for r in rows:
        key = tuple(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    out.sort(key=lambda r: (r[0] != base_path, r[1] != variant))
    return out[:5]


def build_best_fx_pairs(score_rows: List[Tuple[str, float]]) -> List[List[str]]:
    if not score_rows:
        return [["No pair", "No read", "No data"]]
    ordered = [x for x in score_rows if x[0] not in ["EUR", "GBP"]]
    strong = ordered[:3]
    weak = list(reversed(ordered[-3:]))
    pairs = []
    used = set()
    for s_name, s_score in strong:
        for w_name, w_score in weak:
            if s_name == w_name:
                continue
            pair = f"Long {s_name} / Short {w_name}"
            if pair in used:
                continue
            used.add(pair)
            edge = s_score - w_score
            if edge >= 0.7:
                read = "Best expression now"
            elif edge >= 0.45:
                read = "Good expression"
            else:
                read = "Okay / tactical"
            pairs.append([pair, pct(max(0.0, min(1.0, (edge + 1.0) / 2.0))), read])
            if len(pairs) >= 4:
                return pairs
    return pairs[:4] if pairs else [["No pair", "No read", "No data"]]


def build_cross_asset_compact_rows() -> List[List[str]]:
    rows = []
    for asset, now, nxt, read in asset_rank_rows:
        if now in ["Strong", "Above avg"]:
            bias = "Prefer long / overweight"
        elif now == "Mixed":
            bias = "Neutral / tactical"
        else:
            bias = "Lean short / underweight"
        rows.append([asset, now, bias])
    return rows[:8]


def build_stage_handoff_rows() -> List[List[str]]:
    current_stage = STAGE_GUIDE[core['current_q']]
    next_stage = STAGE_GUIDE[core['next_q']]
    return [
        ["Now", core['current_q'], ', '.join(play_cur['US Stocks'][:2]), current_stage['Early'][2]],
        ["If next wins", core['next_q'], ', '.join(play_next['US Stocks'][:2]), next_stage['Early'][2]],
        ["Risk check", _transition_variant(core), "Do not chase if top risk high", variant_explainer(_transition_variant(core))],
    ]


def build_q2_crash_watch_rows() -> List[List[str]]:
    rows = [
        ["Inflation / policy repricing", pct(max(core['stress_infl'], core['top_score'])), "danger if inflation stays sticky and rates reprice harder"],
        ["Liquidity / credit strain", pct(core['stress_liq']), "danger if funding stress tightens fast"],
        ["Breadth failure", pct(1 - core['breadth']), "danger if index holds up but participation narrows"],
        ["Late-cycle extension", pct(core['top_score']), "danger if Q2 keeps running but gets stretched / crowded"],
        ["Transition accident", pct(core['transition_pressure']), "danger if Q2 morphs into Q3/Q4 instead of staying healthy"],
    ]
    return rows


def rel_phase_context(lens: str) -> tuple[str, str, str]:
    map_now = {
        "US/EM": {"Q1":"EM-friendly if USD soft", "Q2":"EM-friendly if USD calm", "Q3":"US over EM", "Q4":"US / defensives over EM"},
        "IHSG/US": {"Q1":"IHSG okay if global improves", "Q2":"IHSG can catch up", "Q3":"US safer", "Q4":"US safer / IHSG mixed"},
        "IHSG/EM": {"Q1":"IHSG mixed", "Q2":"IHSG can lead via commodities", "Q3":"Mixed / conditional", "Q4":"Mixed"},
        "Crypto/Liq": {"Q1":"Crypto-friendly", "Q2":"Crypto-friendly if liquidity okay", "Q3":"Need selectivity / BTC bias", "Q4":"Liquidity matters more than beta"},
        "US Small/Big": {"Q1":"Small improving", "Q2":"Small should lead", "Q3":"Big / quality over small", "Q4":"Big over small"},
        "US Small/Broad": {"Q1":"Small improving", "Q2":"Small should confirm", "Q3":"Broad / big safer", "Q4":"Broad safer"},
        "Alt Basket/BTC": {"Q1":"Alts can lead", "Q2":"Alts can broaden", "Q3":"BTC over alts", "Q4":"BTC / cash-like preference"},
    }
    now_fit = map_now.get(lens, {}).get(core['current_q'], 'Contextual')
    next_fit = map_now.get(lens, {}).get(core['next_q'], 'Contextual')
    if now_fit == next_fit:
        prep = 'No big role change if next wins'
    else:
        prep = f'If {core["next_q"]} wins: {next_fit}'
    return now_fit, next_fit, prep


def leadership_health_rows(df: pd.DataFrame, label: str) -> List[List[str]]:
    if df is None or df.empty:
        return [[label, '0 valid names', 'No benchmark-relative output', 'Use theme proxy / smaller preset']]
    bench = ', '.join(df['Benchmark'].astype(str).value_counts().head(2).index.tolist())
    states = ', '.join([f"{k}:{v}" for k, v in df['State'].value_counts().to_dict().items()])
    return [[label, f"{len(df)} valid names", bench, states]]


@st.cache_data(ttl=60*60*4, show_spinner=False)
def proxy_theme_strength_rows(mapping: Dict[str, set], benchmark_ticker: str, period: str = '6mo', fallback_benchmark: str | None = None, n: int = 6) -> List[List[str]]:
    bench = yahoo_close(benchmark_ticker, period)
    if bench.empty and fallback_benchmark:
        bench = yahoo_close(fallback_benchmark, period)
    if bench.empty:
        return [["No proxy data", '-', '-', '-']]
    rows = []
    for theme, tickers in mapping.items():
        valid = []
        alpha_names = []
        for t in list(tickers)[:8]:
            s = yahoo_close(t, period)
            if s.empty or len(s.dropna()) < 25:
                continue
            valid.append((t, s.dropna()))
        if len(valid) < 2:
            continue
        normed = []
        for t, s in valid:
            normed.append((s / s.iloc[0]).rename(t))
        basket = pd.concat(normed, axis=1).dropna(how='all').mean(axis=1).dropna()
        df = pd.concat([basket.rename('asset'), bench.rename('bench')], axis=1).sort_index().ffill().dropna()
        if len(df) < 25:
            continue
        a21 = ret_n(df['asset'], 21) - ret_n(df['bench'], 21)
        a63 = ret_n(df['asset'], 63) - ret_n(df['bench'], 63)
        rs = clamp01(0.60 * _norm_tanh(a21, 0.08) + 0.40 * _norm_tanh(a63, 0.15))
        start = clamp01(_norm_tanh(a21 - 0.45 * a63, 0.08))
        reps = []
        for t, s in valid:
            d = pd.concat([s.rename('asset'), bench.rename('bench')], axis=1).sort_index().ffill().dropna()
            if len(d) < 25:
                continue
            reps.append((t.replace('.JK',''), ret_n(d['asset'], 21) - ret_n(d['bench'], 21)))
        reps = ', '.join([x[0] for x in sorted(reps, key=lambda z: z[1], reverse=True)[:3]])
        rows.append([theme, pct(rs), pct(start), reps or 'proxy basket'])
    if not rows:
        return [["No proxy data", '-', '-', '-']]
    rows.sort(key=lambda r: float(r[1].strip('%')) if r[1] not in ['-'] else -1, reverse=True)
    return rows[:n]


top_risk_text, bottom_risk_text = explain_turn_risks()
transition_rows = build_transition_library_rows()
beneficiary_rows = build_transition_beneficiary_rows()
fx_rows, fx_strong_txt, fx_weak_txt, fx_score_rows = build_forex_board()
asset_rank_rows = build_cross_asset_rank_rows()
stage_rows = build_stage_rows([core['current_q'], core['next_q']] if core['next_q'] != core['current_q'] else [core['current_q']])
q2_crash_rows = build_q2_crash_watch_rows()

# --------------------
# EVENTS
# --------------------
today = date.today()
event_rows = [["Macro release timing", "dynamic", "Use actual calendar; avoid fake offsets"], ["Released-only cutoff", core["official_date"], "common macro date used in official state"], ["Snapshot archive", f"{SNAPSHOT_COUNT} days", f"last snapshot {SNAPSHOT_LAST}"]]

# --------------------

# RENDER
st.title("QuantFinalV4_Max")
st.markdown("<div class='small-muted'>Core alpha engine: Hedgeye_LiveQuad_Core_v2_5 • Q3-anchored decision support shell • Live backbone: FRED + optional Yahoo</div>", unsafe_allow_html=True)
st.write("")

# ---- attachment-2 style summary ----
cur_stage, _ = infer_cycle_stage(core['current_q'])
variant_now = _transition_variant(core)
crash_now = current_crash_probability()
RISK_BUDGET_ROWS = recommended_risk_budget(core, crash_now) if recommended_risk_budget is not None else []
RISK_CONTROL_ROWS = risk_controls_rows(core, crash_now) if risk_controls_rows is not None else []
TACTICAL_SIZING_ROWS = tactical_sizing_rows(core, crash_now) if tactical_sizing_rows is not None else []
SCENARIO_OVERRIDE_ROWS = scenario_override_rows(core, crash_now) if scenario_override_rows is not None else []

_snapshot_meta = {
    "asof": str(ASOF_TS.date()),
    "app_name": APP_NAME,
    "core_name": CORE_NAME,
    "current_q": core.get("current_q"),
    "next_q": core.get("next_q"),
    "official_date": core.get("official_date"),
    "confidence": float(core.get("confidence", 0.0)),
    "agreement": float(core.get("agreement", 0.0)),
    "fragility": float(core.get("fragility", 0.0)),
    "crash_now": float(crash_now),
    "top_score": float(core.get("top_score", 0.0)),
    "bottom_score": float(core.get("bottom_score", 0.0)),
    "variant_now": str(_transition_variant(core)),
    "sub_phase": str(core.get("sub_phase", "")),
}
for _k, _ticker in {
    "spy_close": "SPY",
    "iwm_close": "IWM",
    "gld_close": "GLD",
    "tlt_close": "TLT",
    "xle_close": "XLE",
    "btc_close": "BTC-USD",
    "eido_close": "EIDO",
}.items():
    try:
        _s = yahoo_close(_ticker, "6mo")
        if _s is not None and len(_s):
            _snapshot_meta[_k] = float(pd.Series(_s).dropna().iloc[-1])
    except Exception:
        pass
SNAPSHOT_PATH = persist_snapshot(SER_RAW, SER, SNAPSHOT_DIR, _snapshot_meta) if persist_snapshot is not None else None
SNAPSHOT_COUNT, SNAPSHOT_LAST = snapshot_coverage(SNAPSHOT_DIR) if snapshot_coverage is not None else (0, "n/a")
VALIDATION_READY = readiness_score(
    snapshot_count=SNAPSHOT_COUNT,
    has_release_lag=adjust_series_dict is not None,
    has_data_health=bool(DATA_HEALTH_ROWS),
    has_risk_budget=bool(RISK_BUDGET_ROWS),
) if readiness_score is not None else None
WFO_DF = run_snapshot_walkforward(SNAPSHOT_DIR, horizon_days=21) if run_snapshot_walkforward is not None else pd.DataFrame()
WFO_SUMMARY_ROWS = walkforward_summary_rows(WFO_DF) if walkforward_summary_rows is not None else []
WFO_CAL_ROWS = calibration_rows(WFO_DF, prob_col="confidence", hit_col="is_correct") if calibration_rows is not None else []
WFO_CRASH_CAL_ROWS = crash_calibration_rows(WFO_DF) if crash_calibration_rows is not None else []
top_state_now = ladder_state(core['top_score'], 'top')
bottom_state_now = ladder_state(core['bottom_score'], 'bottom')

if core['current_q'] == 'Q3':
    action_bias = 'Selective'
    action_sub = 'Defensive / preserve capital; tunggu breadth confirm sebelum broad beta.'
elif core['current_q'] == 'Q2':
    action_bias = 'Risk-on selective'
    action_sub = 'Prefer cyclicals / small caps if breadth confirm.'
elif core['current_q'] == 'Q4':
    action_bias = 'Defensive'
    action_sub = 'Jangan buru-buru base call kalau washout belum bersih.'
else:
    action_bias = 'Balanced'
    action_sub = 'Tactical only sampai confirmation membaik.'

next_read = f"{core['next_q']}"
if core['current_q'] == 'Q3' and core['next_q'] == 'Q2':
    next_sub = 'Early Q2 if breadth + small caps + USD cooling confirm'
elif core['next_q'] == core['current_q']:
    next_sub = f"Stay in {core['current_q']} / pressure persists"
else:
    next_sub = f"Most likely path from {core['current_q']}"

hero_cols = st.columns(6)
hero_items = [
    ("Decision Regime", core['current_q'], pill_html(f"{cur_stage} {core['current_q']}") if cur_stage else pill_html(core['current_q'])),
    ("Confidence", pct(core['confidence']), pill_html(f"Agreement {pct(core['agreement'])}")),
    ("Variant Now", variant_now, pill_html(core['sub_phase'])),
    ("Next Most Likely", next_read, pill_html(next_sub)),
    ("Crash Meter", pct(crash_now), pill_html(crash_meter_label(crash_now))),
    ("Action Bias", action_bias, pill_html(action_sub)),
]
for col, (title, value, sub_html) in zip(hero_cols, hero_items):
    with col:
        st.markdown(f"""
        <div class='hero-card'>
          <div class='metric-title'>{title}</div>
          <div style='font-size:1.2rem;font-weight:800'>{value}</div>
          <div class='metric-sub'>{sub_html}</div>
        </div>
        """, unsafe_allow_html=True)

quick_read = (
    f"Quick read: Sekarang {cur_stage} {core['current_q']} / {variant_now}. "
    f"Base case masih {core['current_q']}, tapi kalau transition lanjut jalur paling mungkin adalah {core['next_q']}. "
    f"Action bias sekarang: {action_bias}; {action_sub}"
)
st.markdown(f"<div class='note-box'><b>{quick_read}</b></div>", unsafe_allow_html=True)

# tabs like attachment-2
t_decision, t_cross, t_risk, t_details = st.tabs(["Decision", "Cross-Asset", "Risk / Relative", "Details"])

with t_decision:
    st.markdown("<div class='card'><div class='section-title'>DECISION SNAPSHOT</div>", unsafe_allow_html=True)
    st.markdown("<div class='mini-caption'>Urutan dari yang paling penting: apa yang dipakai sekarang, kalau next menang bakal jadi apa, apa yang harus dikonfirmasi, lalu event yang paling bisa mengubah bacaannya.</div>", unsafe_allow_html=True)
    st.markdown(table_html(["Priority", "Area", "Use now", "If next wins", "Confirm first"], build_playbook_priority_rows()), unsafe_allow_html=True)
    st.write("")
    st.markdown(table_html(["Focus", "Read", "Why it matters"], build_decision_key_rows()), unsafe_allow_html=True)
    st.write("")
    st.markdown(table_html(["Event", "When", "In", "Why it matters", "Likely impact"], build_macro_catalyst_rows()), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with t_cross:
    st.markdown("<div class='card'><div class='section-title'>CROSS-ASSET DIRECTIONAL BIAS</div>", unsafe_allow_html=True)
    st.markdown("<div class='mini-caption'>Panel operasional utama. Diurutkan dari yang paling langsung kepake dulu: strongest/weakest, area bias sekarang, ekspresi FX, baru detail stage map.</div>", unsafe_allow_html=True)
    st.markdown(f"**Current cycle stage ➜ {cur_stage} {core['current_q']}**")
    st.markdown(f"<div class='small-muted'>{STAGE_GUIDE[core['current_q']][cur_stage][2]}</div>", unsafe_allow_html=True)
    focus_rows = build_cross_asset_focus_rows()
    strong_now = ", ".join([r[0] for r in focus_rows[:4]])
    weak_now = ", ".join([r[0] for r in focus_rows[-4:]])
    st.markdown(f"**Strongest now ➜ {strong_now}**")
    st.markdown(f"**Weakest now ➜ {weak_now}**")
    st.markdown(table_html(["Area", "Bias now", "Use now", "If next wins"], focus_rows), unsafe_allow_html=True)
    st.write("")
    st.markdown(table_html(["Best simple FX expression", "Edge", "Read"], build_fx_expressions_table(fx_score_rows)), unsafe_allow_html=True)
    st.write("")
    st.markdown(f"**FX rank strong → weak ➜ {build_fx_rank_text(fx_score_rows)}**")
    st.markdown(table_html(["FX", "Bias", "Best use"], build_fx_display_rows(fx_score_rows)), unsafe_allow_html=True)
    st.write("")
    st.markdown(table_html(["", "Stage", "Usually strong", "Usually weak", "What confirms"], build_cross_asset_stage_table()), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with t_risk:
    st.markdown("<div class='card'><div class='section-title'>RISK / RELATIVE SNAPSHOT</div>", unsafe_allow_html=True)
    st.markdown("<div class='mini-caption'>Diurutkan dari yang paling penting buat survival: crash timing, state-now per scenario, sizing / risk budget, lalu context tambahan.</div>", unsafe_allow_html=True)
    st.markdown(f"**Current mode ➜ Base case / watch only**")
    st.markdown(f"**Top risk ➜ {top_state_now}**")
    st.markdown(f"**Bottom risk ➜ {bottom_state_now}**")
    st.markdown(f"**Crash meter now ➜ {pct(crash_now)} ({crash_meter_label(crash_now)})**")
    st.markdown(f"**Watch window ➜ {crash_watch_window()}**")
    st.write("")
    st.markdown(table_html(["Risk item", "Now", "How to use"], build_crash_timing_rows(crash_now)), unsafe_allow_html=True)
    st.write("")
    st.markdown(table_html(["Scenario", "State now", "How to use now", "What must improve", "What invalidates"], build_current_scenario_checks()), unsafe_allow_html=True)
    st.write("")
    if RISK_BUDGET_ROWS:
        st.markdown("**Portfolio posture / risk budget**")
        st.markdown(table_html(["Bucket", "Guide weight", "Use"], RISK_BUDGET_ROWS), unsafe_allow_html=True)
        st.write("")
    if TACTICAL_SIZING_ROWS:
        st.markdown(table_html(["Sizing step", "Guide", "Read"], TACTICAL_SIZING_ROWS), unsafe_allow_html=True)
        st.write("")
    if RISK_CONTROL_ROWS:
        st.markdown(table_html(["Control", "Guide", "Why it matters"], RISK_CONTROL_ROWS), unsafe_allow_html=True)
        st.write("")
    if SCENARIO_OVERRIDE_ROWS:
        st.markdown(table_html(["Override trigger", "Action", "Why"], SCENARIO_OVERRIDE_ROWS), unsafe_allow_html=True)
        st.write("")
    st.markdown(table_html(["Lens", "Bias now", "Simple read", "If next wins"], build_relative_compact_rows()), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with t_details:
    st.markdown("<div class='card'><div class='section-title'>DETAILS / DATA / VALIDATION</div>", unsafe_allow_html=True)
    if DATA_HEALTH_SUMMARY:
        st.markdown(table_html(["Health item", "Status", "Read"], DATA_HEALTH_SUMMARY), unsafe_allow_html=True)
        st.write("")
    if VALIDATION_READY is not None:
        validation_rows = [
            ["Validation readiness", f"{VALIDATION_READY.score*100:.1f}% ({VALIDATION_READY.label})", "Higher is better; still needs real WFO results"],
            ["Snapshot coverage", f"{SNAPSHOT_COUNT} days", f"Last archive {SNAPSHOT_LAST}"],
            ["Release-lag mode", "On" if adjust_series_dict is not None else "Off", "Approximate point-in-time availability"],
            ["Data health panel", "On" if DATA_HEALTH_ROWS else "Off", "Fresh/stale/missing tracking"],
            ["Risk budget engine", "On" if RISK_BUDGET_ROWS else "Off", "Sizing / exposure guidance scaffold"],
        ]
        st.markdown(table_html(["Validation item", "Status", "Read"], validation_rows), unsafe_allow_html=True)
        if VALIDATION_READY.notes:
            st.write("")
            st.markdown(table_html(["Open item"], [[x] for x in VALIDATION_READY.notes]), unsafe_allow_html=True)
        st.write("")
    st.markdown(table_html(["Walk-forward item", "Status", "Read"], WFO_SUMMARY_ROWS or [["Walk-forward", "No rows yet", "Need more archived snapshots"]]), unsafe_allow_html=True)
    st.write("")
    if WFO_CAL_ROWS:
        st.markdown(table_html(["Confidence bucket", "Avg confidence", "Realized hit rate"], WFO_CAL_ROWS), unsafe_allow_html=True)
        st.write("")
    if WFO_CRASH_CAL_ROWS:
        st.markdown(table_html(["Crash bucket", "Avg crash meter", "Future drawdown rate"], WFO_CRASH_CAL_ROWS), unsafe_allow_html=True)
        st.write("")
    if DATA_HEALTH_ROWS:
        with st.expander("Show full data health / release-lag rows", expanded=False):
            st.markdown(table_html(["Series", "Status", "Raw last obs", "Usable last obs", "Age days", "Release rule", "Rows dropped"], DATA_HEALTH_ROWS), unsafe_allow_html=True)
    st.write("")
    st.markdown(table_html(["", "Quad", "Base read", "Usually works", "Main crash branch", "Base crash risk"], build_quad_scenario_matrix()), unsafe_allow_html=True)
    st.write("")
    if leaders_status_text() != 'Hidden for now (coverage valid still zero)':
        st.markdown(f"**Leaders status ➜ {leaders_status_text()}**")
        st.write("")
    st.markdown(f"<div class='small-muted'><b>Core model:</b> {CORE_NAME} • <b>Policy:</b> {core.get('anchor_reason','Model-only')} • <b>Raw model:</b> {core.get('model_current_q', core['current_q'])} • <b>Snapshot path:</b> {SNAPSHOT_PATH or 'n/a'}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
