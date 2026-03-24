from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

APP_NAME = "QuantFinalV4_Max"
CORE_NAME = "Baseline_Blended_Core"

st.set_page_config(page_title=APP_NAME, layout="wide")

# =========================
# Config
# =========================
FRED_SERIES = {
    "PAYEMS": "PAYEMS",          # payrolls
    "UNRATE": "UNRATE",          # unemployment
    "ICSA": "ICSA",              # claims
    "INDPRO": "INDPRO",          # industrial production
    "RSAFS": "RSAFS",            # retail sales
    "PCEPI": "PCEPI",            # headline PCE price index
    "CPIAUCSL": "CPIAUCSL",      # CPI
    "CPILFESL": "CPILFESL",      # Core CPI
    "PPIACO": "PPIACO",          # PPI
    "HOUST": "HOUST",            # housing starts
    "PERMIT": "PERMIT",          # building permits
    "FEDFUNDS": "FEDFUNDS",      # policy rate
    "DGS2": "DGS2",              # 2Y
    "DGS10": "DGS10",            # 10Y
    "DGS30": "DGS30",            # 30Y
    "T5YIE": "T5YIE",            # 5Y breakeven
    "BAMLH0A0HYM2": "BAMLH0A0HYM2",  # HY OAS
    "NFCI": "NFCI",              # Chicago NFCI
    "SAHMREALTIME": "SAHMREALTIME",
    "RECPROUSM156N": "RECPROUSM156N",
    "WALCL": "WALCL",            # Fed balance sheet
    "DCOILWTICO": "DCOILWTICO",  # WTI
    "GOLDAMGBD228NLBM": "GOLDAMGBD228NLBM",  # Gold
    "DTWEXBGS": "DTWEXBGS",      # Broad USD
}

YF_SYMBOLS = [
    "SPY", "QQQ", "IWM", "TLT", "GLD", "UUP", "HYG", "DBC",
    "BTC-USD", "ETH-USD", "SOL-USD", "^JKSE", "EEM", "INDA", "EIDO"
]

WHAT_IF_SCENARIO_MATRIX = [
    ["Growth re-acceleration", "Risk-on broadens if growth breadth improves", "Beta / cyclicals / small caps", "Pure duration defensives"],
    ["Disinflationary slowdown", "Growth cools while inflation eases", "Duration / quality / gold", "Deep cyclicals"],
    ["Stagflation fork", "Growth weakens while inflation re-heats", "USD / gold / select energy", "Long-duration beta"],
    ["Bottoming / recovery", "Stress fades and breadth stabilizes", "Beta / cyclicals / EM", "Defensive overcrowding"],
]

DIVERGENCE_RULES = [
    ["US weak but IHSG / Indonesia stronger", "Regional decoupling; check commodity and FX support"],
    ["Crypto strong but equities lag", "Liquidity is selective or equity risk appetite is not confirming"],
    ["Bonds fail to rally in slowdown", "Possible inflation persistence / supply shock / term premium rise"],
    ["Gold and USD both firm", "Stress / stagflation / geopolitical hedge demand"],
]

CORRELATION_TRANSMISSION_PRIORS = [
    ["Oil up → inflation tail", "High", "Supports stagflation and energy-sensitive rotation"],
    ["USD up → EM pressure", "High", "Tighter global financial conditions usually hurt EM beta"],
    ["Real yields down → growth / crypto relief", "Medium-High", "Helps duration-sensitive assets if macro stress is not dominant"],
    ["Fear & Greed extreme + IWM blow-off", "Medium", "Flags exhaustion / crowded beta chase rather than core phase change"],
]

CRASH_TYPES = [
    ["Liquidity accident", "Fast de-risking; breadth and beta usually break first"],
    ["Growth scare", "Cyclicals roll over, duration tends to stabilize first"],
    ["Inflation shock", "Duration weak; USD / gold / select commodities can outperform"],
    ["Geopolitical shock", "Oil / gold / USD can diverge from normal growth playbook"],
]

CRASH_RECOVERY_ORDER = [
    ["Liquidity accident", "Policy / duration → quality → broad beta"],
    ["Growth scare", "Duration / defensives → quality cyclicals → broad beta"],
    ["Inflation shock", "USD / gold → selective cyclicals → duration later"],
    ["Geopolitical shock", "Hedges first, then normalization by region / sector"],
]

FALSE_RECOVERY_MAP = [
    ["Bottoming without breadth", "Provisional bottom only; lower-bottom risk remains"],
    ["Soft-landing trap", "Looks stable, then growth rolls again"],
    ["Re-acceleration fakeout", "Short burst in growth-sensitive assets without durable macro breadth"],
    ["Second-leg / double-dip", "Recovery fails and downside resumes"],
]

DEFAULT_FRED_LOOKBACK_YEARS = 15

# =========================
# Data classes
# =========================
@dataclass
class PhaseState:
    monthly_probs: Dict[str, float]
    quarterly_probs: Dict[str, float]
    blended_probs: Dict[str, float]
    current_phase: str
    confidence: float
    ambiguity: float
    agreement: float
    sub_phase: str
    validity: str
    regime_strength: float
    breadth: float
    fragility: float
    transition_pressure: float
    transition_conviction: float
    stay_probability: float
    next_phase_probs: Dict[str, float]
    top_score: float
    bottom_score: float
    higher_top_risk: float
    lower_bottom_risk: float


# =========================
# Utils
# =========================
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def top1(prob_map: Dict[str, float]) -> str:
    return max(prob_map, key=prob_map.get)


def top2(prob_map: Dict[str, float]) -> List[str]:
    return sorted(prob_map, key=prob_map.get, reverse=True)[:2]


def normalize_prob_map(pm: Dict[str, float]) -> Dict[str, float]:
    s = sum(pm.values())
    if s <= 0:
        n = len(pm)
        return {k: 1.0 / n for k in pm}
    return {k: v / s for k, v in pm.items()}


def pct_fmt(x: float) -> str:
    return f"{x * 100:.1f}%"


def score_color(x: float) -> str:
    if x >= 0.7:
        return "#22c55e"
    if x >= 0.45:
        return "#f59e0b"
    return "#ef4444"


def next_release_watch(now: Optional[pd.Timestamp] = None) -> List[Tuple[str, pd.Timestamp]]:
    now = pd.Timestamp.utcnow().tz_localize(None) if now is None else pd.Timestamp(now).tz_localize(None) if getattr(pd.Timestamp(now), "tzinfo", None) else pd.Timestamp(now)
    y, m = now.year, now.month
    def nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
        d = pd.Timestamp(year=year, month=month, day=1)
        shift = (weekday - d.weekday()) % 7
        return d + pd.Timedelta(days=shift + 7 * (n - 1))
    # rough recurring schedule, good enough as an event watch card
    events = [
        ("CPI", pd.Timestamp(year=y, month=m, day=min(13, pd.Period(f"{y}-{m:02d}").days_in_month))),
        ("PPI", pd.Timestamp(year=y, month=m, day=min(14, pd.Period(f"{y}-{m:02d}").days_in_month))),
        ("Retail Sales", pd.Timestamp(year=y, month=m, day=min(15, pd.Period(f"{y}-{m:02d}").days_in_month))),
        ("FOMC", nth_weekday(y, m, 2, 3)),
        ("NFP", nth_weekday(y, m, 4, 1)),
    ]
    out=[]
    for name, dt in events:
        if dt < now.normalize():
            nm = 1 if m < 12 else 1
            ny = y if m < 12 else y + 1
            if name == "CPI":
                dt = pd.Timestamp(year=ny, month=(m%12)+1, day=min(13, pd.Period(f"{ny}-{(m%12)+1:02d}").days_in_month))
            elif name == "PPI":
                dt = pd.Timestamp(year=ny, month=(m%12)+1, day=min(14, pd.Period(f"{ny}-{(m%12)+1:02d}").days_in_month))
            elif name == "Retail Sales":
                dt = pd.Timestamp(year=ny, month=(m%12)+1, day=min(15, pd.Period(f"{ny}-{(m%12)+1:02d}").days_in_month))
            elif name == "FOMC":
                dt = nth_weekday(ny, (m%12)+1, 2, 3)
            elif name == "NFP":
                dt = nth_weekday(ny, (m%12)+1, 4, 1)
        out.append((name, dt))
    return sorted(out, key=lambda x: x[1])[:3]


def path_status(state: PhaseState) -> Tuple[str, str, str, str]:
    target = f"{state.current_phase} → {top1(state.next_phase_probs)}"
    p = state.transition_pressure
    c = state.confidence
    if top1(state.next_phase_probs) == state.current_phase and state.stay_probability >= 0.62:
        status = "Stable / no clean shift"
    elif p < 0.28:
        status = "Starting"
    elif p < 0.45:
        status = "Building"
    elif p < 0.62:
        status = "Valid"
    else:
        status = "Confirmed"
    fail = "Low" if state.stay_probability > 0.65 else ("Medium" if state.stay_probability > 0.48 else "High")
    conf = "High" if c > 0.68 else ("Medium" if c > 0.48 else "Low")
    return target, status, conf, fail


def risk_snapshot(state: PhaseState, shocks: Dict[str, Tuple[str, str]], fear_greed: float) -> pd.DataFrame:
    growth_stress = clamp(state.transition_pressure * 0.9 + state.top_score * 0.35 - state.bottom_score * 0.2)
    inflation_stress = clamp(1 - state.bottom_score * 0.2 + state.higher_top_risk * 0.35)
    liquidity_stress = 0.2 if shocks.get("Liquidity", ("Low",))[0] == "Low" else (0.55 if shocks.get("Liquidity", ("Low",))[0] == "Medium" else 0.82)
    sentiment_stretch = clamp(abs(fear_greed - 50.0) / 50.0)
    items = [("Growth stress", growth_stress),("Inflation stress", inflation_stress),("Liquidity stress", liquidity_stress),("Sentiment stretch", sentiment_stretch)]
    return pd.DataFrame([(k, pct_fmt(v), "High" if v>=0.7 else ("Medium" if v>=0.45 else "Low")) for k,v in items], columns=["Engine","Score","Read"])


def next_playbook_hint(state: PhaseState) -> Tuple[List[str], List[str]]:
    q = top1(state.next_phase_probs)
    base = {
        "Q1": (["Beta", "Cyclicals", "Quality Growth"], ["USD", "Pure defensives"]),
        "Q2": (["Cyclicals", "Value", "Energy"], ["Long duration"]),
        "Q3": (["USD", "Gold", "Defensives"], ["Small caps", "Deep cyclicals"]),
        "Q4": (["Duration", "Quality", "Gold"], ["Beta", "Value cyclicals"]),
    }
    return base.get(q, (["Balanced"],["Crowded extremes"]))


# =========================
# Data fetch
# =========================
@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def fetch_fred_series(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(pd.compat.StringIO(resp.text)) if hasattr(pd.compat, "StringIO") else pd.read_csv(__import__("io").StringIO(resp.text))
    df.columns = ["DATE", series_id]
    df["DATE"] = pd.to_datetime(df["DATE"])
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    s = df.set_index("DATE")[series_id].dropna()
    return s


@st.cache_data(ttl=2 * 60 * 60, show_spinner=False)
def fetch_all_fred(series_map: Dict[str, str]) -> pd.DataFrame:
    out = {}
    for alias, sid in series_map.items():
        try:
            out[alias] = fetch_fred_series(sid)
        except Exception:
            continue
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, axis=1).sort_index()
    return df


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_yf_prices(symbols: List[str], period: str = "3y") -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if yf is not None:
        try:
            data = yf.download(symbols, period=period, auto_adjust=True, progress=False, threads=False)
            if isinstance(data.columns, pd.MultiIndex) and "Close" in data.columns.get_level_values(0):
                close = data["Close"].copy()
                frames.append(close)
            elif "Close" in getattr(data, "columns", []):
                close = data[["Close"]].copy()
                close.columns = symbols[:1]
                frames.append(close)
        except Exception:
            pass

    # Lightweight fallback for common ETF proxies when Yahoo is unavailable.
    stooq_map = {
        "SPY": "spy.us", "QQQ": "qqq.us", "IWM": "iwm.us", "TLT": "tlt.us",
        "GLD": "gld.us", "UUP": "uup.us", "HYG": "hyg.us", "DBC": "dbc.us",
        "EEM": "eem.us", "EIDO": "eido.us", "INDA": "inda.us",
    }
    stooq_frames = []
    for sym in symbols:
        if sym not in stooq_map:
            continue
        try:
            url = f"https://stooq.com/q/d/l/?s={stooq_map[sym]}&i=d"
            tmp = pd.read_csv(url)
            if {"Date", "Close"}.issubset(tmp.columns):
                tmp["Date"] = pd.to_datetime(tmp["Date"])
                s = pd.to_numeric(tmp["Close"], errors="coerce")
                stooq_frames.append(pd.Series(s.values, index=tmp["Date"], name=sym))
        except Exception:
            continue
    if stooq_frames:
        frames.append(pd.concat(stooq_frames, axis=1))

    if not frames:
        return pd.DataFrame()
    close = pd.concat(frames, axis=1)
    close = close.loc[:, ~close.columns.duplicated(keep="first")]
    return close.sort_index().dropna(how="all")


# =========================
# Feature engineering
# =========================
def safe_zscore(series: pd.Series, window: int) -> float:
    s = series.dropna().tail(window)
    if len(s) < max(8, window // 3):
        return 0.0
    sd = float(s.std(ddof=0))
    if not np.isfinite(sd) or sd < 1e-12:
        return 0.0
    return float((s.iloc[-1] - s.mean()) / sd)


def pct_change_z(series: pd.Series, periods: int, window: int) -> float:
    s = series.pct_change(periods).replace([np.inf, -np.inf], np.nan)
    return safe_zscore(s, window)


def level_change_z(series: pd.Series, periods: int, window: int) -> float:
    s = series.diff(periods)
    return safe_zscore(s, window)


def yoy_z(series: pd.Series, window: int = 36) -> float:
    s = series.pct_change(12).replace([np.inf, -np.inf], np.nan)
    return safe_zscore(s, window)


def latest_value(series: pd.Series) -> Optional[float]:
    s = series.dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


# =========================
# Core engine
# =========================
TRANSITION_BASE = np.array(
    [
        [0.62, 0.22, 0.05, 0.11],  # Q1
        [0.18, 0.58, 0.18, 0.06],  # Q2
        [0.06, 0.18, 0.58, 0.18],  # Q3
        [0.20, 0.05, 0.18, 0.57],  # Q4
    ],
    dtype=float,
)

FAMILY_SCORE_BY_QUAD = {
    "Q1": {"duration": 0.00, "usd": -0.35, "gold": -0.15, "beta": 0.80, "cyclical": 0.70},
    "Q2": {"duration": -0.70, "usd": 0.00, "gold": 0.05, "beta": 0.65, "cyclical": 0.90},
    "Q3": {"duration": 0.20, "usd": 0.70, "gold": 0.85, "beta": -0.75, "cyclical": -0.80},
    "Q4": {"duration": 0.90, "usd": 0.55, "gold": 0.35, "beta": -0.55, "cyclical": -0.60},
}

ASSET_TRANSLATION = {
    "US Stocks": {
        "duration": ["Long-duration growth", "Quality defensives"],
        "usd": ["US defensives", "Dollar beneficiaries"],
        "gold": ["Gold miners", "Precious-metals sensitivity"],
        "beta": ["Small caps", "High beta / momentum"],
        "cyclical": ["Industrials", "Materials / cyclicals"],
    },
    "Futures / Commodities": {
        "duration": ["Treasury longs"],
        "usd": ["Dollar strength basket"],
        "gold": ["Gold / precious metals"],
        "beta": ["Equity index beta"],
        "cyclical": ["Oil / industrial metals / reflation"],
    },
    "Forex": {
        "duration": ["Lower-yield / relief FX"],
        "usd": ["USD strength"],
        "gold": ["Safe-haven FX mix"],
        "beta": ["Risk-on FX"],
        "cyclical": ["Commodity FX"],
    },
    "Crypto": {
        "duration": ["BTC on falling real yields"],
        "usd": ["USD headwind to alts"],
        "gold": ["BTC as alt-safe-beta hybrid"],
        "beta": ["High beta alts"],
        "cyclical": ["Risk-on rotation into beta"],
    },
    "IHSG": {
        "duration": ["Rate-sensitive defensives"],
        "usd": ["Rupiah-sensitive losers / exporters winners"],
        "gold": ["Gold-linked / resource names"],
        "beta": ["Domestic beta / property / banks"],
        "cyclical": ["Coal, nickel, commodity cyclicals"],
    },
}


def build_phase_state(df: pd.DataFrame, fear_greed: Optional[float], iwm_blowoff: Optional[float]) -> PhaseState:
    # Monthly features
    g_m = np.mean([
        pct_change_z(df["PAYEMS"], 3, 24) if "PAYEMS" in df else 0.0,
        pct_change_z(df["INDPRO"], 3, 24) if "INDPRO" in df else 0.0,
        pct_change_z(df["RSAFS"], 3, 24) if "RSAFS" in df else 0.0,
        -level_change_z(df["UNRATE"], 3, 24) if "UNRATE" in df else 0.0,
        -pct_change_z(df["ICSA"], 4, 24) if "ICSA" in df else 0.0,
        pct_change_z(df["HOUST"], 3, 24) if "HOUST" in df else 0.0,
        pct_change_z(df["PERMIT"], 3, 24) if "PERMIT" in df else 0.0,
    ])
    i_m = np.mean([
        yoy_z(df["CPIAUCSL"]) if "CPIAUCSL" in df else 0.0,
        yoy_z(df["CPILFESL"]) if "CPILFESL" in df else 0.0,
        yoy_z(df["PPIACO"]) if "PPIACO" in df else 0.0,
        level_change_z(df["T5YIE"], 3, 24) if "T5YIE" in df else 0.0,
    ])
    stress_m = np.mean([
        level_change_z(df["SAHMREALTIME"], 1, 24) if "SAHMREALTIME" in df else 0.0,
        level_change_z(df["RECPROUSM156N"], 1, 24) if "RECPROUSM156N" in df else 0.0,
        level_change_z(df["BAMLH0A0HYM2"], 4, 24) if "BAMLH0A0HYM2" in df else 0.0,
        level_change_z(df["NFCI"], 4, 24) if "NFCI" in df else 0.0,
    ])
    pol_m = np.mean([
        level_change_z(df["FEDFUNDS"], 3, 24) if "FEDFUNDS" in df else 0.0,
        level_change_z(df["DGS2"], 4, 24) if "DGS2" in df else 0.0,
        level_change_z(df["DTWEXBGS"], 4, 24) if "DTWEXBGS" in df else 0.0,
    ])

    # Quarterly anchor
    g_q = np.mean([
        pct_change_z(df["PAYEMS"], 6, 60) if "PAYEMS" in df else 0.0,
        pct_change_z(df["INDPRO"], 6, 60) if "INDPRO" in df else 0.0,
        pct_change_z(df["RSAFS"], 6, 60) if "RSAFS" in df else 0.0,
        -level_change_z(df["UNRATE"], 6, 60) if "UNRATE" in df else 0.0,
        pct_change_z(df["PERMIT"], 6, 60) if "PERMIT" in df else 0.0,
    ])
    i_q = np.mean([
        yoy_z(df["CPIAUCSL"], 60) if "CPIAUCSL" in df else 0.0,
        yoy_z(df["CPILFESL"], 60) if "CPILFESL" in df else 0.0,
        yoy_z(df["PPIACO"], 60) if "PPIACO" in df else 0.0,
    ])
    stress_q = np.mean([
        level_change_z(df["SAHMREALTIME"], 3, 60) if "SAHMREALTIME" in df else 0.0,
        level_change_z(df["RECPROUSM156N"], 3, 60) if "RECPROUSM156N" in df else 0.0,
        level_change_z(df["NFCI"], 8, 60) if "NFCI" in df else 0.0,
    ])
    pol_q = np.mean([
        level_change_z(df["FEDFUNDS"], 6, 60) if "FEDFUNDS" in df else 0.0,
        level_change_z(df["DGS2"], 8, 60) if "DGS2" in df else 0.0,
        level_change_z(df["DTWEXBGS"], 8, 60) if "DTWEXBGS" in df else 0.0,
    ])

    g_up_m = sigmoid(1.40 * g_m - 0.42 * stress_m - 0.10 * pol_m)
    i_up_m = sigmoid(1.25 * i_m + 0.08 * pol_m)
    g_up_q = sigmoid(1.18 * g_q - 0.35 * stress_q - 0.08 * pol_q)
    i_up_q = sigmoid(1.10 * i_q + 0.05 * pol_q)

    monthly = normalize_prob_map({
        "Q1": g_up_m * (1 - i_up_m),
        "Q2": g_up_m * i_up_m,
        "Q3": (1 - g_up_m) * i_up_m,
        "Q4": (1 - g_up_m) * (1 - i_up_m),
    })
    quarterly = normalize_prob_map({
        "Q1": g_up_q * (1 - i_up_q),
        "Q2": g_up_q * i_up_q,
        "Q3": (1 - g_up_q) * i_up_q,
        "Q4": (1 - g_up_q) * (1 - i_up_q),
    })
    blended = normalize_prob_map({q: 0.35 * monthly[q] + 0.65 * quarterly[q] for q in ["Q1", "Q2", "Q3", "Q4"]})

    # Confidence/agreement
    m_top = top1(monthly)
    q_top = top1(quarterly)
    agreement = 1.0 if m_top == q_top else 0.55
    sorted_blend = sorted(blended.values(), reverse=True)
    ambiguity = 1.0 - (sorted_blend[0] - sorted_blend[1])
    confidence = clamp(0.55 * agreement + 0.45 * (1 - ambiguity))

    current_phase = top1(blended)

    # Sub-phase / validity
    regime_strength = clamp((float(sorted_blend[0] - sorted_blend[-1])) / 0.60)
    growth_parts = [
        pct_change_z(df["PAYEMS"], 3, 24) if "PAYEMS" in df else 0.0,
        pct_change_z(df["INDPRO"], 3, 24) if "INDPRO" in df else 0.0,
        pct_change_z(df["RSAFS"], 3, 24) if "RSAFS" in df else 0.0,
        -level_change_z(df["UNRATE"], 3, 24) if "UNRATE" in df else 0.0,
        -pct_change_z(df["ICSA"], 4, 24) if "ICSA" in df else 0.0,
        pct_change_z(df["HOUST"], 3, 24) if "HOUST" in df else 0.0,
        pct_change_z(df["PERMIT"], 3, 24) if "PERMIT" in df else 0.0,
    ]
    infl_parts = [
        yoy_z(df["CPIAUCSL"]) if "CPIAUCSL" in df else 0.0,
        yoy_z(df["CPILFESL"]) if "CPILFESL" in df else 0.0,
        yoy_z(df["PPIACO"]) if "PPIACO" in df else 0.0,
        level_change_z(df["T5YIE"], 3, 24) if "T5YIE" in df else 0.0,
    ]
    g_sign = 1 if current_phase in ["Q1", "Q2"] else -1
    i_sign = 1 if current_phase in ["Q2", "Q3"] else -1
    g_b = np.mean([1.0 if np.sign(x) == g_sign and abs(x) > 0.10 else 0.0 for x in growth_parts])
    i_b = np.mean([1.0 if np.sign(x) == i_sign and abs(x) > 0.10 else 0.0 for x in infl_parts])
    breadth = clamp(0.55 * g_b + 0.45 * i_b)
    fragility = clamp(0.45 * (1.0 - confidence) + 0.35 * (1.0 - breadth) + 0.20 * max(stress_m, 0.0))
    if confidence > 0.72 and regime_strength > 0.55 and fragility < 0.38:
        validity = "Stable"
    elif confidence > 0.55:
        validity = "Fragile"
    else:
        validity = "Decaying"

    if current_phase == "Q2":
        sub_phase = "Early Reflation" if g_m > 0.25 and stress_m < 0 else ("Late Reflation / Topping" if stress_m > 0.2 else "Mid Reflation")
    elif current_phase == "Q3":
        sub_phase = "Stagflationary Slowdown" if i_m > 0.15 else "Late Growth Rollover"
    elif current_phase == "Q4":
        sub_phase = "Deflationary Slowdown" if stress_m > 0.1 else "Bottoming Attempt"
    else:
        sub_phase = "Goldilocks Expansion" if stress_m < 0 else "Early Recovery"

    # Transition/hazard
    curr_idx = ["Q1", "Q2", "Q3", "Q4"].index(current_phase)
    trans = TRANSITION_BASE.copy()
    # feature-conditioned tilt
    if stress_m > 0.25:
        trans[curr_idx, 2] += 0.05 if current_phase in ["Q1", "Q2"] else 0.0
        trans[curr_idx, 3] += 0.04 if current_phase in ["Q1", "Q2", "Q3"] else 0.0
    if g_m > 0.30 and i_m < 0:
        trans[curr_idx, 0] += 0.05
    if i_m > 0.25 and g_m < 0:
        trans[curr_idx, 2] += 0.05
    trans = trans / trans.sum(axis=1, keepdims=True)
    curr_prob_vec = np.array([blended[q] for q in ["Q1", "Q2", "Q3", "Q4"]], dtype=float)
    next_prob_vec = curr_prob_vec @ trans
    next_phase_probs = normalize_prob_map({q: float(v) for q, v in zip(["Q1", "Q2", "Q3", "Q4"], next_prob_vec)})
    stay_probability = float(next_phase_probs[current_phase])
    transition_pressure = clamp(1.0 - stay_probability)
    _next_sorted_vals = sorted(next_phase_probs.values(), reverse=True)
    transition_conviction = clamp((_next_sorted_vals[0] - _next_sorted_vals[1]) / 0.35)

    # Turning-point ladder with optional sentiment overlays
    late_macro = clamp(0.55 * max(stress_m, 0.0) + 0.45 * transition_pressure)
    bottom_macro = clamp(0.50 * max(-g_m, 0.0) + 0.50 * max(stress_m, 0.0))

    fg_greed = 0.0
    fg_fear = 0.0
    if fear_greed is not None:
        fg = float(fear_greed)
        fg_greed = clamp((fg - 70.0) / 30.0)
        fg_fear = clamp((30.0 - fg) / 30.0)

    iwm_top = clamp(iwm_blowoff or 0.0)
    top_score = clamp(0.72 * late_macro + 0.08 * fg_greed + 0.08 * iwm_top + 0.12 * max(0.0, transition_pressure - 0.40))
    bottom_score = clamp(0.78 * bottom_macro + 0.10 * fg_fear + 0.12 * max(0.0, stress_m))
    higher_top_risk = clamp(0.55 * top_score + 0.25 * fg_greed + 0.20 * iwm_top)
    lower_bottom_risk = clamp(0.65 * bottom_score + 0.35 * fg_fear)

    return PhaseState(
        monthly_probs=monthly,
        quarterly_probs=quarterly,
        blended_probs=blended,
        current_phase=current_phase,
        confidence=confidence,
        ambiguity=ambiguity,
        agreement=agreement,
        sub_phase=sub_phase,
        validity=validity,
        regime_strength=regime_strength,
        breadth=breadth,
        fragility=fragility,
        transition_pressure=transition_pressure,
        transition_conviction=transition_conviction,
        stay_probability=stay_probability,
        next_phase_probs=next_phase_probs,
        top_score=top_score,
        bottom_score=bottom_score,
        higher_top_risk=higher_top_risk,
        lower_bottom_risk=lower_bottom_risk,
    )


# =========================
# Timing / playbook / relative / shocks
# =========================
def timing_engine(state: PhaseState) -> Dict[str, str]:
    entry_quality = "High" if state.confidence > 0.72 and state.transition_pressure < 0.35 else ("Medium" if state.confidence > 0.55 else "Low")
    if state.top_score > 0.65:
        rotation = "Late / Reduce chase"
    elif state.bottom_score > 0.60:
        rotation = "Bottoming / Probe only"
    elif state.transition_pressure > 0.45:
        rotation = "Transitioning / Tight review"
    else:
        rotation = "Stable / Follow base playbook"
    hold = "Longer" if state.stay_probability > 0.68 else ("Medium" if state.stay_probability > 0.50 else "Short")
    invalidation = "Tight" if state.validity != "Stable" or state.transition_pressure > 0.42 else "Normal"
    return {
        "Entry Quality": entry_quality,
        "Rotation Timing": rotation,
        "Hold Bias": hold,
        "Invalidation Window": invalidation,
    }


def family_scores(state: PhaseState) -> Dict[str, float]:
    fam = {k: 0.0 for k in ["duration", "usd", "gold", "beta", "cyclical"]}
    for q, p in state.blended_probs.items():
        for f, s in FAMILY_SCORE_BY_QUAD[q].items():
            fam[f] += 0.70 * p * s
    for q, p in state.next_phase_probs.items():
        for f, s in FAMILY_SCORE_BY_QUAD[q].items():
            fam[f] += 0.30 * p * s

    # adaptive tilts
    if state.top_score > 0.60:
        fam["beta"] -= 0.12
        fam["cyclical"] -= 0.08
        fam["gold"] += 0.06
        fam["usd"] += 0.04
    if state.bottom_score > 0.60:
        fam["beta"] += 0.08
        fam["cyclical"] += 0.08
        fam["duration"] += 0.03
    return fam


def build_playbook(state: PhaseState) -> Dict[str, Dict[str, List[str]]]:
    fam = family_scores(state)
    ordered = sorted(fam, key=fam.get, reverse=True)
    playbook = {}
    for asset_class, mapping in ASSET_TRANSLATION.items():
        playbook[asset_class] = {
            "Winners": mapping[ordered[0]] + mapping[ordered[1]][:1],
            "Losers": mapping[ordered[-1]] + mapping[ordered[-2]][:1],
        }
    return playbook


def relative_engine(prices: pd.DataFrame, df: pd.DataFrame, fear_greed: Optional[float] = None) -> Dict[str, str]:
    out: Dict[str, str] = {}
    rets_3m = prices.ffill().pct_change(63) if not prices.empty else pd.DataFrame()
    last = rets_3m.iloc[-1].to_dict() if not rets_3m.empty else {}

    # Use EIDO as Indonesia/IHSG liquid proxy if ^JKSE is unavailable.
    spy = last.get("SPY", np.nan)
    eem = last.get("EEM", np.nan)
    jkse = last.get("^JKSE", last.get("EIDO", np.nan))
    btc = last.get("BTC-USD", np.nan)
    qqq = last.get("QQQ", np.nan)

    usd_z = safe_zscore(df["DTWEXBGS"], 26) if "DTWEXBGS" in df else 0.0
    oil_z = safe_zscore(df["DCOILWTICO"], 26) if "DCOILWTICO" in df else 0.0
    fci_z = safe_zscore(df["NFCI"], 26) if "NFCI" in df else 0.0
    walcl_z = safe_zscore(df["WALCL"], 26) if "WALCL" in df else 0.0

    if pd.notna(spy) and pd.notna(eem):
        out["US vs EM"] = "US stronger" if spy > eem + 0.03 else ("EM stronger" if eem > spy + 0.03 else "Balanced")
    else:
        out["US vs EM"] = "US stronger" if usd_z > 0.5 and fci_z > 0 else ("EM stronger" if usd_z < -0.4 and oil_z > 0 else "Balanced macro fallback")

    if pd.notna(jkse) and pd.notna(spy):
        out["IHSG vs US"] = "IHSG stronger" if jkse > spy + 0.03 else ("US stronger" if spy > jkse + 0.03 else "Balanced")
    else:
        out["IHSG vs US"] = "IHSG stronger" if oil_z > 0.4 and usd_z <= 0.2 else ("US stronger" if usd_z > 0.5 else "Balanced macro fallback")

    if pd.notna(jkse) and pd.notna(eem):
        out["IHSG vs EM"] = "IHSG stronger" if jkse > eem + 0.03 else ("EM stronger" if eem > jkse + 0.03 else "Balanced")
    else:
        out["IHSG vs EM"] = "IHSG stronger" if oil_z > 0.4 else ("EM stronger" if usd_z > 0.6 else "Balanced macro fallback")

    if pd.notna(btc) and pd.notna(qqq):
        out["Crypto vs Liquidity"] = "Crypto leading" if btc > qqq + 0.05 else ("Liquidity not confirming" if qqq > btc + 0.05 else "Aligned")
    else:
        fg = 50.0 if fear_greed is None else float(fear_greed)
        out["Crypto vs Liquidity"] = "Crypto leading" if walcl_z > 0.4 and fg > 60 else ("Liquidity not confirming" if fci_z > 0.4 or fg < 35 else "Aligned")
    return out


def shock_engine(df: pd.DataFrame, fear_greed: Optional[float], state: PhaseState) -> Dict[str, Tuple[str, str]]:
    oil = latest_value(df["DCOILWTICO"]) if "DCOILWTICO" in df else None
    y2 = latest_value(df["DGS2"]) if "DGS2" in df else None
    usd = latest_value(df["DTWEXBGS"]) if "DTWEXBGS" in df else None
    walcl_mom = pct_change_z(df["WALCL"], 4, 40) if "WALCL" in df else 0.0

    def sev(score: float) -> str:
        return "High" if score > 0.67 else ("Medium" if score > 0.40 else "Low")

    policy = clamp(max(level_change_z(df["FEDFUNDS"], 3, 24) if "FEDFUNDS" in df else 0.0, 0.0) * 0.5 + max(level_change_z(df["DGS2"], 4, 24) if "DGS2" in df else 0.0, 0.0) * 0.5)
    geo = clamp((max(yoy_z(df["DCOILWTICO"], 24) if "DCOILWTICO" in df else 0.0, 0.0) * 0.6) + (0.4 if state.top_score > 0.55 and (fear_greed or 50) > 70 else 0.0))
    liq = clamp(max(-walcl_mom, 0.0) * 0.7 + max(level_change_z(df["NFCI"], 4, 24) if "NFCI" in df else 0.0, 0.0) * 0.3)
    infl = clamp(max(yoy_z(df["CPIAUCSL"], 24) if "CPIAUCSL" in df else 0.0, 0.0) * 0.6 + max(yoy_z(df["PPIACO"], 24) if "PPIACO" in df else 0.0, 0.0) * 0.4)
    growth = clamp(max(-pct_change_z(df["RSAFS"], 3, 24) if "RSAFS" in df else 0.0, 0.0) * 0.4 + max(level_change_z(df["UNRATE"], 3, 24) if "UNRATE" in df else 0.0, 0.0) * 0.6)
    anomaly = clamp(abs(state.monthly_probs[top1(state.monthly_probs)] - state.quarterly_probs[top1(state.quarterly_probs)]) * 0.6 + (0.25 if state.validity != "Stable" else 0.0))

    return {
        "Policy Shock": (sev(policy), "Fed / front-end / policy repricing"),
        "Geopolitical Shock": (sev(geo), "Oil / war / commodity disruption sensitivity"),
        "Liquidity Shock": (sev(liq), "Funding / financial conditions / balance-sheet stress"),
        "Inflation Shock": (sev(infl), "Re-acceleration / commodity / pricing pressure"),
        "Growth Shock": (sev(growth), "Demand / labor / slowdown deterioration"),
        "Anomaly Flag": (sev(anomaly), "Correlation break / data divergence / low-transmission clarity"),
    }


def build_what_if(state: PhaseState, shocks: Dict[str, Tuple[str, str]]) -> List[Dict[str, str]]:
    out = []
    top2_next = top2(state.next_phase_probs)
    out.append({
        "Scenario": "Base Case",
        "Probability": pct_fmt(state.next_phase_probs[top2_next[0]]),
        "Impact": f"Current {state.current_phase} most likely evolves toward {top2_next[0]}",
        "Trigger": "No major shock; blended core remains dominant",
    })
    out.append({
        "Scenario": "Re-acceleration",
        "Probability": pct_fmt(state.next_phase_probs.get("Q1", 0.0) if state.current_phase != "Q1" else state.stay_probability),
        "Impact": "Beta / cyclicals / growth-sensitive risk assets improve",
        "Trigger": "Growth breadth re-accelerates, labor stabilizes, inflation pressure cools",
    })
    out.append({
        "Scenario": "Stagflation Fork",
        "Probability": pct_fmt(state.next_phase_probs.get("Q3", 0.0)),
        "Impact": "Gold / USD / selective commodities strengthen; beta loses edge",
        "Trigger": "Growth rolls while inflation or commodity pressure stays sticky",
    })
    out.append({
        "Scenario": "Bottoming / Recovery",
        "Probability": pct_fmt(state.bottom_score * 0.5),
        "Impact": "Probe duration + quality, then beta only on confirmation",
        "Trigger": "Bottom ladder improves while lower-bottom risk fades",
    })
    if shocks["Geopolitical Shock"][0] in ["Medium", "High"]:
        out.append({
            "Scenario": "War / Oil Shock",
            "Probability": shocks["Geopolitical Shock"][0],
            "Impact": "Commodity, gold, USD, selective exporters outperform; broad EM may lag",
            "Trigger": "Energy, shipping, or sanctions escalation",
        })
    return out


def posture_from_state(state: PhaseState) -> str:
    if state.confidence < 0.42 or state.fragility > 0.68:
        return "Wait / low conviction"
    if state.current_phase in ["Q3", "Q4"] and state.top_score > 0.45:
        return "Defensive"
    if state.current_phase in ["Q1", "Q2"] and state.regime_strength > 0.55 and state.fragility < 0.45:
        return "Aggressive"
    return "Balanced"


def _bucket(v: float, cuts=(0.33,0.66), labels=("Weak","Medium","Strong")) -> str:
    return labels[0] if v < cuts[0] else (labels[1] if v < cuts[1] else labels[2])


def relative_detail_engine(prices: pd.DataFrame, df: pd.DataFrame, fear_greed: Optional[float]=None) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    rets_1m = prices.ffill().pct_change(21) if not prices.empty else pd.DataFrame()
    rets_3m = prices.ffill().pct_change(63) if not prices.empty else pd.DataFrame()
    last1 = rets_1m.iloc[-1].to_dict() if not rets_1m.empty else {}
    last3 = rets_3m.iloc[-1].to_dict() if not rets_3m.empty else {}
    usd_z = safe_zscore(df["DTWEXBGS"], 26) if "DTWEXBGS" in df else 0.0
    oil_z = safe_zscore(df["DCOILWTICO"], 26) if "DCOILWTICO" in df else 0.0
    fci_z = safe_zscore(df["NFCI"], 26) if "NFCI" in df else 0.0

    def build(name, a, b, pos_label, neg_label, confirm_hint=True):
        a1, b1 = last1.get(a, np.nan), last1.get(b, np.nan)
        a3, b3 = last3.get(a, np.nan), last3.get(b, np.nan)
        if pd.notna(a3) and pd.notna(b3):
            diff = float(a3 - b3)
            diff1 = float((0 if pd.isna(a1) else a1) - (0 if pd.isna(b1) else b1))
        else:
            diff = 0.0; diff1 = 0.0
        direction = pos_label if diff > 0.03 else (neg_label if diff < -0.03 else "Balanced")
        strength = clamp(abs(diff) / 0.12)
        state = "Building" if diff1 * diff > 0 and abs(diff1) > 0.01 else ("Fading" if diff1 * diff < 0 and abs(diff1) > 0.01 else "Stable")
        quality = "Healthy" if strength > 0.55 and state != "Fading" else ("Narrow" if strength > 0.35 else "Fragile")
        sustain = "High" if strength > 0.62 and state != "Fading" else ("Medium" if strength > 0.35 else "Low")
        conf = "Confirmed"
        if name == "US vs EM" and usd_z > 0.7 and direction == "EM stronger": conf = "Mixed"
        if name.startswith("IHSG") and usd_z > 0.8 and direction.startswith("IHSG"): conf = "Mixed"
        if name == "Crypto vs Liquidity" and fci_z > 0.4 and direction == "Crypto leading": conf = "Not confirmed"
        out[name] = {"Direction": direction, "Strength": _bucket(strength), "State": state, "Quality": quality, "Sustainability": sustain, "Confirmation": conf}

    build("US vs EM", "SPY", "EEM", "US stronger", "EM stronger")
    build("IHSG vs US", "^JKSE", "SPY", "IHSG stronger", "US stronger")
    if out["IHSG vs US"]["Direction"] == "Balanced" and "EIDO" in last3 and "SPY" in last3:
        diff=float(last3.get("EIDO",0)-last3.get("SPY",0)); out["IHSG vs US"]["Direction"] = "IHSG stronger" if diff>0.03 else ("US stronger" if diff<-0.03 else "Balanced")
    build("IHSG vs EM", "^JKSE", "EEM", "IHSG stronger", "EM stronger")
    if out["IHSG vs EM"]["Direction"] == "Balanced" and "EIDO" in last3 and "EEM" in last3:
        diff=float(last3.get("EIDO",0)-last3.get("EEM",0)); out["IHSG vs EM"]["Direction"] = "IHSG stronger" if diff>0.03 else ("EM stronger" if diff<-0.03 else "Balanced")
    build("Crypto vs Liquidity", "BTC-USD", "QQQ", "Crypto leading", "Liquidity not confirming")
    return out


def size_rotation_engine(prices: pd.DataFrame, df: pd.DataFrame, fear_greed: Optional[float]=None, iwm_blowoff: float=0.0) -> Dict[str, Dict[str, str]]:
    out = {}
    rets_1m = prices.ffill().pct_change(21) if not prices.empty else pd.DataFrame()
    rets_3m = prices.ffill().pct_change(63) if not prices.empty else pd.DataFrame()
    l1 = rets_1m.iloc[-1].to_dict() if not rets_1m.empty else {}
    l3 = rets_3m.iloc[-1].to_dict() if not rets_3m.empty else {}
    fg = 50 if fear_greed is None else float(fear_greed)
    def mk(name, a, b, pos, neg, speculative=False):
        da = float(l3.get(a, np.nan)) if a in l3 else np.nan
        db = float(l3.get(b, np.nan)) if b in l3 else np.nan
        d1a = float(l1.get(a, np.nan)) if a in l1 else np.nan
        d1b = float(l1.get(b, np.nan)) if b in l1 else np.nan
        diff = 0.0 if (pd.isna(da) or pd.isna(db)) else da-db
        diff1 = 0.0 if (pd.isna(d1a) or pd.isna(d1b)) else d1a-d1b
        direction = pos if diff > 0.03 else (neg if diff < -0.03 else "Balanced")
        strength = clamp(abs(diff)/0.15)
        state = "Building" if diff1*diff>0 and abs(diff1)>0.01 else ("Peaking" if abs(diff)>0.08 and abs(diff1)<0.01 else ("Fading" if diff1*diff<0 and abs(diff1)>0.01 else "Stable"))
        quality = "Healthy"
        if speculative and direction==pos and fg>70: quality = "Frothy"
        elif strength < 0.35: quality = "Weak"
        elif state=="Fading": quality = "Exhausting"
        sustain = "High" if strength>0.6 and quality=="Healthy" else ("Low" if quality in ["Frothy","Exhausting","Weak"] else "Medium")
        conf = "Confirmed" if strength>0.60 else ("Partial" if strength>0.35 else "Not confirmed")
        out[name] = {"Direction": direction, "Strength": _bucket(strength), "State": state, "Quality": quality, "Sustainability": sustain, "Confirmation": conf}
    mk("US Small Caps vs Big Caps", "IWM", "SPY", "Small > Big", "Big > Small", speculative=True)
    mk("IHSG Small / 2nd Liners vs Big Caps (proxy)", "EIDO", "SPY", "Small/2nd liners > Big", "Big > Small/2nd liners", speculative=True)
    mk("Crypto Alts vs BTC", "ETH-USD", "BTC-USD", "Alts > BTC", "BTC > Alts", speculative=True)
    return out


# =========================
# Visuals
# =========================
CSS = """
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 1rem; max-width: 1380px;}
html, body, [class*="css"] {font-family: Inter, system-ui, sans-serif;}
.muted {color:#9ca3af; font-size:0.86rem;}
.card {
  border:1px solid rgba(255,255,255,0.10);
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
  padding:14px 16px;
  border-radius:18px;
  margin-bottom:12px;
}
.card h4 {margin:0 0 8px 0; font-size:1rem;}
.kpi {
  padding:12px 14px;
  border-radius:16px;
  border:1px solid rgba(255,255,255,0.08);
  background:rgba(255,255,255,0.03);
}
.kpi-title {font-size:0.8rem; color:#9ca3af; margin-bottom:4px;}
.kpi-val {font-size:1.18rem; font-weight:700;}
.pill {
  display:inline-block;
  padding:3px 10px;
  border-radius:999px;
  font-size:0.78rem;
  margin-right:6px;
  border:1px solid rgba(255,255,255,0.10);
  background:rgba(255,255,255,0.05);
}
.mm-title {font-size: 0.9rem; color:#9ca3af; margin-bottom:8px; text-transform:uppercase; letter-spacing:0.04em;}
.tree {line-height:1.55; font-size:0.96rem;}
.tree .node {margin:2px 0;}
.tree .child {padding-left:16px; color:#d1d5db;}
.good {color:#22c55e;} .warn{color:#f59e0b;} .bad{color:#ef4444;}
.section-title {font-size:1.05rem; font-weight:700; margin: 0 0 10px 0;}
</style>
"""


def pill(label: str, color: Optional[str] = None) -> str:
    style = f" style='border-color:{color};color:{color};'" if color else ""
    return f"<span class='pill'{style}>{label}</span>"


def render_compact_matrix(title: str, rows: list[list[str]], columns: list[str], lens_width: str = "28%"):
    st.markdown(f"**{title}**")
    # Short headers so the table fits without ugly wrapping.
    header_labels = columns[:]
    header_map = {
        "Direction": "Dir",
        "Strength": "Str",
        "Sustainability": "Sustain",
        "Confirmation": "Confirm",
        "Relative Lens": "Relative Lens",
        "Rotation Lens": "Rotation Lens",
    }
    header_labels = [header_map.get(c, c) for c in header_labels]
    headers = ''.join([f"<th>{c}</th>" for c in header_labels])
    body = ''
    for row in rows:
        body += '<tr>' + ''.join([f"<td>{cell}</td>" for cell in row]) + '</tr>'
    html = f"""
    <style>
    .compact-matrix-wrap {{overflow-x:hidden;}}
    .compact-matrix {{width:100%; border-collapse:collapse; table-layout:fixed; font-size:11px;}}
    .compact-matrix th, .compact-matrix td {{border:1px solid rgba(148,163,184,.18); padding:8px 8px; vertical-align:top; overflow-wrap:anywhere;}}
    .compact-matrix th {{color:#94a3b8; font-weight:600; background:rgba(255,255,255,.02); line-height:1.2;}}
    .compact-matrix td:first-child, .compact-matrix th:first-child {{width:{lens_width};}}
    .compact-matrix td:not(:first-child), .compact-matrix th:not(:first-child) {{text-align:left;}}
    </style>
    <div class="compact-matrix-wrap">
    <table class="compact-matrix">
      <thead><tr>{headers}</tr></thead>
      <tbody>{body}</tbody>
    </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def prob_bar(prob_map: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({"Phase": list(prob_map.keys()), "Probability": list(prob_map.values())}).sort_values("Probability", ascending=False)


# =========================
# App
# =========================
st.markdown(CSS, unsafe_allow_html=True)
st.title(f"{APP_NAME}")
st.caption(f"Core alpha engine: {CORE_NAME}  •  Visual shell: mind-map card layout  •  Live backbone: FRED + optional Yahoo")

with st.sidebar:
    st.subheader("Data & Overlay")
    years = st.slider("FRED lookback (years)", 5, 25, DEFAULT_FRED_LOOKBACK_YEARS)
    use_yf = st.toggle("Use live Yahoo prices", value=True)
    fear_greed = st.slider("Fear & Greed (daily overlay)", 0, 100, 55)
    iwm_blowoff = st.slider("IWM blow-off risk", 0.0, 1.0, 0.15, 0.01)
    st.markdown("<div class='muted'>Fear & Greed and IWM blow-off are overlays for timing/top-bottom only, not the core phase engine.</div>", unsafe_allow_html=True)

with st.spinner("Fetching live macro data..."):
    fred = fetch_all_fred(FRED_SERIES)

if fred.empty:
    st.error("No FRED data loaded. Check internet access and try again.")
    st.stop()

cutoff = fred.index.max() - pd.DateOffset(years=years)
fred = fred.loc[fred.index >= cutoff].copy()
prices = fetch_yf_prices(YF_SYMBOLS, period="3y") if use_yf else pd.DataFrame()
state = build_phase_state(fred, fear_greed=float(fear_greed), iwm_blowoff=float(iwm_blowoff))
timing = timing_engine(state)
playbook = build_playbook(state)
# Compact headline order used by hero cards; prefer US Stocks winners, then any winners found.
_playbook_headline = []
if isinstance(playbook, dict):
    if 'US Stocks' in playbook and isinstance(playbook['US Stocks'], dict):
        _playbook_headline = list(playbook['US Stocks'].get('Winners', []))
    if not _playbook_headline:
        for _asset_class, _mapping in playbook.items():
            if isinstance(_mapping, dict):
                _playbook_headline.extend(list(_mapping.get('Winners', [])))
    # preserve order while removing duplicates
    seen = set()
    playbook_order = [x for x in _playbook_headline if not (x in seen or seen.add(x))]
else:
    playbook_order = []
if not playbook_order:
    playbook_order = ['Balanced / wait for stronger edge']
relative = relative_engine(prices, fred, float(fear_greed))
relative_detail = relative_detail_engine(prices, fred, float(fear_greed))
size_rotation = size_rotation_engine(prices, fred, float(fear_greed), float(iwm_blowoff))
shocks = shock_engine(fred, float(fear_greed), state)
what_if_cases = build_what_if(state, shocks)
positioning_posture = posture_from_state(state)
base_override_status = "Override active" if any(v[0] == "High" for v in shocks.values()) else ("Watch" if any(v[0] == "Medium" for v in shocks.values()) else "Base case")

# Top KPIs
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"<div class='kpi'><div class='kpi-title'>Current Phase</div><div class='kpi-val'>{state.current_phase}</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='kpi'><div class='kpi-title'>Confidence</div><div class='kpi-val' style='color:{score_color(state.confidence)}'>{pct_fmt(state.confidence)}</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='kpi'><div class='kpi-title'>Sub-Phase</div><div class='kpi-val'>{state.sub_phase}</div></div>", unsafe_allow_html=True)
with k4:
    st.markdown(f"<div class='kpi'><div class='kpi-title'>Top Risk</div><div class='kpi-val' style='color:{score_color(state.top_score)}'>{pct_fmt(state.top_score)}</div></div>", unsafe_allow_html=True)
with k5:
    st.markdown(f"<div class='kpi'><div class='kpi-title'>Bottom Risk</div><div class='kpi-val' style='color:{score_color(state.bottom_score)}'>{pct_fmt(state.bottom_score)}</div></div>", unsafe_allow_html=True)

# Main mind-map layout
hero1, hero2, hero3, hero4, hero5 = st.columns(5, gap="small")
for col, title, value, extra in [
    (hero1, "CURRENT", state.current_phase, pill(state.validity, score_color(state.confidence))),
    (hero2, "NEXT", top1(state.next_phase_probs), pill(f"Hazard {pct_fmt(state.transition_pressure)}")),
    (hero3, "PLAYBOOK", playbook_order[0], pill(f"Conviction {pct_fmt(state.confidence)}")),
    (hero4, "RELATIVE", relative.get("US vs EM", "Neutral"), pill(relative.get("IHSG vs US", "Neutral"))),
    (hero5, "SHOCKS", "Overlay", pill(f"Top {pct_fmt(state.top_score)} / Bottom {pct_fmt(state.bottom_score)}")),
]:
    with col:
        st.markdown(f"<div class='card'><div class='kpi-title'>{title}</div><div class='kpi-val' style='font-size:18px'>{value}</div><div style='margin-top:8px'>{extra}</div></div>", unsafe_allow_html=True)

left, right = st.columns([1.2, 1.0], gap="large")

with left:
    current_tab, next_tab, playbook_tab = st.tabs(["Current", "Next", "Playbook"])

    with current_tab:
        st.markdown("<div class='card'><div class='section-title'>CURRENT MAP</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='tree'>"
            f"<div class='node'><b>Phase</b> → {state.current_phase}</div>"
            f"<div class='node'><b>Confidence</b> → {pct_fmt(state.confidence)} &nbsp; {pill(state.validity, score_color(state.confidence))}</div>"
            f"<div class='node'><b>Agreement</b> → {pct_fmt(state.agreement)} &nbsp; {pill('Monthly ' + top1(state.monthly_probs))} {pill('Quarterly ' + top1(state.quarterly_probs))}</div>"
            f"<div class='node'><b>Sub-Phase</b> → {state.sub_phase}</div>"
            f"<div class='node'><b>Regime Strength</b> → {pct_fmt(state.regime_strength)}</div>"
            f"<div class='node'><b>Breadth</b> → {pct_fmt(state.breadth)}</div>"
            f"<div class='node'><b>Fragility</b> → {pct_fmt(state.fragility)}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(prob_bar(state.blended_probs), use_container_width=True, hide_index=True)
        c1, c2 = st.columns([1.25, 1.0])
        with c1:
            with st.expander("Top / Bottom ladder", expanded=True):
                ladder = pd.DataFrame([
                    ["Provisional top", pct_fmt(state.top_score)],
                    ["Higher-top / blow-off", pct_fmt(state.higher_top_risk)],
                    ["Provisional bottom", pct_fmt(state.bottom_score)],
                    ["Lower-bottom / capitulation", pct_fmt(state.lower_bottom_risk)],
                ], columns=["Ladder", "Score"])
                st.dataframe(ladder, use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Risk Engine Snapshot**")
            st.dataframe(risk_snapshot(state, shocks, float(fear_greed)), use_container_width=True, hide_index=True)
            st.markdown("**Event Watch**")
            ev = pd.DataFrame([(n, d.strftime("%Y-%m-%d"), f"{max((d.normalize()-pd.Timestamp.utcnow().tz_localize(None).normalize()).days,0)}d") for n,d in next_release_watch()], columns=["Event","Date","In"])
            st.dataframe(ev, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with next_tab:
        st.markdown("<div class='card'><div class='section-title'>NEXT MAP</div>", unsafe_allow_html=True)
        _path_target, _path_status, _path_conf, _path_fail = path_status(state)
        st.markdown(
            f"<div class='tree'>"
            f"<div class='node'><b>Stay Probability</b> → {pct_fmt(state.stay_probability)}</div>"
            f"<div class='node'><b>Transition Pressure</b> → {pct_fmt(state.transition_pressure)}</div>"
            f"<div class='node'><b>Most Likely Next</b> → {top1(state.next_phase_probs)}</div>"
            f"<div class='node'><b>Path to Next Q</b> → {_path_target}</div>"
            f"<div class='node'><b>Status</b> → {_path_status} &nbsp; {pill(_path_conf)} {pill('Failure ' + _path_fail)}</div>"
            f"<div class='node'><b>Transition Conviction</b> → {pct_fmt(state.transition_conviction)}</div>"
            f"<div class='node'><b>Entry Quality</b> → {timing['Entry Quality']}</div>"
            f"<div class='node'><b>Rotation Timing</b> → {timing['Rotation Timing']}</div>"
            f"<div class='node'><b>Hold Bias</b> → {timing['Hold Bias']}</div>"
            f"<div class='node'><b>Invalidation Window</b> → {timing['Invalidation Window']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(prob_bar(state.next_phase_probs), use_container_width=True, hide_index=True)
        _next_sorted = sorted(state.next_phase_probs.items(), key=lambda kv: kv[1], reverse=True)
        _tree = pd.DataFrame([
            ["Most likely", f"{state.current_phase} → {_next_sorted[0][0]}", pct_fmt(_next_sorted[0][1])],
            ["Alt 1", f"{state.current_phase} → {_next_sorted[1][0]}", pct_fmt(_next_sorted[1][1])],
            ["Alt 2", f"{state.current_phase} → {_next_sorted[2][0]}", pct_fmt(_next_sorted[2][1])],
        ], columns=["Route", "Path", "Weight"])
        st.markdown("**Transition Tree mini**")
        st.dataframe(_tree, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with playbook_tab:
        st.markdown("<div class='card'><div class='section-title'>PLAYBOOK MAP</div>", unsafe_allow_html=True)
        _nw, _nl = next_playbook_hint(state)
        ctop1, ctop2 = st.columns(2)
        with ctop1:
            st.markdown("**Current vs Next Playbook mini**")
            _curr = playbook.get('US Stocks', next(iter(playbook.values())))
            mini = pd.DataFrame([
                ["Current winners", ", ".join(_curr['Winners'][:3])],
                ["Current losers", ", ".join(_curr['Losers'][:3])],
                ["Next winners", ", ".join(_nw[:3])],
                ["Next losers", ", ".join(_nl[:3])],
            ], columns=["Lens", "Read"])
            st.dataframe(mini, use_container_width=True, hide_index=True)
        with ctop2:
            st.markdown(f"**Positioning Posture**: {positioning_posture}")
            st.markdown("**Invalidation mini-box**")
            inv = pd.DataFrame([
                ["Growth breadth improves sharply", "Reduce defensive / slowdown bias"],
                ["Inflation re-accelerates", "Upgrade stagflation / USD / gold tilt"],
                ["Shock fades quickly", "Trim hedges and shorten override state"],
            ], columns=["Trigger", "Action"])
            st.dataframe(inv, use_container_width=True, hide_index=True)
        play_tabs = st.tabs(list(playbook.keys()))
        for tab, (asset_class, mapping) in zip(play_tabs, playbook.items()):
            with tab:
                cA, cB = st.columns(2)
                with cA:
                    st.markdown("**Winners**")
                    for x in mapping['Winners']:
                        st.markdown(f"- {x}")
                with cB:
                    st.markdown("**Losers**")
                    for x in mapping['Losers']:
                        st.markdown(f"- {x}")
        st.markdown("</div>", unsafe_allow_html=True)

with right:
    rel_tab, shock_tab, notes_tab = st.tabs(["Relative", "Shocks / What-If", "Notes"])

    with rel_tab:
        st.markdown("<div class='card'><div class='section-title'>RELATIVE</div>", unsafe_allow_html=True)
        rel_rows = []
        for k, v in relative_detail.items():
            rel_rows.append([k, v["Direction"], v["Strength"], v["State"], v["Quality"], v["Sustainability"], v["Confirmation"]])
        render_compact_matrix(
            "RELATIVE MAP",
            rel_rows,
            ["Relative Lens", "Direction", "Strength", "State", "Quality", "Sustainability", "Confirmation"],
            lens_width="26%",
        )
        sr_rows = []
        for k, v in size_rotation.items():
            sr_rows.append([k, v["Direction"], v["Strength"], v["State"], v["Quality"], v["Sustainability"], v["Confirmation"]])
        render_compact_matrix(
            "SIZE ROTATION",
            sr_rows,
            ["Rotation Lens", "Direction", "Strength", "State", "Quality", "Sustainability", "Confirmation"],
            lens_width="30%",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with shock_tab:
        st.markdown("<div class='card'><div class='section-title'>SHOCKS / WHAT-IF MAP</div>", unsafe_allow_html=True)
        s1, s2, s3 = st.tabs(["Overlay", "Transmission / Correlation", "Crash / Recovery"])
        with s1:
            st.markdown(f"**Shock Status**: {base_override_status}")
            for k, (sev, desc) in shocks.items():
                color = "#22c55e" if sev == "Low" else ("#f59e0b" if sev == "Medium" else "#ef4444")
                st.markdown(f"{pill(f'{k}: {sev}', color)} <span class='muted'>{desc}</span>", unsafe_allow_html=True)
            st.markdown("**Scenario overrides**")
            for item in what_if_cases:
                with st.expander(f"{item['Scenario']} • {item['Probability']}", expanded=False):
                    st.markdown(f"**Impact**: {item['Impact']}")
                    st.markdown(f"**Trigger**: {item['Trigger']}")
        with s2:
            trans_df = pd.DataFrame(WHAT_IF_SCENARIO_MATRIX, columns=["Scenario", "Read", "Prefer", "Avoid"])
            st.markdown("**Transmission / What-If scenarios**")
            st.dataframe(trans_df, use_container_width=True, hide_index=True)
            st.markdown("**Divergence rules**")
            st.dataframe(pd.DataFrame(DIVERGENCE_RULES, columns=["Condition", "Interpretation"]), use_container_width=True, hide_index=True)
            st.markdown("**Correlation priors**")
            st.dataframe(pd.DataFrame(CORRELATION_TRANSMISSION_PRIORS, columns=["Transmission", "Strength", "Why"]), use_container_width=True, hide_index=True)
        with s3:
            st.markdown("**Crash types**")
            st.dataframe(pd.DataFrame(CRASH_TYPES, columns=["Crash type", "Read"]), use_container_width=True, hide_index=True)
            st.markdown("**Crash recovery order**")
            st.dataframe(pd.DataFrame(CRASH_RECOVERY_ORDER, columns=["Crash family", "Recovery order"]), use_container_width=True, hide_index=True)
            st.markdown("**False recovery / second-leg map**")
            st.dataframe(pd.DataFrame(FALSE_RECOVERY_MAP, columns=["Flag", "Meaning"]), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with notes_tab:
        st.markdown("<div class='card'><div class='section-title'>MODEL NOTES</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
- **Core model actually used**: `{CORE_NAME}`
- **Final system shell**: `{APP_NAME}`
- **Reading order**: **Current → Next → Playbook → Relative → Shocks / What-If**
- **Live FRED** is used for the core macro backbone.
- **Fear & Greed** and **IWM blow-off** are used only as **timing / top-bottom overlays**, not as core phase inputs.
- **Correlation** is preserved inside **Transmission / Correlation** so it is easier to read instead of being scattered.
- **IHSG small / 2nd-liner rotation** is still **proxy-based** for now, so read it as a local-beta / risk appetite sleeve, not a perfect official small-vs-big index split.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)
