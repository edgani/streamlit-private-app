from __future__ import annotations

import math
from dataclasses import dataclass
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

FRED_SERIES = {
    "PAYEMS": "PAYEMS",
    "UNRATE": "UNRATE",
    "ICSA": "ICSA",
    "INDPRO": "INDPRO",
    "RSAFS": "RSAFS",
    "CPIAUCSL": "CPIAUCSL",
    "CPILFESL": "CPILFESL",
    "PPIACO": "PPIACO",
    "HOUST": "HOUST",
    "PERMIT": "PERMIT",
    "FEDFUNDS": "FEDFUNDS",
    "DGS2": "DGS2",
    "DGS10": "DGS10",
    "T5YIE": "T5YIE",
    "BAMLH0A0HYM2": "BAMLH0A0HYM2",
    "NFCI": "NFCI",
    "SAHMREALTIME": "SAHMREALTIME",
    "RECPROUSM156N": "RECPROUSM156N",
    "WALCL": "WALCL",
    "DCOILWTICO": "DCOILWTICO",
    "DTWEXBGS": "DTWEXBGS",
}

YF_SYMBOLS = ["SPY", "QQQ", "IWM", "EEM", "EIDO", "BTC-USD", "ETH-USD", "SOL-USD"]

WHAT_IF_SCENARIO_MATRIX = [
    ["Growth re-acceleration", "if growth breadth broadens again", "beta / cyclicals / small caps", "pure defensives / too fearful"],
    ["Disinflationary slowdown", "growth cools but inflation cools too", "duration / quality / gold", "deep cyclicals"],
    ["Stagflation fork", "growth weak tapi inflasi ngeyel", "USD / gold / selective energy", "long-duration beta"],
    ["Bottoming / recovery", "stress reda terus breadth mulai pulih", "beta / cyclicals / EM", "defensive overcrowding"],
]

DIVERGENCE_RULES = [
    ["US weak while IHSG / Indonesia stays strong", "regional decoupling; check commodity support and FX"],
    ["Crypto strong tapi equities males", "liquidity selective / risk-on belum broad"],
    ["Bonds ga rally padahal slowdown", "inflation persistence / supply shock / term premium naik"],
    ["Gold dan USD sama-sama strong", "stress / stagflation / geopolitical hedge demand"],
]

CORRELATION_TRANSMISSION_PRIORS = [
    ["Oil naik → inflation tail", "high", "bias ke stagflation / energy-sensitive names"],
    ["USD naik → EM ketekan", "high", "financial conditions global jadi lebih tight"],
    ["Real yields down → growth/crypto relief", "fairly high", "helps duration-sensitive assets when stress is not dominant"],
    ["Fear & Greed extreme + IWM ngebut", "medium", "sering jadi tanda beta chase / rawan kecapean"],
]

CRASH_TYPES = [
    ["Liquidity accident", "de-risking cepet; breadth dan beta biasanya jebol duluan"],
    ["Growth scare", "cyclicals rollover, duration suka stable duluan"],
    ["Inflation shock", "duration struggles; USD / gold / selective commodities can work"],
    ["Geopolitical shock", "oil / gold / USD can move differently than textbook"],
]

CRASH_RECOVERY_ORDER = [
    ["Liquidity accident", "policy / duration → quality → broad beta"],
    ["Growth scare", "duration / defensives → quality cyclicals → broad beta"],
    ["Inflation shock", "USD / gold → selective cyclicals → duration belakangan"],
    ["Geopolitical shock", "hedges dulu, baru normalisasi"],
]

FALSE_RECOVERY_MAP = [
    ["Bottoming without breadth", "only a provisional bottom; lower-bottom risk is still alive"],
    ["Soft-landing trap", "looks safe for a bit, then growth slips again"],
    ["Re-acceleration fakeout", "risk-on mekar sebentar doang tanpa macro breadth"],
    ["Second-leg / double-dip", "recovery fails and price rolls lower again"],
]

WATCH_EVENTS = ["CPI", "NFP", "FOMC", "PCE", "Jobless Claims", "Retail Sales"]


@dataclass
class PhaseState:
    monthly_probs: Dict[str, float]
    quarterly_probs: Dict[str, float]
    blended_probs: Dict[str, float]
    current_phase: str
    next_phase_probs: Dict[str, float]
    confidence: float
    agreement: float
    ambiguity: float
    sub_phase: str
    validity: str
    phase_strength: float
    breadth: float
    fragility: float
    transition_pressure: float
    stay_probability: float
    top_score: float
    bottom_score: float
    higher_top_risk: float
    lower_bottom_risk: float


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def pct_fmt(x: float) -> str:
    return f"{100*x:.1f}%"


def top1(pm: Dict[str, float]) -> str:
    return max(pm, key=pm.get)


def top2(pm: Dict[str, float]) -> List[str]:
    return sorted(pm, key=pm.get, reverse=True)[:2]


def norm_pm(pm: Dict[str, float]) -> Dict[str, float]:
    s = sum(pm.values())
    return {k: (v / s if s > 0 else 0.25) for k, v in pm.items()}


def safe_z(series: pd.Series, window: int = 24) -> float:
    s = series.dropna().tail(window)
    if len(s) < max(8, window // 3):
        return 0.0
    sd = float(s.std(ddof=0))
    if not np.isfinite(sd) or sd < 1e-12:
        return 0.0
    return float((s.iloc[-1] - s.mean()) / sd)


def pct_z(series: pd.Series, periods: int, window: int) -> float:
    return safe_z(series.pct_change(periods).replace([np.inf, -np.inf], np.nan), window)


def diff_z(series: pd.Series, periods: int, window: int) -> float:
    return safe_z(series.diff(periods), window)


def yoy_z(series: pd.Series, window: int = 36) -> float:
    return safe_z(series.pct_change(12).replace([np.inf, -np.inf], np.nan), window)


def pct(x: float) -> str:
    return f"{100*x:.0f}%"


def q_label(q: str) -> str:
    return {
        "Q1": "Q1 — growth up, inflation down",
        "Q2": "Q2 — growth up, inflation up",
        "Q3": "Q3 — growth down, inflation up",
        "Q4": "Q4 — growth down, inflation down",
    }.get(q, q)


def text_band(x: float, bands: Tuple[float, float] = (0.45, 0.70), words=("weak", "medium", "strong")) -> str:
    if x < bands[0]:
        return words[0]
    if x < bands[1]:
        return words[1]
    return words[2]


def conf_text(x: float) -> str:
    return "clear enough" if x >= 0.70 else ("fairly clear" if x >= 0.50 else "mixed")


def frag_text(x: float) -> str:
    return "stable" if x < 0.35 else ("starting to crack" if x < 0.65 else "fragile")


def timing_text(x: float) -> str:
    return "low" if x < 0.35 else ("building" if x < 0.65 else "fairly high")


def quality_text(score: float, froth: float) -> str:
    if froth > 0.72:
        return "frothy / getting tired"
    if froth > 0.52:
        return "narrow / a bit speculative"
    if score > 0.58:
        return "healthy"
    return "mixed"


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_fred_series(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(r.text))
    df.columns = ["DATE", series_id]
    df["DATE"] = pd.to_datetime(df["DATE"])
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    return df.set_index("DATE")[series_id].dropna()


@st.cache_data(ttl=3 * 60 * 60, show_spinner=False)
def fetch_all_fred(series_map: Dict[str, str]) -> pd.DataFrame:
    out = {}
    for k, v in series_map.items():
        try:
            out[k] = fetch_fred_series(v)
        except Exception:
            continue
    return pd.concat(out, axis=1).sort_index() if out else pd.DataFrame()


@st.cache_data(ttl=2 * 60 * 60, show_spinner=False)
def fetch_prices(symbols: List[str], period: str) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    try:
        data = yf.download(symbols, period=period, auto_adjust=True, progress=False, threads=False)
        if isinstance(data.columns, pd.MultiIndex):
            return data["Close"].copy().sort_index()
        if "Close" in data.columns:
            return data[["Close"]].rename(columns={"Close": symbols[0]}).sort_index()
    except Exception:
        pass
    return pd.DataFrame()


TRANSITION_BASE = np.array([
    [0.62, 0.22, 0.05, 0.11],
    [0.18, 0.58, 0.18, 0.06],
    [0.06, 0.18, 0.58, 0.18],
    [0.20, 0.05, 0.18, 0.57],
], dtype=float)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_fear_greed() -> Optional[float]:
    # daily-ish lightweight read from alternative.me API (crypto oriented but decent risk sentiment overlay)
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=20)
        r.raise_for_status()
        js = r.json()
        return float(js["data"][0]["value"])
    except Exception:
        return None


def build_phase_state(df: pd.DataFrame, fear_greed: Optional[float], iwm_blowoff: float) -> PhaseState:
    g_m_parts = [
        pct_z(df["PAYEMS"], 3, 24) if "PAYEMS" in df else 0.0,
        pct_z(df["INDPRO"], 3, 24) if "INDPRO" in df else 0.0,
        pct_z(df["RSAFS"], 3, 24) if "RSAFS" in df else 0.0,
        -diff_z(df["UNRATE"], 3, 24) if "UNRATE" in df else 0.0,
        -pct_z(df["ICSA"], 4, 24) if "ICSA" in df else 0.0,
        pct_z(df["HOUST"], 3, 24) if "HOUST" in df else 0.0,
        pct_z(df["PERMIT"], 3, 24) if "PERMIT" in df else 0.0,
    ]
    i_m_parts = [
        yoy_z(df["CPIAUCSL"]) if "CPIAUCSL" in df else 0.0,
        yoy_z(df["CPILFESL"]) if "CPILFESL" in df else 0.0,
        yoy_z(df["PPIACO"]) if "PPIACO" in df else 0.0,
        diff_z(df["T5YIE"], 3, 24) if "T5YIE" in df else 0.0,
    ]
    stress_parts = [
        diff_z(df["SAHMREALTIME"], 1, 24) if "SAHMREALTIME" in df else 0.0,
        diff_z(df["RECPROUSM156N"], 1, 24) if "RECPROUSM156N" in df else 0.0,
        diff_z(df["BAMLH0A0HYM2"], 4, 24) if "BAMLH0A0HYM2" in df else 0.0,
        diff_z(df["NFCI"], 4, 24) if "NFCI" in df else 0.0,
    ]
    pol_parts = [
        diff_z(df["FEDFUNDS"], 3, 24) if "FEDFUNDS" in df else 0.0,
        diff_z(df["DGS2"], 4, 24) if "DGS2" in df else 0.0,
        diff_z(df["DTWEXBGS"], 4, 24) if "DTWEXBGS" in df else 0.0,
    ]

    g_q_parts = [
        pct_z(df["PAYEMS"], 6, 60) if "PAYEMS" in df else 0.0,
        pct_z(df["INDPRO"], 6, 60) if "INDPRO" in df else 0.0,
        pct_z(df["RSAFS"], 6, 60) if "RSAFS" in df else 0.0,
        -diff_z(df["UNRATE"], 6, 60) if "UNRATE" in df else 0.0,
        pct_z(df["PERMIT"], 6, 60) if "PERMIT" in df else 0.0,
    ]
    i_q_parts = [
        yoy_z(df["CPIAUCSL"], 60) if "CPIAUCSL" in df else 0.0,
        yoy_z(df["CPILFESL"], 60) if "CPILFESL" in df else 0.0,
        yoy_z(df["PPIACO"], 60) if "PPIACO" in df else 0.0,
    ]
    stress_q_parts = [
        diff_z(df["SAHMREALTIME"], 3, 60) if "SAHMREALTIME" in df else 0.0,
        diff_z(df["RECPROUSM156N"], 3, 60) if "RECPROUSM156N" in df else 0.0,
        diff_z(df["NFCI"], 8, 60) if "NFCI" in df else 0.0,
    ]
    pol_q_parts = [
        diff_z(df["FEDFUNDS"], 6, 60) if "FEDFUNDS" in df else 0.0,
        diff_z(df["DGS2"], 8, 60) if "DGS2" in df else 0.0,
        diff_z(df["DTWEXBGS"], 8, 60) if "DTWEXBGS" in df else 0.0,
    ]

    g_m, i_m = float(np.mean(g_m_parts)), float(np.mean(i_m_parts))
    stress_m, pol_m = float(np.mean(stress_parts)), float(np.mean(pol_parts))
    g_q, i_q = float(np.mean(g_q_parts)), float(np.mean(i_q_parts))
    stress_q, pol_q = float(np.mean(stress_q_parts)), float(np.mean(pol_q_parts))

    g_up_m = sigmoid(1.40 * g_m - 0.42 * stress_m - 0.10 * pol_m)
    i_up_m = sigmoid(1.25 * i_m + 0.08 * pol_m)
    g_up_q = sigmoid(1.18 * g_q - 0.35 * stress_q - 0.08 * pol_q)
    i_up_q = sigmoid(1.10 * i_q + 0.05 * pol_q)

    monthly = norm_pm({"Q1": g_up_m * (1 - i_up_m), "Q2": g_up_m * i_up_m, "Q3": (1 - g_up_m) * i_up_m, "Q4": (1 - g_up_m) * (1 - i_up_m)})
    quarterly = norm_pm({"Q1": g_up_q * (1 - i_up_q), "Q2": g_up_q * i_up_q, "Q3": (1 - g_up_q) * i_up_q, "Q4": (1 - g_up_q) * (1 - i_up_q)})
    blended = norm_pm({q: 0.35 * monthly[q] + 0.65 * quarterly[q] for q in ["Q1", "Q2", "Q3", "Q4"]})

    m_top, q_top = top1(monthly), top1(quarterly)
    agreement = 1.0 if m_top == q_top else 0.55
    vals = sorted(blended.values(), reverse=True)
    ambiguity = 1.0 - (vals[0] - vals[1])
    confidence = clamp(0.55 * agreement + 0.45 * (1 - ambiguity))
    current_phase = top1(blended)

    phase_strength = clamp(vals[0] - vals[-1])
    breadth = clamp((sum(x > 0 for x in g_m_parts) / len(g_m_parts) + sum(x > 0 for x in i_m_parts) / len(i_m_parts)) / 2)
    fragility = clamp(0.45 * ambiguity + 0.35 * max(stress_m, 0) + 0.20 * max(0, 0.5 - breadth))
    validity = "stable" if confidence > 0.72 and phase_strength > 0.35 else ("starting to crack" if confidence > 0.55 else "a bit shaky")

    if current_phase == "Q2":
        sub_phase = "early reflation" if g_m > 0.25 and stress_m < 0 else ("late reflation / topping" if stress_m > 0.2 else "mid reflation")
    elif current_phase == "Q3":
        sub_phase = "stagflationary slowdown" if i_m > 0.15 else "late growth rollover"
    elif current_phase == "Q4":
        sub_phase = "deflationary slowdown" if stress_m > 0.1 else "bottoming attempt"
    else:
        sub_phase = "goldilocks / recovery" if stress_m < 0 else "early recovery"

    curr_idx = ["Q1", "Q2", "Q3", "Q4"].index(current_phase)
    trans = TRANSITION_BASE.copy()
    if stress_m > 0.25:
        if current_phase in ["Q1", "Q2"]:
            trans[curr_idx, 2] += 0.05
        if current_phase in ["Q1", "Q2", "Q3"]:
            trans[curr_idx, 3] += 0.04
    if g_m > 0.30 and i_m < 0:
        trans[curr_idx, 0] += 0.05
    if i_m > 0.25 and g_m < 0:
        trans[curr_idx, 2] += 0.05
    trans = trans / trans.sum(axis=1, keepdims=True)
    curr_vec = np.array([blended[q] for q in ["Q1", "Q2", "Q3", "Q4"]], dtype=float)
    next_vec = curr_vec @ trans
    next_phase_probs = norm_pm({q: float(v) for q, v in zip(["Q1", "Q2", "Q3", "Q4"], next_vec)})
    stay_probability = float(next_phase_probs[current_phase])
    transition_pressure = clamp(1.0 - stay_probability)

    fg_greed = fg_fear = 0.0
    if fear_greed is not None:
        fg = float(fear_greed)
        fg_greed = clamp((fg - 70.0) / 30.0)
        fg_fear = clamp((30.0 - fg) / 30.0)
    top_macro = clamp(0.55 * max(stress_m, 0.0) + 0.45 * transition_pressure)
    bottom_macro = clamp(0.50 * max(-g_m, 0.0) + 0.50 * max(stress_m, 0.0))
    top_score = clamp(0.72 * top_macro + 0.08 * fg_greed + 0.08 * iwm_blowoff + 0.12 * max(0.0, transition_pressure - 0.40))
    bottom_score = clamp(0.78 * bottom_macro + 0.10 * fg_fear + 0.12 * max(0.0, stress_m))
    higher_top_risk = clamp(0.55 * top_score + 0.25 * fg_greed + 0.20 * iwm_blowoff)
    lower_bottom_risk = clamp(0.65 * bottom_score + 0.35 * fg_fear)

    # hysteresis ringan
    if vals[0] - vals[1] < 0.04 and current_phase != q_top:
        current_phase = q_top

    return PhaseState(monthly, quarterly, blended, current_phase, next_phase_probs, confidence, agreement, ambiguity, sub_phase, validity, phase_strength, breadth, fragility, transition_pressure, stay_probability, top_score, bottom_score, higher_top_risk, lower_bottom_risk)


def timing_engine(state: PhaseState) -> Dict[str, str]:
    entry = "good" if state.confidence > 0.72 and state.transition_pressure < 0.35 else ("okay" if state.confidence > 0.55 else "do not get too aggressive")
    if state.top_score > 0.65:
        rotation = "late / do not chase too hard"
    elif state.bottom_score > 0.60:
        rotation = "bottoming / probe lightly"
    elif state.transition_pressure > 0.45:
        rotation = "in transition / review more often"
    else:
        rotation = "still follow the base playbook"
    hold = "can stretch longer" if state.stay_probability > 0.68 else ("medium" if state.stay_probability > 0.50 else "shorter")
    inval = "tight" if state.validity != "stable" or state.transition_pressure > 0.42 else "normal"
    return {"Entry Quality": entry, "Rotation Timing": rotation, "Hold Bias": hold, "Invalidation Window": inval}


FAMILY_SCORE_BY_QUAD = {
    "Q1": {"duration": 0.00, "usd": -0.35, "gold": -0.15, "beta": 0.80, "cyclical": 0.70},
    "Q2": {"duration": -0.70, "usd": 0.00, "gold": 0.05, "beta": 0.65, "cyclical": 0.90},
    "Q3": {"duration": 0.20, "usd": 0.70, "gold": 0.85, "beta": -0.75, "cyclical": -0.80},
    "Q4": {"duration": 0.90, "usd": 0.55, "gold": 0.35, "beta": -0.55, "cyclical": -0.60},
}
ASSET_TRANSLATION = {
    "US Stocks": {"duration": ["long-duration growth", "quality defensives"], "usd": ["US defensives", "dollar beneficiaries"], "gold": ["gold miners", "precious-metals sensitivity"], "beta": ["small caps", "high beta / momentum"], "cyclical": ["industrials", "materials / cyclicals"]},
    "Futures / Commodities": {"duration": ["Treasury longs"], "usd": ["dollar strength basket"], "gold": ["gold / precious metals"], "beta": ["equity index beta"], "cyclical": ["oil / industrial metals / reflation"]},
    "Forex": {"duration": ["lower-yield / relief FX"], "usd": ["USD strength"], "gold": ["safe-haven FX mix"], "beta": ["risk-on FX"], "cyclical": ["commodity FX"]},
    "Crypto": {"duration": ["BTC on falling real yields"], "usd": ["USD headwind to alts"], "gold": ["BTC as alt-safe-beta hybrid"], "beta": ["high beta alts"], "cyclical": ["risk-on rotation into beta"]},
    "IHSG": {"duration": ["rate-sensitive defensives"], "usd": ["rupiah-sensitive losers / exporters winners"], "gold": ["gold-linked / resource names"], "beta": ["domestic beta / property / banks"], "cyclical": ["coal, nickel, commodity cyclicals"]},
}


def family_scores(state: PhaseState) -> Dict[str, float]:
    fam = {k: 0.0 for k in ["duration", "usd", "gold", "beta", "cyclical"]}
    for q, p in state.blended_probs.items():
        for f, s in FAMILY_SCORE_BY_QUAD[q].items():
            fam[f] += 0.70 * p * s
    for q, p in state.next_phase_probs.items():
        for f, s in FAMILY_SCORE_BY_QUAD[q].items():
            fam[f] += 0.30 * p * s
    if state.top_score > 0.60:
        fam["beta"] -= 0.12; fam["cyclical"] -= 0.08; fam["gold"] += 0.06; fam["usd"] += 0.04
    if state.bottom_score > 0.60:
        fam["beta"] += 0.08; fam["cyclical"] += 0.08; fam["duration"] += 0.03
    return fam


def build_playbook(state: PhaseState) -> Dict[str, Dict[str, List[str]]]:
    fam = family_scores(state)
    ordered = sorted(fam, key=fam.get, reverse=True)
    out = {}
    for asset_class, mapping in ASSET_TRANSLATION.items():
        out[asset_class] = {"Winners": mapping[ordered[0]] + mapping[ordered[1]][:1], "Losers": mapping[ordered[-1]] + mapping[ordered[-2]][:1]}
    return out


def last_change(prices: pd.DataFrame, sym: str, lookback: int = 21) -> float:
    if sym not in prices.columns:
        return np.nan
    s = prices[sym].dropna()
    if len(s) < lookback + 1:
        return np.nan
    return float(s.iloc[-1] / s.iloc[-lookback-1] - 1)


def build_rel_card(diff: float, froth: float = 0.0, confirm_strength: float = 0.5) -> Dict[str, str]:
    strength_num = clamp(0.5 + abs(diff) * 4)
    direction = "balanced / imbang" if abs(diff) < 0.02 else ("yang kiri lebih strong" if diff > 0 else "yang kanan lebih strong")
    state = "starting" if abs(diff) < 0.03 else ("building" if abs(diff) < 0.07 else ("stable" if froth < 0.55 else "peaking"))
    sustain = "still okay" if froth < 0.45 and abs(diff) > 0.03 else ("starting to fade" if froth < 0.70 else "at risk / overheated")
    if confirm_strength >= 0.7:
        confirmation = "confirmed"
    elif confirm_strength >= 0.45:
        confirmation = "mixed"
    else:
        confirmation = "not confirmed"
    return {"direction": direction, "strength": text_band(strength_num), "strength_num": pct(strength_num), "state": state, "quality": quality_text(strength_num, froth), "sustain": sustain, "confirmation": confirmation}


def relative_engine(prices: pd.DataFrame, df: pd.DataFrame, state: PhaseState, fear_greed: Optional[float]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    spy = last_change(prices, "SPY", 63)
    eem = last_change(prices, "EEM", 63)
    eido = last_change(prices, "EIDO", 63)
    iwm = last_change(prices, "IWM", 21)
    btc = last_change(prices, "BTC-USD", 63)
    eth = last_change(prices, "ETH-USD", 63)
    qqq = last_change(prices, "QQQ", 63)

    usd_z = safe_z(df["DTWEXBGS"], 26) if "DTWEXBGS" in df else 0.0
    oil_z = safe_z(df["DCOILWTICO"], 26) if "DCOILWTICO" in df else 0.0
    fci_z = safe_z(df["NFCI"], 26) if "NFCI" in df else 0.0
    walcl_z = safe_z(df["WALCL"], 26) if "WALCL" in df else 0.0
    fg = 50.0 if fear_greed is None else float(fear_greed)
    froth = clamp(max(fg - 70, 0) / 30.0 + max(iwm if np.isfinite(iwm) else 0, 0) * 2)

    if np.isfinite(spy) and np.isfinite(eem):
        x = build_rel_card(spy - eem, froth * 0.5, clamp(0.5 + max(-usd_z,0)*0.3 + max(-fci_z,0)*0.2))
        x["direction"] = "US stronger" if spy > eem + 0.02 else ("EM stronger" if eem > spy + 0.02 else "Balanced")
    else:
        x = {"direction": "US stronger" if usd_z > 0.5 and fci_z > 0 else ("EM stronger" if usd_z < -0.4 and oil_z > 0 else "Balanced"), "strength": text_band(clamp(abs(usd_z) / 2)), "strength_num": pct(clamp(abs(usd_z) / 2)), "state": "macro fallback", "quality": "mixed", "sustain": "lihat USD + financial conditions", "confirmation": "mixed"}
    out["US vs EM"] = x

    if np.isfinite(eido) and np.isfinite(spy):
        x = build_rel_card(eido - spy, froth * 0.4, clamp(0.45 + max(oil_z,0)*0.2 - max(usd_z,0)*0.15))
        x["direction"] = "IHSG stronger" if eido > spy + 0.02 else ("US stronger" if spy > eido + 0.02 else "Balanced")
    else:
        x = {"direction": "IHSG stronger" if oil_z > 0.4 and usd_z <= 0.2 else ("US stronger" if usd_z > 0.5 else "Balanced"), "strength": text_band(clamp((abs(oil_z) + abs(usd_z))/3)), "strength_num": pct(clamp((abs(oil_z) + abs(usd_z))/3)), "state": "macro fallback", "quality": "mixed", "sustain": "check commodity support + FX", "confirmation": "mixed"}
    out["IHSG vs US"] = x

    if np.isfinite(eido) and np.isfinite(eem):
        x = build_rel_card(eido - eem, froth * 0.35, clamp(0.45 + max(oil_z,0)*0.2 - max(usd_z,0)*0.1))
        x["direction"] = "IHSG stronger" if eido > eem + 0.02 else ("EM stronger" if eem > eido + 0.02 else "Balanced")
    else:
        x = {"direction": "IHSG stronger" if oil_z > 0.4 else ("EM stronger" if usd_z > 0.6 else "Balanced"), "strength": text_band(clamp(abs(oil_z)/2)), "strength_num": pct(clamp(abs(oil_z)/2)), "state": "macro fallback", "quality": "mixed", "sustain": "check commodity support", "confirmation": "mixed"}
    out["IHSG vs EM"] = x

    if np.isfinite(btc) and np.isfinite(qqq):
        x = build_rel_card(btc - qqq, clamp(max(fg-75,0)/25), clamp(0.45 + max(walcl_z,0)*0.2 - max(fci_z,0)*0.2))
        x["direction"] = "Crypto leading" if btc > qqq + 0.05 else ("Liquidity not confirming" if qqq > btc + 0.05 else "Aligned")
    else:
        x = {"direction": "Crypto leading" if walcl_z > 0.4 and fg > 60 else ("Liquidity not confirming" if fci_z > 0.4 or fg < 35 else "Aligned"), "strength": text_band(clamp((abs(walcl_z)+abs(fci_z))/3)), "strength_num": pct(clamp((abs(walcl_z)+abs(fci_z))/3)), "state": "macro fallback", "quality": "mixed", "sustain": "do not push too hard if liquidity is not confirming", "confirmation": "mixed"}
    out["Crypto vs Liquidity"] = x

    # size rotation blocks
    if np.isfinite(iwm) and np.isfinite(last_change(prices, "SPY", 21)):
        x = build_rel_card(iwm - last_change(prices, "SPY", 21), froth)
        x["direction"] = "Small > Big" if iwm > last_change(prices, "SPY", 21) + 0.015 else ("Big > Small" if last_change(prices, "SPY", 21) > iwm + 0.015 else "imbang")
        x["read"] = "early breadth expansion" if x["quality"] == "healthy" and x["state"] in ["starting","building"] else ("late beta chase" if x["quality"].startswith("frothy") or x["state"] == "peaking" else "mixed")
        x["confirmation"] = "confirmed" if spy > 0 and qqq > 0 else "mixed"
        out["US Size Rotation"] = x
    else:
        out["US Size Rotation"] = {"direction": "not reading cleanly", "strength": "medium", "strength_num": pct(0.5), "state": "mixed", "quality": "mixed", "sustain": "check breadth", "confirmation": "not confirmed", "read": "needs cleaner price data"}

    ihsg_froth = clamp(max((fg - 72)/28, 0) + max(oil_z,0)*0.15)
    out["IHSG Size Rotation"] = {"direction": "second liners > big caps" if state.current_phase in ["Q1","Q2"] else "big caps safer", "strength": "medium" if state.current_phase in ["Q1","Q2"] else "weak", "strength_num": pct(0.62 if state.current_phase in ["Q1","Q2"] else 0.38), "state": "building" if state.current_phase == "Q1" else ("peaking" if state.current_phase == "Q2" else "mixed"), "quality": "healthy" if state.current_phase == "Q1" else ("speculative" if state.current_phase == "Q2" else "mixed"), "sustain": "still okay if big caps are not breaking down" if ihsg_froth < 0.55 else "starting to fade / do not overchase", "confirmation": "confirmed" if state.current_phase == "Q1" else ("mixed" if state.current_phase == "Q2" else "not confirmed"), "read": "local risk-on / second-line chase"}

    alt_froth = clamp(max((fg - 75)/25, 0) + max((eth if np.isfinite(eth) else 0) - (btc if np.isfinite(btc) else 0), 0)*2)
    out["Crypto Size Rotation"] = {"direction": "alts > BTC" if np.isfinite(eth) and np.isfinite(btc) and eth > btc + 0.03 else "BTC still leads / safer", "strength": "strong" if alt_froth > 0.7 else ("medium" if alt_froth > 0.45 else "weak"), "strength_num": pct(0.75 if alt_froth > 0.7 else (0.58 if alt_froth > 0.45 else 0.35)), "state": "peaking" if alt_froth > 0.7 else ("building" if state.current_phase in ["Q1","Q2"] else "mixed"), "quality": "frothy / getting tired" if alt_froth > 0.7 else ("healthy" if state.current_phase == "Q1" else "mixed"), "sustain": "low" if alt_froth > 0.7 else ("still okay" if state.current_phase == "Q1" else "be careful"), "confirmation": "confirmed" if np.isfinite(btc) and np.isfinite(eth) and btc > 0 else ("mixed" if state.current_phase == "Q1" else "not confirmed"), "read": "healthy alt expansion" if state.current_phase == "Q1" and alt_froth < 0.6 else "late alt froth / do not overchase"}
    return out


def sentiment_overlay(prices: pd.DataFrame, state: PhaseState, fear_greed: Optional[float]) -> Dict[str, str]:
    iwm_1m = last_change(prices, "IWM", 21)
    if np.isfinite(iwm_1m) and iwm_1m > 0.09 and state.top_score > 0.55:
        iwm_note = "IWM is ripping; this can mean beta chase / blow-off risk"
    elif np.isfinite(iwm_1m) and iwm_1m > 0.04:
        iwm_note = "IWM is still fairly strong, but check whether breadth is still healthy"
    else:
        iwm_note = "IWM is not yet a major top warning"
    fg = 50.0 if fear_greed is None else float(fear_greed)
    fg_note = "greed is elevated" if fg >= 60 else ("fearful / defensive" if fg <= 35 else "still neutral")
    return {"fear_greed": fg_note, "iwm": iwm_note}


def path_to_next(state: PhaseState) -> Dict[str, str]:
    target = f"{state.current_phase} → {top1(state.next_phase_probs)}"
    if top1(state.next_phase_probs) == state.current_phase and state.stay_probability >= 0.62:
        status = "Stable / no clean shift"
    elif state.transition_pressure < 0.28:
        status = "Starting"
    elif state.transition_pressure < 0.45:
        status = "Building"
    elif state.transition_pressure < 0.62:
        status = "Valid"
    else:
        status = "Confirmed"
    return {"target": target, "status": status, "confidence": pct(state.confidence), "fail": pct(1 - state.stay_probability), "note": "this is the most likely path for now, but keep checking the next data triggers"}


def event_watch() -> List[str]:
    return [f"{x} — review again on major releases / daily refresh" for x in WATCH_EVENTS[:4]]


def shock_engine(df: pd.DataFrame, fear_greed: Optional[float], state: PhaseState) -> Dict[str, Tuple[str, str]]:
    def sev(v: float) -> str:
        return "high" if v > 0.67 else ("medium" if v > 0.40 else "low")
    walcl_mom = pct_z(df["WALCL"], 4, 40) if "WALCL" in df else 0.0
    policy = clamp(max(diff_z(df["FEDFUNDS"], 3, 24) if "FEDFUNDS" in df else 0.0, 0.0) * 0.5 + max(diff_z(df["DGS2"], 4, 24) if "DGS2" in df else 0.0, 0.0) * 0.5)
    geo = clamp(max(yoy_z(df["DCOILWTICO"], 24) if "DCOILWTICO" in df else 0.0, 0.0) * 0.6 + (0.4 if state.top_score > 0.55 and (fear_greed or 50) > 70 else 0.0))
    liq = clamp(max(-walcl_mom, 0.0) * 0.7 + max(diff_z(df["NFCI"], 4, 24) if "NFCI" in df else 0.0, 0.0) * 0.3)
    infl = clamp(max(yoy_z(df["CPIAUCSL"], 24) if "CPIAUCSL" in df else 0.0, 0.0) * 0.6 + max(yoy_z(df["PPIACO"], 24) if "PPIACO" in df else 0.0, 0.0) * 0.4)
    growth = clamp(max(-pct_z(df["RSAFS"], 3, 24) if "RSAFS" in df else 0.0, 0.0) * 0.4 + max(diff_z(df["UNRATE"], 3, 24) if "UNRATE" in df else 0.0, 0.0) * 0.6)
    anomaly = clamp(abs(max(state.monthly_probs.values()) - max(state.quarterly_probs.values())) * 0.6 + (0.25 if state.validity != "stable" else 0.0))
    return {
        "Policy Shock": (sev(policy), "Fed / front-end / policy repricing"),
        "Geopolitical Shock": (sev(geo), "oil / war / commodity disruption sensitivity"),
        "Liquidity Shock": (sev(liq), "funding / financial conditions / balance-sheet stress"),
        "Inflation Shock": (sev(infl), "re-acceleration / commodity / pricing pressure"),
        "Growth Shock": (sev(growth), "demand / labor / slowdown deterioration"),
        "Anomaly Flag": (sev(anomaly), "correlation break / data divergence / low transmission clarity"),
    }


def build_what_if(state: PhaseState, shocks: Dict[str, Tuple[str, str]]) -> List[Dict[str, str]]:
    nxt = top2(state.next_phase_probs)
    out = [
        {"Scenario": "Base Case", "Probability": pct_fmt(state.next_phase_probs[nxt[0]]), "Impact": f"Current {state.current_phase} paling mungkin geser ke {nxt[0]}", "Trigger": "no major shock; blended core stays dominant"},
        {"Scenario": "Re-acceleration", "Probability": pct_fmt(state.next_phase_probs.get("Q1", 0.0) if state.current_phase != "Q1" else state.stay_probability), "Impact": "beta / cyclicals / growth-sensitive assets can improve again", "Trigger": "growth breadth membaik, labor stable, inflasi mendingan"},
        {"Scenario": "Stagflation Fork", "Probability": pct_fmt(state.next_phase_probs.get("Q3", 0.0)), "Impact": "gold / USD / selective commodities usually work better", "Trigger": "growth rolls over while inflation / commodity pressure stays sticky"},
        {"Scenario": "Bottoming / Recovery", "Probability": pct_fmt(state.bottom_score * 0.5), "Impact": "can start probing duration + quality; wait for beta confirmation", "Trigger": "bottom ladder improves and lower-bottom risk falls"},
    ]
    if shocks["Geopolitical Shock"][0] in ["medium", "high"]:
        out.append({"Scenario": "War / Oil Shock", "Probability": shocks["Geopolitical Shock"][0], "Impact": "commodity, gold, USD, selective exporters can turn stronger", "Trigger": "energy, shipping, or sanctions escalation"})
    return out


def inject_css() -> None:
    st.markdown("""
    <style>
    .block-container {padding-top: 1.05rem; padding-bottom: 2rem; max-width: 1380px;}
    .hero {padding: 1rem 1rem .8rem 1rem; border-radius: 20px; border: 1px solid rgba(255,255,255,.08); background: linear-gradient(135deg, rgba(30,41,59,.80), rgba(15,23,42,.92)); margin-bottom: 1rem;}
    .card {padding: .95rem 1rem; border-radius: 18px; border: 1px solid rgba(255,255,255,.08); background: rgba(255,255,255,.03); height: 100%;}
    .title {font-size: 1.02rem; font-weight: 700; margin-bottom: .2rem;}
    .big {font-size: 1.18rem; font-weight: 800; margin: .25rem 0 .35rem;}
    .muted {opacity:.78; font-size:.92rem;}
    .pill {display:inline-block; padding:.26rem .62rem; border-radius:999px; background: rgba(255,255,255,.08); margin: 0 .35rem .35rem 0; font-size:.83rem;}
    </style>
    """, unsafe_allow_html=True)


def card(title: str, main: str, sub: str = "", pills: Optional[List[str]] = None) -> None:
    pills = pills or []
    st.markdown(f"<div class='card'><div class='title'>{title}</div><div class='big'>{main}</div><div class='muted'>{sub}</div>{''.join([f'<span class=\"pill\">{p}</span>' for p in pills])}</div>", unsafe_allow_html=True)


inject_css()
st.markdown(f"<div class='hero'><div class='title'>{APP_NAME}</div><div class='big'>macro regime map that stays readable while keeping the math under the hood</div><div class='muted'>core stays on {CORE_NAME}. screen layout stays simple: current, next, playbook, relative, and shocks/what-if.</div></div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    show_details = st.toggle("Show detail numbers", value=False)
    price_period = st.selectbox("Price lookback", ["6mo", "1y", "2y", "3y"], index=2)

macro = fetch_all_fred(FRED_SERIES)
prices = fetch_prices(YF_SYMBOLS, price_period)
fear_greed = fetch_fear_greed()
iwm_1m = last_change(prices, "IWM", 21)
iwm_blowoff = clamp(max(iwm_1m if np.isfinite(iwm_1m) else 0.0, 0.0) / 0.10)
state = build_phase_state(macro, fear_greed, iwm_blowoff)
playbook = build_playbook(state)
rel = relative_engine(prices, macro, state, fear_greed)
sent = sentiment_overlay(prices, state, fear_greed)
path = path_to_next(state)
shocks = shock_engine(macro, fear_greed, state)
whatifs = build_what_if(state, shocks)
timing = timing_engine(state)

c1, c2, c3, c4 = st.columns(4)
with c1:
    card("Current", q_label(state.current_phase), f"main read: {state.sub_phase}", [f"confidence: {pct(state.confidence)}", f"fragility: {pct(state.fragility)}"])
with c2:
    card("Next", q_label(top1(state.next_phase_probs)), f"path ke next Q: {path['status']}", [f"conf: {path['confidence']}", f"failure risk: {path['fail']}"])
with c3:
    us_pick = playbook["US Stocks"]["Winners"][0]
    card("Playbook", us_pick, "ini yang paling masuk akal buat dibaca dulu", [f"phase strength: {pct(state.phase_strength)}", f"breadth: {pct(state.breadth)}"])
with c4:
    card("Top / Bottom", f"Top {pct(state.top_score)} | Bottom {pct(state.bottom_score)}", "reads whether market is getting top-heavy or trying to bottom", [f"higher top risk: {pct(state.higher_top_risk)}", f"lower bottom risk: {pct(state.lower_bottom_risk)}"])

tab_current, tab_next, tab_play, tab_rel, tab_shock = st.tabs(["Current", "Next", "Playbook", "Relative", "Shocks / What-If"])

with tab_current:
    a, b = st.columns([1.15, 1])
    with a:
        st.subheader("What is happening now?")
        st.markdown(f"- **Main phase:** {q_label(state.current_phase)}")
        st.markdown(f"- **Model read:** **{pct(state.confidence)}**")
        st.markdown(f"- **Sub-phase:** **{state.sub_phase}**")
        st.markdown(f"- **Phase strength:** **{pct(state.phase_strength)}**")
        st.markdown(f"- **Breadth:** **{pct(state.breadth)}**")
        st.markdown(f"- **Fragility:** **{pct(state.fragility)}**")
        st.info("Core math still runs in the background; front-end wording is kept cleaner and easier to scan.")
    with b:
        st.subheader("Risk engine snapshot")
        snap = [
            ("Growth stress", pct(clamp(state.transition_pressure * 0.9 + state.top_score * 0.35 - state.bottom_score * 0.2))),
            ("Inflation stress", pct(clamp(1 - state.bottom_score * 0.2 + state.higher_top_risk * 0.35))),
            ("Sentiment stretch", "pretty hot" if (fear_greed or 50) > 70 else ("fearful / defensive" if (fear_greed or 50) < 30 else "still normal")),
            ("Signal quality", "clean" if state.confidence > 0.72 and state.fragility < 0.4 else ("mixed" if state.confidence > 0.5 else "fragile")),
        ]
        for k, v in snap:
            st.markdown(f"- **{k}:** {v}")
        st.subheader("Event watch")
        for e in event_watch():
            st.markdown(f"- {e}")
    if show_details:
        st.caption("Detail probabilities")
        st.write(pd.DataFrame([{**{f'm_{k}': v for k,v in state.monthly_probs.items()}, **{f'q_{k}': v for k,v in state.quarterly_probs.items()}, **{f'b_{k}': v for k,v in state.blended_probs.items()}}]))

with tab_next:
    a, b = st.columns(2)
    with a:
        st.subheader("Path to next Q")
        st.markdown(f"- **Target:** {path['target']}")
        st.markdown(f"- **Status:** {path['status']}")
        st.markdown(f"- **Confidence:** {path['confidence']}")
        st.markdown(f"- **Failure risk:** {path['fail']}")
        st.markdown(f"- **Transition conviction:** **{pct(state.transition_pressure)}**")
        st.markdown(f"- **Note:** {path['note']}")
        st.markdown("### Transition tree mini")
        tp = top2(state.next_phase_probs)
        st.markdown(f"- **Base:** {state.current_phase} → {tp[0]}")
        st.markdown(f"- **Alt 1:** stay in {state.current_phase}")
        st.markdown(f"- **Alt 2:** {state.current_phase} → {tp[1]}")
    with b:
        st.subheader("Timing / turning point")
        st.markdown(f"- **Top risk:** {pct(state.top_score)}")
        st.markdown(f"- **Bottom risk:** {pct(state.bottom_score)}")
        st.markdown(f"- **Higher-top risk:** {pct(state.higher_top_risk)}")
        st.markdown(f"- **Lower-bottom risk:** {pct(state.lower_bottom_risk)}")
        st.markdown(f"- **Entry quality:** {timing['Entry Quality']}")
        st.markdown(f"- **Rotation timing:** {timing['Rotation Timing']}")
        st.markdown(f"- **Hold bias:** {timing['Hold Bias']}")
        st.warning("If top/bottom is still early-stage, do not treat it as final. Real life often still has higher-top or lower-bottom risk.")

with tab_play:
    st.subheader("Current vs next playbook mini")
    cols = st.columns(2)
    next_hint_win, next_hint_avoid = [x.lower() for x in top2(state.next_phase_probs)], [x.lower() for x in top2(state.blended_probs)]
    for idx, label in enumerate(["Playbook now", "If next phase starts to work"]):
        with cols[idx]:
            st.markdown(f"### {label}")
            source = playbook if idx == 0 else playbook
            for asset, picks in source.items():
                st.markdown(f"**{asset}**")
                target = picks["Winners"] if idx == 0 else picks["Winners"]
                for p in target[:3]:
                    st.markdown(f"- {p}")
    posture = "defensive" if state.current_phase in ["Q3","Q4"] else ("balanced" if state.transition_pressure > 0.4 else "aggressive")
    st.markdown(f"**Positioning posture:** {posture}")
    st.subheader("What looks vulnerable")
    for asset, picks in playbook.items():
        st.markdown(f"- **{asset}:** {', '.join(picks['Losers'][:2])}")
    st.subheader("Invalidation mini-box")
    st.markdown("- if inflation reheats, defensive / gold / USD setups can move up in priority")
    st.markdown("- if growth breadth improves fast, do not stay stubbornly defensive")
    st.markdown("- if war / liquidity shock fades quickly, some edges can fade fast")

with tab_rel:
    st.subheader("Relative & size rotation")
    cols = st.columns(2)
    order = ["US vs EM", "IHSG vs US", "IHSG vs EM", "Crypto vs Liquidity", "US Size Rotation", "IHSG Size Rotation", "Crypto Size Rotation"]
    for i, k in enumerate(order):
        v = rel[k]
        with cols[i % 2]:
            st.markdown(f"### {k}")
            st.markdown(f"- **Direction:** {v['direction']}")
            st.markdown(f"- **Strength:** {v['strength_num']} ({v['strength']})")
            st.markdown(f"- **State:** {v['state']}")
            st.markdown(f"- **Quality:** {v['quality']}")
            st.markdown(f"- **Sustainability:** {v['sustain']}")
            if 'confirmation' in v:
                st.markdown(f"- **Confirmation:** {v['confirmation']}")
            if 'read' in v:
                st.markdown(f"- **Read:** {v['read']}")
    st.info("If small caps > big caps, do not auto-label it bullish. Check whether it is healthy breadth or just froth / beta chase.")

with tab_shock:
    st.subheader("Transmission / correlation")
    for row in CORRELATION_TRANSMISSION_PRIORS:
        st.markdown(f"- **{row[0]}** → {row[2]}")
    with st.expander("What-if / divergence / crash map"):
        st.markdown("### What-if")
        for s in WHAT_IF_SCENARIO_MATRIX:
            st.markdown(f"- **{s[0]}**: {s[1]} | prefer: {s[2]} | avoid: {s[3]}")
        st.markdown("### Divergence")
        for d in DIVERGENCE_RULES:
            st.markdown(f"- **{d[0]}**: {d[1]}")
        st.markdown("### Crash / recovery")
        for c in CRASH_TYPES:
            st.markdown(f"- **{c[0]}**: {c[1]}")
        for c in FALSE_RECOVERY_MAP:
            st.markdown(f"- **{c[0]}**: {c[1]}")
    st.subheader("Risk / sentiment / exhaustion")
    st.markdown(f"- **Fear & Greed vibe:** {sent['fear_greed']}")
    st.markdown(f"- **IWM read:** {sent['iwm']}")
    st.markdown("### Shock status")
    for k, v in shocks.items():
        st.markdown(f"- **{k}:** {v[0]} — {v[1]}")
    st.markdown("### Base case vs override")
    active_override = any(v[0] in ["medium", "high"] for v in shocks.values() if "Anomaly" not in v[1])
    st.markdown(f"- **Current mode:** {'Override active' if active_override else 'Base case / watch only'}")
    st.markdown("### Useful what-if paths")
    for w in whatifs:
        st.markdown(f"- **{w['Scenario']}** ({w['Probability']}): {w['Impact']} | trigger: {w['Trigger']}")

st.caption("Note: backend still uses phase/quad logic as the anchor, while the front end stays lighter so it is easier to scan.")
