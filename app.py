from __future__ import annotations

import math
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

st.set_page_config(page_title="Quant Macro Final", layout="wide")

# =========================
# Config
# =========================
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
FRED_API_KEY = st.secrets.get("FRED_API_KEY", "") if hasattr(st, "secrets") else ""

SERIES = {
    "CPI": "CPIAUCSL",
    "CORE_CPI": "CPILFESL",
    "UNRATE": "UNRATE",
    "CLAIMS": "ICSA",
    "INDPRO": "INDPRO",
    "RETAIL": "RSAFS",
    "PAYEMS": "PAYEMS",
    "DGS2": "DGS2",
    "DGS10": "DGS10",
    "OIL": "DCOILWTICO",
    "HY": "BAMLH0A0HYM2",
    "NFCI": "NFCI",
    "SAHM": "SAHMREALTIME",
}

WATCH_EVENTS = [
    "CPI",
    "Jobs",
    "Claims",
    "PCE",
    "Retail Sales",
    "FOMC vibe",
]

QUADS = ["Q1", "Q2", "Q3", "Q4"]
TRANSITION_PRIOR = {
    "Q1": {"Q1": 0.55, "Q2": 0.20, "Q3": 0.05, "Q4": 0.20},
    "Q2": {"Q1": 0.15, "Q2": 0.55, "Q3": 0.22, "Q4": 0.08},
    "Q3": {"Q1": 0.06, "Q2": 0.18, "Q3": 0.56, "Q4": 0.20},
    "Q4": {"Q1": 0.26, "Q2": 0.06, "Q3": 0.18, "Q4": 0.50},
}

# =========================
# Helpers
# =========================
def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def friendly_band(v: float, bands: List[Tuple[float, str]]) -> str:
    for th, label in bands:
        if v <= th:
            return label
    return bands[-1][1]


@st.cache_data(ttl=3600)
def fred_series(series_id: str, limit: int = 360) -> pd.Series:
    params = {
        "series_id": series_id,
        "file_type": "json",
        "sort_order": "asc",
        "limit": limit,
    }
    if FRED_API_KEY:
        params["api_key"] = FRED_API_KEY
    try:
        r = requests.get(FRED_BASE, params=params, timeout=20)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        if not obs:
            return pd.Series(dtype=float)
        idx = pd.to_datetime([x["date"] for x in obs])
        vals = [np.nan if x["value"] == "." else float(x["value"]) for x in obs]
        return pd.Series(vals, index=idx).dropna()
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=3600)
def yf_prices(symbols: List[str], period: str = "1y") -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    if yf is None:
        return out
    try:
        data = yf.download(symbols, period=period, auto_adjust=True, progress=False, group_by="ticker", threads=False)
        if isinstance(data.columns, pd.MultiIndex):
            for sym in symbols:
                try:
                    s = data[sym]["Close"].dropna()
                    if not s.empty:
                        out[sym] = s
                except Exception:
                    pass
        else:
            if "Close" in data and symbols:
                s = data["Close"].dropna()
                if not s.empty:
                    out[symbols[0]] = s
    except Exception:
        pass
    return out


def last_change(series: pd.Series, lookback: int = 21) -> float:
    if series is None or len(series) <= lookback:
        return 0.0
    return float(series.iloc[-1] / series.iloc[-1 - lookback] - 1.0)


def _rolling_z(series: pd.Series, window: int) -> float:
    s = series.dropna()
    if len(s) < max(12, window // 4):
        return 0.0
    tail = s.iloc[-window:] if len(s) >= window else s
    if len(tail) < 12:
        return 0.0
    sd = float(tail.std(ddof=0))
    if not np.isfinite(sd) or sd < 1e-12:
        return 0.0
    return float((tail.iloc[-1] - tail.mean()) / sd)


@dataclass
class CoreState:
    monthly_probs: Dict[str, float]
    quarterly_probs: Dict[str, float]
    blended_probs: Dict[str, float]
    current_q: str
    next_q: str
    confidence: float
    phase_strength: float
    breadth: float
    fragility: float
    transition_pressure: float
    top_score: float
    bottom_score: float
    higher_top_risk: float
    lower_bottom_risk: float
    growth_score: float
    inflation_score: float
    stress_score: float
    path_status: str
    path_conf: float
    path_fail: float
    sub_phase: str
    regime_note: str
    transition_conviction: str


def _probs_from_scores(g: float, i: float, stress: float, polarity: float = 1.0) -> Dict[str, float]:
    g_up = sigmoid(1.30 * g - 0.38 * stress)
    i_up = sigmoid(1.22 * i + 0.08 * polarity)
    probs = {
        "Q1": g_up * (1.0 - i_up),
        "Q2": g_up * i_up,
        "Q3": (1.0 - g_up) * i_up,
        "Q4": (1.0 - g_up) * (1.0 - i_up),
    }
    s = sum(probs.values()) or 1.0
    return {k: v / s for k, v in probs.items()}


def _choose_current_q(blended: Dict[str, float]) -> Tuple[str, float]:
    order = sorted(blended, key=blended.get, reverse=True)
    top1, top2 = order[0], order[1]
    gap = blended[top1] - blended[top2]
    prev_q = st.session_state.get("prev_q")
    prev_probs = st.session_state.get("prev_probs")

    # sticky regime: kalau beda tipis, jangan gampang flip
    if prev_q is not None and prev_q in blended:
        prev_prob = blended[prev_q]
        if top1 != prev_q and (blended[top1] - prev_prob) < 0.07:
            chosen = prev_q
            conf = max(0.01, prev_prob - sorted(blended.values(), reverse=True)[1]) if prev_q == top1 else max(0.01, prev_prob - blended[top1] + 0.03)
            return chosen, clip01(conf * 2.0)

    st.session_state["prev_q"] = top1
    st.session_state["prev_probs"] = blended
    return top1, clip01(gap * 2.0)


def compute_core(data: Dict[str, pd.Series]) -> CoreState:
    growth_cols = ["INDPRO", "RETAIL", "PAYEMS"]
    infl_cols = ["CPI", "CORE_CPI", "OIL"]
    stress_cols = ["HY", "NFCI", "SAHM"]

    g_m = np.mean([_rolling_z(data.get(c, pd.Series(dtype=float)), 12) for c in growth_cols] + [
        -_rolling_z(data.get("UNRATE", pd.Series(dtype=float)), 12),
        -_rolling_z(data.get("CLAIMS", pd.Series(dtype=float)), 12),
    ])
    i_m = np.mean([_rolling_z(data.get(c, pd.Series(dtype=float)), 12) for c in infl_cols])
    s_m = np.mean([_rolling_z(data.get(c, pd.Series(dtype=float)), 12) for c in stress_cols])

    g_q = np.mean([_rolling_z(data.get(c, pd.Series(dtype=float)), 36) for c in growth_cols] + [
        -_rolling_z(data.get("UNRATE", pd.Series(dtype=float)), 36),
        -_rolling_z(data.get("CLAIMS", pd.Series(dtype=float)), 36),
    ])
    i_q = np.mean([_rolling_z(data.get(c, pd.Series(dtype=float)), 36) for c in infl_cols])
    s_q = np.mean([_rolling_z(data.get(c, pd.Series(dtype=float)), 36) for c in stress_cols])

    monthly = _probs_from_scores(g_m, i_m, s_m, polarity=0.0)
    quarterly = _probs_from_scores(g_q, i_q, s_q, polarity=0.0)
    blended = {q: 0.40 * monthly[q] + 0.60 * quarterly[q] for q in QUADS}
    sb = sum(blended.values()) or 1.0
    blended = {k: v / sb for k, v in blended.items()}

    current_q, confidence = _choose_current_q(blended)

    curr_vec = np.array([blended[q] for q in QUADS])
    trans = np.array([[TRANSITION_PRIOR[q1][q2] for q2 in QUADS] for q1 in QUADS])
    next_vec = curr_vec @ trans

    # stay bonus biar nggak liar pindah kalau masih blur
    curr_idx = QUADS.index(current_q)
    stay_bonus = max(0.0, 0.08 - 0.04 * max(s_m, 0.0) + 0.02 * (1.0 - confidence))
    next_vec[curr_idx] += stay_bonus
    next_vec = next_vec / next_vec.sum()
    next_q = QUADS[int(np.argmax(next_vec))]
    if next_q == current_q:
        next_q = QUADS[int(np.argsort(next_vec)[::-1][1])]

    growth_score = clip01(0.5 + 0.20 * (0.5 * g_m + 0.5 * g_q))
    inflation_score = clip01(0.5 + 0.20 * (0.5 * i_m + 0.5 * i_q))
    stress_score = clip01(0.5 + 0.20 * (0.5 * s_m + 0.5 * s_q))

    pos_count = sum(x > 0 for x in [g_m, g_q, i_m, i_q]) + sum(x < 0 for x in [s_m, s_q])
    breadth = clip01(0.30 + 0.10 * pos_count)
    phase_strength = clip01(0.40 + 0.25 * abs(0.5 * g_m + 0.5 * g_q) + 0.18 * abs(0.5 * i_m + 0.5 * i_q))
    fragility = clip01(0.25 + 0.30 * (1.0 - confidence) + 0.22 * stress_score + 0.10 * abs((0.5 * g_m + 0.5 * g_q) - (0.5 * i_m + 0.5 * i_q)))
    transition_pressure = clip01(0.20 + 0.35 * (1.0 - confidence) + 0.20 * fragility + 0.10 * abs(g_m - g_q) + 0.08 * abs(i_m - i_q))

    top_score = clip01(0.08 + 0.22 * (current_q in ["Q2", "Q3"]) + 0.18 * fragility + 0.14 * max(growth_score - 0.55, 0.0) + 0.12 * max(inflation_score - 0.55, 0.0))
    bottom_score = clip01(0.08 + 0.24 * (current_q in ["Q4", "Q1"]) + 0.20 * stress_score + 0.12 * max(0.55 - growth_score, 0.0))
    higher_top_risk = clip01(0.08 + 0.22 * top_score + 0.18 * (1.0 - confidence))
    lower_bottom_risk = clip01(0.08 + 0.22 * bottom_score + 0.18 * (1.0 - confidence))

    if transition_pressure < 0.32:
        path_status = "masih anteng"
    elif transition_pressure < 0.48:
        path_status = "baru mulai"
    elif transition_pressure < 0.64:
        path_status = "lagi kebentuk"
    elif transition_pressure < 0.80:
        path_status = "udah lumayan valid"
    else:
        path_status = "udah kuat banget"

    path_conf = clip01(0.25 + 0.45 * max(next_vec[QUADS.index(next_q)] - curr_vec[curr_idx] + 0.15, 0.0) + 0.20 * confidence)
    path_fail = clip01(0.58 - 0.30 * confidence + 0.18 * fragility)

    if confidence > 0.60 and fragility < 0.40:
        sub_phase = "masih sehat"
    elif confidence > 0.42 and fragility < 0.58:
        sub_phase = "masih jalan tapi mulai rapuh"
    elif top_score > bottom_score and top_score > 0.55:
        sub_phase = "lagi topping / rawan capek"
    elif bottom_score > top_score and bottom_score > 0.55:
        sub_phase = "lagi bottoming / nyoba mantul"
    else:
        sub_phase = "campur aduk / transisi"

    transition_conviction = friendly_band(path_conf, [(0.33, "masih tipis"), (0.66, "lumayan kebaca"), (1.0, "cukup valid")])

    regime_note = {
        "Q1": "growth masih oke, inflasi mendingan",
        "Q2": "growth masih dorong, inflasi juga ikut panas",
        "Q3": "growth mulai loyo tapi inflasi masih ganggu",
        "Q4": "growth sama inflasi sama-sama jinak / dingin",
    }[current_q]

    return CoreState(
        monthly_probs=monthly,
        quarterly_probs=quarterly,
        blended_probs=blended,
        current_q=current_q,
        next_q=next_q,
        confidence=confidence,
        phase_strength=phase_strength,
        breadth=breadth,
        fragility=fragility,
        transition_pressure=transition_pressure,
        top_score=top_score,
        bottom_score=bottom_score,
        higher_top_risk=higher_top_risk,
        lower_bottom_risk=lower_bottom_risk,
        growth_score=growth_score,
        inflation_score=inflation_score,
        stress_score=stress_score,
        path_status=path_status,
        path_conf=path_conf,
        path_fail=path_fail,
        sub_phase=sub_phase,
        regime_note=regime_note,
        transition_conviction=transition_conviction,
    )


# =========================
# View helpers
# =========================
def q_label(q: str) -> str:
    return {
        "Q1": "Q1 / growth oke + inflasi turun",
        "Q2": "Q2 / growth oke + inflasi naik",
        "Q3": "Q3 / growth lemah + inflasi masih tinggi",
        "Q4": "Q4 / growth lemah + inflasi turun",
    }[q]


def level_text(v: float, kind: str) -> str:
    if kind == "conf":
        return friendly_band(v, [(0.33, "masih blur"), (0.66, "lumayan jelas"), (1.0, "cukup clean")])
    if kind == "strength":
        return friendly_band(v, [(0.35, "lemah"), (0.65, "sedang"), (1.0, "kuat")])
    if kind == "fragility":
        return friendly_band(v, [(0.35, "stabil"), (0.65, "mulai rapuh"), (1.0, "rapuh")])
    if kind == "timing":
        return friendly_band(v, [(0.33, "masih awal"), (0.66, "lagi kebentuk"), (1.0, "udah panas")])
    return friendly_band(v, [(0.35, "rendah"), (0.65, "sedang"), (1.0, "tinggi")])


def risk_snapshot(state: CoreState) -> List[Tuple[str, str, str]]:
    return [
        ("Growth", level_text(state.growth_score, "strength"), "dorongan growth sekarang seberapa kuat"),
        ("Inflasi", level_text(state.inflation_score, "strength"), "tekanan inflasi lagi seberapa kenceng"),
        ("Stress", level_text(state.stress_score, "strength"), "stress sistem / funding lagi gimana"),
        ("Fragility", level_text(state.fragility, "fragility"), "fase sekarang gampang goyang atau nggak"),
    ]


def playbook(state: CoreState) -> Dict[str, Dict[str, List[str]]]:
    base = {
        "Q1": {
            "US Stocks": ["quality growth", "semis besar", "consumer yang masih kuat"],
            "Futures / Commodities": ["equity index long lebih enak", "industrial metals tipis-tipis"],
            "Forex": ["FX pro-growth, USD agak jinak"],
            "Crypto": ["BTC dulu, alt kalau breadth ikut"],
            "IHSG": ["big caps sehat", "domestik cyclicals kalau rupiah aman"],
        },
        "Q2": {
            "US Stocks": ["energy", "financials", "cyclicals"],
            "Futures / Commodities": ["oil / commodity beta", "index masih bisa ikut tapi rawan telat"],
            "Forex": ["commodity FX lebih enak"],
            "Crypto": ["risk-on oke, tapi hati-hati kalau udah terlalu panas"],
            "IHSG": ["resources", "second liners kalau flow lokal ikut"],
        },
        "Q3": {
            "US Stocks": ["defensives", "pricing power", "jangan terlalu kejar beta"],
            "Futures / Commodities": ["gold", "oil cuma kalau shock-nya direct"],
            "Forex": ["USD / safety lebih kepake"],
            "Crypto": ["lebih hati-hati, BTC mending daripada alt"],
            "IHSG": ["resource selectif", "defensif", "jangan broad risk-on"],
        },
        "Q4": {
            "US Stocks": ["defensives", "duration-sensitive names", "quality"],
            "Futures / Commodities": ["rates / duration / gold"],
            "Forex": ["USD masih oke, carry pilih-pilih"],
            "Crypto": ["nunggu konfirmasi likuiditas", "jangan buru-buru alt"],
            "IHSG": ["big caps defensif", "nunggu konfirmasi bottoming"],
        },
    }
    avoid = {
        "Q1": ["USD terlalu berat", "bond proxy defensif yang terlalu lambat"],
        "Q2": ["duration panjang", "defensives terlalu dini"],
        "Q3": ["broad beta chase", "small caps yang udah terlalu panas"],
        "Q4": ["cyclicals tanpa konfirmasi", "alt beta tanpa likuiditas"],
    }
    return {"current": base[state.current_q], "next": base[state.next_q], "avoid": {"yang rawan": avoid[state.current_q]}}


def relative_read(prices: Dict[str, pd.Series], state: CoreState) -> Dict[str, Dict[str, str]]:
    def rel(a: str, b: str, fallback_bias: str) -> Dict[str, str]:
        if a in prices and b in prices and len(prices[a]) > 40 and len(prices[b]) > 40:
            diff = last_change(prices[a], 21) - last_change(prices[b], 21)
        else:
            diff = 0.04 if fallback_bias == "a" else (-0.04 if fallback_bias == "b" else 0.0)
        if diff > 0.03:
            direction = f"{a} lebih kuat"
        elif diff < -0.03:
            direction = f"{b} lebih kuat"
        else:
            direction = "masih imbang"
        strength = clip01(0.5 + abs(diff) * 5)
        state_txt = "lagi nambah kuat" if abs(diff) > 0.05 else ("lumayan stabil" if abs(diff) > 0.02 else "masih setengah-setengah")
        sustain = "masih oke" if strength < 0.70 else ("masih bisa lanjut tapi mulai panas" if state.current_q in ["Q1", "Q2"] else "rawan capek")
        return {
            "direction": direction,
            "strength": level_text(strength, "strength"),
            "state": state_txt,
            "sustain": sustain,
        }

    out = {
        "US vs EM": rel("SPY", "EEM", "a"),
        "IHSG vs US": rel("EIDO", "SPY", "b"),
        "IHSG vs EM": rel("EIDO", "EEM", "a"),
    }

    if "BTC-USD" in prices:
        btc = last_change(prices["BTC-USD"], 21)
        liq = 0.04 - (state.stress_score - 0.5) * 0.10
        diff = btc - liq
        out["Crypto vs Liquidity"] = {
            "direction": "crypto keconfirm likuiditas" if diff > 0.03 else "likuiditas belum ngeconfirm crypto",
            "strength": level_text(clip01(0.5 + abs(diff) * 5), "strength"),
            "state": "masih sehat" if diff > 0.03 else "agak maksa / risk-on belum bersih",
            "sustain": "oke" if diff > 0.03 else "hati-hati",
        }
    else:
        out["Crypto vs Liquidity"] = {
            "direction": "likuiditas belum ngeconfirm crypto",
            "strength": "sedang",
            "state": "masih mixed",
            "sustain": "hati-hati",
        }

    out["US Small vs Big"] = rel("IWM", "SPY", "a")
    out["Crypto Alts vs BTC"] = {
        "direction": "alt > BTC" if state.current_q in ["Q1", "Q2"] and state.top_score < 0.6 else "BTC masih leader / lebih aman",
        "strength": "sedang" if state.current_q in ["Q1", "Q2"] else "lemah",
        "state": "healthy breadth" if state.current_q == "Q1" else ("mulai frothy" if state.current_q == "Q2" else "nggak clean"),
        "sustain": "masih oke" if state.current_q == "Q1" else ("jangan terlalu ngebut" if state.current_q == "Q2" else "rendah"),
    }
    return out


def sentiment_overlay(prices: Dict[str, pd.Series], state: CoreState) -> Dict[str, str]:
    iwm = prices.get("IWM")
    ext = last_change(iwm, 21) if iwm is not None and len(iwm) > 60 else 0.0
    if ext > 0.09 and state.top_score > 0.55:
        iwm_note = "IWM lagi ngebut, ini bisa jadi beta chase / rawan blow-off"
    elif ext > 0.04:
        iwm_note = "IWM masih lumayan kuat, tapi cek apakah breadth ikut sehat"
    else:
        iwm_note = "IWM nggak terlalu jadi sinyal top sekarang"
    fear_greed = "greed lumayan tinggi" if state.current_q in ["Q1", "Q2"] else "fear/defensive mode lebih kepake"
    return {"fear_greed": fear_greed, "iwm": iwm_note}


def path_to_next(state: CoreState) -> Dict[str, str]:
    return {
        "target": f"{state.current_q} → {state.next_q}",
        "status": state.path_status,
        "confidence": state.transition_conviction,
        "fail": level_text(state.path_fail, "fragility"),
        "note": "jalur paling mungkin sekarang, tapi tetap cek trigger data berikutnya",
    }


def event_watch() -> List[str]:
    return [f"{x} – cek lagi pas rilis besar / update harian" for x in WATCH_EVENTS[:4]]


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1320px;}
        .hero {padding: 1rem 1rem .9rem 1rem; border-radius: 18px; border: 1px solid rgba(255,255,255,0.08); background: linear-gradient(135deg, rgba(30,41,59,.78), rgba(15,23,42,.94)); margin-bottom: .9rem;}
        .card {padding: .95rem 1rem; border-radius: 16px; border: 1px solid rgba(255,255,255,.08); background: rgba(255,255,255,.03); height:100%;}
        .title {font-size: 1rem; font-weight: 700; margin-bottom: .2rem;}
        .big {font-size: 1.12rem; font-weight: 800; margin:.15rem 0 .25rem;}
        .muted {opacity: .78; font-size: .92rem;}
        .pill {display:inline-block; padding: .24rem .56rem; border-radius:999px; background: rgba(255,255,255,.08); margin: 0 .3rem .3rem 0; font-size:.80rem;}
        .sep {height: .45rem;}
        ul.tight {margin-top:0.2rem;margin-bottom:0.2rem;padding-left:1.1rem}
        </style>
        """,
        unsafe_allow_html=True,
    )


def card(title: str, main: str, sub: str = "", pills: Optional[List[str]] = None) -> None:
    pills = pills or []
    pill_html = "".join([f"<span class='pill'>{p}</span>" for p in pills])
    st.markdown(f"<div class='card'><div class='title'>{title}</div><div class='big'>{main}</div><div class='muted'>{sub}</div>{pill_html}</div>", unsafe_allow_html=True)


# =========================
# App
# =========================
inject_css()

st.markdown(
    "<div class='hero'><div class='title'>QuantFinalV4_Max</div>"
    "<div class='big'>simple, enak dibaca, tapi backend-nya udah dibalikin ke blended core yang lebih stabil</div>"
    "<div class='muted'>intinya: nggak maksa lo baca angka doang. yang keluar di depan bahasa manusia dulu, angka detail opsional kalau mau cek dalemnya.</div></div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Setelan")
    show_details = st.toggle("Buka detail angka", value=False)
    price_period = st.selectbox("Lookback harga", ["6mo", "1y", "2y"], index=1)

macro = {k: fred_series(v) for k, v in SERIES.items()}
prices = yf_prices(["SPY", "EEM", "EIDO", "IWM", "BTC-USD"], period=price_period)
state = compute_core(macro)
book = playbook(state)
rel = relative_read(prices, state)
sent = sentiment_overlay(prices, state)
path = path_to_next(state)

# Hero row
c1, c2, c3, c4 = st.columns(4)
with c1:
    card("Current", q_label(state.current_q), f"inti ceritanya: {state.regime_note}", [f"bacaan: {level_text(state.confidence, 'conf')}", f"sub-phase: {state.sub_phase}"])
with c2:
    card("Next", q_label(state.next_q), f"jalur ke next Q: {path['status']}", [f"conf: {path['confidence']}", f"failure risk: {path['fail']}"])
with c3:
    pick = next(iter(book['current'].get('US Stocks', ['belum ada edge yang clean'])), 'belum ada edge yang clean')
    card("Playbook", pick, "ini yang paling masuk akal buat dibaca dulu", [f"phase strength: {level_text(state.phase_strength, 'strength')}", f"fragility: {level_text(state.fragility, 'fragility')}"])
with c4:
    card("Top / Bottom", f"Top {level_text(state.top_score, 'timing')} | Bottom {level_text(state.bottom_score, 'timing')}", "buat baca lagi rawan topping atau lagi nyoba bottoming", [f"higher top risk: {level_text(state.higher_top_risk, 'strength')}", f"lower bottom risk: {level_text(state.lower_bottom_risk, 'strength')}"])

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

tab_current, tab_next, tab_play, tab_rel, tab_shock = st.tabs(["Current", "Next", "Playbook", "Relative", "Shocks / What-If"])

with tab_current:
    a, b = st.columns([1.15, 1])
    with a:
        st.subheader("Sekarang lagi gimana?")
        st.markdown(f"- **Fase utama:** {q_label(state.current_q)}")
        st.markdown(f"- **Bacaan model:** **{level_text(state.confidence, 'conf')}**")
        st.markdown(f"- **Sub-phase:** **{state.sub_phase}**")
        st.markdown(f"- **Kekuatan fase:** **{level_text(state.phase_strength, 'strength')}**")
        st.markdown(f"- **Breadth:** **{level_text(state.breadth, 'strength')}**")
        st.markdown(f"- **Fragility:** **{level_text(state.fragility, 'fragility')}**")
        st.info("Kalau fase tiba-tiba mau ganti, sekarang harusnya nggak segampang itu flip. Kalau masih mepet, dia bakal kebaca lebih kayak transisi daripada loncat brutal.")
    with b:
        st.subheader("Risk engine snapshot")
        for k, v, note in risk_snapshot(state):
            st.markdown(f"- **{k}:** {v} — {note}")
        st.subheader("Event watch")
        for e in event_watch():
            st.markdown(f"- {e}")
    if show_details:
        st.caption("Monthly / Quarterly / Blended probs")
        st.write(pd.DataFrame([state.monthly_probs, state.quarterly_probs, state.blended_probs], index=["monthly", "quarterly", "blended"]))

with tab_next:
    a, b = st.columns(2)
    with a:
        st.subheader("Path to next Q")
        st.markdown(f"- **Target:** {path['target']}")
        st.markdown(f"- **Status:** {path['status']}")
        st.markdown(f"- **Confidence:** {path['confidence']}")
        st.markdown(f"- **Failure risk:** {path['fail']}")
        st.markdown(f"- **Catatan:** {path['note']}")
        st.markdown("### Transition tree mini")
        st.markdown(f"- **Base:** {state.current_q} → {state.next_q}")
        st.markdown(f"- **Alt 1:** stay di {state.current_q}")
        st.markdown("- **Alt 2:** fakeout ke quad lain kalau data baru nggak confirm")
    with b:
        st.subheader("Timing / turning point")
        st.markdown(f"- **Top risk:** {level_text(state.top_score, 'timing')}")
        st.markdown(f"- **Bottom risk:** {level_text(state.bottom_score, 'timing')}")
        st.markdown(f"- **Masih ada top di atas top?** {level_text(state.higher_top_risk, 'strength')}")
        st.markdown(f"- **Masih ada bottom di bawah bottom?** {level_text(state.lower_bottom_risk, 'strength')}")
        st.warning("Top/bottom di sini dibaca sebagai proses. Jadi jangan langsung anggap final top atau final bottom kalau baru tahap awal.")

with tab_play:
    st.subheader("Current vs next playbook mini")
    cols = st.columns(2)
    for idx, bucket in enumerate(["current", "next"]):
        with cols[idx]:
            st.markdown(f"### {'Playbook sekarang' if bucket == 'current' else 'Kalau next phase jalan'}")
            for asset, picks in book[bucket].items():
                st.markdown(f"**{asset}**")
                for p in picks[:3]:
                    st.markdown(f"- {p}")
    st.subheader("Yang rawan")
    for x in book["avoid"]["yang rawan"]:
        st.markdown(f"- {x}")
    st.subheader("Invalidation mini-box")
    st.markdown("- kalau inflasi panas lagi, playbook defensif / gold / USD bisa naik prioritas")
    st.markdown("- kalau growth breadth mendadak membaik, jangan keras kepala di mode defensif")
    st.markdown("- kalau shock perang / likuiditas mereda cepat, beberapa edge bisa cepat pudar")

with tab_rel:
    st.subheader("Relative & size rotation")
    cols = st.columns(2)
    keys = list(rel.keys())
    for i, k in enumerate(keys):
        with cols[i % 2]:
            v = rel[k]
            st.markdown(f"### {k}")
            st.markdown(f"- **Arah:** {v['direction']}")
            st.markdown(f"- **Kuatnya:** {v['strength']}")
            st.markdown(f"- **Kondisi:** {v['state']}")
            st.markdown(f"- **Masih kuat naik nggak?:** {v['sustain']}")
    st.info("Kalau small caps > big caps, jangan langsung anggap bullish. Lihat juga: breadth sehat apa cuma lagi terlalu panas.")

with tab_shock:
    st.subheader("Transmission / correlation")
    st.markdown("- kalau oil naik bareng USD dan 2Y ikut naik, biasanya broad beta lebih susah napas")
    st.markdown("- kalau real yields turun dan stress reda, duration / BTC / growth names biasanya lebih enak")
    st.markdown("- kalau dollar turun tapi EM masih lemah, berarti masalahnya bukan cuma USD")
    st.subheader("Risk / sentiment / exhaustion")
    st.markdown(f"- **Fear & Greed vibe:** {sent['fear_greed']}")
    st.markdown(f"- **IWM read:** {sent['iwm']}")
    st.subheader("What-if yang kepake")
    st.markdown("- **Soft landing trap:** kelihatan aman bentar, terus growth jeblok lagi")
    st.markdown("- **Double-dip:** bottoming gagal, turun lagi")
    st.markdown("- **Re-acceleration fakeout:** kelihatan pulih, tapi cuma semu")
    st.markdown("- **War / oil shock:** direct commodity menang duluan, broad beta belum tentu")
    st.markdown("- **Correlation break:** textbook-nya bilang A harus ikut, tapi real life kadang nggak")

st.caption("Backend udah dibalikin ke blended core yang lebih stabil. Frontend sengaja dibikin simple dan lebih manusiawi buat dibaca cepat.")
