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

# -------------------------
# Config
# -------------------------
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
    "CPI", "Jobs", "Claims", "FOMC vibe", "PCE", "Retail Sales"
]

# -------------------------
# Helpers
# -------------------------
def pct(x: float) -> str:
    return f"{x*100:.1f}%"


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
def fred_series(series_id: str, limit: int = 240) -> pd.Series:
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
        s = pd.Series(vals, index=idx).dropna()
        return s
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
            if "Close" in data:
                s = data["Close"].dropna()
                if not s.empty and symbols:
                    out[symbols[0]] = s
    except Exception:
        pass
    return out


def last_change(series: pd.Series, lookback: int = 21) -> float:
    if series is None or len(series) <= lookback:
        return 0.0
    return float(series.iloc[-1] / series.iloc[-1 - lookback] - 1.0)


def zscore_tail(series: pd.Series, window: int = 36) -> float:
    if series is None or len(series) < max(12, window):
        return 0.0
    tail = series.dropna().iloc[-window:]
    if len(tail) < 12:
        return 0.0
    sd = float(tail.std(ddof=0))
    if sd < 1e-12:
        return 0.0
    return float((tail.iloc[-1] - tail.mean()) / sd)


@dataclass
class CoreState:
    q_probs: Dict[str, float]
    current_q: str
    next_q: str
    confidence: float
    growth_score: float
    inflation_score: float
    stress_score: float
    phase_strength: float
    breadth: float
    fragility: float
    transition_pressure: float
    top_score: float
    bottom_score: float
    higher_top_risk: float
    lower_bottom_risk: float
    path_status: str
    path_conf: float
    path_fail: float
    sub_phase: str
    regime_note: str


def compute_core(data: Dict[str, pd.Series]) -> CoreState:
    growth_inputs = [
        zscore_tail(data.get("INDPRO", pd.Series(dtype=float))),
        zscore_tail(data.get("RETAIL", pd.Series(dtype=float))),
        zscore_tail(data.get("PAYEMS", pd.Series(dtype=float))),
        -zscore_tail(data.get("UNRATE", pd.Series(dtype=float))),
        -zscore_tail(data.get("CLAIMS", pd.Series(dtype=float))),
    ]
    infl_inputs = [
        zscore_tail(data.get("CPI", pd.Series(dtype=float))),
        zscore_tail(data.get("CORE_CPI", pd.Series(dtype=float))),
        zscore_tail(data.get("OIL", pd.Series(dtype=float))),
    ]
    stress_inputs = [
        zscore_tail(data.get("HY", pd.Series(dtype=float))),
        zscore_tail(data.get("NFCI", pd.Series(dtype=float))),
        zscore_tail(data.get("SAHM", pd.Series(dtype=float))),
    ]

    g = float(np.nanmean(growth_inputs)) if growth_inputs else 0.0
    i = float(np.nanmean(infl_inputs)) if infl_inputs else 0.0
    s = float(np.nanmean(stress_inputs)) if stress_inputs else 0.0

    g_up = sigmoid(1.25 * g - 0.45 * s)
    i_up = sigmoid(1.15 * i + 0.10 * s)

    probs = {
        "Q1": g_up * (1 - i_up),
        "Q2": g_up * i_up,
        "Q3": (1 - g_up) * i_up,
        "Q4": (1 - g_up) * (1 - i_up),
    }
    total = sum(probs.values()) or 1.0
    probs = {k: v / total for k, v in probs.items()}
    order = sorted(probs, key=probs.get, reverse=True)
    current_q = order[0]
    next_q = order[1]

    conf = probs[order[0]] - probs[order[1]]
    phase_strength = clip01(0.5 + 0.22 * abs(g) + 0.18 * abs(i))
    breadth = clip01(0.5 + 0.10 * sum(x > 0 for x in growth_inputs + infl_inputs) - 0.04 * sum(x < -0.75 for x in growth_inputs + infl_inputs))
    fragility = clip01(0.35 + 0.25 * s + 0.20 * (1 - conf) + 0.10 * abs(i - g))
    transition_pressure = clip01(0.25 + 0.30 * (1 - conf) + 0.20 * fragility + 0.10 * abs(g) * (1 if current_q in ["Q1", "Q2"] else 0.6))

    top_score = clip01(0.15 + 0.30 * (current_q in ["Q2", "Q3"]) + 0.20 * fragility + 0.15 * max(g, 0) + 0.10 * max(i, 0))
    bottom_score = clip01(0.10 + 0.30 * (current_q in ["Q4", "Q1"]) + 0.20 * max(s, 0) + 0.15 * max(-g, 0))
    higher_top_risk = clip01(0.10 + 0.25 * top_score + 0.20 * (1 - conf))
    lower_bottom_risk = clip01(0.10 + 0.25 * bottom_score + 0.20 * (1 - conf))

    if transition_pressure < 0.32:
        path_status = "masih anteng"
    elif transition_pressure < 0.48:
        path_status = "baru mulai"
    elif transition_pressure < 0.66:
        path_status = "lagi kebentuk"
    elif transition_pressure < 0.82:
        path_status = "udah lumayan valid"
    else:
        path_status = "udah kuat banget"

    path_conf = clip01(0.4 + 0.4 * conf + 0.2 * transition_pressure)
    path_fail = clip01(0.55 - 0.35 * conf + 0.15 * fragility)

    if conf > 0.22 and fragility < 0.45:
        sub_phase = "masih sehat"
    elif conf > 0.14 and fragility < 0.60:
        sub_phase = "masih jalan tapi mulai rapuh"
    elif top_score > bottom_score and top_score > 0.55:
        sub_phase = "lagi topping / rawan capek"
    elif bottom_score > top_score and bottom_score > 0.55:
        sub_phase = "lagi bottoming / nyoba mantul"
    else:
        sub_phase = "campur aduk / transisi"

    regime_note = {
        "Q1": "growth masih oke, inflasi mendingan",
        "Q2": "growth masih dorong, inflasi juga ikut panas",
        "Q3": "growth mulai loyo tapi inflasi masih ganggu",
        "Q4": "growth sama inflasi sama-sama jinak / dingin",
    }[current_q]

    return CoreState(
        q_probs=probs,
        current_q=current_q,
        next_q=next_q,
        confidence=clip01(conf * 2.2),
        growth_score=clip01(0.5 + 0.25 * g),
        inflation_score=clip01(0.5 + 0.25 * i),
        stress_score=clip01(0.5 + 0.25 * s),
        phase_strength=phase_strength,
        breadth=breadth,
        fragility=fragility,
        transition_pressure=transition_pressure,
        top_score=top_score,
        bottom_score=bottom_score,
        higher_top_risk=higher_top_risk,
        lower_bottom_risk=lower_bottom_risk,
        path_status=path_status,
        path_conf=path_conf,
        path_fail=path_fail,
        sub_phase=sub_phase,
        regime_note=regime_note,
    )


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


def risk_snapshot(state: CoreState) -> List[Tuple[str, str]]:
    return [
        ("Growth", level_text(state.growth_score, "strength")),
        ("Inflasi", level_text(state.inflation_score, "strength")),
        ("Stress", level_text(state.stress_score, "strength")),
        ("Fragility", level_text(state.fragility, "fragility")),
    ]


def playbook(state: CoreState) -> Dict[str, Dict[str, List[str]]]:
    current = state.current_q
    next_q = state.next_q
    base = {
        "Q1": {
            "US Stocks": ["quality growth", "semis besar", "consumer yang masih kuat"],
            "Futures / Commodities": ["industrial metals tipis-tipis", "equity index long lebih oke"],
            "Forex": ["FX pro-growth, USD agak jinak"],
            "Crypto": ["BTC dulu, baru alt kalau breadth ikut"],
            "IHSG": ["big caps sehat, domestik cyclicals kalau rupiah aman"],
        },
        "Q2": {
            "US Stocks": ["energy", "financials", "cyclicals", "small caps kalau breadth sehat"],
            "Futures / Commodities": ["oil / commodity beta", "index masih bisa ikut tapi rawan telat"],
            "Forex": ["commodity FX lebih enak"],
            "Crypto": ["risk-on oke, tapi hati-hati kalau udah frothy"],
            "IHSG": ["resources + second liners kalau flow lokal ikut"],
        },
        "Q3": {
            "US Stocks": ["defensives", "pricing power", "jangan terlalu kejar beta"],
            "Futures / Commodities": ["gold", "oil cuma kalau shock-nya direct"],
            "Forex": ["USD / safety lebih kepake"],
            "Crypto": ["lebih hati-hati, BTC mending daripada alt"],
            "IHSG": ["resource selectif, defensif, jangan broad risk-on"],
        },
        "Q4": {
            "US Stocks": ["defensives", "duration-sensitive names", "quality"],
            "Futures / Commodities": ["rates / duration / gold"],
            "Forex": ["USD masih oke, carry pilih-pilih"],
            "Crypto": ["nunggu konfirmasi likuiditas, jangan buru-buru alt"],
            "IHSG": ["big caps defensif, tunggu konfirmasi bottoming"],
        },
    }
    avoid = {
        "Q1": ["bond proxy defensif yang terlalu lambat", "USD terlalu berat"],
        "Q2": ["duration panjang", "defensives terlalu dini"],
        "Q3": ["small caps yang udah terlalu panas", "broad beta chase"],
        "Q4": ["cyclicals tanpa konfirmasi", "alt beta tanpa likuiditas"],
    }
    return {
        "current": base[current],
        "next": base[next_q],
        "avoid": {"yang rawan": avoid[current]},
    }


def relative_read(prices: Dict[str, pd.Series], state: CoreState) -> Dict[str, Dict[str, str]]:
    def rel(a: str, b: str, fallback_bias: str) -> Dict[str, str]:
        if a in prices and b in prices and len(prices[a]) > 40 and len(prices[b]) > 40:
            xa = last_change(prices[a], 21)
            xb = last_change(prices[b], 21)
            diff = xa - xb
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

    # size rotation
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
    spy = prices.get("SPY")
    if iwm is not None and len(iwm) > 60:
        ext = last_change(iwm, 21)
    else:
        ext = 0.0
    if ext > 0.09 and state.top_score > 0.55:
        iwm_note = "IWM keliatan lagi ngebut, ini bisa jadi beta chase / rawan blow-off"
    elif ext > 0.04:
        iwm_note = "IWM masih lumayan kuat, tapi cek apakah breadth ikut sehat"
    else:
        iwm_note = "IWM nggak terlalu jadi sinyal top sekarang"

    fear_greed = "greed lumayan tinggi" if state.current_q in ["Q1", "Q2"] else "fear/defensive mode lebih kepake"
    return {
        "fear_greed": fear_greed,
        "iwm": iwm_note,
    }


def path_to_next(state: CoreState) -> Dict[str, str]:
    if state.path_conf < 0.35:
        conf = "masih tipis"
    elif state.path_conf < 0.65:
        conf = "lumayan kebaca"
    else:
        conf = "cukup valid"

    return {
        "target": f"{state.current_q} → {state.next_q}",
        "status": state.path_status,
        "confidence": conf,
        "fail": level_text(state.path_fail, "fragility"),
        "note": "ini jalur paling mungkin sekarang, tapi tetap cek trigger data berikutnya",
    }


def event_watch() -> List[str]:
    now = datetime.now(timezone.utc)
    names = WATCH_EVENTS[:4]
    return [f"{x} – cek lagi pas rilis besar / update harian" for x in names]


# -------------------------
# UI helpers
# -------------------------
def inject_css():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1380px;}
        .hero {padding: 1rem 1rem 0.8rem 1rem; border-radius: 20px; border: 1px solid rgba(255,255,255,0.08); background: linear-gradient(135deg, rgba(30,41,59,.80), rgba(15,23,42,.92)); margin-bottom: 1rem;}
        .pill {display:inline-block; padding: 0.26rem 0.62rem; border-radius:999px; background: rgba(255,255,255,.08); margin: 0 .35rem .35rem 0; font-size: .83rem;}
        .card {padding: 0.95rem 1rem; border-radius: 18px; border: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.03); height: 100%;}
        .title {font-size: 1.02rem; font-weight: 700; margin-bottom: .2rem;}
        .muted {opacity: .76; font-size: .92rem;}
        .big {font-size: 1.22rem; font-weight: 800; margin: .2rem 0 .35rem;}
        .mini {font-size: .82rem; opacity: .75;}
        .sep {height: .5rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def card(title: str, main: str, sub: str = "", pills: Optional[List[str]] = None):
    pills = pills or []
    st.markdown(f"<div class='card'><div class='title'>{title}</div><div class='big'>{main}</div><div class='muted'>{sub}</div>{''.join([f'<span class=\"pill\">{p}</span>' for p in pills])}</div>", unsafe_allow_html=True)


# -------------------------
# App
# -------------------------
inject_css()

st.markdown("<div class='hero'><div class='title'>QuantFinalV4_Max</div><div class='big'>macro regime map yang gampang dibaca, nggak kaku, tapi hitungan dalemnya tetap ada</div><div class='muted'>core-nya tetap Baseline_Blended_Core. yang di layar ini udah dibungkus biar lebih gampang dibaca: current, next, playbook, relative, sama shocks/what-if.</div></div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Setelan")
    perf = st.toggle("Mode enteng", value=True)
    show_details = st.toggle("Buka detail angka", value=False)
    price_period = st.selectbox("Lookback harga", ["6mo", "1y", "2y"], index=1)

# Load data
macro = {k: fred_series(v) for k, v in SERIES.items()}
prices = yf_prices(["SPY", "EEM", "EIDO", "IWM", "BTC-USD"], period=price_period)
state = compute_core(macro)
book = playbook(state)
rel = relative_read(prices, state)
sent = sentiment_overlay(prices, state)
path = path_to_next(state)

# Top line
c1, c2, c3, c4 = st.columns(4)
with c1:
    card("Current", q_label(state.current_q), f"inti ceritanya: {state.regime_note}", [f"confidence: {level_text(state.confidence, 'conf')}", f"sub-phase: {state.sub_phase}"])
with c2:
    card("Next", q_label(state.next_q), f"path ke next Q: {path['status']}", [f"conf: {path['confidence']}", f"failure risk: {path['fail']}"])
with c3:
    cur_pick = next(iter(book['current'].get('US Stocks', ['belum ada edge yang clean'])), 'belum ada edge yang clean')
    card("Playbook", cur_pick, "ini yang paling masuk akal buat dibaca dulu, nanti detailnya ada di bawah", [f"phase strength: {level_text(state.phase_strength, 'strength')}", f"fragility: {level_text(state.fragility, 'fragility')}"])
with c4:
    card("Top / Bottom", f"Top {level_text(state.top_score, 'timing')} | Bottom {level_text(state.bottom_score, 'timing')}", "buat baca lagi rawan topping atau lagi nyoba bottoming", [f"higher top risk: {level_text(state.higher_top_risk, 'strength')}", f"lower bottom risk: {level_text(state.lower_bottom_risk, 'strength')}"])

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

tab_current, tab_next, tab_play, tab_rel, tab_shock = st.tabs(["Current", "Next", "Playbook", "Relative", "Shocks / What-If"])

with tab_current:
    a, b = st.columns([1.2, 1])
    with a:
        st.subheader("Sekarang lagi gimana?")
        st.markdown(f"- **Fase utama:** {q_label(state.current_q)}")
        st.markdown(f"- **Bacaan model:** **{level_text(state.confidence, 'conf')}**")
        st.markdown(f"- **Kondisi fase:** **{state.sub_phase}**")
        st.markdown(f"- **Kekuatan fase:** **{level_text(state.phase_strength, 'strength')}**")
        st.markdown(f"- **Breadth:** **{level_text(state.breadth, 'strength')}**")
        st.markdown(f"- **Fragility:** **{level_text(state.fragility, 'fragility')}**")
        st.info("Intinya: dashboard ini nggak maksa lo baca angka mentah. Yang keluar di depan sengaja bahasa manusia dulu.")
    with b:
        st.subheader("Risk engine snapshot")
        for k, v in risk_snapshot(state):
            st.markdown(f"- **{k}:** {v}")
        st.subheader("Event watch")
        for e in event_watch():
            st.markdown(f"- {e}")
    if show_details:
        st.caption("Quad probabilities")
        st.write(pd.DataFrame([state.q_probs]))

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
        st.markdown(f"- **Alt 2:** fakeout ke quad lain kalau data baru nggak confirm")
    with b:
        st.subheader("Timing / turning point")
        st.markdown(f"- **Top risk:** {level_text(state.top_score, 'timing')}")
        st.markdown(f"- **Bottom risk:** {level_text(state.bottom_score, 'timing')}")
        st.markdown(f"- **Di atas top masih ada top?** {level_text(state.higher_top_risk, 'strength')}")
        st.markdown(f"- **Di bawah bottom masih ada bottom?** {level_text(state.lower_bottom_risk, 'strength')}")
        st.warning("Kalau top/bottom baru tahap awal, jangan langsung anggap final. Real life sering masih ada higher-top atau lower-bottom.")

with tab_play:
    st.subheader("Current vs next playbook mini")
    cols = st.columns(2)
    for idx, bucket in enumerate(["current", "next"]):
        with cols[idx]:
            st.markdown(f"### {'Playbook sekarang' if bucket=='current' else 'Kalau next phase jalan'}")
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
    st.info("Kalau lo lihat small caps > big caps, jangan langsung anggap bullish. Harus lihat juga: masih sehat atau udah frothy / capek.")

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

st.caption("Catatan: backend tetap pakai phase/quad logic sebagai anchor, tapi tampilan depan sengaja dibikin lebih manusiawi biar nggak bikin capek baca.")
