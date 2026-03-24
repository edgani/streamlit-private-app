
import math
from datetime import date, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

st.set_page_config(page_title="QuantFinalV4_Max", layout="wide")

APP_NAME = "QuantFinalV4_Max"
CORE_NAME = "Baseline_Blended_Core"

st.markdown("""
<style>
:root {
  --bg:#07101f;
  --card:#0c1526;
  --line:#263246;
  --muted:#9fb0c8;
  --text:#f3f6fb;
}
html, body, [data-testid="stAppViewContainer"] {background: var(--bg); color: var(--text);}
.block-container {padding-top: 1.6rem; padding-bottom: 2.6rem;}
h1,h2,h3,h4,h5,h6,p,span,div,label {color: var(--text);}
.small-muted {color: var(--muted); font-size: .92rem;}
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
}
.metric-title {font-size: .76rem; color: var(--muted); text-transform: uppercase; letter-spacing: .05em;}
.metric-value {font-size: 1.85rem; font-weight: 800; line-height: 1.1;}
.metric-sub {font-size: .88rem; color:#c4d2e6; margin-top:4px;}
.pill {
  display:inline-block;
  border:1px solid #33435d;
  border-radius:999px;
  padding:3px 9px;
  font-size:.82rem;
  color:#dbe7ff;
  background: rgba(30,42,63,.8);
  margin-right:6px;
  margin-top:4px;
}
.tight-table table {
  width:100%;
  border-collapse: collapse;
  table-layout: fixed;
}
.tight-table th, .tight-table td {
  border:1px solid #243147;
  padding:8px 10px;
  font-size:.90rem;
  text-align:left;
  vertical-align:top;
  word-wrap: break-word;
}
.tight-table th {
  color:#9fb0c8;
  font-weight:700;
  background: rgba(20,29,44,.85);
}
.tight-table td {color:#f5f8fd;}
</style>
""", unsafe_allow_html=True)

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def pct(x: float) -> str:
    return f"{100*x:.1f}%"

def pill(txt: str) -> str:
    return f"<span class='pill'>{txt}</span>"

def bucket(x: float, cuts: Tuple[float, float], labels: Tuple[str, str, str]) -> str:
    if x < cuts[0]:
        return labels[0]
    if x < cuts[1]:
        return labels[1]
    return labels[2]

def score_to_label(x: float) -> str:
    return bucket(x, (0.33, 0.66), ("Low", "Medium", "High"))

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

@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fred_series(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        df = pd.read_csv(url)
        date_col = df.columns[0]
        val_col = df.columns[1]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        s = df.dropna().set_index(date_col)[val_col]
        s.name = series_id
        return s
    except Exception:
        return pd.Series(dtype=float, name=series_id)

@st.cache_data(ttl=60 * 60 * 4, show_spinner=False)
def yahoo_close(ticker: str, period: str = "1y", interval: str = "1d") -> pd.Series:
    if yf is None:
        return pd.Series(dtype=float, name=ticker)
    try:
        data = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        if data is None or len(data) == 0:
            return pd.Series(dtype=float, name=ticker)
        if isinstance(data.columns, pd.MultiIndex):
            if ("Close", ticker) in data.columns:
                s = data[("Close", ticker)]
            else:
                s = data["Close"].iloc[:, 0]
        else:
            s = data["Close"]
        s = pd.to_numeric(s, errors="coerce").dropna()
        s.name = ticker
        return s
    except Exception:
        return pd.Series(dtype=float, name=ticker)

@st.cache_data(ttl=60 * 60 * 4, show_spinner=False)
def stooq_close(symbol: str) -> pd.Series:
    # symbol examples: spy.us, iwm.us
    try:
        url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
        df = pd.read_csv(url)
        if "Date" not in df.columns or "Close" not in df.columns:
            return pd.Series(dtype=float, name=symbol)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        s = df.dropna().set_index("Date")["Close"]
        s.name = symbol
        return s
    except Exception:
        return pd.Series(dtype=float, name=symbol)

def market_series(primary: str, stooq_symbol: str = "") -> pd.Series:
    s = yahoo_close(primary, "1y")
    if not s.empty:
        return s
    if stooq_symbol:
        return stooq_close(stooq_symbol)
    return pd.Series(dtype=float, name=primary)

def ret_21(s: pd.Series) -> float:
    if s.empty or len(s) < 22:
        return np.nan
    return float(s.iloc[-21:].pct_change().add(1).prod() - 1)

def stretch_63(s: pd.Series) -> float:
    if s.empty or len(s) < 63:
        return np.nan
    return float((s.iloc[-1] / s.iloc[-63:].mean()) - 1)

def safe_series_mean(series_list: List[pd.Series]) -> pd.Series:
    valid = [s for s in series_list if s is not None and not s.empty]
    if not valid:
        return pd.Series(dtype=float)
    df = pd.concat(valid, axis=1, join="outer").sort_index().ffill().dropna(how="all")
    return df.mean(axis=1, skipna=True).dropna()

def rel_state(spread: float) -> str:
    if spread > 0.08:
        return "Building"
    if spread > 0.025:
        return "Stable"
    if spread > -0.02:
        return "Peaking"
    return "Fading"

def rel_quality(strength: float, breadth: float, stretch: float) -> str:
    if np.isnan(stretch):
        return "Unavailable"
    if stretch > 0.10 and strength > 0.58:
        return "Frothy"
    if breadth < 0.35 and strength > 0.45:
        return "Narrow"
    if strength < 0.28:
        return "Weak"
    return "Healthy"

def rel_sustainability(strength: float, quality: str, fragility: float) -> str:
    if quality == "Unavailable":
        return "Low"
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

SER = {
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
    "M2SL": fred_series("M2SL"),
    "DTWEXBGS": fred_series("DTWEXBGS"),
}

def compute_core() -> Dict[str, object]:
    growth_inputs = [
        robust_z(SER["INDPRO"].pct_change(12)),
        robust_z(SER["RSAFS"].pct_change(12)),
        robust_z(SER["PAYEMS"].pct_change(12)),
        -robust_z(SER["UNRATE"].diff()),
        -robust_z(SER["ICSA"].pct_change(12)),
    ]
    infl_inputs = [
        robust_z(SER["CPI"].pct_change(12)),
        robust_z(SER["CORE_CPI"].pct_change(12)),
        robust_z(SER["PPI"].pct_change(12)),
        robust_z(SER["WTI"].pct_change(12)),
    ]
    stress_inputs = [
        robust_z(SER["SAHM"]),
        robust_z(SER["NFCI"]),
        robust_z(SER["HY"]),
    ]
    g_m = float(np.nanmean(growth_inputs))
    i_m = float(np.nanmean(infl_inputs))
    s_m = float(np.nanmean(stress_inputs))

    growth_q = [
        robust_z(SER["INDPRO"].pct_change(12).rolling(3).mean(), 60),
        robust_z(SER["RSAFS"].pct_change(12).rolling(3).mean(), 60),
        robust_z(SER["PAYEMS"].pct_change(12).rolling(3).mean(), 60),
        -robust_z(SER["UNRATE"].diff().rolling(3).mean(), 60),
        -robust_z(SER["ICSA"].pct_change(12).rolling(3).mean(), 60),
    ]
    infl_q = [
        robust_z(SER["CPI"].pct_change(12).rolling(3).mean(), 60),
        robust_z(SER["CORE_CPI"].pct_change(12).rolling(3).mean(), 60),
        robust_z(SER["PPI"].pct_change(12).rolling(3).mean(), 60),
        robust_z(SER["WTI"].pct_change(12).rolling(3).mean(), 60),
    ]
    g_q = float(np.nanmean(growth_q))
    i_q = float(np.nanmean(infl_q))

    g_up_m = sigmoid(1.25 * g_m - 0.35 * s_m)
    i_up_m = sigmoid(1.20 * i_m + 0.10 * s_m)
    g_up_q = sigmoid(1.05 * g_q - 0.25 * s_m)
    i_up_q = sigmoid(1.05 * i_q + 0.05 * s_m)

    monthly = {
        "Q1": g_up_m * (1 - i_up_m),
        "Q2": g_up_m * i_up_m,
        "Q3": (1 - g_up_m) * i_up_m,
        "Q4": (1 - g_up_m) * (1 - i_up_m),
    }
    quarterly = {
        "Q1": g_up_q * (1 - i_up_q),
        "Q2": g_up_q * i_up_q,
        "Q3": (1 - g_up_q) * i_up_q,
        "Q4": (1 - g_up_q) * (1 - i_up_q),
    }
    blended = {k: 0.4 * monthly[k] + 0.6 * quarterly[k] for k in monthly}
    total = sum(blended.values())
    blended = {k: v / total for k, v in blended.items()}
    ranked = sorted(blended.items(), key=lambda x: x[1], reverse=True)
    current_q, current_p = ranked[0]
    next_q, next_p = ranked[1]

    agreement = 1.0 - 0.5 * sum(abs(monthly[k] - quarterly[k]) for k in monthly)
    agreement = clamp01(agreement)
    confidence = clamp01(0.55 * current_p + 0.45 * agreement)

    phase_strength = clamp01(abs(g_m) * 0.45 + abs(i_m) * 0.35 + max(0, current_p - 0.25) * 0.70)
    growth_breadth = clamp01(sum(1 for x in growth_inputs if x > 0) / max(1, len(growth_inputs)))
    inflation_breadth = clamp01(sum(1 for x in infl_inputs if x > 0) / max(1, len(infl_inputs)))
    breadth = 0.5 * growth_breadth + 0.5 * inflation_breadth
    fragility = clamp01(0.45 * (1 - current_p) + 0.35 * (1 - agreement) + 0.20 * max(0, s_m))

    if current_q == "Q1":
        sub_phase = "Goldilocks / recovery" if fragility < 0.45 else "Recovery but fragile"
    elif current_q == "Q2":
        sub_phase = "Expansion / hot growth" if i_m < 0.35 else "Reflation / overheating risk"
    elif current_q == "Q3":
        sub_phase = "Slowdown with sticky inflation" if i_m < 0.35 else "Stagflation / pressure"
    else:
        sub_phase = "Bottoming attempt" if g_m > -0.20 else "Deflationary slowdown"

    top_score = clamp01(0.35 * max(0, i_m) + 0.25 * phase_strength + 0.20 * fragility + 0.20 * float(current_q in ["Q2", "Q3"]))
    bottom_score = clamp01(0.40 * float(current_q == "Q4") + 0.25 * max(0, -g_m) + 0.20 * (1 - fragility) + 0.15 * float(current_q == "Q4" and g_m > -0.2))
    higher_top = clamp01(top_score * 0.65 * (1 - bottom_score))
    lower_bottom = clamp01(bottom_score * 0.65 * (1 - top_score))

    transition_pressure = clamp01(0.40 * fragility + 0.35 * (1 - current_p) + 0.25 * abs(g_m - i_m) / 3.0)
    transition_conviction = clamp01(0.55 * transition_pressure + 0.45 * next_p)
    stay_probability = clamp01(current_p * (1 - fragility * 0.45))

    # tighter gating so current and next don't conflict too easily
    if current_p > next_p + 0.05:
        if transition_conviction > 0.58:
            path_status = "Valid"
        elif transition_conviction > 0.40:
            path_status = "Building"
        else:
            path_status = "Starting"
    else:
        if transition_conviction > 0.75:
            path_status = "Confirmed"
        elif transition_conviction > 0.58:
            path_status = "Valid"
        elif transition_conviction > 0.42:
            path_status = "Building"
        elif transition_conviction > 0.26:
            path_status = "Starting"
        else:
            path_status = "Stable / no clean shift"

    return {
        "monthly": monthly,
        "quarterly": quarterly,
        "blended": blended,
        "current_q": current_q,
        "next_q": next_q,
        "current_p": current_p,
        "next_p": next_p,
        "confidence": confidence,
        "agreement": agreement,
        "phase_strength": phase_strength,
        "growth_breadth": growth_breadth,
        "inflation_breadth": inflation_breadth,
        "breadth": breadth,
        "fragility": fragility,
        "sub_phase": sub_phase,
        "top_score": top_score,
        "bottom_score": bottom_score,
        "higher_top": higher_top,
        "lower_bottom": lower_bottom,
        "transition_pressure": transition_pressure,
        "transition_conviction": transition_conviction,
        "stay_probability": stay_probability,
        "path_status": path_status,
        "stress_growth": clamp01(abs(g_m)),
        "stress_infl": clamp01(abs(i_m)),
        "stress_liq": clamp01(max(0, robust_z(SER["NFCI"]))),
    }

core = compute_core()

def make_market_row(name: str, a: pd.Series, b: pd.Series, pos_label: str, neg_label: str, breadth_hint: float, proxy_note: str = "", th: float = 0.02) -> Dict[str, str]:
    a_r = ret_21(a)
    b_r = ret_21(b)
    if np.isnan(a_r) or np.isnan(b_r):
        return {"Lens": name, "Direction": "Unavailable", "Strength": "Low", "State": "Stable", "Quality": "Unavailable", "Sustainability": "Low", "Confirmation": "Not confirmed", "Read": f"No live feed{proxy_note}"}
    spread = a_r - b_r
    direction = pos_label if spread > th else (neg_label if spread < -th else "Balanced")
    strength_num = clamp01(min(1.0, abs(spread) / max(th*5, 0.10)))
    state = rel_state(spread)
    stretch = stretch_63(a)
    quality = rel_quality(strength_num, breadth_hint, stretch)
    sustain = rel_sustainability(strength_num, quality, core["fragility"])
    confirm = rel_confirmation(strength_num, state, breadth_hint)
    read = direction if direction != "Balanced" else f"Balanced / weak edge{proxy_note}"
    return {"Lens": name, "Direction": direction, "Strength": score_to_label(strength_num), "State": state, "Quality": quality, "Sustainability": sustain, "Confirmation": confirm, "Read": read}

def liquidity_composite() -> Tuple[float, str]:
    walcl = robust_z(SER["WALCL"].pct_change(26))
    m2 = robust_z(SER["M2SL"].pct_change(12))
    usd = -robust_z(SER["DTWEXBGS"].pct_change(6))
    nfci = -robust_z(SER["NFCI"])
    comp = np.nanmean([walcl, m2, usd, nfci])
    score = clamp01(sigmoid(comp) if np.isfinite(comp) else 0.5)
    label = "Liquidity confirming" if score > 0.62 else ("Liquidity fading" if score < 0.38 else "Liquidity mixed")
    return score, label

def compute_relative() -> List[Dict[str, str]]:
    spy = market_series("SPY", "spy.us")
    eem = market_series("EEM", "eem.us")
    eido = market_series("EIDO", "eido.us")
    btc = market_series("BTC-USD")
    liq_score, liq_label = liquidity_composite()

    rows = [
        make_market_row("US vs EM", spy, eem, "US stronger", "EM stronger", core["growth_breadth"], th=0.018),
        make_market_row("IHSG vs US", eido, spy, "IHSG stronger", "US stronger", core["breadth"], " (proxy-based)", th=0.025),
        make_market_row("IHSG vs EM", eido, eem, "IHSG stronger", "EM stronger", core["breadth"], " (proxy-based)", th=0.022),
    ]

    btc_r = ret_21(btc)
    if np.isnan(btc_r):
        rows.append({"Lens":"Crypto vs Liquidity","Direction":"Unavailable","Strength":"Low","State":"Stable","Quality":"Unavailable","Sustainability":"Low","Confirmation":"Not confirmed","Read":"No live feed (composite liquidity exists)"})
    else:
        spread = btc_r - (liq_score - 0.5) * 0.16
        direction = "Crypto stronger" if spread > 0.025 else ("Liquidity not confirming" if spread < -0.025 else "Balanced")
        strength_num = clamp01(min(1.0, abs(spread) / 0.14))
        state = rel_state(spread)
        quality = rel_quality(strength_num, core["breadth"], stretch_63(btc))
        sustain = rel_sustainability(strength_num, quality, core["fragility"])
        confirm = rel_confirmation(strength_num, state, core["breadth"])
        rows.append({"Lens":"Crypto vs Liquidity","Direction":direction,"Strength":score_to_label(strength_num),"State":state,"Quality":quality,"Sustainability":sustain,"Confirmation":confirm,"Read":liq_label if direction=="Balanced" else direction})
    return rows

def compute_size_rotation() -> List[Dict[str, str]]:
    iwm = market_series("IWM", "iwm.us")
    iwb = market_series("IWB", "iwb.us")
    spy = market_series("SPY", "spy.us")
    btc = market_series("BTC-USD")
    alts = safe_series_mean([
        market_series("ETH-USD"),
        market_series("SOL-USD"),
        market_series("XRP-USD"),
        market_series("ADA-USD"),
        market_series("AVAX-USD"),
        market_series("LINK-USD"),
    ])
    return [
        make_market_row("US Small Caps vs Big Caps (IWM/IWB)", iwm, iwb, "Small > Big", "Big > Small", core["breadth"], th=0.017),
        make_market_row("US Small Caps vs Broad Market (IWM/SPY)", iwm, spy, "Small > Broad", "Broad > Small", core["breadth"], th=0.017),
        make_market_row("Crypto Alt Basket vs BTC", alts, btc, "Alts > BTC", "BTC > Alts", core["breadth"], " (basket proxy)", th=0.03),
    ]

relative_rows = compute_relative()
size_rows = compute_size_rotation()

def fear_greed_value() -> Tuple[int, str]:
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        r = requests.get(url, timeout=8)
        if r.ok:
            j = r.json()
            score = int(j["fear_and_greed"]["score"])
        else:
            score = 35
    except Exception:
        score = 35
    vibe = "Extreme fear" if score < 25 else ("Fear" if score < 45 else ("Neutral" if score < 56 else ("Greed" if score < 76 else "Extreme greed")))
    return score, vibe

def iwm_read() -> str:
    iwm = market_series("IWM", "iwm.us")
    if iwm.empty:
        return "No live read"
    r21 = ret_21(iwm)
    ext = stretch_63(iwm)
    if np.isnan(r21):
        return "No live read"
    if r21 > 0.08 and ext > 0.08:
        return "Blow-off / beta chase risk"
    if r21 > 0.02:
        return "Healthy small-cap strength"
    if r21 < -0.04:
        return "Small-cap stress"
    return "Mixed"

fg_score, fg_vibe = fear_greed_value()
iwm_overlay = iwm_read()

WHAT_IF_SCENARIO_MATRIX = [
    ("Policy rescue", "liquidity eases / squeeze risk", "duration or beta if confirmed", "hard shorting into rescue"),
    ("Growth scare, no recession", "panic without full recession", "duration / defensives", "overpricing depression"),
    ("Commodity spike only", "oil/commodities up without broad growth", "energy / gold", "broad reflation assumption"),
    ("Narrow US leadership", "US up but breadth weak", "quality / megacap only", "assuming healthy broad risk-on"),
]
DIVERGENCE_RULES = [
    ("Macro weak, equities strong", "Market may be front-running a soft landing or a squeeze"),
    ("Crypto strong, liquidity weak", "Speculative move / weaker confirmation"),
    ("IHSG strong, US weak", "Possible local or commodity decoupling"),
]
CORRELATION_TRANSMISSION_PRIORS = [
    ("USD up → EM pressure", "High", "strong USD usually tightens EM conditions"),
    ("Oil up → stagflation tail", "High", "energy shock raises inflation risk"),
    ("Yield down → duration tailwind", "Medium", "works better if growth scare is real"),
]
CRASH_TYPES = [
    ("Liquidity shock", "fast, correlated, violent"),
    ("Growth scare", "duration helps, beta weakens"),
    ("Stagflation shock", "gold / commodity defensives matter more"),
]
FALSE_RECOVERY_MAP = [
    ("Dead-cat bounce", "Bounce without broad confirmation"),
    ("Second leg risk", "Recovery can fail and retest lower"),
]

def build_shocks() -> Dict[str, Tuple[str, str]]:
    shocks = {
        "Policy shock": ("watch", "Policy matters, but not overriding base case"),
        "Geopolitical shock": ("watch", "Use as modifier, not core phase driver"),
        "Liquidity shock": ("watch", "Matters most if liquidity stress rises sharply"),
        "Inflation shock": ("watch", "Important if inflation re-accelerates"),
        "Growth shock": ("watch", "Important if labor/growth roll over harder"),
        "Anomaly flag": ("watch", "Use if market and macro stop confirming each other"),
    }
    if fg_score > 75 or fg_score < 20:
        shocks["Sentiment stretch"] = ("medium", f"Sentiment stretched: {fg_vibe}")
    else:
        shocks["Sentiment stretch"] = ("low", f"Sentiment not stretched: {fg_vibe}")
    return shocks

shocks = build_shocks()
override_active = any(v[0] in ["medium", "high"] for v in shocks.values())

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
    "beta": {"US Stocks": ["small caps / cyclicals"], "Futures / Commodities": ["equity beta"], "Forex": ["high beta FX"], "Crypto": ["alts"], "IHSG": ["2nd liners / local beta"]},
    "cyclical": {"US Stocks": ["industrials", "materials"], "Futures / Commodities": ["industrial commodities"], "Forex": ["commodity FX"], "Crypto": ["risk-on rotation"], "IHSG": ["commodities / cyclicals"]},
}

def current_vs_next_playbook() -> Tuple[Dict[str, List[str]], Dict[str, List[str]], str]:
    cur = core["current_q"]
    nxt = core["next_q"]
    cur_scores = FAMILY_SCORE_BY_QUAD[cur]
    nxt_scores = FAMILY_SCORE_BY_QUAD[nxt]
    cur_sorted = sorted(cur_scores, key=cur_scores.get, reverse=True)
    nxt_sorted = sorted(nxt_scores, key=nxt_scores.get, reverse=True)

    cur_out: Dict[str, List[str]] = {}
    nxt_out: Dict[str, List[str]] = {}
    for bucket_name in ["US Stocks", "Futures / Commodities", "Forex", "Crypto", "IHSG"]:
        cur_assets, nxt_assets = [], []
        for fam in cur_sorted[:2]:
            cur_assets.extend(FAMILY_TO_ASSETS[fam][bucket_name][:1])
        for fam in nxt_sorted[:2]:
            nxt_assets.extend(FAMILY_TO_ASSETS[fam][bucket_name][:1])
        cur_out[bucket_name] = cur_assets
        nxt_out[bucket_name] = nxt_assets

    posture = ("Aggressive" if core["phase_strength"] > 0.65 and core["fragility"] < 0.35
               else "Balanced" if core["confidence"] > 0.45
               else "Defensive" if core["fragility"] > 0.55
               else "Wait / low conviction")
    return cur_out, nxt_out, posture

play_cur, play_next, posture = current_vs_next_playbook()

today = date.today()
events = [
    ("NFP", today + timedelta(days=11)),
    ("CPI", today + timedelta(days=21)),
    ("PPI", today + timedelta(days=22)),
]
event_rows = [[name, dt.isoformat(), f"{(dt - today).days}d"] for name, dt in events]

st.title(APP_NAME)
st.markdown("<div class='small-muted'>Core alpha engine: Baseline_Blended_Core • Visual shell: mind-map card layout • Live backbone: FRED + optional Yahoo/Stooq</div>", unsafe_allow_html=True)
st.write("")

hero_cols = st.columns(5)
hero_items = [
    ("Current Phase", core["current_q"], pill("Decaying") if core["fragility"] > 0.55 else pill("Stable")),
    ("Confidence", pct(core["confidence"]), pill(f"Agreement {pct(core['agreement'])}")),
    ("Sub-Phase", core["sub_phase"], pill(f"Strength {pct(core['phase_strength'])}")),
    ("Top Risk", pct(core["top_score"]), pill(f"Higher-top {pct(core['higher_top'])}")),
    ("Bottom Risk", pct(core["bottom_score"]), pill(f"Lower-bottom {pct(core['lower_bottom'])}")),
]
for col, item in zip(hero_cols, hero_items):
    title, value, sub = item
    with col:
        st.markdown(f"""
        <div class='hero-card'>
          <div class='metric-title'>{title}</div>
          <div class='metric-value'>{value}</div>
          <div class='metric-sub'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

mini_cols = st.columns(5)
mini = [
    ("CURRENT", core["current_q"], pill("Decaying") if core["fragility"] > 0.55 else pill("Stable")),
    ("NEXT", core["next_q"], pill(f"Hazard {pct(core['transition_pressure'])}")),
    ("PLAYBOOK", ", ".join(play_cur["US Stocks"][:1]), pill(f"Conviction {pct(core['confidence'])}")),
    ("RELATIVE", relative_rows[0]["Read"], pill(relative_rows[1]["Read"])),
    ("SHOCKS", "Overlay", pill(f"Top {pct(core['top_score'])} / Bottom {pct(core['bottom_score'])}")),
]
for col, item in zip(mini_cols, mini):
    title, value, sub = item
    with col:
        st.markdown(f"""
        <div class='hero-card'>
          <div class='metric-title'>{title}</div>
          <div style='font-size:1.2rem;font-weight:800'>{value}</div>
          <div class='metric-sub'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

current_tab, next_tab, playbook_tab, relative_tab, shocks_tab, notes_tab = st.tabs(
    ["Current", "Next", "Playbook", "Relative", "Shocks / What-If", "Notes"]
)

with current_tab:
    left, right = st.columns([1.25, 1.0], gap="large")
    with left:
        st.markdown("<div class='card'><div class='section-title'>CURRENT MAP</div>", unsafe_allow_html=True)
        st.markdown(f"**Phase ➜ {core['current_q']}**")
        st.markdown(f"**Confidence ➜ {pct(core['confidence'])}**")
        st.markdown(f"**Agreement ➜ {pct(core['agreement'])}**")
        st.markdown(f"**Sub-Phase ➜ {core['sub_phase']}**")
        st.markdown(f"**Regime Strength ➜ {pct(core['phase_strength'])}**")
        st.markdown(f"**Breadth ➜ {pct(core['breadth'])}**")
        st.markdown(f"**Fragility ➜ {pct(core['fragility'])}**")
        prob_rows = [[k, f"{v:.4f}"] for k, v in sorted(core["blended"].items(), key=lambda x: x[1], reverse=True)]
        st.markdown(table_html(["Phase", "Probability"], prob_rows), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'><div class='section-title'>TOP / BOTTOM LADDER</div>", unsafe_allow_html=True)
        ladder_rows = [
            ["Provisional top", pct(core["top_score"])],
            ["Higher-top / blow-off", pct(core["higher_top"])],
            ["Provisional bottom", pct(core["bottom_score"])],
            ["Lower-bottom / capitulation", pct(core["lower_bottom"])],
        ]
        st.markdown(table_html(["Ladder", "Score"], ladder_rows), unsafe_allow_html=True)
        st.write("")
        risk_rows = [
            ["Growth stress", pct(core["stress_growth"]), score_to_label(core["stress_growth"])],
            ["Inflation stress", pct(core["stress_infl"]), score_to_label(core["stress_infl"])],
            ["Liquidity stress", pct(core["stress_liq"]), score_to_label(core["stress_liq"])],
            ["Sentiment stretch", str(fg_score), fg_vibe],
        ]
        st.markdown("**Risk Engine Snapshot**")
        st.markdown(table_html(["Engine", "Score", "Read"], risk_rows), unsafe_allow_html=True)
        st.write("")
        st.markdown("**Event Watch**")
        st.markdown(table_html(["Event", "Date", "In"], event_rows), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with next_tab:
    left, right = st.columns([1.15, 1.0], gap="large")
    with left:
        st.markdown("<div class='card'><div class='section-title'>NEXT MAP</div>", unsafe_allow_html=True)
        st.markdown(f"**Most likely next ➜ {core['next_q']}**")
        st.markdown(f"**Path to Next Q ➜ {core['current_q']} → {core['next_q']}**")
        st.markdown(f"**Status ➜ {core['path_status']}**")
        st.markdown(f"**Transition Conviction ➜ {pct(core['transition_conviction'])}**")
        st.markdown(f"**Stay Probability ➜ {pct(core['stay_probability'])}**")
        st.markdown(f"**Transition Pressure ➜ {pct(core['transition_pressure'])}**")
        entry_quality = "High" if core["confidence"] > 0.65 and core["fragility"] < 0.4 else ("Medium" if core["confidence"] > 0.45 else "Low")
        rotation_timing = "Building" if core["transition_conviction"] > 0.42 else "Early / watch"
        hold_bias = "Short-Medium" if core["fragility"] > 0.5 else ("Medium" if core["confidence"] > 0.45 else "Short")
        invalid_window = "Tight" if core["fragility"] > 0.55 else ("Normal" if core["confidence"] > 0.45 else "Tight")
        st.markdown(f"**Entry Quality ➜ {entry_quality}**")
        st.markdown(f"**Rotation Timing ➜ {rotation_timing}**")
        st.markdown(f"**Hold Bias ➜ {hold_bias}**")
        st.markdown(f"**Invalidation Window ➜ {invalid_window}**")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'><div class='section-title'>TRANSITION TREE MINI</div>", unsafe_allow_html=True)
        alt2 = "Q3" if core["next_q"] != "Q3" else "Q4"
        tree_rows = [
            ["Base path", f"{core['current_q']} → {core['next_q']}"],
            ["Alt path 1", f"Stay in {core['current_q']}"],
            ["Alt path 2", f"{core['current_q']} → {alt2}"],
        ]
        st.markdown(table_html(["Path", "Read"], tree_rows), unsafe_allow_html=True)
        st.write("")
        st.markdown(f"**Higher-top risk ➜ {pct(core['higher_top'])}**")
        st.markdown(f"**Lower-bottom risk ➜ {pct(core['lower_bottom'])}**")
        st.markdown("</div>", unsafe_allow_html=True)

with playbook_tab:
    left, right = st.columns([1.1, 1.0], gap="large")
    with left:
        st.markdown("<div class='card'><div class='section-title'>CURRENT vs NEXT PLAYBOOK</div>", unsafe_allow_html=True)
        rows = []
        for bucket_name in ["US Stocks", "Futures / Commodities", "Forex", "Crypto", "IHSG"]:
            rows.append([bucket_name, ", ".join(play_cur[bucket_name]), ", ".join(play_next[bucket_name])])
        st.markdown(table_html(["Bucket", "Current", "Next"], rows), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'><div class='section-title'>POSITIONING / INVALIDATION</div>", unsafe_allow_html=True)
        winners = ", ".join(play_cur["US Stocks"])
        losers = "beta if fragility rises"
        st.markdown(f"**Positioning posture ➜ {posture}**")
        st.markdown(f"**Winners ➜ {winners}**")
        st.markdown(f"**Losers ➜ {losers}**")
        st.markdown("**Invalidation mini-box**")
        st.markdown("- Invalid if growth breadth improves sharply")
        st.markdown("- Invalid if inflation re-accelerates against the current path")
        st.markdown("- Invalid if shock override turns active")
        st.markdown("</div>", unsafe_allow_html=True)

with relative_tab:
    left, right = st.columns([1.10, 1.05], gap="large")
    with left:
        st.markdown("<div class='card'><div class='section-title'>RELATIVE</div>", unsafe_allow_html=True)
        rel_rows = [[r["Lens"], r["Direction"], r["Strength"], r["State"], r["Quality"], r["Sustainability"], r["Confirmation"]] for r in relative_rows]
        st.markdown("**RELATIVE MAP**")
        st.markdown(table_html(["Relative Lens", "Dir", "Str", "State", "Quality", "Sustain", "Confirm"], rel_rows), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'><div class='section-title'>SIZE ROTATION</div>", unsafe_allow_html=True)
        sr_rows = [[r["Lens"], r["Direction"], r["Strength"], r["State"], r["Quality"], r["Sustainability"], r["Confirmation"]] for r in size_rows]
        st.markdown(table_html(["Rotation Lens", "Dir", "Str", "State", "Quality", "Sustain", "Confirm"], sr_rows), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with shocks_tab:
    s1, s2, s3 = st.columns([0.95, 1.10, 0.95], gap="large")
    with s1:
        st.markdown("<div class='card'><div class='section-title'>SHOCK STATUS</div>", unsafe_allow_html=True)
        st.markdown(f"**Current mode ➜ {'Override active' if override_active else 'Base case / watch only'}**")
        for k, v in shocks.items():
            st.markdown(f"- **{k}**: {v[0]} — {v[1]}")
        st.markdown("</div>", unsafe_allow_html=True)

    with s2:
        st.markdown("<div class='card'><div class='section-title'>TRANSMISSION / CORRELATION</div>", unsafe_allow_html=True)
        st.markdown(table_html(["Scenario", "Read", "Prefer", "Avoid"], [list(x) for x in WHAT_IF_SCENARIO_MATRIX]), unsafe_allow_html=True)
        st.write("")
        st.markdown(table_html(["Condition", "Interpretation"], [list(x) for x in DIVERGENCE_RULES]), unsafe_allow_html=True)
        st.write("")
        st.markdown(table_html(["Transmission", "Strength", "Why"], [list(x) for x in CORRELATION_TRANSMISSION_PRIORS]), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with s3:
        st.markdown("<div class='card'><div class='section-title'>CRASH / RECOVERY</div>", unsafe_allow_html=True)
        st.markdown(table_html(["Crash type", "Read"], [list(x) for x in CRASH_TYPES]), unsafe_allow_html=True)
        st.write("")
        st.markdown(table_html(["Flag", "Meaning"], [list(x) for x in FALSE_RECOVERY_MAP]), unsafe_allow_html=True)
        st.write("")
        st.markdown(f"**Fear & Greed ➜ {fg_score} ({fg_vibe})**")
        st.markdown(f"**IWM read ➜ {iwm_overlay}**")
        st.markdown("</div>", unsafe_allow_html=True)

with notes_tab:
    st.markdown("<div class='card'><div class='section-title'>NOTES</div>", unsafe_allow_html=True)
    st.markdown(f"""
- **Core model actually used**: `{CORE_NAME}`
- **Reading order**: **Current → Next → Playbook → Relative → Shocks / What-If**
- **US small-cap rotation** now uses **IWM/IWB** and **IWM/SPY**
- **IHSG size rotation** is removed until a proper local small-vs-big split is available
- **Crypto alts vs BTC** uses a broader **alt basket proxy**
- **Crypto vs Liquidity** uses a lighter **liquidity composite**, not just BTC vs QQQ
""")
    st.markdown("</div>", unsafe_allow_html=True)

st.caption("Deep-audited version. Visual stays the same. Old branch behavior removed. Core stays on Baseline_Blended_Core.")
