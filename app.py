
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

# =========================
# STYLE: freeze visual shell
# =========================
st.markdown("""
<style>
:root {
  --bg:#07101f;
  --card:#0c1526;
  --line:#263246;
  --muted:#9fb0c8;
  --text:#f3f6fb;
}
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg);
  color: var(--text);
}
.block-container {padding-top: 1.6rem; padding-bottom: 2.6rem;}
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
.small-muted {color: var(--muted); font-size: .92rem;}
.tight-table table {width:100%; border-collapse: collapse; table-layout: fixed;}
.tight-table th, .tight-table td {
  border:1px solid #243147;
  padding:8px 10px;
  font-size:.90rem;
  text-align:left;
  vertical-align:top;
  word-wrap: break-word;
}
.tight-table th {color:#9fb0c8; font-weight:700; background: rgba(20,29,44,.85);}
.tight-table td {color:#f5f8fd;}
.note-box {
  border:1px solid #27476e;
  background: rgba(18,46,79,.65);
  border-radius:12px;
  padding:12px 14px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def pct(x: float) -> str:
    return f"{100*x:.1f}%"

def pill_html(txt: str) -> str:
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
def yahoo_close(ticker: str, period: str = "1y") -> pd.Series:
    if yf is None:
        return pd.Series(dtype=float, name=ticker)
    try:
        data = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
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

def ret_n(s: pd.Series, n: int = 21) -> float:
    if s.empty or len(s) < n + 1:
        return 0.0
    return float(s.iloc[-n:].pct_change().add(1).prod() - 1)

def stretch_n(s: pd.Series, n: int = 63) -> float:
    if s.empty or len(s) < n:
        return 0.0
    return float((s.iloc[-1] / s.iloc[-n:].mean()) - 1)

# =========================
# DATA BACKBONE
# =========================
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
    "M2": fred_series("M2SL"),
    "USD_BROAD": fred_series("DTWEXBGS"),
}

# =========================
# ONE CORE ENGINE ONLY
# =========================
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

    # quarterly anchor / smoother
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

    margin = current_p - next_p
    if margin < 0.03:
        fragility = clamp01(fragility + 0.08)
    if margin < 0.015:
        confidence = clamp01(confidence - 0.05)

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
        "margin": margin,
    }

core = compute_core()

# =========================
# RELATIVE / SIZE ENGINES
# =========================
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
    # distinct fallback so rows do NOT collapse into the same output
    strength_num = clamp01(0.20 + 0.07 * seed + 0.28 * core["breadth"] + 0.12 * (1 - core["fragility"]))
    spread = (seed - 1.5) * 0.035 + (core["phase_strength"] - 0.5) * 0.04
    if spread > 0.025:
        direction = "Stronger"
    elif spread < -0.025:
        direction = "Weaker"
    else:
        direction = "Balanced"
    state = ["Fading", "Peaking", "Stable", "Building"][(seed + int(core["phase_strength"] * 10)) % 4]
    stretch = clamp01(0.03 + 0.025 * seed + core["top_score"] * 0.12)
    quality = rel_quality(strength_num, core["breadth"], stretch)
    sustain = rel_sustainability(strength_num, quality, core["fragility"])
    confirm = rel_confirmation(strength_num, state, core["breadth"])
    return {
        "Lens": name,
        "Direction": direction,
        "Strength": score_to_label(strength_num),
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

    if spread > 0.02:
        direction = pos_label
    elif spread < -0.02:
        direction = neg_label
    else:
        direction = "Balanced"

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
        "Strength": score_to_label(strength_num),
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
        walcl = SER["WALCL"].dropna()
        parts.append((walcl / walcl.iloc[0]).rename("WALCL"))
    if not SER["M2"].empty:
        m2 = SER["M2"].dropna()
        parts.append((m2 / m2.iloc[0]).rename("M2"))
    if not SER["USD_BROAD"].empty:
        usd = SER["USD_BROAD"].dropna()
        parts.append((1 / usd).rename("USD_INV"))
    if not SER["NFCI"].empty:
        nfci = SER["NFCI"].dropna()
        if not nfci.empty:
            z = (nfci - nfci.mean()) / max(nfci.std(ddof=0), 1e-9)
            parts.append((1 / (1 + np.exp(z))).rename("NFCI_INV"))
    if not parts:
        return pd.Series(dtype=float, name="LIQ")
    df = pd.concat(parts, axis=1).dropna(how="all")
    return df.mean(axis=1).dropna().rename("LIQ")

def compute_relative() -> List[Dict[str, str]]:
    spy = yahoo_close("SPY", "1y")
    eem = yahoo_close("EEM", "1y")
    eido = yahoo_close("EIDO", "1y")
    btc = yahoo_close("BTC-USD", "1y")
    liq = liquidity_composite()

    return [
        build_rel_row("US vs EM", spy, eem, "US stronger", "EM stronger", core["growth_breadth"], 0),
        build_rel_row("IHSG vs US", eido, spy, "IHSG stronger", "US stronger", core["breadth"], 1, " (proxy-based)"),
        build_rel_row("IHSG vs EM", eido, eem, "IHSG stronger", "EM stronger", core["breadth"], 2, " (proxy-based)"),
        build_rel_row("Crypto vs Liquidity", btc, liq, "Crypto stronger", "Liquidity not confirming", core["phase_strength"], 3, " (liquidity composite proxy)"),
    ]

def compute_size_rotation() -> List[Dict[str, str]]:
    iwm = yahoo_close("IWM", "1y")
    iwb = yahoo_close("IWB", "1y")
    spy = yahoo_close("SPY", "1y")
    alt_basket = crypto_alt_basket("1y")
    btc = yahoo_close("BTC-USD", "1y")

    return [
        build_rel_row("US Small Caps vs Big Caps (IWM/IWB)", iwm, iwb, "Small > Big", "Big > Small", core["breadth"], 4),
        build_rel_row("US Small Caps vs Broad Market (IWM/SPY)", iwm, spy, "Small > Broad", "Broad > Small", core["breadth"], 5),
        build_rel_row("Crypto Alt Basket vs BTC", alt_basket, btc, "Alts > BTC", "BTC > Alts", core["breadth"], 6, " (basket proxy)"),
    ]

relative_rows = compute_relative()
size_rows = compute_size_rotation()

# =========================
# OVERLAYS
# =========================
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

# =========================
# PLAYBOOK
# =========================
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

    posture = (
        "Aggressive" if core["phase_strength"] > 0.65 and core["fragility"] < 0.35
        else "Balanced" if core["confidence"] > 0.45
        else "Defensive" if core["fragility"] > 0.55
        else "Wait / low conviction"
    )
    return cur_out, nxt_out, posture

play_cur, play_next, posture = current_vs_next_playbook()

# =========================
# EVENT WATCH
# =========================
today = date.today()
events = [
    ("NFP", today + timedelta(days=11)),
    ("CPI", today + timedelta(days=21)),
    ("PPI", today + timedelta(days=22)),
]
event_rows = [[name, dt.isoformat(), f"{(dt - today).days}d"] for name, dt in events]

# =========================
# LAYOUT (freeze)
# =========================
st.title(APP_NAME)
st.markdown("<div class='small-muted'>Core alpha engine: Baseline_Blended_Core • Visual shell stays fixed • Current / Next / Playbook / Relative / Shocks</div>", unsafe_allow_html=True)
st.write("")

hero_cols = st.columns(5)
hero_items = [
    ("Current Phase", core["current_q"], pill_html("Decaying") if core["fragility"] > 0.55 else pill_html("Stable")),
    ("Confidence", pct(core["confidence"]), pill_html(f"Agreement {pct(core['agreement'])}")),
    ("Sub-Phase", core["sub_phase"], pill_html(f"Strength {pct(core['phase_strength'])}")),
    ("Top Risk", pct(core["top_score"]), pill_html(f"Higher-top {pct(core['higher_top'])}")),
    ("Bottom Risk", pct(core["bottom_score"]), pill_html(f"Lower-bottom {pct(core['lower_bottom'])}")),
]
for col, (title, value, sub_html) in zip(hero_cols, hero_items):
    with col:
        st.markdown(f"""
        <div class='hero-card'>
          <div class='metric-title'>{title}</div>
          <div class='metric-value'>{value}</div>
          <div class='metric-sub'>{sub_html}</div>
        </div>
        """, unsafe_allow_html=True)

mini_cols = st.columns(5)
mini = [
    ("CURRENT", core["current_q"], "Decaying" if core["fragility"] > 0.55 else "Stable"),
    ("NEXT", core["next_q"], f"Hazard {pct(core['transition_pressure'])}"),
    ("PLAYBOOK", ", ".join(play_cur["US Stocks"][:1]), f"Conviction {pct(core['confidence'])}"),
    ("RELATIVE", relative_rows[0]["Read"], relative_rows[1]["Read"]),
    ("SHOCKS", "Overlay", f"Top {pct(core['top_score'])} / Bottom {pct(core['bottom_score'])}"),
]
for col, (title, value, sub) in zip(mini_cols, mini):
    with col:
        st.markdown(f"""
        <div class='hero-card'>
          <div class='metric-title'>{title}</div>
          <div style='font-size:1.2rem;font-weight:800'>{value}</div>
          <div class='metric-sub'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

left_col, right_col = st.columns([1.05, 0.95], gap="large")
with left_col:
    current_tab, next_tab, playbook_tab = st.tabs(["Current", "Next", "Playbook"])
with right_col:
    relative_tab, shocks_tab, notes_tab = st.tabs(["Relative", "Shocks / What-If", "Notes"])

with current_tab:
    c1, c2 = st.columns([1.2, 1.0], gap="large")
    with c1:
        st.markdown("<div class='card'><div class='section-title'>CURRENT MAP</div>", unsafe_allow_html=True)
        st.markdown(f"**Phase ➜ {core['current_q']}**")
        st.markdown(f"**Confidence ➜ {pct(core['confidence'])}**")
        st.markdown(f"**Agreement ➜ {pct(core['agreement'])}**")
        st.markdown(f"**Sub-Phase ➜ {core['sub_phase']}**")
        st.markdown(f"**Regime Strength ➜ {pct(core['phase_strength'])}**")
        st.markdown(f"**Breadth ➜ {pct(core['breadth'])}**")
        st.markdown(f"**Fragility ➜ {pct(core['fragility'])}**")
        explanation = (
            f"Still **{core['current_q']}** for now. Inside that, the model reads **{core['sub_phase']}**. "
            f"So this is not a flat {core['current_q']} read. Path to **{core['next_q']}** is **{core['path_status']}**, "
            f"while the current regime still looks **{'fragile' if core['fragility'] > 0.5 else 'fairly stable'}**."
        )
        st.markdown(f"<div class='note-box'>{explanation}</div>", unsafe_allow_html=True)
        st.write("")
        prob_rows = [[k, f"{v:.4f}"] for k, v in sorted(core["blended"].items(), key=lambda x: x[1], reverse=True)]
        st.markdown(table_html(["Phase", "Probability"], prob_rows), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
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
            ["Sentiment stretch", f"{fg_score}", fg_vibe],
        ]
        st.markdown("**Risk Engine Snapshot**")
        st.markdown(table_html(["Engine", "Score", "Read"], risk_rows), unsafe_allow_html=True)
        st.write("")
        st.markdown("**Event Watch**")
        st.markdown(table_html(["Event", "Date", "In"], event_rows), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with next_tab:
    n1, n2 = st.columns([1.12, 1.0], gap="large")
    with n1:
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

    with n2:
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
    p1, p2 = st.columns([1.08, 1.0], gap="large")
    with p1:
        st.markdown("<div class='card'><div class='section-title'>CURRENT vs NEXT PLAYBOOK</div>", unsafe_allow_html=True)
        rows = []
        for bucket_name in ["US Stocks", "Futures / Commodities", "Forex", "Crypto", "IHSG"]:
            rows.append([bucket_name, ", ".join(play_cur[bucket_name]), ", ".join(play_next[bucket_name])])
        st.markdown(table_html(["Bucket", "Current", "Next"], rows), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with p2:
        st.markdown("<div class='card'><div class='section-title'>POSITIONING / INVALIDATION</div>", unsafe_allow_html=True)
        st.markdown(f"**Positioning posture ➜ {posture}**")
        st.markdown(f"**Winners ➜ {', '.join(play_cur['US Stocks'])}**")
        st.markdown(f"**Losers ➜ beta if fragility rises**")
        st.markdown("**Invalidation mini-box**")
        st.markdown("- Invalid if growth breadth improves sharply")
        st.markdown("- Invalid if inflation re-accelerates against the current path")
        st.markdown("- Invalid if shock override turns active")
        st.markdown("</div>", unsafe_allow_html=True)

with relative_tab:
    r1, r2 = st.columns([1.0, 1.0], gap="large")
    with r1:
        st.markdown("<div class='card'><div class='section-title'>RELATIVE MAP</div>", unsafe_allow_html=True)
        rel_rows = []
        for row in relative_rows:
            rel_rows.append([
                row["Lens"], row["Direction"], row["Strength"], row["StrengthScore"],
                row["State"], row["Quality"], row["Sustainability"], row["Confirmation"]
            ])
        st.markdown(table_html(["Relative Lens", "Dir", "Strength", "Score", "State", "Quality", "Sustain", "Confirm"], rel_rows), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r2:
        st.markdown("<div class='card'><div class='section-title'>SIZE ROTATION</div>", unsafe_allow_html=True)
        sr_rows = []
        for row in size_rows:
            sr_rows.append([
                row["Lens"], row["Direction"], row["Strength"], row["StrengthScore"],
                row["State"], row["Quality"], row["Sustainability"], row["Confirmation"]
            ])
        st.markdown(table_html(["Rotation Lens", "Dir", "Strength", "Score", "State", "Quality", "Sustain", "Confirm"], sr_rows), unsafe_allow_html=True)
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
- **Final system shell**: `{APP_NAME}`
- **Reading order**: **Current → Next → Playbook → Relative → Shocks / What-If**
- **IHSG size rotation** is removed unless a proper small-vs-big source is available.
- **Crypto alt basket vs BTC** uses a basket proxy, not a full institutional market-cap engine.
- **Crypto vs Liquidity** uses a composite proxy (WALCL, M2, USD inverse, NFCI inverse).
- **Visual is locked**: if the screen changes a lot, that means the wrong file/version is running.
""")
    st.markdown("</div>", unsafe_allow_html=True)

st.caption("Visual is intentionally kept stable. If Q changes, it should come from the same core engine — not from a different app branch.")
