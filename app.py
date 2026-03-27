
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

# --------------------
# DATA
# --------------------
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
    "USD": fred_series("DTWEXBGS"),
}

# --------------------
# CORE ENGINE (single source of truth)
# --------------------
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

    agreement = clamp01(1.0 - 0.5 * sum(abs(monthly[k] - quarterly[k]) for k in monthly))
    confidence = clamp01(0.55 * current_p + 0.45 * agreement)

    phase_strength = clamp01(abs(g_m) * 0.45 + abs(i_m) * 0.35 + max(0, current_p - 0.25) * 0.70)
    growth_breadth = clamp01(sum(1 for x in growth_inputs if x > 0) / len(growth_inputs))
    infl_breadth = clamp01(sum(1 for x in infl_inputs if x > 0) / len(infl_inputs))
    breadth = 0.5 * growth_breadth + 0.5 * infl_breadth
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
        "signal_quality": signal_quality_label(confidence, agreement, fragility),
    }

core = compute_core()

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
# EVENTS
# --------------------
today = date.today()
events = [("NFP", today + timedelta(days=11)), ("CPI", today + timedelta(days=21)), ("PPI", today + timedelta(days=22))]
event_rows = [[name, dt.isoformat(), f"{(dt - today).days}d"] for name, dt in events]

# --------------------
# RENDER
# --------------------


def stage_now_label(core: Dict[str, object]) -> str:
    q = core["current_q"]
    top = float(core["top_score"])
    bottom = float(core["bottom_score"])
    breadth = float(core["breadth"])
    if q == "Q3":
        if top < 0.38:
            return "Early Q3"
        if top < 0.72:
            return "Mid Q3"
        return "Late Q3"
    if q == "Q2":
        if breadth < 0.45:
            return "Early Q2"
        if top < 0.62:
            return "Mid Q2"
        return "Late Q2"
    if q == "Q4":
        return "Early Q4" if bottom < 0.28 else ("Mid Q4" if bottom < 0.55 else "Late Q4")
    return "Early Q1" if breadth < 0.48 else ("Mid Q1" if breadth < 0.62 else "Late Q1")


def variant_now_label(core: Dict[str, object]) -> str:
    q = core["current_q"]
    if q == "Q3":
        return "Bad reflation" if core["stress_infl"] > 0.75 or "stagflation" in str(core["sub_phase"]).lower() else "Cleaner Q3"
    if q == "Q2":
        if core["stress_infl"] > 0.75 and core["fragility"] > 0.42:
            return "Dirty / fragile Q2"
        return "Healthy Q2"
    if q == "Q4":
        return "Capitulation / slowdown" if core["bottom_score"] > 0.45 else "Grinding slowdown"
    return "Recovery attempt" if core["breadth"] > 0.50 else "False dawn risk"


def next_read_label(core: Dict[str, object], size_rows: List[Dict[str, str]]) -> str:
    nxt = core["next_q"]
    if nxt == "Q2":
        small_confirm = sum(1 for r in size_rows[:2] if r["Direction"] == "Small > Big" or r["Direction"] == "Small > Broad")
        if variant_now_label(core) == "Bad reflation" or core["top_score"] > 0.65:
            return "Early Q2 (fragile first, not clean yet)"
        if small_confirm >= 1 and core["breadth"] > 0.48:
            return "Early Q2 (healthier reflation)"
        return "Early Q2 (needs breadth + small-cap confirmation)"
    if nxt == "Q4":
        return "Q4 risk (growth scare / cleaner washout)"
    if nxt == "Q3":
        return "Stay in Q3 / pressure persists"
    return "Q1 recovery attempt"


def action_bias(core: Dict[str, object]) -> str:
    q = core["current_q"]
    if q == "Q3":
        return "Defensive-selective; jangan kejar broad beta sampai breadth + small caps confirm."
    if q == "Q2":
        return "Risk-on selektif; size up hanya kalau breadth tetap melebar."
    if q == "Q4":
        return "Defensive / preserve capital; tunggu base yang lebih bersih."
    return "Recovery bias, tapi tetap tunggu confirmation breadth."


def next_catalyst_text(event_rows: List[List[str]]) -> str:
    return " · ".join([f"{name} in {when}" for name, _, when in event_rows[:3]])


def scenario_state_rows(core: Dict[str, object], relative_rows: List[Dict[str, str]], size_rows: List[Dict[str, str]]) -> List[List[str]]:
    variant = variant_now_label(core)
    us_em = relative_rows[0]
    ihsg_us = relative_rows[1]
    crypto_liq = relative_rows[3]
    small_big = size_rows[0]
    alt_btc = size_rows[2]
    gold_state = "Pressured" if core["current_q"] == "Q3" and variant == "Bad reflation" else ("Usable" if core["current_q"] == "Q3" else "Not primary")
    em_state = "Conditional" if us_em["Read"] in ["Still under pressure", "No strong edge yet"] else "Improving"
    small_state = "Pre-confirming next" if small_big["Read"] == "Edge looks usable" and core["next_q"] == "Q2" else "Not confirmed yet"
    duration_state = "Tactical only" if core["stress_infl"] > 0.65 else "Helpful"
    crypto_state = "Strong but stretched" if "stretched" in crypto_liq["Read"].lower() or alt_btc["Quality"] == "Frothy" else ("Selective long only" if crypto_liq["Direction"] == "Stronger" else "Not confirmed")
    ihsg_state = "Conditional" if ihsg_us["Read"] == "No strong edge yet" else ("Improving" if "usable" in ihsg_us["Read"].lower() else "Still mixed")
    return [
        ["Gold in Q3", gold_state, "Selective / no chase", "USD capek + yields tidak makin keras + oil stabil", "USD kuat + yields keras + oil shock"],
        ["EM / IHSG", f"{em_state} / {ihsg_state}", "Tactical only until USD pressure cools", "USD kalem + breadth melebar + commodities confirm", "USD brutal + EM breadth tetap sempit"],
        ["US small caps", small_state, "Watchlist for next-Q confirmation", "Small caps ikut lead + breadth broadens", "Headline Q2 tapi small caps tetap tertinggal"],
        ["Duration / bonds", duration_state, "Tactical hedge / not hero trade", "Growth scare naik dan yields mulai capek", "Inflation shock bertahan / yields terus naik"],
        ["Crypto beta", crypto_state, "Selective, not broad chase", "Liquidity + breadth improve", "Likuiditas lemah dan alts terlalu frothy"],
    ]


def crash_meter_now(core: Dict[str, object]) -> float:
    q_bump = {"Q1": 0.10, "Q2": 0.20, "Q3": 0.34, "Q4": 0.28}[core["current_q"]]
    x = 0.34 * core["top_score"] + 0.26 * core["stress_liq"] + 0.20 * core["fragility"] + 0.10 * core["transition_pressure"] + 0.10 * q_bump
    return clamp01(x)


def cross_asset_rows(core: Dict[str, object], play_cur: Dict[str, List[str]], play_next: Dict[str, List[str]]) -> List[List[str]]:
    rows = []
    area_order = ["Futures / Commodities", "Forex", "US Stocks", "Crypto", "IHSG"]
    for area in area_order:
        if area == "Forex":
            label = "Cross-Asset / FX"
            now_use = ", ".join(play_cur["Forex"][:2])
            nxt_use = ", ".join(play_next["Forex"][:2])
        else:
            label = area
            now_use = ", ".join(play_cur[area][:2])
            nxt_use = ", ".join(play_next[area][:2])
        if area == "Futures / Commodities":
            confirm = "Oil / gold decide apakah ini cleaner reflation atau toxic inflation turn."
        elif area == "Forex":
            confirm = "USD harus kalem dulu untuk high-beta FX / EM jadi lebih usable."
        elif area == "US Stocks":
            confirm = "Breadth + small caps harus confirm kalau next-Q2 mau jadi sehat."
        elif area == "Crypto":
            confirm = "Liquidity + breadth harus improve; kalau tidak tetap tactical."
        else:
            confirm = "IHSG membaik kalau USD pressure turun dan EM/commodity breadth confirm."
        rows.append([label, now_use, nxt_use, confirm])
    return rows


def fx_rank_rows(core: Dict[str, object]) -> List[List[str]]:
    variant = variant_now_label(core)
    q = core["current_q"]
    if q == "Q3" and variant == "Bad reflation":
        order = [
            ("USD", "Strong", "Long / overweight", "Safe-haven + higher-rate bias"),
            ("JPY", "Above avg", "Lean long vs weak FX", "Defensive / funding support"),
            ("CHF", "Above avg", "Lean long vs weak FX", "Defensive quality"),
            ("EUR", "Mixed", "Pair-selective", "Not clean leader"),
            ("GBP", "Mixed", "Pair-selective", "Not clean leader"),
            ("CAD", "Below avg", "Selective only", "Commodity help but USD pressure matters"),
            ("AUD", "Below avg", "Lean short vs strong FX", "China / beta sensitivity"),
            ("NZD", "Weak", "Short / funding leg", "High beta under pressure"),
            ("EMFX / IDR", "Weak", "Short / avoid broad long", "USD + external pressure"),
        ]
    elif core["current_q"] == "Q2":
        order = [
            ("AUD", "Strong", "Long / overweight", "Commodity / cyclical reflation"),
            ("CAD", "Strong", "Long / overweight", "Commodity support"),
            ("NZD", "Above avg", "Lean long", "High-beta reflation"),
            ("GBP", "Mixed", "Pair-selective", "Growth but not pure leader"),
            ("EUR", "Mixed", "Pair-selective", "Growth but not pure leader"),
            ("USD", "Neutral", "Funding only", "Should not be too brutal in healthy Q2"),
            ("JPY", "Below avg", "Lean short", "Funding weakens if reflation is clean"),
            ("CHF", "Below avg", "Lean short", "Defensive demand fades"),
            ("EMFX / IDR", "Above avg", "Tactical long", "Improves only if USD cools"),
        ]
    else:
        order = [
            ("USD", "Above avg", "Long / defensive", "Default anchor"),
            ("JPY", "Mixed", "Pair-selective", "Defensive but not clean"),
            ("CHF", "Mixed", "Pair-selective", "Defensive but not clean"),
            ("EUR", "Mixed", "Neutral", "No clean edge"),
            ("GBP", "Mixed", "Neutral", "No clean edge"),
            ("CAD", "Mixed", "Neutral", "Commodity-sensitive"),
            ("AUD", "Below avg", "Lean short", "Beta-sensitive"),
            ("NZD", "Below avg", "Lean short", "Beta-sensitive"),
            ("EMFX / IDR", "Weak", "Avoid broad long", "External sensitivity"),
        ]
    return [list(x) for x in order]


# --------------------
# RENDER
# --------------------
st.title(APP_NAME)
st.markdown("<div class='small-muted'>Core alpha engine: Baseline_Blended_Core • Q3-anchored decision support shell • Live backbone: FRED + optional Yahoo</div>", unsafe_allow_html=True)
st.write("")

stage_now = stage_now_label(core)
variant_now = variant_now_label(core)
next_profile = next_read_label(core, size_rows)
action_now = action_bias(core)
crash_now = crash_meter_now(core)

hero_cols = st.columns(6)
hero_items = [
    ("Decision Regime", core["current_q"], pill_html(stage_now)),
    ("Confidence", pct(core["confidence"]), pill_html(f"Agreement {pct(core['agreement'])}")),
    ("Variant Now", variant_now, pill_html(core["sub_phase"])),
    ("Next Most Likely", core["next_q"], pill_html(next_profile)),
    ("Crash Meter", pct(crash_now), pill_html("Watch only" if crash_now < 0.45 else ("Elevated" if crash_now < 0.62 else "Respect risk"), red=crash_now>=0.62)),
    ("Action Bias", "Selective", pill_html(action_now)),
]
for col, (title, value, sub_html) in zip(hero_cols, hero_items):
    with col:
        st.markdown(f"""
        <div class='hero-card'>
          <div class='metric-title'>{title}</div>
          <div style='font-size:1.35rem;font-weight:800;line-height:1.15'>{value}</div>
          <div class='metric-sub'>{sub_html}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown(
    f"<div class='note-box'><b>Quick read:</b> Sekarang <b>{stage_now} / {variant_now}</b>. Base case masih <b>{core['current_q']}</b>, "
    f"tapi kalau transition lanjut, jalur paling mungkin adalah <b>{next_profile}</b>. Action bias sekarang: <b>{action_now}</b></div>",
    unsafe_allow_html=True,
)
st.write("")

main_tabs = st.tabs(["Decision", "Cross-Asset", "Risk / Relative", "Details"])

with main_tabs[0]:
    st.markdown("<div class='card'><div class='section-title'>DECISION SNAPSHOT</div>", unsafe_allow_html=True)
    st.markdown("<div class='mini-caption'>Satu panel inti: sekarang di mana, kalau next menang bakal seperti apa, apa yang harus dikonfirmasi, dan event apa yang paling bisa mengubah bacaannya.</div>", unsafe_allow_html=True)

    summary_rows = [
        ["Now", f"{stage_now} / {core['current_q']}", "Fase operasional sekarang"],
        ["Variant now", variant_now, "Bentuk regime sekarang: clean atau toxic"],
        ["If next wins", next_profile, "Read paling realistis kalau transisi lanjut"],
        ["Top / bottom state", f"{ladder_state(core['top_score'], 'top')} / {ladder_state(core['bottom_score'], 'bottom')}", "Top tinggi = jangan kejar atas. Bottom rendah = belum ada washout bersih."],
        ["Crash meter now", pct(crash_now), "Hazard rate kasar, bukan exact timing crash"],
        ["Next catalysts", next_catalyst_text(event_rows), "Checkpoint untuk konfirmasi Q sekarang vs next-Q"],
    ]
    st.markdown(table_html(["Focus", "Read", "Why it matters"], summary_rows), unsafe_allow_html=True)
    st.write("")

    priority_rows = [
        ["Highest", "Futures / Commodities", ", ".join(play_cur["Futures / Commodities"][:2]), ", ".join(play_next["Futures / Commodities"][:2]), "Oil / gold decide apakah ini cleaner reflation atau toxic inflation turn."],
        ["High", "Cross-Asset / FX", ", ".join(play_cur["Forex"][:2]), ", ".join(play_next["Forex"][:2]), "USD harus kalem dulu untuk high-beta FX / EM jadi lebih usable."],
        ["Medium", "US Stocks", ", ".join(play_cur["US Stocks"][:2]), ", ".join(play_next["US Stocks"][:2]), "Breadth + small caps harus confirm kalau next-Q2 mau jadi sehat."],
        ["Lower", "Crypto", ", ".join(play_cur["Crypto"][:2]), ", ".join(play_next["Crypto"][:2]), "Liquidity + breadth harus improve; kalau tidak tetap tactical."],
        ["Low", "IHSG", ", ".join(play_cur["IHSG"][:2]), ", ".join(play_next["IHSG"][:2]), "IHSG membaik kalau USD pressure turun dan EM / commodity breadth confirm."],
    ]
    st.markdown(table_html(["Priority", "Area", "Use now", "If next wins", "Confirm first"], priority_rows), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with main_tabs[1]:
    st.markdown("<div class='card'><div class='section-title'>CROSS-ASSET DIRECTIONAL BIAS</div>", unsafe_allow_html=True)
    st.markdown("<div class='mini-caption'>Panel operasional: stage sekarang, bucket yang paling kuat/lemah, dan cara mengekspresikannya lintas aset / FX.</div>", unsafe_allow_html=True)
    strongest = ", ".join(play_cur["Futures / Commodities"][:1] + play_cur["US Stocks"][:1])
    weakest = ", ".join(play_next["Crypto"][:1] + ["broad beta"])
    st.markdown(f"**Current cycle stage ➜ {stage_now}**")
    st.markdown(f"**Strongest now ➜ {strongest}**")
    st.markdown(f"**Weakest / avoid-chase now ➜ {weakest}**")
    st.write("")
    st.markdown(table_html(["Area", "Bias now", "How to use", "If next wins"], cross_asset_rows(core, play_cur, play_next)), unsafe_allow_html=True)
    st.write("")
    st.markdown("**FX rank (strong → weak)**")
    st.markdown(table_html(["Currency", "Bias", "Expression", "Why now"], fx_rank_rows(core)), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with main_tabs[2]:
    st.markdown("<div class='card'><div class='section-title'>RISK STACK</div>", unsafe_allow_html=True)
    st.markdown("<div class='mini-caption'>Relative + participation + scenario state-now + shock overlay digabung biar bacanya tidak loncat-loncat.</div>", unsafe_allow_html=True)

    risk_top = [
        ["Relative leadership", relative_rows[0]["Read"], "US/EM tetap jadi pembeda paling penting sekarang"],
        ["Participation", size_rows[0]["Read"], "Small caps membantu konfirmasi kalau next-Q2 mau sehat"],
        ["Shock overlay", "Base case / watch only" if not override_active else "Override active", "Shock bukan phase utama, tapi bisa membatalkan read utama"],
        ["Crash branch", "Watch only" if crash_now < 0.45 else ("Elevated" if crash_now < 0.62 else "Respect risk"), "Semakin tinggi = makin hormat ke drawdown / accident window"],
    ]
    st.markdown(table_html(["Stack", "Read", "Why it matters"], risk_top), unsafe_allow_html=True)
    st.write("")

    st.markdown("**Scenario state-now**")
    st.markdown(table_html(["Scenario", "State now", "How to use now", "What flips it", "Still bad when"], scenario_state_rows(core, relative_rows, size_rows)), unsafe_allow_html=True)
    st.write("")

    simple_rel = [[r["Lens"], f"{r['Direction']} / {r['Strength']}", interpret_relative(r["Direction"], r["State"], r["Quality"]), f"If next wins: {('EM-friendly if USD calms' if r['Lens']=='US/EM' else 'Small should lead' if r['Lens']=='US Small/Big' else 'Bias shifts only if confirmation improves')}" ] for r in relative_rows[:3]]
    st.markdown("**Relative snapshot**")
    st.markdown(table_html(["Lens", "Bias now", "Simple read", "If next wins"], simple_rel), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with main_tabs[3]:
    st.markdown("<div class='card'><div class='section-title'>DETAILS</div>", unsafe_allow_html=True)
    with st.expander("Show shock map + divergences + model notes", expanded=False):
        st.markdown("**Shock watch**")
        for k, v in shocks.items():
            st.markdown(f"- **{k}**: {v[0]} — {v[1]}")
        st.write("")
        st.markdown(table_html(["Scenario", "Read", "Prefer", "Avoid"], [list(x) for x in WHAT_IF]), unsafe_allow_html=True)
        st.write("")
        st.markdown(table_html(["Condition", "Interpretation"], [list(x) for x in DIVERGENCE]), unsafe_allow_html=True)
        st.write("")
        st.markdown(table_html(["Transmission", "Strength", "Why"], [list(x) for x in CORR]), unsafe_allow_html=True)
        st.write("")
        st.markdown(table_html(["Crash type", "Read"], [list(x) for x in CRASH]), unsafe_allow_html=True)
        st.write("")
        st.markdown(table_html(["Flag", "Meaning"], [list(x) for x in FALSE_REC]), unsafe_allow_html=True)
        st.write("")
        st.markdown(f"**Fear & Greed ➜ {fg_score} ({fg_vibe})**")
        st.markdown(f"**IWM read ➜ {iwm_overlay}**")
        st.markdown(f"**Core model actually used ➜ {CORE_NAME}**")
        st.markdown("- Dashboard ini fokus ke decision support, bukan execution engine penuh.")
        st.markdown("- Crash meter = hazard window, bukan exact timing crash.")
        st.markdown("- Next-Q baru dianggap sehat kalau breadth + small caps + leadership mulai confirm.")
    st.markdown("</div>", unsafe_allow_html=True)

st.caption("Decision shell condensed. Yang berkorelasi sudah digabung: Decision Snapshot, Risk Stack, dan Cross-Asset Directional Bias.")
