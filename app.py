
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
CORE_NAME = "Q3_Anchored_YieldAware_Core"

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
    "PCE_CORE": fred_series("PCEPILFE"),
    "PPI": fred_series("PPIACO"),
    "SAHM": fred_series("SAHMCURRENT"),
    "NFCI": fred_series("NFCI"),
    "HY": fred_series("BAMLH0A0HYM2"),
    "WTI": fred_series("DCOILWTICO"),
    "WALCL": fred_series("WALCL"),
    "M2": fred_series("M2SL"),
    "USD": fred_series("DTWEXBGS"),
    "DGS2": fred_series("DGS2"),
    "DGS10": fred_series("DGS10"),
    "DFII10": fred_series("DFII10"),
    "T10Y2Y": fred_series("T10Y2Y"),
    "T10YIE": fred_series("T10YIE"),
    "DFF": fred_series("DFF"),
    "SKEW": fred_series("SKEW"),
}

# --------------------
# CORE ENGINE (single source of truth)
# --------------------
def latest_value(s: pd.Series, default: float = 0.0) -> float:
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else float(default)


def diff_pos(s: pd.Series, n: int) -> pd.Series:
    return s.diff(n).clip(lower=0)


def compute_core() -> Dict[str, object]:
    # Base macro blocks
    growth_inputs = [
        robust_z(SER["INDPRO"].pct_change(12)),
        robust_z(SER["RSAFS"].pct_change(12)),
        robust_z(SER["PAYEMS"].pct_change(12)),
        -robust_z(SER["UNRATE"].diff()),
        -robust_z(SER["ICSA"].pct_change(12)),
    ]
    pce_core = SER["PCE_CORE"] if not SER["PCE_CORE"].empty else SER["CORE_CPI"]
    infl_inputs = [
        robust_z(SER["CPI"].pct_change(12)),
        robust_z(pce_core.pct_change(12)),
        robust_z(SER["PPI"].pct_change(12)),
        robust_z(SER["WTI"].pct_change(12)),
    ]
    stress_inputs = [
        robust_z(SER["SAHM"]),
        robust_z(SER["NFCI"]),
        robust_z(SER["HY"]),
    ]

    # Explicit yields block
    y2 = SER["DGS2"].dropna()
    y10 = SER["DGS10"].dropna()
    y10_real = SER["DFII10"].dropna()
    curve = SER["T10Y2Y"].dropna()
    if curve.empty and (not y2.empty and not y10.empty):
        curve = (y10 - y2).dropna()
        curve.name = "T10Y2Y"
    yr_components = [
        robust_z(y2),
        robust_z(y10_real),
        robust_z(diff_pos(y2, 20)),
        robust_z(diff_pos(y10_real, 20)),
    ]
    yr = clamp01(0.20 + 0.18 * np.nanmean([max(0.0, x) for x in yr_components]))
    cs_components = [
        robust_z((-curve).clip(lower=0)),
        robust_z(diff_pos(curve, 20)),
    ]
    cs = clamp01(0.18 + 0.20 * np.nanmean([max(0.0, x) for x in cs_components]))

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
        robust_z(pce_core.pct_change(12).rolling(3).mean(), 60),
        robust_z(SER["PPI"].pct_change(12).rolling(3).mean(), 60),
        robust_z(SER["WTI"].pct_change(12).rolling(3).mean(), 60),
    ]
    g_q = float(np.nanmean(growth_q))
    i_q = float(np.nanmean(infl_q))

    # Raw model (kept for audit only)
    g_up_m = sigmoid(1.25 * g_m - 0.35 * s_m - 0.18 * yr)
    i_up_m = sigmoid(1.20 * i_m + 0.10 * s_m + 0.15 * yr)
    g_up_q = sigmoid(1.05 * g_q - 0.25 * s_m - 0.12 * yr)
    i_up_q = sigmoid(1.05 * i_q + 0.05 * s_m + 0.12 * yr)

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
    total = sum(blended.values()) or 1.0
    blended = {k: v / total for k, v in blended.items()}

    ranked = sorted(blended.items(), key=lambda x: x[1], reverse=True)
    raw_current_q, raw_current_p = ranked[0]
    raw_next_q, raw_next_p = ranked[1]

    # Decision regime: keep live regime Q3-anchored as requested
    current_q = "Q3"
    next_q = "Q2"
    current_p = blended.get(current_q, raw_current_p)
    next_p = blended.get(next_q, raw_next_p)

    agreement = clamp01(1.0 - 0.5 * sum(abs(monthly[k] - quarterly[k]) for k in monthly))
    confidence = clamp01(0.50 * current_p + 0.30 * agreement + 0.20 * (1 - yr))

    phase_strength = clamp01(abs(g_m) * 0.42 + abs(i_m) * 0.32 + max(0, current_p - 0.20) * 0.65 + 0.12 * yr)
    growth_breadth = clamp01(sum(1 for x in growth_inputs if x > 0) / len(growth_inputs))
    infl_breadth = clamp01(sum(1 for x in infl_inputs if x > 0) / len(infl_inputs))
    breadth = 0.55 * growth_breadth + 0.45 * infl_breadth
    fragility = clamp01(0.32 * (1 - current_p) + 0.24 * (1 - agreement) + 0.18 * max(0, s_m) + 0.16 * yr + 0.10 * cs)

    margin = current_p - next_p
    if current_p - next_p < 0.04:
        fragility = clamp01(fragility + 0.06)
    if current_p < 0.30:
        confidence = clamp01(confidence - 0.04)

    variant_now = "Bad reflation" if (yr > 0.52 or i_m > 0.28) else "Slowdown with sticky inflation"
    sub_phase = "Mid Q3 / bad reflation" if variant_now == "Bad reflation" else "Mid Q3 / selective-defensive"

    top_score = clamp01(0.28 * max(0, i_m) + 0.20 * phase_strength + 0.22 * fragility + 0.18 * yr + 0.12 * float(current_q in ["Q2", "Q3"]))
    bottom_score = clamp01(0.18 * max(0, -g_m) + 0.16 * (1 - fragility) + 0.14 * float(current_q == "Q4") + 0.10 * float(g_m < -0.35))
    higher_top = clamp01(top_score * 0.70 * (1 - bottom_score))
    lower_bottom = clamp01(bottom_score * 0.70 * (1 - top_score))

    transition_pressure = clamp01(0.32 * fragility + 0.24 * (1 - current_p) + 0.16 * abs(g_m - i_m) / 3.0 + 0.16 * yr + 0.12 * cs)
    transition_conviction = clamp01(0.55 * transition_pressure + 0.45 * next_p)
    stay_probability = clamp01(current_p * (1 - fragility * 0.45) + 0.10)
    path_status = "Building" if transition_conviction > 0.44 else "Starting"
    if transition_conviction > 0.62:
        path_status = "Valid"

    # Policy-liquidity engine with yields explicit
    IC = clamp01(0.50 * max(0.0, robust_z((pce_core.pct_change(12) - 0.02).dropna())) + 0.30 * max(0.0, robust_z((pce_core.pct_change(3) * 4 - 0.02).dropna())) + 0.20 * max(0.0, robust_z(SER["T10YIE"])))
    GS = clamp01(0.30 * max(0.0, -robust_z(SER["INDPRO"].pct_change(12))) + 0.25 * max(0.0, robust_z(SER["ICSA"].rolling(4).mean())) + 0.25 * max(0.0, -robust_z(SER["PAYEMS"].pct_change(12))) + 0.20 * max(0.0, -robust_z(SER["RSAFS"].pct_change(12))))
    LS = clamp01(0.60 * max(0.0, robust_z(SER["UNRATE"])) + 0.40 * max(0.0, robust_z(SER["ICSA"].rolling(4).mean())))
    FS = clamp01(0.45 * max(0.0, robust_z(SER["HY"])) + 0.35 * max(0.0, robust_z(SER["NFCI"])) + 0.20 * yr)
    RS = clamp01(0.40 * max(0.0, robust_z((SER["DFF"] - 4.0).dropna())) + 0.35 * max(0.0, robust_z(SER["WALCL"].pct_change(12) * -1)) + 0.25 * cs)
    LT = clamp01(0.55 * max(0.0, -robust_z(SER["WALCL"].pct_change(12))) + 0.25 * max(0.0, robust_z(SER["USD"])) + 0.20 * yr)
    MG = clamp01(0.55 * RS + 0.45 * FS)
    q = blended
    z_qe = (1.10 * FS + 0.90 * GS + 0.75 * LS + 0.85 * RS + 0.60 * q["Q4"] + 0.25 * q["Q3"] - 1.15 * IC - 0.70 * yr)
    z_neutral = (0.80 * IC + 0.70 * RS + 0.55 * MG + 0.40 * q["Q1"] + 0.20 * q["Q2"] - 0.45 * FS + 0.15 * yr)
    z_qt = (1.05 * IC + 0.80 * (1 - GS) + 0.60 * (1 - LT) + 0.55 * q["Q2"] + 0.25 * q["Q1"] + 0.85 * yr - 0.85 * FS - 0.70 * RS)
    if (yr > 0.75) and (IC > 0.60) and not ((FS > 0.85) and (RS > 0.85)):
        z_qe = min(z_qe, 0.10)
    if (q["Q4"] > 0.45) and (IC < 0.40) and (yr < 0.45):
        z_qe += 0.25
    # softmax
    mx = max(z_qe, z_neutral, z_qt)
    ex = np.exp(np.array([z_qe - mx, z_neutral - mx, z_qt - mx]))
    probs = ex / ex.sum()
    p_qe, p_neutral, p_qt = [float(x) for x in probs]
    adj_crash = 10 * (0.60 * p_qt + 0.20 * p_neutral - 0.40 * p_qe)
    adj_riskoff = 15 * (0.55 * p_qt + 0.15 * p_neutral - 0.30 * p_qe)
    adj_riskon = 15 * (0.60 * p_qe + 0.20 * p_neutral - 0.50 * p_qt)

    return {
        "monthly": monthly,
        "quarterly": quarterly,
        "blended": blended,
        "raw_current_q": raw_current_q,
        "raw_next_q": raw_next_q,
        "raw_current_p": raw_current_p,
        "raw_next_p": raw_next_p,
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
        "variant_now": variant_now,
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
        "yield_pressure": yr,
        "curve_stress": cs,
        "IC": IC,
        "GS": GS,
        "LS": LS,
        "FS": FS,
        "RS": RS,
        "LT": LT,
        "MG": MG,
        "P_QE": p_qe,
        "P_NEUTRAL": p_neutral,
        "P_QT": p_qt,
        "adj_crash": adj_crash,
        "adj_riskoff": adj_riskoff,
        "adj_riskon": adj_riskon,
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
st.title(APP_NAME)
st.markdown("<div class='small-muted'>Core alpha engine: Q3_Anchored_YieldAware_Core • Visual shell: mind-map card layout • Live backbone: FRED + optional Yahoo</div>", unsafe_allow_html=True)
st.write("")

hero_cols = st.columns(5)
hero_items = [
    ("Current Phase", core["current_q"], pill_html("Decaying", red=True) if core["fragility"] > 0.55 else pill_html("Stable")),
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
    ("CURRENT", core["current_q"], pill_html("Decaying", red=True) if core["fragility"] > 0.55 else pill_html("Stable")),
    ("NEXT", core["next_q"], pill_html(f"Hazard {pct(core['transition_pressure'])}")),
    ("PLAYBOOK", ", ".join(play_cur["US Stocks"][:1]), pill_html(f"Conviction {pct(core['confidence'])}")),
    ("RELATIVE", relative_rows[0]["Read"], pill_html(relative_rows[1]["Read"])),
    ("SHOCKS", "Overlay", pill_html(f"Top {pct(core['top_score'])} / Bottom {pct(core['bottom_score'])}")),
]
for col, (title, value, sub_html) in zip(mini_cols, mini):
    with col:
        st.markdown(f"""
        <div class='hero-card'>
          <div class='metric-title'>{title}</div>
          <div style='font-size:1.2rem;font-weight:800'>{value}</div>
          <div class='metric-sub'>{sub_html}</div>
        </div>
        """, unsafe_allow_html=True)

left_col, right_col = st.columns([1.0, 1.0], gap="large")
with left_col:
    t_current, t_next, t_play = st.tabs(["Current", "Next", "Playbook"])
with right_col:
    t_rel, t_shock, t_notes = st.tabs(["Relative", "Shocks / What-If", "Notes"])

with t_current:
    c1, c2 = st.columns([1.1, 0.9], gap="large")
    with c1:
        st.markdown("<div class='card'><div class='section-title'>CURRENT MAP</div>", unsafe_allow_html=True)
        st.markdown(f"**Phase ➜ {core['current_q']}**")
        st.markdown(f"**Confidence ➜ {pct(core['confidence'])}** {pill_html('Decaying', red=True) if core['fragility'] > 0.55 else pill_html('Stable')}", unsafe_allow_html=True)
        st.markdown(f"**Agreement ➜ {pct(core['agreement'])}** {pill_html('Monthly / Quarterly')}", unsafe_allow_html=True)
        st.markdown(f"**Sub-Phase ➜ {core['sub_phase']}**")
        st.markdown(f"**Regime Strength ➜ {pct(core['phase_strength'])}**")
        st.markdown(f"**Breadth ➜ {pct(core['breadth'])}**")
        st.markdown(f"**Fragility ➜ {pct(core['fragility'])}**")
        st.markdown(f"**Signal quality ➜ {core['signal_quality']}**")
        explanation = (
            f"Current = {core['current_q']} now. Sub-phase = {core['sub_phase']}. "
            f"Next = {core['next_q']} only if the transition keeps building. "
            f"Today the model still treats {core['current_q']} as current because it has the highest blended probability, "
            f"while the regime still looks {'fragile' if core['fragility'] > 0.5 else 'fairly stable'}."
        )
        st.markdown(f"<div class='note-box'>{explanation}</div>", unsafe_allow_html=True)
        st.write("")
        prob_rows = [[k, f"{v:.4f}"] for k, v in sorted(core["blended"].items(), key=lambda x: x[1], reverse=True)]
        st.markdown(table_html(["Phase", "Probability"], prob_rows), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'><div class='section-title'>MARKET / MACRO TURN PROCESS</div>", unsafe_allow_html=True)
        st.markdown("<div class='mini-caption'>Build-up toward a possible phase turn. This is not an exact price top or exact price bottom call.</div>", unsafe_allow_html=True)
        st.markdown(f"**Top state ➜ {ladder_state(core['top_score'], 'top')}**")
        st.markdown(f"**Bottom state ➜ {ladder_state(core['bottom_score'], 'bottom')}**")
        st.write("")
        ladder_rows = [
            ["Top build", pct(core["top_score"])],
            ["Higher-top risk", pct(core["higher_top"])],
            ["Bottom build", pct(core["bottom_score"])],
            ["Lower-bottom risk", pct(core["lower_bottom"])],
        ]
        st.markdown(table_html(["Turn process", "Score"], ladder_rows), unsafe_allow_html=True)
        st.write("")
        risk_rows = [
            ["Growth stress", pct(core["stress_growth"]), score_label(core["stress_growth"])],
            ["Inflation stress", pct(core["stress_infl"]), score_label(core["stress_infl"])],
            ["Liquidity stress", pct(core["stress_liq"]), score_label(core["stress_liq"])],
            ["Sentiment stretch", f"{fg_score}", fg_vibe],
        ]
        st.markdown("**Risk Engine Snapshot**")
        st.markdown(table_html(["Engine", "Score", "Read"], risk_rows), unsafe_allow_html=True)
        st.write("")
        st.markdown("**Event Watch**")
        st.markdown(table_html(["Event", "Date", "In"], event_rows), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with t_next:
    n1, n2 = st.columns([1.05, 0.95], gap="large")
    with n1:
        st.markdown("<div class='card'><div class='section-title'>NEXT MAP</div>", unsafe_allow_html=True)
        st.markdown(f"**Most likely next ➜ {core['next_q']} (not current yet)**")
        st.markdown("<div class='mini-caption'>Next = the most likely path if the transition keeps building. It is not the current phase yet.</div>", unsafe_allow_html=True)
        st.markdown(f"**Path to Next Q ➜ {core['current_q']} → {core['next_q']}**")
        st.markdown(f"**Status ➜ {core['path_status']}**")
        st.markdown(f"**Transition Conviction ➜ {pct(core['transition_conviction'])}**")
        st.markdown(f"**Stay Probability ➜ {pct(core['stay_probability'])}**")
        st.markdown(f"**Transition Pressure ➜ {pct(core['transition_pressure'])}**")
        st.markdown(f"**Why current still wins ➜ highest blended probability ({core['current_q']} = {core['current_p']:.4f})**")
        st.markdown(f"**Why next is not current yet ➜ margin vs next = {pct(max(0.0, core['margin']))}**")
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

with t_play:
    p1, p2 = st.columns([1.05, 0.95], gap="large")
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
        st.markdown(f"**Sizing posture ➜ {exposure_posture(core['confidence'], core['fragility'])}**")
        st.markdown(f"**Winners ➜ {', '.join(play_cur['US Stocks'])}**")
        st.markdown(f"**Losers ➜ beta if fragility rises**")
        st.markdown("**Invalidation mini-box**")
        st.markdown("- Invalid if growth breadth improves sharply")
        st.markdown("- Invalid if inflation re-accelerates against the current path")
        st.markdown("- Invalid if shock override turns active")
        st.markdown("</div>", unsafe_allow_html=True)

with t_rel:
    st.markdown("<div class='card'><div class='section-title'>RELATIVE & SIZE CONTEXT</div>", unsafe_allow_html=True)
    st.markdown("**RELATIVE MAP**")
    st.markdown("<div class='mini-caption'>Relative = who looks stronger right now. Use it as context, not as the main phase call.</div>", unsafe_allow_html=True)
    st.markdown("<div class='mini-caption'>Dir = direction | Str = strength bucket | Score = numeric strength | State = early/building/stable/fading | Qual = clean or messy move | Sustain = how durable it looks | Conf = how much the move is confirmed</div>", unsafe_allow_html=True)
    rel_rows = []
    for row in relative_rows:
        rel_rows.append([row["Lens"], row["Direction"], row["Strength"], row["StrengthScore"], row["State"], row["Quality"], row["Sustainability"], row["Confirmation"], interpret_relative(row["Direction"], row["State"], row["Quality"])])
    st.markdown(table_html(["Lens", "Dir", "Str", "Score", "State", "Qual", "Sustain", "Conf", "Read"], rel_rows), unsafe_allow_html=True)
    st.write("")
    st.markdown("**SIZE ROTATION**")
    st.markdown("<div class='mini-caption'>Size rotation = breadth / participation. It helps confirm the read, but it is not the phase itself.</div>", unsafe_allow_html=True)
    sr_rows = []
    for row in size_rows:
        sr_rows.append([row["Lens"], row["Direction"], row["Strength"], row["StrengthScore"], row["State"], row["Quality"], row["Sustainability"], row["Confirmation"], interpret_relative(row["Direction"], row["State"], row["Quality"])])
    st.markdown(table_html(["Lens", "Dir", "Str", "Score", "State", "Qual", "Sustain", "Conf", "Read"], sr_rows), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with t_shock:
    st.markdown("<div class='card'><div class='section-title'>SHOCKS / WHAT-IF</div>", unsafe_allow_html=True)
    risk_on_meter = clamp01(0.24 * (1 - core["fragility"]) + 0.18 * core["breadth"] + 0.12 * (1 - core["yield_pressure"]) + 0.12 * core["P_QE"] + 0.10 * (1 - core["stress_liq"]) + 0.12 * (1 if 'Healthy' in iwm_overlay else 0.4) + 0.12 * (1 if fg_score > 45 and fg_score < 75 else 0.45))
    risk_off_meter = clamp01(0.20 * core["fragility"] + 0.15 * (1 - core["breadth"]) + 0.15 * core["yield_pressure"] + 0.10 * core["stress_liq"] + 0.10 * (1 if fg_score < 45 else 0.2) + 0.10 * (1 if 'stress' in iwm_overlay.lower() else 0.35) + 0.10 * core["P_QT"] + 0.10 * core["curve_stress"])
    big_crash_meter = clamp01(0.15 * (1 if 'stress' in iwm_overlay.lower() or 'Blow-off' in iwm_overlay else 0.35) + 0.20 * (1 - core["breadth"]) + 0.20 * clamp01(core["stress_liq"] + max(0, robust_z(SER["SKEW"])) * 0.08) + 0.15 * clamp01(max(0, robust_z(SER["HY"])) + max(0, robust_z(SER["NFCI"])) * 0.5) + 0.10 * clamp01(core["top_score"]) + 0.10 * clamp01(core["yield_pressure"] + core["curve_stress"] * 0.5) + 0.10 * (1 if fg_score < 25 else 0.2))
    core_gate_blocks = sum([int((1 if 'stress' in iwm_overlay.lower() or 'Blow-off' in iwm_overlay else 0.35) > 0.60), int((1 - core["breadth"]) > 0.60), int(clamp01(core["stress_liq"] + max(0, robust_z(SER["SKEW"])) * 0.08) > 0.60), int(clamp01(max(0, robust_z(SER["HY"])) + max(0, robust_z(SER["NFCI"])) * 0.5) > 0.60)])
    meter_rows = [
        ["Risk-On", pct(risk_on_meter), bucket(risk_on_meter,(0.33,0.60),("Weak","Building","Strong"))],
        ["Risk-Off", pct(risk_off_meter), bucket(risk_off_meter,(0.33,0.60),("Low","Elevated","High"))],
        ["Big Crash", pct(big_crash_meter), bucket(big_crash_meter,(0.40,0.70),("Low","Watch","High"))],
    ]
    policy_rows = [["P(QE)", pct(core["P_QE"])],["P(Neutral)", pct(core["P_NEUTRAL"])],["P(QT)", pct(core["P_QT"])],["Yield pressure", pct(core["yield_pressure"])],["Curve stress", pct(core["curve_stress"])],["Policy adj crash", f"{core['adj_crash']:.1f}"]]
    st.markdown(table_html(["Meter","Score","Read"], meter_rows), unsafe_allow_html=True)
    st.write("")
    st.markdown(table_html(["Policy liquidity block","Read"], policy_rows), unsafe_allow_html=True)
    st.markdown(f"**Current mode ➜ {'Override active' if override_active else 'Base case / watch only'}** • **Variant now ➜ {core['variant_now']}** • **Core crash gate ➜ {core_gate_blocks}/4**")
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
    st.markdown("</div>", unsafe_allow_html=True)

with t_notes:
    st.markdown("<div class='card'><div class='section-title'>NOTES</div>", unsafe_allow_html=True)
    st.markdown(f"""
- **Core model actually used**: `{CORE_NAME}`
- **Decision regime is Q3-anchored by design**; raw blended read is kept for audit in the engine
- **Yields now enter explicitly** via DGS2, DGS10, DFII10 and T10Y2Y through `yield_pressure` and `curve_stress`
- **Policy liquidity engine** outputs `P(QE) / P(Neutral) / P(QT)` and maps into risk-on / risk-off / big-crash modifiers
- **Layout is frozen to the attachment-2 shell**
- **IHSG size rotation is removed**
- **Crypto alt basket vs BTC** uses a basket proxy
- **Crypto/Liq** uses a composite proxy (WALCL, M2, USD inverse, NFCI inverse)
""")
    st.markdown("</div>", unsafe_allow_html=True)

st.caption("Attachment-2 shell frozen. Final form: Q3-anchored, yields explicit, policy-liquidity + risk-on/off/crash integrated. If the screen shape changes materially, the wrong file/version is running.")
