
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
CORE_NAME = "Hedgeye_Directional_Core_v1"

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

    # Core activity / labor directional signals (do not let liquidity proxies define growth)
    g_indpro = _z_last(_annualized_n(indpro, 3) - indpro.pct_change(12), 36)
    g_sales = _z_last(_annualized_n(rsafs, 3) - rsafs.pct_change(12), 36)
    g_jobs = _z_last((payems.diff(3) / 3.0) - (payems.diff(12) / 12.0), 36)
    g_unrate = -_z_last(unrate.diff(3), 36)
    g_claims = -_z_last(icsa.rolling(4).mean() - icsa.rolling(26).mean(), 52)

    growth_official_inputs = [g_indpro, g_sales, g_jobs, g_unrate, g_claims]
    growth_official_weights = [0.22, 0.22, 0.22, 0.18, 0.16]

    # Inflation directional signals
    i_cpi = _z_last(_annualized_n(cpi, 3) - cpi.pct_change(12), 36)
    i_core = _z_last(_annualized_n(core_cpi, 3) - core_cpi.pct_change(12), 36)
    i_ppi = _z_last(_annualized_n(ppi, 3) - ppi.pct_change(12), 36)
    infl_official_inputs = [i_cpi, i_core, i_ppi]
    infl_official_weights = [0.36, 0.40, 0.24]

    # Fast nowcast modifiers
    wti_m = _monthly_last(SER["WTI"])
    hy = SER["HY"].dropna()
    nfci = SER["NFCI"].dropna()
    sahm = SER["SAHM"].dropna()

    oil_3m = _z_last(_annualized_n(wti_m, 3), 36)
    oil_1m = _z_last(_annualized_n(wti_m, 1), 36)

    # Keep nowcast growth tied to activity/labor; financial conditions only as modifiers
    g_now_inputs = [g_indpro, g_sales, g_jobs, g_unrate, g_claims, g_claims]
    g_now_weights = [0.22, 0.18, 0.24, 0.18, 0.12, 0.06]

    i_now_inputs = [i_cpi, i_core, i_ppi, oil_3m, oil_1m]
    i_now_weights = [0.28, 0.32, 0.18, 0.14, 0.08]

    stress_inputs = [
        _z_last(sahm, 36),
        _z_last(nfci, 52),
        _z_last(hy, 52),
    ]

    g_off = _weighted_mean(growth_official_inputs, growth_official_weights)
    i_off = _weighted_mean(infl_official_inputs, infl_official_weights)
    g_now = _weighted_mean(g_now_inputs, g_now_weights)
    i_now = _weighted_mean(i_now_inputs, i_now_weights)
    s_m = float(np.nanmean(stress_inputs))

    # Thresholds intentionally biased against easy Q2 classification.
    g_up_off = sigmoid(1.30 * (g_off - 0.04) - 0.10 * max(0.0, s_m))
    i_up_off = sigmoid(1.35 * (i_off + 0.02) + 0.08 * max(0.0, s_m))
    g_up_now = sigmoid(1.55 * (g_now - 0.10) - 0.20 * max(0.0, s_m))
    i_up_now = sigmoid(1.60 * (i_now + 0.04) + 0.12 * max(0.0, s_m))

    official_probs = _quad_probs(g_up_off, i_up_off)
    directional_probs = _quad_probs(g_up_now, i_up_now)

    # Guardrail: if labor is deteriorating while inflation is re-accelerating,
    # do not let loose credit / risk appetite masquerade as Q2.
    labor_weak = _weighted_mean([
        max(0.0, -g_jobs),
        max(0.0, -g_unrate),
        max(0.0, -g_claims),
    ], [0.45, 0.30, 0.25])
    inflation_push = max(0.0, i_now)
    stag_pressure = clamp01(0.55 * sigmoid(1.8 * (labor_weak - 0.10)) + 0.45 * sigmoid(1.8 * (inflation_push - 0.08)))
    if stag_pressure > 0.52 and directional_probs["Q2"] > directional_probs["Q3"]:
        shift = min(0.16, 0.55 * (stag_pressure - 0.52) + 0.05)
        take = directional_probs["Q2"] * shift
        directional_probs["Q2"] -= take
        directional_probs["Q3"] += take
        directional_probs = _renorm_probs(directional_probs)

    blended = {k: 0.40 * official_probs[k] + 0.60 * directional_probs[k] for k in official_probs}
    total = sum(blended.values())
    blended = {k: (v / total if total else 0.25) for k, v in blended.items()}

    ranked = sorted(blended.items(), key=lambda x: x[1], reverse=True)
    current_q, current_p = ranked[0]
    next_q, next_p = ranked[1]

    off_ranked = sorted(official_probs.items(), key=lambda x: x[1], reverse=True)
    dir_ranked = sorted(directional_probs.items(), key=lambda x: x[1], reverse=True)
    official_q, official_p = off_ranked[0]
    directional_q, directional_p = dir_ranked[0]

    agreement = clamp01(1.0 - 0.60 * sum(abs(official_probs[k] - directional_probs[k]) for k in official_probs))
    confidence = clamp01(0.58 * current_p + 0.42 * agreement)

    phase_strength = clamp01(0.40 * abs(g_now) + 0.40 * abs(i_now) + 0.20 * max(0.0, current_p - 0.25))
    growth_breadth = clamp01(sum(1 for x in growth_official_inputs if x > 0) / len(growth_official_inputs))
    infl_breadth = clamp01(sum(1 for x in infl_official_inputs if x > 0) / len(infl_official_inputs))
    breadth = 0.5 * growth_breadth + 0.5 * infl_breadth
    regime_divergence = abs(g_now - g_off) + abs(i_now - i_off)
    fragility = clamp01(0.35 * (1 - current_p) + 0.30 * (1 - agreement) + 0.20 * max(0.0, s_m) + 0.15 * min(1.0, regime_divergence / 2.5))

    margin = current_p - next_p
    if margin < 0.03:
        fragility = clamp01(fragility + 0.08)
    if margin < 0.015:
        confidence = clamp01(confidence - 0.05)

    if current_q == "Q1":
        sub_phase = "Goldilocks / recovery" if fragility < 0.45 else "Recovery but fragile"
    elif current_q == "Q2":
        sub_phase = "Expansion / hot growth" if i_now < 0.35 else "Reflation / overheating risk"
    elif current_q == "Q3":
        sub_phase = "Stagflation building" if i_now > i_off + 0.10 else "Slowdown with sticky inflation"
    else:
        sub_phase = "Late Q4 / inflation trying to turn" if directional_q == "Q3" else ("Bottoming attempt" if g_now > -0.20 else "Deflationary slowdown")

    top_score = clamp01(0.35 * max(0.0, i_now) + 0.20 * phase_strength + 0.20 * fragility + 0.15 * float(current_q in ["Q2", "Q3"]) + 0.10 * float(directional_q == "Q3"))
    bottom_score = clamp01(0.35 * float(current_q == "Q4") + 0.20 * max(0.0, -g_now) + 0.20 * (1 - fragility) + 0.15 * float(official_q == "Q4") + 0.10 * float(directional_q == "Q4"))
    higher_top = clamp01(top_score * 0.65 * (1 - bottom_score))
    lower_bottom = clamp01(bottom_score * 0.65 * (1 - top_score))

    transition_pressure = clamp01(0.35 * fragility + 0.30 * (1 - current_p) + 0.20 * min(1.0, regime_divergence / 2.5) + 0.15 * abs(g_now - i_now) / 3.0)
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
        "monthly": official_probs,
        "quarterly": directional_probs,
        "blend": blended,
        "current_q": current_q,
        "current_p": current_p,
        "next_q": next_q,
        "next_p": next_p,
        "official_q": official_q,
        "official_p": official_p,
        "directional_q": directional_q,
        "directional_p": directional_p,
        "official_date": official_dt.strftime("%Y-%m-%d") if official_dt is not None else "n/a",
        "agreement": agreement,
        "confidence": confidence,
        "phase_strength": phase_strength,
        "breadth": breadth,
        "fragility": fragility,
        "sub_phase": sub_phase,
        "higher_top_prob": higher_top,
        "lower_bottom_prob": lower_bottom,
        "top_score": top_score,
        "bottom_score": bottom_score,
        "transition_prob": transition_conviction,
        "transition_pressure": transition_pressure,
        "stay_prob": stay_probability,
        "path_status": path_status,
        "g_off": g_off,
        "i_off": i_off,
        "g_now": g_now,
        "i_now": i_now,
        "stress_m": s_m,
        "labor_weak": labor_weak,
        "stag_pressure": stag_pressure,
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
    bench = yahoo_close(benchmark_ticker, period)
    bench_name = benchmark_ticker
    if bench.empty and fallback_benchmark:
        bench = yahoo_close(fallback_benchmark, period)
        if not bench.empty:
            bench_name = fallback_benchmark
    if bench.empty:
        return pd.DataFrame()

    rows = []
    for ticker in tickers:
        s = yahoo_close(ticker, period)
        if s.empty:
            continue
        df = pd.concat([s.rename('asset'), bench.rename('bench')], axis=1).dropna()
        if len(df) < 80:
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

def compute_leadership_panels() -> Tuple[pd.DataFrame, pd.DataFrame, str, str, str, str]:
    us_custom_raw = st.session_state.get('us_custom_tickers', '')
    ihsg_custom_raw = st.session_state.get('ihsg_custom_tickers', '')

    us_tickers = US_UNIVERSE + _clean_tickers(us_custom_raw)
    ihsg_tickers = IHSG_UNIVERSE + _clean_tickers(ihsg_custom_raw, '.JK')

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
    ihsg_df = rank_market_leaders(ih_final, benchmark_ticker='^JKSE', period='1y', fallback_benchmark='EIDO')

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
# EVENTS
# --------------------
today = date.today()
event_rows = [["Macro release timing", "dynamic", "Use actual calendar; avoid fake offsets"], ["Released-only cutoff", core["official_date"], "common macro date used in official state"]]

# --------------------
# RENDER
# --------------------
st.title(APP_NAME)
st.markdown("<div class='small-muted'>Core alpha engine: Hedgeye_Directional_Core_v1 • Visual shell: mind-map card layout • Live backbone: FRED + optional Yahoo</div>", unsafe_allow_html=True)
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
    t_rel, t_lead, t_shock, t_notes = st.tabs(["Relative", "Leaders", "Shocks / What-If", "Notes"])

with t_current:
    c1, c2 = st.columns([1.1, 0.9], gap="large")
    with c1:
        st.markdown("<div class='card'><div class='section-title'>CURRENT MAP</div>", unsafe_allow_html=True)
        st.markdown(f"**Phase ➜ {core['current_q']}**")
        st.markdown(f"**Confidence ➜ {pct(core['confidence'])}** {pill_html('Decaying', red=True) if core['fragility'] > 0.55 else pill_html('Stable')}", unsafe_allow_html=True)
        st.markdown(f"**Agreement ➜ {pct(core['agreement'])}** {pill_html('Official / Nowcast')}", unsafe_allow_html=True)
        st.markdown(f"**Sub-Phase ➜ {core['sub_phase']}**")
        st.markdown(f"**Released-only current ➜ {core['official_q']} ({core['official_date']})**")
        st.markdown(f"**Directional nowcast ➜ {core['directional_q']}**")
        st.markdown(f"**Labor weakness / stagflation pressure ➜ {core['labor_weak']:+.2f} / {core['stag_pressure']:.2f}**")
        st.markdown(f"**Regime Strength ➜ {pct(core['phase_strength'])}**")
        st.markdown(f"**Breadth ➜ {pct(core['breadth'])}**")
        st.markdown(f"**Fragility ➜ {pct(core['fragility'])}**")
        st.markdown(f"**Signal quality ➜ {core['signal_quality']}**")
        explanation = (
            f"Current = {core['current_q']} using a directional nowcast blend. Released-only macro still says {core['official_q']} as of {core['official_date']}. "
            f"Directional nowcast = {core['directional_q']}. Next = {core['next_q']} only if the transition keeps building. "
            f"The model chooses {core['current_q']} because the nowcast blend has the highest probability, "
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

with t_lead:
    st.markdown("<div class='card'><div class='section-title'>US / IHSG STOCK LEADERSHIP</div>", unsafe_allow_html=True)
    st.markdown("<div class='mini-caption'>This module ranks stocks versus their market benchmark. It separates names already outperforming from names only starting to outperform.</div>", unsafe_allow_html=True)
    setup1, setup2 = st.columns(2, gap='large')
    with setup1:
        st.selectbox('US watchlist preset', options=list(US_PRESETS.keys()), index=list(US_PRESETS.keys()).index(st.session_state.get('us_leader_preset', 'Quantum / speculative tech')), key='us_leader_preset')
        st.text_input('US custom tickers (comma-separated)', value=st.session_state.get('us_custom_tickers', ''), key='us_custom_tickers')
        st.markdown(f"**US leaders now ➜ {us_lead_text}**")
        st.markdown(f"**US starting to outperform ➜ {us_em_text}**")
    with setup2:
        st.selectbox('IHSG watchlist preset', options=list(IHSG_PRESETS.keys()), index=list(IHSG_PRESETS.keys()).index(st.session_state.get('ihsg_leader_preset', 'IHSG large caps')), key='ihsg_leader_preset')
        st.text_input('IHSG custom tickers (comma-separated, no .JK needed)', value=st.session_state.get('ihsg_custom_tickers', ''), key='ihsg_custom_tickers')
        st.markdown(f"**IHSG leaders now ➜ {ih_lead_text}**")
        st.markdown(f"**IHSG starting to outperform ➜ {ih_em_text}**")
    st.write("")

    lead_c1, lead_c2 = st.columns(2, gap='large')
    with lead_c1:
        st.markdown('**US benchmark-relative leadership**')
        st.markdown(table_html(['Ticker', 'State', 'RS', 'Start', 'α 1M / 3M', 'Read'], leadership_table_rows(us_leaders_df, 'top', 10)), unsafe_allow_html=True)
        st.write('')
        st.markdown('**US names starting to outperform**')
        st.markdown(table_html(['Ticker', 'State', 'RS', 'Start', 'α 1M / 3M', 'Read'], leadership_table_rows(us_leaders_df, 'emerging', 6)), unsafe_allow_html=True)
    with lead_c2:
        st.markdown('**IHSG benchmark-relative leadership**')
        st.markdown(table_html(['Ticker', 'State', 'RS', 'Start', 'α 1M / 3M', 'Read'], leadership_table_rows(ihsg_leaders_df, 'top', 10)), unsafe_allow_html=True)
        st.write('')
        st.markdown('**IHSG names starting to outperform**')
        st.markdown(table_html(['Ticker', 'State', 'RS', 'Start', 'α 1M / 3M', 'Read'], leadership_table_rows(ihsg_leaders_df, 'emerging', 6)), unsafe_allow_html=True)
    st.write("")
    lag1, lag2 = st.columns(2, gap='large')
    with lag1:
        st.markdown('**US laggards / fading leaders**')
        st.markdown(table_html(['Ticker', 'State', 'RS', 'Start', 'α 1M / 3M', 'Read'], leadership_table_rows(us_leaders_df, 'fading', 6)), unsafe_allow_html=True)
    with lag2:
        st.markdown('**IHSG laggards / fading leaders**')
        st.markdown(table_html(['Ticker', 'State', 'RS', 'Start', 'α 1M / 3M', 'Read'], leadership_table_rows(ihsg_leaders_df, 'fading', 6)), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with t_shock:
    st.markdown("<div class='card'><div class='section-title'>SHOCKS / WHAT-IF</div>", unsafe_allow_html=True)
    st.markdown(f"**Current mode ➜ {'Override active' if override_active else 'Base case / watch only'}**")
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
- **Layout is frozen to the attachment-2 shell**
- **Current Q should only change if the same core engine changes — not because the shell/layout changed**
- **IHSG size rotation is removed**
- **Crypto alt basket vs BTC** uses a basket proxy
- **Crypto/Liq** uses a composite proxy (WALCL, M2, USD inverse, NFCI inverse)
- **Leaders tab** ranks US stocks vs SPY and IHSG stocks vs ^JKSE with EIDO fallback
- **Emerging** = starting to outperform; **Leader** = already outperforming cleanly
""")
    st.markdown("</div>", unsafe_allow_html=True)

st.caption("Attachment-2 shell frozen. If the screen shape changes materially, the wrong file/version is running.")
