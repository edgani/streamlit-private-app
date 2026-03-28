
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
    "DGS2": fred_series("DGS2"),
    "DGS10": fred_series("DGS10"),
    "DFII10": fred_series("DFII10"),
    "T10Y2Y": fred_series("T10Y2Y"),
    "T10YIE": fred_series("T10YIE"),
    "IORB": fred_series("IORB"),
    "DFF": fred_series("DFF"),
    "SOFR": fred_series("SOFR"),
    "TGCR": fred_series("TGCR"),
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
# FINAL HYBRID HELPERS
# --------------------
MEGACAP_PROXY_WEIGHTS = {
    "AAPL": 6.8, "MSFT": 6.5, "NVDA": 6.0, "AMZN": 3.8,
    "GOOGL": 3.4, "META": 2.5, "BRK-B": 1.7, "AVGO": 1.6,
    "TSLA": 1.5, "LLY": 1.4,
}

PROXY_MAP = {
    "gold": ["GLD", "GDX"],
    "miners": ["GDX", "GDXJ"],
    "energy": ["XLE", "OIH"],
    "oil-gas": ["XLE", "OIH"],
    "coal": ["AMR", "BTU"],
    "metals": ["XME", "COPX"],
    "shipping": ["BDRY", "BOAT"],
    "usd": ["UUP"],
    "duration": ["TLT", "IEF"],
    "small caps": ["IWM"],
    "cyclicals": ["XLI", "XLB"],
    "defensives": ["XLU", "XLP"],
    "em": ["EEM"],
    "crypto": ["BTC-USD", "ETH-USD"],
    "btc": ["BTC-USD"],
}

def _last_valid(s: pd.Series, default: float = np.nan) -> float:
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else float(default)


def _rolling_pct_last(s: pd.Series, window: int = 756) -> float:
    s = s.dropna()
    if len(s) < 20:
        return 0.5
    hist = s.iloc[-min(window, len(s)):]
    return float(hist.rank(pct=True).iloc[-1])


def _stress_hi_last(s: pd.Series, window: int = 756) -> float:
    return clamp01(_rolling_pct_last(s, window) ** 1.5)


def _stress_lo_last(s: pd.Series, window: int = 756) -> float:
    return clamp01((1 - _rolling_pct_last(s, window)) ** 1.5)


def _safe_series(name: str) -> pd.Series:
    return SER.get(name, pd.Series(dtype=float)).dropna()


def _yahoo_pair_ratio(a: str, b: str, period: str = "3y") -> pd.Series:
    sa, sb = yahoo_close(a, period), yahoo_close(b, period)
    if sa.empty or sb.empty:
        return pd.Series(dtype=float)
    df = pd.concat([sa.rename('a'), sb.rename('b')], axis=1).sort_index().ffill().dropna()
    if df.empty:
        return pd.Series(dtype=float)
    return (df['a'] / df['b']).replace([np.inf, -np.inf], np.nan).dropna()


def compute_policy_liquidity() -> Dict[str, object]:
    core_cpi = _safe_series("CORE_CPI")
    core_yoy = core_cpi.pct_change(12)
    core_3m = _annualized_n(core_cpi, 3)
    breakeven = _safe_series("T10YIE")
    dgs2 = _safe_series("DGS2")
    dfii10 = _safe_series("DFII10")
    dgs10 = _safe_series("DGS10")
    curve = _safe_series("T10Y2Y")
    if curve.empty and not dgs10.empty and not dgs2.empty:
        curve = pd.concat([dgs10.rename('d10'), dgs2.rename('d2')], axis=1).dropna().eval('d10-d2')
    walcl = _safe_series("WALCL")
    m2 = _safe_series("M2")
    hy = _safe_series("HY")
    nfci = _safe_series("NFCI")
    unrate = _safe_series("UNRATE")
    claims = _safe_series("ICSA")
    iorb = _safe_series("IORB")
    dff = _safe_series("DFF")
    sofr = _safe_series("SOFR")
    tgcr = _safe_series("TGCR")
    usd = _safe_series("USD")

    vix = yahoo_close("^VIX", "3y")
    vvix = yahoo_close("^VVIX", "3y")
    hyg = yahoo_close("HYG", "3y")
    lqd = yahoo_close("LQD", "3y")
    hyg_lqd = _yahoo_pair_ratio("HYG", "LQD", "3y")

    y2_shock = dgs2.diff(20).clip(lower=0) if not dgs2.empty else pd.Series(dtype=float)
    real10_shock = dfii10.diff(20).clip(lower=0) if not dfii10.empty else pd.Series(dtype=float)
    curve_inv = (-curve).clip(lower=0) if not curve.empty else pd.Series(dtype=float)
    curve_resteepen = curve.diff(20).clip(lower=0) if not curve.empty else pd.Series(dtype=float)

    IC = clamp01(
        0.50 * _stress_hi_last((core_yoy - 0.02).clip(lower=0)) +
        0.30 * _stress_hi_last((core_3m - 0.02).clip(lower=0)) +
        0.20 * _stress_hi_last(breakeven)
    )
    GS = clamp01(0.55 * (core['blended'].get('Q3', 0.25) + core['blended'].get('Q4', 0.25)) + 0.45 * core['stress_growth'])
    LS = clamp01(0.60 * _stress_hi_last(unrate) + 0.40 * _stress_hi_last(claims.rolling(4).mean()))
    FS = clamp01(
        0.35 * _stress_hi_last(hy) +
        0.20 * _stress_hi_last(vix) +
        0.15 * _stress_hi_last(vvix) +
        0.15 * _stress_lo_last(hyg_lqd) +
        0.15 * _stress_hi_last(nfci)
    )
    reserve_gap = pd.Series(dtype=float)
    if not iorb.empty and not dff.empty:
        reserve_gap = (iorb - dff).abs()
    elif not iorb.empty and not sofr.empty:
        reserve_gap = (sofr - iorb).abs()
    elif not iorb.empty and not tgcr.empty:
        reserve_gap = (tgcr - iorb).abs()
    RS = clamp01(
        0.45 * _stress_hi_last(reserve_gap) +
        0.25 * _stress_hi_last(nfci) +
        0.30 * clamp01(0.50 * _stress_lo_last(walcl.diff(13)) + 0.50 * _stress_lo_last(walcl))
    )
    LT = clamp01(
        0.40 * _stress_lo_last(walcl.diff(13)) +
        0.30 * _stress_lo_last(m2.pct_change(12)) +
        0.30 * _stress_hi_last(usd)
    )
    MG = clamp01(0.55 * RS + 0.45 * FS)
    YR = clamp01(
        0.35 * _stress_hi_last(dgs2) +
        0.30 * _stress_hi_last(dfii10) +
        0.20 * _stress_hi_last(y2_shock) +
        0.15 * _stress_hi_last(real10_shock)
    )
    CS = clamp01(0.55 * _stress_hi_last(curve_inv) + 0.45 * _stress_hi_last(curve_resteepen))

    q = core['blended']
    z_qe = (1.10*FS + 0.90*RS + 0.80*GS + 0.55*LS + 0.70*LT + 0.60*q['Q4'] + 0.20*q['Q3'] - 1.20*IC - 0.70*YR)
    z_n  = (0.85*RS + 0.50*LT + 0.45*MG + 0.35*q['Q1'] + 0.20*q['Q2'] + 0.25*IC + 0.15*YR - 0.35*FS)
    z_qt = (1.10*IC + 0.80*(1-GS) + 0.70*(1-LT) + 0.55*q['Q2'] + 0.25*q['Q1'] + 0.85*YR - 0.85*FS - 0.75*RS)

    if (YR > 0.75 and IC > 0.60) and not (FS > 0.85 and RS > 0.85):
        z_qe = min(z_qe, 0.10)
    if (q['Q4'] > 0.45) and (IC < 0.40) and (YR < 0.45):
        z_qe += 0.25
    if (FS > 0.8 and RS > 0.8):
        z_qt -= 0.35
        z_n += 0.15
        z_qe += 0.20

    arr = np.array([z_qe, z_n, z_qt], dtype=float)
    ex = np.exp(arr - arr.max())
    probs = ex / ex.sum()
    P_QE, P_N, P_QT = [float(x) for x in probs]

    adj_crash = 10 * (0.60*P_QT + 0.20*P_N - 0.40*P_QE)
    adj_riskoff = 15 * (0.55*P_QT + 0.15*P_N - 0.30*P_QE)
    adj_riskon = 15 * (0.60*P_QE + 0.20*P_N - 0.50*P_QT)
    return {
        'IC': IC, 'GS': GS, 'LS': LS, 'FS': FS, 'RS': RS, 'LT': LT, 'MG': MG,
        'YR': YR, 'CS': CS,
        'P_QE': P_QE, 'P_NEUTRAL': P_N, 'P_QT': P_QT,
        'adj_crash': adj_crash, 'adj_riskoff': adj_riskoff, 'adj_riskon': adj_riskon,
        'y2': _last_valid(dgs2), 'y10': _last_valid(dgs10), 'real10': _last_valid(dfii10),
        'curve': _last_valid(curve), 'y2_shock': _last_valid(y2_shock, 0.0), 'real10_shock': _last_valid(real10_shock, 0.0),
    }


def compute_leadership_quality() -> Dict[str, object]:
    spy = yahoo_close('SPY', '1y'); rsp = yahoo_close('RSP', '1y'); iwm = yahoo_close('IWM', '1y')
    panel = yahoo_panel(list(MEGACAP_PROXY_WEIGHTS.keys()), period='3mo')
    ret1d = panel.pct_change().iloc[-1].dropna() if not panel.empty and len(panel) > 2 else pd.Series(dtype=float)
    rows = []
    raw = []
    for t, w in MEGACAP_PROXY_WEIGHTS.items():
        r = float(ret1d.get(t, 0.0))
        impact = w * r
        raw.append((t, w, r, impact))
    total_abs = sum(abs(x[3]) for x in raw) or 1.0
    concentration = sum(abs(x[3]) for x in sorted(raw, key=lambda z: abs(z[3]), reverse=True)[:5]) / total_abs
    pos = [(t,w,r,imp) for t,w,r,imp in raw if imp > 0]
    neg = [(t,w,r,imp) for t,w,r,imp in raw if imp < 0]
    pos_total = sum(x[3] for x in pos) or 1.0
    neg_total = sum(abs(x[3]) for x in neg) or 1.0
    pos_rows = [[t, f"{w:.2f}%", pct(r), pct(imp/pos_total)] for t,w,r,imp in sorted(pos, key=lambda z: z[3], reverse=True)[:5]] or [["None","-","-","-"]]
    neg_rows = [[t, f"{w:.2f}%", pct(r), pct(abs(imp)/neg_total)] for t,w,r,imp in sorted(neg, key=lambda z: z[3])[:5]] or [["None","-","-","-"]]

    eq_div = 0.0
    russell_rel = 0.0
    if not spy.empty and not rsp.empty:
        eq_div = ret_n(spy,21) - ret_n(rsp,21)
    if not spy.empty and not iwm.empty:
        russell_rel = ret_n(iwm,21) - ret_n(spy,21)
    breadth = core['breadth']
    if breadth < 0.30 and russell_rel < -0.04 and concentration > 0.55:
        label = 'Breaking'
    elif breadth < 0.40 or russell_rel < -0.02 or concentration > 0.48:
        label = 'Fragile'
    elif breadth > 0.58 and russell_rel > 0.01 and concentration < 0.34 and eq_div < 0.02:
        label = 'Broad'
    else:
        label = 'Narrow'
    score = clamp01(0.35*(1-concentration) + 0.35*breadth + 0.15*clamp01((russell_rel+0.08)/0.16) + 0.15*clamp01((0.04-eq_div)/0.08))
    summary_rows = [
        ['Leadership quality', label, 'Healthy move = breadth confirms and top-5 concentration not too extreme.'],
        ['Cap vs equal-weight', pct(eq_div), 'SPY outrunning RSP = index strength getting narrower.'],
        ['Russell confirm', pct(russell_rel), 'IWM lagging SPY = small caps not validating broad risk-on.'],
        ['Impact concentration', pct(concentration), 'Higher = move is increasingly dependent on fewer mega-caps.'],
    ]
    return {
        'label': label, 'score': score, 'summary_rows': summary_rows,
        'pos_rows': pos_rows, 'neg_rows': neg_rows,
        'concentration': concentration, 'eq_div': eq_div, 'russell_rel': russell_rel,
    }


def compute_risk_meters(policy: Dict[str, object], lead: Dict[str, object], fg_score: int) -> Dict[str, object]:
    vix = yahoo_close('^VIX', '3y')
    vvix = yahoo_close('^VVIX', '3y')
    skew = yahoo_close('^SKEW', '3y')
    hyg_lqd = _yahoo_pair_ratio('HYG', 'LQD', '3y')
    russell_block = 15 * clamp01(0.45 * clamp01((-lead['russell_rel'])/0.08) + 0.30 * _stress_lo_last(_yahoo_pair_ratio('IWM','SPY','3y')) + 0.25 * clamp01(abs(min(0.0, lead['russell_rel']))/0.10))
    breadth_block = 20 * clamp01(0.55*(1-core['breadth']) + 0.25*lead['concentration'] + 0.20*core['fragility'])
    vol_block = 20 * clamp01(0.35*_stress_hi_last(vix) + 0.25*_stress_hi_last(vvix) + 0.15*_stress_hi_last(skew) + 0.25*clamp01((_last_valid(vix)-16)/20))
    credit_block = 15 * clamp01(0.45*_stress_hi_last(_safe_series('HY')) + 0.30*_stress_lo_last(hyg_lqd) + 0.25*_stress_hi_last(_safe_series('NFCI')))
    trend_block = 10 * clamp01(0.50*core['fragility'] + 0.30*core['top_score'] + 0.20*clamp01((-ret_n(yahoo_close('SPY','1y'),21))/0.12))
    rates_block = 10 * clamp01(0.35*policy['YR'] + 0.25*_stress_hi_last(_safe_series('DGS2').diff(20).clip(lower=0)) + 0.20*_stress_hi_last(_safe_series('DFII10').diff(20).clip(lower=0)) + 0.20*policy['CS'])
    sentiment_block = 10 * clamp01(0.60*clamp01((55-fg_score)/55) + 0.40*_stress_hi_last(vix))
    core_hits = sum([
        russell_block >= 9.0,
        breadth_block >= 12.0,
        vol_block >= 12.0,
        credit_block >= 9.0,
    ])
    crash = clamp01((russell_block+breadth_block+vol_block+credit_block+trend_block+rates_block+sentiment_block + policy['adj_crash']) / 100.0)
    if core_hits < 3:
        crash *= 0.82
    riskoff = clamp01((
        0.18*clamp01((55-fg_score)/55) +
        0.16*clamp01((-lead['russell_rel'])/0.08) +
        0.15*_stress_hi_last(vix) +
        0.15*_stress_hi_last(_safe_series('USD')) +
        0.14*(1-core['breadth']) +
        0.12*core['fragility'] +
        0.10*clamp01(policy['adj_riskoff']/15)
    ))
    riskon = clamp01((
        0.22*core['breadth'] +
        0.18*clamp01((lead['russell_rel']+0.08)/0.16) +
        0.14*clamp01((0.45-lead['concentration'])/0.45) +
        0.14*(1-_stress_hi_last(vix)) +
        0.12*(1-_stress_hi_last(_safe_series('HY'))) +
        0.10*(1-core['fragility']) +
        0.10*clamp01((policy['adj_riskon']+7.5)/15)
    ))
    breakdown_rows = [
        ['Russell / small caps', pct(russell_block/15), 'Core crash block'],
        ['Breadth damage', pct(breadth_block/20), 'Core crash block'],
        ['Vol complex', pct(vol_block/20), 'Core crash block'],
        ['Credit / liquidity', pct(credit_block/15), 'Core crash block'],
        ['Trend / structure', pct(trend_block/10), 'Overlap block'],
        ['Rates / yields', pct(rates_block/10), 'Overlap block'],
        ['Sentiment / Fear & Greed', pct(sentiment_block/10), 'Supporting block'],
    ]
    riskoff_rows = [
        ['Sentiment', pct(clamp01((55-fg_score)/55)), 'Fear & Greed / early de-risking'],
        ['Russell weakness', pct(clamp01((-lead['russell_rel'])/0.08)), 'Small caps stop confirming'],
        ['Vol stress', pct(_stress_hi_last(vix)), 'Risk-off / hedge demand'],
        ['Dollar pressure', pct(_stress_hi_last(_safe_series('USD'))), 'Tighter global conditions'],
        ['Breadth narrowing', pct(1-core['breadth']), 'Internals worsen before index breaks'],
    ]
    return {
        'riskon': riskon, 'riskoff': riskoff, 'crash': crash,
        'breakdown_rows': breakdown_rows, 'riskoff_rows': riskoff_rows, 'core_hits': core_hits,
        'riskon_label': bucket(riskon, (0.33,0.66), ('Weak','Building','Strong')),
        'riskoff_label': bucket(riskoff, (0.33,0.66), ('Low','Elevated','High')),
        'crash_label': bucket(crash, (0.40,0.70), ('Contained','High risk','Severe setup')),
    }


def build_decision_snapshot_rows(policy: Dict[str, object]) -> List[List[str]]:
    q2_type = 'Fragile early Q2' if policy['P_QT'] > policy['P_QE'] or variant_now in ['Bad reflation','Crash-prone Q2'] else 'Healthier early Q2'
    return [
        ['Now', f"{cur_stage} {core['current_q']}", 'Current decision regime in use now.'],
        ['If next wins', f"{core['next_q']} ({q2_type if core['next_q']=='Q2' else 'next path'})", 'Most likely path if transition keeps extending.'],
        ['Variant now', variant_now, 'Clean/dirty branch matters more than quad label alone.'],
        ['Global state now', 'Selective, not broad clean risk-on', 'Use this as posture, not as a guarantee of direction.'],
        ['What confirms', 'Breadth repair + Russell confirm + USD/yields calm down', 'Those matter most for a cleaner Q3→Q2 handoff.'],
        ['What invalidates', 'Breadth stays narrow, small caps fail, credit/vol worsen', 'That keeps the tape in fragile / bad reflation mode.'],
        ['Next catalysts', ', '.join([r[0] for r in event_rows[:3]]), 'Use releases as timing checkpoints rather than prediction certainties.'],
    ]


def build_commodity_resource_rows() -> List[List[str]]:
    def _state_for(bucket: str) -> str:
        q = core['current_q']
        if bucket == 'Oil-gas':
            return 'Useful hedge / strong' if q == 'Q3' else 'Cyclical if breadth confirms'
        if bucket == 'Coal':
            return 'Inflation-linked / selective' if q == 'Q3' else 'Commodity beta if reflation cleans up'
        if bucket == 'Metals':
            return 'Needs China / growth confirmation' if q in ['Q3','Q4'] else 'Better if next reflation wins'
        if bucket == 'Shipping':
            return 'Depends on branch: tanker/LNG different from dry bulk' 
        if bucket == 'Positive spillovers':
            return 'Rail / logistics / suppliers can benefit after core energy/resources confirm'
        return 'Fuel-sensitive / cost-sensitive groups stay under pressure if energy shock dominates'
    rows = [
        ['Global','Oil-gas',_state_for('Oil-gas'),'Use as direct energy expression / hedge now','Still fine if next wins, but less unique edge'],
        ['Global','Coal',_state_for('Coal'),'Use selectively as consumable-fuels / inflation-linked expression','Can rotate into cyclical commodity beta if next Q2 cleans up'],
        ['Global','Metals',_state_for('Metals'),'Needs cleaner breadth or China-demand confirmation','Improves if growth breadth broadens'],
        ['Global','Shipping',_state_for('Shipping'),'Tanker/LNG can differ from bulk — treat as sub-branch','More cyclical if next wins and trade breadth improves'],
        ['Global','Positive spillovers',_state_for('Positive spillovers'),'Think rail / logistics / heavy suppliers','Better after core commodity leaders confirm'],
        ['Global','Pressured losers',_state_for('Pressured losers'),'Airlines / fuel-sensitive users can get squeezed','Pressure eases if energy shock fades'],
        ['IHSG','Oil-gas','Resource leadership / hedge','Useful when commodity tape is firm and IDR pressure manageable','Can stay fine if next wins'],
        ['IHSG','Coal','Direct resource leadership','Use when consumable-fuels theme still leading','Can broaden into local beta only if breadth confirms'],
        ['IHSG','Metals','Needs global demand / China help','Treat as watchlist if metals breadth improves','Cleaner if next wins with better EM backdrop'],
        ['IHSG','Shipping','Selective / branch-driven','More tied to freight branch and local flows','Improves if trade / commodity breadth broadens'],
    ]
    return rows


def build_proxy_ticker_rows() -> List[List[str]]:
    current_map = {
        'Q3': ['GLD / GDX', 'XLE / OIH', 'UUP / quality defensives'],
        'Q2': ['XLI / XLB', 'IWM', 'commodity FX proxies'],
        'Q4': ['TLT / IEF', 'XLU / XLP', 'cash-like quality'],
        'Q1': ['QQQ / SOXX', 'XLY', 'quality growth'],
    }
    next_map = {
        'Q2': ['XLI / XLB', 'IWM', 'EEM / cyclicals'],
        'Q3': ['GLD / GDX', 'UUP', 'defensives'],
        'Q4': ['TLT / IEF', 'quality defensives', 'cash-like'],
        'Q1': ['QQQ / SOXX', 'quality growth', 'consumer beta'],
    }
    avoid_map = {
        'Q3': 'Lower-quality beta / weak EMFX / broad chase longs',
        'Q2': 'Long duration proxies / bond-like defensives',
        'Q4': 'Junky beta / lower-quality cyclicals',
        'Q1': 'Deep defensives / panic hedges',
    }
    return [
        ['Proxy longs now', ', '.join(current_map.get(core['current_q'], [])), 'Use proxies first; stock-level timing still needs structure and risk ranges.'],
        ['Watch if next wins', ', '.join(next_map.get(core['next_q'], [])), 'Only activate if confirms actually show up.'],
        ['Avoid / underweight', avoid_map.get(core['current_q'], 'Contextual'), 'This is posture, not a guaranteed price path.'],
    ]


def build_policy_rows(policy: Dict[str, object], meters: Dict[str, object]) -> List[List[str]]:
    return [
        ['Risk-On', f"{pct(meters['riskon'])} ({meters['riskon_label']})", 'Cleaner broadening / healthier beta environment.'],
        ['Risk-Off', f"{pct(meters['riskoff'])} ({meters['riskoff_label']})", 'Correction / de-risking / vol-spike environment.'],
        ['Big Crash', f"{pct(meters['crash'])} ({meters['crash_label']})", f"Needs {meters['core_hits']}/4 core crash blocks to really confirm."],
        ['P(QE)', pct(policy['P_QE']), 'Supportive liquidity regime probability.'],
        ['P(Neutral)', pct(policy['P_NEUTRAL']), 'Reserve-management / non-crisis support probability.'],
        ['P(QT)', pct(policy['P_QT']), 'Restrictive liquidity backdrop probability.'],
        ['Yield pressure', pct(policy['YR']), '2Y + real 10Y + shock component.'],
        ['Curve stress', pct(policy['CS']), 'Inversion depth + re-steepening stress.'],
    ]


def build_rates_detail_rows(policy: Dict[str, object]) -> List[List[str]]:
    return [
        ['2Y Treasury', f"{policy['y2']:.2f}" if np.isfinite(policy['y2']) else 'n/a', 'Front-end policy repricing proxy'],
        ['10Y Treasury', f"{policy['y10']:.2f}" if np.isfinite(policy['y10']) else 'n/a', 'Long nominal rate context'],
        ['10Y real yield', f"{policy['real10']:.2f}" if np.isfinite(policy['real10']) else 'n/a', 'Valuation / real-rate pressure'],
        ['2s10s slope', f"{policy['curve']:.2f}" if np.isfinite(policy['curve']) else 'n/a', 'Growth scare vs policy-tightening context'],
        ['2Y shock 20d', f"{policy['y2_shock']:.2f}" if np.isfinite(policy['y2_shock']) else 'n/a', 'Restrictive repricing shock'],
        ['10Y real shock 20d', f"{policy['real10_shock']:.2f}" if np.isfinite(policy['real10_shock']) else 'n/a', 'Real-rate spike risk'],
    ]

# --------------------
# EVENTS
# --------------------
today = date.today()
events = [("NFP", today + timedelta(days=11)), ("CPI", today + timedelta(days=21)), ("PPI", today + timedelta(days=22))]
event_rows = [[name, dt.isoformat(), f"{(dt - today).days}d"] for name, dt in events]

# --------------------

# RENDER
policy = compute_policy_liquidity()
lead = compute_leadership_quality()
fg_score, fg_vibe = fear_greed_value()
meters = compute_risk_meters(policy, lead, fg_score)
cur_stage = 'Early' if core['phase_strength'] < 0.33 else ('Mid' if core['phase_strength'] < 0.66 else 'Late')
variant_now = core['sub_phase']
leadership_label = lead['label']

st.title(APP_NAME)
st.markdown("<div class='small-muted'>Core alpha engine: Hedgeye_LiveQuad_Core_v2_5 • Visual shell: attachment-2 merge • Decision regime anchored to Q3 while raw model remains visible in Details.</div>", unsafe_allow_html=True)
st.write("")

hero_cols = st.columns(7)
hero_items = [
    ("Current", core['current_q'], pill_html(f"{cur_stage} {core['current_q']}")),
    ("Next", core['next_q'], pill_html("Fragile early Q2" if core['next_q']=='Q2' else f"Path to {core['next_q']}")),
    ("Variant", variant_now, pill_html('Bad reflation', red=('bad' in variant_now.lower())) if 'reflation' in variant_now.lower() else pill_html(variant_now)),
    ("Risk-On", pct(meters['riskon']), pill_html(meters['riskon_label'])),
    ("Risk-Off", pct(meters['riskoff']), pill_html(meters['riskoff_label'], red=meters['riskoff']>0.66)),
    ("Big Crash", pct(meters['crash']), pill_html(meters['crash_label'], red=meters['crash']>0.55)),
    ("Leadership Quality", leadership_label, pill_html(f"Concentration {pct(lead['concentration'])}" if np.isfinite(lead['concentration']) else 'Proxy unavailable', red=leadership_label in ['Fragile','Breaking'])),
]
for col, (title, value, sub_html) in zip(hero_cols, hero_items):
    with col:
        st.markdown(f"""
        <div class='hero-card'>
          <div class='metric-title'>{title}</div>
          <div style='font-size:1.15rem;font-weight:800'>{value}</div>
          <div class='metric-sub'>{sub_html}</div>
        </div>
        """, unsafe_allow_html=True)

quick_read = (
    f"Quick read: sekarang {cur_stage} {core['current_q']} / {variant_now}. "
    f"Next paling mungkin {core['next_q']}. Leadership quality {leadership_label.lower()} — jadi jangan cuma lihat headline indeks. "
    f"Risk-On {pct(meters['riskon'])}, Risk-Off {pct(meters['riskoff'])}, Big Crash {pct(meters['crash'])}."
)
st.markdown(f"<div class='note-box'><b>{quick_read}</b></div>", unsafe_allow_html=True)

# Approved final structure
t_decision, t_cross, t_risk, t_details = st.tabs(["Decision", "Cross-Asset", "Risk / Relative", "Details"])

with t_decision:
    st.markdown("<div class='card'><div class='section-title'>DECISION SNAPSHOT</div>", unsafe_allow_html=True)
    st.markdown("<div class='mini-caption'>Satu panel inti: sekarang di mana, kalau next menang bakal seperti apa, apa yang harus dikonfirmasi, dan event apa yang paling bisa mengubah bacaannya.</div>", unsafe_allow_html=True)
    st.markdown(table_html(["Focus", "Read", "Why it matters"], build_decision_snapshot_rows(policy)), unsafe_allow_html=True)
    st.write("")
    play_rows = []
    for bucket_name in ["US Stocks", "Futures / Commodities", "Forex", "Crypto", "IHSG"]:
        play_rows.append([bucket_name, ", ".join(play_cur[bucket_name]), ", ".join(play_next[bucket_name]), 'Use only if confirms match the handoff'])
    st.markdown(table_html(["Bucket", "Now", "If next wins", "Confirm first"], play_rows), unsafe_allow_html=True)
    st.write("")
    st.markdown(f"**Global state now ➜** Selective / not broad clean risk-on")
    st.markdown(f"**Next catalysts ➜** {', '.join([f'{r[0]} ({r[2]})' for r in event_rows[:3]])}")
    st.markdown("</div>", unsafe_allow_html=True)

with t_cross:
    st.markdown("<div class='card'><div class='section-title'>CROSS-ASSET DIRECTIONAL BIAS</div>", unsafe_allow_html=True)
    st.markdown("<div class='mini-caption'>Panel operasional utama: strongest vs weakest, stage sekarang, FX, commodity-resource leadership, dan proxy ticker layer.</div>", unsafe_allow_html=True)
    # keep current simplified current/next playbook as directional bias
    rows = []
    for bucket_name in ["US Stocks", "Futures / Commodities", "Forex", "Crypto", "IHSG"]:
        rows.append([bucket_name, ", ".join(play_cur[bucket_name]), ", ".join(play_next[bucket_name]), 'Rotate only if breadth / Russell confirm'])
    st.markdown(table_html(["Area", "Bias now", "If next wins", "Read"], rows), unsafe_allow_html=True)
    st.write("")
    commodity_rows = build_commodity_resource_rows()
    st.markdown("**Commodity & Resource Leadership Map**")
    st.markdown(table_html(["Scope", "Bucket", "State now", "How to use now", "If next wins"], commodity_rows), unsafe_allow_html=True)
    st.write("")
    st.markdown("**Ticker / proxy layer**")
    st.markdown("<div class='mini-caption'>Proxy first, not guaranteed winners. Use together with regime, structure, and risk ranges — not as standalone buy/sell certainty.</div>", unsafe_allow_html=True)
    st.markdown(table_html(["Layer", "Read", "How to use"], build_proxy_ticker_rows()), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with t_risk:
    st.markdown("<div class='card'><div class='section-title'>RISK / RELATIVE SNAPSHOT</div>", unsafe_allow_html=True)
    st.markdown("<div class='mini-caption'>Semua yang berkorelasi gue merge di sini: liquidity / rates / stress engine, leadership quality / index internals, dan scenario state-now.</div>", unsafe_allow_html=True)
    st.markdown("**Liquidity / Rates / Stress Engine**")
    st.markdown(table_html(["Item", "Score / Read", "Why"], build_policy_rows(policy, meters)), unsafe_allow_html=True)
    st.write("")
    st.markdown(table_html(["Rates block", "Value", "Why it matters"], build_rates_detail_rows(policy)), unsafe_allow_html=True)
    st.write("")
    st.markdown(table_html(["Crash architecture", "Contribution", "Type"], meters['breakdown_rows']), unsafe_allow_html=True)
    st.write("")
    st.markdown("**Leadership Quality / Index Internals**")
    st.markdown(table_html(["Item", "Read", "Meaning"], lead['summary_rows']), unsafe_allow_html=True)
    st.write("")
    with st.expander("Show Index Impact Board", expanded=False):
        st.markdown("<div class='mini-caption'>Gunanya buat bilang apakah indeks kelihatan kuat karena broad market sehat, atau cuma diselamatkan sedikit mega-cap. Ini overlay leadership / concentration, bukan mesin utama buy-sell.</div>", unsafe_allow_html=True)
        st.markdown(table_html(["Top positive contributors", "Weight", "1D return", "Positive impact share"], lead['pos_rows']), unsafe_allow_html=True)
        st.write("")
        st.markdown(table_html(["Top negative detractors", "Weight", "1D return", "Negative impact share"], lead['neg_rows']), unsafe_allow_html=True)
        st.write("")
        impact_read = 'Index strength looks broad and healthy.' if leadership_label=='Broad' else ('Index strength is narrow / concentrated.' if leadership_label=='Narrow' else ('Internals look fragile; headline index can mislead.' if leadership_label=='Fragile' else 'Internals are breaking; treat index strength/weakness as potentially systemic.'))
        st.markdown(f"**Interpretation ➜ {impact_read}**")
    st.write("")
    st.markdown("**Scenario state-now**")
    scen_rows = [
        ['Gold in Q3', 'Selective long only' if core['current_q']=='Q3' else 'Contextual', 'Better if USD / yields stop squeezing.', 'Still bad if oil shock keeps USD / yields rising.'],
        ['EM / IHSG', 'Conditional only', 'Needs softer USD and better commodity breadth.', 'Still bad if USD / yields stay hard.'],
        ['US small caps', 'Confirmation asset', 'Need breadth + orderly rates.', 'Still bad if rates rise for bad reasons.'],
        ['Bonds / duration', 'Mostly tactical / hedge', 'Need growth fear to outrun inflation fear.', 'Still bad if nominal and real yields keep rising.'],
        ['Crypto beta', 'Selective only', 'Needs liquidity + breadth.', 'Still bad if USD is strong and funding tightens.'],
    ]
    st.markdown(table_html(["Scenario", "State now", "What must improve", "Still bad when"], scen_rows), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with t_details:
    st.markdown("<div class='card'><div class='section-title'>DETAILS / DATA / VALIDATION</div>", unsafe_allow_html=True)
    st.markdown(table_html(["Scenario family", "Read", "Why"], [list(x) for x in WHAT_IF]), unsafe_allow_html=True)
    st.write("")
    st.markdown(table_html(["Transmission", "Strength", "Why"], [list(x) for x in CORR]), unsafe_allow_html=True)
    st.write("")
    st.markdown(table_html(["Crash type", "Read"], [list(x) for x in CRASH]), unsafe_allow_html=True)
    st.write("")
    st.markdown(table_html(["Event", "Date", "In"], event_rows), unsafe_allow_html=True)
    st.write("")
    st.markdown("**Fear & Greed / tape context**")
    st.markdown(f"Fear & Greed ➜ {fg_score} ({fg_vibe})")
    st.markdown(f"IWM overlay ➜ {iwm_overlay}")
    st.write("")
    st.markdown(f"<div class='small-muted'><b>Core model:</b> {CORE_NAME} • <b>Decision regime:</b> {core['current_q']} • <b>Anchor:</b> {'Q3 on' if Q3_CONSENSUS_ANCHOR else 'off'} • <b>Raw model current:</b> {core.get('model_current_q', core['current_q'])}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.caption("Approved final structure loaded: Hero → Decision Snapshot → Cross-Asset → Risk/Relative → Details. If the screen shape changes materially, the wrong file/version is running.")
