import html
import math
import re
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

st.set_page_config(page_title="Macro Quad Transition Dashboard", layout="wide")

SERIES = {
    "WEI": "WEI",
    "ICSA": "ICSA",
    "T10Y2Y": "T10Y2Y",
    "CPI": "CPIAUCSL",
    "CORE_CPI": "CPILFESL",
    "BREAKEVEN_5Y": "T5YIE",
    "OIL": "DCOILWTICO",
    "HY_OAS": "BAMLH0A0HYM2",
    "NFCI": "NFCI",
    "STLFSI4": "STLFSI4",
    "VIX": "VIXCLS",
    "SAHM": "SAHMREALTIME",
    "RECPRO": "RECPROUSM156N",
}

RELEASE_WATCH = {
    "CPI": "CPIAUCSL",
    "Jobs": "UNRATE",
    "Claims": "ICSA",
    "GDP": "GDP",
    "Retail Sales": "RSAFS",
}

# User-preferred quad numbering from the screenshots:
# Q1 = Growth Up / Inflation Down
# Q2 = Growth Up / Inflation Up
# Q3 = Growth Down / Inflation Up
# Q4 = Growth Down / Inflation Down
QUAD_META = {
    "Q1": {
        "name": "Quad 1",
        "phase": "Goldilocks / Early Recovery",
        "logic": "Growth Up / Inflation Down",
        "winners": "small caps, semis, emerging markets, hard metals",
    },
    "Q2": {
        "name": "Quad 2",
        "phase": "Reflation / Strong Nominal Growth",
        "logic": "Growth Up / Inflation Up",
        "winners": "energy, hard metals, emerging markets, nominal-growth beta",
    },
    "Q3": {
        "name": "Quad 3",
        "phase": "Stagflation Stress",
        "logic": "Growth Down / Inflation Up",
        "winners": "energy, gold, hard assets, defensives, emerging markets",
    },
    "Q4": {
        "name": "Quad 4",
        "phase": "Disinflation Slowdown / Recession Risk",
        "logic": "Growth Down / Inflation Down",
        "winners": "rates, duration, defensives, gold, recession hedges",
    },
}

PHASE_GUIDE = {
    "Q1": {
        "meaning": {
            "Macro": [
                "Growth improves while inflation cools.",
                "The economy feels healthier without a major inflation scare.",
            ],
            "Market": [
                "Breadth usually improves.",
                "Risk assets can broaden beyond just mega caps.",
            ],
            "Positioning": [
                "Rotation into cyclicals, beta, and selected growth.",
                "Cleaner risk-on works best if credit stays stable.",
            ],
        },
        "winners": {
            "Small Caps": {
                "Direct": ["IWM", "small-cap cyclicals", "equal-weight beta"],
                "Correlated / Spillover": ["regional banks", "industrials", "consumer discretionary"],
                "Confirmation": ["breadth improving", "credit stable", "inflation cooling"],
            },
            "Semis / AI Infra": {
                "Direct": ["SMH", "SOXX", "NVDA", "AVGO", "ANET"],
                "Correlated / Spillover": ["networking", "power infra", "cooling / HVAC"],
                "Confirmation": ["semis lead", "capex strong", "risk-on broadens"],
            },
            "Emerging Markets": {
                "Direct": ["EEM", "EM beta", "Asia cyclicals"],
                "Correlated / Spillover": ["materials exporters", "internet platforms", "global cyclicals"],
                "Confirmation": ["USD calmer", "global growth okay", "inflation cooling"],
            },
            "Hard Metals": {
                "Direct": ["copper beta", "silver beta", "selected miners"],
                "Correlated / Spillover": ["industrial metals", "resource exporters"],
                "Confirmation": ["growth improving", "China impulse better", "risk appetite healthy"],
            },
        },
        "losers": {
            "Deep Defensives": {
                "Direct": ["staples laggards", "utilities laggards", "slow defensive beta"],
                "Correlated / Spillover": ["bond-proxy defensives", "low-beta hiding spots"],
                "Pressure Signs": ["breadth broadens", "beta works", "fear fades"],
            },
            "Pure Inflation Hedges": {
                "Direct": ["energy leadership", "commodity beta", "inflation hedges"],
                "Correlated / Spillover": ["refiners", "materials torque", "hard-asset only trades"],
                "Pressure Signs": ["oil cooling", "breakevens softer", "inflation losing momentum"],
            },
        },
    },
    "Q2": {
        "meaning": {
            "Macro": [
                "Nominal growth is strong and inflation is still firm.",
                "The market starts asking whether inflation cools first or growth cracks first.",
            ],
            "Market": [
                "Reflation-sensitive groups usually lead while pure duration lags.",
                "Hard assets and commodity beta stay relevant if inflation stays sticky.",
            ],
            "Positioning": [
                "Lean into strong nominal-growth winners, but watch for transition risk.",
                "Need tighter risk management because this quad can flip into Goldilocks or stagflation.",
            ],
        },
        "winners": {
            "Energy": {
                "Direct": ["XLE", "OIH", "oil names", "commodity producers"],
                "Correlated / Spillover": ["refiners", "shipping", "energy services"],
                "Confirmation": ["oil firm", "breakevens up", "inflation sticky"],
            },
            "Hard Metals": {
                "Direct": ["copper", "silver", "selected miners"],
                "Correlated / Spillover": ["materials", "industrials", "resource exporters"],
                "Confirmation": ["global growth okay", "hard-asset demand", "inflation pressure persistent"],
            },
            "Emerging Markets": {
                "Direct": ["EEM", "commodity EM", "resource-heavy EM"],
                "Correlated / Spillover": ["Asia cyclicals", "LatAm beta", "global exporters"],
                "Confirmation": ["commodities firm", "USD not too strong", "nominal growth strong"],
            },
        },
        "losers": {
            "Long-Duration Tech": {
                "Direct": ["rate-sensitive software", "long-duration growth", "high-multiple tech"],
                "Correlated / Spillover": ["profitless growth", "duration-heavy quality"],
                "Pressure Signs": ["rates firm", "inflation sticky", "value / hard assets lead"],
            },
        },
    },
    "Q3": {
        "meaning": {
            "Macro": [
                "Growth slows while inflation stays too firm.",
                "Consumers and margins both feel pressure in this mix.",
            ],
            "Market": [
                "Hard assets and selective defensives tend to outperform broad beta.",
                "Rally quality usually deteriorates and balance-sheet quality matters more.",
            ],
            "Positioning": [
                "Prefer inflation hedges, hard assets, and selective defense.",
                "Avoid assuming every cyclical dip is a clean buy before growth truly bottoms.",
            ],
        },
        "winners": {
            "Energy": {
                "Direct": ["XLE", "oil names", "commodity producers"],
                "Correlated / Spillover": ["refiners", "shipping", "resource exporters"],
                "Confirmation": ["oil firm", "breakevens firm", "inflation still sticky"],
            },
            "Gold / Hard Assets": {
                "Direct": ["GLD", "GDX", "hard-asset equities"],
                "Correlated / Spillover": ["uranium", "silver", "select real assets"],
                "Confirmation": ["real growth weak", "policy uncertainty", "inflation not breaking"],
            },
            "Defensives": {
                "Direct": ["staples", "utilities", "healthcare"],
                "Correlated / Spillover": ["quality balance sheets", "cash-flow growers"],
                "Confirmation": ["growth rolls", "beta weakens", "credit worsens"],
            },
            "Emerging Markets": {
                "Direct": ["commodity EM", "resource exporters", "select EM value"],
                "Correlated / Spillover": ["LatAm beta", "energy-linked FX"],
                "Confirmation": ["hard assets lead", "commodity complex firm", "USD not disorderly"],
            },
        },
        "losers": {
            "Weak Consumer": {
                "Direct": ["low-end consumer", "discretionary laggards", "retail beta"],
                "Correlated / Spillover": ["housing beta", "transport", "travel beta"],
                "Pressure Signs": ["real income squeezed", "growth weak", "consumer stress rises"],
            },
            "Long-Duration Tech": {
                "Direct": ["rate-sensitive growth", "spec tech", "long-duration software"],
                "Correlated / Spillover": ["profitless growth", "multiple-heavy tech"],
                "Pressure Signs": ["inflation sticky", "rates volatile", "real growth weak"],
            },
            "Junk Beta / Weak Balance Sheets": {
                "Direct": ["weak balance-sheet beta", "junk rallies", "fragile cyclicals"],
                "Correlated / Spillover": ["small-cap junk", "high-beta laggards"],
                "Pressure Signs": ["credit worsens", "funding tightens", "risk appetite narrows"],
            },
        },
    },
    "Q4": {
        "meaning": {
            "Macro": [
                "Growth slows and inflation cools at the same time.",
                "Recession risk starts to matter more than inflation fear.",
            ],
            "Market": [
                "Defensives, quality, and duration usually matter more.",
                "Broad beta needs clearer evidence of a new growth bottom before leading again.",
            ],
            "Positioning": [
                "Stay selective with risk and emphasize balance-sheet strength.",
                "Watch for either a Q1 recovery turn or a Q3 re-heating inflation scare.",
            ],
        },
        "winners": {
            "Rates / Duration": {
                "Direct": ["TLT", "IEF", "duration-sensitive equities"],
                "Correlated / Spillover": ["utilities", "REIT quality", "software quality"],
                "Confirmation": ["inflation cooling", "growth slowing", "yields easing"],
            },
            "Defensives": {
                "Direct": ["XLP", "XLV", "quality large caps"],
                "Correlated / Spillover": ["cash-flow growers", "low-beta names"],
                "Confirmation": ["claims worsen", "PMI soft", "recession risk rises"],
            },
            "Gold": {
                "Direct": ["GLD", "GDX", "select hard-asset protection"],
                "Correlated / Spillover": ["macro hedges", "policy uncertainty trades"],
                "Confirmation": ["real yields behave", "macro fear rises", "growth slows"],
            },
        },
        "losers": {
            "High Beta / Weak Balance Sheets": {
                "Direct": ["small-cap cyclicals", "deep value beta", "weak balance-sheet beta"],
                "Correlated / Spillover": ["consumer discretionary", "transport", "banks"],
                "Pressure Signs": ["growth slowing", "claims worsening", "PMI soft"],
            },
            "Cyclicals": {
                "Direct": ["industrials lag", "financials lag", "consumer beta lag"],
                "Correlated / Spillover": ["capex plays", "aggressive reflation trades"],
                "Pressure Signs": ["recession risk up", "growth soft", "risk appetite weak"],
            },
        },
    },
}

PLAYBOOK_NEXT = {
    "Q1": ["Q2"],
    "Q2": ["Q3"],
    "Q3": ["Q4", "Q1"],
    "Q4": ["Q1"],
}

PATH_META = {
    ("Q1", "Q2"): {
        "possible": "Inflation re-heats while growth stays okay, so Goldilocks morphs back into reflation.",
        "if_right": "Bias tilts toward cyclicals, banks, value, industrials, and stronger nominal-growth trades.",
        "laggards": "Long-duration defensives and pure disinflation beneficiaries can lose relative strength.",
    },
    ("Q1", "Q4"): {
        "possible": "Growth loses momentum first while inflation keeps cooling, opening a slowdown / disinflation path.",
        "if_right": "Bonds, defensives, staples, utilities, and quality balance sheets get more support.",
        "laggards": "Small caps, cyclicals, and aggressive beta lose sponsorship.",
    },
    ("Q2", "Q1"): {
        "possible": "Inflation cools first while growth stays resilient, creating a cleaner Goldilocks handoff.",
        "if_right": "Duration, quality growth, semis, and broader beta become easier to own.",
        "laggards": "Pure reflation trades, hard assets, and sticky-inflation hedges lose relative edge.",
    },
    ("Q2", "Q3"): {
        "possible": "Growth cracks first while inflation stays sticky, opening the stagflation path.",
        "if_right": "Energy, gold, hard assets, and defensives become more relevant than pure beta.",
        "laggards": "Small caps, banks, cyclicals, and domestic beta usually struggle.",
    },
    ("Q3", "Q4"): {
        "possible": "Inflation finally breaks lower while growth is still weak, shifting toward disinflation / slowdown.",
        "if_right": "Bonds, quality, and defensives gain support as inflation pressure fades.",
        "laggards": "Commodities and inflation-sensitive trades usually lose leadership.",
    },
    ("Q3", "Q2"): {
        "possible": "Growth recovers first while inflation remains firm, pulling the regime back into reflation.",
        "if_right": "Cyclicals, industrials, value, and nominal-growth winners usually improve.",
        "laggards": "Deep defensives and pure duration trades lose edge.",
    },
    ("Q4", "Q1"): {
        "possible": "Growth bottoms while inflation stays cool enough, setting up early recovery / Goldilocks.",
        "if_right": "Small caps, semis, and broader risk-on participation can start to improve.",
        "laggards": "Pure safety trades become less dominant as breadth returns.",
    },
    ("Q4", "Q3"): {
        "possible": "Inflation re-accelerates while growth stays weak, creating a stagflation risk.",
        "if_right": "Energy, gold, and hard assets get more relative support than broad beta.",
        "laggards": "Bonds and clean Goldilocks expectations get hit.",
    },
}

CURRENT_PHASE_TEXT = {
    "Q1": "Growth is holding up while inflation cools. Focus on whether inflation re-heats or growth loses momentum first.",
    "Q2": "Nominal growth is strong. Focus on whether inflation cools first or growth rolls first.",
    "Q3": "Growth is weak while inflation stays firm. Focus on whether inflation finally breaks or growth bottoms first.",
    "Q4": "Growth and inflation are both cooling. Focus on whether growth bottoms into recovery or inflation re-heats into stagflation.",
}

STATUS_MAP = {
    "confirmed": ("✅", "Confirmed", 1.0),
    "starting": ("🟡", "Starting", 0.55),
    "notyet": ("⚪", "Not Yet", 0.15),
    "invalid": ("❌", "Invalidated", 0.0),
}

CNN_FG_URL = "https://edition.cnn.com/markets/fear-and-greed"
CNN_FG_JSON_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
    "Referer": "https://edition.cnn.com/",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
            .main-card {
                border: 2px solid #19e68c;
                border-radius: 20px;
                padding: 22px;
                background: linear-gradient(90deg, rgba(14,35,29,.95), rgba(6,25,30,.95));
                margin-bottom: 14px;
            }
            .soft-card {
                padding: 16px;
                border-radius: 16px;
                border: 1px solid rgba(255,255,255,.10);
                background: rgba(255,255,255,.03);
                height: 100%;
            }
            .mini-card {
                padding: 14px;
                border-radius: 14px;
                background: rgba(255,255,255,.03);
                border: 1px solid rgba(255,255,255,.08);
                height: 100%;
            }
            .section-note {
                padding: 14px 16px;
                border-radius: 14px;
                background: rgba(37, 99, 235, .14);
                border: 1px solid rgba(96, 165, 250, .18);
                margin-bottom: 12px;
            }
            .tree-box {
                padding: 14px;
                border-radius: 14px;
                background: rgba(255,255,255,.02);
                border: 1px solid rgba(255,255,255,.07);
                margin-bottom: 10px;
            }
            .small-muted { opacity: .8; font-size: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sigmoid(x: float) -> float:
    x = float(np.clip(x, -8, 8))
    return 1.0 / (1.0 + math.exp(-x))


def clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def safe_last(series: pd.Series, default: float = 0.0) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return default
    return float(s.iloc[-1])


def last_date(series: pd.Series) -> Optional[pd.Timestamp]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return pd.Timestamp(s.index[-1])


def rolling_z_last(series: pd.Series, window: int = 126, default: float = 0.0) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 20:
        return default
    w = min(window, max(20, len(s)))
    tail = s.iloc[-w:]
    std = float(tail.std(ddof=0))
    if std == 0 or np.isnan(std):
        return default
    return float((tail.iloc[-1] - tail.mean()) / std)


def ann_roc(series: pd.Series, periods: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return ((s / s.shift(periods)) ** (12 / periods) - 1.0) * 100.0


def pct_change(series: pd.Series, periods: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.pct_change(periods) * 100.0


def fred_series(series_id: str, api_key: str, start: str = "2000-01-01") -> pd.Series:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    obs = r.json().get("observations", [])
    if not obs:
        return pd.Series(dtype=float, name=series_id)
    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"])
    df[series_id] = pd.to_numeric(df["value"].replace(".", np.nan), errors="coerce")
    s = df.set_index("date")[series_id].dropna().sort_index()
    s.name = series_id
    return s


@st.cache_data(ttl=3600, show_spinner=False)
def load_fred_bundle(api_key: str) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for key, sid in SERIES.items():
        try:
            out[key] = fred_series(sid, api_key)
        except Exception:
            out[key] = pd.Series(dtype=float, name=sid)
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def load_yf_close(symbols: Tuple[str, ...], period: str = "3y") -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    try:
        data = yf.download(list(symbols), period=period, auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.get_level_values(0):
                closes = data["Close"].copy()
            else:
                closes = data.xs("Close", axis=1, level=0, drop_level=True)
        else:
            closes = data.copy()
        closes = pd.DataFrame(closes)
        closes.columns = [str(c) for c in closes.columns]
        return closes.dropna(how="all")
    except Exception:
        return pd.DataFrame()


def _recursive_find_score(obj, preferred_keys=("fear_and_greed", "fearAndGreed", "feargreed")) -> Optional[int]:
    def maybe_score(d: dict) -> Optional[int]:
        score = d.get("score")
        if isinstance(score, (int, float)) and 0 <= float(score) <= 100:
            return int(round(float(score)))
        value = d.get("value")
        if isinstance(value, (int, float)) and 0 <= float(value) <= 100:
            return int(round(float(value)))
        return None

    if isinstance(obj, dict):
        for key in preferred_keys:
            if key in obj and isinstance(obj[key], dict):
                sc = maybe_score(obj[key])
                if sc is not None:
                    return sc
        sc = maybe_score(obj)
        if sc is not None and any(k in obj for k in ("rating", "status", "previous_close")):
            return sc
        for v in obj.values():
            sc = _recursive_find_score(v, preferred_keys)
            if sc is not None:
                return sc
    elif isinstance(obj, list):
        for item in obj:
            sc = _recursive_find_score(item, preferred_keys)
            if sc is not None:
                return sc
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_cnn_fear_greed() -> Dict[str, object]:
    result = {"value": None, "source": "cnn", "status": "unavailable", "updated": None}

    try:
        r = requests.get(CNN_FG_URL, headers=UA, timeout=20)
        if r.ok:
            text = r.text
            patterns = [
                r'"fear_and_greed"\s*:\s*\{[^\}]*?"score"\s*:\s*(\d{1,3})',
                r'"score"\s*:\s*(\d{1,3})\s*,\s*"rating"',
                r'"current"\s*:\s*\{[^\}]*?"value"\s*:\s*(\d{1,3})',
                r'Fear\s*&\s*Greed[^\d]{0,40}(\d{1,3})',
            ]
            for pat in patterns:
                m = re.search(pat, text, flags=re.I | re.S)
                if m:
                    value = int(m.group(1))
                    if 0 <= value <= 100:
                        result.update({
                            "value": value,
                            "status": "ok",
                            "updated": datetime.now(timezone.utc).isoformat(),
                        })
                        return result
    except Exception:
        pass

    try:
        r = requests.get(CNN_FG_JSON_URL, headers=UA, timeout=20)
        r.raise_for_status()
        data = r.json()
        value = _recursive_find_score(data)
        if value is None and isinstance(data, dict):
            for key in ("fear_and_greed_historical", "fear_and_greed"):
                if key in data and isinstance(data[key], list) and data[key]:
                    value = _recursive_find_score(data[key][-1])
                    if value is not None:
                        break
        if value is not None and 0 <= value <= 100:
            updated = data.get("timestamp") if isinstance(data, dict) else None
            result.update({"value": int(value), "status": "ok", "updated": updated})
            return result
    except Exception:
        pass

    return result


def get_status(prob: float) -> Tuple[str, str, float]:
    if prob >= 0.75:
        return STATUS_MAP["confirmed"]
    if prob >= 0.45:
        return STATUS_MAP["starting"]
    if prob >= 0.15:
        return STATUS_MAP["notyet"]
    return STATUS_MAP["invalid"]


def escape_text(text: str) -> str:
    return html.escape(str(text))


def state_chip_html(label: str) -> str:
    colors = {
        "Valid": "#22c55e",
        "At Risk": "#eab308",
        "Losing Validity": "#f97316",
        "Invalid": "#ef4444",
        "Watch": "#3b82f6",
    }
    bg = colors.get(label, "#3b82f6")
    return (
        f"<span style='display:inline-block;padding:6px 10px;border-radius:999px;background:{bg};"
        "color:#08111f;font-weight:800;font-size:13px'>"
        f"{escape_text(label)}</span>"
    )


def status_badge_html(icon: str, text: str) -> str:
    bg = {
        "✅": "rgba(34,197,94,.18)",
        "🟡": "rgba(250,204,21,.18)",
        "⚪": "rgba(255,255,255,.08)",
        "❌": "rgba(239,68,68,.18)",
    }.get(icon, "rgba(255,255,255,.08)")
    return (
        f"<span style='display:inline-block;padding:6px 10px;border-radius:999px;background:{bg};font-weight:700'>"
        f"{escape_text(icon)} {escape_text(text)}</span>"
    )


def regime_stage(max_path_score: float) -> str:
    if max_path_score < 35:
        return "Early"
    if max_path_score < 60:
        return "Mid"
    return "Late"


def classify_validity(current_score: float, max_path_score: float) -> str:
    if current_score > 55 and max_path_score < 50:
        return "Valid"
    if current_score > 50 and 40 <= max_path_score < 60:
        return "At Risk"
    if current_score < 50 and max_path_score >= 60:
        return "Losing Validity"
    if current_score < 45 and max_path_score >= 75:
        return "Invalid"
    return "At Risk"


def meter_label(score: float) -> str:
    if score >= 80:
        return "Extreme"
    if score >= 65:
        return "High"
    if score >= 45:
        return "Elevated"
    if score >= 25:
        return "Moderate"
    return "Low"


def fear_greed_overlays(fg_value: int) -> Dict[str, float]:
    if fg_value is None or fg_value < 0:
        return {
            "fg_norm": np.nan,
            "fg_greed": 0.0,
            "fg_fear": 0.0,
            "fg_extreme_greed": 0.0,
            "fg_extreme_fear": 0.0,
            "fg_big_crash_overlay": 0.0,
            "fg_short_risk_on": 0.0,
        }
    fg_norm = max(0, min(100, fg_value)) / 100.0
    fg_greed = max(0.0, (fg_norm - 0.5) / 0.5)
    fg_fear = max(0.0, (0.5 - fg_norm) / 0.5)
    fg_extreme_greed = max(0.0, (fg_norm - 0.8) / 0.2)
    fg_extreme_fear = max(0.0, (0.25 - fg_norm) / 0.25)
    sweet = 1.0 - min(1.0, abs(fg_norm - 0.65) / 0.35)
    fg_short_risk_on = sweet
    fg_big_crash_overlay = max(0.65 * fg_extreme_greed, fg_extreme_fear)
    return {
        "fg_norm": fg_norm,
        "fg_greed": fg_greed,
        "fg_fear": fg_fear,
        "fg_extreme_greed": fg_extreme_greed,
        "fg_extreme_fear": fg_extreme_fear,
        "fg_big_crash_overlay": fg_big_crash_overlay,
        "fg_short_risk_on": fg_short_risk_on,
    }


def latest_signal_snapshot(bundle: Dict[str, pd.Series], fg_value: int) -> Dict[str, float]:
    wei = bundle["WEI"]
    icsa = bundle["ICSA"]
    t10y2y = bundle["T10Y2Y"]
    cpi = bundle["CPI"]
    core_cpi = bundle["CORE_CPI"]
    breakeven = bundle["BREAKEVEN_5Y"]
    oil = bundle["OIL"]
    hy = bundle["HY_OAS"]
    nfci = bundle["NFCI"]
    stlfsi = bundle["STLFSI4"]
    vix = bundle["VIX"]
    sahm = bundle["SAHM"]
    recpro = bundle["RECPRO"]

    claims4 = icsa.rolling(4).mean()
    claims_yoy = claims4.pct_change(52) * 100.0
    claims_13w = pct_change(claims4, 13)

    cpi3 = ann_roc(cpi, 3)
    cpi6 = ann_roc(cpi, 6)
    core3 = ann_roc(core_cpi, 3)
    core6 = ann_roc(core_cpi, 6)
    cpi_gap = cpi3 - cpi6
    core_gap = core3 - core6

    oil21 = pct_change(oil, 21)
    oil63 = pct_change(oil, 63)
    breakeven20 = breakeven.diff(20)
    wei4 = wei.diff(4)

    signals: Dict[str, float] = {
        "wei_level_z": rolling_z_last(wei, 156),
        "wei_4w_z": rolling_z_last(wei4, 156),
        "claims_yoy_z": rolling_z_last(claims_yoy, 156),
        "claims_13w_z": rolling_z_last(claims_13w, 156),
        "curve_z": rolling_z_last(t10y2y, 252),
        "cpi3_z": rolling_z_last(cpi3, 60),
        "core3_z": rolling_z_last(core3, 60),
        "cpi_gap_z": rolling_z_last(cpi_gap, 60),
        "core_gap_z": rolling_z_last(core_gap, 60),
        "breakeven_z": rolling_z_last(breakeven, 252),
        "breakeven20_z": rolling_z_last(breakeven20, 252),
        "oil21_z": rolling_z_last(oil21, 252),
        "oil63_z": rolling_z_last(oil63, 252),
        "hy_z": rolling_z_last(hy, 252),
        "nfci_z": rolling_z_last(nfci, 156),
        "stlfsi_z": rolling_z_last(stlfsi, 156),
        "vix_z": rolling_z_last(vix, 252),
        "sahm_z": rolling_z_last(sahm, 60),
        "recpro_z": rolling_z_last(recpro, 60),
        "sahm_last": safe_last(sahm),
        "recpro_last": safe_last(recpro),
        "cpi3_last": safe_last(cpi3),
        "core3_last": safe_last(core3),
        "curve_last": safe_last(t10y2y),
        "wei_last": safe_last(wei),
        "claims_last": safe_last(claims4),
        "breakeven_last": safe_last(breakeven),
        "hy_last": safe_last(hy),
        "last_WEI": last_date(wei),
        "last_ICSA": last_date(icsa),
        "last_T10Y2Y": last_date(t10y2y),
        "last_CPI": last_date(cpi),
        "last_CORE_CPI": last_date(core_cpi),
        "last_OIL": last_date(oil),
        "last_HY_OAS": last_date(hy),
        "last_VIX": last_date(vix),
    }

    yf_close = load_yf_close(("IWM", "SPY", "QQQ"))
    if not yf_close.empty and all(col in yf_close.columns for col in ["IWM", "SPY"]):
        ratio = (yf_close["IWM"] / yf_close["SPY"]).dropna()
        rel63 = pct_change(ratio, 63)
        rel20 = pct_change(ratio, 20)
        dist_high = ((ratio / ratio.rolling(252).max()) - 1.0) * 100.0
        iwm_20 = pct_change(yf_close["IWM"], 20)
        iwm_63 = pct_change(yf_close["IWM"], 63)
        signals.update(
            {
                "iwm_rel63_z": rolling_z_last(rel63, 252),
                "iwm_rel20_z": rolling_z_last(rel20, 252),
                "iwm_dist_z": rolling_z_last(dist_high, 252),
                "iwm_20_z": rolling_z_last(iwm_20, 252),
                "iwm_63_z": rolling_z_last(iwm_63, 252),
                "iwm_rel_last": safe_last(ratio),
            }
        )
    else:
        signals.update(
            {
                "iwm_rel63_z": 0.0,
                "iwm_rel20_z": 0.0,
                "iwm_dist_z": 0.0,
                "iwm_20_z": 0.0,
                "iwm_63_z": 0.0,
                "iwm_rel_last": np.nan,
            }
        )

    signals.update(fear_greed_overlays(fg_value))

    growth_level_raw = 0.50 * signals["wei_level_z"] + 0.20 * (-signals["claims_yoy_z"]) + 0.15 * signals["curve_z"]
    growth_mom_raw = 0.55 * signals["wei_4w_z"] + 0.45 * (-signals["claims_13w_z"])
    inflation_level = (
        0.35 * signals["cpi3_z"]
        + 0.25 * signals["core3_z"]
        + 0.20 * signals["breakeven_z"]
        + 0.20 * signals["oil63_z"]
    )
    inflation_mom = (
        0.40 * signals["cpi_gap_z"]
        + 0.30 * signals["core_gap_z"]
        + 0.15 * signals["breakeven20_z"]
        + 0.15 * signals["oil21_z"]
    )
    credit_stress = 0.35 * signals["hy_z"] + 0.25 * signals["nfci_z"] + 0.20 * signals["stlfsi_z"] + 0.20 * signals["vix_z"]
    labor_soft = 0.65 * signals["claims_yoy_z"] + 0.35 * signals["sahm_z"]
    recession_risk = 0.40 * signals["sahm_z"] + 0.35 * signals["recpro_z"] + 0.25 * (-signals["curve_z"])
    iwm_fragility = 0.45 * (-signals["iwm_rel63_z"]) + 0.35 * (-signals["iwm_dist_z"]) + 0.20 * (-signals["iwm_rel20_z"])
    iwm_euphoria = 0.45 * signals["iwm_63_z"] + 0.35 * signals["iwm_20_z"] + 0.20 * signals["iwm_dist_z"]
    breadth_health = 0.55 * signals["iwm_rel63_z"] + 0.25 * signals["iwm_rel20_z"] + 0.20 * signals["iwm_dist_z"]

    # Make growth classification less "sticky Q2" by penalizing transition / slowdown risk.
    growth_level = growth_level_raw - 0.18 * credit_stress - 0.15 * labor_soft
    growth_mom = growth_mom_raw - 0.22 * labor_soft - 0.18 * recession_risk
    growth_transition = sigmoid(
        -1.15 * growth_mom_raw - 0.25 * growth_level_raw + 0.40 * labor_soft + 0.35 * recession_risk + 0.10 * credit_stress
    )

    signals.update(
        {
            "growth_level_raw": growth_level_raw,
            "growth_mom_raw": growth_mom_raw,
            "growth_level": growth_level,
            "growth_mom": growth_mom,
            "growth_transition": growth_transition,
            "inflation_level": inflation_level,
            "inflation_mom": inflation_mom,
            "credit_stress": credit_stress,
            "labor_soft": labor_soft,
            "recession_risk": recession_risk,
            "iwm_fragility": iwm_fragility,
            "iwm_euphoria": iwm_euphoria,
            "breadth_health": breadth_health,
        }
    )

    credit_prob = sigmoid(credit_stress)
    labor_prob = sigmoid(labor_soft)
    recession_prob = sigmoid(recession_risk)
    breadth_prob = sigmoid(breadth_health)

    g_up = sigmoid(
        0.75 * growth_level
        + 0.95 * growth_mom
        - 0.35 * credit_stress
        - 0.15 * iwm_fragility
        - 0.10 * np.maximum(inflation_mom, 0.0)
    )
    g_down = sigmoid(
        -0.60 * growth_level
        - 1.05 * growth_mom
        + 0.65 * recession_risk
        + 0.40 * labor_soft
        + 0.20 * credit_stress
        + 0.10 * iwm_fragility
    )
    i_up = sigmoid(0.65 * inflation_level + 1.10 * inflation_mom)
    signals["g_up"] = g_up
    signals["g_down"] = g_down
    signals["i_up"] = i_up

    short_risk_on = clamp01(
        sigmoid(
            0.95 * breadth_health
            - 0.75 * credit_stress
            - 0.65 * signals["vix_z"]
            - 0.35 * iwm_fragility
            + 0.55 * (signals["fg_short_risk_on"] - 0.5)
        )
    )
    short_risk_off = clamp01(
        sigmoid(
            0.85 * credit_stress
            + 0.80 * signals["vix_z"]
            + 0.70 * iwm_fragility
            + 0.50 * signals["fg_fear"]
            - 0.35 * growth_mom
        )
    )
    big_crash = clamp01(
        sigmoid(
            0.95 * credit_stress
            + 0.95 * recession_risk
            + 0.55 * labor_soft
            + 0.45 * iwm_fragility
            + 0.35 * signals["fg_big_crash_overlay"]
            - 0.20 * breadth_health
        )
    )
    long_risk_on = clamp01(
        sigmoid(
            0.85 * growth_level
            + 0.95 * growth_mom
            - 0.55 * credit_stress
            - 0.55 * recession_risk
            - 0.25 * np.maximum(inflation_level, 0)
            + 0.40 * breadth_health
            + 0.15 * (signals["fg_norm"] - 0.5 if not np.isnan(signals["fg_norm"]) else 0.0)
        )
    )
    signals.update(
        {
            "short_risk_on": short_risk_on,
            "short_risk_off": short_risk_off,
            "big_crash": big_crash,
            "long_risk_on": long_risk_on,
        }
    )

    q1_score = clamp01((g_up + (1 - i_up)) / 2 + 0.08 * breadth_prob - 0.12 * growth_transition - 0.05 * credit_prob)
    q2_score = clamp01((g_up + i_up) / 2 - 0.20 * growth_transition - 0.10 * recession_prob - 0.06 * labor_prob)
    q3_score = clamp01((g_down + i_up) / 2 + 0.20 * growth_transition + 0.08 * labor_prob + 0.04 * credit_prob)
    q4_score = clamp01((g_down + (1 - i_up)) / 2 + 0.10 * recession_prob + 0.06 * credit_prob - 0.04 * breadth_prob)

    signals["quad_scores"] = {
        "Q1": q1_score * 100,
        "Q2": q2_score * 100,
        "Q3": q3_score * 100,
        "Q4": q4_score * 100,
    }
    return signals


def reason_lines(signals: Dict[str, float], quad: str) -> Tuple[List[str], List[str]]:
    gl = signals["growth_level"]
    gm = signals["growth_mom"]
    il = signals["inflation_level"]
    im = signals["inflation_mom"]
    cs = signals["credit_stress"]
    rr = signals["recession_risk"]

    if quad == "Q1":
        valid = [
            "Growth composite is still positive / stabilizing" if gl >= -0.1 else "Growth resilience is fading",
            "Inflation cooling is still largely intact" if im < 0 else "Inflation is trying to re-heat",
            "Breadth / beta quality is improving more than in a pure slowdown regime" if signals["breadth_health"] >= -0.2 else "Breadth is not yet convincingly broad",
        ]
        invalid = [
            "Inflation re-accelerates across CPI / breakevens / oil",
            "Growth rolls over hard enough to lose the early-recovery setup",
            "Credit stress rises enough to shut down broader risk-on",
        ]
    elif quad == "Q2":
        valid = [
            "Growth composite is still positive" if gl >= 0 else "Growth composite is no longer strongly positive",
            "Inflation trend is still firm / sticky" if il >= 0 or im >= 0 else "Inflation is cooling materially",
            "Credit stress is not yet dominant" if cs < 0.8 else "Credit stress is already elevated",
        ]
        invalid = [
            "Growth momentum rolls over decisively",
            "Disinflation accelerates across CPI / breakevens / oil",
            "Credit stress and recession risk confirm deterioration",
        ]
    elif quad == "Q3":
        valid = [
            "Growth remains weak / vulnerable" if gm < 0 else "Growth weakness is fading",
            "Inflation stays firm / sticky" if il >= 0 or im >= 0 else "Inflation pressure is easing",
            "Hard-asset regime is still supported" if cs < 1.5 else "Systemic stress is becoming too dominant",
        ]
        invalid = [
            "Inflation rolls over materially",
            "Growth bottoms and starts recovering",
            "Credit shock overwhelms the stagflation setup",
        ]
    else:
        valid = [
            "Growth remains soft / slowing" if gm < 0 else "Growth is trying to bottom",
            "Inflation is still cooling" if im < 0 else "Inflation has stopped cooling",
            "Recession / slowdown risk remains present" if rr > 0 else "Recession risk is easing",
        ]
        invalid = [
            "Growth bottoms and turns up",
            "Inflation stops falling and starts to firm",
            "Breadth and cyclicals start confirming recovery",
        ]
    return valid, invalid


def req(name: str, prob: float) -> Dict[str, object]:
    icon, label, status_score = get_status(prob)
    return {"name": name, "prob": prob, "icon": icon, "label": label, "status_score": status_score}


def build_paths(signals: Dict[str, float], quad: str) -> List[Dict[str, object]]:
    gl, gm = signals["growth_level"], signals["growth_mom"]
    il, im = signals["inflation_level"], signals["inflation_mom"]
    cs, ls, rr = signals["credit_stress"], signals["labor_soft"], signals["recession_risk"]
    iwm_f, iwm_e = signals["iwm_fragility"], signals["iwm_euphoria"]

    infl_roll = sigmoid(-1.15 * im - 0.20 * il + 0.05 * cs)
    infl_reheat = sigmoid(1.15 * im + 0.20 * il + 0.10 * iwm_e)
    growth_resilient = sigmoid(0.85 * gl + 0.55 * gm - 0.25 * ls - 0.15 * cs)
    growth_roll = sigmoid(-1.20 * gm - 0.20 * gl + 0.25 * cs + 0.20 * iwm_f)
    growth_bottom = sigmoid(1.00 * gm + 0.30 * gl - 0.20 * rr)
    labor_soft = sigmoid(0.85 * ls + 0.20 * cs)
    labor_stable = sigmoid(-0.85 * ls + 0.20 * growth_resilient)
    infl_sticky = sigmoid(0.80 * il + 0.55 * im)
    infl_cooling = sigmoid(-0.80 * il - 0.70 * im)
    commodity_cool = sigmoid(-0.75 * signals["oil21_z"] - 0.75 * signals["breakeven20_z"])
    commodity_rise = sigmoid(0.75 * signals["oil21_z"] + 0.75 * signals["breakeven20_z"])
    growth_weak = sigmoid(-0.85 * gm - 0.20 * gl + 0.20 * rr)

    if quad == "Q1":
        raw = [
            {
                "target": "Q2",
                "title": "Path to Quad 2 — Inflation Reheats",
                "requirements": [
                    req("Inflation stops cooling and turns up", infl_reheat),
                    req("Commodities / breakevens rising", commodity_rise),
                    req("Growth remains stable", growth_resilient),
                ],
                "weights": [0.40, 0.35, 0.25],
                "winners": "cyclicals, banks, value, industrials, reflation trades",
            },
            {
                "target": "Q4",
                "title": "Path to Quad 4 — Growth Down First",
                "requirements": [
                    req("Growth nowcast rolling over", growth_roll),
                    req("Labor softening", labor_soft),
                    req("Inflation keeps cooling", infl_cooling),
                ],
                "weights": [0.40, 0.30, 0.30],
                "winners": "bonds, defensives, quality, staples",
            },
        ]
    elif quad == "Q2":
        raw = [
            {
                "target": "Q1",
                "title": "Path to Quad 1 — Inflation Down First",
                "requirements": [
                    req("Inflation trend rolling over", infl_roll),
                    req("Commodities / breakevens cooling", commodity_cool),
                    req("Growth still resilient", growth_resilient),
                ],
                "weights": [0.40, 0.35, 0.25],
                "winners": "duration, quality growth, semis, broader beta",
            },
            {
                "target": "Q3",
                "title": "Path to Quad 3 — Growth Down First",
                "requirements": [
                    req("Growth nowcast rolling over", growth_roll),
                    req("Labor softening", labor_soft),
                    req("Inflation still sticky", infl_sticky),
                ],
                "weights": [0.40, 0.35, 0.25],
                "winners": "energy, gold, hard assets, defensives",
            },
        ]
    elif quad == "Q3":
        raw = [
            {
                "target": "Q4",
                "title": "Path to Quad 4 — Inflation Down First",
                "requirements": [
                    req("Inflation trend rolling over", infl_roll),
                    req("Commodities / breakevens cooling", commodity_cool),
                    req("Growth still weak", growth_weak),
                ],
                "weights": [0.40, 0.35, 0.25],
                "winners": "bonds, defensives, quality, disinflation beneficiaries",
            },
            {
                "target": "Q2",
                "title": "Path to Quad 2 — Growth Recovers First",
                "requirements": [
                    req("Growth bottoms and turns up", growth_bottom),
                    req("Labor stabilizes", labor_stable),
                    req("Inflation remains firm", infl_sticky),
                ],
                "weights": [0.40, 0.30, 0.30],
                "winners": "cyclicals, industrials, value, nominal-growth winners",
            },
        ]
    else:
        raw = [
            {
                "target": "Q1",
                "title": "Path to Quad 1 — Growth Bottoms First",
                "requirements": [
                    req("Growth bottoms and turns up", growth_bottom),
                    req("Labor stabilizes", labor_stable),
                    req("Inflation keeps cooling / stays tame", infl_cooling),
                ],
                "weights": [0.40, 0.30, 0.30],
                "winners": "small caps, semis, quality growth, broader risk-on",
            },
            {
                "target": "Q3",
                "title": "Path to Quad 3 — Inflation Rises While Growth Stays Weak",
                "requirements": [
                    req("Inflation bottoms and turns up", infl_reheat),
                    req("Commodities rising", commodity_rise),
                    req("Growth still weak", growth_weak),
                ],
                "weights": [0.40, 0.35, 0.25],
                "winners": "energy, gold, real assets",
            },
        ]

    for p in raw:
        p.update(PATH_META.get((quad, p["target"]), {}))
    return raw


def finalize_paths(paths: List[Dict[str, object]]) -> List[Dict[str, object]]:
    final = []
    for p in paths:
        weights = p["weights"]
        score = 100 * sum(r["status_score"] * w for r, w in zip(p["requirements"], weights))
        color = "#22c55e" if score < 60 else "#f59e0b" if score < 75 else "#ef4444"
        final.append({**p, "score": score, "color": color})
    return final


def forecast_summary(current_quad: str, current_score: float, paths: List[Dict[str, object]], validity: str) -> Dict[str, str]:
    ordered = sorted(paths, key=lambda x: x["score"], reverse=True)
    primary = ordered[0]
    alternate = ordered[1]
    base_hold = current_quad if current_score >= primary["score"] or validity in ("Valid", "At Risk") else primary["target"]
    return {
        "phase_text": CURRENT_PHASE_TEXT[current_quad],
        "bias": current_quad,
        "next_likely": primary["target"],
        "alternate": alternate["target"],
        "base_hold": base_hold,
        "transition": primary["target"],
        "primary_score": f"{primary['score']:.0f}",
        "alternate_score": f"{alternate['score']:.0f}",
    }


def overview_metrics(signals: Dict[str, float], fg_info: Dict[str, object]) -> None:
    fg_text = "N/A" if np.isnan(signals["fg_norm"]) else f"{signals['fg_norm'] * 100:.0f}"
    fg_source = "CNN" if fg_info.get("status") == "ok" else "Manual / N.A."
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Growth Up Prob", f"{signals['g_up'] * 100:.0f}%")
    c2.metric("Inflation Up Prob", f"{signals['i_up'] * 100:.0f}%")
    c3.metric("Credit Stress", f"{signals['credit_stress']:.2f}")
    c4.metric("Recession Risk", f"{signals['recession_risk']:.2f}")
    c5.metric("IWM Fragility", f"{signals['iwm_fragility']:.2f}")
    c6.metric("Fear & Greed", fg_text, fg_source)


def render_forecast_summary_row(current_quad: str, current_quad_score: float, validity: str, primary_path: Dict[str, object]) -> None:
    target_quad = str(primary_path["target"])
    target_meta = QUAD_META[target_quad]
    st.markdown("### Forecast Snapshot")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Current Quad", current_quad)
    with c2:
        st.metric("Forecast Bias", current_quad)
    with c3:
        st.metric("Current Validity", validity)
    with c4:
        st.metric("Next Likely Quad", target_quad)
    with c5:
        st.metric("Transition Score", f"{primary_path['score']:.0f}/100")
    with c6:
        st.metric("Quad Fit", f"{current_quad_score:.0f}/100")

    st.markdown(
        f"""
        <div class='section-note'>
            <div style='font-weight:800;margin-bottom:8px'>If forecast benar → {escape_text(target_meta['name'])}: {escape_text(target_meta['phase'])}</div>
            <div style='margin-bottom:6px'><b>Possible:</b> {escape_text(primary_path.get('possible', ''))}</div>
            <div style='margin-bottom:6px'><b>Likely strong:</b> {escape_text(primary_path.get('winners', ''))}</div>
            <div><b>Likely laggards:</b> {escape_text(primary_path.get('laggards', ''))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_meter_cards(signals: Dict[str, float]) -> None:
    st.markdown("### Crash / Risk Meters")
    cards = [
        ("Risk-Off Jangka Pendek", signals["short_risk_off"] * 100, "buat baca panic / de-risking cepat"),
        ("Risk-On Jangka Pendek", signals["short_risk_on"] * 100, "buat baca appetite ambil beta sekarang"),
        ("BIG CRASH", signals["big_crash"] * 100, "buat baca kerusakan sistemik / recession / credit event"),
        ("Risk-On Jangka Panjang", signals["long_risk_on"] * 100, "buat baca dukungan regime beberapa minggu-bulan"),
    ]
    cols = st.columns(4)
    for col, (name, score, desc) in zip(cols, cards):
        label = meter_label(score)
        color = "#22c55e" if score < 35 else "#eab308" if score < 60 else "#f97316" if score < 80 else "#ef4444"
        with col:
            st.markdown(
                f"""
                <div class='soft-card'>
                    <div style='font-size:14px;opacity:.9;margin-bottom:8px'>{escape_text(name)}</div>
                    <div style='font-size:30px;font-weight:900;margin-bottom:8px'>{score:.0f}/100</div>
                    <div style='display:inline-block;padding:4px 10px;border-radius:999px;background:{color};color:#07110f;font-weight:800;font-size:12px;margin-bottom:10px'>{escape_text(label)}</div>
                    <div style='font-size:12px;opacity:.8'>{escape_text(desc)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_hero(current_quad: str, quad_score: float, validity: str, primary_path: Dict[str, object], stage: str) -> None:
    meta = QUAD_META[current_quad]
    st.markdown(
        f"""
        <div class='main-card'>
            <div style='display:flex;justify-content:space-between;gap:24px;align-items:flex-start;flex-wrap:wrap'>
                <div>
                    <div style='font-size:18px;font-weight:800;margin-bottom:14px'>{escape_text(meta['name'])}</div>
                    <div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px'>
                        <span style='display:inline-block;padding:6px 10px;border-radius:999px;background:#19e68c;color:#07110f;font-weight:800;font-size:12px'>WE ARE HERE</span>
                        <span style='display:inline-block;padding:6px 10px;border-radius:999px;background:#2596ff;color:white;font-weight:800;font-size:12px'>FORECAST</span>
                    </div>
                    <div style='margin-bottom:8px'><span style='font-weight:700'>Phase Name:</span> {escape_text(meta['phase'])}</div>
                    <div style='margin-bottom:8px'><span style='font-weight:700'>Stage:</span> {escape_text(stage)}</div>
                    <div><span style='font-weight:700'>Logic:</span> {escape_text(meta['logic'])}</div>
                </div>
                <div style='min-width:290px'>
                    <div style='margin-bottom:10px'><span style='opacity:.8'>Current Validity</span><br>{state_chip_html(validity)}</div>
                    <div style='margin-bottom:10px'><span style='opacity:.8'>Primary Transition</span><br><span style='font-weight:800'>{escape_text(primary_path['target'])} Watch</span></div>
                    <div style='margin-bottom:6px'><span style='opacity:.8'>Transition Score</span><br><span style='font-size:24px;font-weight:900'>{primary_path['score']:.0f}/100</span></div>
                    <div style='opacity:.8'>Quad Fit Score: <b>{quad_score:.0f}/100</b></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_buckets_column(title: str, buckets: Dict[str, Dict[str, List[str]]], expand_first: bool = True) -> None:
    st.markdown(f"#### {title}")
    first_bucket = True
    for bucket, content in buckets.items():
        with st.expander(bucket, expanded=expand_first and first_bucket):
            first_sub = True
            for subhead, items in content.items():
                with st.expander(subhead, expanded=first_sub):
                    for item in items:
                        st.write(f"• {item}")
                first_sub = False
        first_bucket = False


def render_phase_matrix(quad: str) -> None:
    guide = PHASE_GUIDE[quad]
    c1, c2, c3 = st.columns([1, 1.15, 1.15])
    with c1:
        st.markdown("#### Meaning")
        for section, items in guide["meaning"].items():
            with st.expander(section, expanded=(section == "Macro")):
                for item in items:
                    st.write(f"• {item}")
    with c2:
        render_buckets_column("Winners", guide["winners"])
    with c3:
        render_buckets_column("Losers", guide["losers"])


def render_phase_guide(quad: str) -> None:
    st.markdown("### Current Phase")
    st.markdown(
        f"<div class='section-note'><b>{escape_text(QUAD_META[quad]['name'])}:</b> {escape_text(CURRENT_PHASE_TEXT[quad])}</div>",
        unsafe_allow_html=True,
    )
    render_phase_matrix(quad)


def render_requirement_row(requirement: Dict[str, object]) -> None:
    left, right = st.columns([3.2, 1])
    left.write(requirement["name"])
    right.markdown(status_badge_html(requirement["icon"], requirement["label"]), unsafe_allow_html=True)


def render_target_quad_snapshot(target_quad: str) -> None:
    meta = QUAD_META[target_quad]
    guide = PHASE_GUIDE[target_quad]
    with st.expander(f"{meta['name']} Snapshot", expanded=False):
        st.write(f"**Phase:** {meta['phase']}")
        st.write(f"**Logic:** {meta['logic']}")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Meaning**")
            for section, items in guide["meaning"].items():
                with st.expander(section, expanded=(section == "Macro")):
                    for item in items:
                        st.write(f"• {item}")
        with c2:
            st.markdown("**Winners**")
            for bucket, content in guide["winners"].items():
                with st.expander(bucket, expanded=False):
                    for subhead, items in content.items():
                        st.write(f"**{subhead}**")
                        for item in items:
                            st.write(f"• {item}")
        with c3:
            st.markdown("**Losers**")
            for bucket, content in guide["losers"].items():
                with st.expander(bucket, expanded=False):
                    for subhead, items in content.items():
                        st.write(f"**{subhead}**")
                        for item in items:
                            st.write(f"• {item}")


def render_path_card(path: Dict[str, object]) -> None:
    with st.container(border=True):
        st.markdown(f"#### {path['title']}")
        for requirement in path["requirements"]:
            render_requirement_row(requirement)
        st.caption(f"Path Score {path['score']:.0f}/100")
        st.progress(max(0.0, min(1.0, path["score"] / 100.0)))
        st.write(f"**Possible:** {path.get('possible', '-')}")
        st.write(f"**If forecast benar:** {path.get('if_right', '-')}")
        st.write(f"**Likely strong:** {path.get('winners', '-')}")
        st.write(f"**Likely laggards:** {path.get('laggards', '-')}")
        render_target_quad_snapshot(path["target"])


def render_path_tree(paths: List[Dict[str, object]]) -> None:
    st.markdown("### Transition Tree")
    for idx, path in enumerate(paths):
        prefix = "├─" if idx == 0 else "└─"
        with st.expander(f"{prefix} {path['title']}", expanded=False):
            st.markdown(f"<div class='tree-box'><b>{escape_text(path['title'])}</b><br><span class='small-muted'>Path Score {path['score']:.0f}/100</span></div>", unsafe_allow_html=True)
            for i, requirement in enumerate(path["requirements"], start=1):
                st.write(f"{i}. {requirement['name']} — {requirement['icon']} {requirement['label']}")
            st.progress(max(0.0, min(1.0, path["score"] / 100.0)))
            with st.expander("Possible", expanded=True):
                st.write(path.get("possible", "-"))
            with st.expander("If Forecast Benar", expanded=True):
                st.write(path.get("if_right", "-"))
            with st.expander("Likely Strong / Laggards", expanded=True):
                st.write(f"**Likely strong:** {path.get('winners', '-')}")
                st.write(f"**Likely laggards:** {path.get('laggards', '-')}")


def render_quad_detail(quad: str, signals: Dict[str, float], current_quad: str) -> None:
    meta = QUAD_META[quad]
    paths = finalize_paths(build_paths(signals, quad))
    quad_score = signals["quad_scores"][quad]
    primary_path = max(paths, key=lambda x: x["score"])
    validity = classify_validity(quad_score, max(p["score"] for p in paths)) if quad == current_quad else "Watch"
    stage = regime_stage(max(p["score"] for p in paths))
    valid_lines, invalid_lines = reason_lines(signals, quad)

    top_c1, top_c2, top_c3 = st.columns([1.2, 1, 1])
    with top_c1:
        st.markdown(f"**{meta['name']}**")
        st.write(meta["phase"])
        st.caption(meta["logic"])
    with top_c2:
        if quad == current_quad:
            st.markdown("**Current Validity**")
            st.markdown(state_chip_html(validity), unsafe_allow_html=True)
        else:
            st.markdown("**Current Relevance**")
            st.metric("Quad Fit", f"{quad_score:.0f}/100")
    with top_c3:
        st.metric("Primary Path", f"{primary_path['target']} | {primary_path['score']:.0f}")
        st.caption(f"Stage: {stage}")

    c1, c2 = st.columns(2)
    with c1:
        render_path_card(paths[0])
    with c2:
        render_path_card(paths[1])

    b1, b2 = st.columns(2)
    with b1:
        st.markdown("### Why Valid / Why This Quad Fits")
        for line in valid_lines:
            st.write(f"• {line}")
    with b2:
        st.markdown("### What Invalidates It")
        for line in invalid_lines:
            st.write(f"• {line}")

    st.markdown("### Positioning Prep")
    st.markdown(
        f"<div class='section-note'><b>If {paths[0]['target']} confirms</b> → likely strong: {escape_text(paths[0]['winners'])}<br><br>"
        f"<b>If {paths[1]['target']} confirms</b> → likely strong: {escape_text(paths[1]['winners'])}</div>",
        unsafe_allow_html=True,
    )


def build_forecast_tables(signals: Dict[str, float], current_quad: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    current_paths = finalize_paths(build_paths(signals, current_quad))
    path_rows = []
    for p in sorted(current_paths, key=lambda x: x["score"], reverse=True):
        path_rows.append(
            {
                "From": current_quad,
                "To": p["target"],
                "Path Score": round(float(p["score"]), 1),
                "Possible": p.get("possible", ""),
                "If Forecast Right": p.get("if_right", ""),
                "Likely Strong": p["winners"],
                "Likely Laggards": p.get("laggards", ""),
            }
        )
    path_df = pd.DataFrame(path_rows)

    quad_rows = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        paths = finalize_paths(build_paths(signals, q))
        primary = max(paths, key=lambda x: x["score"])
        quad_rows.append(
            {
                "Quad": q,
                "Phase": QUAD_META[q]["phase"],
                "Logic": QUAD_META[q]["logic"],
                "Fit Score": round(float(signals["quad_scores"][q]), 1),
                "Primary Path": primary["target"],
                "Transition Score": round(float(primary["score"]), 1),
                "Likely If Right": primary["winners"],
                "Likely Laggards": primary.get("laggards", ""),
            }
        )
    quad_df = pd.DataFrame(quad_rows).sort_values(["Fit Score", "Transition Score"], ascending=False)
    return path_df, quad_df


def raw_signal_table(signals: Dict[str, float]) -> None:
    rows = []
    for k, v in signals.items():
        if isinstance(v, dict):
            continue
        if isinstance(v, pd.Timestamp):
            rows.append((k, str(v.date())))
        elif isinstance(v, (int, float, np.floating)):
            rows.append((k, float(v)))
    df = pd.DataFrame(rows, columns=["signal", "value"]).sort_values("signal")
    st.dataframe(df, use_container_width=True, hide_index=True)


@st.cache_data(ttl=21600, show_spinner=False)
def fred_next_release_for_series(series_id: str, api_key: str) -> Optional[Dict[str, str]]:
    try:
        rel_url = "https://api.stlouisfed.org/fred/series/release"
        rel_params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
        rel_resp = requests.get(rel_url, params=rel_params, timeout=30)
        rel_resp.raise_for_status()
        releases = rel_resp.json().get("releases", [])
        if not releases:
            return None
        release_id = releases[0].get("id")
        release_name = releases[0].get("name")
        if release_id is None:
            return None

        dates_url = "https://api.stlouisfed.org/fred/release/dates"
        date_params = {
            "release_id": release_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "asc",
            "include_release_dates_with_no_data": "true",
        }
        dates_resp = requests.get(dates_url, params=date_params, timeout=30)
        dates_resp.raise_for_status()
        release_dates = dates_resp.json().get("release_dates", [])
        today = pd.Timestamp.utcnow().normalize().date()
        future = [d for d in release_dates if pd.to_datetime(d.get("date")).date() >= today]
        if not future:
            return None
        next_date = pd.to_datetime(future[0]["date"]).date()
        days_left = (next_date - today).days
        return {
            "series_id": series_id,
            "release_name": release_name,
            "next_date": str(next_date),
            "days_left": str(days_left),
        }
    except Exception:
        return None


def render_countdown_cards(api_key: str) -> None:
    st.markdown("### Countdown to New Economy Data")
    cols = st.columns(len(RELEASE_WATCH))
    for col, (label, series_id) in zip(cols, RELEASE_WATCH.items()):
        info = fred_next_release_for_series(series_id, api_key)
        with col:
            if info is None:
                st.markdown(
                    f"""
                    <div class='mini-card'>
                        <div style='font-size:14px;opacity:.9'>{escape_text(label)}</div>
                        <div style='font-size:20px;font-weight:900;margin-top:6px'>N/A</div>
                        <div style='font-size:12px;opacity:.75;margin-top:6px'>{escape_text(series_id)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                continue
            days_left = int(info["days_left"])
            color = "#22c55e" if days_left <= 2 else "#eab308" if days_left <= 7 else "#3b82f6"
            st.markdown(
                f"""
                <div class='mini-card'>
                    <div style='font-size:14px;opacity:.9'>{escape_text(label)}</div>
                    <div style='font-size:26px;font-weight:900;margin-top:6px'>T-{days_left}d</div>
                    <div style='display:inline-block;padding:3px 8px;border-radius:999px;background:{color};color:#07110f;font-size:12px;font-weight:800;margin-top:8px'>{escape_text(info['next_date'])}</div>
                    <div style='font-size:12px;opacity:.75;margin-top:8px'>{escape_text(info['release_name'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_playbook_meaning(quad: str) -> None:
    st.markdown("### Meaning")
    for section, items in PHASE_GUIDE[quad]["meaning"].items():
        with st.expander(section, expanded=False):
            for item in items:
                st.write(f"• {item}")


def render_playbook_section(title: str, buckets: Dict[str, Dict[str, List[str]]]) -> None:
    st.markdown(f"### {title}")
    for bucket, content in buckets.items():
        with st.expander(bucket, expanded=False):
            for subhead, items in content.items():
                st.write(f"**{subhead}**")
                for item in items:
                    st.write(f"• {item}")


def render_possible_next_playbook(quad: str) -> None:

    st.markdown("### Possible Next")
    for target in PLAYBOOK_NEXT.get(quad, []):
        info = PATH_META.get((quad, target), {})
        with st.expander(f"Open transition → {target}", expanded=False):
            if info.get("possible"):
                st.write(f"**Possible:** {info['possible']}")
            if info.get("if_right"):
                st.write(f"**If forecast benar:** {info['if_right']}")
            if QUAD_META.get(target, {}).get("winners"):
                st.write(f"**Likely strong:** {QUAD_META[target]['winners']}")
            if info.get("laggards"):
                st.write(f"**Likely laggards:** {info['laggards']}")

def render_playbook_all_quads(signals: Dict[str, float], current_quad: str) -> None:
    st.markdown("### Quad Playbook (All Quads)")
    current_paths = finalize_paths(build_paths(signals, current_quad))
    next_likely = max(current_paths, key=lambda x: x["score"])["target"]

    for q in ["Q1", "Q2", "Q3", "Q4"]:
        meta = QUAD_META[q]
        paths = finalize_paths(build_paths(signals, q))
        chip = ""
        if q == current_quad:
            chip = "<span style='display:inline-block;padding:4px 8px;border-radius:999px;background:#19e68c;color:#07110f;font-size:11px;font-weight:800;margin-bottom:8px'>CURRENT QUAD</span>"
        elif q == next_likely:
            chip = "<span style='display:inline-block;padding:4px 8px;border-radius:999px;background:#f59e0b;color:#07110f;font-size:11px;font-weight:800;margin-bottom:8px'>NEXT LIKELY</span>"

        with st.expander(f"Open {meta['name']}", expanded=False):
            st.markdown(
                f"""
                <div class='soft-card' style='margin-bottom:14px'>
                    {chip}
                    <div style='font-size:18px;font-weight:800'>{meta['name']}</div>
                    <div style='margin-top:8px'><b>Phase Name:</b> {meta['phase']}</div>
                    <div style='margin-top:6px'><b>Stage:</b> N/A</div>
                    <div style='margin-top:6px'><b>Logic:</b> {meta['logic']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_playbook_meaning(q)
            render_playbook_section("Winners", PHASE_GUIDE[q]["winners"])
            render_playbook_section("Losers", PHASE_GUIDE[q]["losers"])
            render_possible_next_playbook(q)


def main() -> None:
    inject_css()
    st.title("Macro Quad Transition Dashboard")
    st.caption(
        "Fokus: quad sekarang masih valid apa nggak, requirement pindah quad, crash meter, sama ancang-ancang positioning kalau transisi mulai confirm."
    )

    st.sidebar.header("Settings")
    default_key = ""
    try:
        default_key = st.secrets.get("FRED_API_KEY", "")
    except Exception:
        default_key = ""
    fred_key = st.sidebar.text_input("FRED API Key", value=default_key, type="password")
    fg_mode = st.sidebar.radio("Fear & Greed Source", ["CNN Auto", "Manual Override"], index=0)
    manual_fg = st.sidebar.number_input("Manual Fear & Greed (0-100)", min_value=0, max_value=100, value=50)
    show_raw = st.sidebar.checkbox("Show raw signal table", value=False)
    st.sidebar.caption("Fear & Greed default-nya auto dari CNN. Kalau gagal kebaca, fallback pakai manual override.")

    if not fred_key:
        st.warning("Masukin FRED API key dulu di sidebar.")
        st.stop()

    if fg_mode == "CNN Auto":
        with st.spinner("Loading FRED + market overlays + CNN Fear & Greed..."):
            bundle = load_fred_bundle(fred_key)
            fg_info = fetch_cnn_fear_greed()
            fg_value = int(fg_info["value"]) if fg_info.get("status") == "ok" and fg_info.get("value") is not None else int(manual_fg)
            signals = latest_signal_snapshot(bundle, fg_value)
    else:
        with st.spinner("Loading FRED + market overlays..."):
            bundle = load_fred_bundle(fred_key)
            fg_info = {"status": "manual", "value": int(manual_fg), "source": "manual"}
            fg_value = int(manual_fg)
            signals = latest_signal_snapshot(bundle, fg_value)

    if fg_mode == "CNN Auto" and fg_info.get("status") != "ok":
        st.warning("CNN Fear & Greed lagi gagal kebaca. Dashboard sementara pakai manual fallback dari sidebar.")

    quad_scores = signals["quad_scores"]
    current_quad = max(quad_scores, key=quad_scores.get)
    current_paths = finalize_paths(build_paths(signals, current_quad))
    primary_path = max(current_paths, key=lambda x: x["score"])
    current_quad_score = quad_scores[current_quad]
    max_path_score = max(p["score"] for p in current_paths)
    validity = classify_validity(current_quad_score, max_path_score)
    stage = regime_stage(max_path_score)

    overview_metrics(signals, fg_info)
    render_forecast_summary_row(current_quad, current_quad_score, validity, primary_path)
    st.markdown("---")
    render_meter_cards(signals)
    st.markdown("---")
    render_countdown_cards(fred_key)
    st.markdown("---")
    render_hero(current_quad, current_quad_score, validity, primary_path, stage)
    render_phase_guide(current_quad)

    with st.expander(f"Open {QUAD_META[current_quad]['name']}", expanded=True):
        render_quad_detail(current_quad, signals, current_quad)

    render_playbook_all_quads(signals, current_quad)


    if show_raw:
        st.markdown("### Raw Signal Table")
        raw_signal_table(signals)

    st.markdown("---")
    st.caption(
        "Catatan: ini dashboard regime / transition, bukan alat timing intraday. Status requirement pakai multi-signal composite dan path score, jadi lebih susah flip-flop cuma karena satu print data."
    )


if __name__ == "__main__":
    main()
