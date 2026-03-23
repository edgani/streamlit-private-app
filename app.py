from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="IHSG Regime Dashboard", layout="wide")


# =====================================================
# STYLE
# =====================================================

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
            .path-card {
                padding: 16px;
                border-radius: 18px;
                border: 1px solid rgba(255,255,255,.10);
                background: rgba(255,255,255,.03);
                min-height: 250px;
            }
            .section-note {
                padding: 14px 16px;
                border-radius: 14px;
                background: rgba(37, 99, 235, .14);
                border: 1px solid rgba(96, 165, 250, .18);
                margin-bottom: 12px;
            }
            .good {
                color: #21e39b;
                font-weight: 700;
            }
            .bad {
                color: #ff7878;
                font-weight: 700;
            }
            .muted {
                opacity: .82;
                font-size: 12px;
            }
            .pill {
                display:inline-block;
                padding: 4px 10px;
                border-radius: 999px;
                border: 1px solid rgba(255,255,255,.12);
                background: rgba(255,255,255,.05);
                font-size: 12px;
                margin-right: 6px;
                margin-bottom: 6px;
            }
            .big-number {
                font-size: 28px;
                font-weight: 800;
                line-height: 1.1;
            }
            .small-title {
                font-size: 12px;
                opacity: .72;
                text-transform: uppercase;
                letter-spacing: .05em;
            }
            .divider-space {
                margin-top: 8px;
                margin-bottom: 8px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =====================================================
# DATA MODEL
# =====================================================

@dataclass
class DriverState:
    driver: str
    current_quad: str
    current_prob: float
    next_quad: str
    next_prob: float
    phase: str
    growth_axis: float
    inflation_axis: float
    regime_score: float
    risk_on_score: float
    meters: Dict[str, float]
    sector_bias: List[Tuple[str, float, str]]
    winners: List[str]
    losers: List[str]
    focus: List[str]
    validation: List[str]
    invalidation: List[str]
    what_if_rows: List[Dict[str, str]]
    transition_text: str
    signal_table: pd.DataFrame


QUAD_META: Dict[str, Dict[str, List[str] | str]] = {
    "Q1": {
        "name": "Quad 1",
        "phase": "Growth Up / Inflation Down",
        "winners": [
            "Banks besar",
            "Property",
            "Consumer discretionary",
            "Retail domestik",
            "Beta domestik berkualitas",
        ],
        "losers": [
            "Pure defensives yang over-owned",
            "Nama yang cuma menang saat risk-off",
            "Resource names kalau commodity tape melemah",
        ],
    },
    "Q2": {
        "name": "Quad 2",
        "phase": "Growth Up / Inflation Up",
        "winners": [
            "Energy",
            "Coal",
            "Nickel / metals",
            "Banks selektif",
            "Nominal-growth cyclicals",
        ],
        "losers": [
            "Property yang sensitif rates",
            "Bond-proxy defensives",
            "Import-heavy names",
        ],
    },
    "Q3": {
        "name": "Quad 3",
        "phase": "Growth Down / Inflation Up",
        "winners": [
            "Coal dan energy cash-flow",
            "Nickel / gold proxy selektif",
            "Telco defensif",
            "Staples",
            "Healthcare / low-beta",
        ],
        "losers": [
            "Banks beta tinggi",
            "Property",
            "Domestic cyclicals",
            "Nama yang perlu rupiah kuat + rates turun",
        ],
    },
    "Q4": {
        "name": "Quad 4",
        "phase": "Growth Down / Inflation Down",
        "winners": [
            "Defensives cash-flow",
            "Staples",
            "Telco",
            "Healthcare",
            "Quality high-dividend",
        ],
        "losers": [
            "Commodity beta yang butuh hot nominal growth",
            "Property",
            "Lower-quality cyclicals",
            "Banks kalau growth belum bottom",
        ],
    },
}

SECTOR_PROXIES: Dict[str, List[str]] = {
    "Banks": ["BBCA", "BBRI", "BMRI", "BBNI"],
    "Property": ["CTRA", "PWON", "BSDE", "SMRA"],
    "Defensives": ["ICBP", "INDF", "KLBF", "TLKM"],
    "Domestic Cyclicals": ["ASII", "ACES", "AMRT", "ERAA"],
    "Energy": ["ADRO", "ITMG", "PTBA", "PGAS"],
    "Metals / Mining": ["ANTM", "INCO", "MDKA", "UNTR"],
}

SCENARIO_PRESETS: Dict[str, Dict[str, int]] = {
    "Base Case - Late Q3": {
        "growth": -28,
        "inflation": 34,
        "rupiah": -12,
        "foreign": -6,
        "commodity": 30,
        "bi": 22,
        "banks": -18,
        "property": -26,
        "defensives": 22,
        "resources": 34,
    },
    "Domestic Recovery - Q1 Bias": {
        "growth": 38,
        "inflation": -18,
        "rupiah": 24,
        "foreign": 20,
        "commodity": 4,
        "bi": -16,
        "banks": 36,
        "property": 28,
        "defensives": -10,
        "resources": 2,
    },
    "Reflation Commodity - Q2 Bias": {
        "growth": 28,
        "inflation": 30,
        "rupiah": 6,
        "foreign": 12,
        "commodity": 42,
        "bi": 10,
        "banks": 20,
        "property": -6,
        "defensives": -12,
        "resources": 40,
    },
    "Risk-Off Disinflation - Q4 Bias": {
        "growth": -34,
        "inflation": -26,
        "rupiah": 18,
        "foreign": -8,
        "commodity": -18,
        "bi": 8,
        "banks": -22,
        "property": -30,
        "defensives": 34,
        "resources": -10,
    },
}

DEFAULT_CALENDAR = {
    "BI Event": 12,
    "Inflation Update": 10,
    "Trade Balance": 15,
    "GDP Release": 44,
}


# =====================================================
# HELPERS
# =====================================================

def clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def sigmoid(x: float) -> float:
    x = clamp(x, -8, 8)
    return 1.0 / (1.0 + math.exp(-x))


def score_band(x: float) -> str:
    if x >= 25:
        return "Strong"
    if x >= 8:
        return "Positive"
    if x <= -25:
        return "Strong Negative"
    if x <= -8:
        return "Negative"
    return "Neutral"


def traffic_color(value: float) -> str:
    if value >= 65:
        return "#21e39b"
    if value >= 45:
        return "#f6c244"
    return "#ff7878"


def to_0_100(x: float) -> float:
    return clamp(50 + x / 2, 0, 100)


def softmax_probs(scores: Dict[str, float], temp: float = 18.0) -> Dict[str, float]:
    values = np.array(list(scores.values()), dtype=float) / temp
    values = values - np.max(values)
    probs = np.exp(values)
    probs = probs / probs.sum()
    return {k: float(v * 100) for k, v in zip(scores.keys(), probs)}


def display_score_html(title: str, value: str, note: str) -> str:
    return f"""
    <div class='mini-card'>
        <div class='small-title'>{title}</div>
        <div class='big-number'>{value}</div>
        <div class='muted'>{note}</div>
    </div>
    """


# =====================================================
# ENGINE
# =====================================================

def build_driver_inputs(raw: Dict[str, float], driver: str) -> Dict[str, float]:
    growth = raw["growth"]
    inflation = raw["inflation"]
    rupiah = raw["rupiah"]
    foreign = raw["foreign"]
    commodity = raw["commodity"]
    bi = raw["bi"]
    banks = raw["banks"]
    property_ = raw["property"]
    defensives = raw["defensives"]
    resources = raw["resources"]

    if driver == "Monthly (Flow + Sector Pulse)":
        growth_axis = 0.38 * growth + 0.18 * foreign + 0.18 * banks + 0.12 * property_ + 0.14 * rupiah - 0.18 * bi
        inflation_axis = 0.42 * inflation + 0.22 * commodity + 0.16 * resources - 0.18 * rupiah + 0.08 * defensives
    elif driver == "Quarterly Anchor":
        growth_axis = 0.62 * growth + 0.12 * foreign + 0.10 * banks + 0.06 * property_ + 0.10 * rupiah - 0.18 * bi
        inflation_axis = 0.60 * inflation + 0.24 * commodity + 0.10 * resources - 0.14 * rupiah + 0.06 * defensives
    else:
        growth_axis = 0.50 * growth + 0.15 * foreign + 0.15 * banks + 0.10 * property_ + 0.10 * rupiah - 0.18 * bi
        inflation_axis = 0.51 * inflation + 0.23 * commodity + 0.13 * resources - 0.16 * rupiah + 0.07 * defensives

    return {
        **raw,
        "growth_axis": clamp(growth_axis, -100, 100),
        "inflation_axis": clamp(inflation_axis, -100, 100),
    }


def quad_scores(inputs: Dict[str, float]) -> Dict[str, float]:
    g = inputs["growth_axis"]
    i = inputs["inflation_axis"]
    rupiah = inputs["rupiah"]
    foreign = inputs["foreign"]
    commodity = inputs["commodity"]
    bi = inputs["bi"]
    banks = inputs["banks"]
    property_ = inputs["property"]
    defensives = inputs["defensives"]
    resources = inputs["resources"]

    return {
        "Q1": 0.66 * g - 0.58 * i + 0.18 * foreign + 0.16 * rupiah - 0.15 * bi + 0.14 * banks + 0.12 * property_ - 0.08 * defensives,
        "Q2": 0.56 * g + 0.60 * i + 0.10 * foreign + 0.06 * rupiah - 0.10 * bi + 0.10 * banks - 0.06 * property_ + 0.18 * commodity + 0.20 * resources,
        "Q3": -0.56 * g + 0.66 * i - 0.06 * foreign - 0.08 * rupiah + 0.06 * bi - 0.16 * banks - 0.18 * property_ + 0.16 * defensives + 0.24 * resources,
        "Q4": -0.66 * g - 0.56 * i - 0.02 * foreign + 0.14 * rupiah + 0.10 * bi - 0.12 * banks - 0.18 * property_ + 0.26 * defensives - 0.10 * resources,
    }


def phase_label(current_prob: float, next_prob: float) -> str:
    gap = current_prob - next_prob
    if gap >= 18:
        return "Early"
    if gap >= 10:
        return "Mid"
    if gap >= 5:
        return "Late"
    return "Transitioning"


def meters_from_inputs(inputs: Dict[str, float], current_quad: str) -> Dict[str, float]:
    g = inputs["growth_axis"]
    i = inputs["inflation_axis"]
    rupiah = inputs["rupiah"]
    foreign = inputs["foreign"]
    commodity = inputs["commodity"]
    bi = inputs["bi"]
    banks = inputs["banks"]
    defensives = inputs["defensives"]

    risk_on = clamp(50 + 0.28 * g - 0.18 * i + 0.16 * rupiah + 0.16 * foreign - 0.15 * bi + 0.10 * banks - 0.07 * defensives, 0, 100)
    macro_heat = clamp(50 + 0.34 * i + 0.16 * commodity - 0.10 * rupiah, 0, 100)
    rupiah_stress = clamp(50 - 0.58 * rupiah + 0.16 * i + 0.10 * bi, 0, 100)
    domestic_breadth = clamp(50 + 0.34 * g + 0.16 * banks + 0.12 * inputs["property"] + 0.10 * foreign, 0, 100)
    resource_support = clamp(50 + 0.38 * commodity + 0.22 * inputs["resources"] + 0.08 * i, 0, 100)
    defense_need = clamp(50 - 0.20 * g + 0.18 * i - 0.10 * foreign + 0.16 * defensives, 0, 100)

    return {
        "IHSG Risk-On": risk_on,
        "Inflation Pressure": macro_heat,
        "IDR Stress": rupiah_stress,
        "Domestic Breadth": domestic_breadth,
        "Resource Support": resource_support,
        "Defense Need": defense_need,
        "Regime Fit": {
            "Q1": clamp(56 + 0.35 * risk_on - 0.30 * defense_need, 0, 100),
            "Q2": clamp(52 + 0.24 * resource_support + 0.16 * macro_heat + 0.08 * domestic_breadth, 0, 100),
            "Q3": clamp(46 + 0.22 * macro_heat + 0.20 * defense_need + 0.14 * resource_support - 0.18 * domestic_breadth, 0, 100),
            "Q4": clamp(50 + 0.28 * defense_need - 0.22 * macro_heat - 0.10 * domestic_breadth, 0, 100),
        }[current_quad],
    }


def sector_bias_table(inputs: Dict[str, float], current_quad: str, next_quad: str) -> List[Tuple[str, float, str]]:
    scores = {
        "Banks": 0.46 * inputs["growth_axis"] - 0.18 * inputs["inflation_axis"] + 0.22 * inputs["rupiah"] + 0.24 * inputs["foreign"] - 0.22 * inputs["bi"] + 0.26 * inputs["banks"],
        "Property": 0.42 * inputs["growth_axis"] - 0.18 * inputs["inflation_axis"] + 0.18 * inputs["rupiah"] - 0.34 * inputs["bi"] + 0.28 * inputs["property"],
        "Defensives": -0.20 * inputs["growth_axis"] - 0.16 * inputs["foreign"] + 0.10 * inputs["rupiah"] + 0.24 * inputs["defensives"] + 0.16 * inputs["inflation_axis"],
        "Domestic Cyclicals": 0.42 * inputs["growth_axis"] + 0.14 * inputs["foreign"] + 0.10 * inputs["banks"] - 0.18 * inputs["inflation_axis"] - 0.12 * inputs["bi"],
        "Energy": 0.16 * inputs["growth_axis"] + 0.30 * inputs["inflation_axis"] + 0.36 * inputs["commodity"] + 0.20 * inputs["resources"],
        "Metals / Mining": 0.12 * inputs["growth_axis"] + 0.28 * inputs["inflation_axis"] + 0.26 * inputs["commodity"] + 0.34 * inputs["resources"] - 0.08 * inputs["rupiah"],
    }

    rows: List[Tuple[str, float, str]] = []
    for sector, score in scores.items():
        tilt = score_band(score)
        if current_quad == "Q4" and sector in {"Banks", "Property", "Domestic Cyclicals"}:
            tilt += " / Watchlist Only"
        if current_quad == "Q3" and next_quad == "Q4" and sector == "Defensives":
            tilt += " / Improving"
        rows.append((sector, round(score, 1), tilt))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def action_focus(current_quad: str, next_quad: str, phase: str) -> Tuple[List[str], List[str], List[str], str]:
    if current_quad == "Q1":
        focus = [
            "Prioritaskan banks besar + property + domestic cyclicals berkualitas.",
            "Cari breadth domestik yang melebar, bukan cuma satu dua big caps.",
            "Kalau phase udah late, jangan kejar beta paling jelek.",
        ]
        validation = [
            "Banks tetap lead.",
            "Property tidak gagal follow-through.",
            "Rupiah stabil / foreign flow tidak balik kabur.",
        ]
        invalidation = [
            "Inflation naik lagi terlalu cepat.",
            "BI harus jauh lebih hawkish.",
            "Defensives kembali jadi pemimpin utama.",
        ]
        transition = "Jalur berikutnya biasanya ke Q2 kalau inflasi re-heat, atau ke Q4 kalau growth crack duluan."
    elif current_quad == "Q2":
        focus = [
            "Pegang resource winners dulu, lalu banks selektif yang monetise nominal growth.",
            "Jangan over-berat di property kalau rates masih keras.",
            "Kalau breadth sempit, utamakan first-order winners dulu.",
        ]
        validation = [
            "Commodity support tetap hidup.",
            "Banks tidak kehilangan leadership total.",
            "IHSG broad tape masih sanggup ikut, bukan cuma energy / metals saja.",
        ]
        invalidation = [
            "Growth retak tapi inflasi masih tinggi -> risiko pindah Q3.",
            "Foreign flow mulai keluar.",
            "Property dan cyclicals drop jauh lebih dulu dari resource.",
        ]
        transition = "Jalur berikutnya biasanya ke Q3 kalau growth retak, atau kembali ke Q1 kalau inflasi mendingin duluan."
    elif current_quad == "Q3":
        focus = [
            "Utamakan exporters / hard-asset cash-flow dan defensives yang benar-benar tahan banting.",
            "Kurangi property dan domestic beta yang perlu rupiah kuat + rate relief.",
            "Kalau phase makin akhir, perhatikan defensives makin rapih atau belum sebagai sinyal Q4.",
        ]
        validation = [
            "Resource names tetap mengalahkan property dan banks.",
            "Defensives tidak ambruk.",
            "Foreign flow belum kembali agresif ke beta domestik.",
        ]
        invalidation = [
            "Banks + property mulai memimpin bareng.",
            "Rupiah membaik kuat dan BI pressure mereda.",
            "Resource tape kehilangan leadership tajam.",
        ]
        transition = "Jalur berikutnya biasanya ke Q4 jika inflasi patah, atau ke Q2 bila growth recover tapi inflasi masih tinggi."
    else:
        focus = [
            "Fokus ke defensives cash-flow, telco, staples, healthcare, dan quality high-dividend.",
            "Banks/property cukup jadi watchlist bottoming, bukan core exposure.",
            "Cari tanda early recovery, tapi jangan paksa risk-on sebelum breadth nyata.",
        ]
        validation = [
            "Defensives terus menang relatif.",
            "Growth proxies masih lemah.",
            "Property belum benar-benar break out.",
        ]
        invalidation = [
            "Banks + cyclicals mulai outperform konsisten.",
            "Property ikut hidup, bukan cuma short squeeze.",
            "Foreign flow kembali masuk ke beta domestik.",
        ]
        transition = "Jalur berikutnya biasanya ke Q1 saat growth bottom dan inflasi tetap jinak, atau ke Q3 kalau inflasi hidup lagi saat growth masih lemah."

    if phase == "Transitioning":
        focus = focus + [f"Sekarang fase transisi, jadi treat {next_quad} sebagai jalur aktif, bukan cuma background noise."]
    elif phase == "Late":
        focus = focus + [f"Sekarang fase late, jadi porsi winner utama boleh dipertahankan tapi jangan tambah agresif pada tier-3 names."]

    return focus, validation, invalidation, transition


def build_what_if(inputs: Dict[str, float], current_quad: str, next_quad: str) -> List[Dict[str, str]]:
    rows = [
        {
            "Scenario": "Rupiah membaik + foreign inflow masuk",
            "Impact": "Banks, property, dan domestic cyclicals membaik.",
            "Regime Read": "Mendorong Q1 / mempercepat keluar dari Q3-Q4.",
            "What To Watch": "Leadership pindah dari resource/defensives ke beta domestik.",
        },
        {
            "Scenario": "Commodity naik lagi tapi growth domestik lemah",
            "Impact": "Resource names masih kuat, broad IHSG tertahan.",
            "Regime Read": "Mengunci Q3 atau menarik Q4 kembali ke Q3.",
            "What To Watch": "Coal, metals, exporters menang; property tetap lemah.",
        },
        {
            "Scenario": "BI pressure turun lebih cepat dari ekspektasi",
            "Impact": "Property dan bank sensitif-rates dapat napas.",
            "Regime Read": "Positif ke Q1, netral-negatif ke Q3.",
            "What To Watch": "Apakah cyclicals ikut hidup atau cuma one-day squeeze.",
        },
        {
            "Scenario": "Inflasi turun tapi growth belum bottom",
            "Impact": "Defensives makin bagus, resource mulai kehilangan edge.",
            "Regime Read": "Mendorong Q4.",
            "What To Watch": "Staples / telco / healthcare makin rapih dibanding energy/metals.",
        },
        {
            "Scenario": "Growth recover duluan sementara inflasi tetap firm",
            "Impact": "Banks, cyclicals, industrial-linked names membaik.",
            "Regime Read": f"Mendorong {next_quad if next_quad in {'Q1', 'Q2'} else 'Q2'}.",
            "What To Watch": "Property ikut konfirmasi atau belum.",
        },
    ]

    if current_quad == "Q3":
        rows.insert(
            0,
            {
                "Scenario": "Resource leadership gagal confirm",
                "Impact": "Q3 jadi rapuh; broad tape bisa masuk fase disinflation lebih cepat.",
                "Regime Read": "Positif ke Q4, negatif ke Q3.",
                "What To Watch": "ADRO/PTBA/ITMG/ANTM kehilangan leadership sementara defensives rapih.",
            },
        )
    return rows


def signal_table(inputs: Dict[str, float], scores: Dict[str, float], probs: Dict[str, float], meters: Dict[str, float]) -> pd.DataFrame:
    rows = [
        ["Growth Momentum", inputs["growth"], score_band(inputs["growth"]), "Manual input"],
        ["Inflation Momentum", inputs["inflation"], score_band(inputs["inflation"]), "Manual input"],
        ["Rupiah Health", inputs["rupiah"], score_band(inputs["rupiah"]), "Manual input"],
        ["Foreign Flow", inputs["foreign"], score_band(inputs["foreign"]), "Manual input"],
        ["Commodity Support", inputs["commodity"], score_band(inputs["commodity"]), "Manual input"],
        ["BI Stance", inputs["bi"], score_band(inputs["bi"]), "Positive = tighter / hawkish"],
        ["Banks Leadership", inputs["banks"], score_band(inputs["banks"]), "Proxy sector pulse"],
        ["Property Leadership", inputs["property"], score_band(inputs["property"]), "Proxy sector pulse"],
        ["Defensives Leadership", inputs["defensives"], score_band(inputs["defensives"]), "Proxy sector pulse"],
        ["Resources Leadership", inputs["resources"], score_band(inputs["resources"]), "Proxy sector pulse"],
        ["Growth Axis", round(inputs["growth_axis"], 1), score_band(inputs["growth_axis"]), "Engine output"],
        ["Inflation Axis", round(inputs["inflation_axis"], 1), score_band(inputs["inflation_axis"]), "Engine output"],
        ["Q1 Score", round(scores["Q1"], 1), f"{probs['Q1']:.1f}%", "Growth up / inflation down"],
        ["Q2 Score", round(scores["Q2"], 1), f"{probs['Q2']:.1f}%", "Growth up / inflation up"],
        ["Q3 Score", round(scores["Q3"], 1), f"{probs['Q3']:.1f}%", "Growth down / inflation up"],
        ["Q4 Score", round(scores["Q4"], 1), f"{probs['Q4']:.1f}%", "Growth down / inflation down"],
        ["IHSG Risk-On Meter", round(meters["IHSG Risk-On"], 1), "0-100", "Higher = easier beta tape"],
        ["Inflation Pressure Meter", round(meters["Inflation Pressure"], 1), "0-100", "Higher = hotter macro"],
        ["IDR Stress Meter", round(meters["IDR Stress"], 1), "0-100", "Higher = more pressure"],
        ["Domestic Breadth Meter", round(meters["Domestic Breadth"], 1), "0-100", "Higher = broader domestic participation"],
        ["Resource Support Meter", round(meters["Resource Support"], 1), "0-100", "Higher = commodity helps IDX"],
        ["Defense Need Meter", round(meters["Defense Need"], 1), "0-100", "Higher = prefer defense"],
    ]
    return pd.DataFrame(rows, columns=["Signal", "Value", "State", "Notes"])


def build_driver_state(raw: Dict[str, float], driver: str) -> DriverState:
    inputs = build_driver_inputs(raw, driver)
    scores = quad_scores(inputs)
    probs = softmax_probs(scores)

    current_quad = max(probs, key=probs.get)
    current_prob = probs[current_quad]
    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    next_quad, next_prob = ranked[1]
    phase = phase_label(current_prob, next_prob)
    meters = meters_from_inputs(inputs, current_quad)
    sector_bias = sector_bias_table(inputs, current_quad, next_quad)
    focus, validation, invalidation, transition_text = action_focus(current_quad, next_quad, phase)
    what_if_rows = build_what_if(inputs, current_quad, next_quad)
    quad_info = QUAD_META[current_quad]

    return DriverState(
        driver=driver,
        current_quad=current_quad,
        current_prob=round(current_prob, 1),
        next_quad=next_quad,
        next_prob=round(next_prob, 1),
        phase=phase,
        growth_axis=round(inputs["growth_axis"], 1),
        inflation_axis=round(inputs["inflation_axis"], 1),
        regime_score=round(scores[current_quad], 1),
        risk_on_score=round(meters["IHSG Risk-On"], 1),
        meters=meters,
        sector_bias=sector_bias,
        winners=list(quad_info["winners"]),
        losers=list(quad_info["losers"]),
        focus=focus,
        validation=validation,
        invalidation=invalidation,
        what_if_rows=what_if_rows,
        transition_text=transition_text,
        signal_table=signal_table(inputs, scores, probs, meters),
    )


# =====================================================
# RENDER
# =====================================================

def render_overview(states: Dict[str, DriverState], selected_driver: str) -> None:
    state = states[selected_driver]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            display_score_html(
                "Current Regime",
                f"{state.current_quad} · {state.phase}",
                f"Confidence {state.current_prob:.1f}%",
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            display_score_html(
                "Next Likely",
                state.next_quad,
                f"Transition odds {state.next_prob:.1f}%",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            display_score_html(
                "Growth / Inflation Axis",
                f"{state.growth_axis:+.1f} / {state.inflation_axis:+.1f}",
                "IHSG-only engine",
            ),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            display_score_html(
                "Risk-On Meter",
                f"{state.risk_on_score:.1f}",
                "Higher = easier beta tape",
            ),
            unsafe_allow_html=True,
        )


def render_manual_countdowns(days_map: Dict[str, int]) -> None:
    st.markdown("### Manual update rhythm")
    cols = st.columns(len(days_map))
    for idx, (label, days) in enumerate(days_map.items()):
        with cols[idx]:
            st.markdown(
                f"""
                <div class='mini-card'>
                    <div class='small-title'>{label}</div>
                    <div class='big-number'>{int(days)}d</div>
                    <div class='muted'>Manual placeholder countdown</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_meter_cards(state: DriverState) -> None:
    st.markdown("### IHSG macro + regime meters")
    meter_items = [
        ("IHSG Risk-On", state.meters["IHSG Risk-On"], "Kemudahan tape untuk beta domestik"),
        ("Inflation Pressure", state.meters["Inflation Pressure"], "Tekanan nominal / inflation pulse"),
        ("IDR Stress", state.meters["IDR Stress"], "Semakin tinggi = semakin berat untuk broad tape"),
        ("Domestic Breadth", state.meters["Domestic Breadth"], "Semakin tinggi = leadership makin menyebar"),
        ("Resource Support", state.meters["Resource Support"], "Semakin tinggi = commodity bantu IDX"),
        ("Defense Need", state.meters["Defense Need"], "Semakin tinggi = pilih defensives"),
    ]
    cols = st.columns(3)
    for idx, (label, value, note) in enumerate(meter_items):
        with cols[idx % 3]:
            color = traffic_color(value)
            st.markdown(
                f"""
                <div class='soft-card'>
                    <div class='small-title'>{label}</div>
                    <div style='font-size:24px;font-weight:800;color:{color}'>{value:.1f}</div>
                    <div class='muted'>{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_condition_summary(state: DriverState) -> None:
    focus_html = "".join([f"<li>{item}</li>" for item in state.focus])
    winners_html = "".join([f"<span class='pill'>{item}</span>" for item in state.winners])
    losers_html = "".join([f"<span class='pill'>{item}</span>" for item in state.losers])

    st.markdown("### Condition sekarang bagusnya?")
    st.markdown(
        f"""
        <div class='main-card'>
            <div class='small-title'>Action Summary</div>
            <div class='big-number'>{state.current_quad} · {state.phase}</div>
            <div class='divider-space'></div>
            <div><span class='good'>Prioritas:</span> {winners_html}</div>
            <div style='margin-top:10px'><span class='bad'>Kurangi:</span> {losers_html}</div>
            <div style='margin-top:14px' class='muted'>Fokus taktis</div>
            <ul>{focus_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_driver_triptych(states: Dict[str, DriverState], selected_driver: str) -> None:
    st.markdown("### Current phase compare")
    cols = st.columns(3)
    for idx, driver in enumerate(states.keys()):
        state = states[driver]
        border = "2px solid #19e68c" if driver == selected_driver else "1px solid rgba(255,255,255,.10)"
        with cols[idx]:
            top_sector = state.sector_bias[0][0]
            st.markdown(
                f"""
                <div class='main-card' style='border:{border};padding:18px 18px 16px 18px;min-height:280px'>
                    <div class='small-title'>{driver}</div>
                    <div class='big-number'>{state.current_quad}</div>
                    <div>{state.phase} · confidence {state.current_prob:.1f}%</div>
                    <div class='muted' style='margin-top:8px'>Next: {state.next_quad} ({state.next_prob:.1f}%)</div>
                    <hr style='opacity:.18'>
                    <div>Growth axis: <b>{state.growth_axis:+.1f}</b></div>
                    <div>Inflation axis: <b>{state.inflation_axis:+.1f}</b></div>
                    <div>Risk-On: <b>{state.risk_on_score:.1f}</b></div>
                    <div>Top sector tilt: <b>{top_sector}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_path_compare(state: DriverState) -> None:
    st.markdown("### Path to next quad")
    c1, c2 = st.columns(2)
    with c1:
        validation_html = "".join([f"<li>{x}</li>" for x in state.validation])
        st.markdown(
            f"""
            <div class='path-card'>
                <div class='small-title'>Validation for {state.current_quad}</div>
                <div class='big-number'>{state.current_quad} stays valid</div>
                <ul>{validation_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        invalidation_html = "".join([f"<li>{x}</li>" for x in state.invalidation])
        st.markdown(
            f"""
            <div class='path-card'>
                <div class='small-title'>Invalidation / shift trigger</div>
                <div class='big-number'>Toward {state.next_quad}</div>
                <ul>{invalidation_html}</ul>
                <div class='muted' style='margin-top:8px'>{state.transition_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sector_bias(state: DriverState) -> None:
    st.markdown("#### Sector / proxy bias")
    df = pd.DataFrame(state.sector_bias, columns=["Sector", "Score", "Tilt"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("#### Proxy basket")
    cols = st.columns(3)
    for idx, (sector, tickers) in enumerate(SECTOR_PROXIES.items()):
        with cols[idx % 3]:
            pills = " ".join([f"<span class='pill'>{t}</span>" for t in tickers])
            st.markdown(
                f"""
                <div class='soft-card'>
                    <div style='font-weight:700'>{sector}</div>
                    <div style='margin-top:8px'>{pills}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_playbook(state: DriverState) -> None:
    meta = QUAD_META[state.current_quad]
    st.markdown("#### IHSG playbook")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div class='soft-card'>
                <div class='small-title'>{meta['name']}</div>
                <div class='big-number'>{meta['phase']}</div>
                <div style='margin-top:12px'><b>Fokus utama</b></div>
                <ul>{''.join([f'<li>{x}</li>' for x in state.focus])}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class='soft-card'>
                <div class='small-title'>Current vs next</div>
                <div class='big-number'>{state.current_quad} → {state.next_quad}</div>
                <div style='margin-top:12px'><b>Transition logic</b></div>
                <div>{state.transition_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_what_if_table(state: DriverState) -> None:
    st.markdown("#### What-if scenario matrix")
    df = pd.DataFrame(state.what_if_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_raw_table(state: DriverState) -> None:
    st.markdown("#### Raw signal table")
    st.dataframe(state.signal_table, use_container_width=True, hide_index=True)


# =====================================================
# MAIN
# =====================================================

def main() -> None:
    inject_css()

    st.title("IHSG Regime Dashboard")
    st.caption(
        "Struktur visual dibikin ngikut bahasa dashboard contoh lu: main-card / soft-card / mini-card, urutan overview → meter → condition summary → compare cards → path cards → bottom toggles, tapi engine-nya sekarang full IHSG-only dan manual-input."
    )

    st.sidebar.header("Settings")
    preset = st.sidebar.selectbox("Preset Scenario", list(SCENARIO_PRESETS.keys()))
    selected_driver = st.sidebar.selectbox(
        "Current Driver",
        ["Monthly (Flow + Sector Pulse)", "Blended Regime", "Quarterly Anchor"],
    )
    show_raw = st.sidebar.checkbox("Show raw signal table", value=False)
    st.sidebar.caption("No live data. Semua angka di bawah manual / preset dan bisa lu update sendiri tiap hari.")

    base = SCENARIO_PRESETS[preset].copy()
    with st.sidebar.expander("Manual overrides", expanded=True):
        growth = st.slider("Growth momentum", -100, 100, int(base["growth"]), 1)
        inflation = st.slider("Inflation momentum", -100, 100, int(base["inflation"]), 1)
        rupiah = st.slider("Rupiah health", -100, 100, int(base["rupiah"]), 1)
        foreign = st.slider("Foreign flow", -100, 100, int(base["foreign"]), 1)
        commodity = st.slider("Commodity support", -100, 100, int(base["commodity"]), 1)
        bi = st.slider("BI stance (tight + / loose -)", -100, 100, int(base["bi"]), 1)
        banks = st.slider("Banks leadership", -100, 100, int(base["banks"]), 1)
        property_ = st.slider("Property leadership", -100, 100, int(base["property"]), 1)
        defensives = st.slider("Defensives leadership", -100, 100, int(base["defensives"]), 1)
        resources = st.slider("Resources leadership", -100, 100, int(base["resources"]), 1)

    with st.sidebar.expander("Manual countdown placeholders", expanded=False):
        bi_days = st.number_input("Days to BI event", min_value=0, value=DEFAULT_CALENDAR["BI Event"])
        infl_days = st.number_input("Days to inflation update", min_value=0, value=DEFAULT_CALENDAR["Inflation Update"])
        trade_days = st.number_input("Days to trade balance", min_value=0, value=DEFAULT_CALENDAR["Trade Balance"])
        gdp_days = st.number_input("Days to GDP release", min_value=0, value=DEFAULT_CALENDAR["GDP Release"])

    raw = {
        "growth": float(growth),
        "inflation": float(inflation),
        "rupiah": float(rupiah),
        "foreign": float(foreign),
        "commodity": float(commodity),
        "bi": float(bi),
        "banks": float(banks),
        "property": float(property_),
        "defensives": float(defensives),
        "resources": float(resources),
    }

    driver_order = ["Monthly (Flow + Sector Pulse)", "Blended Regime", "Quarterly Anchor"]
    states = {driver: build_driver_state(raw, driver) for driver in driver_order}
    state = states[selected_driver]

    st.markdown(
        """
        <div class='section-note'>
            Engine ini sengaja <b>IHSG-only</b>: nggak ada world asset clutter. Yang dihitung cuma jalur yang relevan ke tape IDX: growth, inflation, rupiah, foreign flow, BI stance, commodity support, lalu proxy sektor utama (banks, property, defensives, resources).
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_overview(states, selected_driver)
    st.markdown("---")

    render_manual_countdowns(
        {
            "BI Event": int(bi_days),
            "Inflation": int(infl_days),
            "Trade Balance": int(trade_days),
            "GDP": int(gdp_days),
        }
    )
    st.markdown("---")

    render_meter_cards(state)
    st.markdown("---")

    render_condition_summary(state)
    st.markdown("---")

    render_driver_triptych(states, selected_driver)
    st.markdown("---")

    render_path_compare(state)
    st.markdown("---")

    with st.expander("Selected-driver detail", expanded=True):
        render_sector_bias(state)

    with st.expander("IHSG playbook + integrated overlays", expanded=False):
        render_playbook(state)

    with st.expander("What-if / correlation matrix", expanded=False):
        render_what_if_table(state)

    if show_raw:
        with st.expander("Raw signal table", expanded=False):
            render_raw_table(state)

    st.markdown("---")
    st.caption(
        "Catatan: ini versi tanpa live data. Jadi dashboard ini cocok buat diisi manual tiap hari atau dijadikan template sebelum nanti gue sambungin ke source live."
    )


if __name__ == "__main__":
    main()
