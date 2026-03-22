import html
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
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
    "DGS2": "DGS2",
    "DGS5": "DGS5",
    "DGS10": "DGS10",
    "DGS30": "DGS30",
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

DEFAULT_NEWS_QUERY = "Iran war oil Strait of Hormuz private credit treasury auction rollover"

WHAT_IF_SCENARIO_MATRIX = [
    ("Oil up + USD up + 2Y up", "Stagflationary / geopolitical shock", "Direct oil, pure upstream, integrated majors, selective IHSG resource names", "Broad EM, alt beta, weak duration growth"),
    ("Oil up + gold down + 2Y up", "Front-end hawkish shock dominates safe-haven bid", "Oil chain first; treat gold weakness as rates/dollar pressure, not automatic negation of Quad 3", "Do not force broad-beta risk-on just because gold is soft"),
    ("Oil spikes but tanker / second-order names do not confirm", "Chain incomplete", "Stick to first-order commodity winners", "Avoid assuming every spillover proxy must immediately follow"),
    ("USD down + 2Y down + real yields down", "Risk-on relief / duration relief", "Nasdaq, BTC, selective EM, quality cyclicals", "Oil-shock winners usually lose relative edge"),
    ("Dollar down but EM still weak", "Local / credit / growth problem still dominates", "Prefer selective exporters or US quality over broad EM", "Avoid treating EM as automatic winner"),
    ("Real yields down but crypto breadth stays weak", "Quality-led relief only", "BTC first, then ETH, then alt beta only if breadth expands", "Avoid forcing alt rotation too early"),
    ("30Y up faster than 2Y", "Long-end term-premium pressure", "Hard assets / pricing-power / selective value", "Long-duration valuation-sensitive names"),
    ("Credit worsens while oil stays high", "War + funding stress", "Direct commodity proxies and safest defensives only", "Broad beta, weak EM, low-quality junk rallies"),
]

DIVERGENCE_RULES = [
    ("Oil naik tapi gold turun", "Bukan otomatis kontradiksi quad. Sering artinya dollar dan yields naik lebih dominan daripada safe-haven bid ke gold."),
    ("Oil naik tapi broad EM tidak ikut", "Biasanya dollar terlalu kuat atau flow ke EM masih jelek. Fokus ke selective exporters, bukan broad EM."),
    ("Oil naik tapi tanker tidak ikut", "Harga crude saja tidak cukup. Freight, route stress, dan shipping supply juga harus mendukung."),
    ("Coal / metals naik tapi dry bulk tidak ikut", "Supply shock / price shock bisa terjadi tanpa volume shipping yang benar-benar pulih."),
    ("Real yields turun tapi crypto belum ikut", "Dollar, liquidity, atau breadth crypto belum confirm. Biasanya BTC duluan, alt paling belakang."),
    ("Dollar turun tapi IHSG / EM tetap lemah", "Masih bisa ada masalah lokal, growth global lemah, atau credit belum membaik."),
]

CORRELATION_TRANSMISSION_PRIORS = [
    ("Oil -> WTI/Brent", "Very High", "Underlying paling murni untuk war/oil shock."),
    ("Oil -> pure upstream oil", "Very High", "Producer upstream paling cepat menyerap harga crude."),
    ("Oil -> integrated majors", "High", "Masih direct, tapi lebih defensif dari upstream murni."),
    ("Oil -> IHSG resource proxies", "Medium-High", "Nama lokal berbasis resource bisa ikut, tapi kena drag FX/flow lokal."),
    ("Oil -> broad EM", "Low / conditional", "Oil naik tidak otomatis bikin broad EM bullish."),
    ("2Y naik keras -> front-end stress", "Very High", "Paling cepat menekan duration, crypto beta, dan broad risk appetite."),
    ("30Y naik keras -> long-end pressure", "High", "Valuation duration-heavy dan long bonds biasanya paling sensitif."),
    ("Real yields turun -> Nasdaq", "Very High", "Duration asset paling sensitif ke relief real yields."),
    ("Real yields turun -> BTC", "High", "BTC biasanya crypto leader saat conditions membaik."),
    ("Dollar naik -> broad EM / IDR", "Very High negative", "Salah satu jalur transmisi terpenting ke EM dan IHSG."),
]

REGIME_SWITCH_ARCHETYPES = [
    ("Quad 4 -> Quad 3 -> Quad 1", "Shock inflasi mereda, lalu disinflation stabil, lalu duration/risk-on menang."),
    ("Quad 2 overheating -> Quad 3", "Growth masih oke, tapi inflasi terlalu keras dan broad beta mulai capek."),
    ("Quad 3 relief -> false Quad 1 -> back to Quad 3", "Yield relief ada, tapi breadth/credit tidak confirm sehingga risk-on gagal lanjut."),
    ("War / oil shock -> selective commodity leadership -> failure", "Direct commodity proxy menang, tapi second-order / broad beta tidak confirm lalu move kehilangan tenaga."),
]

FALSE_RECOVERY_MAP = [
    ("2Y belum relief", "Risk-on bounce sering belum tahan lama kalau front-end stress masih keras."),
    ("Dollar tetap kuat", "EM / crypto / small caps sering gagal confirm walau ada pantulan."),
    ("Credit tetap jelek", "Broad beta bisa memantul, tapi kualitasnya rendah dan rawan gagal lanjut."),
    ("Breadth tidak ikut", "Kalau hanya old winners atau short covering, itu sering dead-cat bounce."),
    ("Second-order proxies belum ikut", "Kalau hanya first-order winner hidup, regime belum tentu sehat menyebar."),
]

CRASH_TYPES = [
    ("Liquidity shock", "Semua dijual karena market cari cash. Correlation naik cepat."),
    ("Credit shock", "Spread melebar, trust turun, balance-sheet stress jadi pusat."),
    ("Growth collapse", "Commodity dan cyclicals bisa turun bareng karena demand fear."),
    ("Inflation / commodity shock", "Hard assets bisa naik duluan, tapi broad beta sering rapuh."),
    ("Policy shock", "2Y / front-end repricing bikin duration dan beta sesak."),
    ("Geopolitical shock", "Oil, tanker, direct commodity proxies bisa menang, broad EM belum tentu."),
]

CRASH_RECOVERY_ORDER = [
    ("Deflationary / liquidity crash", "Bonds -> defensives -> Nasdaq -> BTC -> EM -> small caps -> alt beta"),
    ("Inflation / oil shock", "Direct commodity -> pricing-power defensives -> selective exporters -> broad beta belakangan"),
    ("Credit shock", "Treasuries / safest duration -> gold / defensives -> quality equities -> broad beta jauh belakangan"),
]

# User-preferred quad numbering from the screenshots:
# Q1 = Growth Up / Inflation Down
# Q2 = Growth Up / Inflation Up
# Q3 = Growth Down / Inflation Up
# Q4 = Growth Down / Inflation Down
QUAD_META = {
    "Q1": {
        "name": "Quad 1",
        "phase": "Goldilocks",
        "logic": "Growth Up / Inflation Down",
        "winners": "equities, credit, commodities, FX; tech, consumer discretionary, communication services, industrials, materials, REITs",
    },
    "Q2": {
        "name": "Quad 2",
        "phase": "Reflation",
        "logic": "Growth Up / Inflation Up",
        "winners": "commodities, equities, credit, FX; tech, industrials, financials, energy, consumer discretionary",
    },
    "Q3": {
        "name": "Quad 3",
        "phase": "Stagflation",
        "logic": "Growth Down / Inflation Up",
        "winners": "gold, commodities, fixed income; utilities, energy, REITs, tech, consumer staples, health care",
    },
    "Q4": {
        "name": "Quad 4",
        "phase": "Deflation",
        "logic": "Growth Down / Inflation Down",
        "winners": "fixed income, gold, USD; consumer staples, health care, utilities",
    },
}

PHASE_GUIDE = {
    "Q1": {
        "meaning": {
            "Macro": [
                "Growth is accelerating while inflation is decelerating.",
                "This is the Goldilocks regime in the Hedgeye framework.",
            ],
            "Market": [
                "Historically favorable for risk assets, especially equities and credit.",
                "Fixed income and the USD tend to be the weaker major asset-class buckets in the playbook.",
            ],
            "Positioning": [
                "Favor offense through the actual Quad 1 winners rather than generic beta everywhere.",
                "Hedgeye's backtests tilt toward high beta, momentum, leverage, secular growth, and mid caps—not toward bond proxies or deep defensives.",
            ],
        },
        "winners": {
            "Best Asset Classes": {
                "Historically Favored": ["equities", "credit", "commodities", "FX"],
            },
            "Best Equity Sectors": {
                "Historically Favored": [
                    "tech",
                    "consumer discretionary",
                    "communication services",
                    "industrials",
                    "materials",
                    "REITs",
                ],
            },
            "Best Equity Style Factors": {
                "Historically Favored": [
                    "high beta",
                    "momentum",
                    "leverage",
                    "secular growth",
                    "mid caps",
                ],
            },
            "Best Fixed Income Sectors": {
                "Historically Favored": [
                    "BDCs",
                    "convertibles",
                    "high-yield credit",
                    "EM dollar debt",
                    "leveraged loans",
                ],
            },
            "FX Lens": {
                "Prefer": [
                    "pro-cyclical FX and growth-sensitive majors",
                    "AUD, CAD, NOK when growth breadth and credit remain healthy",
                    "selected EM FX only when dollar trend is soft and credit stress is contained",
                ],
            },
            "Emerging Markets Lens": {
                "Prefer Selectively": [
                    "country selection over generic EM beta when global demand is broadening",
                    "EM exporters / reform stories only when USD is not the dominant macro trend",
                    "broad EM works best when credit is calm and the dollar is softening, not tightening",
                ],
            },
            "Crypto / Digital Assets Lens": {
                "Tactical, Not Core": [
                    "BTC and liquid crypto beta can participate when breadth, liquidity, and credit all improve together",
                    "use signal confirmation rather than assuming all alts are automatic Quad 1 winners",
                    "higher-quality, liquid beta is cleaner than illiquid alt beta",
                ],
            },
        },
        "losers": {
            "Worst Asset Classes": {
                "Historically Weak": ["fixed income", "USD"],
            },
            "Worst Equity Sectors": {
                "Historically Weak": ["utilities", "consumer staples", "health care"],
            },
            "Worst Equity Style Factors": {
                "Historically Weak": [
                    "low beta",
                    "defensives",
                    "value",
                    "dividend yield",
                    "small caps",
                ],
            },
            "Worst Fixed Income Sectors": {
                "Historically Weak": [
                    "TIPS",
                    "short-duration Treasuries",
                    "MBS",
                    "Treasury belly",
                    "long bond",
                ],
            },
            "FX Lens": {
                "Avoid / Weak": [
                    "USD and defensive funding currencies when reflation breadth is broad",
                    "pure safe-haven FX beta over cyclical FX",
                ],
            },
            "Emerging Markets Lens": {
                "Avoid / Weak": [
                    "fragile EM that need a weak dollar and easy credit to work",
                    "broad EM beta if credit stress rises before growth breadth truly improves",
                ],
            },
            "Crypto / Digital Assets Lens": {
                "Avoid / Weak": [
                    "illiquid alts and weak-balance-sheet miners when the rally is too narrow",
                    "treating all crypto beta as equal-quality risk-on exposure",
                ],
            },
        },
    },
    "Q2": {
        "meaning": {
            "Macro": [
                "Growth and inflation are both accelerating.",
                "This is the Reflation regime and, in Hedgeye's public materials, often the most bullish environment for stocks.",
            ],
            "Market": [
                "Commodities, equities, credit, and FX are the strongest major asset-class buckets in the playbook.",
                "Fixed income and the USD usually lag while nominal-growth-sensitive areas lead.",
            ],
            "Positioning": [
                "Lean into cyclical offense and nominal-growth beta.",
                "The historical winners tilt toward secular growth, high beta, small caps, cyclical growth, and momentum.",
            ],
        },
        "winners": {
            "Best Asset Classes": {
                "Historically Favored": ["commodities", "equities", "credit", "FX"],
            },
            "Best Equity Sectors": {
                "Historically Favored": [
                    "tech",
                    "industrials",
                    "financials",
                    "energy",
                    "consumer discretionary",
                ],
            },
            "Best Equity Style Factors": {
                "Historically Favored": [
                    "secular growth",
                    "high beta",
                    "small caps",
                    "cyclical growth",
                    "momentum",
                ],
            },
            "Best Fixed Income Sectors": {
                "Historically Favored": [
                    "convertibles",
                    "BDCs",
                    "preferreds",
                    "leveraged loans",
                    "high-yield credit",
                ],
            },
            "FX Lens": {
                "Prefer": [
                    "commodity and cyclical FX rather than defensive funding currencies",
                    "AUD, CAD, NOK when reflation is broad and commodities confirm",
                    "selected exporter FX only if dollar trend is not dominant",
                ],
            },
            "Emerging Markets Lens": {
                "Prefer Selectively": [
                    "commodity exporters and nominal-growth beneficiaries rather than generic EM beta",
                    "country ETFs tied to energy, industrials, or reflation capex can outperform broad EM",
                    "broad EM only if USD is not tightening financial conditions",
                ],
            },
            "Crypto / Digital Assets Lens": {
                "Tactical, Not Core": [
                    "this is usually the best macro backdrop for BTC and higher-beta liquid crypto if breadth and liquidity confirm",
                    "alts can work, but only when volume, breadth, and funding are broad rather than speculative and narrow",
                    "infrastructure / exchange / blockchain beta is cleaner than low-quality alt beta",
                ],
            },
        },
        "losers": {
            "Worst Asset Classes": {
                "Historically Weak": ["fixed income", "USD"],
            },
            "Worst Equity Sectors": {
                "Historically Weak": [
                    "utilities",
                    "communication services",
                    "consumer staples",
                    "REITs",
                    "health care",
                ],
            },
            "Worst Equity Style Factors": {
                "Historically Weak": [
                    "low beta",
                    "dividend yield",
                    "value",
                    "defensives",
                    "size",
                ],
            },
            "Worst Fixed Income Sectors": {
                "Historically Weak": [
                    "long bond",
                    "Treasury belly",
                    "munis",
                    "MBS",
                    "IG credit",
                ],
            },
            "FX Lens": {
                "Avoid / Weak": [
                    "USD shorts when rates / credit are re-tightening and reflation breadth is narrow",
                    "generic broad EM beta without commodity confirmation",
                ],
            },
            "Emerging Markets Lens": {
                "Avoid / Weak": [
                    "importer-heavy EM or weak-balance-sheet countries if oil and yields rise too fast",
                    "generic broad-EM beta when the dollar is reasserting leadership",
                ],
            },
            "Crypto / Digital Assets Lens": {
                "Avoid / Weak": [
                    "chasing low-quality alts after narrow vertical squeezes",
                    "assuming every crypto rally is durable if rates and liquidity are working against it",
                ],
            },
        },
    },
    "Q3": {
        "meaning": {
            "Macro": [
                "Growth is decelerating while inflation is accelerating.",
                "This is the Stagflation regime in the Hedgeye framework.",
            ],
            "Market": [
                "The historical asset-class winners are gold, commodities, and fixed income—not broad credit.",
                "Inside equities, leadership narrows toward utilities, energy, REITs, tech, staples, and health care while many cyclicals lag.",
            ],
            "Positioning": [
                "Do not treat broad emerging markets or generic small-cap beta as default Quad 3 winners.",
                "The historical style-factor winners tilt toward secular growth, momentum, mid caps, low beta, and quality, while financials, discretionary, industrials, and materials are weaker.",
            ],
        },
        "winners": {
            "Best Asset Classes": {
                "Historically Favored": ["gold", "commodities", "fixed income"],
            },
            "Best Equity Sectors": {
                "Historically Favored": [
                    "utilities",
                    "energy",
                    "REITs",
                    "tech",
                    "consumer staples",
                    "health care",
                ],
            },
            "Best Equity Style Factors": {
                "Historically Favored": [
                    "secular growth",
                    "momentum",
                    "mid caps",
                    "low beta",
                    "quality",
                ],
            },
            "Best Fixed Income Sectors": {
                "Historically Favored": [
                    "munis",
                    "EM dollar debt",
                    "long bond",
                    "TIPS",
                    "Treasury belly",
                ],
            },
            "FX Lens": {
                "Use Selectively": [
                    "USD is often the clean tactical expression when oil, inflation, and yields are all pressing higher",
                    "major-pair expressions are cleaner than broad EM FX beta in stagflationary episodes",
                    "commodity-exporter FX can work tactically, but only as exceptions rather than the default broad-EM playbook",
                ],
            },
            "Emerging Markets Lens": {
                "Use Selectively": [
                    "country dispersion dominates: selective energy/resource exporters or idiosyncratic reform/stimulus stories only",
                    "broad EM is not a default Quad 3 winner in a USD-up stagflationary episode",
                    "country-level longs are cleaner than EEM-style broad beta",
                ],
            },
            "Crypto / Digital Assets Lens": {
                "Tactical, Not Core": [
                    "BTC is not a core historical Quad 3 winner in Hedgeye's public playbook",
                    "at best, treat BTC as a tactical signal-driven trade rather than a regime-led allocation",
                    "alts are usually much weaker than gold, commodities, or duration in this environment",
                ],
            },
        },
        "losers": {
            "Worst Asset Classes": {
                "Historically Weak": ["credit"],
            },
            "Worst Equity Sectors": {
                "Historically Weak": [
                    "communication services",
                    "financials",
                    "consumer discretionary",
                    "industrials",
                    "materials",
                ],
            },
            "Worst Equity Style Factors": {
                "Historically Weak": [
                    "small caps",
                    "dividend yield",
                    "value",
                    "defensives",
                    "size",
                ],
            },
            "Worst Fixed Income Sectors": {
                "Historically Weak": [
                    "BDCs",
                    "preferreds",
                    "convertibles",
                    "leveraged loans",
                    "high-yield credit",
                ],
            },
            "FX Lens": {
                "Avoid / Weak": [
                    "broad EM FX beta",
                    "EUR, GBP, and high-beta cyclical FX during USD-up episodes",
                    "JPY when USD rates impulse dominates and trend breaks lower",
                ],
            },
            "Emerging Markets Lens": {
                "Avoid / Weak": [
                    "broad EM equities / FX, especially importers or countries reliant on easy dollar liquidity",
                    "fragile balance-sheet EM and generic EEM beta during USD-up, oil-up stress",
                ],
            },
            "Crypto / Digital Assets Lens": {
                "Avoid / Weak": [
                    "alts, miners, and blockchain-beta equities when crash / stagflation risk is high",
                    "treating BTC like a guaranteed inflation hedge when the dollar and yields are dominating the tape",
                ],
            },
        },
    },
    "Q4": {
        "meaning": {
            "Macro": [
                "Growth and inflation are both decelerating.",
                "This is the Deflation regime in the Hedgeye framework.",
            ],
            "Market": [
                "Fixed income, gold, and USD are the main asset-class winners in the historical playbook.",
                "Commodities, equities, credit, and FX are the broad losers while defensive sectors and quality factors outperform.",
            ],
            "Positioning": [
                "Favor defense, duration, and quality over aggressive beta.",
                "The strongest equity style factors are low beta, dividend yield, quality, defensives, and value; the weakest are high beta, momentum, leverage, secular growth, and cyclical growth.",
            ],
        },
        "winners": {
            "Best Asset Classes": {
                "Historically Favored": ["fixed income", "gold", "USD"],
            },
            "Best Equity Sectors": {
                "Historically Favored": ["consumer staples", "health care", "utilities"],
            },
            "Best Equity Style Factors": {
                "Historically Favored": [
                    "low beta",
                    "dividend yield",
                    "quality",
                    "defensives",
                    "value",
                ],
            },
            "Best Fixed Income Sectors": {
                "Historically Favored": [
                    "long bond",
                    "Treasury belly",
                    "IG credit",
                    "munis",
                    "MBS",
                ],
            },
            "FX Lens": {
                "Prefer": [
                    "USD and defensive reserve FX over cyclical beta",
                    "long USD expressions rather than commodity / growth-sensitive FX",
                ],
            },
            "Emerging Markets Lens": {
                "Prefer Selectively": [
                    "very selective country stories only; broad EM is usually not the right default expression",
                    "reserve-quality or reform-driven idiosyncratic winners are cleaner than broad EM beta",
                ],
            },
            "Crypto / Digital Assets Lens": {
                "Tactical, Not Core": [
                    "wait for bottoming and signal confirmation rather than assuming crypto beta should lead",
                    "capital preservation matters more than forcing digital-asset risk into a deflationary slowdown",
                ],
            },
        },
        "losers": {
            "Worst Asset Classes": {
                "Historically Weak": ["commodities", "equities", "credit", "FX"],
            },
            "Worst Equity Sectors": {
                "Historically Weak": [
                    "energy",
                    "tech",
                    "financials",
                    "industrials",
                    "consumer discretionary",
                ],
            },
            "Worst Equity Style Factors": {
                "Historically Weak": [
                    "high beta",
                    "momentum",
                    "leverage",
                    "secular growth",
                    "cyclical growth",
                ],
            },
            "Worst Fixed Income Sectors": {
                "Historically Weak": [
                    "preferreds",
                    "EM local currency",
                    "BDCs",
                    "leveraged loans",
                    "TIPS",
                ],
            },
            "FX Lens": {
                "Avoid / Weak": [
                    "commodity FX and high-beta growth FX",
                    "broad EM FX beta when global growth is decelerating",
                ],
            },
            "Emerging Markets Lens": {
                "Avoid / Weak": [
                    "broad EM beta, especially cyclical importers and countries needing soft global growth conditions",
                    "generic EM risk when dollar liquidity is tightening or global demand is weakening",
                ],
            },
            "Crypto / Digital Assets Lens": {
                "Avoid / Weak": [
                    "BTC and especially alts when deflationary slowdown pressure is still dominating",
                    "miners and levered crypto-equity beta before the macro and risk backdrop actually bottoms",
                ],
            },
        },
    },
}

PLAYBOOK_NEXT = {
    "Q1": ["Q2", "Q4"],
    "Q2": ["Q1", "Q3"],
    "Q3": ["Q4", "Q2"],
    "Q4": ["Q1", "Q3"],
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

OFFICIAL_WINNER_BUCKETS = [
    "Best Asset Classes",
    "Best Equity Sectors",
    "Best Equity Style Factors",
    "Best Fixed Income Sectors",
]

OFFICIAL_LOSER_BUCKETS = [
    "Worst Asset Classes",
    "Worst Equity Sectors",
    "Worst Equity Style Factors",
    "Worst Fixed Income Sectors",
]

OVERLAY_BUCKETS = [
    "FX Lens",
    "Emerging Markets Lens",
    "Crypto / Digital Assets Lens",
]

CURRENT_PHASE_TEXT = {
    "Q1": "Growth is holding up while inflation cools. Focus on whether inflation re-heats or growth loses momentum first.",
    "Q2": "Nominal growth is strong. Focus on whether inflation cools first or growth rolls first.",
    "Q3": "Growth is weak while inflation stays firm. Focus on whether inflation finally breaks or growth bottoms first.",
    "Q4": "Growth and inflation are both cooling. Focus on whether growth bottoms into recovery or inflation re-heats into stagflation.",
}

EM_IHSG_MATRIX = {
    "Q1": {
        "Tier 1 — strongest direct sensitivity": [
            "IHSG large banks / domestic liquidity leaders: BBCA, BBRI, BMRI, BBNI",
            "quality domestic-demand / retail proxies: AMRT, ACES, MAPI",
            "high-quality Indonesia beta once USD stress is easing: EIDO / broad liquid IHSG",
        ],
        "Tier 2 — strong but needs confirmation": [
            "property beta after rates stop backing up: PWON, SMRA, CTRA",
            "industrials / cyclicals with clean balance sheets: ASII, UNTR, AKRA",
        ],
        "Tier 3 — spillover / tactical only": [
            "select reform / exporter EM beta only after dollar and credit improve together",
            "second-line domestic beta once banks and liquid leaders already confirm",
        ],
        "Weak / avoid": [
            "fragile broad EM beta if USD refuses to soften",
            "pure defensives as primary longs if growth breadth is broadening",
        ],
    },
    "Q2": {
        "Tier 1 — strongest direct sensitivity": [
            "commodity / energy / materials proxies: ADRO, ITMG, PGAS, ANTM, MDKA",
            "nominal-growth beneficiaries and reflation cyclicals: AKRA, ASII, UNTR",
            "selected broad EM / exporter beta if USD is not the dominant tightening force",
        ],
        "Tier 2 — strong but secondary": [
            "banks that monetize stronger nominal growth: BBRI, BMRI, BBCA, BBNI",
            "shipping / logistics / capex spillover once the core commodity tape is already working",
        ],
        "Tier 3 — spillover / tactical only": [
            "property beta only after rates stop bear-steepening too hard: PWON, SMRA, CTRA",
            "broad IHSG beta after energy / materials / banks already lead",
        ],
        "Weak / avoid": [
            "pure defensives and bond proxies as primary longs",
            "fragile EM importers during an oil-up, inflation-sensitive reflation move",
        ],
    },
    "Q3": {
        "Tier 1 — strongest direct sensitivity": [
            "Indonesia exporter / hard-asset / energy complex: ADRO, ITMG, PGAS, ANTM, MDKA",
            "broadest direct macro expressions are USD strength and broad-EM weakness, not generic EM beta",
            "inside IHSG, commodity cash-flow names usually feel the regime first and most directly",
        ],
        "Tier 2 — strong but more defensive": [
            "defensive cash-flow / domestic defense: TLKM, ICBP, INDF, KLBF",
            "selected idiosyncratic exporter / reform stories only after the core USD / commodity story is already in place",
        ],
        "Tier 3 — spillover / tactical only": [
            "selective tactical bounces in broad EM or IHSG beta only when short-term risk appetite improves",
            "country-specific exceptions, not generic EEM-style broad exposure",
        ],
        "Weak / avoid": [
            "broad EM beta and vulnerable EM importers",
            "IHSG domestic-beta cyclicals and rates-sensitive property: BBRI, BMRI, BBNI, PWON, SMRA, CTRA",
        ],
    },
    "Q4": {
        "Tier 1 — strongest direct sensitivity": [
            "IHSG defensives / cash-flow resilience: TLKM, ICBP, INDF, KLBF",
            "quality balance sheets and lower-beta large caps before the next recovery is confirmed",
            "USD defense over broad EM beta",
        ],
        "Tier 2 — strong but selective": [
            "selected gold / defensive commodity exposures only if duration is helping the tape: ANTM, MDKA",
            "cash-generative franchises that can hold margins in a slowdown",
        ],
        "Tier 3 — spillover / watchlist only": [
            "banks and domestic cyclicals as early-recovery watchlist, not core longs yet",
            "broad EM only after a real bottoming process begins",
        ],
        "Weak / avoid": [
            "commodity beta that still needs hot nominal growth",
            "property, lower-quality cyclicals, and fragile EM balance-sheet exposure",
        ],
    },
}

CRYPTO_MATRIX = {
    "Q1": {
        "Tier 1 — strongest / highest-quality": [
            "BTC first",
            "ETH as secondary quality beta once breadth and liquidity confirm",
        ],
        "Tier 2 — strong but needs confirmation": [
            "SOL and liquid large-cap beta only after BTC / ETH leadership is already broadening",
            "blockchain / exchange infrastructure as cleaner equity expressions than low-quality alts",
        ],
        "Tier 3 — spillover / tactical only": [
            "select liquid alts only after the major-beta complex is already working",
        ],
        "Weak / avoid": [
            "illiquid alts, memecoins, and levered miner beta if breadth is still narrow",
        ],
    },
    "Q2": {
        "Tier 1 — strongest direct sensitivity": [
            "BTC, ETH, and SOL / liquid large-cap beta",
            "the cleanest crypto backdrop for broad beta only if liquidity and breadth actually confirm",
        ],
        "Tier 2 — strong but secondary": [
            "blockchain / exchange / miner beta as higher-volatility expressions",
            "selected liquid L1 / L2 / infra names after BTC leadership is already healthy",
        ],
        "Tier 3 — spillover / tactical only": [
            "broader alt beta after majors, liquidity, and funding all confirm",
        ],
        "Weak / avoid": [
            "low-quality illiquid alt beta treated as if it were the same as BTC quality",
        ],
    },
    "Q3": {
        "Tier 1 — strongest relative quality": [
            "cash / stables first, then BTC as the only cleaner tactical bounce candidate",
            "if risk appetite improves briefly, BTC is still cleaner than alt beta",
        ],
        "Tier 2 — tactical only": [
            "ETH / SOL only after BTC and broader risk appetite both improve",
            "blockchain / miner beta only for tactical trades, not core regime-led exposure",
        ],
        "Tier 3 — spillover / fragile": [
            "selected liquid alts for quick tactical rebounds only",
        ],
        "Weak / avoid": [
            "broad alt beta, memecoins, miners, and levered crypto-equity exposure as core holdings",
        ],
    },
    "Q4": {
        "Tier 1 — strongest relative quality": [
            "cash / stables first, then BTC watchlist logic only after bottoming evidence appears",
        ],
        "Tier 2 — tactical only": [
            "ETH and liquid majors only after the macro and liquidity backdrop truly turns",
        ],
        "Tier 3 — spillover / watchlist only": [
            "miners, exchanges, and alt beta as later-cycle recovery expressions rather than first longs",
        ],
        "Weak / avoid": [
            "broad alt beta and low-quality speculative crypto risk during an ongoing deflationary slowdown",
        ],
    },
}

PROXY_IMPACT_MATRIX = {
    "Q1": {
        "Strongest direct proxies": [
            "growth RoC up, inflation RoC down",
            "banks / cyclicals / high beta / momentum leadership",
            "soft-to-stable 2Y / 5Y and calmer USD",
        ],
        "Second-order confirmations": [
            "better breadth, cleaner credit, selective EM / IHSG beta participation",
            "BTC first, then broader crypto only if breadth confirms",
        ],
        "Spillover / weaker confirmations": [
            "property and lower-liquidity beta after the liquid leaders already confirm",
        ],
    },
    "Q2": {
        "Strongest direct proxies": [
            "commodities and nominal-growth cyclicals",
            "firm 2Y / 5Y / 10Y, still-orderly credit, high-beta leadership",
            "energy / materials / reflation equity leadership",
        ],
        "Second-order confirmations": [
            "banks, small caps, exporter EM / IHSG beta, liquid alt beta",
        ],
        "Spillover / weaker confirmations": [
            "property / late-cycle domestic beta after the core reflation tape is already working",
        ],
    },
    "Q3": {
        "Strongest direct proxies": [
            "oil / hard-asset impulse",
            "USD strength and front-end / belly repricing via 2Y and 5Y",
            "broad EM weakness rather than broad EM strength",
        ],
        "Second-order confirmations": [
            "defensive cash-flow equities, selected energy / gold / exporter names, mixed 10Y / 30Y behavior",
            "BTC tactical only after risk appetite improves; alt beta lags",
        ],
        "Spillover / weaker confirmations": [
            "short-lived tactical bounces in broad beta or alts that do not change the underlying regime",
        ],
    },
    "Q4": {
        "Strongest direct proxies": [
            "falling 2Y / 5Y / 10Y / 30Y and duration leadership",
            "defensives, USD, and slowdown winners",
        ],
        "Second-order confirmations": [
            "quality large caps, selected defensive EM / IHSG cash-flow names",
        ],
        "Spillover / weaker confirmations": [
            "early-recovery bounces in banks / crypto / cyclicals that need far more confirmation",
        ],
    },
}

QUAD_LONG_SHORT_LADDER = {
    "Q1": {
        "Winner Ladder (most direct → spillover)": [
            "Tier 1 — direct regime winners: tech, consumer discretionary, communication services, industrials, materials, REITs",
            "Tier 2 — strong secondary: high beta, momentum, leverage, secular growth, mid caps, credit beta",
            "Tier 3 — spillover / tactical: selected EM / IHSG banks, exporters, and then liquid crypto quality after breadth confirms",
        ],
        "Loser Ladder (best shorts / weakest first)": [
            "Tier 1 — most direct laggards: utilities, staples, health care, pure duration, USD defense",
            "Tier 2 — secondary laggards: low beta, deep defensives, dividend yield, bond proxies",
            "Tier 3 — weaker spillover shorts: fragile EM or illiquid alt beta only if the rally stays narrow and selective",
        ],
    },
    "Q2": {
        "Winner Ladder (most direct → spillover)": [
            "Tier 1 — direct regime winners: commodities, energy, materials, industrials, financials, high-beta cyclicals",
            "Tier 2 — strong secondary: banks, small caps, exporter EM / IHSG beta, leveraged credit, liquid crypto beta",
            "Tier 3 — spillover / tactical: property, logistics, and broader domestic cyclicals after the core reflation tape already leads",
        ],
        "Loser Ladder (best shorts / weakest first)": [
            "Tier 1 — most direct laggards: long duration, utilities, staples, health care, pure USD defense",
            "Tier 2 — secondary laggards: bond proxies, low beta, deep defensives, quality if rates back up too hard",
            "Tier 3 — weaker spillover shorts: fragile importers and lower-quality growth if inflation pressure keeps broadening",
        ],
    },
    "Q3": {
        "Winner Ladder (most direct → spillover)": [
            "Tier 1 — direct regime winners: oil / hard assets, USD strength, selected energy exporters, utilities / staples / health care cash-flow defense",
            "Tier 2 — strong secondary: gold if inflation-hedge demand catches up, REITs / duration only when long-end growth fear is helping, selected exporter IHSG names",
            "Tier 3 — spillover / tactical: BTC tactical only after risk appetite improves; selective country-specific EM exceptions, not broad EM beta",
        ],
        "Loser Ladder (best shorts / weakest first)": [
            "Tier 1 — most direct laggards: broad EM, EM FX, domestic-beta cyclicals, financials, consumer discretionary, industrials, materials",
            "Tier 2 — secondary laggards: IHSG banks / property / lower-quality cyclicals, broad alt beta, miners, small-cap beta",
            "Tier 3 — weaker spillover shorts: lower-quality semis / tech beta only when yields and the dollar keep tightening the tape",
        ],
    },
    "Q4": {
        "Winner Ladder (most direct → spillover)": [
            "Tier 1 — direct regime winners: duration, Treasuries, USD, gold, staples, health care, utilities",
            "Tier 2 — strong secondary: high quality cash-flow large caps, selected defensive IHSG names, quality growth if yields keep falling",
            "Tier 3 — spillover / watchlist only: BTC first, then banks / cyclicals only after a real bottoming process begins",
        ],
        "Loser Ladder (best shorts / weakest first)": [
            "Tier 1 — most direct laggards: commodities that need hot nominal growth, lower-quality cyclicals, property, broad EM beta, alt beta",
            "Tier 2 — secondary laggards: financials, industrials, materials, consumer discretionary, lower-quality credit",
            "Tier 3 — weaker spillover shorts: exporter beta if duration is falling because global demand is rolling over, not because inflation is reaccelerating",
        ],
    },
}

STAGE_ROTATION_GUIDE = {
    "Q1": {
        "Early": {
            "Winner Ladder": [
                "Direct longs: tech, discretionary, communication services, industrials",
                "Secondary: high beta, momentum, credit beta, selective EM / IHSG banks",
                "Spillover: BTC first, then broader liquid crypto only if breadth is widening",
            ],
            "Loser Ladder": [
                "Best shorts: utilities, staples, health care, long-duration defensives",
                "Secondary shorts: USD defense and low-beta bond proxies",
            ],
            "If the phase weakens, rotate toward": [
                "Q2 if inflation re-accelerates → add energy, materials, financials, commodity beta",
                "Q4 if growth rolls first → add duration, defensives, USD, quality",
            ],
        },
        "Mid": {
            "Winner Ladder": [
                "Keep core longs in tech / cyclicals but reduce weakest beta",
                "Prefer quality growth over lower-quality speculative beta",
                "EM / crypto only if breadth and credit still confirm",
            ],
            "Loser Ladder": [
                "Utilities / staples still lag, but the cleaner shorts shift toward late movers and over-owned beta",
                "Watch for rate-sensitive losers if yields start backing up",
            ],
            "If the phase weakens, rotate toward": [
                "Toward Q2 if oil / breakevens / 2Y / 5Y all firm together",
                "Toward Q4 if breadth narrows and duration starts to outperform",
            ],
        },
        "Late": {
            "Winner Ladder": [
                "Take profits in crowded high-beta winners and keep only liquid leadership",
                "Shrink lower-quality EM / crypto spillover first",
            ],
            "Loser Ladder": [
                "Best new shorts become lower-quality cyclicals if breadth is cracking",
                "Defensives stop being clean shorts once rotation begins",
            ],
            "If the phase weakens, rotate toward": [
                "Toward Q2 when inflation-led reflation takes over",
                "Toward Q4 when growth cracks and duration / defense take leadership",
            ],
        },
    },
    "Q2": {
        "Early": {
            "Winner Ladder": [
                "Direct longs: energy, materials, industrials, financials, commodity beta",
                "Secondary: banks, small caps, exporter EM / IHSG beta, liquid crypto beta",
                "Spillover: property / domestic cyclicals after the reflation tape is clearly leading",
            ],
            "Loser Ladder": [
                "Best shorts: duration, utilities, staples, health care, bond proxies",
                "Secondary shorts: low beta / quality if rates keep backing up",
            ],
            "If the phase weakens, rotate toward": [
                "Toward Q1 if inflation cools while growth stays okay → back to tech / growth / discretionary",
                "Toward Q3 if growth cracks while inflation stays hot → add USD, hard assets, defensives",
            ],
        },
        "Mid": {
            "Winner Ladder": [
                "Keep commodity / financial leadership but be stricter on balance-sheet quality",
                "Reduce lower-quality beta if credit or USD stress starts to rise",
            ],
            "Loser Ladder": [
                "Duration and defensives can still lag, but crowded cyclicals become two-way risk",
                "Fragile importers become cleaner shorts than broad defensives if oil stays hot",
            ],
            "If the phase weakens, rotate toward": [
                "Toward Q1 if rates calm and growth stays resilient",
                "Toward Q3 if USD / oil / front-end yields start doing the heavy lifting",
            ],
        },
        "Late": {
            "Winner Ladder": [
                "Take profits in crowded commodity / small-cap / crypto beta",
                "Keep only the strongest cash-flow reflation winners",
            ],
            "Loser Ladder": [
                "Best new shorts become lower-quality cyclicals, fragile EM, and over-owned small caps",
                "Defensives stop being clean shorts if slowdown risk rises",
            ],
            "If the phase weakens, rotate toward": [
                "Toward Q1 if inflation cools fast",
                "Toward Q3 if growth fades but inflation / oil stay sticky",
            ],
        },
    },
    "Q3": {
        "Early": {
            "Winner Ladder": [
                "Direct longs: USD, oil / hard assets, energy exporters, utilities / staples / health care",
                "Secondary: selected gold, selected exporter IHSG names, defensive cash-flow EM exceptions",
                "Spillover: BTC tactical only, not broad alt beta",
            ],
            "Loser Ladder": [
                "Best shorts: broad EM, EM FX, domestic-beta cyclicals, financials, discretionary, industrials, materials",
                "Secondary shorts: IHSG banks / property / lower-quality cyclicals, miners, alt beta",
            ],
            "If the phase weakens, rotate toward": [
                "Toward Q4 if 2Y / 5Y / 10Y / 30Y all start falling and duration takes leadership",
                "Toward Q2 only if growth stabilizes and inflation stays hot enough to reflate cyclicals",
            ],
        },
        "Mid": {
            "Winner Ladder": [
                "Keep hard-asset / USD / defensive cash-flow winners, but size down anything that relied on a one-off oil squeeze",
                "Gold improves if inflation-hedge demand catches up and the long end stops backing up",
            ],
            "Loser Ladder": [
                "Broad EM and domestic cyclicals remain the cleaner shorts than pure defensives",
                "Alt beta stays weaker than BTC even during tactical bounces",
            ],
            "If the phase weakens, rotate toward": [
                "Toward Q4 if growth fear starts dominating inflation fear",
                "Toward Q2 only if breadth, credit, and nominal growth all re-accelerate together",
            ],
        },
        "Late": {
            "Winner Ladder": [
                "Harvest hard-asset winners as duration / defensives begin taking over",
                "Keep only the strongest exporter / cash-flow names and shrink tactical crypto",
            ],
            "Loser Ladder": [
                "Best shorts start migrating from broad EM / cyclicals toward the weakest residual commodity and alt-beta names",
                "Do not overstay shorts in already-broken beta if yields are now falling fast",
            ],
            "If the phase weakens, rotate toward": [
                "Toward Q4 first in most classic late-Q3 transitions",
                "Toward Q2 only if the slowdown aborts and growth breadth re-accelerates",
            ],
        },
    },
    "Q4": {
        "Early": {
            "Winner Ladder": [
                "Direct longs: duration, Treasuries, USD, staples, health care, utilities",
                "Secondary: quality growth and selected defensive IHSG names",
                "Spillover: BTC watchlist only, not broad crypto beta",
            ],
            "Loser Ladder": [
                "Best shorts: commodities needing hot nominal growth, lower-quality cyclicals, property, broad EM, alt beta",
                "Secondary shorts: financials, industrials, materials, lower-quality credit",
            ],
            "If the phase weakens, rotate toward": [
                "Toward Q1 if growth stabilizes while inflation keeps cooling → quality growth / cyclicals improve",
                "Toward Q3 if inflation re-heats before growth recovers → hard assets / USD re-take leadership",
            ],
        },
        "Mid": {
            "Winner Ladder": [
                "Keep duration / defensives / quality, but start ranking recovery candidates",
                "BTC can move from watchlist to tactical only after liquidity and breadth stabilize",
            ],
            "Loser Ladder": [
                "Lower-quality cyclicals and alt beta stay cleaner shorts than already-defensive winners",
                "Broad EM remains weak until USD and rates stop pressing it",
            ],
            "If the phase weakens, rotate toward": [
                "Toward Q1 on cleaner disinflationary recovery",
                "Toward Q3 if the slowdown gets replaced by another inflation shock",
            ],
        },
        "Late": {
            "Winner Ladder": [
                "Take profits in pure duration and defensives as early recovery leaders begin to emerge",
                "Upgrade watchlist names in banks, cyclicals, EM, and BTC only when the bottoming evidence is real",
            ],
            "Loser Ladder": [
                "Best shorts shift away from the already-broken areas and toward residual late-defensive crowding if rates reverse higher",
                "Do not press commodity / cyclicals shorts if the recovery handoff is becoming obvious",
            ],
            "If the phase weakens, rotate toward": [
                "Toward Q1 in the cleaner recovery path",
                "Toward Q3 if inflation re-accelerates before the recovery is secure",
            ],
        },
    },
}

LEADERSHIP_ROTATION_MAP = {
    "Q1": {
        "Early": {
            "Who usually moves first": [
                "quality growth leadership first: tech, communication services, consumer discretionary, liquid cyclicals",
                "banks / broad cyclicals join only after breadth and credit confirm the initial move",
            ],
            "If leaders cool, handoff usually goes to": [
                "high beta, momentum, credit beta, then selected EM / IHSG banks",
                "BTC first before broader liquid crypto if breadth keeps widening",
            ],
            "Who usually moves last / spillover": [
                "property, lower-liquidity domestic beta, and weaker crypto beta",
            ],
            "Exhaustion / invalidation signs": [
                "low-quality beta or illiquid spillover starts outperforming the original liquid leaders",
                "defensives stop lagging and breadth narrows while the original leaders lose relative strength",
            ],
        },
        "Mid": {
            "Who usually moves first": [
                "the original liquid quality-growth leaders should still hold relative strength",
            ],
            "If leaders cool, handoff usually goes to": [
                "banks, industrials, selective EM / IHSG domestic beta, and then cleaner crypto quality",
            ],
            "Who usually moves last / spillover": [
                "property and lower-liquidity beta after the larger liquid buckets already worked",
            ],
            "Exhaustion / invalidation signs": [
                "handoff fails and only the weakest beta keeps running",
                "2Y / 5Y back up, oil / breakevens reheat, and the tape starts looking more like Q2 than Q1",
            ],
        },
        "Late": {
            "Who usually moves first": [
                "remaining liquid quality leaders and a narrower set of cyclicals",
            ],
            "If leaders cool, handoff usually goes to": [
                "there should be less handoff and more profit-taking unless a true Q2 or Q4 transition is forming",
            ],
            "Who usually moves last / spillover": [
                "lower-quality EM / crypto / domestic-beta names",
            ],
            "Exhaustion / invalidation signs": [
                "late junky spillover outperforms while original leaders stall",
                "duration / defensives begin taking leadership, pointing toward Q4, or energy / materials take over, pointing toward Q2",
            ],
        },
    },
    "Q2": {
        "Early": {
            "Who usually moves first": [
                "energy, materials, industrials, financials, commodity-linked cyclicals",
                "rates and nominal-growth proxies usually confirm early through firm 2Y / 5Y / 10Y",
            ],
            "If leaders cool, handoff usually goes to": [
                "banks, small caps, exporter EM / IHSG beta, then liquid crypto beta",
            ],
            "Who usually moves last / spillover": [
                "property, logistics, and broader domestic cyclicals after reflation leadership is already proven",
            ],
            "Exhaustion / invalidation signs": [
                "small caps / alt beta start going vertical while the original reflation leaders stall",
                "commodities stop confirming but lower-quality beta keeps squeezing",
            ],
        },
        "Mid": {
            "Who usually moves first": [
                "core reflation leaders should still be energy / materials / industrials / financials",
            ],
            "If leaders cool, handoff usually goes to": [
                "banks, exporter EM, selected IHSG cyclicals, and then liquid crypto beta if liquidity remains friendly",
            ],
            "Who usually moves last / spillover": [
                "property and lower-quality domestic beta",
            ],
            "Exhaustion / invalidation signs": [
                "the handoff shifts too quickly into junkier beta while core commodities and financials stop leading",
                "USD / oil / front-end yields become the only leaders, which can foreshadow a Q3 handoff",
            ],
        },
        "Late": {
            "Who usually moves first": [
                "only the strongest cash-flow reflation names should still be leading",
            ],
            "If leaders cool, handoff usually goes to": [
                "there is usually less healthy handoff and more rotation into weaker beta or into next-regime winners",
            ],
            "Who usually moves last / spillover": [
                "small caps, fragile EM, lower-quality cyclicals, alt beta",
            ],
            "Exhaustion / invalidation signs": [
                "late spillover is strongest while commodities / financials lose leadership",
                "defensives and USD stop lagging, or duration begins outperforming",
            ],
        },
    },
    "Q3": {
        "Early": {
            "Who usually moves first": [
                "USD strength, oil / hard assets, energy exporters, and defensive cash-flow equities",
                "front-end and belly repricing through 2Y / 5Y often confirm before every asset proxy aligns",
            ],
            "If leaders cool, handoff usually goes to": [
                "selected gold and exporter names if inflation-hedge demand broadens",
                "REITs / duration only if the long end begins to fall on growth fear",
            ],
            "Who usually moves last / spillover": [
                "BTC tactical only, then very selective country-specific exceptions; broad EM and broad alts usually do not deserve the handoff",
            ],
            "Exhaustion / invalidation signs": [
                "broad EM, domestic cyclicals, or alt beta start ripping while USD / oil / defensive leaders fade",
                "2Y / 5Y stop backing up and duration begins taking over decisively, pointing toward Q4",
            ],
        },
        "Mid": {
            "Who usually moves first": [
                "hard-asset / USD / defensive leaders should still dominate the tape",
            ],
            "If leaders cool, handoff usually goes to": [
                "gold and selected exporter / cash-flow names if the move broadens in a cleaner way",
                "duration only on clear growth-fear confirmation rather than on blind hope",
            ],
            "Who usually moves last / spillover": [
                "tactical BTC only; broad alt beta remains a poor-quality spillover",
            ],
            "Exhaustion / invalidation signs": [
                "only low-quality residual commodity or alt-beta names are still moving",
                "broad EM and cyclicals stop underperforming, which weakens the clean Q3 map",
            ],
        },
        "Late": {
            "Who usually moves first": [
                "the remaining winners are usually the strongest exporter / cash-flow / defensive names",
            ],
            "If leaders cool, handoff usually goes to": [
                "duration and deeper defensives first in the classic Q3→Q4 path",
                "only in the less-common abort path does the handoff go back toward Q2 cyclicals",
            ],
            "Who usually moves last / spillover": [
                "residual commodity beta and low-quality hard-asset chasers",
            ],
            "Exhaustion / invalidation signs": [
                "the only things still working are lower-quality residual winners while duration takes over",
                "already-broken shorts stop making new downside progress because the market is handing off into Q4",
            ],
        },
    },
    "Q4": {
        "Early": {
            "Who usually moves first": [
                "duration, Treasuries, USD, staples, health care, utilities",
                "falling 2Y / 5Y usually confirm before high-beta recovery trades deserve attention",
            ],
            "If leaders cool, handoff usually goes to": [
                "quality growth and selected defensive cash-flow names",
                "BTC only moves from watchlist to tactical if liquidity and breadth genuinely stabilize",
            ],
            "Who usually moves last / spillover": [
                "banks, cyclicals, broad EM, and alt beta only after a true bottoming process starts",
            ],
            "Exhaustion / invalidation signs": [
                "recovery beta starts outperforming while duration and defensives lose leadership",
                "oil / breakevens and front-end yields reheat, creating a Q3 risk rather than a clean Q1 recovery",
            ],
        },
        "Mid": {
            "Who usually moves first": [
                "duration / defensives / quality should still dominate",
            ],
            "If leaders cool, handoff usually goes to": [
                "quality growth first, then selected banks / BTC only after bottoming evidence strengthens",
            ],
            "Who usually moves last / spillover": [
                "broad EM, commodity beta, and broad alt beta",
            ],
            "Exhaustion / invalidation signs": [
                "broad recovery beta starts improving before the old defensive winners can extend",
                "yields stop falling and curve dynamics shift away from the clean deflationary setup",
            ],
        },
        "Late": {
            "Who usually moves first": [
                "pure duration and defensives usually stop being the only game in town",
            ],
            "If leaders cool, handoff usually goes to": [
                "quality growth, then banks / cyclicals / BTC on a cleaner Q1 handoff",
                "or back to hard assets / USD if inflation shock risk revives Q3 instead",
            ],
            "Who usually moves last / spillover": [
                "lower-quality cyclicals, broad EM, and alt beta only after the recovery handoff is already real",
            ],
            "Exhaustion / invalidation signs": [
                "late defensives remain crowded while recovery leaders keep broadening",
                "or, alternatively, oil and yields re-accelerate before recovery is secure, which breaks the clean Q4→Q1 path",
            ],
        },
    },
}


RATES_POLICY_GUIDE = {
    "Q1": [
        "2Y and 5Y usually soften or at least stop backing up aggressively as inflation cools and policy pressure eases.",
        "10Y and 30Y do not need to collapse, but a major bear-steepening backup is usually not a clean Goldilocks confirmation.",
        "Best read: front-end calm to easier, belly controlled, long-end not disorderly, while growth remains resilient.",
    ],
    "Q2": [
        "2Y, 5Y, and 10Y often stay firm because nominal growth and inflation are both strong.",
        "30Y can also rise, especially in a bear-steepening reflation move that hurts pure duration trades.",
        "Best read: front-end firm, belly firm, long-end firm but orderly, credit calm, and cyclicals leading.",
    ],
    "Q3": [
        "2Y and 5Y matter a lot here: if the front-end and belly back up, the market is repricing sticky inflation / fewer cuts, which is compatible with a stagflationary read.",
        "10Y is the broad nominal-conditions lens; 30Y is the duration / growth-fear / term-premium lens. The long end can rise in an oil / term-premium shock or fall if growth fear dominates.",
        "That is why Oil Up + USD Up + Gold down for a few sessions does not invalidate Q3 by itself; it can simply mean front-end policy repricing is leading the tape.",
    ],
    "Q4": [
        "2Y and 5Y usually fall as the market prices easing / disinflation / growth stress.",
        "10Y and 30Y often fall too, which is why long duration tends to be a core winner in deflationary slowdowns.",
        "Best read: front-end, belly, and long-end yields all easing while defensives and duration outperform.",
    ],
}

for _q, _items in RATES_POLICY_GUIDE.items():
    PHASE_GUIDE[_q]["meaning"]["Rates / Policy Lens"] = _items

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


@dataclass
class QuadSelectionState:
    driver_label: str
    active_scores: Dict[str, float]
    current_quad: str
    fit_score: float
    source_key: str


@dataclass
class DashboardState:
    signals: Dict[str, float]
    fg_info: Dict[str, object]
    quad: QuadSelectionState
    current_paths: List[Dict[str, object]]
    primary_path: Dict[str, object]
    validity: str
    stage: str


def choose_quad_selection(signals: Dict[str, float], driver_label: str) -> QuadSelectionState:
    if driver_label == "Monthly (Hedgeye-style current call)":
        active_scores = signals["quad_scores_monthly"]
        current_quad = str(signals["macro_quad_monthly"])
        source_key = "monthly"
    elif driver_label == "Quarterly Anchor":
        active_scores = signals["quad_scores_quarterly"]
        current_quad = str(signals["macro_quad_quarterly"])
        source_key = "quarterly"
    else:
        active_scores = signals["quad_scores"]
        current_quad = max(active_scores, key=active_scores.get)
        source_key = "blend"
    return QuadSelectionState(
        driver_label=driver_label,
        active_scores=active_scores,
        current_quad=current_quad,
        fit_score=float(active_scores[current_quad]),
        source_key=source_key,
    )


def build_dashboard_state(signals: Dict[str, float], fg_info: Dict[str, object], driver_label: str) -> DashboardState:
    quad = choose_quad_selection(signals, driver_label)
    current_paths = finalize_paths(build_paths(signals, quad.current_quad, quad.source_key), signals=signals)
    primary_path = max(current_paths, key=lambda x: x["score"])
    max_path_score = max(p["score"] for p in current_paths)
    validity = classify_validity(quad.fit_score, max_path_score)
    stage = transition_stage(signals, quad.fit_score, current_paths, quad.source_key)
    return DashboardState(
        signals=signals,
        fg_info=fg_info,
        quad=quad,
        current_paths=current_paths,
        primary_path=primary_path,
        validity=validity,
        stage=stage,
    )


def quad_scores_for_source(signals: Dict[str, float], source_key: str) -> Dict[str, float]:
    if source_key == "monthly":
        return signals["quad_scores_monthly"]
    if source_key == "quarterly":
        return signals["quad_scores_quarterly"]
    return signals["quad_scores"]


def transition_axes(signals: Dict[str, float], source_key: str) -> Dict[str, float]:
    g_m = float(signals.get("gdp_nowcast_monthly", 0.0))
    g_q = float(signals.get("gdp_nowcast_quarterly", 0.0))
    i_m = float(signals.get("cpi_nowcast_monthly", 0.0))
    i_q = float(signals.get("cpi_nowcast_quarterly", 0.0))
    if source_key == "monthly":
        gl = 0.70 * g_m + 0.30 * g_q
        gm = g_m
        il = 0.70 * i_m + 0.30 * i_q
        im = i_m
        source_blend = 0.75
    elif source_key == "quarterly":
        gl = g_q
        gm = 0.70 * g_q + 0.30 * g_m
        il = i_q
        im = 0.70 * i_q + 0.30 * i_m
        source_blend = 0.25
    else:
        gl = 0.60 * g_q + 0.40 * g_m
        gm = 0.75 * g_m + 0.25 * g_q
        il = 0.60 * i_q + 0.40 * i_m
        im = 0.75 * i_m + 0.25 * i_q
        source_blend = 0.50
    ls = 0.55 * float(signals.get("claims_yoy_z", 0.0)) + 0.25 * float(signals.get("sahm_z", 0.0)) + 0.20 * float(signals.get("recpro_z", 0.0))
    rr = 0.45 * float(signals.get("sahm_z", 0.0)) + 0.35 * float(signals.get("recpro_z", 0.0)) + 0.20 * (-gl)
    return {"gl": gl, "gm": gm, "il": il, "im": im, "ls": ls, "rr": rr, "source_blend": source_blend}


def path_signal_alignment(signals: Dict[str, float], target_quad: str) -> float:
    front = float(signals.get("front_end_policy", 0.0))
    duration = float(signals.get("duration_tailwind", 0.0))
    usd = float(signals.get("usd_signal", 0.0))
    commodity = float(signals.get("commodity_breadth", 0.0))
    breadth = float(signals.get("breadth_health", 0.0))
    credit = float(signals.get("credit_stress", 0.0))
    em = float(signals.get("broad_em_equity_signal", 0.0))
    nasdaq = float(signals.get("nasdaq_signal", 0.0))
    top = float(signals.get("behavioral_top_score", 0.0))
    dist = float(signals.get("distribution_risk", 0.0))

    checks = {
        "Q1": [duration, nasdaq, breadth, -front, -credit],
        "Q2": [commodity, breadth, em, -usd, -credit],
        "Q3": [commodity, usd, front, -em, -duration],
        "Q4": [duration, credit, dist, usd, -breadth],
    }[target_quad]
    vals = [clamp01(0.5 + 0.30 * x) for x in checks]
    # Penalize noisy late-stage risk when chasing reflation/recovery; reward it a bit for Q4-type downside.
    if target_quad in {"Q1", "Q2"}:
        vals.append(clamp01(1.0 - 0.55 * top))
    elif target_quad == "Q4":
        vals.append(clamp01(0.55 + 0.45 * dist))
    else:
        vals.append(clamp01(0.55 + 0.25 * max(usd, 0.0)))
    return float(sum(vals) / len(vals))


def transition_stage(signals: Dict[str, float], current_fit: float, paths: List[Dict[str, object]], source_key: str) -> str:
    if not paths:
        return "Mid"
    max_path_score = max(float(p.get("score", 0.0)) for p in paths) / 100.0
    fit = clamp01(current_fit / 100.0)
    top = float(signals.get("behavioral_top_score", 0.0))
    dist = float(signals.get("distribution_risk", 0.0))
    credit = clamp01(0.5 + 0.25 * float(signals.get("credit_stress", 0.0)))
    breadth_fragility = clamp01(0.5 - 0.25 * float(signals.get("breadth_health", 0.0)))
    monthly_quad = str(signals.get("macro_quad_monthly", ""))
    quarterly_quad = str(signals.get("macro_quad_quarterly", ""))
    divergence = 1.0 if monthly_quad and quarterly_quad and monthly_quad != quarterly_quad else 0.0
    source_penalty = 0.10 if source_key == "blend" and divergence else 0.0
    maturity = (
        0.38 * max_path_score
        + 0.18 * (1.0 - fit)
        + 0.14 * top
        + 0.12 * dist
        + 0.10 * credit
        + 0.08 * breadth_fragility
        + source_penalty
    )
    if maturity < 0.42:
        return "Early"
    if maturity < 0.67:
        return "Mid"
    return "Late"


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


def _safe_pct_series(df: pd.DataFrame, col: str, sign: float = 1.0) -> Optional[pd.Series]:
    if df.empty or col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return None
    return sign * s.pct_change().dropna()


def _safe_diff_series(bundle: Dict[str, pd.Series], key: str, sign: float = 1.0) -> Optional[pd.Series]:
    if key not in bundle:
        return None
    s = pd.to_numeric(bundle[key], errors="coerce").dropna()
    if s.empty:
        return None
    return sign * s.diff().dropna()


def _aligned_corr_stats(series_list: List[pd.Series], tail: int = 126, min_obs: int = 40) -> Tuple[Optional[float], Optional[float], Optional[pd.Series]]:
    valid = [s for s in series_list if s is not None and not s.dropna().empty]
    if len(valid) < 2:
        return None, None, None
    df = pd.concat(valid, axis=1, join="inner").dropna().tail(tail)
    if len(df) < min_obs or df.shape[1] < 2:
        return None, None, None
    corr = df.corr()
    vals = []
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[1]):
            v = corr.iat[i, j]
            if pd.notna(v):
                vals.append(abs(float(v)))
    if not vals:
        return None, None, None
    rep = df.mean(axis=1)
    return float(np.mean(vals)), float(np.min(vals)), rep


def build_correlation_audit(bundle: Dict[str, pd.Series], yf_close: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    families = [
        {
            "name": "Front-End Rates Family",
            "members": [
                _safe_diff_series(bundle, "DGS2"),
                _safe_diff_series(bundle, "DGS5"),
            ],
            "note": "2Y dan 5Y biasanya bergerak sangat rapat. Ini lebih bersih dibaca sebagai satu family policy / front-end repricing.",
        },
        {
            "name": "Long-End Rates Family",
            "members": [
                _safe_diff_series(bundle, "DGS10"),
                _safe_diff_series(bundle, "DGS30"),
            ],
            "note": "10Y dan 30Y idealnya dibaca sebagai satu family duration / long-end pressure dulu, baru dibedah divergence-nya bila perlu.",
        },
        {
            "name": "USD Complex",
            "members": [
                _safe_pct_series(yf_close, "UUP", 1.0),
                _safe_pct_series(yf_close, "FXE", -1.0),
                _safe_pct_series(yf_close, "FXB", -1.0),
                _safe_pct_series(yf_close, "FXY", -1.0),
                _safe_pct_series(yf_close, "CEW", -1.0),
            ],
            "note": "Kalau korelasinya rapat, baca ini sebagai satu family dollar-strength / EM-FX pressure, bukan lima sinyal terpisah.",
        },
        {
            "name": "Europe Equity Complex",
            "members": [
                _safe_pct_series(yf_close, "EZU"),
                _safe_pct_series(yf_close, "VGK"),
            ],
            "note": "EZU dan VGK biasanya redundant. Kalau rapat, cukup satu European risk family.",
        },
        {
            "name": "China Equity Complex",
            "members": [
                _safe_pct_series(yf_close, "FXI"),
                _safe_pct_series(yf_close, "MCHI"),
            ],
            "note": "FXI dan MCHI sering cukup rapat untuk dibaca sebagai satu China risk family terlebih dulu.",
        },
        {
            "name": "India Equity Complex",
            "members": [
                _safe_pct_series(yf_close, "INDA"),
                _safe_pct_series(yf_close, "EPI"),
            ],
            "note": "Kalau rapat, cukup satu India family; pecah lagi hanya saat ada divergence local factor.",
        },
        {
            "name": "IHSG Banks Family",
            "members": [
                _safe_pct_series(yf_close, "BBCA.JK"),
                _safe_pct_series(yf_close, "BBRI.JK"),
                _safe_pct_series(yf_close, "BMRI.JK"),
                _safe_pct_series(yf_close, "BBNI.JK"),
            ],
            "note": "Empat bank besar biasanya sangat rapat. Ini cocok dibaca sebagai satu leadership family dulu.",
        },
        {
            "name": "IHSG Property Family",
            "members": [
                _safe_pct_series(yf_close, "PWON.JK"),
                _safe_pct_series(yf_close, "SMRA.JK"),
                _safe_pct_series(yf_close, "CTRA.JK"),
            ],
            "note": "Property names sering bergerak bersama, tapi tetap pantau divergence kalau funding/local rates berubah cepat.",
        },
        {
            "name": "IHSG Defensive Staples/Telecom/Pharma Family",
            "members": [
                _safe_pct_series(yf_close, "TLKM.JK"),
                _safe_pct_series(yf_close, "ICBP.JK"),
                _safe_pct_series(yf_close, "INDF.JK"),
                _safe_pct_series(yf_close, "KLBF.JK"),
            ],
            "note": "Defensive IHSG names kadang cukup rapat, tapi sering lebih conditional daripada banks/property.",
        },
        {
            "name": "IHSG Resource Family",
            "members": [
                _safe_pct_series(yf_close, "ADRO.JK"),
                _safe_pct_series(yf_close, "ITMG.JK"),
                _safe_pct_series(yf_close, "PGAS.JK"),
                _safe_pct_series(yf_close, "ANTM.JK"),
                _safe_pct_series(yf_close, "MDKA.JK"),
            ],
            "note": "Resource names boleh dikelompokkan kalau korelasinya rapat, tapi tetap pecah lagi untuk coal vs oil-gas vs metals bila chain-nya berbeda.",
        },
        {
            "name": "Commodity Breadth ex-Gold",
            "members": [
                _safe_pct_series(yf_close, "DBC"),
                _safe_pct_series(yf_close, "DBB"),
                _safe_pct_series(yf_close, "DBA"),
                _safe_pct_series(yf_close, "UNG"),
            ],
            "note": "Kalau cukup rapat, ini cocok dibaca sebagai commodity breadth family; gold tetap dipisah karena perannya bisa beda.",
        },
        {
            "name": "Alt-Beta Crypto Family",
            "members": [
                _safe_pct_series(yf_close, "ETH-USD"),
                _safe_pct_series(yf_close, "SOL-USD"),
                _safe_pct_series(yf_close, "BLOK"),
                _safe_pct_series(yf_close, "WGMI"),
            ],
            "note": "ETH/SOL dan crypto-equity beta sering lebih masuk akal dibaca sebagai satu alt-beta family; BTC sebaiknya tetap dipisah sebagai leader utama.",
        },
    ]

    merged_rows: List[Dict[str, object]] = []
    toggle_rows: List[Dict[str, object]] = []
    reps: Dict[str, pd.Series] = {}

    for fam in families:
        avg_abs, min_abs, rep = _aligned_corr_stats(fam["members"])
        if rep is not None:
            reps[fam["name"]] = rep
        if avg_abs is None:
            continue
        row = {
            "Family": fam["name"],
            "Avg |Corr|": round(avg_abs, 2),
            "Min |Corr|": round(min_abs if min_abs is not None else 0.0, 2),
            "Action": "Merge into one family" if avg_abs >= 0.78 and (min_abs or 0.0) >= 0.55 else "Keep toggle / conditional",
            "Note": fam["note"],
        }
        if row["Action"] == "Merge into one family":
            merged_rows.append(row)
        elif avg_abs >= 0.45:
            toggle_rows.append(row)

    def _pair_corr(a: pd.Series, b: pd.Series, tail: int = 126, min_obs: int = 40) -> Optional[float]:
        df = pd.concat([a, b], axis=1, join="inner").dropna().tail(tail)
        if len(df) < min_obs:
            return None
        c = df.corr().iat[0, 1]
        if pd.isna(c):
            return None
        return float(c)

    pair_notes = {
        frozenset(["Front-End Rates Family", "USD Complex"]): "Front-end hawkish repricing dan USD strength sering jalan bareng dalam shock regime.",
        frozenset(["Commodity Breadth ex-Gold", "IHSG Resource Family"]): "Commodity breadth yang sehat biasanya menetes ke resource proxies, tapi local flow bisa delay.",
        frozenset(["Alt-Beta Crypto Family", "USD Complex"]): "Dollar kuat sering jadi headwind untuk alt beta; lihat ini sebagai conditional inverse pair.",
        frozenset(["IHSG Banks Family", "IHSG Property Family"]): "Banks dan property sering nyambung lewat domestic liquidity / rates, tapi property lebih lagging.",
        frozenset(["Europe Equity Complex", "USD Complex"]): "Europe risk complex sering sensitif ke dollar direction dan global growth tone.",
        frozenset(["China Equity Complex", "Commodity Breadth ex-Gold"]): "China complex bisa nyambung ke cyclicals/commodities, tapi tidak selalu sinkron bila policy lokal mendominasi.",
        frozenset(["India Equity Complex", "USD Complex"]): "India tetap equity-heavy, tetapi dollar/funding backdrop tetap penting.",
        frozenset(["IHSG Resource Family", "USD Complex"]): "USD kuat bisa membantu sebagian resource revenues tapi menekan broad local flows; treat as mixed pair.",
    }

    pair_rows: List[Dict[str, object]] = []
    rep_names = list(reps.keys())
    for i in range(len(rep_names)):
        for j in range(i + 1, len(rep_names)):
            a_name, b_name = rep_names[i], rep_names[j]
            c = _pair_corr(reps[a_name], reps[b_name])
            if c is None or abs(c) < 0.45:
                continue
            pair_rows.append({
                "Pair": f"{a_name} ↔ {b_name}",
                "Corr": round(c, 2),
                "Type": "Strong" if abs(c) >= 0.70 else "Moderate",
                "Read": pair_notes.get(frozenset([a_name, b_name]), "Treat as regime-dependent cross-family relationship; do not fully merge unless the linkage stays persistent."),
            })

    merged_df = pd.DataFrame(merged_rows).sort_values(["Avg |Corr|", "Min |Corr|"], ascending=False).reset_index(drop=True) if merged_rows else pd.DataFrame(columns=["Family", "Avg |Corr|", "Min |Corr|", "Action", "Note"])
    toggle_df = pd.DataFrame(toggle_rows).sort_values(["Avg |Corr|", "Min |Corr|"], ascending=False).reset_index(drop=True) if toggle_rows else pd.DataFrame(columns=["Family", "Avg |Corr|", "Min |Corr|", "Action", "Note"])
    pair_df = pd.DataFrame(pair_rows).sort_values("Corr", key=lambda s: s.abs(), ascending=False).reset_index(drop=True) if pair_rows else pd.DataFrame(columns=["Pair", "Corr", "Type", "Read"])
    return merged_df, toggle_df, pair_df


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


def _quad_scores_from_probs(g_up: float, i_up: float) -> Dict[str, float]:
    q1 = clamp01((g_up + (1.0 - i_up)) / 2.0)
    q2 = clamp01((g_up + i_up) / 2.0)
    q3 = clamp01(((1.0 - g_up) + i_up) / 2.0)
    q4 = clamp01(((1.0 - g_up) + (1.0 - i_up)) / 2.0)
    return {"Q1": q1 * 100.0, "Q2": q2 * 100.0, "Q3": q3 * 100.0, "Q4": q4 * 100.0}


def latest_signal_snapshot(bundle: Dict[str, pd.Series], fg_value: int) -> Dict[str, float]:
    """
    Internal architecture (visual intentionally unchanged):
    1) GDP nowcast engine (macro only)
    2) CPI nowcast engine (macro + inflation impulse only)
    3) Policy / rates engine (2Y, 5Y, 10Y, 30Y)
    4) Signal / risk engines (IWM, VIX, HY, Fear & Greed)
    """
    wei = bundle["WEI"]
    icsa = bundle["ICSA"]
    t10y2y = bundle["T10Y2Y"]
    dgs2 = bundle["DGS2"]
    dgs5 = bundle["DGS5"]
    dgs10 = bundle["DGS10"]
    dgs30 = bundle["DGS30"]
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
    claims_26w = pct_change(claims4, 26)

    wei4 = wei.diff(4)
    wei13 = wei.diff(13)
    sahm13 = sahm.diff(13)
    recpro13 = recpro.diff(13)

    dgs2_20 = dgs2.diff(20)
    dgs2_63 = dgs2.diff(63)
    dgs5_20 = dgs5.diff(20)
    dgs5_63 = dgs5.diff(63)
    dgs10_20 = dgs10.diff(20)
    dgs10_63 = dgs10.diff(63)
    dgs30_20 = dgs30.diff(20)
    dgs30_63 = dgs30.diff(63)

    curve5s2s = dgs5 - dgs2
    curve10s5s = dgs10 - dgs5
    curve30s10s = dgs30 - dgs10
    curve30s2s = dgs30 - dgs2
    curve5s2s_20 = curve5s2s.diff(20)
    curve10s5s_20 = curve10s5s.diff(20)
    curve30s10s_20 = curve30s10s.diff(20)
    curve30s2s_20 = curve30s2s.diff(20)

    cpi3 = ann_roc(cpi, 3)
    cpi6 = ann_roc(cpi, 6)
    core3 = ann_roc(core_cpi, 3)
    core6 = ann_roc(core_cpi, 6)
    cpi12 = pct_change(cpi, 12)
    core12 = pct_change(core_cpi, 12)
    cpi_gap = cpi3 - cpi6
    core_gap = core3 - core6
    cpi6_gap = cpi6 - cpi12
    core6_gap = core6 - core12

    oil21 = pct_change(oil, 21)
    oil63 = pct_change(oil, 63)
    breakeven20 = breakeven.diff(20)
    breakeven63 = breakeven.diff(63)

    signals: Dict[str, float] = {
        # Real-economy / GDP nowcast features
        "wei_level_z": rolling_z_last(wei, 156),
        "wei_4w_z": rolling_z_last(wei4, 156),
        "wei_13w_z": rolling_z_last(wei13, 156),
        "claims_yoy_z": rolling_z_last(claims_yoy, 156),
        "claims_13w_z": rolling_z_last(claims_13w, 156),
        "claims_26w_z": rolling_z_last(claims_26w, 156),
        "sahm_z": rolling_z_last(sahm, 60),
        "sahm_13w_z": rolling_z_last(sahm13, 60),
        "recpro_z": rolling_z_last(recpro, 60),
        "recpro_13w_z": rolling_z_last(recpro13, 60),
        # Policy / rates features
        "curve_z": rolling_z_last(t10y2y, 252),
        "dgs2_z": rolling_z_last(dgs2, 252),
        "dgs5_z": rolling_z_last(dgs5, 252),
        "dgs10_z": rolling_z_last(dgs10, 252),
        "dgs30_z": rolling_z_last(dgs30, 252),
        "dgs2_20_z": rolling_z_last(dgs2_20, 252),
        "dgs2_63_z": rolling_z_last(dgs2_63, 252),
        "dgs5_20_z": rolling_z_last(dgs5_20, 252),
        "dgs5_63_z": rolling_z_last(dgs5_63, 252),
        "dgs10_20_z": rolling_z_last(dgs10_20, 252),
        "dgs10_63_z": rolling_z_last(dgs10_63, 252),
        "dgs30_20_z": rolling_z_last(dgs30_20, 252),
        "dgs30_63_z": rolling_z_last(dgs30_63, 252),
        "curve5s2s_z": rolling_z_last(curve5s2s, 252),
        "curve10s5s_z": rolling_z_last(curve10s5s, 252),
        "curve30s10s_z": rolling_z_last(curve30s10s, 252),
        "curve30s2s_z": rolling_z_last(curve30s2s, 252),
        "curve5s2s_20_z": rolling_z_last(curve5s2s_20, 252),
        "curve10s5s_20_z": rolling_z_last(curve10s5s_20, 252),
        "curve30s10s_20_z": rolling_z_last(curve30s10s_20, 252),
        "curve30s2s_20_z": rolling_z_last(curve30s2s_20, 252),
        # Inflation / CPI nowcast features
        "cpi3_z": rolling_z_last(cpi3, 60),
        "cpi6_z": rolling_z_last(cpi6, 60),
        "core3_z": rolling_z_last(core3, 60),
        "core6_z": rolling_z_last(core6, 60),
        "cpi_gap_z": rolling_z_last(cpi_gap, 60),
        "core_gap_z": rolling_z_last(core_gap, 60),
        "cpi6_gap_z": rolling_z_last(cpi6_gap, 60),
        "core6_gap_z": rolling_z_last(core6_gap, 60),
        "breakeven_z": rolling_z_last(breakeven, 252),
        "breakeven20_z": rolling_z_last(breakeven20, 252),
        "breakeven63_z": rolling_z_last(breakeven63, 252),
        "oil21_z": rolling_z_last(oil21, 252),
        "oil63_z": rolling_z_last(oil63, 252),
        # Financial / market raw features
        "hy_z": rolling_z_last(hy, 252),
        "nfci_z": rolling_z_last(nfci, 156),
        "stlfsi_z": rolling_z_last(stlfsi, 156),
        "vix_z": rolling_z_last(vix, 252),
        # Last values / dates
        "sahm_last": safe_last(sahm),
        "recpro_last": safe_last(recpro),
        "cpi3_last": safe_last(cpi3),
        "core3_last": safe_last(core3),
        "curve_last": safe_last(t10y2y),
        "dgs2_last": safe_last(dgs2),
        "dgs5_last": safe_last(dgs5),
        "dgs10_last": safe_last(dgs10),
        "dgs30_last": safe_last(dgs30),
        "curve5s2s_last": safe_last(curve5s2s),
        "curve10s5s_last": safe_last(curve10s5s),
        "curve30s10s_last": safe_last(curve30s10s),
        "curve30s2s_last": safe_last(curve30s2s),
        "wei_last": safe_last(wei),
        "claims_last": safe_last(claims4),
        "breakeven_last": safe_last(breakeven),
        "hy_last": safe_last(hy),
        "last_WEI": last_date(wei),
        "last_ICSA": last_date(icsa),
        "last_T10Y2Y": last_date(t10y2y),
        "last_DGS2": last_date(dgs2),
        "last_DGS5": last_date(dgs5),
        "last_DGS10": last_date(dgs10),
        "last_DGS30": last_date(dgs30),
        "last_CPI": last_date(cpi),
        "last_CORE_CPI": last_date(core_cpi),
        "last_OIL": last_date(oil),
        "last_HY_OAS": last_date(hy),
        "last_VIX": last_date(vix),
    }

    # Market overlays - separate from macro quad engine.
    yf_close = load_yf_close(("IWM", "SPY", "QQQ", "UUP", "FXE", "FXB", "FXY", "CEW", "GLD", "DBC", "DBB", "DBA", "UNG", "SPHB", "SPLV", "MTUM", "QUAL", "IWF", "IWD", "EEM", "EIDO", "^JKSE", "TLT", "HYG", "BLOK", "WGMI", "BTC-USD", "ETH-USD", "SOL-USD", "FXI", "MCHI", "EWJ", "EZU", "VGK", "INDA", "EPI", "EWZ", "EWA", "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK", "ICBP.JK", "INDF.JK", "KLBF.JK", "ADRO.JK", "ITMG.JK", "PGAS.JK", "ANTM.JK", "MDKA.JK", "PWON.JK", "SMRA.JK", "CTRA.JK", "AKRA.JK", "UNTR.JK", "ASII.JK", "AMRT.JK", "ACES.JK", "MAPI.JK"))
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

    # Broader market-signal families (kept separate from macro quad engine)
    def _zret(col: str, lookback: int = 20) -> float:
        if yf_close.empty or col not in yf_close.columns:
            return 0.0
        return rolling_z_last(pct_change(yf_close[col], lookback), 252)

    def _zratio(numer: str, denom: str, lookback: int = 20) -> float:
        if yf_close.empty or numer not in yf_close.columns or denom not in yf_close.columns:
            return 0.0
        ratio = (yf_close[numer] / yf_close[denom]).dropna()
        return rolling_z_last(pct_change(ratio, lookback), 252)

    def _zbasket(cols: Tuple[str, ...], lookback: int = 20, min_names: int = 1) -> float:
        if yf_close.empty:
            return 0.0
        vals: List[float] = []
        for col in cols:
            if col in yf_close.columns:
                vals.append(rolling_z_last(pct_change(yf_close[col], lookback), 252))
        if len(vals) < min_names:
            return 0.0
        return float(np.nanmean(vals))

    usd_signal = (
        0.30 * _zret("UUP", 20)
        + 0.20 * _zret("UUP", 63)
        + 0.15 * (-_zret("FXE", 20))
        + 0.10 * (-_zret("FXB", 20))
        + 0.10 * (-_zret("FXY", 20))
        + 0.15 * (-_zret("CEW", 20))
    )
    emfx_signal = (
        0.60 * _zret("CEW", 20)
        + 0.20 * _zret("CEW", 63)
        + 0.20 * (-_zret("UUP", 20))
    )
    commodity_breadth = (
        0.28 * _zret("DBC", 20)
        + 0.16 * _zret("DBC", 63)
        + 0.18 * _zret("GLD", 20)
        + 0.14 * _zret("DBB", 20)
        + 0.12 * _zret("DBA", 20)
        + 0.12 * _zret("UNG", 20)
    )
    hard_asset_breadth = (
        0.45 * _zret("GLD", 20)
        + 0.35 * _zret("DBC", 20)
        + 0.20 * _zret("DBB", 20)
    )
    cyclical_style_signal = (
        0.30 * _zratio("SPHB", "SPLV", 20)
        + 0.20 * _zret("MTUM", 20)
        + 0.20 * _zratio("IWF", "IWD", 20)
        + 0.15 * _zret("IWM", 20)
        + 0.15 * _zret("QQQ", 20)
    )
    defensive_style_signal = (
        0.35 * (-_zratio("SPHB", "SPLV", 20))
        + 0.25 * _zret("QUAL", 20)
        + 0.20 * (-_zret("IWM", 20))
        + 0.20 * (-_zret("QQQ", 20))
    )
    duration_market_signal = (
        0.60 * _zret("TLT", 20)
        + 0.20 * _zret("TLT", 63)
        + 0.20 * (-_zret("HYG", 20))
    )
    crypto_equity_signal = (
        0.55 * _zret("BLOK", 20)
        + 0.45 * _zret("WGMI", 20)
    )
    broad_em_equity_signal = (
        0.50 * _zret("EEM", 20)
        + 0.20 * _zret("EEM", 63)
        + 0.20 * _zret("EIDO", 20)
        + 0.10 * (-_zret("UUP", 20))
    )
    indo_equity_signal = (
        0.40 * _zret("EIDO", 20)
        + 0.20 * _zret("EIDO", 63)
        + 0.20 * _zret("^JKSE", 20)
        + 0.20 * _zret("^JKSE", 63)
    )
    ihsg_bank_signal = 0.65 * _zbasket(("BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK"), 20, 2) + 0.35 * _zbasket(("BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK"), 63, 2)
    ihsg_defensive_signal = 0.65 * _zbasket(("TLKM.JK", "ICBP.JK", "INDF.JK", "KLBF.JK"), 20, 2) + 0.35 * _zbasket(("TLKM.JK", "ICBP.JK", "INDF.JK", "KLBF.JK"), 63, 2)
    ihsg_commodity_signal = 0.60 * _zbasket(("ADRO.JK", "ITMG.JK", "PGAS.JK", "ANTM.JK", "MDKA.JK"), 20, 2) + 0.40 * _zbasket(("ADRO.JK", "ITMG.JK", "PGAS.JK", "ANTM.JK", "MDKA.JK"), 63, 2)
    ihsg_property_signal = 0.65 * _zbasket(("PWON.JK", "SMRA.JK", "CTRA.JK"), 20, 2) + 0.35 * _zbasket(("PWON.JK", "SMRA.JK", "CTRA.JK"), 63, 2)
    ihsg_cyclical_signal = 0.45 * _zbasket(("AKRA.JK", "UNTR.JK", "ASII.JK"), 20, 2) + 0.25 * _zbasket(("AMRT.JK", "ACES.JK", "MAPI.JK"), 20, 1) + 0.30 * ihsg_bank_signal
    btc_signal = 0.60 * _zret("BTC-USD", 20) + 0.40 * _zret("BTC-USD", 63)
    eth_signal = 0.60 * _zret("ETH-USD", 20) + 0.40 * _zret("ETH-USD", 63)
    sol_signal = 0.60 * _zret("SOL-USD", 20) + 0.40 * _zret("SOL-USD", 63)
    crypto_major_signal = 0.50 * btc_signal + 0.30 * eth_signal + 0.20 * sol_signal
    alt_beta_signal = 0.35 * eth_signal + 0.25 * sol_signal + 0.40 * crypto_equity_signal
    crypto_quality_spread = btc_signal - alt_beta_signal

    us_equity_signal = 0.60 * _zret("SPY", 20) + 0.40 * _zret("SPY", 63)
    nasdaq_signal = 0.60 * _zret("QQQ", 20) + 0.40 * _zret("QQQ", 63)
    tlt_signal = 0.60 * _zret("TLT", 20) + 0.40 * _zret("TLT", 63)
    gld_signal = 0.60 * _zret("GLD", 20) + 0.40 * _zret("GLD", 63)
    dbc_signal = 0.60 * _zret("DBC", 20) + 0.40 * _zret("DBC", 63)
    europe_signal = 0.45 * _zret("EZU", 20) + 0.20 * _zret("EZU", 63) + 0.20 * _zret("VGK", 20) + 0.15 * (-_zret("UUP", 20))
    china_signal = 0.40 * _zret("FXI", 20) + 0.20 * _zret("FXI", 63) + 0.25 * _zret("MCHI", 20) + 0.15 * (-_zret("UUP", 20))
    japan_signal = 0.60 * _zret("EWJ", 20) + 0.25 * _zret("EWJ", 63) + 0.15 * (-_zret("UUP", 20))
    india_signal = 0.45 * _zret("INDA", 20) + 0.25 * _zret("INDA", 63) + 0.15 * _zret("EPI", 20) + 0.15 * (-_zret("UUP", 20))
    brazil_signal = 0.60 * _zret("EWZ", 20) + 0.25 * _zret("EWZ", 63) + 0.15 * commodity_breadth
    australia_signal = 0.60 * _zret("EWA", 20) + 0.25 * _zret("EWA", 63) + 0.15 * commodity_breadth

    correlation_merge_df, correlation_toggle_df, cross_family_corr_df = build_correlation_audit(bundle, yf_close)

    signals.update({
        "usd_signal": usd_signal,
        "emfx_signal": emfx_signal,
        "commodity_breadth": commodity_breadth,
        "hard_asset_breadth": hard_asset_breadth,
        "cyclical_style_signal": cyclical_style_signal,
        "defensive_style_signal": defensive_style_signal,
        "duration_market_signal": duration_market_signal,
        "crypto_equity_signal": crypto_equity_signal,
        "broad_em_equity_signal": broad_em_equity_signal,
        "indo_equity_signal": indo_equity_signal,
        "ihsg_bank_signal": ihsg_bank_signal,
        "ihsg_defensive_signal": ihsg_defensive_signal,
        "ihsg_commodity_signal": ihsg_commodity_signal,
        "ihsg_property_signal": ihsg_property_signal,
        "ihsg_cyclical_signal": ihsg_cyclical_signal,
        "btc_signal": btc_signal,
        "eth_signal": eth_signal,
        "sol_signal": sol_signal,
        "crypto_major_signal": crypto_major_signal,
        "alt_beta_signal": alt_beta_signal,
        "crypto_quality_spread": crypto_quality_spread,
        "us_equity_signal": us_equity_signal,
        "nasdaq_signal": nasdaq_signal,
        "tlt_signal": tlt_signal,
        "gld_signal": gld_signal,
        "dbc_signal": dbc_signal,
        "europe_signal": europe_signal,
        "china_signal": china_signal,
        "japan_signal": japan_signal,
        "india_signal": india_signal,
        "brazil_signal": brazil_signal,
        "australia_signal": australia_signal,
    })

    signals.update(fear_greed_overlays(fg_value))

    # =========================
    # 1) GDP & CPI NOWCAST ENGINE
    # =========================
    gdp_nowcast_monthly = (
        0.38 * signals["wei_4w_z"]
        + 0.32 * (-signals["claims_13w_z"])
        - 0.18 * signals["sahm_13w_z"]
        - 0.12 * signals["recpro_13w_z"]
    )
    gdp_nowcast_quarterly = (
        0.28 * signals["wei_level_z"]
        + 0.20 * signals["wei_13w_z"]
        + 0.20 * (-signals["claims_yoy_z"])
        + 0.10 * (-signals["claims_26w_z"])
        - 0.12 * signals["sahm_z"]
        - 0.10 * signals["recpro_z"]
    )

    inflation_market_impulse_monthly = 0.55 * signals["breakeven20_z"] + 0.45 * signals["oil21_z"]
    inflation_market_impulse_quarterly = 0.55 * signals["breakeven_z"] + 0.45 * signals["oil63_z"]

    cpi_nowcast_monthly = (
        0.30 * signals["cpi_gap_z"]
        + 0.25 * signals["core_gap_z"]
        + 0.20 * signals["cpi3_z"]
        + 0.05 * signals["core3_z"]
        + 0.20 * inflation_market_impulse_monthly
    )
    cpi_nowcast_quarterly = (
        0.22 * signals["cpi6_z"]
        + 0.22 * signals["core6_z"]
        + 0.14 * signals["cpi6_gap_z"]
        + 0.12 * signals["core6_gap_z"]
        + 0.30 * inflation_market_impulse_quarterly
    )

    growth_monthly_axis = gdp_nowcast_monthly
    growth_quarterly_axis = gdp_nowcast_quarterly
    inflation_monthly_axis = cpi_nowcast_monthly
    inflation_quarterly_axis = cpi_nowcast_quarterly

    g_up_monthly = sigmoid(1.20 * growth_monthly_axis)
    g_up_quarterly = sigmoid(1.10 * growth_quarterly_axis)
    i_up_monthly = sigmoid(1.20 * inflation_monthly_axis)
    i_up_quarterly = sigmoid(1.10 * inflation_quarterly_axis)

    growth_blend_axis = 0.60 * growth_quarterly_axis + 0.40 * growth_monthly_axis
    inflation_blend_axis = 0.60 * inflation_quarterly_axis + 0.40 * inflation_monthly_axis
    g_up = sigmoid(1.15 * growth_blend_axis)
    i_up = sigmoid(1.15 * inflation_blend_axis)
    g_down = 1.0 - g_up

    quad_scores_monthly = _quad_scores_from_probs(g_up_monthly, i_up_monthly)
    quad_scores_quarterly = _quad_scores_from_probs(g_up_quarterly, i_up_quarterly)
    quad_scores_blend = {
        q: 0.65 * quad_scores_quarterly[q] + 0.35 * quad_scores_monthly[q] for q in ["Q1", "Q2", "Q3", "Q4"]
    }

    # Macro-only transition helpers.
    labor_soft = 0.55 * signals["claims_yoy_z"] + 0.25 * signals["sahm_z"] + 0.20 * signals["recpro_z"]
    recession_risk = 0.45 * signals["sahm_z"] + 0.35 * signals["recpro_z"] + 0.20 * (-growth_quarterly_axis)
    growth_transition = sigmoid(-1.10 * growth_monthly_axis - 0.55 * growth_quarterly_axis)

    # =========================
    # 2) POLICY / RATES ENGINE
    # =========================
    front_end_policy = 0.50 * signals["dgs2_20_z"] + 0.25 * signals["dgs2_63_z"] + 0.25 * signals["dgs5_20_z"]
    belly_policy = 0.45 * signals["dgs5_20_z"] + 0.30 * signals["dgs5_63_z"] + 0.25 * signals["dgs10_20_z"]
    long_end_pressure = (
        0.30 * signals["dgs10_20_z"]
        + 0.20 * signals["dgs10_63_z"]
        + 0.30 * signals["dgs30_20_z"]
        + 0.20 * signals["dgs30_63_z"]
    )
    duration_tailwind = (
        0.30 * (-signals["dgs10_20_z"])
        + 0.20 * (-signals["dgs10_63_z"])
        + 0.30 * (-signals["dgs30_20_z"])
        + 0.20 * (-signals["dgs30_63_z"])
    )
    policy_easing_impulse = (
        0.45 * (-signals["dgs2_20_z"])
        + 0.25 * (-signals["dgs5_20_z"])
        + 0.15 * (-signals["dgs10_20_z"])
        + 0.15 * (-signals["dgs30_20_z"])
    )
    steepening_impulse = (
        0.30 * signals["curve5s2s_20_z"]
        + 0.20 * signals["curve10s5s_20_z"]
        + 0.30 * signals["curve30s10s_20_z"]
        + 0.20 * signals["curve30s2s_20_z"]
    )
    curve_regime = (
        0.40 * signals["curve5s2s_z"]
        + 0.20 * signals["curve10s5s_z"]
        + 0.20 * signals["curve30s10s_z"]
        + 0.20 * signals["curve30s2s_z"]
    )
    bear_steepener = 0.55 * max(long_end_pressure, 0.0) + 0.45 * max(steepening_impulse, 0.0)

    # =========================
    # 2.5) BAYESIAN-LITE OUT-QUARTER MODULE
    # =========================
    growth_base_effect = (
        0.30 * signals["wei_13w_z"]
        + 0.25 * (-signals["claims_26w_z"])
        - 0.25 * signals["sahm_13w_z"]
        - 0.20 * signals["recpro_13w_z"]
    )
    inflation_base_effect = (
        -0.25 * signals["cpi6_z"]
        - 0.20 * signals["core6_z"]
        + 0.20 * signals["cpi_gap_z"]
        + 0.20 * signals["core_gap_z"]
        + 0.15 * signals["cpi6_gap_z"]
        + 0.20 * inflation_market_impulse_monthly
    )
    gdp_nowcast_outquarter = 0.55 * gdp_nowcast_quarterly + 0.25 * gdp_nowcast_monthly + 0.20 * growth_base_effect
    cpi_nowcast_outquarter = 0.55 * cpi_nowcast_quarterly + 0.25 * cpi_nowcast_monthly + 0.20 * inflation_base_effect
    g_up_outquarter = sigmoid(1.10 * gdp_nowcast_outquarter)
    i_up_outquarter = sigmoid(1.10 * cpi_nowcast_outquarter)
    quad_scores_outquarter = _quad_scores_from_probs(g_up_outquarter, i_up_outquarter)

    # =========================
    # 3) SIGNAL / RISK ENGINES
    # =========================
    credit_stress = 0.40 * signals["hy_z"] + 0.30 * signals["nfci_z"] + 0.20 * signals["stlfsi_z"] + 0.10 * signals["vix_z"]
    breadth_health = 0.50 * signals["iwm_rel63_z"] + 0.30 * signals["iwm_rel20_z"] + 0.20 * signals["iwm_dist_z"]
    iwm_fragility = 0.45 * (-signals["iwm_rel63_z"]) + 0.25 * (-signals["iwm_rel20_z"]) + 0.30 * (-signals["iwm_dist_z"])
    iwm_euphoria = 0.50 * signals["iwm_63_z"] + 0.30 * signals["iwm_20_z"] + 0.20 * signals["iwm_dist_z"]

    recession_prob = sigmoid(recession_risk)

    short_risk_on = clamp01(
        sigmoid(
            0.62 * breadth_health
            - 0.48 * credit_stress
            - 0.38 * signals["vix_z"]
            + 0.25 * (signals["fg_short_risk_on"] - 0.5)
            + 0.18 * duration_tailwind
            - 0.18 * front_end_policy
        )
    )
    short_risk_off = clamp01(
        sigmoid(
            0.50 * credit_stress
            + 0.48 * signals["vix_z"]
            + 0.40 * iwm_fragility
            + 0.28 * signals["fg_fear"]
            + 0.18 * front_end_policy
            + 0.08 * long_end_pressure
            - 0.16 * breadth_health
        )
    )
    big_crash = clamp01(
        sigmoid(
            0.48 * credit_stress
            + 0.28 * recession_risk
            + 0.20 * labor_soft
            + 0.14 * iwm_fragility
            + 0.14 * signals["fg_big_crash_overlay"]
            + 0.10 * front_end_policy
            + 0.10 * long_end_pressure
            - 0.10 * breadth_health
        )
    )
    long_risk_on = clamp01(
        sigmoid(
            0.40 * growth_quarterly_axis
            + 0.18 * growth_monthly_axis
            + 0.16 * breadth_health
            + 0.14 * duration_tailwind
            + 0.10 * policy_easing_impulse
            - 0.22 * credit_stress
            - 0.15 * max(inflation_quarterly_axis, 0.0)
            - 0.10 * recession_prob
            - 0.12 * front_end_policy
        )
    )

    speculative_heat = (
        0.28 * max(iwm_euphoria, 0.0)
        + 0.24 * max(alt_beta_signal, 0.0)
        + 0.16 * max(cyclical_style_signal, 0.0)
        + 0.16 * max(signals["fg_greed"], 0.0)
        + 0.16 * max(broad_em_equity_signal, 0.0)
    )
    quality_break = (
        0.28 * max(-crypto_quality_spread, 0.0)
        + 0.24 * max(iwm_fragility, 0.0)
        + 0.24 * max(credit_stress, 0.0)
        + 0.24 * max(front_end_policy, 0.0)
    )
    behavioral_top_score = clamp01(sigmoid(0.95 * speculative_heat + 1.00 * quality_break - 0.30 * breadth_health))
    distribution_risk = clamp01(sigmoid(0.85 * iwm_fragility + 0.55 * credit_stress + 0.35 * max(iwm_euphoria, 0.0) - 0.25 * breadth_health))

    secular_hard_asset_pressure = (
        0.28 * commodity_breadth
        + 0.18 * hard_asset_breadth
        + 0.16 * inflation_quarterly_axis
        + 0.16 * long_end_pressure
        + 0.12 * front_end_policy
        + 0.10 * bear_steepener
    )
    secular_duration_disinflation = (
        0.30 * duration_tailwind
        + 0.20 * policy_easing_impulse
        + 0.20 * (-max(inflation_quarterly_axis, 0.0))
        + 0.15 * (-max(front_end_policy, 0.0))
        + 0.15 * defensive_style_signal
    )

    signals.update(
        {
            # GDP / CPI nowcast axes
            "gdp_nowcast_quarterly": gdp_nowcast_quarterly,
            "gdp_nowcast_monthly": gdp_nowcast_monthly,
            "cpi_nowcast_quarterly": cpi_nowcast_quarterly,
            "cpi_nowcast_monthly": cpi_nowcast_monthly,
            "gdp_nowcast_outquarter": gdp_nowcast_outquarter,
            "cpi_nowcast_outquarter": cpi_nowcast_outquarter,
            "growth_base_effect": growth_base_effect,
            "inflation_base_effect": inflation_base_effect,
            "inflation_market_impulse_monthly": inflation_market_impulse_monthly,
            "inflation_market_impulse_quarterly": inflation_market_impulse_quarterly,
            # Backward-compatible macro axis keys
            "growth_level": growth_quarterly_axis,
            "growth_mom": growth_monthly_axis,
            "inflation_level": inflation_quarterly_axis,
            "inflation_mom": inflation_monthly_axis,
            "growth_level_raw": growth_quarterly_axis,
            "growth_mom_raw": growth_monthly_axis,
            "growth_transition": growth_transition,
            "labor_soft": labor_soft,
            "recession_risk": recession_risk,
            # Policy / rates engine
            "front_end_policy": front_end_policy,
            "belly_policy": belly_policy,
            "long_end_pressure": long_end_pressure,
            "duration_tailwind": duration_tailwind,
            "policy_easing_impulse": policy_easing_impulse,
            "steepening_impulse": steepening_impulse,
            "curve_regime": curve_regime,
            "bear_steepener": bear_steepener,
            # Legacy-compatible aliases used by the UI
            "policy_repricing": front_end_policy,
            "long_end_inflation_stress": long_end_pressure,
            # Market / risk overlays
            "credit_stress": credit_stress,
            "breadth_health": breadth_health,
            "iwm_fragility": iwm_fragility,
            "iwm_euphoria": iwm_euphoria,
            # Macro probabilities and quad scores
            "g_up_monthly": g_up_monthly,
            "g_up_quarterly": g_up_quarterly,
            "i_up_monthly": i_up_monthly,
            "i_up_quarterly": i_up_quarterly,
            "g_up_outquarter": g_up_outquarter,
            "i_up_outquarter": i_up_outquarter,
            "g_up": g_up,
            "g_down": g_down,
            "i_up": i_up,
            "quad_scores_monthly": quad_scores_monthly,
            "quad_scores_quarterly": quad_scores_quarterly,
            "quad_scores_outquarter": quad_scores_outquarter,
            "quad_scores": quad_scores_blend,
            "macro_quad_monthly": max(quad_scores_monthly, key=quad_scores_monthly.get),
            "macro_quad_quarterly": max(quad_scores_quarterly, key=quad_scores_quarterly.get),
            "macro_quad_outquarter": max(quad_scores_outquarter, key=quad_scores_outquarter.get),
            # Risk engines
            "short_risk_on": short_risk_on,
            "short_risk_off": short_risk_off,
            "big_crash": big_crash,
            "long_risk_on": long_risk_on,
            "behavioral_top_score": behavioral_top_score,
            "distribution_risk": distribution_risk,
            "speculative_heat": speculative_heat,
            "quality_break": quality_break,
            "secular_hard_asset_pressure": secular_hard_asset_pressure,
            "secular_duration_disinflation": secular_duration_disinflation,
            "correlation_merge_df": correlation_merge_df,
            "correlation_toggle_df": correlation_toggle_df,
            "cross_family_corr_df": cross_family_corr_df,
            # Components for UI transparency
            "risk_on_components": {
                "breadth_health": breadth_health,
                "credit_stress_inv": -credit_stress,
                "vix_inv": -signals["vix_z"],
                "fear_greed_sweet": signals["fg_short_risk_on"] - 0.5,
                "duration_tailwind": duration_tailwind,
                "front_end_policy_inv": -front_end_policy,
            },
            "risk_off_components": {
                "credit_stress": credit_stress,
                "vix_z": signals["vix_z"],
                "iwm_fragility": iwm_fragility,
                "fg_fear": signals["fg_fear"],
                "front_end_policy": front_end_policy,
                "long_end_pressure": long_end_pressure,
            },
            "big_crash_components": {
                "credit_stress": credit_stress,
                "recession_risk": recession_risk,
                "labor_soft": labor_soft,
                "iwm_fragility": iwm_fragility,
                "fg_big_crash_overlay": signals["fg_big_crash_overlay"],
                "front_end_policy": front_end_policy,
                "long_end_pressure": long_end_pressure,
            },
            "long_risk_on_components": {
                "growth_quarterly": growth_quarterly_axis,
                "growth_monthly": growth_monthly_axis,
                "breadth_health": breadth_health,
                "credit_stress_inv": -credit_stress,
                "inflation_drag": -max(inflation_quarterly_axis, 0.0),
                "duration_tailwind": duration_tailwind,
                "policy_easing_impulse": policy_easing_impulse,
                "front_end_policy_inv": -front_end_policy,
            },
        }
    )
    return signals


def _state_label(score: float, positive_label: str = "Bullish", negative_label: str = "Bearish") -> str:
    if score >= 0.85:
        return f"{positive_label} Trend"
    if score >= 0.35:
        return f"{positive_label} / Improving"
    if score <= -0.85:
        return f"{negative_label} Trend"
    if score <= -0.35:
        return f"{negative_label} / Weakening"
    return "Neutral / Mixed"


def build_formal_signal_state_df(signals: Dict[str, float]) -> pd.DataFrame:
    rows = [
        ("USD", signals.get("usd_signal", 0.0), "Dollar and funding conditions"),
        ("Oil / broad commodities", 0.60 * signals.get("oil21_z", 0.0) + 0.40 * signals.get("dbc_signal", 0.0), "Inflation shock / hard-asset pulse"),
        ("Gold", signals.get("gld_signal", 0.0), "Precious-metals / hedge response"),
        ("2Y front-end", signals.get("front_end_policy", 0.0), "Policy repricing / hawkish-dovish shift"),
        ("5Y belly", signals.get("belly_policy", 0.0), "Medium-term inflation-policy repricing"),
        ("10Y/30Y long-end", signals.get("long_end_pressure", 0.0), "Term premium / duration pressure"),
        ("Duration / TLT", signals.get("duration_tailwind", 0.0), "Long-duration relief or stress"),
        ("Nasdaq / duration equity", signals.get("nasdaq_signal", 0.0), "Growth-duration leadership"),
        ("Broad EM", signals.get("broad_em_equity_signal", 0.0), "Cross-border cyclicality / dollar sensitivity"),
        ("Indonesia / IHSG", signals.get("indo_equity_signal", 0.0), "Local EM / domestic risk appetite"),
        ("BTC", signals.get("btc_signal", 0.0), "Crypto quality leader"),
        ("Alt beta", signals.get("alt_beta_signal", 0.0), "Speculative crypto / narrative beta"),
    ]
    out = []
    for asset, score, why in rows:
        if asset in {"USD", "2Y front-end", "5Y belly", "10Y/30Y long-end"}:
            state = _state_label(score, positive_label="Higher", negative_label="Lower")
        else:
            state = _state_label(score)
        out.append({"Signal": asset, "State": state, "Score": round(float(score), 2), "Why it matters": why})
    return pd.DataFrame(out)


def build_country_engine_df(signals: Dict[str, float], current_quad: str) -> pd.DataFrame:
    rows = [
        ("US", signals.get("us_equity_signal", 0.0), "Core benchmark / quality leadership"),
        ("Europe", signals.get("europe_signal", 0.0), "DM cyclical / FX-sensitive"),
        ("China", signals.get("china_signal", 0.0), "Stimulus / industrial-demand sensitivity"),
        ("Japan", signals.get("japan_signal", 0.0), "Global industrial / yen-policy mix"),
        ("India", signals.get("india_signal", 0.0), "Domestic growth / higher-quality EM"),
        ("Indonesia", signals.get("indo_equity_signal", 0.0), "Commodity + banks + local liquidity"),
        ("Brazil", signals.get("brazil_signal", 0.0), "Commodity exporter / hard-asset beta"),
        ("Australia", signals.get("australia_signal", 0.0), "Commodity / China-sensitive DM"),
        ("Broad EM", signals.get("broad_em_equity_signal", 0.0), "Catch-all EM beta; often weaker than country selection"),
    ]
    adj_rows = []
    for country, raw_score, why in rows:
        bonus = 0.0
        if current_quad == "Q2" and country in {"Brazil", "Australia", "India", "Indonesia"}:
            bonus += 0.15
        if current_quad == "Q3" and country in {"Brazil", "Australia"}:
            bonus += 0.18
        if current_quad == "Q3" and country in {"Broad EM", "Indonesia"}:
            bonus -= 0.10 if country == "Broad EM" else 0.05
        if current_quad == "Q4" and country in {"US", "Japan"}:
            bonus += 0.10
        final = raw_score + bonus
        if final >= 0.55:
            bias = "Strong / favored"
        elif final >= 0.20:
            bias = "Constructive / selective"
        elif final <= -0.55:
            bias = "Weak / avoid"
        elif final <= -0.20:
            bias = "Soft / underweight"
        else:
            bias = "Mixed / watch"
        adj_rows.append({"Country": country, "Bias": bias, "Score": round(float(final), 2), "Why": why})
    return pd.DataFrame(adj_rows).sort_values("Score", ascending=False).reset_index(drop=True)


def build_outquarter_df(signals: Dict[str, float]) -> pd.DataFrame:
    rows = [
        ("GDP nowcast (Monthly)", signals.get("gdp_nowcast_monthly", 0.0)),
        ("GDP nowcast (Quarterly)", signals.get("gdp_nowcast_quarterly", 0.0)),
        ("GDP nowcast (Out-quarter)", signals.get("gdp_nowcast_outquarter", 0.0)),
        ("CPI nowcast (Monthly)", signals.get("cpi_nowcast_monthly", 0.0)),
        ("CPI nowcast (Quarterly)", signals.get("cpi_nowcast_quarterly", 0.0)),
        ("CPI nowcast (Out-quarter)", signals.get("cpi_nowcast_outquarter", 0.0)),
        ("Growth base-effect adjustment", signals.get("growth_base_effect", 0.0)),
        ("Inflation base-effect adjustment", signals.get("inflation_base_effect", 0.0)),
    ]
    return pd.DataFrame(rows, columns=["Module", "Score"])


def behavioral_process_summary(signals: Dict[str, float]) -> Dict[str, object]:
    top = float(signals.get("behavioral_top_score", 0.0))
    dist = float(signals.get("distribution_risk", 0.0))
    if top >= 0.78:
        label = "Behavioral Excess / Blow-off Risk"
    elif top >= 0.58:
        label = "Hot / crowded tape"
    elif top >= 0.38:
        label = "Warm / watch for distribution"
    else:
        label = "No major behavioral excess"
    lines = [
        "Leader-first moves are healthy; low-quality spillover leading too early usually means the move is maturing.",
        "If BTC leadership weakens while alt beta still pushes, or IWM euphoria rises while breadth deteriorates, treat that as a late-stage warning.",
        "If credit stress and front-end yields rise while speculative beta stays hot, that is closer to distribution than healthy rotation.",
    ]
    return {"label": label, "top": round(top * 100, 1), "distribution": round(dist * 100, 1), "lines": lines}


def secular_cycle_summary(signals: Dict[str, float]) -> Dict[str, object]:
    hard = float(signals.get("secular_hard_asset_pressure", 0.0))
    dur = float(signals.get("secular_duration_disinflation", 0.0))
    if hard - dur >= 0.35:
        regime = "Hard-asset / inflation-pressure secular tilt"
    elif dur - hard >= 0.35:
        regime = "Duration / disinflation secular tilt"
    else:
        regime = "Mixed / transition secular tilt"
    lines = [
        "Use this as a backdrop overlay, not as a direct replacement for the quad engine.",
        "Hard-asset secular tilt matters most when commodity breadth, yields, and inflation impulse reinforce each other.",
        "Duration secular tilt matters most when front-end pressure eases, long-end relief broadens, and defensives / quality duration lead.",
    ]
    return {"regime": regime, "hard_asset_score": round(hard, 2), "duration_score": round(dur, 2), "lines": lines}


def render_advanced_process_overlay(signals: Dict[str, float], current_quad: str) -> None:
    st.markdown("### Advanced Process Overlay")
    with st.expander("Open signal states + out-quarter + country engine", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Formal Signal States**")
            st.dataframe(build_formal_signal_state_df(signals), use_container_width=True, hide_index=True)
            beh = behavioral_process_summary(signals)
            st.markdown("**Behavioral / Topping Process**")
            st.markdown(
                f"<div class='section-note'><b>{escape_text(beh['label'])}</b><br>Top score: {beh['top']}/100 | Distribution risk: {beh['distribution']}/100</div>",
                unsafe_allow_html=True,
            )
            for line in beh["lines"]:
                st.write(f"• {line}")
        with c2:
            st.markdown("**Bayesian-lite Out-Quarter Module**")
            st.dataframe(build_outquarter_df(signals), use_container_width=True, hide_index=True)
            st.write("**Out-quarter quad scores:**", {k: round(v, 1) for k, v in signals.get("quad_scores_outquarter", {}).items()})
            st.write("**Out-quarter quad:**", signals.get("macro_quad_outquarter", "N/A"))
            sec = secular_cycle_summary(signals)
            st.markdown("**Secular Commodity / Rates Overlay**")
            st.markdown(
                f"<div class='section-note'><b>{escape_text(sec['regime'])}</b><br>Hard-asset score: {sec['hard_asset_score']} | Duration score: {sec['duration_score']}</div>",
                unsafe_allow_html=True,
            )
            for line in sec["lines"]:
                st.write(f"• {line}")
        st.markdown("**Global / Country Engine**")
        st.dataframe(build_country_engine_df(signals, current_quad), use_container_width=True, hide_index=True)


def reason_lines(signals: Dict[str, float], quad: str, source_key: str = "blend") -> Tuple[List[str], List[str]]:
    ax = transition_axes(signals, source_key)
    gl = ax["gl"]
    gm = ax["gm"]
    il = ax["il"]
    im = ax["im"]
    cs = signals["credit_stress"]
    rr = ax["rr"]

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


def build_paths(signals: Dict[str, float], quad: str, source_key: str = "blend") -> List[Dict[str, object]]:
    # Macro transition engine only. No IWM / Fear & Greed inputs here.
    ax = transition_axes(signals, source_key)
    gl, gm = ax["gl"], ax["gm"]
    il, im = ax["il"], ax["im"]
    ls, rr = ax["ls"], ax["rr"]

    infl_roll = sigmoid(-1.15 * im - 0.55 * il)
    infl_reheat = sigmoid(1.15 * im + 0.55 * il)
    growth_resilient = sigmoid(1.00 * gm + 0.65 * gl - 0.25 * ls)
    growth_roll = sigmoid(-1.10 * gm - 0.55 * gl + 0.35 * ls + 0.20 * rr)
    growth_bottom = sigmoid(0.95 * gm - 0.35 * gl - 0.10 * ls)
    labor_soft = sigmoid(0.90 * ls + 0.25 * rr)
    labor_stable = sigmoid(-0.90 * ls + 0.25 * growth_resilient)
    infl_sticky = sigmoid(0.85 * il + 0.75 * im)
    infl_cooling = sigmoid(-0.85 * il - 0.75 * im)
    commodity_cool = sigmoid(-0.75 * signals["oil21_z"] - 0.75 * signals["breakeven20_z"])
    commodity_rise = sigmoid(0.75 * signals["oil21_z"] + 0.75 * signals["breakeven20_z"])
    growth_weak = sigmoid(-0.85 * gm - 0.65 * gl + 0.25 * rr)

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
        p["source_key"] = source_key
    return raw

def finalize_paths(paths: List[Dict[str, object]], signals: Optional[Dict[str, float]] = None) -> List[Dict[str, object]]:
    final = []
    for p in paths:
        weights = p["weights"]
        raw_score = 100 * sum(r["status_score"] * w for r, w in zip(p["requirements"], weights))
        alignment = path_signal_alignment(signals, p["target"]) if signals is not None else 0.5
        conviction_boost = (alignment - 0.5) * 16.0
        score = max(0.0, min(100.0, raw_score + conviction_boost))
        color = "#22c55e" if score < 60 else "#f59e0b" if score < 75 else "#ef4444"
        final.append({**p, "raw_score": raw_score, "alignment": alignment, "score": score, "color": color})
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
    st.markdown("### Macro Quad Engine")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Blended Regime", str(max(signals["quad_scores"], key=signals["quad_scores"].get)))
    c2.metric("Quarterly Quad", str(signals.get("macro_quad_quarterly", "N/A")))
    c3.metric("Monthly Quad", str(signals.get("macro_quad_monthly", "N/A")))
    c4.metric("GDP Nowcast (Q)", f"{signals['gdp_nowcast_quarterly']:.2f}")
    c5.metric("CPI Nowcast (Q)", f"{signals['cpi_nowcast_quarterly']:.2f}")
    c6.metric("Out-Quarter Quad", str(signals.get("macro_quad_outquarter", "N/A")))
    st.caption(f"Macro roll risk: {signals['growth_transition'] * 100:.0f}% | Behavioral top: {signals['behavioral_top_score'] * 100:.0f}%")

    fg_text = "N/A" if np.isnan(signals["fg_norm"]) else f"{signals['fg_norm'] * 100:.0f}"
    fg_source = "CNN" if fg_info.get("status") == "ok" else "Manual / N.A."
    st.markdown("### Market / Risk Overlay")
    d1, d2, d3, d4, d5, d6 = st.columns(6)
    d1.metric("Credit Stress", f"{signals['credit_stress']:.2f}")
    d2.metric("Breadth / IWM", f"{signals['breadth_health']:.2f}")
    d3.metric("Front-End Policy", f"{signals['front_end_policy']:.2f}")
    d4.metric("Duration Tailwind", f"{signals['duration_tailwind']:.2f}")
    d5.metric("Fear & Greed", fg_text, fg_source)
    d6.metric("Recession Risk", f"{signals['recession_risk']:.2f}")

def render_forecast_summary_row(current_quad: str, current_quad_score: float, validity: str, primary_path: Dict[str, object], driver_label: str) -> None:
    target_quad = str(primary_path["target"])
    target_meta = QUAD_META[target_quad]
    st.markdown("### Forecast Snapshot")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Current Quad", current_quad)
    with c2:
        st.metric("Forecast Bias", current_quad)
        st.caption(driver_label)
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
            <div style='font-weight:800;margin-bottom:8px'>What If / Next Likely → {escape_text(target_meta['name'])}: {escape_text(target_meta['phase'])}</div>
            <div style='margin-bottom:6px'><b>Possible:</b> {escape_text(primary_path.get('possible', ''))}</div>
            <div style='margin-bottom:6px'><b>Likely strong:</b> {escape_text(primary_path.get('winners', ''))}</div>
            <div><b>Likely laggards:</b> {escape_text(primary_path.get('laggards', ''))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_engine_components(signals: Dict[str, float]) -> None:
    st.markdown("### Engine Components")
    macro_left, macro_right = st.columns(2)
    with macro_left:
        with st.expander("Macro Quad Engine", expanded=False):
            st.write("**GDP nowcast (quarterly):**", round(float(signals["gdp_nowcast_quarterly"]), 3))
            st.write("**GDP nowcast (monthly):**", round(float(signals["gdp_nowcast_monthly"]), 3))
            st.write("**CPI nowcast (quarterly):**", round(float(signals["cpi_nowcast_quarterly"]), 3))
            st.write("**CPI nowcast (monthly):**", round(float(signals["cpi_nowcast_monthly"]), 3))
            st.write("**Inflation market impulse (monthly):**", round(float(signals["inflation_market_impulse_monthly"]), 3))
            st.write("**Inflation market impulse (quarterly):**", round(float(signals["inflation_market_impulse_quarterly"]), 3))
            st.write("**Quarterly quad scores:**", {k: round(v, 1) for k, v in signals["quad_scores_quarterly"].items()})
            st.write("**Monthly quad scores:**", {k: round(v, 1) for k, v in signals["quad_scores_monthly"].items()})
            st.write("**Out-quarter quad scores:**", {k: round(v, 1) for k, v in signals["quad_scores_outquarter"].items()})
            st.write("**Blended regime scores:**", {k: round(v, 1) for k, v in signals["quad_scores"].items()})
    with macro_right:
        with st.expander("Policy / Transition Engine", expanded=False):
            st.write("**Growth transition risk:**", round(float(signals["growth_transition"]), 3))
            st.write("**Labor softening:**", round(float(signals["labor_soft"]), 3))
            st.write("**Recession risk:**", round(float(signals["recession_risk"]), 3))
            st.caption("Macro quad remains growth-vs-inflation. 2Y / 5Y / 10Y / 30Y live here as policy, rates, timing, and divergence lenses.")
            st.write("**2Y front-end policy repricing:**", round(float(signals["front_end_policy"]), 3))
            st.write("**5Y belly policy repricing:**", round(float(signals["belly_policy"]), 3))
            st.write("**10Y/30Y long-end pressure:**", round(float(signals["long_end_pressure"]), 3))
            st.write("**Duration tailwind:**", round(float(signals["duration_tailwind"]), 3))
            st.write("**Policy easing impulse:**", round(float(signals["policy_easing_impulse"]), 3))
            st.write("**Steepening impulse:**", round(float(signals["steepening_impulse"]), 3))
            st.write("**Bear steepener risk:**", round(float(signals["bear_steepener"]), 3))
            st.write("**Commodity impulse (oil21 z):**", round(float(signals["oil21_z"]), 3))
            st.write("**Breakeven impulse (20d z):**", round(float(signals["breakeven20_z"]), 3))
            st.write("**USD signal complex:**", round(float(signals["usd_signal"]), 3))
            st.write("**EM FX signal:**", round(float(signals["emfx_signal"]), 3))
            st.write("**Commodity breadth:**", round(float(signals["commodity_breadth"]), 3))
            st.write("**Hard-asset breadth:**", round(float(signals["hard_asset_breadth"]), 3))
            st.write("**Cyclical style signal:**", round(float(signals["cyclical_style_signal"]), 3))
            st.write("**Defensive style signal:**", round(float(signals["defensive_style_signal"]), 3))
            st.write("**Duration market signal:**", round(float(signals["duration_market_signal"]), 3))
            st.write("**Crypto-equity signal:**", round(float(signals["crypto_equity_signal"]), 3))
            st.write("**Behavioral top score:**", round(float(signals["behavioral_top_score"]), 3))
            st.write("**Distribution risk:**", round(float(signals["distribution_risk"]), 3))
            st.write("**Secular hard-asset pressure:**", round(float(signals["secular_hard_asset_pressure"]), 3))
            st.write("**Secular duration/disinflation:**", round(float(signals["secular_duration_disinflation"]), 3))

    risk_names = [
        ("Short Risk-On Engine", signals.get("risk_on_components", {})),
        ("Short Risk-Off Engine", signals.get("risk_off_components", {})),
        ("Big Crash Engine", signals.get("big_crash_components", {})),
        ("Long Risk-On Engine", signals.get("long_risk_on_components", {})),
    ]
    cols = st.columns(2)
    for i, (title, comp) in enumerate(risk_names):
        with cols[i % 2]:
            with st.expander(title, expanded=False):
                for k, v in comp.items():
                    st.write(f"**{k.replace('_', ' ').title()}:** {float(v):.3f}")

def render_meter_cards(signals: Dict[str, float]) -> None:
    st.markdown("### Separate Market / Risk Engines")
    cards = [
        ("Risk-Off Jangka Pendek", signals["short_risk_off"] * 100, "panic / de-risking tactical; driven by credit, vol, IWM fragility, fear", "short_risk_off"),
        ("Risk-On Jangka Pendek", signals["short_risk_on"] * 100, "breadth + sentiment + low vol / low credit stress tactical window", "short_risk_on"),
        ("BIG CRASH", signals["big_crash"] * 100, "systemic stress / recession / credit break engine", "big_crash"),
        ("Risk-On Jangka Panjang", signals["long_risk_on"] * 100, "multi-week to multi-month backdrop; macro growth + breadth - credit drag", "long_risk_on"),
    ]
    cols = st.columns(4)
    for col, (name, score, desc, engine) in zip(cols, cards):
        label = meter_label(score)
        action, action_desc = risk_action(score, engine)
        color = "#22c55e" if score < 35 else "#eab308" if score < 60 else "#f97316" if score < 80 else "#ef4444"
        with col:
            st.markdown(
                f"""
                <div class='soft-card'>
                    <div style='font-size:14px;opacity:.9;margin-bottom:8px'>{escape_text(name)}</div>
                    <div style='font-size:30px;font-weight:900;margin-bottom:8px'>{score:.0f}/100</div>
                    <div style='display:inline-block;padding:4px 10px;border-radius:999px;background:{color};color:#07110f;font-weight:800;font-size:12px;margin-bottom:10px'>{escape_text(label)}</div>
                    <div style='font-size:12px;font-weight:800;margin-bottom:8px'>{escape_text(action)}</div>
                    <div style='font-size:12px;opacity:.8;margin-bottom:8px'>{escape_text(action_desc)}</div>
                    <div style='font-size:12px;opacity:.75'>{escape_text(desc)}</div>
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




def _normalize_item_key(item: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", item.lower()).strip()


def _display_label_from_key(key: str) -> str:
    label_map = {
        "fixed income": "fixed income",
        "usd": "USD",
        "fx": "FX",
        "consumer staples": "consumer staples",
        "health care": "health care",
        "consumer discretionary": "consumer discretionary",
        "communication services": "communication services",
        "small caps": "small caps",
        "reits": "REITs",
        "reit s": "REITs",
        "high beta": "high beta",
        "low beta": "low beta",
        "em dollar debt": "EM dollar debt",
        "high yield credit": "high yield credit",
        "short duration treasuries": "short duration treasuries",
        "treasury belly": "Treasury belly",
        "long bond": "long bond",
        "tips": "TIPS",
    }
    return label_map.get(key, key)


RELATED_ITEM_HINTS = {
    "commodities": ["energy", "gold", "materials", "industrials", "financials", "fixed income", "usd"],
    "energy": ["commodities", "materials", "industrials", "financials", "usd", "fixed income"],
    "gold": ["commodities", "fixed income", "usd", "consumer staples", "health care"],
    "fixed income": ["usd", "financials", "utilities", "reits", "gold"],
    "equities": ["tech", "financials", "industrials", "materials", "energy", "consumer discretionary", "communication services", "consumer staples", "health care", "utilities", "reits"],
    "credit": ["financials", "fixed income", "high yield credit", "leveraged loans", "em dollar debt"],
    "fx": ["usd", "commodities", "energy", "gold"],
    "usd": ["fixed income", "fx", "gold", "commodities", "energy"],
    "utilities": ["fixed income", "reits", "consumer staples", "health care"],
    "reits": ["fixed income", "utilities", "financials"],
    "tech": ["communication services", "consumer discretionary", "financials"],
    "consumer staples": ["health care", "utilities", "fixed income"],
    "health care": ["consumer staples", "utilities", "fixed income"],
    "industrials": ["materials", "financials", "energy", "consumer discretionary"],
    "materials": ["commodities", "energy", "industrials", "financials"],
    "financials": ["industrials", "materials", "consumer discretionary", "fixed income"],
    "consumer discretionary": ["communication services", "tech", "financials", "industrials"],
    "communication services": ["tech", "consumer discretionary"],
    "small caps": ["financials", "industrials", "consumer discretionary"],
    "high beta": ["small caps", "momentum", "consumer discretionary", "tech"],
    "momentum": ["high beta", "tech", "consumer discretionary"],
}

KNOWN_RELATED_KEYS = sorted({
    *RELATED_ITEM_HINTS.keys(),
    "equities", "credit", "commodities", "gold", "fixed income", "fx", "usd", "energy", "utilities", "reits",
    "tech", "consumer staples", "health care", "industrials", "materials", "financials", "consumer discretionary",
    "communication services", "small caps", "high beta", "momentum", "secular growth", "mid caps", "bdcs",
    "convertibles", "high yield credit", "em dollar debt", "leveraged loans", "low beta", "defensives", "value",
    "dividend yield", "tips", "short duration treasuries", "mbs", "treasury belly", "long bond"
})


def _extract_related_keys(item: str, tree: Dict[str, List[str]]) -> List[str]:
    item_key = _normalize_item_key(item)
    related = set(RELATED_ITEM_HINTS.get(item_key, []))
    normalized_lines = []
    for lines in tree.values():
        normalized_lines.extend(_normalize_item_key(line) for line in lines)
    for candidate in KNOWN_RELATED_KEYS:
        if candidate == item_key:
            continue
        for line in normalized_lines:
            if candidate in line:
                related.add(candidate)
                break
    cleaned = [r for r in related if r and r != item_key]
    return sorted(cleaned)


def _fallback_item_tree(item: str, direction: str) -> Dict[str, List[str]]:
    if direction == "winner":
        return {
            "Most Direct": [f"the cleanest, most liquid {item} leaders", f"the names with the strongest relative strength inside {item}"],
            "Secondary / Confirm": [f"high-quality second-line names inside {item}", f"related proxies only after the direct leaders confirm"],
            "Spillover / Late": [f"lower-liquidity and late-chasing names around {item}", f"only the small spillover beneficiaries after the main move is already working"],
            "If leaders cool, watch rotation to": [f"the next quality bucket next to {item}", "broader spillover only if breadth keeps improving"],
        }
    return {
        "Most Direct Shorts / Weakness": [f"the weakest, most crowded, or most rate-sensitive names inside {item}", f"the names already losing relative strength inside {item}"],
        "Secondary Shorts": [f"second-line names that usually weaken after the leaders crack", f"related proxies only after the direct weak leaders confirm"],
        "Spillover / Late Weakness": [f"lower-liquidity names around {item} that often get hit later", "late spillover after the main downside trend is already established"],
        "If downside cools, watch rotation to": ["the next weakest adjacent bucket", "a squeeze first in the most crowded shorts before broader weakness resumes"],
    }


def _specific_item_tree(quad: str, key: str, direction: str) -> Optional[Dict[str, List[str]]]:
    # Winners
    if direction == "winner":
        if key == "energy":
            if quad == "Q3":
                return {
                    "Most Direct": ["upstream oil & gas E&P", "oil-linked exporters / direct crude beta", "integrated majors when oil and USD stay firm"],
                    "Secondary / Confirm": ["refiners and selected oil services after crude leadership is confirmed", "shipping / tanker only if route stress and freight also confirm"],
                    "Spillover / Late": ["coal / metals / local resource spillover", "second-order industrial names that only catch a small energy tailwind"],
                    "If leaders cool, watch rotation to": ["gold and exporter cash-flow names if inflation-hedge demand broadens", "duration / defensives only if growth fear begins to dominate"],
                }
            return {
                "Most Direct": ["upstream oil & gas E&P", "integrated majors", "oil services once crude strength is confirmed"],
                "Secondary / Confirm": ["refiners and fuel logistics", "selected exporter EM / resource-linked local names"],
                "Spillover / Late": ["adjacent resource proxies and lower-liquidity local names", "second-order industrial beneficiaries"],
                "If leaders cool, watch rotation to": ["financials / industrials in Q2", "defensive cash-flow names in Q3"],
            }
        if key == "commodities":
            if quad == "Q3":
                return {
                    "Most Direct": ["oil / direct energy beta", "gold once inflation-hedge demand broadens", "hard-asset exporters"],
                    "Secondary / Confirm": ["industrial metals and selected ags only if breadth confirms", "commodity-linked currencies / exporters"],
                    "Spillover / Late": ["shipping, dry bulk, and second-order resource names", "local resource beta that only gets partial spillover"],
                    "If leaders cool, watch rotation to": ["defensives and duration if growth fear starts to dominate", "stay selective rather than assuming broad EM must join"],
                }
            return {
                "Most Direct": ["the cleanest liquid commodity futures / ETFs", "energy and industrial-metals producers", "resource exporters"],
                "Secondary / Confirm": ["materials, miners, and commodity FX", "selected transport / logistics only after the core move confirms"],
                "Spillover / Late": ["second-order commodity-sensitive local names", "weaker resource beta that only moves after the leaders"],
                "If leaders cool, watch rotation to": ["financials / cyclicals if the reflation move broadens", "gold / defensives if the move turns more stagflationary"],
            }
        if key == "gold":
            return {
                "Most Direct": ["spot / front-month gold exposure", "large liquid gold miners", "royalty / streaming names when the move is clean"],
                "Secondary / Confirm": ["silver and precious-metals miners only after gold leadership confirms", "defensive cash-flow hard-asset equities"],
                "Spillover / Late": ["smaller miners and lower-liquidity precious-metals beta", "country-specific gold proxies that only catch a partial spillover"],
                "If leaders cool, watch rotation to": ["duration if real yields are falling", "energy / exporters if oil is becoming the dominant inflation leg"],
            }
        if key == "fixed income":
            return {
                "Most Direct": ["the part of the Treasury curve that matches the regime first", "liquid duration expressions before lower-quality spread products"],
                "Secondary / Confirm": ["investment grade duration and high-quality rate-sensitive proxies", "only selected spread sectors after Treasuries confirm"],
                "Spillover / Late": ["lower-quality spread products and residual credit beta", "weaker duration-adjacent equities"],
                "If leaders cool, watch rotation to": ["defensives in Q4", "equity beta only when yields and credit both confirm a handoff"],
            }
        if key == "equities":
            return {
                "Most Direct": ["the official best sectors for the active quad", "liquid index leadership and highest-quality sector leaders"],
                "Secondary / Confirm": ["style-factor winners that fit the quad", "country / EM beta only after the US core leaders confirm"],
                "Spillover / Late": ["lower-liquidity sector laggards", "local spillover names that only move after the main equity leadership is obvious"],
                "If leaders cool, watch rotation to": ["the next-best official sector bucket for the same quad", "or toward the next quad's early leaders if the regime is fading"],
            }
        if key == "credit":
            return {
                "Most Direct": ["highest-quality spread products that fit the regime", "liquid credit beta before lower-quality tails"],
                "Secondary / Confirm": ["convertibles / HY / loans only after spreads and rates both confirm", "EM dollar debt only if USD pressure is calm"],
                "Spillover / Late": ["lower-liquidity spread beta", "the weakest carry chasers late in the move"],
                "If leaders cool, watch rotation to": ["equities in risk-on regimes", "Treasuries / cash-flow defensives when stress rises"],
            }
        if key == "fx":
            return {
                "Most Direct": ["the clearest major-currency expression of the active regime", "liquid developed-market FX before EMFX"],
                "Secondary / Confirm": ["selected EMFX only after the dollar and rates backdrop confirms", "commodity FX when commodities are truly leading"],
                "Spillover / Late": ["narrower country-specific FX themes", "the weakest spillover currencies that only move after the majors"],
                "If leaders cool, watch rotation to": ["the next-strongest major cross", "stay selective on EMFX until credit and USD both confirm"],
            }
        if key == "usd":
            return {
                "Most Direct": ["DXY and the clearest USD-major crosses", "USD vs the weakest cyclical / EMFX expressions"],
                "Secondary / Confirm": ["USD vs commodity importers and fragile balance-sheet currencies", "country-specific weak FX only after the broad USD move is confirmed"],
                "Spillover / Late": ["narrow local FX themes", "late spillover to already-weak EMFX"],
                "If leaders cool, watch rotation to": ["duration and defensives in Q4", "selected pro-cyclical FX only when rates and credit both improve"],
            }
        if key == "utilities":
            return {
                "Most Direct": ["regulated utilities and defensive cash-flow names", "high-quality yield defensives"],
                "Secondary / Confirm": ["adjacent defensives and infrastructure-like proxies", "higher-duration utilities only if yields behave"],
                "Spillover / Late": ["smaller / lower-liquidity defensive names", "local defensive utility proxies"],
                "If leaders cool, watch rotation to": ["staples / health care if defense is still needed", "duration if the market shifts harder into Q4"],
            }
        if key == "reit s" or key == "reits":
            return {
                "Most Direct": ["large liquid REITs with clean balance sheets", "property cash-flow vehicles that match the rates backdrop"],
                "Secondary / Confirm": ["higher-beta REIT sub-industries only after the leaders confirm", "rate-sensitive property names with improving liquidity"],
                "Spillover / Late": ["smaller / weaker property beta", "local property names that only catch partial spillover"],
                "If leaders cool, watch rotation to": ["utilities / staples if the move gets more defensive", "banks / cyclicals only when rates and credit clearly improve"],
            }
        if key == "tech":
            return {
                "Most Direct": ["large liquid quality tech", "software / semis only when rates are not fighting the move"],
                "Secondary / Confirm": ["mid-cap growth and platform names after large-cap leadership confirms", "selected quality crypto-beta equities only after tech breadth expands"],
                "Spillover / Late": ["lower-quality growth and narrative beta", "smaller speculative tech only after the quality leaders already worked"],
                "If leaders cool, watch rotation to": ["other official growth winners if the regime still supports risk", "defensives or duration if rates start to dominate against growth"],
            }
        if key == "consumer staples":
            return {
                "Most Direct": ["large liquid staples with pricing power", "cash-flow defensives with stable margins"],
                "Secondary / Confirm": ["household / food / beverage leaders after the primary defensive move is confirmed", "quality dividend defensives"],
                "Spillover / Late": ["smaller and lower-liquidity staples", "local defensive names with only partial spillover"],
                "If leaders cool, watch rotation to": ["health care / utilities if defense remains needed", "quality growth only when a genuine recovery handoff develops"],
            }
        if key == "health care":
            return {
                "Most Direct": ["large liquid health-care defensives", "pharma / managed-care / resilient cash-flow names"],
                "Secondary / Confirm": ["medtech and broader defensive health-care buckets once the main move confirms", "quality biotech only very selectively"],
                "Spillover / Late": ["small / lower-liquidity health-care beta", "story-driven names that only catch late spillover"],
                "If leaders cool, watch rotation to": ["staples / utilities if the regime remains defensive", "quality growth if a clean Q4→Q1 handoff appears"],
            }
        if key == "industrials":
            return {
                "Most Direct": ["large liquid machinery / transports / capital-goods leaders", "cyclical industrials with strong operating leverage"],
                "Secondary / Confirm": ["logistics / rails / selective defense-related industry after the core leaders confirm", "selected local industrial beta"],
                "Spillover / Late": ["lower-liquidity industrial names", "second-order suppliers that only move after the leaders"],
                "If leaders cool, watch rotation to": ["materials / financials in a clean reflation broadening", "defensives if the cycle starts rolling over"],
            }
        if key == "materials":
            return {
                "Most Direct": ["diversified miners and chemicals / materials leaders", "industrial metals producers"],
                "Secondary / Confirm": ["specialty materials and exporter proxies after base metals confirm", "selected local commodity processors"],
                "Spillover / Late": ["weaker small-cap materials beta", "downstream spillover names"],
                "If leaders cool, watch rotation to": ["energy if inflation pressure broadens", "industrials / financials if the growth leg is cleaner"],
            }
        if key == "financials":
            return {
                "Most Direct": ["money-center banks and liquid financial beta", "insurers / brokers when the rates backdrop confirms"],
                "Secondary / Confirm": ["regional / local banks and lender proxies after the primary leaders confirm", "selected IHSG banks on improving domestic risk appetite"],
                "Spillover / Late": ["weaker lenders and lower-quality local financial beta", "property-adjacent finance spillover"],
                "If leaders cool, watch rotation to": ["industrials / materials if reflation is broadening", "defensives if the rates backdrop turns less friendly"],
            }
        if key == "consumer discretionary":
            return {
                "Most Direct": ["large liquid discretionary leaders", "retail / travel / consumer-beta winners with clear earnings leverage"],
                "Secondary / Confirm": ["autos / leisure / domestic-beta names after the leaders confirm", "selected local consumer cyclicals"],
                "Spillover / Late": ["lower-liquidity discretionary beta", "weak-quality consumer names that only squeeze late"],
                "If leaders cool, watch rotation to": ["communication services / tech if risk appetite stays firm", "staples if the consumer cycle weakens"],
            }
        if key == "communication services":
            return {
                "Most Direct": ["large liquid internet / media / platform names", "the communication-services names that behave like quality growth leaders"],
                "Secondary / Confirm": ["advertising / entertainment beta after the primary leaders confirm", "selected mid-cap growth comms names"],
                "Spillover / Late": ["smaller narrative-driven names", "local media spillover"],
                "If leaders cool, watch rotation to": ["tech / discretionary if the growth leg broadens", "defensives if rates begin to bite"],
            }
        if key in {"high beta","momentum","leverage","secular growth","mid caps","bdcs","convertibles","high yield credit","em dollar debt","leveraged loans"}:
            return _fallback_item_tree(item=key, direction=direction)
    else:
        if key in {"fixed income", "usd", "utilities", "consumer staples", "health care", "low beta", "defensives", "value", "dividend yield", "tips", "short duration treasuries", "mbs", "treasury belly", "long bond"}:
            return {
                "Most Direct Shorts / Weakness": [f"the cleanest liquid expressions of weak {key}", f"the names already losing relative strength inside {key}"],
                "Secondary Shorts": [f"second-line {key} proxies after the leaders crack", "adjacent lower-liquidity names after the core move confirms"],
                "Spillover / Late Weakness": ["crowded laggards and smaller names after the core downside is established", "residual late weakness rather than the first clean short"],
                "If downside cools, watch rotation to": ["the active quad's real winners rather than forcing stale shorts", "a squeeze first in the most crowded weak names"],
            }
        if key in {"equities","credit","tech","consumer discretionary","communication services","industrials","materials","financials","energy","commodities","small caps"}:
            return {
                "Most Direct Shorts / Weakness": [f"the most rate-sensitive / cyclically-exposed {key} leaders that have rolled over", f"the crowded prior winners inside {key} once relative strength breaks"],
                "Secondary Shorts": [f"second-line and lower-quality {key} names after the liquid leaders weaken", "adjacent spillover beta once the main downside trend is proven"],
                "Spillover / Late Weakness": ["illiquid and residual beta", "the names that only get hit after broad selling is already obvious"],
                "If downside cools, watch rotation to": ["the next weakest adjacent bucket", "or a handoff into the next regime's winners if macro conditions are changing"],
            }
    return None


def _playbook_item_tree(quad: str, item: str, direction: str) -> Dict[str, List[str]]:
    key = _normalize_item_key(item)
    tree = _specific_item_tree(quad, key, direction)
    return tree if tree is not None else _fallback_item_tree(item, direction)


def render_item_tree(quad: str, item: str, direction: str, depth: int = 0, visited: Optional[set] = None) -> None:
    if visited is None:
        visited = set()
    item_key = _normalize_item_key(item)
    tree = _playbook_item_tree(quad, item, direction)
    for section, lines in tree.items():
        st.markdown(f"<div class='small-muted' style='margin-top:6px;margin-bottom:4px'><b>{escape_text(section)}</b></div>", unsafe_allow_html=True)
        for line in lines:
            st.write(f"• {line}")
    if depth >= 2:
        return
    related_keys = [rk for rk in _extract_related_keys(item, tree) if rk not in visited]
    if related_keys:
        with st.expander("Open sub-matrix / sleeves", expanded=False):
            for rk in related_keys:
                label = _display_label_from_key(rk)
                with st.expander(label, expanded=False):
                    render_item_tree(quad, label, direction, depth=depth + 1, visited=visited | {item_key, rk})
def render_buckets_column(title: str, buckets: Dict[str, Dict[str, List[str]]], quad: str, direction: str, expand_first: bool = True) -> None:
    st.markdown(f"#### {title}")
    first_bucket = True
    for bucket, content in buckets.items():
        with st.expander(bucket, expanded=expand_first and first_bucket):
            first_sub = True
            for subhead, items in content.items():
                with st.expander(subhead, expanded=first_sub):
                    for idx, item in enumerate(items):
                        with st.expander(item, expanded=(idx == 0)):
                            render_item_tree(quad, item, direction)
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
        render_buckets_column("Winners", filter_playbook_buckets("Winners", guide["winners"], official_only=True), quad, "winner")
    with c3:
        render_buckets_column("Losers", filter_playbook_buckets("Losers", guide["losers"], official_only=True), quad, "loser")


def current_fx_overlay(quad: str, signals: Dict[str, float]) -> List[str]:
    oil_hot = float(signals.get("oil21_z", 0.0)) > 0
    usd_stress = float(signals.get("big_crash", 0.0)) > 0.50 or float(signals.get("short_risk_off", 0.0)) > 0.55

    if quad == "Q1":
        lines = [
            "Base FX tilt: underweight USD; prefer pro-growth FX rather than safe havens.",
            "Cleaner expressions: AUD, CAD, NOK, or other cyclical FX when breadth and credit are healthy.",
            "Avoid leaning too hard into CHF/JPY-style defense while Q1 remains intact.",
        ]
        if usd_stress:
            lines.append("If risk-off stress suddenly rises, reduce cyclical-FX aggression even if the macro map still says Q1.")
        return lines

    if quad == "Q2":
        lines = [
            "Base FX tilt: prefer commodity and cyclical FX over defensive reserve currencies.",
            "Cleaner expressions: AUD, CAD, NOK, or exporter FX when commodities confirm the reflation pulse.",
            "USD usually loses relative appeal versus broad reflation winners unless rates and stress re-tighten sharply.",
        ]
        if usd_stress:
            lines.append("If dollar stress reappears through rates/credit, cut reflation-FX size and wait for cleaner confirmation.")
        return lines

    if quad == "Q3":
        lines = [
            "Base FX tilt now: long USD rather than broad EM FX beta.",
            "Clean major-pair expressions: short EUR/USD, short GBP/USD, and long USD/JPY.",
            "Broad EM FX is usually the fragile bucket; only selective commodity-exporter FX deserve tactical attention.",
            "Proxy note: Q3 is a regime, not a promise that every proxy rises together every day.",
            "Oil and USD can lead the move while gold pauses if yields and the dollar are rising faster than inflation-hedge demand.",
        ]
        if oil_hot:
            lines.append("With oil pressure still positive, keep stagflation-sensitive FX expressions cleaner via USD majors instead of broad EM beta.")
        if float(signals.get("usd_signal", 0.0)) > 0.20:
            lines.append("USD signal lattice is confirming the defensive tilt; stay with major-pair USD expressions before reaching for EM FX.")
        return lines

    lines = [
        "Base FX tilt: prefer USD and reserve-currency defense over cyclical FX beta.",
        "Cleaner expressions: long USD versus high-beta or commodity-sensitive FX.",
        "Avoid broad EM FX and growth-sensitive currencies while global growth is decelerating.",
    ]
    if float(signals.get("long_risk_on", 0.0)) > 0.55:
        lines.append("If long-risk-on starts to improve from depressed levels, prepare a watchlist for the next rotation but do not front-run it too early.")
    return lines


def current_em_overlay(quad: str, signals: Dict[str, float]) -> List[str]:
    usd_stress = float(signals.get("big_crash", 0.0)) > 0.50 or float(signals.get("short_risk_off", 0.0)) > 0.55

    if quad == "Q1":
        lines = [
            "Base EM tilt: selective long bias only when the dollar is soft and credit is calm.",
            "Country selection beats generic broad-EM beta.",
            "Cleaner expressions: reform / exporter / demand-sensitive countries rather than weak balance-sheet EM.",
        ]
        if float(signals.get("ihsg_bank_signal", 0.0)) > float(signals.get("ihsg_defensive_signal", 0.0)):
            lines.append("Current IHSG confirmation: banks / domestic liquidity leaders are outperforming defensives, which fits the higher-beta side of Q1.")
        if float(signals.get("ihsg_property_signal", 0.0)) > 0.10:
            lines.append("Rates-sensitive property beta is starting to confirm the recovery side of the tape.")
        if usd_stress:
            lines.append("If USD stress rises, cut broad EM aggression first and keep only the highest-conviction country stories.")
        return lines

    if quad == "Q2":
        lines = [
            "Base EM tilt: selective reflation beneficiaries and commodity exporters.",
            "Cleaner expressions: country ETFs tied to energy, industrial capex, and exporter strength.",
            "Broad EM only works cleanly if USD is not the dominant tightening force.",
        ]
        if float(signals.get("ihsg_commodity_signal", 0.0)) >= max(float(signals.get("ihsg_bank_signal", 0.0)), float(signals.get("ihsg_defensive_signal", 0.0))):
            lines.append("Current IHSG confirmation: commodity / exporter complex is leading the local tape, which is the cleanest reflation read.")
        elif float(signals.get("ihsg_bank_signal", 0.0)) > 0.10:
            lines.append("Current IHSG confirmation: banks are monetizing stronger nominal-growth expectations.")
        if usd_stress:
            lines.append("If dollar / credit stress rises, downgrade broad EM to tactical only.")
        return lines

    if quad == "Q3":
        lines = [
            "Base EM tilt now: underweight broad EM beta.",
            "Cleaner expressions: selective exporters or idiosyncratic reform / stimulus stories only.",
            "Avoid assuming EEM-style broad exposure is a Quad 3 winner.",
        ]
        if float(signals.get("ihsg_commodity_signal", 0.0)) >= max(float(signals.get("ihsg_bank_signal", 0.0)), float(signals.get("ihsg_defensive_signal", 0.0))):
            lines.append("Current IHSG confirmation: exporter / hard-asset names are beating banks and domestic beta, which is the cleanest local Q3 expression.")
        elif float(signals.get("ihsg_defensive_signal", 0.0)) > float(signals.get("ihsg_bank_signal", 0.0)):
            lines.append("Current IHSG confirmation: defensives are holding up better than banks / property, which also fits a tighter Q3 tape.")
        if usd_stress:
            lines.append("If USD-up and risk-off pressure stay firm, keep EM exposure very selective or absent.")
        if float(signals.get("emfx_signal", 0.0)) < -0.15:
            lines.append("EM FX signal is not confirming broad EM risk-taking right now.")
        return lines

    lines = [
        "Base EM tilt: defensive and highly selective only.",
        "Broad EM beta is usually the wrong default expression in a Q4-style slowdown.",
        "Wait for a real bottoming process before leaning back into cyclical EM risk.",
    ]
    if float(signals.get("ihsg_defensive_signal", 0.0)) >= max(float(signals.get("ihsg_bank_signal", 0.0)), float(signals.get("ihsg_commodity_signal", 0.0))):
        lines.append("Current IHSG confirmation: defensives / cash-flow names are the cleaner local expression than cyclicals.")
    if float(signals.get("long_risk_on", 0.0)) > 0.55:
        lines.append("If long-risk-on improves from depressed levels, start building an EM watchlist rather than front-running full beta.")
    return lines


def current_crypto_overlay(quad: str, signals: Dict[str, float]) -> List[str]:
    crash = float(signals.get("big_crash", 0.0))
    sro = float(signals.get("short_risk_off", 0.0))
    lro = float(signals.get("long_risk_on", 0.0))

    if quad == "Q1":
        lines = [
            "Crypto lens: constructive only if breadth, liquidity, and credit are all improving together.",
            "BTC is cleaner than broad alt beta when the risk-on move is still maturing.",
            "Use market signals to size exposure; do not treat crypto as a core Hedgeye quad bucket.",
        ]
        if float(signals.get("crypto_quality_spread", 0.0)) > 0.10:
            lines.append("Current confirmation: BTC quality still leads alt beta, which is usually the healthier sequence early in a crypto risk-on move.")
        if crash > 0.60 or sro > 0.60:
            lines.append("If crash / risk-off pressure stays high, keep crypto tactical and smaller than core macro winners.")
        return lines

    if quad == "Q2":
        lines = [
            "Crypto lens: this is usually the friendliest macro backdrop for BTC and higher-beta digital assets, but only if liquidity actually confirms.",
            "BTC first, then selective liquid alts; avoid assuming every alt deserves equal weight.",
            "Treat blockchain beta and miners as higher-volatility expressions, not substitutes for BTC quality.",
        ]
        if float(signals.get("alt_beta_signal", 0.0)) > float(signals.get("btc_signal", 0.0)) and float(signals.get("short_risk_on", 0.0)) > 0.55:
            lines.append("Current confirmation: liquid alt beta is starting to outrun BTC, which is a stronger reflation / risk-appetite tell.")
        if crash > 0.55:
            lines.append("If crash pressure is still elevated, stay with BTC / liquid beta and avoid low-quality alts.")
        return lines

    if quad == "Q3":
        lines = [
            "Crypto lens now: defensive to bearish by default.",
            "BTC can bounce tactically, but this is not the public-Hedgeye-style regime to treat crypto as a core winner.",
            "Alts, miners, and blockchain-beta equities are the weakest part of the stack when stagflation and crash risk dominate.",
        ]
        if float(signals.get("crypto_quality_spread", 0.0)) > 0.15:
            lines.append("Current confirmation: BTC quality is materially stronger than alt beta, which is the cleaner defensive crypto sequence in Q3.")
        if lro > 0.60 and crash < 0.45:
            lines.append("Only if short-term risk appetite improves meaningfully should you consider tactical BTC exposure; alts still deserve tighter risk limits.")
        if float(signals.get("crypto_equity_signal", 0.0)) < -0.15:
            lines.append("Blockchain/miner beta is not confirming broad crypto risk appetite yet.")
        return lines

    lines = [
        "Crypto lens: wait for a bottoming process rather than forcing buy-and-hold risk into a Q4 slowdown.",
        "BTC is still cleaner than alt beta, but neither is a core Quad 4 winner in Hedgeye's public playbook.",
        "Alts and miners are usually the weakest expressions until macro and liquidity actually turn.",
    ]
    if float(signals.get("btc_signal", 0.0)) > float(signals.get("alt_beta_signal", 0.0)):
        lines.append("Current confirmation: BTC remains the cleaner watchlist candidate than broader alt beta.")
    if lro > 0.60 and crash < 0.40:
        lines.append("If long-risk-on is genuinely recovering, start with BTC watchlist logic before broad alt beta.")
    return lines


def current_rates_note(quad: str, signals: Dict[str, float]) -> List[str]:
    d2 = float(signals.get("dgs2_20_z", 0.0))
    d5 = float(signals.get("dgs5_20_z", 0.0))
    d10 = float(signals.get("dgs10_20_z", 0.0))
    d30 = float(signals.get("dgs30_20_z", 0.0))
    s52 = float(signals.get("curve5s2s_20_z", 0.0))
    s3010 = float(signals.get("curve30s10s_20_z", 0.0))

    if quad == "Q3":
        lines = [
            "2Y = front-end policy / inflation repricing lens. 5Y = belly lens that bridges policy expectations with medium-term inflation pressure.",
            "10Y = broad nominal-conditions lens. 30Y = duration / growth-fear / term-premium lens. Those do not have to move the same way inside Quad 3.",
            "That is why oil can rip while gold stalls or falls for a stretch: if 2Y/5Y and the dollar are doing the leading, the short-end repricing can dominate the tape.",
        ]
        if d2 > 0.35 and d5 > 0.20 and d10 >= 0 and d30 >= 0:
            lines.append("Current scenario bias: front-end and belly inflation-policy repricing are leading, while the long end is not offering clean duration relief yet. Oil-up + USD-up + Gold-soft remains compatible with Q3.")
        elif d2 > 0.20 and d5 > 0.10 and d10 < 0 and d30 < 0:
            lines.append("Current scenario bias: stagflation with growth fear. Front-end stays sticky, but 10Y/30Y are catching a duration bid.")
        elif d2 < -0.20 and d5 < -0.15 and d10 < -0.15 and d30 < -0.15:
            lines.append("Watch for a Q3→Q4 handoff if the whole curve keeps easing while inflation proxies cool.")
        if s52 > 0.30 or s3010 > 0.30:
            lines.append("A steeper 2s5s / 30s10s curve means the rates tape is changing shape; that matters for banks, REITs, housing, and duration even before the quad label changes.")
        return lines

    if quad == "Q2":
        lines = [
            "In Q2, firm 2Y / 5Y / 10Y usually confirm nominal-growth strength and keep pure duration trades on the back foot.",
            "30Y can rise too, especially in a bear-steepening reflation move where commodities and cyclicals lead.",
        ]
        if d2 < -0.20 and d5 < -0.15 and d10 < -0.10:
            lines.append("If front-end and belly yields start easing first, reflation may be fading toward Q1 or Q3.")
        return lines

    if quad == "Q1":
        lines = [
            "In Q1, stable-to-lower 2Y and 5Y fit the cooling-inflation story better than an aggressive front-end backup.",
            "10Y and 30Y can stay mixed, but a disorderly bear steepener is usually not a clean Goldilocks confirmation.",
        ]
        if d2 > 0.30 and d5 > 0.20:
            lines.append("If 2Y and 5Y start backing up sharply, watch for an inflation reheat that can push the regime toward Q2.")
        return lines

    lines = [
        "In Q4, falling 2Y / 5Y / 10Y / 30Y usually confirm the duration-led slowdown / deflation backdrop.",
        "If the long end starts backing up while oil and breakevens reheat, watch for a Q4→Q3 stagflation risk rather than forcing a clean disinflation story.",
    ]
    if d2 > 0.20 and d5 > 0.15 and d10 > 0.15 and d30 > 0.15:
        lines.append("A whole-curve backup would weaken the clean Q4 read and argue for closer monitoring of stagflation risk.")
    return lines

def current_proxy_note(quad: str, signals: Dict[str, float]) -> List[str]:
    if quad == "Q3":
        lines = [
            "Quad maps are expected-value playbooks, not a requirement that oil, gold, bonds, and FX all move in lockstep every session.",
            "A Q3 read stays valid as long as growth is rolling over while inflation pressure is sticky or re-accelerating.",
            "Short-term divergence is normal: oil plus USD can be leading while gold consolidates under stronger dollar / yield pressure.",
            "The key check is whether GDP nowcast is still rolling over while CPI nowcast stays firm; that matters more than forcing all proxies to move together every day.",
            "Cross-check the signal families: USD complex, commodity breadth, curve shape, and style-factor leadership do not all need to agree at once, but they should not tell the exact opposite story for long.",
        ]
    elif quad == "Q4":
        lines = [
            "Q4 usually rewards duration, USD, and defense, but high-volatility bear-market rallies can still happen inside the regime.",
            "Do not confuse a tactical bounce in beta with a confirmed regime change until growth and inflation trajectories actually turn.",
        ]
    elif quad == "Q2":
        lines = [
            "Q2 usually rewards broad reflation, but not every cyclical pocket has to move together every day.",
            "If commodities stop confirming and credit/vol worsen, reflation trades can fade before the macro quad fully changes.",
        ]
    else:
        lines = [
            "Q1 supports risk-on with cooling inflation, but breadth and credit still matter for confirmation.",
            "If defensives suddenly lead and growth breadth narrows, treat it as an early warning rather than a reason to instantly relabel the quad.",
        ]
    return lines + current_rates_note(quad, signals)


def current_proxy_strength_ladder(quad: str, signals: Dict[str, float]) -> List[str]:
    if quad == "Q3":
        base = [
            "Strongest direct regime expressions now: oil / hard-asset impulse, USD strength, and 2Y / 5Y repricing.",
            "Second-order expressions: selective IHSG exporters / defensives, broad EM weakness, and only tactical BTC quality.",
            "Weakest / most fragile spillover: broad EM beta, IHSG property / domestic-beta cyclicals, and broad alt beta.",
        ]
    elif quad == "Q2":
        base = [
            "Strongest direct regime expressions now: commodities, cyclicals, and firm 2Y / 5Y / 10Y nominal-growth confirmation.",
            "Second-order expressions: banks, exporter EM / IHSG beta, and then liquid crypto beta.",
            "Weakest / lagging spillover: bond proxies and deep defensives if reflation breadth stays broad.",
        ]
    elif quad == "Q1":
        base = [
            "Strongest direct regime expressions now: quality growth, banks / cyclicals, improving breadth, and calmer 2Y / 5Y.",
            "Second-order expressions: selected EM / IHSG domestic beta and then cleaner crypto quality.",
            "Weakest / lagging spillover: pure defensives and pure duration winners if growth breadth keeps improving.",
        ]
    else:
        base = [
            "Strongest direct regime expressions now: falling yields / duration, USD, and defensive cash-flow equities.",
            "Second-order expressions: selected defensive IHSG names and only early watchlist work in BTC / banks.",
            "Weakest / lagging spillover: cyclical EM, commodity beta that needs hot nominal growth, and broad alt beta.",
        ]
    return base


def render_ranked_overlay_matrix(title: str, matrix: Dict[str, List[str]]) -> None:
    st.markdown(f"**{title}**")
    for tier, items in matrix.items():
        with st.expander(tier, expanded=False):
            for idx, item in enumerate(items, start=1):
                st.write(f"{idx}. {item}")


def render_compact_driver_compare(states_by_driver: Dict[str, DashboardState], active_driver: str) -> None:
    order = [
        "Monthly (Hedgeye-style current call)",
        "Blended Regime",
        "Quarterly Anchor",
    ]
    st.markdown("**Current Phase Driver Compare**")
    cols = st.columns(3)
    for col, driver in zip(cols, order):
        s = states_by_driver[driver]
        is_active = driver == active_driver
        tone_bg = "#19e68c" if is_active else "#1f2937"
        tone_fg = "#07110f" if is_active else "#d1d5db"
        with col:
            st.markdown(
                f"""
                <div class='soft-card'>
                    <div style='display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:8px'>
                        <div style='font-size:12px;font-weight:800'>{escape_text(driver)}</div>
                        <div style='display:inline-block;padding:3px 8px;border-radius:999px;background:{tone_bg};color:{tone_fg};font-weight:800;font-size:10px'>{'ACTIVE' if is_active else 'COMPARE'}</div>
                    </div>
                    <div style='display:flex;justify-content:space-between;gap:10px;align-items:flex-end;margin-bottom:8px'>
                        <div>
                            <div style='font-size:22px;font-weight:900;line-height:1'>{escape_text(s.quad.current_quad)}</div>
                            <div style='font-size:11px;opacity:.8;margin-top:6px'>{escape_text(QUAD_META[s.quad.current_quad]['phase'])}</div>
                        </div>
                        <div style='text-align:right'>
                            <div style='font-size:11px;opacity:.75'>Fit</div>
                            <div style='font-size:18px;font-weight:900'>{s.quad.fit_score:.0f}</div>
                        </div>
                    </div>
                    <div style='font-size:11px;margin-bottom:3px'><b>Stage:</b> {escape_text(s.stage)}</div>
                    <div style='font-size:11px;margin-bottom:3px'><b>Validity:</b> {escape_text(s.validity)}</div>
                    <div style='font-size:11px'><b>Next:</b> {escape_text(s.primary_path['target'])} ({s.primary_path['score']:.0f})</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_phase_overlay_bundle(quad: str, signals: Dict[str, float], stage: str) -> None:
    c1, c2 = st.columns(2)
    with c1:
        with st.expander("Current FX Overlay", expanded=False):
            for line in current_fx_overlay(quad, signals):
                st.write(f"• {line}")
            render_ranked_overlay_matrix("Proxy Impact Ladder (strongest → spillover)", PROXY_IMPACT_MATRIX[quad])
        with st.expander("Current Emerging Markets Overlay", expanded=False):
            for line in current_em_overlay(quad, signals):
                st.write(f"• {line}")
            render_ranked_overlay_matrix("EM / IHSG Matrix (strongest → spillover)", EM_IHSG_MATRIX[quad])
        with st.expander("Winner / Loser Ladder (strongest → spillover)", expanded=False):
            render_ranked_overlay_matrix("Quad Long / Short Ladder", QUAD_LONG_SHORT_LADDER[quad])
    with c2:
        with st.expander("Proxy / Divergence Note", expanded=False):
            for line in current_proxy_note(quad, signals):
                st.write(f"• {line}")
            for line in current_proxy_strength_ladder(quad, signals):
                st.write(f"• {line}")
        with st.expander("Current Crypto Overlay", expanded=False):
            for line in current_crypto_overlay(quad, signals):
                st.write(f"• {line}")
            render_ranked_overlay_matrix("Crypto Matrix (strongest → spillover)", CRYPTO_MATRIX[quad])
        with st.expander(f"Stage Rotation — {stage}", expanded=False):
            render_ranked_overlay_matrix("Stage Winner / Loser / Rotation Map", STAGE_ROTATION_GUIDE[quad][stage])
        with st.expander(f"Leadership / Handoff Map — {stage}", expanded=False):
            render_ranked_overlay_matrix("Who usually moves first → who takes over → who moves last", LEADERSHIP_ROTATION_MAP[quad][stage])


def render_driver_playbook_compare(states_by_driver: Dict[str, DashboardState], active_driver: str, signals: Dict[str, float]) -> None:
    order = [
        "Monthly (Hedgeye-style current call)",
        "Blended Regime",
        "Quarterly Anchor",
    ]
    with st.expander("Compare driver playbooks / overlays", expanded=False):
        for driver in order:
            s = states_by_driver[driver]
            label = f"{driver} — {s.quad.current_quad} | {QUAD_META[s.quad.current_quad]['phase']} | Fit {s.quad.fit_score:.0f} | Next {s.primary_path['target']} ({s.primary_path['score']:.0f})"
            with st.expander(label, expanded=(driver == active_driver)):
                render_phase_matrix(s.quad.current_quad)
                render_phase_overlay_bundle(s.quad.current_quad, signals, s.stage)

def render_phase_guide(quad: str, signals: Dict[str, float], stage: str, states_by_driver: Dict[str, DashboardState], active_driver: str) -> None:
    st.markdown("### Current Phase")
    render_compact_driver_compare(states_by_driver, active_driver)
    st.markdown(
        f"<div class='section-note'><b>{escape_text(QUAD_META[quad]['name'])}:</b> {escape_text(CURRENT_PHASE_TEXT[quad])}<br><br><b>Playbook lens:</b> Fokus ke winners quad ini, hindari losers yang paling sensitif, lalu monitor what-if / next likely quad sebelum ubah agresi.</div>",
        unsafe_allow_html=True,
    )
    render_phase_matrix(quad)
    render_phase_overlay_bundle(quad, signals, stage)
    render_driver_playbook_compare(states_by_driver, active_driver, signals)


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
            for bucket, content in filter_playbook_buckets("Winners", guide["winners"], official_only=True).items():
                with st.expander(bucket, expanded=False):
                    for subhead, items in content.items():
                        st.write(f"**{subhead}**")
                        for item in items:
                            st.write(f"• {item}")
        with c3:
            st.markdown("**Losers**")
            for bucket, content in filter_playbook_buckets("Losers", guide["losers"], official_only=True).items():
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
        st.caption(f"Path Score {path['score']:.0f}/100 | Raw {path.get('raw_score', path['score']):.0f}/100 | Signal confirm {100*path.get('alignment', 0.5):.0f}%")
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
            st.markdown(f"<div class='tree-box'><b>{escape_text(path['title'])}</b><br><span class='small-muted'>Path Score {path['score']:.0f}/100 | Raw {path.get('raw_score', path['score']):.0f}/100 | Signal confirm {100*path.get('alignment', 0.5):.0f}%</span></div>", unsafe_allow_html=True)
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




def risk_action(score: float, engine: str) -> Tuple[str, str]:
    if engine == "short_risk_off":
        if score >= 75:
            return "RISK OFF NOW", "Kurangi beta agresif, tighten stop, hedge, dan fokus defense."
        if score >= 60:
            return "DE-RISK", "Kurangi size, hindari tambah exposure agresif, pilih quality saja."
        if score >= 45:
            return "WATCH", "Belum full risk-off, tapi jangan terlalu agresif."
        return "NOT DOMINANT", "Risk-off belum dominan."
    if engine == "short_risk_on":
        if score >= 70:
            return "TACTICAL BUY", "Window trading long jangka pendek terbuka; cari entry, bukan FOMO chase."
        if score >= 55:
            return "SELECTIVE BUY", "Boleh nibble / trading long selektif."
        if score >= 45:
            return "NEUTRAL", "Belum ada edge risk-on jangka pendek yang kuat."
        return "STAND DOWN", "Jangan paksa long tactical."
    if engine == "big_crash":
        if score >= 80:
            return "EXIT MARKET / MAX DEFENSE", "Lindungi modal; hindari tambah beta, fokus cash / hedge / defense."
        if score >= 65:
            return "CRASH RISK HIGH", "Jangan treat dip as easy buy; utamakan defense dan likuiditas."
        if score >= 50:
            return "CRASH WATCH", "Belum full crash, tapi market sudah rapuh."
        return "NO CRASH REGIME", "Belum ada sinyal crash dominan."
    if engine == "long_risk_on":
        if score >= 70:
            return "BUY & HOLD WINDOW", "Backdrop mendukung bangun posisi swing / position bertahap."
        if score >= 55:
            return "START ACCUMULATING", "Boleh mulai beli bertahap dan hold selektif."
        if score >= 45:
            return "WAIT / BUILD LIST", "Belum ideal untuk buy & hold agresif."
        return "AVOID HOLDING AGGRESSIVE BETA", "Belum saatnya bangun hold besar."
    return "WATCH", "-"


def overall_market_call(signals: Dict[str, float]) -> Dict[str, str]:
    sro = float(signals["short_risk_on"] * 100.0)
    srf = float(signals["short_risk_off"] * 100.0)
    bc = float(signals["big_crash"] * 100.0)
    lro = float(signals["long_risk_on"] * 100.0)

    if bc >= 80 or (bc >= 70 and srf >= 65):
        return {
            "headline": "KELUAR / MAX DEFENSE",
            "detail": "Probabilitas stress besar sudah terlalu tinggi. Prioritas utama: jaga modal, kecilkan beta, tahan agresi beli.",
        }
    if srf >= 65 or bc >= 60:
        return {
            "headline": "SAATNYA RISK OFF",
            "detail": "Ini fase de-risk: kurangi posisi rapuh, hindari nambah beta agresif, fokus quality / hedge / cash buffer.",
        }
    if lro >= 65 and bc < 40 and srf < 50:
        return {
            "headline": "MULAI BELI & HOLD",
            "detail": "Backdrop multi-week / multi-month cukup sehat untuk akumulasi bertahap dan tahan posisi lebih lama.",
        }
    if sro >= 65 and bc < 45 and srf < 50:
        return {
            "headline": "SAATNYA BELI TACTICAL",
            "detail": "Window risk-on jangka pendek terbuka. Fokus entry taktis dan tetap disiplin karena ini belum tentu buy & hold regime.",
        }
    return {
        "headline": "NETRAL / SELECTIVE",
        "detail": "Belum ada sinyal dominan untuk all-in risk-on atau full risk-off. Pilih setup terbaik dan jaga size.",
    }


def render_market_action_summary(signals: Dict[str, float]) -> None:
    call = overall_market_call(signals)
    st.markdown("### Market Action Summary")
    st.markdown(
        f"""
        <div class='section-note'>
            <div style='font-size:18px;font-weight:900;margin-bottom:8px'>{escape_text(call['headline'])}</div>
            <div>{escape_text(call['detail'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
def render_quad_detail(quad: str, signals: Dict[str, float], current_quad: str, active_scores: Dict[str, float], source_key: str = "blend") -> None:
    meta = QUAD_META[quad]
    paths = sorted(finalize_paths(build_paths(signals, quad, source_key), signals=signals), key=lambda x: x["score"], reverse=True)
    quad_score = float(active_scores[quad])
    primary_path = paths[0]
    validity = classify_validity(quad_score, max(p["score"] for p in paths)) if quad == current_quad else "Watch"
    stage = transition_stage(signals, quad_score, paths, source_key)
    valid_lines, invalid_lines = reason_lines(signals, quad, source_key)

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
        f"<div class='section-note'><b>Risk management if {paths[0]['target']} confirms</b> → lean toward: {escape_text(paths[0]['winners'])}<br><br>"
        f"<b>Alternate if {paths[1]['target']} confirms</b> → lean toward: {escape_text(paths[1]['winners'])}</div>",
        unsafe_allow_html=True,
    )


def build_forecast_tables(signals: Dict[str, float], current_quad: str, active_scores: Dict[str, float], source_key: str = "blend") -> Tuple[pd.DataFrame, pd.DataFrame]:
    active_scores = signals["quad_scores"] if active_scores is None else active_scores
    current_paths = finalize_paths(build_paths(signals, current_quad, source_key), signals=signals)
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
        paths = finalize_paths(build_paths(signals, q, source_key), signals=signals)
        primary = max(paths, key=lambda x: x["score"])
        quad_rows.append(
            {
                "Quad": q,
                "Phase": QUAD_META[q]["phase"],
                "Logic": QUAD_META[q]["logic"],
                "Fit Score": round(float(active_scores[q]), 1),
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


def filter_playbook_buckets(title: str, buckets: Dict[str, Dict[str, List[str]]], official_only: bool = False) -> Dict[str, Dict[str, List[str]]]:
    if not official_only:
        return buckets
    keep = OFFICIAL_WINNER_BUCKETS if title.lower().startswith("winner") else OFFICIAL_LOSER_BUCKETS
    return {k: v for k, v in buckets.items() if k in keep}


def render_playbook_section(title: str, quad: str, buckets: Dict[str, Dict[str, List[str]]], official_only: bool = False) -> None:
    st.markdown(f"### {title}")
    direction = "winner" if title.lower().startswith("winner") else "loser"
    buckets = filter_playbook_buckets(title, buckets, official_only=official_only)
    for bucket, content in buckets.items():
        with st.expander(bucket, expanded=False):
            for subhead, items in content.items():
                with st.expander(subhead, expanded=False):
                    for idx, item in enumerate(items):
                        with st.expander(item, expanded=(idx == 0)):
                            render_item_tree(quad, item, direction)


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

def render_playbook_all_quads(signals: Dict[str, float], current_quad: str, active_scores: Optional[Dict[str, float]] = None, source_key: str = "blend") -> None:
    st.markdown("### Quad Playbook (All Quads)")
    active_scores = signals["quad_scores"] if active_scores is None else active_scores
    current_paths = finalize_paths(build_paths(signals, current_quad, source_key), signals=signals)
    next_likely = max(current_paths, key=lambda x: x["score"])["target"]

    for q in ["Q1", "Q2", "Q3", "Q4"]:
        meta = QUAD_META[q]
        paths = finalize_paths(build_paths(signals, q, source_key), signals=signals)
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
            render_playbook_section("Winners", q, PHASE_GUIDE[q]["winners"], official_only=True)
            render_playbook_section("Losers", q, PHASE_GUIDE[q]["losers"], official_only=True)
            render_ranked_overlay_matrix("EM / IHSG Matrix (strongest → spillover)", EM_IHSG_MATRIX[q])
            render_ranked_overlay_matrix("Crypto Matrix (strongest → spillover)", CRYPTO_MATRIX[q])
            render_ranked_overlay_matrix("Proxy Impact Ladder (strongest → spillover)", PROXY_IMPACT_MATRIX[q])
            render_ranked_overlay_matrix("Winner / Loser Ladder (strongest → spillover)", QUAD_LONG_SHORT_LADDER[q])
            for _stage in ["Early", "Mid", "Late"]:
                with st.expander(f"Stage Rotation — {_stage}", expanded=False):
                    render_ranked_overlay_matrix(f"{_stage} Winner / Loser / Rotation Map", STAGE_ROTATION_GUIDE[q][_stage])
            render_possible_next_playbook(q)



@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news_rss(query: str, max_items: int = 8) -> List[Dict[str, str]]:
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    root = ET.fromstring(r.text)
    items: List[Dict[str, str]] = []
    for item in root.findall('.//item')[:max_items]:
        title = item.findtext('title', default='')
        link = item.findtext('link', default='')
        pub = item.findtext('pubDate', default='')
        desc = re.sub(r'<[^>]+>', '', item.findtext('description', default=''))
        items.append({'title': title, 'link': link, 'published': pub, 'desc': desc})
    return items

def infer_news_case(title: str, desc: str) -> Tuple[str, str, str]:
    text = f"{title} {desc}".lower()
    if any(k in text for k in ["iran", "israel", "hormuz", "middle east", "war", "missile", "gulf", "gas hub"]):
        return (
            "Geopolitical energy shock",
            "Baca sebagai war/oil-risk premium dulu, bukan otomatis broad risk-off tunggal. Direct oil proxies bisa naik sementara gold bisa tertahan kalau dollar dan yields dominan.",
            "Prefer first-order commodity proxies dan hindari broad EM / alt beta kalau USD + 2Y ikut naik.",
        )
    if any(k in text for k in ["private credit", "bdc", "direct lending", "spread", "stress", "default"]):
        return (
            "Credit / funding stress",
            "Ini lebih dekat ke spread / balance-sheet shock. Broad beta bisa kelihatan murah terlalu cepat.",
            "Prioritaskan defense, quality, dan tunggu credit relief sebelum agresif risk-on.",
        )
    if any(k in text for k in ["auction", "refund", "bill issuance", "treasury issuance", "rollover"]):
        return (
            "Funding / auction shock",
            "Front-end dan term funding bisa lebih penting daripada headline growth biasa.",
            "Kalau 2Y naik lebih cepat daripada long-end, treat as front-end stress bukan risk-on setup.",
        )
    if any(k in text for k in ["fed", "powell", "yield", "2-year", "treasury", "rate"]):
        return (
            "Policy / yield shock",
            "2Y, real yields, dan dollar menentukan napas duration, crypto, dan EM.",
            "Kalau 2Y naik cepat, hati-hati duration beta; kalau 2Y turun, lihat relief ladder.",
        )
    if any(k in text for k in ["china", "stimulus", "copper", "iron ore", "steel", "property"]):
        return (
            "China / commodity-demand shock",
            "Biasanya paling kena ke miners, bulk metals, dry bulk, commodity EM, lalu second-order equipment.",
            "Mulai dari first-order miners dulu, baru lihat spillover second-order jika volume confirm.",
        )
    return (
        "General macro headline",
        "Perlu dibaca bareng quad, 2Y, dollar, oil, liquidity, dan credit; jangan trade headline sendirian.",
        "Sinkronkan headline dengan current quad dan current signal families.",
    )

def build_live_news_table(news_items: List[Dict[str, str]]) -> pd.DataFrame:
    rows = []
    for item in news_items:
        case, read, action = infer_news_case(item.get('title', ''), item.get('desc', ''))
        rows.append({
            'Published': item.get('published', ''),
            'Headline': item.get('title', ''),
            'Case': case,
            'Read': read,
            'Positioning hint': action,
            'Link': item.get('link', ''),
        })
    return pd.DataFrame(rows)

def build_dynamic_war_overlay(signals: Dict[str, float], current_quad: str, news_items: List[Dict[str, str]]) -> Dict[str, object]:
    text_blob = ' '.join([(x.get('title','') + ' ' + x.get('desc','')) for x in news_items]).lower()
    geo_news = any(k in text_blob for k in ['iran', 'israel', 'hormuz', 'war', 'missile', 'gulf', 'middle east'])
    oil_up = max(signals.get('oil21_z', 0.0), signals.get('oil63_z', 0.0))
    usd_up = max(signals.get('usd_signal', 0.0), 0.0)
    front_up = max(signals.get('front_end_policy', 0.0), 0.0)
    long_up = max(signals.get('long_end_pressure', 0.0), 0.0)
    gold_breadth = signals.get('hard_asset_breadth', 0.0)
    crash = max(signals.get('big_crash', 0.0), 0.0)
    shock_score = float(np.clip((0.35 if geo_news else 0.0) + 0.20 * max(oil_up, 0.0) + 0.15 * usd_up + 0.15 * front_up + 0.05 * long_up + 0.10 * crash, 0.0, 1.0))
    if shock_score >= 0.70:
        label = 'HIGH WAR PREMIUM'
    elif shock_score >= 0.45:
        label = 'ACTIVE WAR PREMIUM'
    elif shock_score >= 0.25:
        label = 'WATCH WAR PREMIUM'
    else:
        label = 'LOW WAR PREMIUM'

    if oil_up > 0 and (usd_up > 0 or front_up > 0):
        read = 'Oil-up plus dollar/front-end repricing means the war shock is transmitting through inflation and financial conditions, not just through classic safe-haven channels.'
    elif oil_up > 0 and gold_breadth > 0:
        read = 'Oil and hard-asset breadth are confirming together; this is the cleaner hard-asset shock expression.'
    else:
        read = 'War headlines exist, but transmission into oil / dollar / yields is still incomplete or fading.'

    lines = [
        f'Current quad tetap {QUAD_META[current_quad]["name"]}; war headline tidak otomatis mengganti quad inti.',
        'Kalau oil naik sementara gold turun, itu masih konsisten dengan stagflationary / hawkish shock bila dollar dan yields memimpin.',
        'Gunakan headline perang sebagai overlay premium dan transmission test, bukan pengganti GDP/CPI nowcast engine.',
    ]
    if geo_news and oil_up > 0 and (usd_up > 0 or front_up > 0):
        lines.append('Cleanest leadership biasanya: WTI/Brent -> pure upstream -> integrated majors -> selective IHSG resource names; broad EM dan alt beta biasanya tertinggal.')
    elif geo_news and oil_up > 0:
        lines.append('Kalau cuma direct commodity proxies yang hidup, treat second-order names sebagai wait-for-confirmation.')
    else:
        lines.append('Kalau premium perang memudar dan 2Y/USD ikut reda, window rotasi bisa pindah ke duration, Nasdaq, BTC, lalu EM.')
    return {'score': shock_score, 'label': label, 'read': read, 'lines': lines}

def build_active_transmission_df(signals: Dict[str, float]) -> pd.DataFrame:
    oil_imp = max(signals.get('oil21_z', 0.0), signals.get('oil63_z', 0.0), 0.0)
    usd_imp = max(signals.get('usd_signal', 0.0), 0.0)
    fe_imp = max(signals.get('front_end_policy', 0.0), 0.0)
    le_imp = max(signals.get('long_end_pressure', 0.0), 0.0)
    dur_relief = max(signals.get('duration_tailwind', 0.0), 0.0)
    em_imp = max(signals.get('broad_em_equity_signal', 0.0), 0.0)
    ihsg_com = max(signals.get('ihsg_commodity_signal', 0.0), 0.0)
    btc = max(signals.get('btc_signal', 0.0), 0.0)
    rows = [
        ('Oil -> WTI/Brent', oil_imp * 1.25, 'First-order', 'Underlying paling murni untuk oil shock.'),
        ('Oil -> pure upstream oil', oil_imp * 1.10, 'First-order', 'Producer upstream paling cepat menyerap perubahan crude.'),
        ('Oil -> integrated majors', oil_imp * 0.90, 'Strong second-order', 'Masih direct, tapi lebih defensif dari upstream murni.'),
        ('Oil -> IHSG resource proxies', max(oil_imp * 0.55, ihsg_com * 0.95), 'Selective spillover', 'Bisa ikut, tapi tetap kena drag FX/flow lokal.'),
        ('Dollar / front-end -> broad EM negative', max(usd_imp, fe_imp) * 1.10, 'Very high negative', 'Dollar dan front-end stress paling cepat menekan broad EM / IDR.'),
        ('2Y up -> front-end stress', fe_imp * 1.05, 'Very high', 'Paling cepat terasa ke duration, crypto beta, dan risk appetite.'),
        ('30Y up -> long-end pressure', le_imp * 0.95, 'High', 'Long-end pressure sering menekan valuation duration-heavy.'),
        ('Real-yield / duration relief -> Nasdaq', dur_relief * 1.00, 'Very high', 'Nasdaq biasanya paling sensitif ke relief real yields / duration.'),
        ('Real-yield / duration relief -> BTC', max(dur_relief * 0.75, btc * 0.60), 'High', 'BTC sering jadi crypto leader saat conditions membaik.'),
        ('Commodity breadth -> selective EM exporters', max(signals.get('commodity_breadth', 0.0), em_imp * 0.60), 'Conditional', 'Valid hanya jika dollar tidak terlalu dominan.'),
    ]
    df = pd.DataFrame(rows, columns=['Transmission', 'Active Score', 'Rank', 'Why'])
    df['Active Score'] = df['Active Score'].clip(lower=0.0)
    return df.sort_values('Active Score', ascending=False).reset_index(drop=True)

def build_dynamic_what_if_df(signals: Dict[str, float], current_quad: str) -> pd.DataFrame:
    oil_up = max(signals.get('oil21_z', 0.0), signals.get('oil63_z', 0.0)) > 0
    usd_up = signals.get('usd_signal', 0.0) > 0
    front_up = signals.get('front_end_policy', 0.0) > 0
    duration_relief = signals.get('duration_tailwind', 0.0) > 0
    credit_bad = signals.get('credit_stress', 0.0) > 0
    rows = []
    rows.append({
        'What-if': 'War escalates / Strait risk stays tight',
        'Signal combo': 'oil up + USD up + 2Y up',
        'Read': 'Treat as hard-inflation / stagflation overlay, not automatic broad-beta crash.',
        'Works best': 'WTI/Brent, pure upstream, integrated majors, selective IHSG resource names',
        'Usually struggles': 'Broad EM, IDR-sensitive beta, alt beta, weak duration growth',
        'Now': 'Active' if oil_up and (usd_up or front_up) else 'Watch',
    })
    rows.append({
        'What-if': 'War premium fades',
        'Signal combo': 'oil down + USD down + 2Y down',
        'Read': 'Classic relief ladder toward duration and cleaner risk-on.',
        'Works best': 'Nasdaq, BTC, selective EM, quality cyclicals',
        'Usually struggles': 'Late hard-asset chasers and stale war-premium trades',
        'Now': 'Active' if duration_relief and not oil_up and not usd_up else 'Watch',
    })
    rows.append({
        'What-if': 'Oil up but gold soft',
        'Signal combo': 'oil up + front-end stress + strong USD',
        'Read': 'Still logically consistent with shock quad behavior; dollar/yields can suppress gold short-term.',
        'Works best': 'Direct oil chain first, then selective hard assets if breadth widens',
        'Usually struggles': 'Narratives that require every hedge proxy to rise together',
        'Now': 'Active' if oil_up and front_up else 'Watch',
    })
    rows.append({
        'What-if': 'Broad beta tries to bounce without credit relief',
        'Signal combo': 'duration relief but credit still bad',
        'Read': 'Dead-cat / false recovery risk remains high.',
        'Works best': 'Quality over junk; BTC before alt beta; selective over broad EM',
        'Usually struggles': 'Small-cap junk, weak balance sheets, alt basket',
        'Now': 'Active' if duration_relief and credit_bad else 'Watch',
    })
    return pd.DataFrame(rows)

def build_static_reference_df(rows: List[Tuple[str, str]], cols: Tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(cols))

def render_live_news_overlay(signals: Dict[str, float], current_quad: str, news_query: str) -> None:
    st.markdown('### Live News / What-If / Correlation Engine')
    with st.expander('Open live news + what-if + correlation overlay', expanded=False):
        try:
            news_items = fetch_news_rss(news_query, max_items=8)
        except Exception as e:
            news_items = []
            st.warning(f'Live news feed gagal kebaca: {e}')

        overlay = build_dynamic_war_overlay(signals, current_quad, news_items)
        st.markdown(
            f"""
            <div class='section-note'>
                <div style='font-size:18px;font-weight:900;margin-bottom:8px'>{escape_text(overlay['label'])}</div>
                <div>{escape_text(overlay['read'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for line in overlay['lines']:
            st.write(f'• {line}')

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('**Live News Feed**')
            news_df = build_live_news_table(news_items)
            if not news_df.empty:
                st.dataframe(news_df.drop(columns=['Link']), use_container_width=True, hide_index=True)
                for idx, row in news_df.head(5).iterrows():
                    st.markdown(f"[{escape_text(row['Headline'])}]({row['Link']})")
            else:
                st.caption('Belum ada headline yang kebaca sekarang.')
            st.markdown('**Dynamic What-If Scenarios**')
            st.dataframe(build_dynamic_what_if_df(signals, current_quad), use_container_width=True, hide_index=True)
        with c2:
            st.markdown('**Current Transmission / Correlation Map**')
            st.dataframe(build_active_transmission_df(signals), use_container_width=True, hide_index=True)
            st.markdown('**Static Scenario Reference**')
            st.dataframe(pd.DataFrame(WHAT_IF_SCENARIO_MATRIX, columns=['Case', 'Read', 'Direct winners', 'Likely strugglers']), use_container_width=True, hide_index=True)

        d1, d2 = st.columns(2)
        with d1:
            st.markdown('**Deep Correlation Audit — Merge these first**')
            merge_df = signals.get('correlation_merge_df', pd.DataFrame())
            if isinstance(merge_df, pd.DataFrame) and not merge_df.empty:
                st.dataframe(merge_df, use_container_width=True, hide_index=True)
            else:
                st.caption('Belum ada family yang cukup rapat untuk digabung penuh pada data sekarang.')
            with st.expander('Open moderate / conditional correlations', expanded=False):
                toggle_df = signals.get('correlation_toggle_df', pd.DataFrame())
                if isinstance(toggle_df, pd.DataFrame) and not toggle_df.empty:
                    st.dataframe(toggle_df, use_container_width=True, hide_index=True)
                else:
                    st.caption('Belum ada moderate-correlation family yang menonjol sekarang.')
                pair_df = signals.get('cross_family_corr_df', pd.DataFrame())
                if isinstance(pair_df, pd.DataFrame) and not pair_df.empty:
                    st.markdown('**Cross-Family Correlation Watchlist**')
                    st.dataframe(pair_df, use_container_width=True, hide_index=True)
            st.markdown('**Divergence Rules**')
            st.dataframe(build_static_reference_df(DIVERGENCE_RULES, ('Case', 'Why')), use_container_width=True, hide_index=True)
            st.markdown('**Regime Switch Archetypes**')
            st.dataframe(build_static_reference_df(REGIME_SWITCH_ARCHETYPES, ('Archetype', 'Read')), use_container_width=True, hide_index=True)
        with d2:
            st.markdown('**Correlation / Transmission Priors**')
            st.dataframe(pd.DataFrame(CORRELATION_TRANSMISSION_PRIORS, columns=['Chain', 'Impact', 'Why']), use_container_width=True, hide_index=True)
            st.markdown('**False Recovery / Crash Map**')
            st.dataframe(build_static_reference_df(FALSE_RECOVERY_MAP, ('Trap', 'Why')), use_container_width=True, hide_index=True)
            st.dataframe(pd.DataFrame(CRASH_TYPES, columns=['Crash type', 'Read']), use_container_width=True, hide_index=True)
            st.dataframe(build_static_reference_df(CRASH_RECOVERY_ORDER, ('Crash family', 'Recovery order')), use_container_width=True, hide_index=True)



def driver_short_label(driver: str) -> str:
    if driver.startswith("Monthly"):
        return "Current Phase (Monthly)"
    if driver.startswith("Quarterly"):
        return "Current Phase (Quarterly)"
    return "Blended Phase"


def driver_compare_note(driver: str, states_by_driver: Dict[str, DashboardState]) -> str:
    if driver.startswith("Blended"):
        m = states_by_driver["Monthly (Hedgeye-style current call)"].quad.current_quad
        q = states_by_driver["Quarterly Anchor"].quad.current_quad
        if m == q:
            return f"Blend sekarang hampir sama dengan monthly/quarterly karena keduanya sama-sama {m}."
        return f"Blend sekarang beda karena monthly = {m} sementara quarterly = {q}; blended menjelaskan titik tengah / handoff di antara dua source itu."
    if driver.startswith("Monthly"):
        return "Ini baca cuaca/taktikal sekarang ala monthly call — paling cepat nangkep perubahan jangka pendek."
    return "Ini anchor iklim dominan / backdrop yang lebih lambat berubah dan dipakai buat cek apakah monthly cuma noise atau benar-benar handoff." 


def render_compact_playbook_buckets(quad: str, which: str) -> None:
    direction = "winner" if which.lower().startswith("winner") else "loser"
    buckets = PHASE_GUIDE[quad]["winners" if direction == "winner" else "losers"]
    buckets = filter_playbook_buckets(which, buckets, official_only=True)
    for bucket, content in buckets.items():
        with st.expander(bucket, expanded=False):
            for subhead, items in content.items():
                with st.expander(subhead, expanded=False):
                    for idx, item in enumerate(items):
                        with st.expander(item, expanded=(idx == 0)):
                            render_item_tree(quad, item, direction)


def render_phase_consistency_bundle(driver: str, state: DashboardState, signals: Dict[str, float], states_by_driver: Dict[str, DashboardState]) -> None:
    quad = state.quad.current_quad
    stage = state.stage
    st.caption(driver_compare_note(driver, states_by_driver))
    for section, items in PHASE_GUIDE[quad]["meaning"].items():
        with st.expander(section, expanded=(section == "Macro")):
            for item in items:
                st.write(f"• {item}")
    with st.expander("Winners", expanded=False):
        render_compact_playbook_buckets(quad, "Winners")
    with st.expander("Losers", expanded=False):
        render_compact_playbook_buckets(quad, "Losers")
    with st.expander("Rates / Policy Lens", expanded=False):
        for line in current_rates_note(quad, signals):
            st.write(f"• {line}")
    with st.expander("Current FX Overlay", expanded=False):
        for line in current_fx_overlay(quad, signals):
            st.write(f"• {line}")
    with st.expander("Current Emerging Markets Overlay", expanded=False):
        for line in current_em_overlay(quad, signals):
            st.write(f"• {line}")
        render_ranked_overlay_matrix("EM / IHSG Matrix (strongest → spillover)", EM_IHSG_MATRIX[quad])
    with st.expander("Current Crypto Overlay", expanded=False):
        for line in current_crypto_overlay(quad, signals):
            st.write(f"• {line}")
        render_ranked_overlay_matrix("Crypto Matrix (strongest → spillover)", CRYPTO_MATRIX[quad])
    with st.expander("Proxy / Divergence Note", expanded=False):
        for line in current_proxy_note(quad, signals):
            st.write(f"• {line}")
        for line in current_proxy_strength_ladder(quad, signals):
            st.write(f"• {line}")
        render_ranked_overlay_matrix("Proxy Impact Ladder (strongest → spillover)", PROXY_IMPACT_MATRIX[quad])
    with st.expander("Winner / Loser Ladder (strongest → spillover)", expanded=False):
        render_ranked_overlay_matrix("Quad Long / Short Ladder", QUAD_LONG_SHORT_LADDER[quad])
    with st.expander(f"Stage Rotation — {stage}", expanded=False):
        render_ranked_overlay_matrix("Stage Winner / Loser / Rotation Map", STAGE_ROTATION_GUIDE[quad][stage])
    with st.expander(f"Leadership / Handoff Map — {stage}", expanded=False):
        render_ranked_overlay_matrix("Who usually moves first → who takes over → who moves last", LEADERSHIP_ROTATION_MAP[quad][stage])


def render_driver_triptych(states_by_driver: Dict[str, DashboardState], signals: Dict[str, float], active_driver: str) -> None:
    st.markdown("### Current Phase Compare")
    order = [
        "Monthly (Hedgeye-style current call)",
        "Quarterly Anchor",
        "Blended Regime",
    ]
    cols = st.columns(3)
    for col, driver in zip(cols, order):
        s = states_by_driver[driver]
        q = s.quad.current_quad
        meta = QUAD_META[q]
        is_active = driver == active_driver
        tag_bg = "#19e68c" if is_active else "#20304d"
        tag_fg = "#07110f" if is_active else "#dbeafe"
        border = "1.5px solid rgba(0,255,200,.85)" if is_active else "1px solid rgba(255,255,255,.10)"
        with col:
            st.markdown(
                f"""
                <div style='border:{border};border-radius:18px;padding:16px 16px 14px 16px;background:linear-gradient(135deg, rgba(3,20,18,0.96), rgba(6,16,33,0.96));min-height:218px'>
                    <div style='display:flex;justify-content:space-between;gap:10px;align-items:flex-start'>
                        <div style='font-size:14px;font-weight:800;line-height:1.25'>{escape_text(driver_short_label(driver))}</div>
                        <div style='display:inline-block;padding:4px 8px;border-radius:999px;background:{tag_bg};color:{tag_fg};font-weight:800;font-size:10px'>{'ACTIVE' if is_active else 'COMPARE'}</div>
                    </div>
                    <div style='margin-top:14px;font-size:34px;font-weight:900;line-height:1'>{escape_text(q)}</div>
                    <div style='margin-top:6px;font-weight:700'>{escape_text(meta['phase'])}</div>
                    <div style='margin-top:12px;display:grid;grid-template-columns:1fr auto;gap:8px 12px;font-size:13px'>
                        <div><b>Stage:</b> {escape_text(s.stage)}</div>
                        <div style='text-align:right'><span style='opacity:.75'>Fit</span> <b>{s.quad.fit_score:.0f}</b></div>
                        <div><b>Validity:</b> {escape_text(s.validity)}</div>
                        <div style='text-align:right'><span style='opacity:.75'>Next</span> <b>{escape_text(s.primary_path['target'])} ({s.primary_path['score']:.0f})</b></div>
                    </div>
                    <div style='margin-top:12px;font-size:12px;opacity:.82'>{escape_text(meta['logic'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            toggle_key = f"toggle_consist_{re.sub(r'[^a-z0-9]+', '_', driver.lower())}"
            if st.toggle("Open consist", value=False, key=toggle_key):
                render_phase_consistency_bundle(driver, s, signals, states_by_driver)

def render_driver_path_compare(states_by_driver: Dict[str, DashboardState]) -> None:
    st.markdown("### Path to Q?")
    order = [
        "Monthly (Hedgeye-style current call)",
        "Quarterly Anchor",
        "Blended Regime",
    ]

    def _path_card(path: Dict[str, object], compact: bool = False) -> str:
        status_html = ""
        reqs = list(path.get("requirements", []))[:3]
        for req in reqs:
            icon = escape_text(req.get("icon", "•"))
            label = escape_text(req.get("label", "-"))
            name = escape_text(req.get("name", "-"))
            status_html += f"<div style='display:flex;justify-content:space-between;gap:8px;margin-top:8px'><div style='font-size:13px'>{name}</div><div style='font-weight:800;font-size:12px;opacity:.95'>{icon} {label}</div></div>"
        score = float(path.get("score", 0))
        winners = escape_text(path.get("winners", "-"))
        laggards = escape_text(path.get("laggards", "-"))
        possible = escape_text(path.get("possible", "-"))
        title = escape_text(path.get("title", path.get("target", "Path")))
        if_right = escape_text(path.get("if_right", "-"))
        bar_w = max(6, min(100, int(round(score))))
        extra = f"<div style='margin-top:10px'><b>Possible:</b> {possible}</div><div style='margin-top:8px'><b>If forecast benar:</b> {if_right}</div><div style='margin-top:8px'><b>Likely strong:</b> {winners}</div><div style='margin-top:8px'><b>Likely laggards:</b> {laggards}</div>"
        if compact:
            extra = f"<div style='margin-top:10px'><b>Possible:</b> {possible}</div><div style='margin-top:8px'><b>Strong:</b> {winners}</div><div style='margin-top:8px'><b>Laggards:</b> {laggards}</div>"
        return f"""
        <div class='main-card' style='padding:16px 16px 14px 16px;margin-top:12px'>
            <div style='display:flex;justify-content:space-between;gap:12px;align-items:flex-start'>
                <div style='font-size:17px;font-weight:900;line-height:1.25'>{title}</div>
                <div style='font-size:13px;font-weight:900;white-space:nowrap'>{score:.0f}/100</div>
            </div>
            {status_html}
            <div style='margin-top:12px;font-size:13px;opacity:.78'>Path Score {score:.0f}/100</div>
            <div style='margin-top:8px;width:100%;height:8px;background:rgba(255,255,255,.08);border-radius:999px;overflow:hidden'>
                <div style='height:100%;width:{bar_w}%;background:linear-gradient(90deg,#29a3ff,#3b82f6);border-radius:999px'></div>
            </div>
            {extra}
        </div>
        """

    cols = st.columns(3)
    for col, driver in zip(cols, order):
        s = states_by_driver[driver]
        ordered = sorted(s.current_paths, key=lambda x: x['score'], reverse=True)
        primary = ordered[0]
        alt = ordered[1] if len(ordered) > 1 else None
        with col:
            st.markdown(f"**{escape_text(driver_short_label(driver))}**")
            st.markdown(_path_card(primary), unsafe_allow_html=True)
            if alt is not None:
                with st.expander(f"Alt path: {escape_text(alt['target'])}", expanded=False):
                    st.markdown(_path_card(alt, compact=True), unsafe_allow_html=True)



def build_macro_only_asset_table(signals: Dict[str, float]) -> pd.DataFrame:
    """Macro-only probability / range / timing panel.
    Uses existing quad signals plus market overlays already loaded in the app.
    It is intentionally probabilistic, not a point forecast engine.
    """
    G = float(np.clip(0.30*signals.get('wei_4w_z',0)+0.25*signals.get('wei_13w_z',0)-0.20*signals.get('claims_13w_z',0)-0.15*signals.get('sahm_13w_z',0)-0.10*signals.get('recpro_13w_z',0), -3, 3))
    I = float(np.clip(0.24*signals.get('cpi3_z',0)+0.20*signals.get('core3_z',0)+0.16*signals.get('cpi_gap_z',0)+0.14*signals.get('core_gap_z',0)+0.12*signals.get('breakeven20_z',0)+0.14*signals.get('oil21_z',0), -3, 3))
    R = float(np.clip(0.30*signals.get('sahm_z',0)+0.25*signals.get('recpro_z',0)+0.20*signals.get('claims_26w_z',0)+0.15*signals.get('hy_z',0)+0.10*signals.get('vix_z',0), -3, 3))
    Y = float(np.clip(0.45*signals.get('dgs10_20_z',0)+0.35*signals.get('dgs30_20_z',0)+0.20*signals.get('dgs5_20_z',0), -3, 3))
    D = float(np.clip(0.70*signals.get('usd_signal',0)+0.30*(signals.get('dgs2_20_z',0)+signals.get('dgs5_20_z',0))/2, -3, 3))
    F = float(np.clip(-0.45*signals.get('hy_z',0)-0.25*signals.get('nfci_z',0)-0.20*signals.get('stlfsi_z',0)-0.10*signals.get('vix_z',0), -3, 3))
    V = float(np.clip(0.65*signals.get('vix_z',0)+0.35*signals.get('stlfsi_z',0), -3, 3))
    surprise = abs(signals.get('cpi_gap_z',0))+abs(signals.get('core_gap_z',0))+0.5*abs(signals.get('wei_4w_z',0)-signals.get('wei_13w_z',0))
    breadth = np.mean([
        1 if signals.get('cpi3_z',0) > 0 else -1,
        1 if signals.get('core3_z',0) > 0 else -1,
        1 if signals.get('wei_4w_z',0) > 0 else -1,
        1 if -signals.get('claims_13w_z',0) > 0 else -1,
        1 if signals.get('breakeven20_z',0) > 0 else -1,
    ])
    persistence = np.mean([
        np.sign(signals.get('wei_4w_z',0))*np.sign(signals.get('wei_13w_z',0)),
        np.sign(signals.get('cpi3_z',0))*np.sign(signals.get('cpi6_z',0)),
        np.sign(signals.get('core3_z',0))*np.sign(signals.get('core6_z',0)),
    ])
    E = float(np.clip(0.55*surprise + 0.45*max(abs(breadth), 0), 0, 3))
    S = float(np.clip(0.60*signals.get('oil21_z',0)+0.20*signals.get('oil63_z',0)+0.20*signals.get('commodity_breadth',0), -3, 3))
    CM = float(np.clip(0.60*signals.get('btc_signal',0)+0.40*signals.get('crypto_major_signal',0), -3, 3))
    CV = float(np.clip(0.60*signals.get('alt_beta_signal',0)-0.40*signals.get('vix_z',0), -3, 3))

    cfg = {
        'Gold': {'score': (-0.30*Y + 0.24*V + 0.16*I + 0.14*R - 0.10*D + 0.06*E), 'cap': 0.80, 'vol20': 0.07, 'k': 0.90, 'timing_bias': 1.05},
        'Oil': {'score': (0.60*(0.24*G + 0.12*I - 0.14*R - 0.10*D + 0.10*E) + 0.40*S), 'cap': 0.70 if abs(S) > 0.35 else 0.45, 'vol20': 0.12, 'k': 1.40, 'timing_bias': 0.85},
        'Long Bonds': {'score': (-0.28*G - 0.26*I + 0.20*R - 0.14*Y + 0.07*V + 0.05*E), 'cap': 0.85, 'vol20': 0.05, 'k': 0.80, 'timing_bias': 1.10},
        'SPX': {'score': (0.30*G - 0.16*I - 0.14*R - 0.12*Y + 0.12*F + 0.08*E - 0.06*D), 'cap': 0.65, 'vol20': 0.08, 'k': 1.00, 'timing_bias': 0.95},
        'IWM': {'score': (0.34*G - 0.18*I - 0.18*R + 0.16*F - 0.10*D - 0.04*Y + 0.06*E), 'cap': 0.60, 'vol20': 0.11, 'k': 1.20, 'timing_bias': 0.90},
        'DXY': {'score': (0.32*D + 0.16*R - 0.14*G - 0.12*F - 0.08*I + 0.06*E), 'cap': 0.80, 'vol20': 0.04, 'k': 0.80, 'timing_bias': 1.05},
        'Broad EM': {'score': (0.20*G + 0.16*F - 0.18*D - 0.16*R + 0.12*signals.get('broad_em_equity_signal',0) + 0.08*signals.get('emfx_signal',0)), 'cap': 0.60, 'vol20': 0.10, 'k': 1.10, 'timing_bias': 0.90},
        'IHSG': {'score': (0.18*G + 0.10*F - 0.18*D - 0.14*R + 0.16*signals.get('indo_equity_signal',0) + 0.10*signals.get('ihsg_bank_signal',0) + 0.10*signals.get('ihsg_commodity_signal',0) - 0.06*signals.get('ihsg_property_signal',0)), 'cap': 0.58, 'vol20': 0.11, 'k': 1.10, 'timing_bias': 0.85},
        'BTC': {'score': (0.45*(0.16*F - 0.14*R - 0.10*D + 0.08*G - 0.06*Y + 0.06*E) + 0.35*CM + 0.20*CV), 'cap': 0.65 if (abs(CM) + abs(CV)) > 0.30 else 0.40, 'vol20': 0.18, 'k': 1.80, 'timing_bias': 0.80},
    }

    rows = []
    breadth_score = float(np.clip(50 + 25*breadth + 15*persistence, 5, 95))
    event_strength = float(np.clip(40 + 18*E, 5, 95))
    catalyst_density = 60.0
    for asset, c in cfg.items():
        score = float(np.clip(c['score'], -3, 3))
        win = float(np.clip(100*sigmoid(1.25*score), 5, 95))
        adj_win = float(np.clip(50 + (win-50)*c['cap'], 5, 95))
        timing = float(np.clip((0.40*event_strength + 0.25*breadth_score + 0.20*(50 + 35*persistence) + 0.15*catalyst_density) * c['timing_bias']/100, 0, 100))
        exp20 = score * c['vol20'] * c['k'] * c['cap']
        band = 0.60 * c['vol20']
        base_low = exp20 - band
        base_high = exp20 + band
        best = exp20 + np.sign(score if score != 0 else 1) * (1.20*c['vol20'])
        worst = exp20 - np.sign(score if score != 0 else 1) * (1.20*c['vol20'])
        rows.append({
            'Asset': asset,
            'Score': round(score, 2),
            'Winner %': round(adj_win, 1),
            'Loser %': round(100-adj_win, 1),
            '20D Base': f"{base_low*100:+.1f}% to {base_high*100:+.1f}%",
            '20D Best': f"{best*100:+.1f}%",
            '20D Worst': f"{worst*100:+.1f}%",
            'Timing': ('High' if timing >= 70 else 'Medium' if timing >= 52 else 'Low'),
            'Timing Score': round(timing, 1),
            'Confidence': round(100*c['cap'], 0),
        })
    df = pd.DataFrame(rows)
    df['Rank'] = 0.45*df['Winner %'] + 0.30*df['Timing Score'] + 0.25*df['Confidence']
    return df.sort_values('Rank', ascending=False).drop(columns=['Rank'])


def render_macro_only_asset_panel(signals: Dict[str, float]) -> None:
    st.markdown("### Macro-Only Probability / Range / Timing")
    st.caption("Probabilitas + range + timing window. Bukan target pasti atau exact top/bottom.")
    df = build_macro_only_asset_table(signals)
    topw, topl = st.columns(2)
    with topw:
        st.markdown("**Top Winner Candidates**")
        for _, r in df.head(4).iterrows():
            st.write(f"• {r['Asset']} — {r['Winner %']:.0f}% | {r['20D Base']} | Timing {r['Timing']}")
    with topl:
        st.markdown("**Top Loser / Short Candidates**")
        for _, r in df.sort_values('Loser %', ascending=False).head(4).iterrows():
            st.write(f"• {r['Asset']} — {r['Loser %']:.0f}% | {r['20D Base']} | Timing {r['Timing']}")
    with st.expander("Open per-asset detail", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)

def render_bottom_toggle_sections(signals: Dict[str, float], state: DashboardState, news_query: str, states_by_driver: Optional[Dict[str, DashboardState]] = None) -> None:
    st.markdown("### Bottom Toggles")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        show_selected = st.toggle("Selected-driver detail", value=False, key="bottom_selected_driver")
    with t2:
        show_playbook = st.toggle("Quad playbook", value=False, key="bottom_quad_playbook")
    with t3:
        show_news = st.toggle("Live news / what-if / correlation", value=False, key="bottom_live_news")
    with t4:
        show_advanced = st.toggle("Advanced process", value=False, key="bottom_advanced_process")

    if show_selected:
        if states_by_driver:
            with st.expander("Current Phase Compare Detail (Monthly / Quarterly / Blended)", expanded=False):
                for driver in ["Monthly (Hedgeye-style current call)", "Quarterly Anchor", "Blended Regime"]:
                    s = states_by_driver[driver]
                    st.markdown(f"#### {escape_text(driver_short_label(driver))} — {escape_text(s.quad.current_quad)} / {escape_text(QUAD_META[s.quad.current_quad]['phase'])}")
                    render_quad_detail(s.quad.current_quad, signals, s.quad.current_quad, s.quad.active_scores, s.quad.source_key)
                    st.markdown("---")
        else:
            render_quad_detail(state.quad.current_quad, signals, state.quad.current_quad, state.quad.active_scores, state.quad.source_key)
    if show_playbook:
        render_playbook_all_quads(signals, state.quad.current_quad, state.quad.active_scores, state.quad.source_key)
        render_macro_only_asset_panel(signals)
    if show_news:
        render_live_news_overlay(signals, state.quad.current_quad, news_query)
    if show_advanced:
        render_advanced_process_overlay(signals, state.quad.current_quad)

def main() -> None:
    inject_css()
    st.title("Macro Quad Transition Dashboard")
    st.caption(
        "Struktur utama: Macro Quad Engine → Separate Market / Risk Engines → Condition Sekarang Bagusnya? → Current Phase Compare (Monthly / Quarterly / Blended) → Path to Q? → bottom toggles."
    )

    st.sidebar.header("Settings")
    try:
        default_key = st.secrets.get("FRED_API_KEY", "")
    except Exception:
        default_key = ""

    defaults = {
        "cfg_fred_key": default_key,
        "cfg_fg_mode": "CNN Auto",
        "cfg_manual_fg": 50,
        "cfg_show_raw": False,
        "cfg_quad_driver": "Monthly (Hedgeye-style current call)",
        "cfg_news_query": DEFAULT_NEWS_QUERY,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    with st.sidebar.form("settings_form", clear_on_submit=False):
        fred_key_in = st.text_input("FRED API Key", value=st.session_state["cfg_fred_key"], type="password")
        fg_mode_in = st.radio("Fear & Greed Source", ["CNN Auto", "Manual Override"], index=0 if st.session_state["cfg_fg_mode"] == "CNN Auto" else 1)
        manual_fg_in = st.number_input("Manual Fear & Greed (0-100)", min_value=0, max_value=100, value=int(st.session_state["cfg_manual_fg"]))
        show_raw_in = st.checkbox("Show raw signal table", value=bool(st.session_state["cfg_show_raw"]))
        quad_driver_in = st.selectbox(
            "Current Quad Driver",
            ["Monthly (Hedgeye-style current call)", "Blended Regime", "Quarterly Anchor"],
            index=["Monthly (Hedgeye-style current call)", "Blended Regime", "Quarterly Anchor"].index(st.session_state["cfg_quad_driver"]),
        )
        news_query_in = st.text_input("Live News Query", value=st.session_state["cfg_news_query"])
        applied = st.form_submit_button("Apply settings")

    if applied:
        st.session_state["cfg_fred_key"] = fred_key_in
        st.session_state["cfg_fg_mode"] = fg_mode_in
        st.session_state["cfg_manual_fg"] = int(manual_fg_in)
        st.session_state["cfg_show_raw"] = bool(show_raw_in)
        st.session_state["cfg_quad_driver"] = quad_driver_in
        st.session_state["cfg_news_query"] = news_query_in

    fred_key = st.session_state["cfg_fred_key"]
    fg_mode = st.session_state["cfg_fg_mode"]
    manual_fg = int(st.session_state["cfg_manual_fg"])
    show_raw = bool(st.session_state["cfg_show_raw"])
    quad_driver = st.session_state["cfg_quad_driver"]
    news_query = st.session_state["cfg_news_query"]
    st.sidebar.caption("Sidebar sekarang pakai Apply settings supaya nggak rerun tiap kali ketik/ganti opsi.")

    if not fred_key:
        st.warning("Masukin FRED API key dulu di sidebar, lalu klik Apply settings.")
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

    driver_order = ["Monthly (Hedgeye-style current call)", "Blended Regime", "Quarterly Anchor"]
    states_by_driver = {driver: build_dashboard_state(signals, fg_info, driver) for driver in driver_order}
    state = states_by_driver[quad_driver]

    overview_metrics(signals, fg_info)
    render_countdown_cards(fred_key)
    st.markdown("---")

    render_meter_cards(signals)
    st.markdown("---")

    st.markdown("### Condition Sekarang Bagusnya?")
    render_market_action_summary(signals)
    st.markdown("---")

    render_driver_triptych(states_by_driver, signals, quad_driver)
    st.markdown("---")

    render_driver_path_compare(states_by_driver)
    st.markdown("---")

    render_bottom_toggle_sections(signals, state, news_query, states_by_driver)

    if show_raw:
        st.markdown("### Raw Signal Table")
        raw_signal_table(signals)

    st.markdown("---")
    st.caption(
        "Performa dibenerin dengan Apply-settings form di sidebar dan detail compare/playbook/news/advanced overlay yang sekarang benar-benar lazy-render lewat expander, jadi bagian berat nggak ikut dirender kalau belum dibuka."
    )



if __name__ == "__main__":
    main()
