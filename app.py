from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Macro Scenario Matrix Final", layout="wide")

BASE_URL = "https://api.stlouisfed.org/fred"
DEFAULT_START = "2010-01-01"
DEFAULT_NEWS_QUERY = "Iran war oil Strait of Hormuz private credit treasury auction rollover"

SERIES = {
    "growth_conf": ["BSCICP02USM460S"],
    "indpro": ["INDPRO"],
    "payrolls": ["PAYEMS"],
    "unrate": ["UNRATE"],
    "cpi": ["CPIAUCSL"],
    "core_cpi": ["CPILFESL"],
    "pce": ["PCEPI"],
    "breakeven10y": ["T10YIE"],
    "us2y": ["DGS2"],
    "us10y": ["DGS10"],
    "us30y": ["DGS30"],
    "real10y": ["DFII10"],
    "dxy": ["DTWEXBGS"],
    "gold": ["GOLDAMGBD228NLBM", "GOLDPMGBD228NLBM"],
    "walcl": ["WALCL"],
    "rrp": ["RRPONTSYD"],
    "hy": ["BAMLH0A0HYM2"],
    "nfci": ["NFCI"],
    "wti": ["DCOILWTICO", "DCOILBRENTEU"],
}
OPTIONAL_SERIES = {"gold"}

QUAD_PLAYBOOK = {
    "Quad 1": {
        "meaning": "Growth naik, inflasi turun. Biasanya ini fase paling ramah buat duration dan risk assets berkualitas.",
        "good": "Nasdaq, quality growth, BTC, broad EM, quality cyclicals",
        "avoid": "Inflation hedge yang tertinggal, defensives mahal, low-quality laggards",
        "early": "Biasanya Nasdaq / quality growth dulu, lalu BTC, lalu EM.",
        "mid": "Risk-on makin luas. Kalau bersih, breadth ikut membaik.",
        "late": "Winners mulai crowded. Hati-hati kejar beta terlambat.",
        "transition": "Kalau inflasi balik keras atau yields naik lagi, jalur sering ke Quad 2.",
    },
    "Quad 2": {
        "meaning": "Growth dan inflasi sama-sama naik. Biasanya industrials, cyclicals, energy, commodity-linked lebih kuat.",
        "good": "Industrials, cyclicals, energy, commodity-linked, selected small caps",
        "avoid": "Long-duration defensives, unprofitable growth, rate-sensitive defensives",
        "early": "Industrials, cyclicals, energy biasanya memimpin.",
        "mid": "Leadership paling rapi ada di cyclicals / commodity names.",
        "late": "Trade cyclical mulai crowded. Pantau oil, 2Y, 30Y, dan growth rollover.",
        "transition": "Kalau growth capek tapi inflasi tetap keras, jalur ke Quad 4. Kalau growth patah dan inflasi turun, ke Quad 3.",
    },
    "Quad 3": {
        "meaning": "Growth melemah, inflasi turun. Biasanya bonds, defensives, gold, dan quality balance-sheet names lebih kuat.",
        "good": "Bonds, defensives, gold, quality balance-sheet names",
        "avoid": "Deep cyclicals, broad EM beta, alt beta",
        "early": "Bonds, defensives, gold biasanya membaik duluan.",
        "mid": "Defensives masih dominan. Broad beta belum tentu bersih.",
        "late": "Kalau real yields dan dollar turun, Nasdaq / BTC / EM mulai menarik.",
        "transition": "Kalau disinflation lanjut dan growth stabil, jalur ke Quad 1. Kalau growth re-accelerate bareng inflasi, ke Quad 2.",
    },
    "Quad 4": {
        "meaning": "Growth melemah, inflasi naik. Ini salah satu fase paling susah untuk broad risk assets.",
        "good": "Energy, hard assets, commodity-linked, defensives dengan pricing power",
        "avoid": "Crypto beta, alt beta, Nasdaq duration-heavy, broad beta rallies",
        "early": "Pure oil / energy, hard assets, commodity-linked biasanya duluan.",
        "mid": "Winners fase awal masih bekerja, tapi broad beta tetap rapuh.",
        "late": "Old winners mulai crowded. Jangan terlalu agresif tambah trade lama.",
        "transition": "Kalau oil / yields / dollar mulai melemah, path berikutnya sering ke Quad 3 atau Quad 1.",
    },
}

ASSET_META = {
    "Nasdaq": {
        "drivers": "real yields, 2Y relief, liquidity, dollar",
        "sensitivity": {"real_yield": 0.32, "usd": 0.10, "liquidity": 0.20, "credit": 0.14, "oil": 0.04},
        "reversal": {"liquidity_relief": 0.24, "real_yield_relief": 0.32, "usd_relief": 0.08, "credit_relief": 0.18, "inflation_relief": 0.06},
        "proxies": [
            ("QQQ / quality growth", "Strongest", "Paling sensitif ke real yield relief dan duration squeeze turun."),
            ("Big-cap semis/software", "Strong", "Masih sangat related, tapi sedikit lebih sector-specific."),
            ("Broad weak growth", "Weak", "Bisa ikut memantul, tapi kualitasnya lebih buruk dan gampang gagal lanjut."),
        ],
    },
    "Crypto": {
        "drivers": "real yields, dollar, liquidity, credit",
        "sensitivity": {"real_yield": 0.32, "usd": 0.22, "liquidity": 0.20, "credit": 0.14, "oil": 0.05},
        "reversal": {"liquidity_relief": 0.30, "real_yield_relief": 0.30, "usd_relief": 0.20, "credit_relief": 0.14, "inflation_relief": 0.06},
        "proxies": [
            ("BTC", "Strongest", "Quality leader. Biasanya paling awal pulih kalau setup macro membaik."),
            ("ETH", "Strong", "Masih ikut risk-on, tapi butuh konfirmasi lebih besar daripada BTC."),
            ("High beta L1 / SOL-type", "Medium", "Butuh liquidity lebih lebar dan breadth lebih bersih."),
            ("Alt beta basket", "Weakest", "Paling belakangan dan paling gampang false start."),
        ],
    },
    "Emerging Markets": {
        "drivers": "dollar, real yields, liquidity, commodities",
        "sensitivity": {"real_yield": 0.18, "usd": 0.32, "liquidity": 0.16, "credit": 0.14, "oil": 0.10},
        "reversal": {"liquidity_relief": 0.20, "real_yield_relief": 0.16, "usd_relief": 0.30, "credit_relief": 0.14, "inflation_relief": 0.06},
        "proxies": [
            ("Selective commodity exporter EM", "Strongest", "Lebih direct ke commodity tailwind dan sering lebih bersih dari broad EM."),
            ("EM oil / resource-linked equities", "Strong", "Masih bagus, tapi ada country risk / FX risk."),
            ("Broad EM ETF", "Weak", "Sering misleading kalau dollar masih kuat atau risk-off global belum selesai."),
            ("Weak importer EM", "Weakest", "Biasanya paling jelek saat oil shock / dollar shock."),
        ],
    },
    "IHSG": {
        "drivers": "dollar, commodities, EM appetite, liquidity",
        "sensitivity": {"real_yield": 0.10, "usd": 0.28, "liquidity": 0.16, "credit": 0.12, "oil": 0.18},
        "reversal": {"liquidity_relief": 0.18, "real_yield_relief": 0.10, "usd_relief": 0.28, "credit_relief": 0.12, "inflation_relief": 0.06},
        "proxies": [
            ("MEDC / direct oil-linked local proxy", "Strongest", "Paling dekat ke oil chain di IHSG."),
            ("Coal / resource-linked local names", "Strong", "Masih direct ke commodity chain, tergantung komoditasnya."),
            ("Banks / domestic cyclicals", "Medium", "Bagus kalau domestic risk-on sehat, tapi bukan direct commodity proxy."),
            ("Property / speculative small caps", "Weakest", "Paling sensitif ke rates, dollar, dan flow risk."),
        ],
    },
    "Energy": {
        "drivers": "oil impulse, geopolitical premium, capex cycle",
        "sensitivity": {"real_yield": 0.05, "usd": 0.08, "liquidity": 0.08, "credit": 0.08, "oil": 0.55},
        "reversal": {"liquidity_relief": 0.10, "real_yield_relief": 0.08, "usd_relief": 0.08, "credit_relief": 0.10, "inflation_relief": 0.14},
        "proxies": [
            ("WTI / Brent", "Strongest", "Ekspresi paling murni untuk oil move."),
            ("COP / EOG / OXY / APA", "Strongest equity", "Pure upstream. Paling sensitif ke crude."),
            ("XOM / CVX / BP / SHEL", "Strong", "Integrated majors. Lebih kalem tapi lebih defensif."),
            ("SLB / HAL / BKR", "Medium", "Second-order. Butuh oil tinggi bertahan agar capex jalan."),
            ("FRO / TNK crude tankers", "Conditional", "Butuh freight dan route stress, bukan harga oil doang."),
        ],
    },
}

FX_BIAS = {
    "Quad 1": [("EURUSD long", "Strong", "Dollar biasanya melemah kalau real yields turun dan risk appetite membaik."), ("AUDUSD long", "Strong", "Lebih beta ke growth dan risk-on daripada EURUSD."), ("USDJPY short", "Medium-Strong", "Kalau yields AS turun dan dollar melemah, pair ini sering kehilangan tenaga."), ("USDCNH short", "Medium", "Dollar lemah + risk-on global biasanya bantu pair ini turun."), ("USDIDR short", "Medium", "Kalau dollar global melemah dan EM appetite membaik, tekanan ke rupiah biasanya berkurang.")],
    "Quad 2": [("AUDUSD long", "Strong", "Growth + inflasi naik sering bantu cyclical FX dan commodity FX."), ("USDCAD short", "Medium-Strong", "CAD bisa terbantu kalau commodity impulse kuat."), ("EURUSD long", "Medium", "Masih bisa naik, tapi tidak sebersih AUDUSD."), ("USDJPY mixed", "Medium", "Kalau yields AS juga naik, pair ini bisa tetap tinggi."), ("EURAUD short", "Lower", "Lebih tactical untuk relative strength AUD vs EUR.")],
    "Quad 3": [("USDJPY short", "Strong", "Disinflation + yields turun biasanya bantu sisi short USDJPY paling jelas."), ("EURUSD long", "Medium", "Kalau dollar melemah karena yields turun, EURUSD bisa terbantu."), ("AUDUSD short / avoid", "Medium-Strong", "Growth melemah biasanya bikin AUD relatif lebih lemah."), ("USDCNH long", "Medium", "Kalau growth dunia lemah, FX beta biasanya belum bagus."), ("USDIDR long", "Medium", "EM FX kadang belum enak kalau growth global dan risk appetite masih lemah.")],
    "Quad 4": [("USDJPY long", "Strong", "Dollar dan yields AS cenderung kuat, pair ini sering paling bersih."), ("EURUSD short", "Strong", "Dollar kuat biasanya bikin EURUSD cenderung lemah."), ("AUDUSD short", "Strong", "AUD sering lebih kena sisi growth risk daripada EUR."), ("USDCNH long", "Medium-Strong", "Dollar kuat + tekanan growth global biasanya dukung pair ini naik."), ("USDIDR long", "Medium", "Dollar kuat jadi tekanan untuk FX EM, tapi local policy bisa bikin geraknya lebih lambat.")],
}

CRYPTO_RANK = {
    "Quad 1": [("BTC", "Strongest", "Kualitas tertinggi, biasanya paling duluan pulih."), ("ETH", "Strong", "Masih ikut risk-on, tapi sedikit di bawah BTC."), ("SOL / high beta L1", "Medium", "Butuh liquidity yang benar-benar bagus."), ("Alt beta basket", "Late winner", "Paling dibantu liquidity, jadi biasanya belakangan.")],
    "Quad 2": [("BTC", "Strong", "Masih bisa jalan kalau growth kuat."), ("ETH", "Medium-Strong", "Masih oke, tapi makin sensitif kalau yields terlalu keras."), ("SOL / high beta L1", "Medium", "Masih bisa naik, tapi lebih rapuh."), ("Alt beta basket", "Weak", "Cepat rapuh kalau inflation / yields makin keras.")],
    "Quad 3": [("BTC", "Strongest defensive", "Kalau crypto mau hidup, biasanya BTC duluan."), ("ETH", "Medium", "Butuh real yields turun lebih jelas."), ("SOL / high beta L1", "Weak", "Beta tinggi belum bersih di Quad 3."), ("Alt beta basket", "Weakest", "Biasanya bukan tempat terbaik duluan.")],
    "Quad 4": [("BTC", "Least weak", "Masih paling kuat di dalam crypto walau tetap tertekan."), ("ETH", "Weak", "Masih sensitif ke yields dan liquidity."), ("SOL / high beta L1", "Very weak", "Biasanya kena lebih keras."), ("Alt beta basket", "Weakest", "Dollar kuat + real yields naik = kombinasi jelek.")],
}

TERMS = [
    ("Winner", "Aset yang paling cocok dengan regime sekarang."),
    ("Old winner", "Winner yang sudah lari duluan dan edge-nya mulai menipis."),
    ("Crowded", "Trade yang sudah terlalu obvious; banyak orang sudah di sana, jadi risk/reward makin jelek."),
    ("Beta", "Aset yang biasanya bergerak lebih besar dari market atau tema utamanya."),
    ("Broad beta", "Risk asset umum seperti index, small caps, broad EM, alt basket."),
    ("Quality", "Nama atau aset yang balance sheet, earnings, atau struktur risikonya lebih bagus."),
    ("Duration", "Aset yang paling sensitif ke perubahan yields atau real yields."),
    ("Hard asset", "Aset fisik atau komoditas nyata seperti oil, gold, copper."),
    ("Commodity-linked", "Nama yang untung kalau komoditas relevan naik."),
    ("Pricing power", "Perusahaan yang masih bisa jaga margin walau biaya naik."),
    ("Leader", "Yang biasanya bergerak paling awal dan paling bersih."),
    ("Follower", "Yang ikut tema, tapi bukan yang paling awal."),
    ("Lagging proxy", "Yang harusnya bisa ikut, tapi sering baru nyala belakangan kalau tema makin kuat."),
]

PROXY_GROUPS = {
    "Oil / energy chain": [
        ("WTI / Brent", "Strongest", "Direct instrument", "Ekspresi paling murni untuk oil move."),
        ("COP / EOG / OXY / APA", "Strongest equity", "Direct upstream", "Pure upstream. Paling sensitif ke crude."),
        ("XOM / CVX / BP / SHEL", "Strong", "Direct but calmer", "Integrated majors. Lebih defensif tapi kurang tajam."),
        ("SLB / HAL / BKR", "Medium", "Second-order", "Butuh oil tinggi bertahan agar capex jalan."),
        ("FRO / TNK", "Conditional", "Second-order / freight", "Butuh freight dan route stress, bukan oil doang."),
        ("Selective EM oil proxies", "Mixed", "Conditional local", "Bisa ikut, tapi kena drag dollar/flow/country risk."),
        ("Broad EM", "Misleading", "Weak spillover", "Oil naik tidak otomatis bikin broad EM bullish."),
    ],
    "Gas / LNG chain": [
        ("Gas / LNG direct instruments", "Strongest", "Direct instrument", "Paling murni ke gas/LNG."),
        ("Gas producers", "Strong", "Direct equity", "Paling dekat ke gas chain di equity."),
        ("LNG carriers", "Medium-Strong", "Transport second-order", "Valid kalau transport gas/LNG benar-benar ketat."),
        ("Integrated majors with gas exposure", "Medium", "Mixed direct", "Masih related, tapi tidak semurni producer murni."),
        ("Broad exporter markets", "Weak", "Weak spillover", "Harus sangat selektif, broad market sering noisy."),
    ],
    "Coal / bulk chain": [
        ("Coal miners", "Strongest", "Direct equity", "Hubungan paling langsung ke harga coal."),
        ("Dry bulk shipping", "Strong", "Second-order", "Coal memang butuh shipping bulk, tapi butuh volume confirm."),
        ("Ports / logistics", "Medium", "Third-order", "Butuh throughput naik nyata."),
        ("Heavy equipment", "Medium", "Third-order", "Butuh capex cycle mining benar-benar jalan."),
        ("Broad exporter markets", "Weak", "Weak spillover", "Lebih noisy dan kalah bersih dari direct proxies."),
    ],
    "Metals / mining chain": [
        ("Miners / metal producers", "Strongest", "Direct equity", "Hubungan paling langsung ke harga metals / ore."),
        ("Direct metal instruments", "Strong", "Direct instrument", "Ekspresi bersih untuk move komoditasnya."),
        ("Dry bulk", "Medium-Strong", "Second-order", "Iron ore / bulk metals sangat nyambung ke dry bulk."),
        ("Heavy equipment", "Medium", "Third-order", "Butuh capex dan project activity ikut naik."),
        ("Industrial cyclicals", "Weak", "Weak spillover", "Lebih umum dan tidak sebersih miners."),
    ],
    "Rates / dollar / liquidity chain": [
        ("Long bonds / duration", "Strongest", "Direct duration", "Paling awal menang saat disinflation / yield relief."),
        ("Nasdaq / quality growth", "Strong", "Direct duration equity", "Sangat sensitif ke real yield relief."),
        ("BTC", "Strong", "Direct crypto leader", "Biasanya crypto leader saat financial conditions membaik."),
        ("Broad EM", "Medium", "Conditional", "Butuh dollar relief lebih jelas."),
        ("Small caps", "Weak", "Late spillover", "Butuh breadth dan confidence lebih bersih."),
        ("Alt beta", "Weakest", "Late spillover", "Paling belakangan dan paling rapuh."),
    ],
}

DIVERGENCE_RULES = [
    ("Oil naik tapi EM broad ga naik", "Biasanya dollar terlalu kuat atau flow ke EM jelek. Jadi pure oil proxy masih bisa naik, broad EM belum tentu."),
    ("Oil naik tapi tanker ga naik", "Harga oil doang tidak cukup. Freight rates, supply kapal, dan trade flow juga harus mendukung."),
    ("Coal naik tapi dry bulk ga naik", "Harga commodity bisa naik karena supply shock, tapi volume angkut belum tentu ikut bagus."),
    ("Real yields turun tapi crypto belum naik", "Bisa karena dollar belum benar-benar lemah, liquidity belum membaik, atau positioning masih jelek."),
    ("Dollar turun tapi EM belum gerak", "Bisa karena growth dunia masih lemah, credit stress masih tinggi, atau local issues masih berat."),
]

CHAIN_STRENGTH = [
    ("Oil -> WTI/Brent", "Very High", "Underlying paling murni."),
    ("Oil -> pure upstream oil", "Very High", "Harga crude paling cepat terasa ke producer upstream murni."),
    ("Oil -> integrated majors", "High", "Masih kuat, tapi lebih kalem dari upstream murni."),
    ("Oil -> oil services", "Medium", "Butuh oil tinggi lebih lama untuk capex response."),
    ("Oil -> crude tankers", "Medium", "Butuh freight rates dan trade flows ikut support."),
    ("Coal -> coal miners", "Very High", "Hubungan paling langsung."),
    ("Coal -> dry bulk shipping", "High", "Coal memang butuh shipping bulk."),
    ("Metals -> miners", "Very High", "Hubungan paling langsung."),
    ("Metals -> dry bulk", "High", "Bulk ores sangat nyambung ke dry bulk."),
    ("LNG / gas -> LNG carriers", "High", "Valid untuk gas/LNG chain, bukan oil chain murni."),
    ("2Y naik keras -> front-end stress", "Very High", "Paling cepat terasa ke duration / beta dan pembacaan hawkish."),
    ("30Y naik keras -> long-end pressure", "High", "Long-end naik keras sering menekan duration dan valuation."),
    ("Real yields turun -> Nasdaq", "Very High", "Duration asset sangat sensitif."),
    ("Real yields turun -> BTC", "High", "BTC biasanya crypto leader."),
    ("Dollar naik -> broad EM", "Very High negative", "Salah satu transmission line terpenting ke EM."),
]

SCENARIO_MATRIX = [
    ("Oil up + USD up + 2Y up", "Shock reflation / hard inflation pressure", "Long WTI / pure oil / integrated majors; avoid broad EM, alt beta, duration growth"),
    ("Oil up + USD weak + 2Y flat/down", "Healthy reflation", "Long energy + selective EM commodity exporters + miners"),
    ("Oil down + 2Y down + real yields down + USD down", "Risk-on relief / disinflation relief", "Nasdaq, BTC, broad EM, quality cyclicals"),
    ("Growth down + inflation down + 2Y down, but credit still bad", "Defensive relief", "Bonds, defensives, gold; broad beta belum tentu bersih"),
    ("Growth up + inflation up + 30Y up faster", "Cyclical leadership, duration pressure", "Industrials, cyclicals, energy; avoid weak duration"),
    ("BTC up, alts not following", "Early risk-on / quality-led crypto", "BTC/ETH lebih bersih daripada alt basket"),
    ("Commodity exporter EM up, broad EM weak", "Selective EM only", "Long selective exporters, avoid broad EM ETF"),
    ("Oil up, tankers not following", "Chain incomplete", "Stick to first-order winners, wait for freight confirmation"),
    ("Coal up, dry bulk not following", "Supply shock, not full volume recovery", "Prefer coal miners over shipping"),
    ("2Y spikes, 30Y muted", "Front-end hawkish shock", "Avoid duration beta; prefer defensives / selective hard assets"),
    ("30Y spikes faster than 2Y", "Long-end term premium pressure", "Avoid long-duration valuation-sensitive names"),
    ("Dollar down, EM still weak", "Growth / credit / local risk still bad", "Prefer US quality or selective exporters over broad EM"),
    ("Real yields down, crypto still weak", "Liquidity or credit not confirming", "Prefer Nasdaq / bonds first, wait on crypto breadth"),
]

REGIME_SWITCH_ARCHETYPES = [
    ("Quad 4 -> Quad 3 -> Quad 1", "Shock inflation mereda, lalu disinflation stabil, lalu duration/risk-on menang."),
    ("Quad 2 overheating -> Quad 4", "Growth masih panas tapi inflasi/commodities kebablasan dan broad beta mulai capek."),
    ("Quad 3 relief -> false Quad 1 -> back to Quad 3", "Yield relief ada, tapi breadth/credit tidak confirm sehingga risk-on gagal lanjut."),
    ("Quad 4 commodity shock -> selective reflation -> failure", "Direct commodity menang, tapi broad cyclicals/EM tidak confirm lalu move kehilangan tenaga."),
]

FALSE_RECOVERY_MAP = [
    ("2Y belum relief", "Risk-on bounce sering belum tahan lama kalau front-end stress masih keras."),
    ("Dollar tetap kuat", "EM/crypto/small caps sering gagal confirm walau ada pantulan."),
    ("Credit tetap jelek", "Broad beta bisa memantul, tapi kualitasnya rendah dan rawan gagal lanjut."),
    ("Breadth tidak ikut", "Kalau hanya old winners atau short covering, itu sering dead-cat bounce."),
    ("Crypto beta / small caps / EM tidak confirm", "Biasanya pertanda recovery masih sempit atau palsu."),
]

CRASH_ENGINE = {
    "types": [
        ("Liquidity shock", "Semua dijual karena market cari cash. Correlation naik cepat."),
        ("Credit shock", "Spread melebar, trust turun, balance-sheet stress jadi pusat."),
        ("Growth collapse", "Commodity dan cyclicals bisa turun bareng karena demand fear."),
        ("Inflation / commodity shock", "Hard assets bisa naik duluan, tapi broad beta sering rapuh."),
        ("Policy shock", "2Y / front-end repricing bikin duration dan beta sesak."),
        ("Geopolitical shock", "Oil, tanker, direct commodity proxies bisa menang, broad EM belum tentu."),
        ("Systemic confidence shock", "Lehman-style. Fokus market pindah ke survival dan funding."),
    ],
    "early": [
        "Breadth makin sempit.",
        "Credit mulai rusak.",
        "Dollar makin dominan.",
        "2Y/front-end stress naik.",
        "Proxy chain mulai putus: direct winner hidup, second-order tidak ikut.",
    ],
    "mid": [
        "Broad beta mulai nyerah: small caps, EM, alt beta rapuh.",
        "Forced deleveraging mulai terasa.",
        "Dollar dan credit jadi makin penting.",
        "Even winners awal bisa mulai ikut dijual kalau likuidasi membesar.",
    ],
    "late": [
        "Correlation naik mendekati satu.",
        "Forced selling / capitulation lebih jelas.",
        "Market fokus ke survival dan policy response.",
        "Orang berhenti cari winner, mulai cari siapa yang paling tahan.",
    ],
    "recovery": [
        ("Deflationary / liquidity crash", "Bonds -> defensives -> Nasdaq -> BTC -> EM -> small caps -> alt beta"),
        ("Inflation / oil shock", "Direct commodity -> pricing-power defensives -> selective exporters -> broad beta belakangan"),
        ("Credit shock", "Treasuries / safest duration -> gold / defensives -> quality equities -> broad beta jauh belakangan"),
    ],
}

TOP_BOTTOM_ASSETS = ["Nasdaq", "Crypto", "Emerging Markets", "IHSG", "Energy"]

YAHOO_SYMBOLS = {
    "IHSG": "^JKSE",
    "EIDO": "EIDO",
    "EEM": "EEM",
    "QQQ": "QQQ",
    "SPY": "SPY",
    "XLE": "XLE",
    "USO": "USO",
    "GLD": "GLD",
    "SLV": "SLV",
    "CPER": "CPER",
    "PICK": "PICK",
    "HYG": "HYG",
    "TLT": "TLT",
    "BIZD": "BIZD",
    "OWL": "OWL",
    "KOL": "KOL",
    "TAN": "TAN",
    "USDIDR": "IDR=X",
    "BTC": "BTC-USD",
}

@dataclass
class MacroState:
    growth_now: float
    growth_prev: float
    inflation_now: float
    inflation_prev: float
    y2_now: float
    y2_prev: float
    y10_now: float
    y10_prev: float
    y30_now: float
    y30_prev: float
    ry10_now: float
    ry10_prev: float
    oil_now: float
    oil_prev: float
    gold_now: float
    gold_prev: float
    dxy_now: float
    dxy_prev: float
    liq_now: float
    liq_prev: float
    credit_now: float
    credit_prev: float
    breadth_now: float
    breadth_prev: float

# --------------------------
# Data fetch helpers
# --------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fred_single(series_id: str, api_key: str, observation_start: str = DEFAULT_START) -> pd.DataFrame:
    url = f"{BASE_URL}/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": observation_start,
        "sort_order": "asc",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    df = pd.DataFrame(js.get("observations", []))
    if df.empty:
        return pd.DataFrame(columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["date", "value"]].dropna().sort_values("date").reset_index(drop=True)


def fetch_with_fallback(name: str, api_key: str, observation_start: str):
    last_error = None
    for sid in SERIES[name]:
        try:
            df = fred_single(sid, api_key, observation_start)
            if not df.empty:
                return df, sid, None
            last_error = f"{sid}: empty"
        except Exception as e:
            last_error = f"{sid}: {e}"
    return pd.DataFrame(columns=["date", "value"]), None, last_error


def pct_change_annual(values: pd.Series, periods: int) -> pd.Series:
    return (values / values.shift(periods) - 1.0) * 100.0


def zscore(series: pd.Series, window: int = 36) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return (series - mean) / std


def merge_on_date(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for name, df in frames.items():
        temp = df.rename(columns={"value": name})
        merged = temp if merged is None else pd.merge(merged, temp, on="date", how="outer")
    return merged.sort_values("date").reset_index(drop=True)


def prev_value(df: pd.DataFrame) -> float:
    c = df.dropna().reset_index(drop=True)
    if len(c) > 22:
        return float(c["value"].iloc[-22])
    if len(c) > 1:
        return float(c["value"].iloc[-2])
    return float(c["value"].iloc[-1])


def build_live_macro_state(api_key: str, observation_start: str):
    dfs, ids, errs = {}, {}, {}
    for k in SERIES:
        df, sid, err = fetch_with_fallback(k, api_key, observation_start)
        dfs[k], ids[k], errs[k] = df, sid, err

    missing = [k for k in SERIES if dfs[k].empty and k not in OPTIONAL_SERIES]
    if missing:
        raise ValueError("Missing required series: " + ", ".join(missing))

    cpi = dfs["cpi"].copy(); cpi["yoy"] = pct_change_annual(cpi["value"], 12)
    core_cpi = dfs["core_cpi"].copy(); core_cpi["yoy"] = pct_change_annual(core_cpi["value"], 12)
    pce = dfs["pce"].copy(); pce["yoy"] = pct_change_annual(pce["value"], 12)
    indpro = dfs["indpro"].copy(); indpro["yoy"] = pct_change_annual(indpro["value"], 12)
    payrolls = dfs["payrolls"].copy(); payrolls["yoy"] = pct_change_annual(payrolls["value"], 12)

    growth_inputs = merge_on_date({
        "indpro_yoy": indpro[["date", "yoy"]].rename(columns={"yoy": "value"}),
        "payrolls_yoy": payrolls[["date", "yoy"]].rename(columns={"yoy": "value"}),
        "growth_conf": dfs["growth_conf"],
        "unrate_inv": dfs["unrate"].assign(value=lambda x: -x["value"])[["date", "value"]],
    }).ffill()

    infl_inputs = merge_on_date({
        "cpi_yoy": cpi[["date", "yoy"]].rename(columns={"yoy": "value"}),
        "core_cpi_yoy": core_cpi[["date", "yoy"]].rename(columns={"yoy": "value"}),
        "pce_yoy": pce[["date", "yoy"]].rename(columns={"yoy": "value"}),
        "breakeven10y": dfs["breakeven10y"],
    }).ffill()

    growth_comp = pd.DataFrame({"date": growth_inputs["date"]})
    for col in ["indpro_yoy", "payrolls_yoy", "growth_conf", "unrate_inv"]:
        growth_comp[col] = zscore(growth_inputs[col], 36)
    growth_comp["growth_composite"] = growth_comp.drop(columns=["date"]).mean(axis=1)

    infl_comp = pd.DataFrame({"date": infl_inputs["date"]})
    for col in [c for c in infl_inputs.columns if c != "date"]:
        infl_comp[col] = zscore(infl_inputs[col], 36)
    infl_comp["inflation_composite"] = infl_comp.drop(columns=["date"]).mean(axis=1)

    liq_in = merge_on_date({"walcl": dfs["walcl"], "rrp": dfs["rrp"]}).ffill()
    liq = pd.DataFrame({"date": liq_in["date"]})
    liq["walcl_roc"] = liq_in["walcl"].pct_change(4) * 100
    liq["rrp_roc_inv"] = -(liq_in["rrp"].pct_change(4) * 100)
    liq["liquidity_composite"] = pd.concat([zscore(liq["walcl_roc"], 36), zscore(liq["rrp_roc_inv"], 36)], axis=1).mean(axis=1)

    credit_in = merge_on_date({"hy": dfs["hy"], "nfci": dfs["nfci"]}).ffill()
    credit = pd.DataFrame({"date": credit_in["date"]})
    credit["credit_composite"] = pd.concat([zscore(credit_in["hy"], 36), zscore(credit_in["nfci"], 36)], axis=1).mean(axis=1)

    comp = (
        growth_comp[["date", "growth_composite"]]
        .merge(infl_comp[["date", "inflation_composite"]], on="date", how="outer")
        .merge(liq[["date", "liquidity_composite"]], on="date", how="outer")
        .merge(credit[["date", "credit_composite"]], on="date", how="outer")
        .sort_values("date").ffill()
    )
    comp["breadth"] = (comp["growth_composite"].fillna(0) - comp["credit_composite"].fillna(0) + comp["liquidity_composite"].fillna(0)) * 10 + 50
    comp = comp.dropna().reset_index(drop=True)

    state = MacroState(
        growth_now=float(comp["growth_composite"].iloc[-1]),
        growth_prev=float(comp["growth_composite"].iloc[-2]),
        inflation_now=float(comp["inflation_composite"].iloc[-1]),
        inflation_prev=float(comp["inflation_composite"].iloc[-2]),
        y2_now=float(dfs["us2y"]["value"].dropna().iloc[-1]),
        y2_prev=prev_value(dfs["us2y"]),
        y10_now=float(dfs["us10y"]["value"].dropna().iloc[-1]),
        y10_prev=prev_value(dfs["us10y"]),
        y30_now=float(dfs["us30y"]["value"].dropna().iloc[-1]),
        y30_prev=prev_value(dfs["us30y"]),
        ry10_now=float(dfs["real10y"]["value"].dropna().iloc[-1]),
        ry10_prev=prev_value(dfs["real10y"]),
        oil_now=float(dfs["wti"]["value"].dropna().iloc[-1]),
        oil_prev=prev_value(dfs["wti"]),
        gold_now=float(dfs["gold"]["value"].dropna().iloc[-1]) if not dfs["gold"].empty else float("nan"),
        gold_prev=prev_value(dfs["gold"]) if not dfs["gold"].empty else float("nan"),
        dxy_now=float(dfs["dxy"]["value"].dropna().iloc[-1]),
        dxy_prev=prev_value(dfs["dxy"]),
        liq_now=float(comp["liquidity_composite"].iloc[-1]),
        liq_prev=float(comp["liquidity_composite"].iloc[-2]),
        credit_now=float(comp["credit_composite"].iloc[-1]),
        credit_prev=float(comp["credit_composite"].iloc[-2]),
        breadth_now=float(comp["breadth"].iloc[-1]),
        breadth_prev=float(comp["breadth"].iloc[-2]),
    )
    return state, comp, dfs, ids, errs

# --------------------------
# News helpers
# --------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news_rss(query: str, max_items: int = 8) -> List[Dict[str, str]]:
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    root = ET.fromstring(r.text)
    items = []
    for item in root.findall(".//item")[:max_items]:
        title = item.findtext("title", default="")
        link = item.findtext("link", default="")
        pub = item.findtext("pubDate", default="")
        desc = re.sub(r"<[^>]+>", "", item.findtext("description", default=""))
        items.append({"title": title, "link": link, "published": pub, "desc": desc})
    return items


def infer_news_case(title: str, desc: str) -> Tuple[str, str, str]:
    text = f"{title} {desc}".lower()
    if any(k in text for k in ["private credit", "bdc", "owl", "direct lending", "alt manager", "apollo", "ares"]):
        return (
            "Private credit stress",
            "Baca ini sebagai potensi credit accident / spillover watch, bukan otomatis bottom.",
            "Kalau spreads melebar, BIZD/OWL lemah, dan 2Y/front-end stres naik, treat as stress not cheapness.",
        )
    if any(k in text for k in ["auction", "refund", "bill issuance", "treasury issuance", "maturity wall", "rollover"]):
        return (
            "Funding / auction shock",
            "Front-end dan term funding bisa lebih penting daripada headline makro umum.",
            "Kalau 2Y naik lebih cepat daripada 10Y/30Y dan dollar firm, treat as front-end stress.",
        )
    if any(k in text for k in ["iran", "hormuz", "war", "missile", "middle east"]):
        return (
            "Geopolitical energy shock",
            "Biasanya oil dan tanker risk premium bisa naik, tapi broad EM bisa tetap lemah kalau dollar dan 2Y keras.",
            "Prefer WTI / pure oil / integrated majors daripada broad EM beta.",
        )
    if any(k in text for k in ["fed", "powell", "yield", "2-year", "treasury", "rate"]):
        return (
            "Policy / yield shock",
            "2Y dan real yields menentukan napas untuk duration, crypto, dan EM.",
            "Kalau 2Y naik cepat, hati-hati Nasdaq/crypto beta; kalau 2Y turun, lihat risk-on relief.",
        )
    if any(k in text for k in ["coal", "power shortage", "blackout", "solar", "renewable", "lng", "gas"]):
        return (
            "Energy mix / scarcity shock",
            "Bedakan legacy energy scarcity, power shortage, coal leadership, dan solar catch-up. Itu tidak selalu memberi pesan yang sama.",
            "Jangan treat semua energy headline sebagai pure oil beta; lihat siapa yang benar-benar memimpin chain-nya.",
        )
    if any(k in text for k in ["china", "stimulus", "copper", "iron ore", "steel", "property"]):
        return (
            "China / commodity demand shock",
            "Paling kena ke miners, dry bulk, commodity EM, lalu second-order equipment.",
            "Prefer miners dulu, lalu dry bulk kalau volume benar-benar confirm.",
        )
    if any(k in text for k in ["credit", "default", "bank", "spread", "stress"]):
        return (
            "Credit stress",
            "Broad beta, EM, small caps, alt beta sering lebih rapuh.",
            "Prefer defensives / bonds / gold sampai credit relief muncul.",
        )
    return (
        "General macro news",
        "Perlu dibaca bareng quad, 2Y, dollar, oil, liquidity, dan credit.",
        "Jangan trade headline sendirian; sinkronkan dengan regime.",
    )

# --------------------------
# Macro logic
# --------------------------
def nowcast_scores(state: MacroState) -> Dict[str, float]:
    g = state.growth_now - state.growth_prev
    i = state.inflation_now - state.inflation_prev
    y2 = state.y2_now - state.y2_prev
    y10 = state.y10_now - state.y10_prev
    y30 = state.y30_now - state.y30_prev
    ry = state.ry10_now - state.ry10_prev
    oil = state.oil_now - state.oil_prev
    dxy = state.dxy_now - state.dxy_prev
    liq = state.liq_now - state.liq_prev
    credit = state.credit_now - state.credit_prev
    breadth = state.breadth_now - state.breadth_prev

    growth_score = 1.0*(g/0.7) + 0.55*(liq/1.0) + 0.40*(breadth/10.0) - 0.55*(credit/0.7) - 0.35*(ry/0.35) - 0.20*(dxy/2.5) - 0.15*(oil/8.0) - 0.20*(y2/0.50)
    inflation_score = 1.0*(i/0.5) + 0.60*(oil/8.0) + 0.20*(y10/0.50) + 0.15*(y30/0.50) - 0.15*(credit/0.7) - 0.15*(dxy/2.5)
    return {
        "growth_up_prob": float(1 / (1 + math.exp(-growth_score))),
        "inflation_up_prob": float(1 / (1 + math.exp(-inflation_score))),
        "growth_score": float(growth_score),
        "inflation_score": float(inflation_score),
    }


def current_quad(state: MacroState) -> str:
    nc = nowcast_scores(state)
    if nc["growth_up_prob"] >= 0.5 and nc["inflation_up_prob"] < 0.5:
        return "Quad 1"
    if nc["growth_up_prob"] >= 0.5 and nc["inflation_up_prob"] >= 0.5:
        return "Quad 2"
    if nc["growth_up_prob"] < 0.5 and nc["inflation_up_prob"] < 0.5:
        return "Quad 3"
    return "Quad 4"


def next_quad_probabilities(state: MacroState) -> Dict[str, float]:
    nc = nowcast_scores(state)
    pg, pi = nc["growth_up_prob"], nc["inflation_up_prob"]
    return {
        "Quad 1": float(pg*(1-pi)),
        "Quad 2": float(pg*pi),
        "Quad 3": float((1-pg)*(1-pi)),
        "Quad 4": float((1-pg)*pi),
    }


def phase_confidence(probs: Dict[str, float]) -> float:
    vals = sorted(probs.values(), reverse=True)
    return float(vals[0] - vals[1]) if len(vals) >= 2 else 0.0


def next_likely_quad(state: MacroState) -> str:
    cur = current_quad(state)
    for q, _ in sorted(next_quad_probabilities(state).items(), key=lambda x: x[1], reverse=True):
        if q != cur:
            return q
    return cur


def lifecycle_stage(state: MacroState) -> str:
    front = abs(state.y2_now - state.y2_prev)
    long = abs(state.y30_now - state.y30_prev)
    stress = abs(state.oil_now - state.oil_prev) + abs(state.credit_now - state.credit_prev)
    conf = phase_confidence(next_quad_probabilities(state))
    raw = (front/0.5 + long/0.5 + stress/12.0) / 3.0
    if conf >= 0.20 and raw < 0.35:
        return "Early"
    if conf >= 0.12 and raw < 0.55:
        return "Mid"
    if conf >= 0.08 and raw < 0.80:
        return "Late"
    return "Transition"


def macro_components(state: MacroState) -> Dict[str, float]:
    return {
        "real_yield": float(np.clip((state.ry10_now - state.ry10_prev)/0.35, 0, 1)),
        "real_yield_relief": float(np.clip(-(state.ry10_now - state.ry10_prev)/0.35, 0, 1)),
        "usd": float(np.clip((state.dxy_now - state.dxy_prev)/2.5, 0, 1)),
        "usd_relief": float(np.clip(-(state.dxy_now - state.dxy_prev)/2.5, 0, 1)),
        "liquidity": float(np.clip(-(state.liq_now - state.liq_prev)/1.0, 0, 1)),
        "liquidity_relief": float(np.clip((state.liq_now - state.liq_prev)/1.0, 0, 1)),
        "credit": float(np.clip((state.credit_now - state.credit_prev)/0.7, 0, 1)),
        "credit_relief": float(np.clip(-(state.credit_now - state.credit_prev)/0.7, 0, 1)),
        "oil": float(np.clip((state.oil_now - state.oil_prev)/12.0, 0, 1)),
        "inflation_relief": float(np.clip(-(state.inflation_now - state.inflation_prev)/0.5, 0, 1)),
        "front_end_stress": float(np.clip((state.y2_now - state.y2_prev)/0.5, 0, 1)),
        "long_end_pressure": float(np.clip((state.y30_now - state.y30_prev)/0.5, 0, 1)),
    }


def asset_risk(state: MacroState, asset: str) -> float:
    c = macro_components(state)
    meta = ASSET_META[asset]
    base = sum(meta["sensitivity"][k] * c[k] for k in meta["sensitivity"])
    extra = 0.0
    if asset in ["Nasdaq", "Crypto"]:
        extra += 0.10*c["front_end_stress"] + 0.08*c["long_end_pressure"]
    if asset in ["Emerging Markets", "IHSG"]:
        extra += 0.06*c["front_end_stress"]
    return float(np.clip(base + extra, 0, 1))


def asset_bottom(state: MacroState, asset: str) -> float:
    c = macro_components(state)
    meta = ASSET_META[asset]
    base = sum(meta["reversal"][k] * c[k] for k in meta["reversal"])
    if asset in ["Nasdaq", "Crypto"] and state.y2_now <= state.y2_prev:
        base += 0.08
    return float(np.clip(base, 0, 1))


def asset_opportunity(state: MacroState, asset: str) -> float:
    return float(np.clip(0.6*(1-asset_risk(state, asset)) + 0.4*asset_bottom(state, asset), 0, 1))


def top_risk_stage(score: float) -> str:
    if score >= 0.75:
        return "Extreme top risk"
    if score >= 0.55:
        return "High top risk"
    if score >= 0.35:
        return "Rising top risk"
    return "Low top risk"


def bottom_stage(score: float) -> str:
    if score >= 0.75:
        return "High probability bottoming"
    if score >= 0.55:
        return "Maturing base"
    if score >= 0.35:
        return "Building base"
    if score >= 0.20:
        return "Early stabilization"
    return "No clear bottoming"


def phase_transition_stage(state: MacroState) -> str:
    score = 0.0
    if state.y2_now <= state.y2_prev:
        score += 0.25
    if state.ry10_now <= state.ry10_prev:
        score += 0.20
    if state.dxy_now <= state.dxy_prev:
        score += 0.20
    if state.credit_now <= state.credit_prev:
        score += 0.15
    if state.breadth_now >= state.breadth_prev:
        score += 0.10
    if state.oil_now <= state.oil_prev:
        score += 0.10
    if score >= 0.80:
        return "Late transition / ready to flip"
    if score >= 0.60:
        return "Mid transition"
    if score >= 0.35:
        return "Early transition"
    return "No clear transition yet"

def top_signal_stage(state: MacroState, asset: str) -> str:
    top = asset_risk(state, asset)
    bottom = asset_bottom(state, asset)
    if top >= 0.75:
        return "Late / crowded topping"
    if top >= 0.55:
        return "Mid topping development"
    if top >= 0.35 and bottom < 0.35:
        return "Early top warning"
    return "No clear toping signal"

def bottom_signal_stage(state: MacroState, asset: str) -> str:
    bottom = asset_bottom(state, asset)
    top = asset_risk(state, asset)
    if bottom >= 0.75:
        return "Late bottom / ready for next phase"
    if bottom >= 0.55:
        return "Mid bottom confirmation"
    if bottom >= 0.35 and top < 0.55:
        return "Building / early-mid bottoming"
    if bottom >= 0.20:
        return "Early stabilization"
    return "No clear bottoming"

def build_phase_transition_table(state: MacroState) -> pd.DataFrame:
    rows = []
    conditions = [
        ("2Y relief", state.y2_now <= state.y2_prev, "Front-end stress mulai reda."),
        ("Real yields relief", state.ry10_now <= state.ry10_prev, "Duration / quality assets mulai dapat napas."),
        ("Dollar relief", state.dxy_now <= state.dxy_prev, "EM / crypto / risk-on lebih mungkin ikut."),
        ("Credit stabil", state.credit_now <= state.credit_prev, "Stress sistemik tidak makin buruk."),
        ("Breadth membaik", state.breadth_now >= state.breadth_prev, "Leadership mulai melebar."),
        ("Oil relief", state.oil_now <= state.oil_prev, "Tekanan hard-inflation / commodity shock mereda."),
    ]
    for signal, ok, why in conditions:
        rows.append({"Signal": signal, "Status": "Yes" if ok else "No", "Why it matters": why})
    return pd.DataFrame(rows)

def build_topping_signal_table(state: MacroState) -> pd.DataFrame:
    assets = ["Energy", "Nasdaq", "Crypto", "Emerging Markets", "IHSG"]
    rows = []
    for asset in assets:
        rows.append({
            "Aset": asset,
            "Top stage": top_signal_stage(state, asset),
            "Top risk": f"{asset_risk(state, asset):.0%}",
            "Why": ASSET_META[asset]["drivers"],
        })
    return pd.DataFrame(rows)

def build_bottoming_signal_table(state: MacroState) -> pd.DataFrame:
    assets = ["Energy", "Nasdaq", "Crypto", "Emerging Markets", "IHSG"]
    rows = []
    for asset in assets:
        rows.append({
            "Aset": asset,
            "Bottom stage": bottom_signal_stage(state, asset),
            "Bottom formation": f"{asset_bottom(state, asset):.0%}",
            "Why": ASSET_META[asset]["drivers"],
        })
    return pd.DataFrame(rows)

def build_false_transition_table() -> pd.DataFrame:
    rows = [
        ("2Y relief doang", "Kalau dollar dan credit belum ikut membaik, itu sering baru relief pendek, belum regime turn."),
        ("Oil turun tapi credit makin rusak", "Bisa bikin orang kira risk-on sehat, padahal itu growth scare / deflationary stress."),
        ("Nasdaq kuat tapi breadth sempit", "Bisa jadi old winner extension, belum broad phase turn."),
        ("BTC naik tapi ETH/alts nggak ikut", "Biasanya baru early / quality-led, belum full crypto turn."),
        ("EM exporter kuat tapi broad EM lemah", "Selective theme, bukan broad EM confirmation."),
    ]
    return pd.DataFrame(rows, columns=["False transition trap", "Why"])

def build_false_bottom_table() -> pd.DataFrame:
    rows = [
        ("Asset weak memantul tapi 2Y masih naik", "Sering cuma technical bounce, belum bottoming macro."),
        ("Dollar tetap kuat", "EM / crypto / broad beta bottom sering belum valid kalau dollar masih dominan."),
        ("Credit masih memburuk", "Bottoming risk assets belum bersih kalau stress sistemik belum reda."),
        ("Breadth tidak ikut", "Kalau cuma sedikit nama yang stabil, base masih rapuh."),
        ("Old winners belum menyerah", "Kadang rotation belum benar-benar pindah; weak asset bounce bisa cepat gagal."),
    ]
    return pd.DataFrame(rows, columns=["False bottom trap", "Why"])


def expression_ranking_for_current_quad(quad: str) -> List[Tuple[str, str, str]]:
    if quad == "Quad 1":
        return [
            ("QQQ / quality growth", "Strongest", "Paling bersih saat real yields turun dan duration relief jalan."),
            ("BTC", "Strong", "Leader crypto saat conditions membaik."),
            ("Broad EM / selective exporters", "Medium", "Butuh dollar relief lebih jelas."),
            ("Small caps", "Weak but related", "Sering ikut belakangan setelah breadth melebar."),
        ]
    if quad == "Quad 2":
        return [
            ("Industrials / cyclicals", "Strongest", "Growth + inflasi naik mendukung cyclicals lebih dulu."),
            ("Energy / commodity-linked", "Strong", "Masih didukung reflation / commodity impulse."),
            ("Selected small caps", "Medium", "Butuh confidence risk-on cukup bersih."),
            ("Weak duration growth", "Avoid", "Sering kalah saat yields lebih keras."),
        ]
    if quad == "Quad 3":
        return [
            ("Bonds / duration", "Strongest", "Paling awal menang dalam disinflation / weak growth."),
            ("Defensives / gold", "Strong", "Masih sangat masuk akal di lingkungan defensif."),
            ("Nasdaq / BTC watchlist", "Medium", "Baru lebih bersih jika real yields dan dollar benar-benar relief."),
            ("Broad beta / alt beta", "Avoid", "Sering belum waktunya."),
        ]
    return [
        ("WTI / Brent / pure upstream oil", "Strongest", "Ekspresi paling bersih untuk oil shock / hard inflation regime."),
        ("Integrated majors / selective hard assets", "Strong", "Masih sangat valid, lebih defensif dari pure beta."),
        ("Crude tankers / services", "Medium / conditional", "Butuh freight atau capex confirmation."),
        ("Broad EM / crypto beta / duration growth", "Avoid", "Sering jadi false proxy atau korban shock."),
    ]


def expression_ranking_for_next_quad(quad: str) -> List[Tuple[str, str, str]]:
    if quad == "Quad 1":
        return [
            ("QQQ / quality growth", "Strongest", "Biasanya duration leader paling duluan."),
            ("BTC", "Strong", "Quality crypto leader."),
            ("ETH / broad EM", "Medium", "Perlu breadth dan dollar relief lebih bersih."),
            ("Small caps / alt beta", "Late", "Biasanya belakangan."),
        ]
    if quad == "Quad 2":
        return [
            ("Industrials / cyclicals", "Strongest", "Growth dan inflasi sama-sama naik mendukung cyclicals."),
            ("Energy / commodity-linked", "Strong", "Masih sangat nyambung ke reflation yang sehat."),
            ("Commodity FX / selective exporters", "Medium", "Butuh dollar tidak terlalu dominan."),
            ("Unprofitable duration", "Avoid", "Sering kalah di reflation yang lebih keras."),
        ]
    if quad == "Quad 3":
        return [
            ("Bonds / defensives", "Strongest", "Disinflation + weak growth biasanya menguntungkan mereka duluan."),
            ("Gold / quality balance sheet", "Strong", "Masih paling logis secara defensif."),
            ("Nasdaq watchlist", "Medium", "Butuh real yields turun lebih meyakinkan."),
            ("Broad beta", "Avoid", "Sering belum waktunya."),
        ]
    return [
        ("WTI / Brent / pure oil", "Strongest", "Shock inflation / hard-asset regime biasanya mulai dari sini."),
        ("Integrated majors / selective hard assets", "Strong", "Masih logis kalau move makin matang."),
        ("Tanker / services", "Medium / conditional", "Butuh konfirmasi tambahan."),
        ("Broad beta", "Avoid", "Sering rapuh di Quad 4."),
    ]


def build_full_narrative(state: MacroState) -> str:
    quad = current_quad(state)
    nxt = next_likely_quad(state)
    stage = lifecycle_stage(state)
    conf = phase_confidence(next_quad_probabilities(state))
    two_ten = state.y10_now - state.y2_now
    ten_thirty = state.y30_now - state.y10_now
    c = macro_components(state)

    macro_flags = []
    macro_flags.append("growth membaik" if state.growth_now > state.growth_prev else "growth melemah")
    macro_flags.append("inflasi membaik" if state.inflation_now < state.inflation_prev else "inflasi mengeras")
    macro_flags.append("2Y relief" if state.y2_now <= state.y2_prev else "2Y menekan")
    macro_flags.append("real yields relief" if state.ry10_now <= state.ry10_prev else "real yields menekan")
    macro_flags.append("dollar relief" if state.dxy_now <= state.dxy_prev else "dollar menekan")
    macro_flags.append("liquidity membaik" if state.liq_now >= state.liq_prev else "liquidity memburuk")
    macro_flags.append("credit stabil/membaik" if state.credit_now <= state.credit_prev else "credit memburuk")
    macro_flags.append("oil menekan" if state.oil_now > state.oil_prev else "oil relief")

    quality = "fase sempit / pilih-pilih" if state.breadth_now < 50 or state.breadth_now < state.breadth_prev else "fase lebih sehat / breadth membaik"
    conf_text = "tinggi" if conf >= 0.20 else "menengah" if conf >= 0.10 else "rendah"

    return (
        f"Sekarang market paling masuk akal dibaca sebagai {quad} dalam stage {stage}. "
        f"Artinya backdrop inti sekarang adalah {QUAD_PLAYBOOK[quad]['meaning']} "
        f"Kalau diterjemahkan ke bahasa sederhana: {', '.join(macro_flags)}. "
        f"Yield curve penting juga: 2Y di {state.y2_now:.2f}%, 10Y di {state.y10_now:.2f}%, 30Y di {state.y30_now:.2f}%, "
        f"2s10s di {two_ten:.2f}, 10s30s di {ten_thirty:.2f}. Ini mengarah ke pembacaan bahwa market sedang {quality}. "
        f"Di fase ini yang biasanya lebih bersih adalah {QUAD_PLAYBOOK[quad]['good']}, sementara yang perlu dihindari adalah {QUAD_PLAYBOOK[quad]['avoid']}. "
        f"Kalau urutan waktunya: early biasanya {QUAD_PLAYBOOK[quad]['early']} Mid biasanya {QUAD_PLAYBOOK[quad]['mid']} "
        f"Late biasanya {QUAD_PLAYBOOK[quad]['late']} Transition clue: {QUAD_PLAYBOOK[quad]['transition']} "
        f"Most likely next quad saat ini adalah {nxt} dengan confidence {conf_text}. "
        f"Itu berarti secara positioning, fokus utama tetap current quad dulu, tapi sudah perlu pantau sinyal transisi ke {nxt} terutama kalau driver lawan mulai muncul."
    )


def current_quad_detail(quad: str) -> str:
    p = QUAD_PLAYBOOK[quad]
    return (
        f"{quad} berarti {p['meaning']} Good: {p['good']}. Avoid: {p['avoid']}. "
        f"Early sign: {p['early']} Mid: {p['mid']} Late: {p['late']} Transition: {p['transition']}"
    )


def next_quad_detail(quad: str) -> str:
    p = QUAD_PLAYBOOK[quad]
    return (
        f"{quad} sebagai fase berikutnya berarti {p['meaning']} "
        f"Kalau transisi ke sana benar-benar jalan, yang biasanya mulai dibangun lebih dulu adalah {p['good']}. "
        f"Yang sering masih salah waktu / perlu dihindari adalah {p['avoid']}. "
        f"Tanda awal fase ini valid biasanya mulai terlihat dari: {p['early']}"
    )


def build_current_proxy_table(quad: str) -> pd.DataFrame:
    return pd.DataFrame(expression_ranking_for_current_quad(quad), columns=["Expression", "Rank", "Why"])


def build_next_proxy_table(quad: str) -> pd.DataFrame:
    return pd.DataFrame(expression_ranking_for_next_quad(quad), columns=["Expression", "Rank", "Why"])


def build_cross_market_rank(state: MacroState) -> pd.DataFrame:
    rows = []
    for asset in ASSET_META:
        rows.append({
            "Aset": asset,
            "Opportunity": f"{asset_opportunity(state, asset):.0%}",
            "Top Risk": f"{asset_risk(state, asset):.0%}",
            "Bottom Formation": f"{asset_bottom(state, asset):.0%}",
            "Drivers": ASSET_META[asset]["drivers"],
        })
    return pd.DataFrame(rows).sort_values(by=["Opportunity", "Top Risk"], ascending=[False, True])


def build_top_bottom_table(state: MacroState) -> pd.DataFrame:
    rows = []
    for asset in TOP_BOTTOM_ASSETS:
        top = asset_risk(state, asset)
        bottom = asset_bottom(state, asset)
        rows.append({
            "Aset": asset,
            "Macro Top Risk": f"{top:.0%}",
            "Top Stage": top_risk_stage(top),
            "Bottom Formation": f"{bottom:.0%}",
            "Bottom Stage": bottom_stage(bottom),
            "Why": ASSET_META[asset]["drivers"],
        })
    return pd.DataFrame(rows)


def build_curve_table(state: MacroState) -> pd.DataFrame:
    rows = [
        ("2Y", f"{state.y2_now:.2f}%", "Front-end / Fed sensitivity"),
        ("10Y", f"{state.y10_now:.2f}%", "Macro benchmark"),
        ("30Y", f"{state.y30_now:.2f}%", "Long-end / term premium"),
        ("2s10s", f"{state.y10_now - state.y2_now:.2f}", "Inverted = policy/growth stress, re-steepen = structure changing"),
        ("10s30s", f"{state.y30_now - state.y10_now:.2f}", "Long-end conviction / pressure"),
        ("10Y real yield", f"{state.ry10_now:.2f}%", "Duration / crypto / EM sensitivity"),
        ("Dollar delta", f"{state.dxy_now - state.dxy_prev:.2f}", "Broad EM / crypto / FX pressure or relief"),
        ("Oil delta", f"{state.oil_now - state.oil_prev:.2f}", "Commodity / inflation impulse"),
        ("Liquidity delta", f"{state.liq_now - state.liq_prev:.2f}", "Risk-on breadth / easing or tightening"),
        ("Credit delta", f"{state.credit_now - state.credit_prev:.2f}", "Stress / spread / confidence"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value", "Read"])


def build_forex_df(quad: str) -> pd.DataFrame:
    return pd.DataFrame(FX_BIAS[quad], columns=["Pair", "Strength", "Why"])


def build_crypto_df(quad: str) -> pd.DataFrame:
    return pd.DataFrame(CRYPTO_RANK[quad], columns=["Bucket", "Rank", "Why"])


def build_terms_df() -> pd.DataFrame:
    return pd.DataFrame(TERMS, columns=["Istilah", "Arti"])


def build_proxy_df(group: str) -> pd.DataFrame:
    return pd.DataFrame(PROXY_GROUPS[group], columns=["Proxy", "Rank", "Directness", "Why"])


def build_divergence_df() -> pd.DataFrame:
    return pd.DataFrame(DIVERGENCE_RULES, columns=["Kasus", "Kenapa"])


def build_chain_df() -> pd.DataFrame:
    return pd.DataFrame(CHAIN_STRENGTH, columns=["Chain", "Impact", "Why"])


def build_scenario_df() -> pd.DataFrame:
    return pd.DataFrame(SCENARIO_MATRIX, columns=["Case", "Read", "What tends to work"])


def build_regime_switch_df() -> pd.DataFrame:
    return pd.DataFrame(REGIME_SWITCH_ARCHETYPES, columns=["Archetype", "Read"])


def build_false_recovery_df() -> pd.DataFrame:
    return pd.DataFrame(FALSE_RECOVERY_MAP, columns=["Trap", "Why it matters"])


def build_crash_types_df() -> pd.DataFrame:
    return pd.DataFrame(CRASH_ENGINE["types"], columns=["Crash type", "Read"])


def build_crash_recovery_df() -> pd.DataFrame:
    return pd.DataFrame(CRASH_ENGINE["recovery"], columns=["Crash family", "Recovery order"])


def confirmation_ladder_text(state: MacroState) -> str:
    parts = []
    if state.oil_now > state.oil_prev:
        parts.append("oil naik")
    if state.y2_now <= state.y2_prev:
        parts.append("2Y relief")
    if state.dxy_now <= state.dxy_prev:
        parts.append("dollar relief")
    if state.credit_now <= state.credit_prev:
        parts.append("credit stabil")
    if state.breadth_now >= state.breadth_prev:
        parts.append("breadth membaik")
    return " -> ".join(parts) if parts else "Belum ada confirmation ladder yang rapi sekarang."


def invalidation_ladder_text(state: MacroState) -> str:
    parts = []
    if state.y2_now > state.y2_prev:
        parts.append("2Y makin keras")
    if state.dxy_now > state.dxy_prev:
        parts.append("dollar makin kuat")
    if state.credit_now > state.credit_prev:
        parts.append("credit memburuk")
    if state.breadth_now < state.breadth_prev:
        parts.append("breadth menyempit")
    if state.oil_now > state.oil_prev and current_quad(state) in ["Quad 3", "Quad 1"]:
        parts.append("oil balik menekan")
    return " -> ".join(parts) if parts else "Belum ada invalidation ladder dominan sekarang."


def opportunity_purity_matrix(state: MacroState) -> pd.DataFrame:
    rows = []
    purity = {
        "Nasdaq": "High",
        "Crypto": "High",
        "Emerging Markets": "Medium",
        "IHSG": "Medium",
        "Energy": "High",
    }
    for asset in ASSET_META:
        rows.append({
            "Aset": asset,
            "Purity": purity[asset],
            "Opportunity": f"{asset_opportunity(state, asset):.0%}",
            "Read": "Paling direct" if purity[asset] == "High" else "Lebih mixed / conditional",
        })
    return pd.DataFrame(rows)


def proxy_migration_text(quad: str) -> str:
    if quad == "Quad 4":
        return "Kalau pure oil sudah terlalu crowded, migrasi alpha paling logis biasanya ke integrated majors, lalu services, lalu selective tanker jika freight confirm."
    if quad == "Quad 1":
        return "Kalau Nasdaq / quality growth sudah terlalu crowded, migrasi biasanya ke BTC, lalu broad EM, lalu small caps kalau breadth benar-benar melebar."
    if quad == "Quad 3":
        return "Kalau bonds dan defensives sudah terlalu crowded, migrasi paling logis biasanya ke quality growth watchlist, lalu BTC, lalu broad risk-on kalau yield/dollar benar-benar relief."
    return "Kalau industrials / cyclicals awal sudah terlalu ramai, migrasi paling logis biasanya ke commodity-linked second-order dan selected small caps yang baru confirm."


def economy_news_positioning_text(state: MacroState, selected_news: List[Dict[str, str]]) -> str:
    quad = current_quad(state)
    base = []
    base.append(f"Current quad sekarang {quad}.")
    base.append("Economic backdrop utama: ")
    base.append(f"growth {'membaik' if state.growth_now > state.growth_prev else 'melemah'}, inflasi {'membaik' if state.inflation_now < state.inflation_prev else 'mengeras'}, 2Y {'relief' if state.y2_now <= state.y2_prev else 'menekan'}, dollar {'relief' if state.dxy_now <= state.dxy_prev else 'menekan'}, credit {'stabil/membaik' if state.credit_now <= state.credit_prev else 'memburuk'}. ")
    if selected_news:
        title = selected_news[0]["title"]
        case, read, action = infer_news_case(selected_news[0]["title"], selected_news[0]["desc"])
        base.append(f"News teratas yang sedang beredar: {title}. Itu paling masuk akal dibaca sebagai {case}. {read} Positioning hint: {action}")
    else:
        base.append("Belum ada news yang terbaca, jadi fokus tetap ke macro state utama.")
    base.append("Pertanyaan utamanya: tetap ikut current quad atau siap next quad? Jawabannya tergantung apakah confirmation ladder current quad masih rapi atau justru invalidation ladder mulai dominan.")
    return " ".join(base)



# --------------------------
# Detailed decision engines
# --------------------------
QUAD_THEMES = {
    "Quad 1": {
        "theme": "Duration / quality risk-on",
        "current_winners": "Nasdaq, quality growth, BTC, broad EM quality, quality cyclicals",
        "first_next": "Cyclicals / reflation sleeves",
        "top_asset": "Nasdaq",
    },
    "Quad 2": {
        "theme": "Cyclical reflation",
        "current_winners": "Industrials, cyclicals, energy, commodity-linked, selected small caps",
        "first_next": "Hard asset / stagflation or defensive disinflation sleeves",
        "top_asset": "Energy",
    },
    "Quad 3": {
        "theme": "Disinflation / defensive relief",
        "current_winners": "Bonds, defensives, gold, quality balance-sheet names",
        "first_next": "Duration growth / BTC / broad risk-on quality",
        "top_asset": "Nasdaq",
    },
    "Quad 4": {
        "theme": "Stagflation / hard-asset defense",
        "current_winners": "Energy, hard assets, pricing-power defensives, selective exporters",
        "first_next": "Defensives / bonds first, then duration if yields and dollar really fade",
        "top_asset": "Energy",
    },
}


def safe_pct_delta(now: float, prev: float) -> float:
    if prev in [0, None] or pd.isna(prev) or pd.isna(now):
        return 0.0
    return float((now / prev - 1.0) * 100.0)


def delta(now: float, prev: float) -> float:
    if pd.isna(now) or pd.isna(prev):
        return 0.0
    return float(now - prev)


def pct_bucket(value: float, pos_hi: float, pos_lo: float, neg_lo: float, neg_hi: float) -> str:
    if value >= pos_hi:
        return "Naik keras"
    if value >= pos_lo:
        return "Naik"
    if value <= neg_hi:
        return "Turun keras"
    if value <= neg_lo:
        return "Turun"
    return "Flat / stall"


def bool_word(x: bool) -> str:
    return "Yes" if x else "No"


def treasury_regime_text(state: MacroState) -> Tuple[str, str]:
    dy2, dy10 = delta(state.y2_now, state.y2_prev), delta(state.y10_now, state.y10_prev)
    curve_now = state.y10_now - state.y2_now
    curve_prev = state.y10_prev - state.y2_prev
    dcurve = curve_now - curve_prev

    if dy2 > 0 and dy10 > 0:
        label = "Bear steepening" if dcurve > 0 else "Bear flattening"
    elif dy2 < 0 and dy10 < 0:
        label = "Bull steepening" if dcurve > 0 else "Bull flattening"
    elif dy2 > 0 and dy10 <= 0:
        label = "Front-end hawkish squeeze"
    elif dy2 <= 0 and dy10 > 0:
        label = "Long-end term premium pressure"
    else:
        label = "Mixed / indecisive"

    if label == "Bear steepening":
        read = "Long-end naik lebih cepat. Reflation atau term premium hidup; duration valuation tetap berat."
    elif label == "Bear flattening":
        read = "Front-end lebih agresif. Biasanya hawkish / late-cycle dan tidak ramah buat broad beta."
    elif label == "Bull steepening":
        read = "Front-end relief lebih cepat dari long-end. Sering cocok untuk early transition ke next phase."
    elif label == "Bull flattening":
        read = "Seluruh curve turun, tapi long-end lebih jinak. Defensive relief masih dominan."
    elif label == "Front-end hawkish squeeze":
        read = "2Y naik sementara 10Y tidak ikut. Ini sering bikin duration, crypto, dan EM susah bernapas."
    elif label == "Long-end term premium pressure":
        read = "10Y/30Y naik tanpa front-end shock besar. Duration equity masih bisa sesak walau Fed tone tidak ekstrem."
    else:
        read = "Rates belum kasih pesan yang benar-benar bersih."
    return label, read


def news_premium_status(state: MacroState, news_items: List[Dict[str, str]]) -> Tuple[str, str]:
    if not news_items:
        return "No strong news premium", "Tidak ada headline dominan yang cukup kuat untuk mengubah pembacaan macro."
    geo = 0
    growth = 0
    inflation = 0
    for item in news_items[:6]:
        case, _, _ = infer_news_case(item["title"], item["desc"])
        title = (item["title"] + " " + item["desc"]).lower()
        if case == "Geopolitical / oil shock" or any(k in title for k in ["war", "iran", "strait", "hormuz", "attack", "missile"]):
            geo += 1
        elif case in ["Growth slowdown / demand scare", "Growth recovery / reflation"]:
            growth += 1
        elif case == "Inflation / oil shock":
            inflation += 1

    oil_pct = safe_pct_delta(state.oil_now, state.oil_prev)
    gold_pct = safe_pct_delta(state.gold_now, state.gold_prev)

    if geo > 0 and oil_pct > 6:
        return "War premium alive", "Headline geopolitik masih benar-benar diterjemahkan market menjadi premium di oil."
    if geo > 0 and oil_pct <= 1:
        return "War premium fading", "Headline geopolitik ada, tapi oil tidak lagi merespons sebesar sebelumnya."
    if inflation > 0 and oil_pct > 3:
        return "Inflation premium alive", "Headline inflasi / energi masih sinkron dengan harga oil."
    if growth > 0 and oil_pct < -3 and gold_pct > 0:
        return "Growth scare hedge bid", "News lebih cocok dibaca sebagai fear / slowdown; oil turun sementara gold tetap dicari."
    return "Mixed news premium", "Headline ada, tetapi belum ada satu premium dominan yang benar-benar mengontrol tape."


def current_phase_maturity(state: MacroState, quad: str, news_items: List[Dict[str, str]]) -> Tuple[str, float, List[str], List[str]]:
    lead_asset = QUAD_THEMES[quad]["top_asset"]
    top_score = asset_risk(state, lead_asset)
    bottom_score = asset_bottom(state, lead_asset)
    transition_score = transition_quality_score(state, news_items)["score"]
    confirms, failures = [], []

    if state.breadth_now >= state.breadth_prev:
        confirms.append("Breadth tidak menyempit.")
    else:
        failures.append("Breadth menyempit; leadership makin sempit.")

    if state.credit_now <= state.credit_prev:
        confirms.append("Credit tidak memburuk.")
    else:
        failures.append("Credit memburuk; fase sekarang jadi lebih rapuh.")

    if quad in ["Quad 2", "Quad 4"]:
        if state.oil_now > state.oil_prev:
            confirms.append("Oil masih ikut mendukung current phase.")
        else:
            failures.append("Oil berhenti mendukung current winners.")
    else:
        if state.y2_now <= state.y2_prev and state.ry10_now <= state.ry10_prev:
            confirms.append("Front-end dan real yields masih memberi napas ke current winners.")
        else:
            failures.append("2Y / real yields tidak lagi sebersih sebelumnya untuk current winners.")

    if top_score >= 0.72:
        maturity = "Topping / crowded"
    elif top_score >= 0.56 or transition_score >= 0.60:
        maturity = "Late"
    elif top_score >= 0.36:
        maturity = "Mid"
    else:
        maturity = "Early"
    return maturity, top_score, confirms, failures


def transition_quality_score(state: MacroState, news_items: List[Dict[str, str]]) -> Dict[str, object]:
    points = []
    score = 0.0

    dy2 = delta(state.y2_now, state.y2_prev)
    dry = delta(state.ry10_now, state.ry10_prev)
    ddxy = delta(state.dxy_now, state.dxy_prev)
    dcredit = delta(state.credit_now, state.credit_prev)
    dbreadth = delta(state.breadth_now, state.breadth_prev)
    doil = delta(state.oil_now, state.oil_prev)
    dgold = safe_pct_delta(state.gold_now, state.gold_prev)

    if dy2 <= 0:
        score += 0.18
        points.append(("2Y relief", "pass", "Front-end stress mereda."))
    else:
        points.append(("2Y relief", "fail", "Front-end masih menekan."))

    if dry <= 0:
        score += 0.16
        points.append(("Real yields relief", "pass", "Duration / crypto / quality growth dapat napas."))
    else:
        points.append(("Real yields relief", "fail", "Real yields masih jadi lawan."))

    if ddxy <= 0:
        score += 0.14
        points.append(("Dollar relief", "pass", "EM / crypto / FX beta lebih mungkin ikut."))
    else:
        points.append(("Dollar relief", "fail", "Dollar masih dominan."))

    if dcredit <= 0:
        score += 0.12
        points.append(("Credit stable", "pass", "Stress sistemik tidak bertambah."))
    else:
        points.append(("Credit stable", "fail", "Credit memburuk; transisi berisiko rapuh."))

    if dbreadth >= 0:
        score += 0.10
        points.append(("Breadth improves", "pass", "Leadership melebar."))
    else:
        points.append(("Breadth improves", "fail", "Leadership tetap sempit."))

    if doil <= 0:
        score += 0.10
        points.append(("Oil relief", "pass", "Tekanan hard-inflation mereda."))
    else:
        points.append(("Oil relief", "fail", "Oil masih menambah tekanan."))

    if pd.notna(state.gold_now):
        if dgold > 1.5 and doil <= 0:
            score += 0.05
            points.append(("Gold vs oil split", "pass", "Gold dicari saat oil melemah: hedge/fear signal, bukan reflation murni."))
        elif dgold < -1 and doil < 0:
            points.append(("Gold vs oil split", "mixed", "Oil turun tapi gold juga lemah: lebih cocok ke rate / dollar story daripada fear hedge."))
        else:
            points.append(("Gold vs oil split", "mixed", "Gold belum memberi sinyal yang benar-benar bersih."))

    premium, premium_read = news_premium_status(state, news_items)
    if premium in ["War premium fading", "Growth scare hedge bid"]:
        score += 0.05
    points.append(("News premium", "mixed", premium_read))

    if score >= 0.82:
        label = "Confirmed transition"
    elif score >= 0.68:
        label = "Clean transition"
    elif score >= 0.54:
        label = "Fragile transition"
    elif score >= 0.40:
        label = "Narrow / early transition"
    else:
        label = "No clean transition"
    return {"score": float(score), "label": label, "points": points}


def current_top_diagnosis(state: MacroState, quad: str, news_items: List[Dict[str, str]]) -> Dict[str, object]:
    dy2 = delta(state.y2_now, state.y2_prev)
    dry = delta(state.ry10_now, state.ry10_prev)
    ddxy = delta(state.dxy_now, state.dxy_prev)
    doil = safe_pct_delta(state.oil_now, state.oil_prev)
    dgold = safe_pct_delta(state.gold_now, state.gold_prev)
    dcredit = delta(state.credit_now, state.credit_prev)
    dbreadth = delta(state.breadth_now, state.breadth_prev)
    premium, premium_read = news_premium_status(state, news_items)

    starting, confirm, invalidate, traps = [], [], [], []
    score = 0.0
    theme = QUAD_THEMES[quad]["theme"]

    if quad == "Quad 1":
        if dy2 > 0:
            starting.append("2Y berhenti relief / naik lagi. Biasanya ini starting point paling awal buat top duration-risk-on.")
            score += 0.18
        if dry > 0:
            starting.append("10Y real yield naik lagi. Valuation support buat Nasdaq/BTC menipis.")
            score += 0.18
        if ddxy > 0:
            starting.append("Dollar balik kuat. Risk-on quality bisa masih hijau, tapi breadth sering ikut menyempit.")
            score += 0.12
        if dbreadth < 0:
            confirm.append("Breadth menyempit. Ini tanda top lebih valid daripada sekadar price masih bikin high.")
            score += 0.14
        if dcredit > 0:
            confirm.append("Credit memburuk. Quality masih bisa bertahan sebentar, tapi extension jadi lebih rapuh.")
            score += 0.14
        traps.append("Nasdaq masih naik bukan berarti regime masih bersih; kalau hanya mega-cap yang jalan, itu justru pola late-stage.")
        traps.append("BTC naik tapi alts tidak ikut biasanya quality-led saja, belum broad crypto confirmation.")
        invalidate.append("Kalau 2Y dan real yields kembali relief, dollar tidak dominan, dan breadth membaik, top risk turun lagi.")
    elif quad == "Quad 2":
        if 0 <= doil < 4:
            starting.append("Oil masih naik tapi dorongannya tidak sekuat fase awal. Itu sering jadi starting point reflation mulai capek.")
            score += 0.12
        if ddxy > 0:
            starting.append("Dollar ikut menguat. Ini bikin reflation sehat berubah jadi trade yang lebih sempit dan rapuh.")
            score += 0.12
        if dbreadth < 0:
            confirm.append("Cyclicals tetap hidup tapi breadth tidak melebar. Leadership mulai terlalu sempit.")
            score += 0.14
        if dcredit > 0:
            confirm.append("Credit memburuk. Reflation berubah dari broad constructive ke late-cycle pressure.")
            score += 0.14
        if dgold > 1.5 and doil <= 2:
            confirm.append("Gold mulai dicari sementara oil kehilangan tenaga. Market pindah dari reflation ke hedge/fear.")
            score += 0.12
        traps.append("Industrials / cyclicals masih kuat tidak otomatis berarti top belum dekat; cek apakah second-order proxies masih confirm.")
        invalidate.append("Kalau oil kembali impulsif, breadth melebar, dan credit membaik, reflation top belum matang.")
    elif quad == "Quad 3":
        if dy2 <= 0 and ddxy < 0 and dbreadth > 0:
            starting.append("Rates relief + dollar relief + breadth membaik. Ini starting point paling umum untuk defensive phase mulai kehilangan dominasi.")
            score += 0.18
        if dry <= 0 and dgold <= 0:
            starting.append("Gold tidak lagi menjadi satu-satunya refuge. Defensive crowding mulai kelihatan.")
            score += 0.10
        if dbreadth > 0:
            confirm.append("Breadth membaik. Quality growth dan risk assets mulai dapat sponsor baru.")
            score += 0.14
        if dcredit <= 0:
            confirm.append("Credit tidak memburuk. Itu syarat penting agar defensive top lebih valid.")
            score += 0.12
        traps.append("Bonds/gold kuat terus bukan berarti defensive tidak mau top; cek apakah growth stabilization diam-diam jalan.")
        invalidate.append("Kalau breadth balik jelek, credit memburuk, dan dollar balik dominan, defensive top tertunda.")
    else:  # Quad 4
        if 0 <= doil < 4:
            starting.append("Oil masih tinggi tapi impulse-nya melemah. Itu starting point klasik hard-asset winner mulai capek.")
            score += 0.14
        if premium == "War premium fading":
            confirm.append("Headline geopolitik masih ramai tapi oil tidak merespons sebesar sebelumnya. Premium mulai pudar.")
            score += 0.16
        if dgold > 1.5 and doil <= 0:
            confirm.append("Gold naik saat oil melemah. Itu tanda market pindah dari inflation beta ke hedge / fear hedge.")
            score += 0.14
        if dbreadth < 0:
            confirm.append("Broad beta tetap rapuh; old winners jadi makin crowded dan risk/reward menipis.")
            score += 0.10
        if dcredit > 0:
            confirm.append("Credit memburuk; shock theme makin sempit dan rentan gagal lanjut.")
            score += 0.12
        traps.append("Energy equities bisa tetap hijau walau oil sudah kehilangan tenaga. Harga indeks belum tentu bilang top belum dekat.")
        invalidate.append("Kalau oil re-accelerate, breakeven tetap naik, dan second-order chain seperti services ikut confirm, top tertunda.")

    if not starting:
        starting.append("Belum ada starting point yang benar-benar tegas; current winners masih relatif ditopang driver utamanya.")
    if not confirm:
        confirm.append("Belum ada confirmation yang cukup kuat untuk bilang top sudah matang.")
    if score >= 0.72:
        label = "Late / crowded topping"
    elif score >= 0.52:
        label = "Rising top risk"
    elif score >= 0.34:
        label = "Early top warning"
    else:
        label = "No clean top yet"

    interpretation = (
        f"Current phase sekarang bertema {theme}. Pembacaan top yang benar bukan hanya lihat price masih naik atau tidak, "
        f"tapi apakah driver pendukungnya melemah, breadth menyempit, dan premium berita makin tidak efektif. {premium_read}"
    )
    return {
        "label": label,
        "score": float(np.clip(score, 0, 1)),
        "theme": theme,
        "starting": starting,
        "confirm": confirm,
        "invalidate": invalidate,
        "traps": traps,
        "interpretation": interpretation,
    }


def next_phase_activation(state: MacroState, next_quad: str, news_items: List[Dict[str, str]]) -> Dict[str, object]:
    dy2 = delta(state.y2_now, state.y2_prev)
    dry = delta(state.ry10_now, state.ry10_prev)
    ddxy = delta(state.dxy_now, state.dxy_prev)
    doil = safe_pct_delta(state.oil_now, state.oil_prev)
    dgold = safe_pct_delta(state.gold_now, state.gold_prev)
    dcredit = delta(state.credit_now, state.credit_prev)
    dbreadth = delta(state.breadth_now, state.breadth_prev)

    starting, stabilization, confirmation, traps = [], [], [], []
    score = 0.0
    theme = QUAD_THEMES[next_quad]["theme"]

    if next_quad == "Quad 1":
        if dy2 <= 0:
            starting.append("2Y tidak naik lagi / mulai relief. Ini starting point utama untuk phase duration-friendly.")
            score += 0.18
        if dry <= 0:
            starting.append("Real yields berhenti menekan. Support valuasi mulai muncul.")
            score += 0.16
        if ddxy <= 0:
            stabilization.append("Dollar relief. EM / crypto / beta quality punya ruang ikut.")
            score += 0.14
        if dcredit <= 0:
            stabilization.append("Credit tidak memburuk. Bounce lebih mungkin berubah jadi base-building.")
            score += 0.10
        if dbreadth >= 0:
            confirmation.append("Breadth membaik. Itu pembeda utama antara quality-led bounce vs true phase activation.")
            score += 0.12
        if doil <= 0:
            confirmation.append("Oil relief. Tekanan hard-inflation mereda dan membantu disinflation / rate relief.")
            score += 0.08
        traps.append("Kalau cuma Nasdaq atau BTC yang pulih sementara breadth, EM, dan credit tidak confirm, itu belum full Quad 1 activation.")
    elif next_quad == "Quad 2":
        if state.growth_now > state.growth_prev:
            starting.append("Growth composite membaik. Ini fondasi paling awal untuk cyclical reflation.")
            score += 0.16
        if state.inflation_now > state.inflation_prev or doil > 2:
            starting.append("Inflation / commodity impulse hidup kembali. Reflation mulai plausible.")
            score += 0.14
        if dbreadth >= 0:
            stabilization.append("Breadth membaik. Ciri baik untuk cyclical expansion.")
            score += 0.10
        if dcredit <= 0:
            confirmation.append("Credit stabil. Reflation lebih sehat kalau bukan sekadar short squeeze.")
            score += 0.10
        if ddxy <= 1.0:
            confirmation.append("Dollar tidak terlalu mengganggu. Itu penting agar move tetap broad.")
            score += 0.08
        traps.append("Kalau cuma oil yang naik tapi broad cyclicals dan breadth tidak ikut, itu selective shock, bukan clean Quad 2.")
    elif next_quad == "Quad 3":
        if doil <= 0:
            starting.append("Oil turun / shock energi mereda. Ini starting point utama untuk defensive-disinflation phase.")
            score += 0.18
        if state.inflation_now <= state.inflation_prev:
            stabilization.append("Inflation composite melunak. Defensives, bonds, gold dapat sponsor.")
            score += 0.14
        if dy2 <= 0:
            stabilization.append("2Y relief. Front-end tidak lagi menambah tekanan.")
            score += 0.10
        if dgold > 0:
            confirmation.append("Gold ikut dicari. Ini selaras dengan defensive / hedge bid.")
            score += 0.08
        if dcredit > 0:
            traps.append("Kalau credit memburuk terlalu cepat, jangan kira semua defensive strength itu sehat; bisa juga crash dynamic.")
        traps.append("Oil turun sendirian tidak cukup; harus dibedakan antara good disinflation vs bad growth scare.")
    else:  # Quad 4
        if doil >= 2:
            starting.append("Oil / hard-asset impulse hidup lagi. Ini starting point utama untuk Quad 4 style pressure.")
            score += 0.16
        if state.inflation_now > state.inflation_prev:
            stabilization.append("Inflation composite membaik ke atas. Shock tidak lagi cuma headline.")
            score += 0.12
        if ddxy >= 0:
            confirmation.append("Dollar tidak relief. Broad beta jadi susah, sementara selective hard asset tetap lebih menarik.")
            score += 0.10
        if dbreadth < 0:
            confirmation.append("Breadth sempit. Itu cocok dengan pola stagflation: selective winners, broad losers.")
            score += 0.10
        traps.append("Kalau oil naik tapi gold, breakeven, dan second-order commodity chain tidak confirm, bisa cuma spike pendek.")
    if not starting:
        starting.append("Belum ada starting point yang benar-benar bersih untuk next phase.")
    if not stabilization:
        stabilization.append("Stabilization belum rapi; tekanan lama masih terlalu dominan.")
    if not confirmation:
        confirmation.append("Confirmation chain next phase belum lengkap.")
    if score >= 0.72:
        label = "Confirmed activation"
    elif score >= 0.55:
        label = "Base-building"
    elif score >= 0.38:
        label = "Early stabilization"
    elif score >= 0.22:
        label = "Starting point only"
    else:
        label = "Not active yet"
    interpretation = (
        f"Next phase yang paling plausible adalah {next_quad} ({theme}). Bottoming / activation yang sehat selalu datang bertahap: "
        f"tekanan utama berhenti memburuk -> leaders tahan banting -> breadth ikut -> bad news makin tidak efektif."
    )
    return {
        "label": label,
        "score": float(np.clip(score, 0, 1)),
        "theme": theme,
        "starting": starting,
        "stabilization": stabilization,
        "confirmation": confirmation,
        "traps": traps,
        "interpretation": interpretation,
    }


def build_cross_signal_dashboard(state: MacroState, current_quad_name: str, next_quad_name: str, news_items: List[Dict[str, str]]) -> pd.DataFrame:
    premium, premium_read = news_premium_status(state, news_items)
    oil_pct = safe_pct_delta(state.oil_now, state.oil_prev)
    gold_pct = safe_pct_delta(state.gold_now, state.gold_prev)
    dy2 = delta(state.y2_now, state.y2_prev)
    dry = delta(state.ry10_now, state.ry10_prev)
    ddxy = delta(state.dxy_now, state.dxy_prev)
    dcredit = delta(state.credit_now, state.credit_prev)
    dbreadth = delta(state.breadth_now, state.breadth_prev)

    rows = [
        {
            "Signal": "Oil",
            "Now": f"{pct_bucket(oil_pct, 6, 1, -1, -6)} ({oil_pct:.1f}%)",
            "Current-phase top bias": ("Top risk naik" if current_quad_name in ["Quad 2", "Quad 4"] and oil_pct <= 2 else ("Top risk turun" if current_quad_name in ["Quad 1", "Quad 3"] and oil_pct <= -2 else "Mixed")),
            "Next-phase bottom bias": ("Good for next phase" if next_quad_name in ["Quad 1", "Quad 3"] and oil_pct <= -2 else ("Supports next phase" if next_quad_name in ["Quad 2", "Quad 4"] and oil_pct >= 2 else "Mixed")),
            "Interpretation": "Oil turun bisa berarti disinflation relief atau growth scare; baca bersama credit, dollar, dan breadth.",
        },
        {
            "Signal": "Gold",
            "Now": f"{pct_bucket(gold_pct, 4, 1, -1, -4)} ({gold_pct:.1f}%)",
            "Current-phase top bias": ("Fear hedge rising" if gold_pct > 1.5 and oil_pct <= 0 else ("Inflation hedge still alive" if gold_pct > 1.5 and oil_pct > 0 else "Mixed")),
            "Next-phase bottom bias": ("Helpful for Quad 3" if next_quad_name == "Quad 3" and gold_pct > 0 else ("Can support transition" if gold_pct > 0 and oil_pct <= 0 else "Mixed")),
            "Interpretation": "Gold naik + oil turun biasanya lebih dekat ke hedge/fear bid; gold naik + oil naik lebih dekat ke inflation/geopolitics mix.",
        },
        {
            "Signal": "2Y",
            "Now": f"{pct_bucket(dy2, 0.35, 0.05, -0.05, -0.35)} ({dy2:.2f})",
            "Current-phase top bias": ("Top risk naik" if current_quad_name in ["Quad 1", "Quad 3"] and dy2 > 0 else ("Top risk turun" if current_quad_name in ["Quad 2", "Quad 4"] and dy2 <= 0 else "Mixed")),
            "Next-phase bottom bias": ("Good for duration next phase" if next_quad_name in ["Quad 1", "Quad 3"] and dy2 <= 0 else ("Bad for risk-on bottom" if next_quad_name == "Quad 1" and dy2 > 0 else "Mixed")),
            "Interpretation": "Front-end biasanya starting point paling cepat untuk perubahan regime.",
        },
        {
            "Signal": "10Y real yield",
            "Now": f"{pct_bucket(dry, 0.25, 0.05, -0.05, -0.25)} ({dry:.2f})",
            "Current-phase top bias": ("Duration squeeze" if current_quad_name == "Quad 1" and dry > 0 else ("Defensive top risk" if current_quad_name == "Quad 3" and dry <= 0 else "Mixed")),
            "Next-phase bottom bias": ("Supports Quad 1" if next_quad_name == "Quad 1" and dry <= 0 else ("Supports hard-asset resistance" if next_quad_name == "Quad 4" and dry > 0 else "Mixed")),
            "Interpretation": "Real yields sering lebih tajam dari nominal yields untuk baca Nasdaq / crypto / gold.",
        },
        {
            "Signal": "Dollar",
            "Now": f"{pct_bucket(ddxy, 2.0, 0.3, -0.3, -2.0)} ({ddxy:.2f})",
            "Current-phase top bias": ("Top risk naik" if current_quad_name in ["Quad 1", "Quad 2"] and ddxy > 0 else ("Helps current phase" if current_quad_name in ["Quad 3", "Quad 4"] and ddxy > 0 else "Mixed")),
            "Next-phase bottom bias": ("Helps next phase" if next_quad_name in ["Quad 1", "Quad 3"] and ddxy <= 0 else ("Bad for EM/crypto bottom" if next_quad_name == "Quad 1" and ddxy > 0 else "Mixed")),
            "Interpretation": "Dollar kuat sering membuat move menjadi selective, bukan broad.",
        },
        {
            "Signal": "Credit",
            "Now": f"{pct_bucket(dcredit, 0.5, 0.1, -0.1, -0.5)} ({dcredit:.2f})",
            "Current-phase top bias": ("Top risk naik" if dcredit > 0 else "Top risk turun"),
            "Next-phase bottom bias": ("More valid" if dcredit <= 0 else "Fragile only"),
            "Interpretation": "Credit membedakan antara healthy transition dan sekadar oversold bounce.",
        },
        {
            "Signal": "Breadth",
            "Now": f"{pct_bucket(dbreadth, 4, 0.3, -0.3, -4)} ({dbreadth:.2f})",
            "Current-phase top bias": ("Top risk naik" if dbreadth < 0 else "Current phase still broad"),
            "Next-phase bottom bias": ("More valid" if dbreadth >= 0 else "Still narrow"),
            "Interpretation": "Breadth adalah filter paling penting untuk bedakan extension sempit vs regime flip yang bersih.",
        },
        {
            "Signal": "News premium",
            "Now": premium,
            "Current-phase top bias": ("Top risk naik" if premium in ["War premium fading", "Growth scare hedge bid"] else "Mixed"),
            "Next-phase bottom bias": ("Can help next phase" if premium in ["War premium fading", "Growth scare hedge bid"] else "Mixed"),
            "Interpretation": premium_read,
        },
    ]
    return pd.DataFrame(rows)


def build_executive_summary_table(state: MacroState, current_quad_name: str, next_quad_name: str, news_items: List[Dict[str, str]]) -> pd.DataFrame:
    maturity, top_score, confirms, failures = current_phase_maturity(state, current_quad_name, news_items)
    top_diag = current_top_diagnosis(state, current_quad_name, news_items)
    next_diag = next_phase_activation(state, next_quad_name, news_items)
    trans = transition_quality_score(state, news_items)
    treasury_label, treasury_read = treasury_regime_text(state)
    premium, _ = news_premium_status(state, news_items)

    rows = [
        ("Current Quad", current_quad_name, QUAD_THEMES[current_quad_name]["theme"]),
        ("Current maturity", maturity, "; ".join((confirms or failures)[:2])),
        ("Current top risk", top_diag["label"], f"Score {top_diag['score']:.0%}"),
        ("Next likely Quad", next_quad_name, QUAD_THEMES[next_quad_name]["theme"]),
        ("Next-phase activation", next_diag["label"], f"Score {next_diag['score']:.0%}"),
        ("Transition quality", trans["label"], f"Score {trans['score']:.0%}"),
        ("Treasury regime", treasury_label, treasury_read),
        ("Dominant news premium", premium, "Headline premium yang paling mungkin sedang mengganggu tape."),
    ]
    return pd.DataFrame(rows, columns=["Item", "Status", "Why it matters"])


def build_em_region_table(state: MacroState, current_quad_name: str, next_quad_name: str) -> pd.DataFrame:
    ddxy = delta(state.dxy_now, state.dxy_prev)
    doil = safe_pct_delta(state.oil_now, state.oil_prev)
    rows = []

    selective = "Strong" if doil > 2 and ddxy < 2 else "Selective only"
    broad = "Improving" if ddxy <= 0 and state.breadth_now >= state.breadth_prev else "Fragile"
    emfx = "Relief" if ddxy <= 0 else "Pressure"
    weak_importer = "Under pressure" if doil > 2 or ddxy > 0 else "Less bad"
    ihsg = "Selective commodity support" if doil > 2 else ("Macro relief support" if ddxy <= 0 else "Mixed")

    rows.extend([
        ("Selective exporter EM", selective, "Oil / commodity pulse, dollar not too dominant", "Dollar squeeze + growth scare"),
        ("Broad EM", broad, "Dollar relief + breadth improve + credit stable", "Dollar up / credit worse / breadth weak"),
        ("EM FX", emfx, "DXY and real-yield relief", "Dollar dominance"),
        ("Weak importer EM", weak_importer, "Oil relief + dollar relief", "Oil up + dollar up"),
        ("IHSG sub-case", ihsg, "Commodity support or dollar relief", "Dollar squeeze / global stress"),
    ])
    return pd.DataFrame(rows, columns=["Sleeve", "Now", "What helps", "What breaks"])


def build_us_crypto_fx_table(current_quad_name: str, next_quad_name: str) -> pd.DataFrame:
    rows = [
        ("US quality growth", QUAD_THEMES[current_quad_name]["current_winners"], QUAD_THEMES[next_quad_name]["first_next"]),
        ("Crypto", "BTC first, then ETH, then high beta if liquidity broadens", "Watch whether breadth broadens beyond BTC"),
        ("FX", "Read DXY first, then AUDUSD / EURUSD / USDJPY / EM FX", "Need dollar relief for broad next-phase confirmation"),
    ]
    return pd.DataFrame(rows, columns=["Block", "Current read", "Next-phase read"])


def bullet_block(title: str, items: List[str]) -> str:
    if not items:
        return f"**{title}**\n- None"
    return "**" + title + "**\n" + "\n".join([f"- {x}" for x in items])


def render_list(items: List[str]) -> None:
    for item in items:
        st.markdown(f"- {item}")


def compact_kpi(label: str, value: str, help_text: str = "") -> None:
    st.metric(label, value, help=help_text if help_text else None)




def first_nonempty(items: List[str], fallback: str = "-") -> str:
    for item in items:
        if item and str(item).strip():
            return str(item)
    return fallback


def join_top(df: pd.DataFrame, label_col: str, rank_col: str | None = None, n: int = 3) -> str:
    if df is None or df.empty:
        return "-"
    take = df.head(n)
    parts = []
    for _, row in take.iterrows():
        label = str(row[label_col])
        if rank_col and rank_col in row:
            parts.append(f"{label} ({row[rank_col]})")
        else:
            parts.append(label)
    return ", ".join(parts)


def best_proxy_group_for_quad(quad: str) -> str:
    mapping = {
        "Quad 1": "Rates / dollar / liquidity chain",
        "Quad 2": "Metals / mining chain",
        "Quad 3": "Rates / dollar / liquidity chain",
        "Quad 4": "Oil / energy chain",
    }
    return mapping.get(quad, "Rates / dollar / liquidity chain")


def correlation_focus_text(quad: str) -> str:
    mapping = {
        "Quad 1": "2Y relief -> real yields turun -> Nasdaq/BTC hidup -> broad EM -> baru late beta. Kalau hanya mega-cap/BTC yang jalan, move masih sempit.",
        "Quad 2": "Growth/inflation naik -> cyclicals/industrials/commodity-linked memimpin. Kalau broad beta tak ikut dan dollar terlalu kuat, reflation belum bersih.",
        "Quad 3": "Oil/inflation turun -> bonds/defensives/gold duluan. Kalau later 2Y, real yields, dan dollar ikut relief, rotation ke duration growth mulai valid.",
        "Quad 4": "Oil / inflation shock -> pure energy / hard asset duluan. Kalau broad EM atau broad beta tak ikut, itu normal; fokus ke first-order winners dulu.",
    }
    return mapping.get(quad, "Baca chain dari driver utama ke leader, lalu baru second-order proxies.")


def phase_em_focus_text(state: MacroState, quad: str) -> str:
    em_df = build_em_region_table(state, quad, quad)
    def get_row(name: str) -> str:
        hit = em_df.loc[em_df["Sleeve"] == name, "Now"]
        return str(hit.iloc[0]) if not hit.empty else "Mixed"
    if quad in ["Quad 2", "Quad 4"]:
        return (
            f"Selective exporter EM = {get_row('Selective exporter EM')}; "
            f"Broad EM = {get_row('Broad EM')}; IHSG = {get_row('IHSG sub-case')}. "
            "Kalau commodity shock masih hidup, selective exporter dan local commodity proxies biasanya lebih bersih daripada broad EM."
        )
    return (
        f"Broad EM = {get_row('Broad EM')}; EM FX = {get_row('EM FX')}; IHSG = {get_row('IHSG sub-case')}. "
        "Kalau dollar dan real yields relief, broad EM/EM FX bisa ikut; kalau belum, move sering cuma selective."
    )


def phase_divergence_focus(quad: str) -> str:
    mapping = {
        "Quad 1": "Nasdaq/BTC naik tapi breadth, alts, dan broad EM tidak ikut = masih quality-led, belum full regime confirmation.",
        "Quad 2": "Oil/cyclicals naik tapi breadth dan second-order industrial chain tidak ikut = reflation masih sempit.",
        "Quad 3": "Bonds/defensives/gold kuat tapi growth stabilization mulai muncul = defensive strength bisa mulai crowded.",
        "Quad 4": "Oil naik tapi tanker/services/broad EM tak ikut = shock lebih sempit daripada kelihatannya.",
    }
    return mapping.get(quad, "Cek apakah direct proxy lebih kuat daripada followers; kalau iya, chain belum lengkap.")


def build_compare_board_table(state: MacroState, current_quad_name: str, next_quad_name: str, news_items: List[Dict[str, str]], mode: str) -> pd.DataFrame:
    treasury_label, treasury_read = treasury_regime_text(state)
    premium_label, premium_read = news_premium_status(state, news_items)
    trans = transition_quality_score(state, news_items)
    if mode == "current":
        maturity, _, _, _ = current_phase_maturity(state, current_quad_name, news_items)
        top_diag = current_top_diagnosis(state, current_quad_name, news_items)
        rows = [
            ("Quad", f"{current_quad_name} — {QUAD_THEMES[current_quad_name]['theme']}"),
            ("Lifecycle", maturity),
            ("Treasury", f"{treasury_label}: {treasury_read}"),
            ("Current winners", QUAD_THEMES[current_quad_name]["current_winners"]),
            ("Top status", f"{top_diag['label']} ({top_diag['score']:.0%})"),
            ("Starting point", first_nonempty(top_diag["starting"])),
            ("Confirmation", first_nonempty(top_diag["confirm"])),
            ("Invalidation", first_nonempty(top_diag["invalidate"])),
            ("Trap", first_nonempty(top_diag["traps"])),
            ("News premium", f"{premium_label}: {premium_read}"),
            ("Transition quality", trans["label"]),
        ]
    else:
        next_diag = next_phase_activation(state, next_quad_name, news_items)
        rows = [
            ("Next likely quad", f"{next_quad_name} — {QUAD_THEMES[next_quad_name]['theme']}"),
            ("Activation", f"{next_diag['label']} ({next_diag['score']:.0%})"),
            ("Treasury", f"{treasury_label}: {treasury_read}"),
            ("First likely winners", QUAD_THEMES[next_quad_name]["first_next"]),
            ("Starting point", first_nonempty(next_diag["starting"])),
            ("Early stabilization", first_nonempty(next_diag["stabilization"])),
            ("Confirmation", first_nonempty(next_diag["confirmation"])),
            ("False bottom trap", first_nonempty(next_diag["traps"])),
            ("News premium", f"{premium_label}: {premium_read}"),
            ("Transition quality", trans["label"]),
        ]
    return pd.DataFrame(rows, columns=["Field", "Read"])


def build_phase_signal_story_table(state: MacroState, current_quad_name: str, next_quad_name: str, news_items: List[Dict[str, str]], mode: str) -> pd.DataFrame:
    dash = build_cross_signal_dashboard(state, current_quad_name, next_quad_name, news_items)
    bias_col = "Current-phase top bias" if mode == "current" else "Next-phase bottom bias"
    rows = []
    for _, row in dash.iterrows():
        rows.append({
            "Signal": row["Signal"],
            "Read": f"{row['Now']}. {row[bias_col]}. {row['Interpretation']}"
        })
    return pd.DataFrame(rows)


def build_integrated_stack_table(state: MacroState, current_quad_name: str, next_quad_name: str, news_items: List[Dict[str, str]], mode: str) -> pd.DataFrame:
    quad_name = current_quad_name if mode == "current" else next_quad_name
    proxy_df = build_current_proxy_table(quad_name) if mode == "current" else build_next_proxy_table(quad_name)
    fx_df = build_forex_df(quad_name)
    crypto_df = build_crypto_df(quad_name)
    chain_group = best_proxy_group_for_quad(quad_name)
    chain_df = build_proxy_df(chain_group)
    market_df = build_cross_market_rank(state)
    top_diag = current_top_diagnosis(state, current_quad_name, news_items)
    next_diag = next_phase_activation(state, next_quad_name, news_items)

    if mode == "current":
        confirm_text = first_nonempty(top_diag["confirm"])
        break_text = first_nonempty(top_diag["traps"])
        lead_text = QUAD_THEMES[current_quad_name]["current_winners"]
        migration = proxy_migration_text(current_quad_name)
    else:
        confirm_text = first_nonempty(next_diag["confirmation"])
        break_text = first_nonempty(next_diag["traps"])
        lead_text = QUAD_THEMES[next_quad_name]["first_next"]
        migration = proxy_migration_text(next_quad_name)

    rows = [
        ("Leadership core", lead_text),
        ("Strongest proxies", join_top(proxy_df, proxy_df.columns[0], proxy_df.columns[1], 4)),
        ("Commodity sleeve", join_top(chain_df, "Proxy", "Rank", 3)),
        ("FX sleeve", join_top(fx_df, "Pair", "Strength", 3)),
        ("Crypto sleeve", join_top(crypto_df, "Bucket", "Rank", 3)),
        ("Cross-asset leaders", join_top(market_df, "Aset", None, 3)),
        ("EM / IHSG read", phase_em_focus_text(state, quad_name)),
        ("Correlation focus", correlation_focus_text(quad_name)),
        ("Proxy chain", f"{chain_group}: {join_top(chain_df, 'Proxy', 'Rank', 4)}"),
        ("Divergence to watch", phase_divergence_focus(quad_name)),
        ("Migration path", migration),
        ("What confirms", confirm_text),
        ("What breaks", break_text),
    ]
    return pd.DataFrame(rows, columns=["Field", "Read"])


def build_transition_vertical_table(state: MacroState, news_items: List[Dict[str, str]]) -> pd.DataFrame:
    trans = transition_quality_score(state, news_items)
    rows = []
    for signal, status, why in trans["points"]:
        rows.append((signal, f"{status.upper()} — {why}"))
    return pd.DataFrame(rows, columns=["Signal", "Read"])


def build_curve_table_v2(state: MacroState, news_items: List[Dict[str, str]]) -> pd.DataFrame:
    premium, premium_read = news_premium_status(state, news_items)
    rows = [
        ("2Y", f"{state.y2_now:.2f}%", "Starting point paling cepat untuk hawkish/relief shift."),
        ("10Y", f"{state.y10_now:.2f}%", "Benchmark macro umum."),
        ("30Y", f"{state.y30_now:.2f}%", "Long-end / term premium pressure."),
        ("2s10s", f"{state.y10_now - state.y2_now:.2f}", "Curve shape: stress vs re-steepening."),
        ("10s30s", f"{state.y30_now - state.y10_now:.2f}", "Long-end conviction."),
        ("10Y real yield", f"{state.ry10_now:.2f}%", "Nasdaq / BTC / gold sensitivity utama."),
        ("DXY delta", f"{state.dxy_now - state.dxy_prev:.2f}", "Broad EM / crypto / FX pressure atau relief."),
        ("Oil delta", f"{state.oil_now - state.oil_prev:.2f}", "Inflation impulse atau shock growth filter."),
        ("Gold delta", f"{state.gold_now - state.gold_prev:.2f}" if pd.notna(state.gold_now) and pd.notna(state.gold_prev) else "n/a", "Hedge / fear bid vs inflation hedge filter."),
        ("Liquidity delta", f"{state.liq_now - state.liq_prev:.2f}", "Breadth dan easing/tightening helper."),
        ("Credit delta", f"{state.credit_now - state.credit_prev:.2f}", "Healthy transition vs fake bounce filter."),
        ("Breadth delta", f"{state.breadth_now - state.breadth_prev:.2f}", "Narrow vs broad confirmation."),
        ("News premium", premium, premium_read),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value", "Read"])


def _seq_state(ok: bool, mixed: bool = False) -> str:
    if ok:
        return "LIVE NOW"
    if mixed:
        return "WATCH / MIXED"
    return "NOT YET"


def build_top_sequence_table(state: MacroState, quad: str, news_items: List[Dict[str, str]]) -> pd.DataFrame:
    dy2 = delta(state.y2_now, state.y2_prev)
    dry = delta(state.ry10_now, state.ry10_prev)
    ddxy = delta(state.dxy_now, state.dxy_prev)
    doil = safe_pct_delta(state.oil_now, state.oil_prev)
    dgold = safe_pct_delta(state.gold_now, state.gold_prev)
    dcredit = delta(state.credit_now, state.credit_prev)
    dbreadth = delta(state.breadth_now, state.breadth_prev)
    premium, _ = news_premium_status(state, news_items)
    rows: List[Dict[str, str]] = []

    if quad == "Quad 1":
        rows = [
            {
                "Step": "1",
                "Sequence": "Breadth dulu menyempit walau leader masih kelihatan kuat.",
                "Typical order": "alts / small caps / broad EM / weak beta mulai gagal duluan sebelum Nasdaq leader",
                "Now": _seq_state(dbreadth < 0, abs(dbreadth) < 1.0),
                "Why it matters": "Top risk jarang mulai dari leader langsung ambruk. Biasanya market lebih sempit dulu.",
            },
            {
                "Step": "2",
                "Sequence": "Followers gagal confirm quality leader.",
                "Typical order": "BTC tetap kuat tapi alts tidak; Nasdaq tetap hijau tapi breadth jelek; EM tidak ikut",
                "Now": _seq_state((ddxy > 0) or (dcredit > 0), (ddxy > -0.1) or (dcredit > -0.05)),
                "Why it matters": "Kalau cuma quality yang hidup, regime belum broad lagi dan extension makin rapuh.",
            },
            {
                "Step": "3",
                "Sequence": "Driver makro mulai melawan: 2Y naik, real yields naik.",
                "Typical order": "front-end stress -> duration valuation kepukul -> quality jadi capek",
                "Now": _seq_state((dy2 > 0) or (dry > 0), (dy2 > -0.02) or (dry > -0.02)),
                "Why it matters": "Ini biasanya titik awal yang benar-benar menggerus napas Nasdaq / BTC.",
            },
            {
                "Step": "4",
                "Sequence": "Good news makin tidak efektif, dollar/credit ikut bikin tape sempit.",
                "Typical order": "headline masih bagus, tapi respon harga makin kecil dan market makin pilih-pilih",
                "Now": _seq_state((ddxy > 0 and dcredit > 0) or premium == "Mixed news premium", ddxy > 0 or dcredit > 0),
                "Why it matters": "Saat respon harga mengecil, fase lama biasanya sudah masuk late-stage.",
            },
            {
                "Step": "5",
                "Sequence": "Leader exhaustion / top confirm.",
                "Typical order": "baru di tahap ini Nasdaq leader ikut melemah atau minimal gagal extend",
                "Now": _seq_state(asset_risk(state, 'Nasdaq') >= 0.72, asset_risk(state, 'Nasdaq') >= 0.52),
                "Why it matters": "Top confirm biasanya datang paling akhir, bukan paling awal.",
            },
        ]
    elif quad == "Quad 2":
        rows = [
            {
                "Step": "1",
                "Sequence": "Oil / reflation winners masih naik, tapi impulse mulai pendek.",
                "Typical order": "direct commodity tetap kuat, tapi dorongannya tidak seganas fase awal",
                "Now": _seq_state(0 <= doil < 4, doil < 6),
                "Why it matters": "Reflation top sering mulai dari tenaga impulse yang memendek, bukan dari crash mendadak.",
            },
            {
                "Step": "2",
                "Sequence": "Second-order cyclical / industrial followers berhenti confirm.",
                "Typical order": "metals/miners/selected industrials masih oke, tapi broad cyclicals tidak ikut melebar",
                "Now": _seq_state(dbreadth < 0, abs(dbreadth) < 1.0),
                "Why it matters": "Kalau chain tidak meluas, trade reflation berubah jadi sempit dan rentan telat.",
            },
            {
                "Step": "3",
                "Sequence": "Dollar / credit mulai bikin reflation tidak sehat.",
                "Typical order": "USD naik -> credit seret -> reflation berubah dari constructive ke pressure trade",
                "Now": _seq_state((ddxy > 0) or (dcredit > 0), (ddxy > -0.1) or (dcredit > -0.05)),
                "Why it matters": "Dollar/credit menentukan apakah cyclical move masih broad atau tinggal selective saja.",
            },
            {
                "Step": "4",
                "Sequence": "Gold/hedge bid mulai pisah dari oil.",
                "Typical order": "gold naik sendiri sementara oil stall / tidak meledak lagi",
                "Now": _seq_state((dgold > 1.5 and doil <= 2) or premium == "Growth scare hedge bid", dgold > 0),
                "Why it matters": "Ini tanda market pindah dari reflation ke hedge/fear, bukan menambah reflation lagi.",
            },
            {
                "Step": "5",
                "Sequence": "Reflation top confirm.",
                "Typical order": "old winners tetap terlihat hidup, tapi risk/reward sudah jelek dan chain rusak",
                "Now": _seq_state(asset_risk(state, 'Energy') >= 0.72, asset_risk(state, 'Energy') >= 0.52),
                "Why it matters": "Konfirmasi akhir datang saat extension sudah tidak efisien lagi.",
            },
        ]
    elif quad == "Quad 3":
        rows = [
            {
                "Step": "1",
                "Sequence": "Growth stabilization mulai muncul di bawah permukaan.",
                "Typical order": "breadth membaik, growth quality lebih tahan, defensive tidak lagi satu-satunya refuge",
                "Now": _seq_state(dbreadth > 0, abs(dbreadth) < 1.0),
                "Why it matters": "Defensive top biasanya mulai saat market berhenti cuma cari aman.",
            },
            {
                "Step": "2",
                "Sequence": "2Y / dollar / real yields mulai relief.",
                "Typical order": "front-end dulu -> real yields -> dollar -> risk assets dapat napas",
                "Now": _seq_state((dy2 <= 0 and dry <= 0) or ddxy < 0, dy2 <= 0 or dry <= 0 or ddxy < 0),
                "Why it matters": "Tanpa relief rates/dollar, defensive bisa tetap dominan lebih lama.",
            },
            {
                "Step": "3",
                "Sequence": "Defensive crowding terlihat: gold/bonds kuat tapi tidak sebersih sebelumnya.",
                "Typical order": "defensives masih green, tapi sponsor baru justru pindah ke quality growth",
                "Now": _seq_state((dgold <= 0 and dry <= 0) or asset_risk(state, 'Nasdaq') > 0.35, dgold <= 0),
                "Why it matters": "Crowding tidak selalu berarti harga langsung turun; sering berarti alpha pindah diam-diam.",
            },
            {
                "Step": "4",
                "Sequence": "Quality growth / BTC mulai pegang bid lebih dulu.",
                "Typical order": "Nasdaq/quality dulu -> BTC -> baru broad beta kalau chain lengkap",
                "Now": _seq_state(transition_quality_score(state, news_items)["score"] >= 0.54, transition_quality_score(state, news_items)["score"] >= 0.40),
                "Why it matters": "Itu urutan normal saat defensive phase mulai mau top.",
            },
            {
                "Step": "5",
                "Sequence": "Defensive top confirm.",
                "Typical order": "market berhenti bayar premium berlebih untuk safety",
                "Now": _seq_state(asset_risk(state, 'Nasdaq') >= 0.55 and dbreadth > 0, asset_risk(state, 'Nasdaq') >= 0.36),
                "Why it matters": "Saat next winners tahan pullback, defensive top lebih valid.",
            },
        ]
    else:  # Quad 4
        rows = [
            {
                "Step": "1",
                "Sequence": "Oil masih tinggi, tapi impulse makin capek.",
                "Typical order": "harga tetap tinggi dulu, tapi spike makin pendek dan susah bikin ekstensi",
                "Now": _seq_state(0 <= doil < 4, doil < 6),
                "Why it matters": "Hard-asset top hampir selalu mulai dari hilangnya tenaga, bukan harga langsung rontok.",
            },
            {
                "Step": "2",
                "Sequence": "Second-order chain berhenti confirm.",
                "Typical order": "services / tankers / broad EM exporter tidak sekuat direct oil lagi",
                "Now": _seq_state(dbreadth < 0 or premium == "War premium fading", abs(dbreadth) < 1.0 or premium == "Mixed news premium"),
                "Why it matters": "Kalau chain putus, winners lama jadi makin crowded dan rawan telat.",
            },
            {
                "Step": "3",
                "Sequence": "Gold naik saat oil stall / turun; hedge bid muncul.",
                "Typical order": "market pindah dari inflation beta ke fear/hedge beta",
                "Now": _seq_state((dgold > 1.5 and doil <= 0) or premium == "Growth scare hedge bid", dgold > 0),
                "Why it matters": "Ini salah satu split paling penting untuk baca top hard-asset phase.",
            },
            {
                "Step": "4",
                "Sequence": "Credit memburuk, breadth tetap sempit, old winners jadi crowded.",
                "Typical order": "broad tape tetap jelek meski energy names belum jatuh",
                "Now": _seq_state((dcredit > 0 and dbreadth < 0), dcredit > 0 or dbreadth < 0),
                "Why it matters": "Fase lama bisa terlihat kuat di permukaan tapi secara internal sudah rapuh.",
            },
            {
                "Step": "5",
                "Sequence": "Hard-asset top confirm.",
                "Typical order": "baru belakangan direct winners ikut gagal extend atau kehilangan bid",
                "Now": _seq_state(asset_risk(state, 'Energy') >= 0.72, asset_risk(state, 'Energy') >= 0.52),
                "Why it matters": "Top confirm adalah step terakhir, bukan starting point.",
            },
        ]
    return pd.DataFrame(rows)


def build_bottom_sequence_table(state: MacroState, next_quad: str, news_items: List[Dict[str, str]]) -> pd.DataFrame:
    dy2 = delta(state.y2_now, state.y2_prev)
    dry = delta(state.ry10_now, state.ry10_prev)
    ddxy = delta(state.dxy_now, state.dxy_prev)
    doil = safe_pct_delta(state.oil_now, state.oil_prev)
    dgold = safe_pct_delta(state.gold_now, state.gold_prev)
    dcredit = delta(state.credit_now, state.credit_prev)
    dbreadth = delta(state.breadth_now, state.breadth_prev)
    trans = transition_quality_score(state, news_items)
    rows: List[Dict[str, str]] = []

    if next_quad == "Quad 1":
        rows = [
            {
                "Step": "1",
                "Sequence": "Tekanan utama berhenti memburuk: 2Y dulu relief.",
                "Typical order": "2Y stop naik -> front-end stress reda -> valuation napas",
                "Now": _seq_state(dy2 <= 0, dy2 <= 0.05),
                "Why it matters": "Ini starting point tercepat untuk bottom duration-friendly assets.",
            },
            {
                "Step": "2",
                "Sequence": "Real yields berhenti menekan, quality leader tahan banting.",
                "Typical order": "Nasdaq / quality growth / BTC bertahan dulu sebelum broad beta",
                "Now": _seq_state(dry <= 0, dry <= 0.05),
                "Why it matters": "Bottom sehat biasanya dimulai dari quality leader, bukan junk duluan.",
            },
            {
                "Step": "3",
                "Sequence": "Dollar dan credit ikut relief.",
                "Typical order": "DXY reda -> credit stabil -> EM / crypto dapat ruang ikut",
                "Now": _seq_state(ddxy <= 0 and dcredit <= 0, ddxy <= 0 or dcredit <= 0),
                "Why it matters": "Tanpa ini, pantulan sering hanya quality-led dan gampang gagal.",
            },
            {
                "Step": "4",
                "Sequence": "Breadth memperbaiki cerita.",
                "Typical order": "follower ikut, no new lows on bad news, move mulai melebar",
                "Now": _seq_state(dbreadth >= 0, abs(dbreadth) < 1.0),
                "Why it matters": "Breadth membedakan bottom asli vs sekadar dead-cat relief.",
            },
            {
                "Step": "5",
                "Sequence": "Followers confirm.",
                "Typical order": "ETH/alts quality, broad EM, selected cyclicals baru ikut setelah leader",
                "Now": _seq_state(trans["score"] >= 0.68, trans["score"] >= 0.54),
                "Why it matters": "Kalau hanya leader yang pulih, fase baru belum matang.",
            },
            {
                "Step": "6",
                "Sequence": "Bottom / activation confirmed.",
                "Typical order": "pullback ditahan, bad news makin tidak efektif, old winners capek",
                "Now": _seq_state(next_phase_activation(state, next_quad, news_items)["score"] >= 0.72, next_phase_activation(state, next_quad, news_items)["score"] >= 0.55),
                "Why it matters": "Konfirmasi akhir datang saat market tahan ujian kedua, bukan saat pantulan pertama.",
            },
        ]
    elif next_quad == "Quad 2":
        rows = [
            {
                "Step": "1",
                "Sequence": "Growth berhenti memburuk dulu.",
                "Typical order": "data/activity stabilization dulu sebelum cyclical rally melebar",
                "Now": _seq_state(state.growth_now > state.growth_prev, abs(state.growth_now - state.growth_prev) < 0.05),
                "Why it matters": "Quad 2 butuh fondasi growth, bukan cuma headline.",
            },
            {
                "Step": "2",
                "Sequence": "Early cyclicals / industrials stop making new lows.",
                "Typical order": "industrials, miners, selected cyclical winners tahan banting dulu",
                "Now": _seq_state(dbreadth >= 0, abs(dbreadth) < 1.0),
                "Why it matters": "Leaders reflation biasanya stabil dulu sebelum broad beta ikut.",
            },
            {
                "Step": "3",
                "Sequence": "Inflation / commodity impulse hidup lagi.",
                "Typical order": "metals/oil/commodity-linked move duluan sebelum broad cyclicals",
                "Now": _seq_state((state.inflation_now > state.inflation_prev) or (doil > 2), abs(state.inflation_now - state.inflation_prev) < 0.05 or doil > 0),
                "Why it matters": "Tanpa impulse ini, move reflation gampang hanya jadi bounce pendek.",
            },
            {
                "Step": "4",
                "Sequence": "Dollar tidak terlalu mengganggu, credit tetap oke.",
                "Typical order": "reflation sehat butuh dollar tidak terlalu galak dan credit tidak patah",
                "Now": _seq_state(ddxy <= 1.0 and dcredit <= 0, ddxy <= 1.0 or dcredit <= 0),
                "Why it matters": "Kalau USD/credit melawan, Quad 2 sering jadi selective shock saja.",
            },
            {
                "Step": "5",
                "Sequence": "Followers / second-order chain confirm.",
                "Typical order": "metals/miners -> industrial followers -> selected small caps",
                "Now": _seq_state(trans["score"] >= 0.54 and dbreadth >= 0, trans["score"] >= 0.40),
                "Why it matters": "Chain yang lengkap jauh lebih penting daripada satu headline reflation.",
            },
            {
                "Step": "6",
                "Sequence": "Reflation activation confirmed.",
                "Typical order": "pullback shallow, growth and inflation both support, breadth broad",
                "Now": _seq_state(next_phase_activation(state, next_quad, news_items)["score"] >= 0.72, next_phase_activation(state, next_quad, news_items)["score"] >= 0.55),
                "Why it matters": "Konfirmasi akhir datang saat broad cyclical beta bertahan, bukan hanya oil naik.",
            },
        ]
    elif next_quad == "Quad 3":
        rows = [
            {
                "Step": "1",
                "Sequence": "Oil / inflation shock mulai reda dulu.",
                "Typical order": "oil turun / stall -> inflation pressure mereda -> defensives dapat sponsor",
                "Now": _seq_state(doil <= 0, doil <= 2),
                "Why it matters": "Ini starting point utama untuk defensive-disinflation bottoming.",
            },
            {
                "Step": "2",
                "Sequence": "Bonds / gold / defensives stop making fresh lows dan mulai dicari.",
                "Typical order": "UST / defensives / gold dulu, broad beta tetap belum bersih",
                "Now": _seq_state((state.inflation_now <= state.inflation_prev) or (dgold > 0), abs(state.inflation_now - state.inflation_prev) < 0.05 or dgold > 0),
                "Why it matters": "Bottom Quad 3 biasanya dimulai dari demand for safety yang sehat, bukan risk-on.",
            },
            {
                "Step": "3",
                "Sequence": "2Y relief supaya pressure tidak bertambah.",
                "Typical order": "front-end berhenti menekan, crash risk menurun",
                "Now": _seq_state(dy2 <= 0, dy2 <= 0.05),
                "Why it matters": "Kalau 2Y terus keras, defensive strength bisa tercampur panic, bukan clean disinflation.",
            },
            {
                "Step": "4",
                "Sequence": "Credit tidak tambah rusak.",
                "Typical order": "downside beta masih rapuh, tapi stress sistemik berhenti melebar",
                "Now": _seq_state(dcredit <= 0, dcredit <= 0.05),
                "Why it matters": "Membedakan good disinflation dengan bad crash dynamic.",
            },
            {
                "Step": "5",
                "Sequence": "Defensive breadth broadens.",
                "Typical order": "bonds/defensives/gold bukan cuma satu nama, tapi satu sleeve ikut",
                "Now": _seq_state(dbreadth >= 0 or dgold > 0, abs(dbreadth) < 1.0),
                "Why it matters": "Tanpa breadth, bisa jadi cuma panic bid sempit.",
            },
            {
                "Step": "6",
                "Sequence": "Defensive/disinflation phase confirmed.",
                "Typical order": "headline negatif tidak bikin panic baru, safety bid tetap bertahan",
                "Now": _seq_state(next_phase_activation(state, next_quad, news_items)["score"] >= 0.72, next_phase_activation(state, next_quad, news_items)["score"] >= 0.55),
                "Why it matters": "Konfirmasi akhir datang saat safety sleeve tahan retest.",
            },
        ]
    else:  # Quad 4
        rows = [
            {
                "Step": "1",
                "Sequence": "Oil / hard assets stop falling dan mulai firm lagi.",
                "Typical order": "direct commodity duluan, broad beta belum perlu ikut",
                "Now": _seq_state(doil >= 2, doil >= 0),
                "Why it matters": "Stagflation bottoming dimulai dari hard asset pulse, bukan broad equity relief.",
            },
            {
                "Step": "2",
                "Sequence": "Inflation / breakeven harden meski growth tidak sehat.",
                "Typical order": "inflation side dulu pulih, growth side tetap rapuh",
                "Now": _seq_state(state.inflation_now > state.inflation_prev, abs(state.inflation_now - state.inflation_prev) < 0.05),
                "Why it matters": "Ini yang membedakan Quad 4 dari Quad 2.",
            },
            {
                "Step": "3",
                "Sequence": "Dollar tetap firm, breadth tetap sempit.",
                "Typical order": "selective winners hidup, broad tape belum ikut",
                "Now": _seq_state(ddxy >= 0 and dbreadth < 0, ddxy >= -0.1 or dbreadth < 1.0),
                "Why it matters": "Stagflation yang bersih memang selective, bukan broad risk-on.",
            },
            {
                "Step": "4",
                "Sequence": "Pure energy / exporter proxies memimpin duluan.",
                "Typical order": "WTI/upstream -> majors -> selected exporters -> followers belakangan",
                "Now": _seq_state(asset_opportunity(state, 'Energy') >= 0.55, asset_opportunity(state, 'Energy') >= 0.40),
                "Why it matters": "Urutan leader penting; kalau broad beta ikut terlalu cepat justru patut curiga.",
            },
            {
                "Step": "5",
                "Sequence": "Second-order commodity chain ikut confirm.",
                "Typical order": "services/tanker/exporter sleeves baru ikut kalau pulse bertahan",
                "Now": _seq_state(next_phase_activation(state, next_quad, news_items)["score"] >= 0.55, next_phase_activation(state, next_quad, news_items)["score"] >= 0.38),
                "Why it matters": "Tanpa second-order chain, move bisa cuma spike headline.",
            },
            {
                "Step": "6",
                "Sequence": "Stagflation / hard-asset pressure confirmed.",
                "Typical order": "current squeeze bertahan melewati retest, broad beta tetap kalah",
                "Now": _seq_state(next_phase_activation(state, next_quad, news_items)["score"] >= 0.72, next_phase_activation(state, next_quad, news_items)["score"] >= 0.55),
                "Why it matters": "Konfirmasi akhir datang saat selective winners konsisten, bukan saat headline pertama keluar.",
            },
        ]
    return pd.DataFrame(rows)


def build_compare_board_table(state: MacroState, current_quad_name: str, next_quad_name: str, news_items: List[Dict[str, str]], mode: str) -> pd.DataFrame:
    treasury_label, treasury_read = treasury_regime_text(state)
    premium_label, premium_read = news_premium_status(state, news_items)
    trans = transition_quality_score(state, news_items)
    top_seq = build_top_sequence_table(state, current_quad_name, news_items)
    bot_seq = build_bottom_sequence_table(state, next_quad_name, news_items)
    if mode == "current":
        maturity, _, _, _ = current_phase_maturity(state, current_quad_name, news_items)
        top_diag = current_top_diagnosis(state, current_quad_name, news_items)
        rows = [
            ("Quad", f"{current_quad_name} — {QUAD_THEMES[current_quad_name]['theme']}"),
            ("Lifecycle", maturity),
            ("Treasury", f"{treasury_label}: {treasury_read}"),
            ("Current winners", QUAD_THEMES[current_quad_name]["current_winners"]),
            ("Top status", f"{top_diag['label']} ({top_diag['score']:.0%})"),
            ("Step 1", top_seq.iloc[0]["Sequence"]),
            ("Step 2", top_seq.iloc[1]["Sequence"]),
            ("Step 3", top_seq.iloc[2]["Sequence"]),
            ("Step 4", top_seq.iloc[3]["Sequence"]),
            ("Final confirm", top_seq.iloc[4]["Sequence"]),
            ("News premium", f"{premium_label}: {premium_read}"),
            ("Transition quality", trans["label"]),
        ]
    else:
        next_diag = next_phase_activation(state, next_quad_name, news_items)
        rows = [
            ("Next likely quad", f"{next_quad_name} — {QUAD_THEMES[next_quad_name]['theme']}"),
            ("Activation", f"{next_diag['label']} ({next_diag['score']:.0%})"),
            ("Treasury", f"{treasury_label}: {treasury_read}"),
            ("First likely winners", QUAD_THEMES[next_quad_name]["first_next"]),
            ("Step 1", bot_seq.iloc[0]["Sequence"]),
            ("Step 2", bot_seq.iloc[1]["Sequence"]),
            ("Step 3", bot_seq.iloc[2]["Sequence"]),
            ("Step 4", bot_seq.iloc[3]["Sequence"]),
            ("Step 5", bot_seq.iloc[4]["Sequence"]),
            ("Final confirm", bot_seq.iloc[5]["Sequence"]),
            ("News premium", f"{premium_label}: {premium_read}"),
            ("Transition quality", trans["label"]),
        ]
    return pd.DataFrame(rows, columns=["Field", "Read"])



@st.cache_data(ttl=1800, show_spinner=False)
def yahoo_chart(symbol: str, range_: str = "18mo", interval: str = "1d") -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(symbol, safe='')}"
    r = requests.get(
        url,
        params={"range": range_, "interval": interval, "includeAdjustedClose": "true"},
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    r.raise_for_status()
    payload = r.json()
    results = payload.get("chart", {}).get("result") or []
    if not results:
        return pd.DataFrame(columns=["date", "close"])
    item = results[0]
    ts = item.get("timestamp") or []
    if not ts:
        return pd.DataFrame(columns=["date", "close"])
    indicators = item.get("indicators", {})
    adj = (indicators.get("adjclose") or [{}])[0].get("adjclose")
    close = (indicators.get("quote") or [{}])[0].get("close") or []
    values = adj if adj and any(v is not None for v in adj) else close
    df = pd.DataFrame({"date": pd.to_datetime(ts, unit="s"), "close": pd.to_numeric(values, errors="coerce")})
    df["date"] = df["date"].dt.tz_localize(None).dt.normalize()
    return df.dropna().drop_duplicates("date").sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_market_bundle() -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for name, symbol in YAHOO_SYMBOLS.items():
        try:
            out[name] = yahoo_chart(symbol)
        except Exception:
            out[name] = pd.DataFrame(columns=["date", "close"])
    return out


def build_market_price_frame(bundle: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    wide: pd.DataFrame | None = None
    for name, df in bundle.items():
        if df.empty:
            continue
        cur = df[["date", "close"]].rename(columns={"close": name})
        wide = cur if wide is None else wide.merge(cur, on="date", how="outer")
    if wide is None:
        return pd.DataFrame()
    wide = wide.sort_values("date").ffill().dropna(how="all")
    return wide.reset_index(drop=True)


def safe_trailing_return(prices: pd.DataFrame, col: str, days: int = 20) -> float:
    if prices.empty or col not in prices.columns:
        return float("nan")
    s = prices[col].dropna()
    if len(s) <= days:
        return float("nan")
    prev = s.iloc[-days - 1]
    now = s.iloc[-1]
    if prev == 0 or pd.isna(prev) or pd.isna(now):
        return float("nan")
    return float((now / prev - 1) * 100)


def safe_corr(prices: pd.DataFrame, a: str, b: str, window: int = 63) -> float:
    if prices.empty or a not in prices.columns or b not in prices.columns:
        return float("nan")
    ret = prices[[a, b]].pct_change().dropna()
    if len(ret) < max(20, window // 2):
        return float("nan")
    take = ret.iloc[-min(window, len(ret)):]
    return float(take[a].corr(take[b]))


def safe_beta(prices: pd.DataFrame, a: str, b: str, window: int = 63) -> float:
    if prices.empty or a not in prices.columns or b not in prices.columns:
        return float("nan")
    ret = prices[[a, b]].pct_change().dropna()
    if len(ret) < max(20, window // 2):
        return float("nan")
    take = ret.iloc[-min(window, len(ret)):]
    var_b = float(take[b].var())
    if var_b == 0 or pd.isna(var_b):
        return float("nan")
    return float(take[a].cov(take[b]) / var_b)


def score_label(score: float) -> str:
    if pd.isna(score):
        return "n/a"
    if score >= 0.75:
        return "Strong"
    if score >= 0.55:
        return "Good"
    if score >= 0.40:
        return "Mixed"
    return "Weak"


def corr_label(x: float, inverse: bool = False) -> str:
    if pd.isna(x):
        return "n/a"
    value = -x if inverse else x
    if value >= 0.65:
        return "Very high"
    if value >= 0.40:
        return "High"
    if value >= 0.20:
        return "Medium"
    if value >= -0.20:
        return "Low / unstable"
    return "Inverse / opposite"


def bool_score(*flags: bool) -> float:
    vals = [1.0 if f else 0.0 for f in flags]
    return float(np.mean(vals)) if vals else 0.0


def build_ihsg_overlay(state: MacroState, prices: pd.DataFrame, current_quad_name: str, next_quad_name: str) -> Dict[str, object]:
    dy2 = delta(state.y2_now, state.y2_prev)
    dry = delta(state.ry10_now, state.ry10_prev)
    ddxy = delta(state.dxy_now, state.dxy_prev)
    dcredit = delta(state.credit_now, state.credit_prev)
    doil = safe_pct_delta(state.oil_now, state.oil_prev)
    dgold = safe_pct_delta(state.gold_now, state.gold_prev)

    ihsg20 = safe_trailing_return(prices, "IHSG", 20)
    eem20 = safe_trailing_return(prices, "EEM", 20)
    eido20 = safe_trailing_return(prices, "EIDO", 20)
    qqq20 = safe_trailing_return(prices, "QQQ", 20)
    xle20 = safe_trailing_return(prices, "XLE", 20)
    uso20 = safe_trailing_return(prices, "USO", 20)
    pick20 = safe_trailing_return(prices, "PICK", 20)
    gld20 = safe_trailing_return(prices, "GLD", 20)
    usd20 = safe_trailing_return(prices, "USDIDR", 20)
    btc20 = safe_trailing_return(prices, "BTC", 20)

    corr_eem = safe_corr(prices, "IHSG", "EEM", 63)
    corr_eido = safe_corr(prices, "IHSG", "EIDO", 63)
    corr_qqq = safe_corr(prices, "IHSG", "QQQ", 63)
    corr_xle = safe_corr(prices, "IHSG", "XLE", 63)
    corr_uso = safe_corr(prices, "IHSG", "USO", 63)
    corr_pick = safe_corr(prices, "IHSG", "PICK", 63)
    corr_gld = safe_corr(prices, "IHSG", "GLD", 63)
    corr_btc = safe_corr(prices, "IHSG", "BTC", 63)
    corr_usdidr = safe_corr(prices, "IHSG", "USDIDR", 63)

    global_flow = bool_score(dy2 <= 0, dry <= 0, ddxy <= 0, dcredit <= 0)
    em_relief = bool_score((not pd.isna(usd20) and usd20 <= 0), (not pd.isna(eem20) and eem20 >= 0), (not pd.isna(corr_eem) and corr_eem >= 0.30))
    commodity_selective = bool_score((not pd.isna(xle20) and xle20 > 0), (not pd.isna(pick20) and pick20 > 0), (not pd.isna(corr_xle) and corr_xle >= 0.20) or (not pd.isna(corr_pick) and corr_pick >= 0.20))
    oil_tax = bool_score(doil > 2, ddxy > 0, (not pd.isna(usd20) and usd20 > 0))
    broad_risk = bool_score((not pd.isna(qqq20) and qqq20 > 0), (not pd.isna(btc20) and btc20 > 0), state.breadth_now >= state.breadth_prev)
    defensive_relief = bool_score(doil <= 0, dgold > 0, dy2 <= 0, dcredit <= 0)
    domestic_decoupling = bool_score((not pd.isna(ihsg20) and ihsg20 > 0), (not pd.isna(eido20) and not pd.isna(ihsg20) and ihsg20 > eido20), (not pd.isna(usd20) and usd20 <= 2.0))
    em_stress = bool_score(ddxy > 0, dy2 > 0, dcredit > 0, (not pd.isna(usd20) and usd20 > 0))

    scenario_scores = {
        "Broad EM / rate-relief confirm": float(np.mean([global_flow, em_relief, broad_risk])),
        "Selective commodity confirm": float(np.mean([commodity_selective, 1 - min(oil_tax, 1.0), max(0.0, 1 - em_relief)])),
        "Oil shock hurts broad IHSG": float(np.mean([oil_tax, commodity_selective, max(0.0, 1 - em_relief)])),
        "EM-stress override": float(np.mean([em_stress, max(0.0, 1 - global_flow), max(0.0, 1 - em_relief)])),
        "Defensive / large-cap relief": float(np.mean([defensive_relief, global_flow, max(0.0, 1 - oil_tax)])),
        "Domestic decoupling / local hold": float(np.mean([domestic_decoupling, max(0.0, 1 - em_stress), max(0.0, 1 - oil_tax / 1.0)])),
    }
    dominant = max(scenario_scores, key=scenario_scores.get)

    current_fit = {
        "Quad 1": float(np.mean([global_flow, em_relief, broad_risk, max(0.0, 1 - oil_tax)])),
        "Quad 2": float(np.mean([commodity_selective, max(global_flow, 0.35), max(0.0, 1 - em_stress)])),
        "Quad 3": float(np.mean([defensive_relief, max(0.0, 1 - oil_tax), max(0.0, 1 - em_stress / 1.0)])),
        "Quad 4": float(np.mean([commodity_selective, oil_tax, max(0.0, 1 - em_relief)])),
    }

    lead_order = [
        "1) DXY + USDIDR dulu: ini filter tercepat apakah broad IHSG bisa ikut atau masih kena tekanan EM.",
        "2) 2Y + real yields: kalau front-end/real yields relief, broad IHSG dan banks lebih punya ruang. Kalau naik, broad tape lebih rapuh.",
        "3) EEM + EIDO: cek apakah broad EM dan Indonesia versi USD ikut confirm atau hanya local tape yang kelihatan lebih baik.",
        "4) Oil + metals/mining: kalau ini yang hidup tapi DXY/EM jelek, baca sebagai selective commodity, bukan broad IHSG bullish.",
        "5) Baru broad IHSG: index sering telat atau tersaring oleh rupiah, flow asing, dan komposisi banks/commodities.",
    ]

    if dominant == "Broad EM / rate-relief confirm":
        headline = "IHSG broad lebih mungkin confirm quad kalau relief rates/dollar/credit bersih dan broad EM ikut."
    elif dominant == "Selective commodity confirm":
        headline = "IHSG lebih cocok dibaca selective commodity confirm, bukan broad confirm. Resource pockets bisa hidup lebih dulu daripada whole index."
    elif dominant == "Oil shock hurts broad IHSG":
        headline = "US oil-led strength tidak otomatis ditransmisikan ke IHSG broad. Indonesia dapat benefit ke sebagian resource names, tapi broad index tetap kena oil-import + EM filter."
    elif dominant == "EM-stress override":
        headline = "Filter EM sedang mengalahkan cerita quad global. Broad IHSG lebih tunduk ke DXY/USDIDR/foreign-flow daripada ke thematic US move."
    elif dominant == "Defensive / large-cap relief":
        headline = "Kalau oil/inflation reda dan front-end relief, IHSG broad terutama large caps/banks bisa lebih enak walau bukan pure high-beta risk-on."
    else:
        headline = "Tape domestik relatif tahan, tapi tetap harus dicek apakah itu betul-betul local decoupling atau cuma noise sebelum global filter menekan lagi."

    notes = [
        "US strength di oil/XLE belum tentu bikin IHSG broad ikut; biasanya baca dulu apakah DXY/USDIDR sedang menekan atau tidak.",
        "Kalau EEM/EIDO dan IHSG sama-sama membaik saat 2Y/real yields/DXY relief, broad confirm lebih bersih.",
        "Kalau oil/metals kuat tapi EEM lemah dan USDIDR naik, lebih cocok dibaca sebagai selective exporter/commodity move, bukan broad IHSG bullish.",
        "Kalau gold naik sementara oil turun dan credit tidak tambah rusak, baca sebagai relief/hedge transition; ini bisa lebih ramah untuk large caps daripada commodity beta murni.",
    ]

    return {
        "headline": headline,
        "dominant": dominant,
        "scenario_scores": scenario_scores,
        "current_fit": current_fit.get(current_quad_name, float("nan")),
        "next_fit": current_fit.get(next_quad_name, float("nan")),
        "fit_by_quad": current_fit,
        "global_flow": global_flow,
        "em_relief": em_relief,
        "commodity_selective": commodity_selective,
        "oil_tax": oil_tax,
        "broad_risk": broad_risk,
        "defensive_relief": defensive_relief,
        "domestic_decoupling": domestic_decoupling,
        "em_stress": em_stress,
        "lead_order": lead_order,
        "notes": notes,
        "snapshot": {
            "IHSG 20d": ihsg20,
            "EIDO 20d": eido20,
            "EEM 20d": eem20,
            "QQQ 20d": qqq20,
            "XLE 20d": xle20,
            "USO 20d": uso20,
            "PICK 20d": pick20,
            "GLD 20d": gld20,
            "BTC 20d": btc20,
            "USDIDR 20d": usd20,
            "Corr IHSG-EEM": corr_eem,
            "Corr IHSG-EIDO": corr_eido,
            "Corr IHSG-QQQ": corr_qqq,
            "Corr IHSG-XLE": corr_xle,
            "Corr IHSG-USO": corr_uso,
            "Corr IHSG-PICK": corr_pick,
            "Corr IHSG-GLD": corr_gld,
            "Corr IHSG-BTC": corr_btc,
            "Corr IHSG-USDIDR": corr_usdidr,
        },
    }


def build_ihsg_correlation_table(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty or "IHSG" not in prices.columns:
        return pd.DataFrame([{
            "Factor": "No market data",
            "63d corr": "n/a",
            "126d corr": "n/a",
            "Beta 63d": "n/a",
            "20d move": "n/a",
            "Read": "Yahoo market overlay belum tersedia sekarang."
        }])
    factors = [
        ("EEM", "Broad EM filter", False),
        ("EIDO", "Indonesia in USD / foreign-flow view", False),
        ("QQQ", "US duration / quality spillover", False),
        ("XLE", "US energy sleeve", False),
        ("USO", "Crude oil pulse", False),
        ("PICK", "Metals / mining exporter pulse", False),
        ("GLD", "Gold / hedge pulse", False),
        ("BTC", "Global risk appetite spillover", False),
        ("USDIDR", "Rupiah pressure filter", True),
    ]
    rows = []
    for name, desc, inverse in factors:
        c63 = safe_corr(prices, "IHSG", name, 63)
        c126 = safe_corr(prices, "IHSG", name, 126)
        beta63 = safe_beta(prices, "IHSG", name, 63)
        move20 = safe_trailing_return(prices, name, 20)
        read = corr_label(c63, inverse=inverse)
        if name == "USDIDR":
            interp = "Semakin inverse, biasanya semakin sehat untuk broad IHSG di phase relief. Kalau malah searah kuat, filter EM/rupiah sedang dominan."
        elif name in ["XLE", "USO", "PICK"]:
            interp = "Kalau ini tinggi tapi EEM/QQQ rendah, baca IHSG sebagai selective commodity market, bukan broad quad confirm."
        elif name in ["EEM", "EIDO"]:
            interp = "Ini paling penting untuk tahu apakah broad IHSG benar-benar ikut flow EM atau hanya local noise/selective move."
        else:
            interp = desc
        rows.append({
            "Factor": f"{name} — {desc}",
            "63d corr": f"{c63:.2f}" if not pd.isna(c63) else "n/a",
            "126d corr": f"{c126:.2f}" if not pd.isna(c126) else "n/a",
            "Beta 63d": f"{beta63:.2f}" if not pd.isna(beta63) else "n/a",
            "20d move": f"{move20:.1f}%" if not pd.isna(move20) else "n/a",
            "Read": f"{read}. {interp}",
        })
    return pd.DataFrame(rows)


def build_ihsg_filter_table(state: MacroState, overlay: Dict[str, object], current_quad_name: str, next_quad_name: str) -> pd.DataFrame:
    rows = [
        ("Current quad fit", f"{score_label(float(overlay['current_fit']))} ({float(overlay['current_fit']):.0%})", f"Seberapa broad/clean IHSG cocok dengan {current_quad_name} sekarang."),
        ("Next quad fit", f"{score_label(float(overlay['next_fit']))} ({float(overlay['next_fit']):.0%})", f"Seberapa mungkin IHSG ikut kalau {next_quad_name} yang aktif."),
        ("Global flow filter", f"{score_label(float(overlay['global_flow']))} ({float(overlay['global_flow']):.0%})", "2Y, real yields, DXY, credit. Ini yang paling sering menentukan broad IHSG bisa ikut atau tidak."),
        ("EM / rupiah relief", f"{score_label(float(overlay['em_relief']))} ({float(overlay['em_relief']):.0%})", "Cross-check broad EM, EIDO, dan tekanan USDIDR."),
        ("Commodity selective support", f"{score_label(float(overlay['commodity_selective']))} ({float(overlay['commodity_selective']):.0%})", "Kalau ini tinggi sendirian, baca sebagai selective commodity pockets."),
        ("Oil-import penalty", f"{score_label(float(overlay['oil_tax']))} ({float(overlay['oil_tax']):.0%})", "Crude oil terlalu tinggi + dollar keras sering buruk untuk broad IHSG walau resource names bisa hidup."),
        ("Broad risk spillover", f"{score_label(float(overlay['broad_risk']))} ({float(overlay['broad_risk']):.0%})", "QQQ/BTC/breadth dipakai hanya sebagai spillover check, bukan driver utama IHSG."),
        ("Defensive relief", f"{score_label(float(overlay['defensive_relief']))} ({float(overlay['defensive_relief']):.0%})", "Lebih ramah untuk large caps / banks / relief tape daripada commodity beta murni."),
        ("Domestic decoupling", f"{score_label(float(overlay['domestic_decoupling']))} ({float(overlay['domestic_decoupling']):.0%})", "Tape lokal tahan vs foreign-flow view, tapi harus dicek apakah berumur panjang atau tidak."),
        ("EM-stress override", f"{score_label(float(overlay['em_stress']))} ({float(overlay['em_stress']):.0%})", "Kalau ini tinggi, filter EM mengalahkan cerita quad global."),
        ("Dominant IHSG read", str(overlay['dominant']), str(overlay['headline'])),
    ]
    return pd.DataFrame(rows, columns=["Field", "Now", "Why it matters"])


def build_ihsg_scenario_table(state: MacroState, overlay: Dict[str, object]) -> pd.DataFrame:
    scen = overlay["scenario_scores"]
    rows = [
        ("Broad EM / rate-relief confirm", f"{scen['Broad EM / rate-relief confirm']:.0%}", "2Y/real yields/DXY/credit relief + EEM/EIDO ikut", "Broad IHSG, banks, large caps, domestic cyclicals lebih mungkin ikut bersih", "Kalau broad EM dan USDIDR tidak confirm, jangan terlalu cepat sebut broad confirm."),
        ("Selective commodity confirm", f"{scen['Selective commodity confirm']:.0%}", "Metals/mining/oil-related pulse ada, tapi broad EM tidak sebersih itu", "Selective resource names / commodity-linked pockets lebih menarik daripada whole index", "Jangan salah baca resource strength sebagai seluruh IHSG bullish."),
        ("Oil shock hurts broad IHSG", f"{scen['Oil shock hurts broad IHSG']:.0%}", "US oil/XLE kuat, tapi DXY/USDIDR juga menekan", "US energy bisa kuat sementara broad IHSG tetap mixed atau rapuh", "Indonesia dapat benefit ke sebagian exporters, tapi broad tape bisa kena oil-import tax + EM filter."),
        ("EM-stress override", f"{scen['EM-stress override']:.0%}", "DXY naik, 2Y naik, credit ketat, USDIDR lemah", "Broad IHSG lebih tunduk ke risk-off EM daripada ke cerita quad US", "Ini salah satu override paling penting. Kalau aktif, banyak narasi lain kalah."),
        ("Defensive / large-cap relief", f"{scen['Defensive / large-cap relief']:.0%}", "Oil/inflation reda, front-end relief, credit tidak tambah rusak", "Banks / large caps / defensive quality lokal bisa lebih enak daripada beta tinggi", "Ini bukan selalu risk-on full; sering hanya relief yang lebih bersih untuk quality domestik."),
        ("Domestic decoupling / local hold", f"{scen['Domestic decoupling / local hold']:.0%}", "IHSG lokal tahan, tapi EIDO/foreign-flow view tidak sekuat itu", "Ada ruang untuk local resilience, tapi wajib cek umur dan sponsor-nya", "Kalau DXY/USDIDR makin jelek, local hold bisa cepat patah."),
    ]
    return pd.DataFrame(rows, columns=["Scenario", "Live fit", "What it means", "Likely IHSG behavior", "Trap"])


def build_ihsg_order_table(overlay: Dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame({"Read order": overlay["lead_order"]})


def build_ihsg_snapshot_table(overlay: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for key, value in overlay["snapshot"].items():
        if pd.isna(value):
            shown = "n/a"
        elif "Corr" in key:
            shown = f"{value:.2f}"
        else:
            shown = f"{value:.1f}%"
        rows.append((key, shown))
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def build_ihsg_quad_fit_matrix(overlay: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for quad_name, score in overlay["fit_by_quad"].items():
        if quad_name == "Quad 1":
            why = "Paling butuh relief di 2Y/real yields/DXY/credit agar broad IHSG dan banks benar-benar ikut."
        elif quad_name == "Quad 2":
            why = "Bagus kalau growth+commodity hidup, tapi broadness tetap tergantung dollar/EM filter."
        elif quad_name == "Quad 3":
            why = "Lebih cocok untuk relief quality/large caps kalau oil reda dan defensive demand sehat."
        else:
            why = "Biasanya selective commodity/hard-asset only, broad IHSG belum tentu enak."
        rows.append((quad_name, f"{score_label(score)} ({score:.0%})", why))
    return pd.DataFrame(rows, columns=["Quad", "IHSG fit", "Read"])





def headline_text(news_items: List[Dict[str, str]]) -> str:
    return " ".join(f"{item.get('title','')} {item.get('desc','')}" for item in news_items).lower()


def headline_hit_count(news_items: List[Dict[str, str]], keywords: List[str]) -> int:
    text = headline_text(news_items)
    return sum(1 for k in keywords if k in text)


def short_ret(prices: pd.DataFrame, col: str, days: int = 5) -> float:
    return safe_trailing_return(prices, col, days)


def move_text(x: float, pct: bool = True) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x:.1f}%" if pct else f"{x:.2f}"


def pct_sign(x: float, up_good: bool = True) -> str:
    if pd.isna(x):
        return "n/a"
    good = x > 0 if up_good else x < 0
    if good:
        return "supports"
    if x == 0:
        return "flat"
    return "pressures"


def classify_energy_shock(state: MacroState, prices: pd.DataFrame, news_items: List[Dict[str, str]]) -> Dict[str, object]:
    oil_pct = safe_pct_delta(state.oil_now, state.oil_prev)
    xle20 = safe_trailing_return(prices, "XLE", 20)
    uso20 = safe_trailing_return(prices, "USO", 20)
    kol20 = safe_trailing_return(prices, "KOL", 20)
    tan20 = safe_trailing_return(prices, "TAN", 20)
    geo_hits = headline_hit_count(news_items, ["iran", "war", "hormuz", "missile", "middle east", "ceasefire"])
    premium_label, _ = news_premium_status(state, news_items)

    temporary_score = bool_score(geo_hits > 0, oil_pct > 3, (pd.isna(kol20) or kol20 <= 0))
    structural_score = bool_score((not pd.isna(xle20) and xle20 > 0) or (not pd.isna(uso20) and uso20 > 0), not pd.isna(kol20) and kol20 > 0, premium_label == "War premium fading" or geo_hits == 0)

    if not pd.isna(kol20) and not pd.isna(tan20) and kol20 > tan20 + 6:
        theme = "Coal / scarcity leads"
    elif not pd.isna(tan20) and not pd.isna(kol20) and tan20 > kol20 + 6:
        theme = "Solar / transition catch-up"
    elif oil_pct > 2 or (not pd.isna(uso20) and uso20 > 2):
        theme = "Oil / legacy energy leads"
    else:
        theme = "Mixed energy leadership"

    if structural_score >= 0.66 and temporary_score < 0.66:
        label = "Structural energy tightness"
        read = "Bahkan kalau headline perang mereda, chain energy masih punya tenaga. Ini lebih besar dari sekadar war premium sesaat."
    elif temporary_score >= 0.66 and structural_score < 0.50:
        label = "Temporary war premium"
        read = "Move energy lebih dekat ke headline geopolitik daripada krisis struktural yang benar-benar menetap."
    elif premium_label == "War premium fading" and structural_score >= 0.50:
        label = "War fades, structural stress stays"
        read = "Headline geopolitik mulai kehilangan daya, tapi legacy energy / scarcity proxies belum benar-benar mati."
    else:
        label = "Mixed energy shock"
        read = "Chain energy memberi pesan campuran; jangan treat semua energy move sebagai oil beta murni."

    rows = [
        ("Oil impulse", move_text(oil_pct), "Headline-level" if abs(oil_pct) > 3 else "Mild"),
        ("Legacy energy 20d", move_text(xle20), "Broad energy equity read"),
        ("Coal 20d", move_text(kol20), "Scarcity / coal leadership"),
        ("Solar 20d", move_text(tan20), "Transition catch-up"),
        ("Dominant energy theme", theme, label),
    ]
    return {"label": label, "read": read, "theme": theme, "rows": rows, "score": max(structural_score, temporary_score)}


def classify_war_damage_persistence(state: MacroState, prices: pd.DataFrame, news_items: List[Dict[str, str]]) -> Dict[str, object]:
    geo_hits = headline_hit_count(news_items, ["iran", "war", "hormuz", "missile", "ceasefire", "truce", "netanyahu"])
    oil_pct = safe_pct_delta(state.oil_now, state.oil_prev)
    dcredit = delta(state.credit_now, state.credit_prev)
    dbreadth = delta(state.breadth_now, state.breadth_prev)
    premium_label, _ = news_premium_status(state, news_items)
    if geo_hits > 0 and premium_label == "War premium fading" and (dcredit > 0 or dbreadth < 0):
        label = "War fades, damage persists"
        read = "Headline perang mulai kurang efektif mengangkat oil, tapi stress ke growth / credit / breadth belum hilang."
        score = 0.72
    elif geo_hits > 0 and oil_pct > 3:
        label = "War premium alive"
        read = "Headline perang masih diterjemahkan market menjadi premium langsung di energy."
        score = 0.70
    elif geo_hits > 0 and oil_pct <= 0 and dcredit <= 0:
        label = "Market absorbing war shock"
        read = "Headline tetap ada, tapi market mulai absorb; baca apakah ini temporary relief atau benar-benar structural improvement."
        score = 0.45
    else:
        label = "No dominant war filter"
        read = "War bukan lagi filter tunggal yang mengontrol tape; driver lain lebih dominan."
        score = 0.25
    return {"label": label, "read": read, "score": score}


def classify_front_end_stress(state: MacroState, news_items: List[Dict[str, str]]) -> Dict[str, object]:
    dy2 = delta(state.y2_now, state.y2_prev)
    d10 = delta(state.y10_now, state.y10_prev)
    d30 = delta(state.y30_now, state.y30_prev)
    ddxy = delta(state.dxy_now, state.dxy_prev)
    dcredit = delta(state.credit_now, state.credit_prev)
    auction_hits = headline_hit_count(news_items, ["auction", "refund", "bill", "issuance", "maturity", "rollover", "treasury auction"])
    front_score = bool_score(dy2 > 0.05, dy2 > d10 + 0.03, dy2 > d30 + 0.03, ddxy >= 0, dcredit > 0 or auction_hits > 0)
    if front_score >= 0.75:
        label = "Front-end funding stress"
        read = "2Y memimpin naik. Ini lebih dekat ke funding / rollover / front-end pressure daripada sekadar reflation sehat."
    elif front_score >= 0.55:
        label = "Auction / rollover watch"
        read = "Short-end lebih tegang daripada long-end. Belum pasti accident, tapi cukup untuk menahan risk-on bersih."
    elif dy2 <= 0:
        label = "Front-end relief"
        read = "2Y mereda; ini salah satu syarat terpenting untuk bottoming risk assets yang lebih sehat."
    else:
        label = "Mixed front-end"
        read = "2Y belum memberi pesan yang benar-benar bersih."
    return {"label": label, "read": read, "score": front_score, "dy2": dy2, "d10": d10, "d30": d30}


def classify_private_credit_stress(state: MacroState, prices: pd.DataFrame, news_items: List[Dict[str, str]]) -> Dict[str, object]:
    dcredit = delta(state.credit_now, state.credit_prev)
    bizd20 = safe_trailing_return(prices, "BIZD", 20)
    owl20 = safe_trailing_return(prices, "OWL", 20)
    hyg20 = safe_trailing_return(prices, "HYG", 20)
    pc_hits = headline_hit_count(news_items, ["private credit", "bdc", "direct lending", "owl", "alt manager", "hedge funds way to short private credit"])
    score = bool_score(dcredit > 0.10, (not pd.isna(bizd20) and bizd20 < -3) or (not pd.isna(owl20) and owl20 < -5), not pd.isna(hyg20) and hyg20 < -1, pc_hits > 0)
    if score >= 0.75:
        label = "Private credit spillover risk"
        read = "Bukan otomatis bottom. Treat sebagai stress engine sampai spreads / BDC proxies / funding pressure mulai stabil."
    elif score >= 0.55:
        label = "Private credit stress rising"
        read = "Headlines dan tape mulai sinkron. Ini lebih cocok dibaca sebagai crack watch daripada cheapness."
    elif pc_hits > 0 and dcredit <= 0:
        label = "Private credit fear, tape not confirming"
        read = "Narasi seram ada, tapi spread / proxies belum benar-benar rusak. Bisa jadi fear premium duluan."
        score = max(score, 0.35)
    else:
        label = "Private credit contained"
        read = "Belum ada cukup bukti spillover ke tape yang lebih luas."
        score = max(score, 0.20)
    rows = [
        ("Credit spread delta", move_text(dcredit, pct=False), "Worse" if dcredit > 0 else "Stable / better"),
        ("BIZD 20d", move_text(bizd20), "BDC / listed private credit proxy"),
        ("OWL 20d", move_text(owl20), "Alt manager / exposure proxy"),
        ("HYG 20d", move_text(hyg20), "High yield tape"),
    ]
    return {"label": label, "read": read, "score": score, "rows": rows}


def classify_gold_mode(state: MacroState, prices: pd.DataFrame, news_items: List[Dict[str, str]]) -> Dict[str, object]:
    gold_pct = safe_pct_delta(state.gold_now, state.gold_prev)
    oil_pct = safe_pct_delta(state.oil_now, state.oil_prev)
    dy2 = delta(state.y2_now, state.y2_prev)
    dry = delta(state.ry10_now, state.ry10_prev)
    qqq5 = short_ret(prices, "QQQ", 5)
    btc5 = short_ret(prices, "BTC", 5)
    if gold_pct > 1.5 and oil_pct <= 0 and delta(state.credit_now, state.credit_prev) > 0:
        label = "Fear / hedge bid"
        read = "Gold dipakai sebagai hedge saat growth / credit tape memburuk."
    elif gold_pct > 1.5 and oil_pct > 0:
        label = "Inflation / geopolitics hedge"
        read = "Gold naik bersama oil. Ini lebih dekat ke inflation / geopolitics mix daripada disinflation relief."
    elif gold_pct < -1.5 and ((not pd.isna(qqq5) and qqq5 < 0) or (not pd.isna(btc5) and btc5 < 0)) and oil_pct < 0:
        label = "Liquidity squeeze gold down"
        read = "Gold ikut dijual bareng risk assets dan commodities. Ini lebih dekat ke deleveraging / collateral stress."
    elif gold_pct < -1.0 and (dy2 > 0 or dry > 0):
        label = "Real-yield / funding pressure"
        read = "Gold lemah bukan karena uncertainty hilang, tapi karena rates / funding masih menekan."
    elif gold_pct < 0 and oil_pct < 0 and delta(state.credit_now, state.credit_prev) <= 0:
        label = "Uncertainty unwind"
        read = "Gold dan oil sama-sama melemah, sementara credit tidak memburuk. Itu lebih dekat ke unwind daripada panic."
    else:
        label = "Mixed gold signal"
        read = "Gold belum memberi pesan tunggal; baca bersama oil, 2Y, real yields, credit, dan breadth."
    return {"label": label, "read": read, "score": bool_score(gold_pct < -1.5, dy2 > 0 or dry > 0)}


def classify_liquidity_event(state: MacroState, prices: pd.DataFrame) -> Dict[str, object]:
    moves = {
        "Stocks": short_ret(prices, "SPY", 5) if "SPY" in prices.columns else short_ret(prices, "QQQ", 5),
        "Dollar": safe_pct_delta(state.dxy_now, state.dxy_prev),
        "Gold": safe_pct_delta(state.gold_now, state.gold_prev),
        "Silver": short_ret(prices, "SLV", 5),
        "Copper": short_ret(prices, "CPER", 5),
        "Oil": safe_pct_delta(state.oil_now, state.oil_prev),
        "Bitcoin": short_ret(prices, "BTC", 5),
    }
    vals = [v for v in moves.values() if not pd.isna(v)]
    neg = sum(v < 0 for v in vals)
    total = len(vals)
    score = neg / total if total else 0.0
    if total >= 5 and neg >= max(5, total - 1) and moves.get("Gold", 0) < 0:
        label = "Everything-down liquidation"
        read = "Ini bukan rotasi normal. Likuiditas / deleveraging / collateral stress lebih mungkin dominan."
    elif total >= 5 and neg >= 4:
        label = "Cross-asset liquidation risk"
        read = "Banyak sleeve turun bareng. Bottoming butuh credit dan front-end relief dulu."
    elif neg >= 3:
        label = "Broad risk-off"
        read = "Risk-off cukup luas, tapi belum tentu full liquidation regime."
    else:
        label = "Normal rotation / mixed tape"
        read = "Belum ada bukti kuat event everything-down."
    rows = [(k, move_text(v), "Down" if not pd.isna(v) and v < 0 else "Up / mixed") for k, v in moves.items()]
    return {"label": label, "read": read, "score": score, "rows": rows}


def classify_bottom_quality(state: MacroState, prices: pd.DataFrame, news_items: List[Dict[str, str]], next_quad: str) -> Dict[str, object]:
    next_diag = next_phase_activation(state, next_quad, news_items)
    trans = transition_quality_score(state, news_items)
    front = classify_front_end_stress(state, news_items)
    pc = classify_private_credit_stress(state, prices, news_items)
    liq = classify_liquidity_event(state, prices)
    breadth_ok = state.breadth_now >= state.breadth_prev
    credit_ok = state.credit_now <= state.credit_prev
    score = 0.35 * next_diag["score"] + 0.25 * trans["score"] + 0.15 * (1.0 if breadth_ok else 0.0) + 0.15 * (1.0 if credit_ok else 0.0) + 0.10 * (1.0 - min(1.0, front["score"]))
    if liq["score"] > 0.80 and (not breadth_ok or not credit_ok):
        label = "Not a bottom yet"
        read = "Masih lebih dekat ke liquidation / stress daripada durable low."
        score = min(score, 0.30)
    elif liq["score"] > 0.75 and breadth_ok and trans["score"] >= 0.45:
        label = "Capitulation candidate"
        read = "Panic sudah luas, tapi baru boleh dibaca bottom kalau breadth / credit / front-end mulai membaik."
        score = max(score, 0.42)
    elif score >= 0.75 and pc["score"] < 0.55:
        label = "Durable bottom building"
        read = "Drivers bawahnya mulai lengkap: transition lebih bersih, funding tidak sejahat sebelumnya, breadth/credit membaik."
    elif score >= 0.58:
        label = "Tradable / base-building"
        read = "Ada alasan untuk bounce lebih dari sekadar dead cat, tapi belum semua filter bersih."
    elif score >= 0.42:
        label = "Panic bounce risk"
        read = "Relief bisa terjadi, tapi kualitas bottom masih rapuh."
    else:
        label = "Not a bottom yet"
        read = "Belum ada cukup bukti bahwa pain terburuk sudah lewat."
    return {"label": label, "read": read, "score": float(np.clip(score, 0, 1))}


def build_structural_stress_table(state: MacroState, prices: pd.DataFrame, news_items: List[Dict[str, str]], next_quad: str) -> pd.DataFrame:
    energy = classify_energy_shock(state, prices, news_items)
    war = classify_war_damage_persistence(state, prices, news_items)
    pc = classify_private_credit_stress(state, prices, news_items)
    gold = classify_gold_mode(state, prices, news_items)
    front = classify_front_end_stress(state, news_items)
    liq = classify_liquidity_event(state, prices)
    bottom = classify_bottom_quality(state, prices, news_items, next_quad)
    rows = [
        ("Energy shock", energy["label"], energy["read"]),
        ("War damage", war["label"], war["read"]),
        ("Private credit", pc["label"], pc["read"]),
        ("Gold mode", gold["label"], gold["read"]),
        ("Front-end / rollover", front["label"], front["read"]),
        ("Everything-down tape", liq["label"], liq["read"]),
        ("Bottom quality", bottom["label"], bottom["read"]),
    ]
    return pd.DataFrame(rows, columns=["Module", "State", "Read"])


def build_energy_theme_split_table(prices: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("Legacy energy / oil", move_text(safe_trailing_return(prices, "XLE", 20)), move_text(safe_trailing_return(prices, "USO", 20)), "Best for legacy oil / integrated / upstream read."),
        ("Coal / scarcity", move_text(safe_trailing_return(prices, "KOL", 20)), "n/a", "Useful when power scarcity / coal leadership is the real winner, not oil alone."),
        ("Solar / transition", move_text(safe_trailing_return(prices, "TAN", 20)), "n/a", "Useful to test whether market is already looking beyond near-term scarcity."),
        ("Metals / miners", move_text(safe_trailing_return(prices, "PICK", 20)), move_text(safe_trailing_return(prices, "CPER", 20)), "Tells you whether commodity strength is broadening beyond energy."),
    ]
    return pd.DataFrame(rows, columns=["Theme", "Equity proxy 20d", "Underlying 20d", "Read"])


def build_private_credit_snapshot(prices: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("BIZD", move_text(safe_trailing_return(prices, "BIZD", 5)), move_text(safe_trailing_return(prices, "BIZD", 20)), "Listed BDC / private credit proxy"),
        ("OWL", move_text(safe_trailing_return(prices, "OWL", 5)), move_text(safe_trailing_return(prices, "OWL", 20)), "Alt manager exposure proxy"),
        ("HYG", move_text(safe_trailing_return(prices, "HYG", 5)), move_text(safe_trailing_return(prices, "HYG", 20)), "Public HY tape / spillover monitor"),
        ("TLT", move_text(safe_trailing_return(prices, "TLT", 5)), move_text(safe_trailing_return(prices, "TLT", 20)), "Duration relief vs long-end stress"),
    ]
    return pd.DataFrame(rows, columns=["Proxy", "5d", "20d", "Read"])


def build_liquidation_snapshot(prices: pd.DataFrame, state: MacroState) -> pd.DataFrame:
    rows = [
        ("SPY / stocks", move_text(short_ret(prices, "SPY", 5) if "SPY" in prices.columns else short_ret(prices, "QQQ", 5))),
        ("Dollar", move_text(safe_pct_delta(state.dxy_now, state.dxy_prev))),
        ("Gold", move_text(safe_pct_delta(state.gold_now, state.gold_prev))),
        ("Silver", move_text(short_ret(prices, "SLV", 5))),
        ("Copper", move_text(short_ret(prices, "CPER", 5))),
        ("Oil", move_text(safe_pct_delta(state.oil_now, state.oil_prev))),
        ("Bitcoin", move_text(short_ret(prices, "BTC", 5))),
    ]
    return pd.DataFrame(rows, columns=["Sleeve", "5d move"])


def build_compare_board_table_v2(state: MacroState, prices: pd.DataFrame, current_quad_name: str, next_quad_name: str, news_items: List[Dict[str, str]], mode: str) -> pd.DataFrame:
    treasury_label, treasury_read = treasury_regime_text(state)
    premium_label, premium_read = news_premium_status(state, news_items)
    trans = transition_quality_score(state, news_items)
    top_seq = build_top_sequence_table(state, current_quad_name, news_items)
    bot_seq = build_bottom_sequence_table(state, next_quad_name, news_items)
    energy = classify_energy_shock(state, prices, news_items)
    front = classify_front_end_stress(state, news_items)
    gold = classify_gold_mode(state, prices, news_items)
    pc = classify_private_credit_stress(state, prices, news_items)
    liq = classify_liquidity_event(state, prices)
    bottomq = classify_bottom_quality(state, prices, news_items, next_quad_name)
    if mode == "current":
        maturity, _, _, _ = current_phase_maturity(state, current_quad_name, news_items)
        top_diag = current_top_diagnosis(state, current_quad_name, news_items)
        rows = [
            ("Quad", f"{current_quad_name} — {QUAD_THEMES[current_quad_name]['theme']}"),
            ("Lifecycle", maturity),
            ("Treasury", f"{treasury_label}: {treasury_read}"),
            ("Current winners", QUAD_THEMES[current_quad_name]["current_winners"]),
            ("Energy mode", f"{energy['label']}: {energy['theme']}"),
            ("Funding mode", front["label"]),
            ("Tape mode", liq["label"]),
            ("Top status", f"{top_diag['label']} ({top_diag['score']:.0%})"),
            ("Step 1", top_seq.iloc[0]["Sequence"]),
            ("Step 2", top_seq.iloc[1]["Sequence"]),
            ("Step 3", top_seq.iloc[2]["Sequence"]),
            ("Step 4", top_seq.iloc[3]["Sequence"]),
            ("Final confirm", top_seq.iloc[4]["Sequence"]),
            ("News premium", f"{premium_label}: {premium_read}"),
            ("Transition quality", trans["label"]),
        ]
    else:
        next_diag = next_phase_activation(state, next_quad_name, news_items)
        rows = [
            ("Next likely quad", f"{next_quad_name} — {QUAD_THEMES[next_quad_name]['theme']}"),
            ("Activation", f"{next_diag['label']} ({next_diag['score']:.0%})"),
            ("Treasury", f"{treasury_label}: {treasury_read}"),
            ("First likely winners", QUAD_THEMES[next_quad_name]["first_next"]),
            ("Bottom quality", f"{bottomq['label']} ({bottomq['score']:.0%})"),
            ("Gold mode", gold["label"]),
            ("Private credit", pc["label"]),
            ("Funding mode", front["label"]),
            ("Step 1", bot_seq.iloc[0]["Sequence"]),
            ("Step 2", bot_seq.iloc[1]["Sequence"]),
            ("Step 3", bot_seq.iloc[2]["Sequence"]),
            ("Step 4", bot_seq.iloc[3]["Sequence"]),
            ("Step 5", bot_seq.iloc[4]["Sequence"]),
            ("Final confirm", bot_seq.iloc[5]["Sequence"]),
            ("Transition quality", trans["label"]),
        ]
    return pd.DataFrame(rows, columns=["Field", "Read"])


def merge_field_tables(*dfs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seen = set()
    for df in dfs:
        if df is None or df.empty:
            continue
        if len(df.columns) < 2:
            continue
        c0, c1 = df.columns[:2]
        for _, row in df.iterrows():
            field = str(row[c0]).strip()
            read = str(row[c1]).strip()
            if not field or field.lower() == 'nan' or field in seen:
                continue
            seen.add(field)
            rows.append((field, read))
    return pd.DataFrame(rows, columns=['Field', 'Read'])


def build_context_board_table(state: MacroState, prices: pd.DataFrame, news_items: List[Dict[str, str]], next_quad: str, ihsg_overlay: Dict[str, object]) -> pd.DataFrame:
    trans = transition_quality_score(state, news_items)
    treasury_label, treasury_read = treasury_regime_text(state)
    energy = classify_energy_shock(state, prices, news_items)
    war = classify_war_damage_persistence(state, prices, news_items)
    pc = classify_private_credit_stress(state, prices, news_items)
    gold = classify_gold_mode(state, prices, news_items)
    front = classify_front_end_stress(state, news_items)
    liq = classify_liquidity_event(state, prices)
    bottom = classify_bottom_quality(state, prices, news_items, next_quad)
    premium_label, premium_read = news_premium_status(state, news_items)
    rows = [
        ('Transition quality', f"{trans['label']} ({float(trans.get('score', 0.0)):.0%}) — {trans['read'] if 'read' in trans else 'Cross-driver check'}"),
        ('Treasury regime', f"{treasury_label} — {treasury_read}"),
        ('Front-end / rollover', f"{front['label']} — {front['read']}"),
        ('Private credit', f"{pc['label']} — {pc['read']}"),
        ('Structural energy', f"{energy['label']} — {energy['read']}"),
        ('War damage persistence', f"{war['label']} — {war['read']}"),
        ('Gold mode', f"{gold['label']} — {gold['read']}"),
        ('Everything-down tape', f"{liq['label']} — {liq['read']}"),
        ('Bottom quality', f"{bottom['label']} ({float(bottom.get('score', 0.0)):.0%}) — {bottom['read']}"),
        ('IHSG filter', f"{ihsg_overlay['dominant']} — {ihsg_overlay['headline']}"),
        ('Dominant premium', f"{premium_label} — {premium_read}"),
    ]
    return pd.DataFrame(rows, columns=['Field', 'Read'])


def _maturity_bias(label: str) -> float:
    lab = (label or '').lower()
    if 'early' in lab:
        return 0.95
    if 'mid' in lab:
        return 0.75
    if 'late' in lab:
        return 0.35
    if 'crowd' in lab or 'topping' in lab:
        return 0.15
    if 'transition' in lab:
        return 0.25
    return 0.50


def build_scenario_probability_table(state: MacroState, prices: pd.DataFrame, current_quad: str, next_quad: str, news_items: List[Dict[str, str]], ihsg_overlay: Dict[str, object]) -> pd.DataFrame:
    maturity, _, _, _ = current_phase_maturity(state, current_quad, news_items)
    top = current_top_diagnosis(state, current_quad, news_items)
    nxt = next_phase_activation(state, next_quad, news_items)
    trans = transition_quality_score(state, news_items)
    front = classify_front_end_stress(state, news_items)
    pc = classify_private_credit_stress(state, prices, news_items)
    liq = classify_liquidity_event(state, prices)
    energy = classify_energy_shock(state, prices, news_items)
    war = classify_war_damage_persistence(state, prices, news_items)
    bottom = classify_bottom_quality(state, prices, news_items, next_quad)

    top_s = float(top.get('score', 0.0))
    nxt_s = float(nxt.get('score', 0.0))
    trans_s = float(trans.get('score', 0.0))
    front_s = float(front.get('score', 0.0))
    pc_s = float(pc.get('score', 0.0))
    liq_s = float(liq.get('score', 0.0))
    energy_s = float(energy.get('score', 0.0))
    war_s = float(war.get('score', 0.0))
    bottom_s = float(bottom.get('score', 0.0))
    ihsg_em = float(ihsg_overlay.get('em_stress', 0.0))
    maturity_s = _maturity_bias(maturity)

    raw = {
        'Current continues': max(0.05, 0.42 * (1 - top_s) + 0.20 * maturity_s + 0.18 * (1 - liq_s) + 0.10 * (1 - pc_s) + 0.10 * (1 - trans_s)),
        'Current late / topping': max(0.05, 0.45 * top_s + 0.18 * front_s + 0.12 * liq_s + 0.12 * pc_s + 0.13 * (1 - maturity_s)),
        'Clean transition': max(0.05, 0.38 * nxt_s + 0.28 * trans_s + 0.20 * bottom_s + 0.07 * (1 - pc_s) + 0.07 * (1 - liq_s)),
        'Fragile transition / false bottom': max(0.05, 0.30 * nxt_s + 0.15 * trans_s + 0.20 * (1 - bottom_s) + 0.15 * front_s + 0.10 * pc_s + 0.10 * liq_s),
        'Liquidity event / forced selling': max(0.05, 0.48 * liq_s + 0.22 * front_s + 0.15 * pc_s + 0.15 * (1 - trans_s)),
        'Credit spillover': max(0.05, 0.55 * pc_s + 0.20 * front_s + 0.15 * liq_s + 0.10 * (1 - bottom_s)),
        'Structural energy persistence': max(0.05, 0.58 * energy_s + 0.25 * war_s + 0.17 * top_s),
        'EM stress override': max(0.05, 0.60 * ihsg_em + 0.20 * front_s + 0.20 * pc_s),
    }
    total = sum(raw.values())
    probs = {k: v / total for k, v in raw.items()}

    meta = {
        'Current continues': ('old winners still carry', 'breadth not worsening too fast, front-end stress cools', 'new highs fail and breadth shrinks further', 'stay with current winners but trim if leadership narrows'),
        'Current late / topping': ('crowding + breadth decay', '2Y / real yields / dollar pressure confirms', 'followers re-expand and credit improves', 'reduce chasing, watch first cracks'),
        'Clean transition': ('next leaders stabilize first', 'breadth + credit + 2Y relief confirm together', 'transition stays narrow', 'rotate gradually toward next winners'),
        'Fragile transition / false bottom': ('bounce exists but internals weak', 'bad news stops making lows only briefly', '2Y, credit, breadth fail together', 'keep size smaller, demand confirmation'),
        'Liquidity event / forced selling': ('everything-down tape', 'cross-asset weakness broadens and hedges fail', 'selling pressure stops broadening', 'raise cash / reduce gross / avoid junk beta'),
        'Credit spillover': ('private-credit narrative spreads', 'BIZD/OWL/HYG and credit spreads worsen', 'stress stays contained in headlines only', 'avoid weak credit beta, focus quality'),
        'Structural energy persistence': ('war fades but energy still tight', 'coal/power/miners confirm beyond crude', 'energy complex fades together', 'prefer selective scarcity plays, not broad risk chase'),
        'EM stress override': ('USDIDR / DXY / EM tape dominate', 'EEM/EIDO/FX fail to confirm risk-on', 'dollar and front-end finally relax', 'treat IHSG as selective only until flow improves'),
    }

    rows = []
    for scen, prob in sorted(probs.items(), key=lambda kv: kv[1], reverse=True):
        trig, conf, brk, act = meta[scen]
        rows.append((scen, f'{prob:.0%}', trig, conf, brk, act))
    return pd.DataFrame(rows, columns=['Scenario', 'Probability', 'Trigger', 'What confirms', 'What breaks it', 'What to do'])




def clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def shock_word(score: float) -> str:
    if score >= 0.75:
        return "Very bullish"
    if score >= 0.55:
        return "Bullish"
    if score >= 0.35:
        return "Constructive"
    if score <= -0.75:
        return "Crash risk"
    if score <= -0.55:
        return "Bearish"
    if score <= -0.35:
        return "Weak"
    return "Mixed"


def _scenario_signal_flags(state: MacroState, war_diag: Dict[str, object], energy_diag: Dict[str, object]) -> Dict[str, float]:
    dy2 = delta(state.y2_now, state.y2_prev)
    dry = delta(state.ry10_now, state.ry10_prev)
    ddxy = delta(state.dxy_now, state.dxy_prev)
    dcredit = delta(state.credit_now, state.credit_prev)
    dbreadth = delta(state.breadth_now, state.breadth_prev)
    doil_pct = safe_pct_delta(state.oil_now, state.oil_prev)
    dgrowth = delta(state.growth_now, state.growth_prev)
    dinfl = delta(state.inflation_now, state.inflation_prev)

    war_label = str(war_diag.get("label", ""))
    energy_label = str(energy_diag.get("label", ""))

    return {
        "y2_relief": clamp01(-dy2 / 0.35),
        "y2_stress": clamp01(dy2 / 0.35),
        "real_relief": clamp01(-dry / 0.25),
        "real_stress": clamp01(dry / 0.25),
        "dxy_relief": clamp01(-ddxy / 2.0),
        "dxy_stress": clamp01(ddxy / 2.0),
        "credit_relief": clamp01(-dcredit / 0.45),
        "credit_stress": clamp01(dcredit / 0.45),
        "breadth_repair": clamp01(dbreadth / 4.0),
        "breadth_decay": clamp01(-dbreadth / 4.0),
        "oil_relief": clamp01(-doil_pct / 8.0),
        "oil_pressure": clamp01(doil_pct / 8.0),
        "growth_accel": clamp01(dgrowth / 0.55),
        "growth_slow": clamp01(-dgrowth / 0.55),
        "infl_cool": clamp01(-dinfl / 0.45),
        "infl_heat": clamp01(dinfl / 0.45),
        "war_alive": 1.0 if war_label == "War premium alive" else 0.0,
        "war_damage": 1.0 if war_label == "War fades, damage persists" else 0.0,
        "war_absorb": 1.0 if war_label in ["Market absorbing war shock", "No dominant war filter"] else 0.0,
        "energy_structural": 1.0 if energy_label in ["Structural energy tightness", "War fades, structural stress stays"] else 0.0,
        "energy_temporary": 1.0 if energy_label == "Temporary war premium" else 0.0,
    }


WHAT_NEXT_IMPACT_MAP = {
    "Current regime extends": {
        "path": "Stay with current quad",
        "impacts_by_quad": {
            "Quad 1": {"US stocks": 0.90, "Index futures": 0.75, "Treasury futures": 0.55, "Commodities": -0.10, "Energy": -0.35, "Emerging Mkts": 0.65, "IHSG": 0.45, "Crypto": 0.75, "USD": -0.45, "Gold": 0.05},
            "Quad 2": {"US stocks": 0.55, "Index futures": 0.65, "Treasury futures": -0.55, "Commodities": 0.70, "Energy": 0.80, "Emerging Mkts": 0.45, "IHSG": 0.60, "Crypto": 0.20, "USD": -0.05, "Gold": 0.15},
            "Quad 3": {"US stocks": -0.20, "Index futures": -0.30, "Treasury futures": 0.85, "Commodities": -0.45, "Energy": -0.55, "Emerging Mkts": -0.45, "IHSG": -0.15, "Crypto": -0.35, "USD": 0.35, "Gold": 0.60},
            "Quad 4": {"US stocks": -0.50, "Index futures": -0.60, "Treasury futures": -0.55, "Commodities": 0.65, "Energy": 0.95, "Emerging Mkts": -0.25, "IHSG": 0.20, "Crypto": -0.55, "USD": 0.45, "Gold": 0.65},
        },
    },
    "Clean Quad 1 relief": {
        "path": "Transition -> Quad 1",
        "impacts": {"US stocks": 0.90, "Index futures": 0.85, "Treasury futures": 0.75, "Commodities": -0.10, "Energy": -0.45, "Emerging Mkts": 0.80, "IHSG": 0.55, "Crypto": 0.90, "USD": -0.75, "Gold": 0.10},
    },
    "Healthy Quad 2 reflation": {
        "path": "Transition -> Quad 2",
        "impacts": {"US stocks": 0.55, "Index futures": 0.70, "Treasury futures": -0.65, "Commodities": 0.75, "Energy": 0.80, "Emerging Mkts": 0.55, "IHSG": 0.65, "Crypto": 0.20, "USD": -0.10, "Gold": 0.05},
    },
    "Quad 3 defensive slowdown": {
        "path": "Transition -> Quad 3",
        "impacts": {"US stocks": -0.35, "Index futures": -0.35, "Treasury futures": 0.90, "Commodities": -0.55, "Energy": -0.60, "Emerging Mkts": -0.45, "IHSG": -0.10, "Crypto": -0.45, "USD": 0.35, "Gold": 0.80},
    },
    "Quad 4 stagflation": {
        "path": "Transition -> Quad 4",
        "impacts": {"US stocks": -0.55, "Index futures": -0.60, "Treasury futures": -0.70, "Commodities": 0.80, "Energy": 0.95, "Emerging Mkts": -0.30, "IHSG": 0.20, "Crypto": -0.60, "USD": 0.50, "Gold": 0.70},
    },
    "War de-escalation clean relief": {
        "path": "Geo shock fades -> Quad 3 -> Quad 1",
        "impacts": {"US stocks": 0.80, "Index futures": 0.75, "Treasury futures": 0.70, "Commodities": -0.35, "Energy": -0.75, "Emerging Mkts": 0.85, "IHSG": 0.60, "Crypto": 0.75, "USD": -0.70, "Gold": -0.05},
    },
    "War fades, damage persists": {
        "path": "Headline relief -> damaged growth / credit",
        "impacts": {"US stocks": 0.05, "Index futures": 0.00, "Treasury futures": 0.55, "Commodities": -0.10, "Energy": -0.05, "Emerging Mkts": -0.15, "IHSG": 0.05, "Crypto": -0.05, "USD": 0.10, "Gold": 0.55},
    },
    "Front-end funding squeeze": {
        "path": "Funding squeeze -> risk compression",
        "impacts": {"US stocks": -0.75, "Index futures": -0.80, "Treasury futures": -0.45, "Commodities": -0.25, "Energy": -0.20, "Emerging Mkts": -0.70, "IHSG": -0.55, "Crypto": -0.85, "USD": 0.80, "Gold": -0.10},
    },
    "Private credit spillover": {
        "path": "Credit accident -> defensive / policy watch",
        "impacts": {"US stocks": -0.70, "Index futures": -0.75, "Treasury futures": 0.70, "Commodities": -0.35, "Energy": -0.35, "Emerging Mkts": -0.70, "IHSG": -0.60, "Crypto": -0.75, "USD": 0.65, "Gold": 0.65},
    },
    "Liquidity / forced deleveraging": {
        "path": "Everything-down tape -> capitulation",
        "impacts": {"US stocks": -0.95, "Index futures": -0.95, "Treasury futures": 0.55, "Commodities": -0.75, "Energy": -0.70, "Emerging Mkts": -0.95, "IHSG": -0.90, "Crypto": -0.98, "USD": 0.90, "Gold": -0.05},
    },
    "Structural energy persistence": {
        "path": "Scarcity / legacy energy persists",
        "impacts": {"US stocks": -0.15, "Index futures": -0.20, "Treasury futures": -0.45, "Commodities": 0.75, "Energy": 0.92, "Emerging Mkts": 0.10, "IHSG": 0.35, "Crypto": -0.30, "USD": 0.15, "Gold": 0.55},
    },
    "EM dollar stress override": {
        "path": "Dollar up -> EM selective only",
        "impacts": {"US stocks": -0.05, "Index futures": -0.10, "Treasury futures": 0.10, "Commodities": 0.00, "Energy": 0.05, "Emerging Mkts": -0.95, "IHSG": -0.80, "Crypto": -0.45, "USD": 0.95, "Gold": 0.25},
    },
}


WHAT_NEXT_META = {
    "Current regime extends": {
        "driver": "current winners still hold, top risk not extreme, stress not broadening",
        "confirm": "breadth does not worsen too much and front-end stress cools",
        "break": "leaders fail to extend while breadth and credit deteriorate together",
        "action": "stay with current winners but stop chasing late followers",
    },
    "Clean Quad 1 relief": {
        "driver": "2Y + real yields + dollar all relax together",
        "confirm": "credit stabilizes, breadth broadens, and quality leaders hold pullbacks",
        "break": "move stays limited to Nasdaq/BTC while credit or EM still fail",
        "action": "rotate gradually into duration / quality risk-on",
    },
    "Healthy Quad 2 reflation": {
        "driver": "growth improves without front-end stress exploding",
        "confirm": "cyclicals, commodities, and selected EM all participate",
        "break": "only oil rises while breadth stays narrow and dollar dominates",
        "action": "favor cyclicals / commodity-linked sleeves over long duration",
    },
    "Quad 3 defensive slowdown": {
        "driver": "growth cools and inflation also cools",
        "confirm": "bonds, gold, and defensives outperform while credit is contained",
        "break": "growth fear morphs into systemic credit stress or oil re-accelerates",
        "action": "lean defensive first, then wait for cleaner next leaders",
    },
    "Quad 4 stagflation": {
        "driver": "inflation / commodity pressure stays hot while growth rolls over",
        "confirm": "energy and hard assets outperform while broad beta stays weak",
        "break": "oil rolls over and front-end / dollar pressure fades",
        "action": "favor hard assets and pricing power, avoid weak beta",
    },
    "War de-escalation clean relief": {
        "driver": "geo premium fades and oil loses its shock bid",
        "confirm": "energy underperforms while bonds, EM, and quality growth all improve",
        "break": "headline calm arrives but credit, breadth, and shipping damage stay broken",
        "action": "buy relief leaders, not lagging war hedges",
    },
    "War fades, damage persists": {
        "driver": "headline premium fades but macro damage remains in credit / breadth / growth",
        "confirm": "oil stops helping but broad risk still cannot broaden",
        "break": "credit and breadth repair quickly, turning relief into a clean transition",
        "action": "treat relief as selective only until internals heal",
    },
    "Front-end funding squeeze": {
        "driver": "2Y leads higher and funding pressure dominates",
        "confirm": "dollar firms, beta compresses, and risk-on breadth fails",
        "break": "2Y quickly reverses lower and auction / funding fears fade",
        "action": "cut weak beta and respect short-end stress",
    },
    "Private credit spillover": {
        "driver": "credit cracks spread from private credit / BDC / HY complex",
        "confirm": "BIZD / OWL / HYG worsen while breadth and credit tighten together",
        "break": "stress stays contained to headlines and spreads stop worsening",
        "action": "stay in quality / defensives until spillover stops expanding",
    },
    "Liquidity / forced deleveraging": {
        "driver": "everything-down tape with correlations rising toward one",
        "confirm": "cross-asset selling broadens and hedges stop working cleanly",
        "break": "selling pressure stops broadening and policy / balance-sheet relief shows up",
        "action": "raise cash, cut gross, and avoid false bottoms",
    },
    "Structural energy persistence": {
        "driver": "energy scarcity survives beyond a simple war premium",
        "confirm": "coal / power / energy equities hold up even if headlines cool",
        "break": "energy chain fades together, not just crude headlines",
        "action": "favor direct scarcity expressions over broad macro beta",
    },
    "EM dollar stress override": {
        "driver": "DXY / EM FX pressure dominates local beta",
        "confirm": "EEM / EIDO / IHSG fail even when US quality looks okay",
        "break": "dollar and front-end both relax and EM breadth improves",
        "action": "treat EM and IHSG as selective only until flow turns",
    },
}


def _scenario_impacts_for_name(name: str, current_quad_name: str) -> Dict[str, float]:
    cfg = WHAT_NEXT_IMPACT_MAP[name]
    if "impacts" in cfg:
        return cfg["impacts"]
    return cfg["impacts_by_quad"].get(current_quad_name, cfg["impacts_by_quad"]["Quad 1"])


def build_what_next_scenario_engine(state: MacroState, prices: pd.DataFrame, current_quad_name: str, next_quad_name: str, news_items: List[Dict[str, str]], ihsg_overlay: Dict[str, object]) -> Dict[str, object]:
    maturity, _, _, _ = current_phase_maturity(state, current_quad_name, news_items)
    top = current_top_diagnosis(state, current_quad_name, news_items)
    nxt = next_phase_activation(state, next_quad_name, news_items)
    trans = transition_quality_score(state, news_items)
    front = classify_front_end_stress(state, news_items)
    pc = classify_private_credit_stress(state, prices, news_items)
    liq = classify_liquidity_event(state, prices)
    energy = classify_energy_shock(state, prices, news_items)
    war = classify_war_damage_persistence(state, prices, news_items)
    bottom = classify_bottom_quality(state, prices, news_items, next_quad_name)
    qprobs = next_quad_probabilities(state)
    flags = _scenario_signal_flags(state, war, energy)

    top_s = float(top.get("score", 0.0))
    nxt_s = float(nxt.get("score", 0.0))
    trans_s = float(trans.get("score", 0.0))
    front_s = float(front.get("score", 0.0))
    pc_s = float(pc.get("score", 0.0))
    liq_s = float(liq.get("score", 0.0))
    energy_s = float(energy.get("score", 0.0))
    war_s = float(war.get("score", 0.0))
    bottom_s = float(bottom.get("score", 0.0))
    ihsg_em = float(ihsg_overlay.get("em_stress", 0.0))
    maturity_s = _maturity_bias(maturity)

    raw = {
        "Current regime extends": max(0.04, 0.34 * (1 - top_s) + 0.16 * maturity_s + 0.12 * (1 - front_s) + 0.10 * (1 - pc_s) + 0.10 * (1 - liq_s) + 0.08 * (1 - trans_s) + 0.10 * max(qprobs.values())),
        "Clean Quad 1 relief": max(0.04, 0.30 * qprobs["Quad 1"] + 0.20 * trans_s + 0.14 * bottom_s + 0.10 * flags["y2_relief"] + 0.08 * flags["real_relief"] + 0.08 * flags["dxy_relief"] + 0.05 * flags["credit_relief"] + 0.05 * flags["breadth_repair"]),
        "Healthy Quad 2 reflation": max(0.04, 0.30 * qprobs["Quad 2"] + 0.14 * flags["growth_accel"] + 0.12 * flags["infl_heat"] + 0.10 * flags["breadth_repair"] + 0.08 * flags["credit_relief"] + 0.08 * (1 - flags["dxy_stress"]) + 0.08 * (1 - front_s) + 0.10 * (1 - top_s)),
        "Quad 3 defensive slowdown": max(0.04, 0.30 * qprobs["Quad 3"] + 0.14 * flags["infl_cool"] + 0.12 * flags["growth_slow"] + 0.10 * flags["y2_relief"] + 0.08 * flags["credit_stress"] + 0.08 * (1 - energy_s) + 0.08 * (1 - qprobs["Quad 2"]) + 0.10 * (1 - bottom_s)),
        "Quad 4 stagflation": max(0.04, 0.30 * qprobs["Quad 4"] + 0.14 * energy_s + 0.12 * flags["oil_pressure"] + 0.10 * flags["infl_heat"] + 0.08 * flags["dxy_stress"] + 0.08 * flags["breadth_decay"] + 0.08 * (1 - flags["y2_relief"]) + 0.10 * war_s),
        "War de-escalation clean relief": max(0.04, 0.20 * flags["war_absorb"] + 0.16 * flags["oil_relief"] + 0.14 * flags["dxy_relief"] + 0.14 * flags["y2_relief"] + 0.10 * flags["credit_relief"] + 0.10 * flags["breadth_repair"] + 0.08 * trans_s + 0.08 * bottom_s),
        "War fades, damage persists": max(0.04, 0.28 * flags["war_damage"] + 0.16 * war_s + 0.14 * flags["credit_stress"] + 0.12 * flags["breadth_decay"] + 0.10 * energy_s + 0.10 * (1 - bottom_s) + 0.10 * (1 - trans_s)),
        "Front-end funding squeeze": max(0.04, 0.34 * front_s + 0.14 * flags["dxy_stress"] + 0.12 * flags["breadth_decay"] + 0.10 * liq_s + 0.10 * pc_s + 0.10 * flags["real_stress"] + 0.10 * (1 - bottom_s)),
        "Private credit spillover": max(0.04, 0.40 * pc_s + 0.16 * front_s + 0.12 * liq_s + 0.10 * flags["breadth_decay"] + 0.10 * flags["dxy_stress"] + 0.12 * (1 - bottom_s)),
        "Liquidity / forced deleveraging": max(0.04, 0.42 * liq_s + 0.16 * front_s + 0.12 * pc_s + 0.10 * flags["breadth_decay"] + 0.10 * flags["dxy_stress"] + 0.10 * (1 - trans_s)),
        "Structural energy persistence": max(0.04, 0.30 * energy_s + 0.16 * flags["energy_structural"] + 0.12 * flags["oil_pressure"] + 0.10 * flags["infl_heat"] + 0.10 * flags["war_alive"] + 0.10 * flags["breadth_decay"] + 0.12 * (1 - flags["credit_relief"])),
        "EM dollar stress override": max(0.04, 0.34 * ihsg_em + 0.16 * flags["dxy_stress"] + 0.14 * front_s + 0.10 * pc_s + 0.10 * flags["breadth_decay"] + 0.08 * liq_s + 0.08 * (1 - bottom_s)),
    }
    total = sum(raw.values())
    probs = {k: float(v / total) for k, v in raw.items()}

    ranked = []
    for name, prob in sorted(probs.items(), key=lambda kv: kv[1], reverse=True):
        meta = WHAT_NEXT_META[name]
        ranked.append({
            "Scenario": name,
            "Probability": prob,
            "Path": WHAT_NEXT_IMPACT_MAP[name]["path"],
            "Driver": meta["driver"],
            "What confirms": meta["confirm"],
            "What breaks it": meta["break"],
            "What to do": meta["action"],
            "Impacts": _scenario_impacts_for_name(name, current_quad_name),
        })

    return {
        "rows": ranked,
        "top": ranked[0] if ranked else None,
        "raw": raw,
        "qprobs": qprobs,
        "flags": flags,
        "context": {
            "top": top,
            "next": nxt,
            "transition": trans,
            "front": front,
            "private_credit": pc,
            "liquidity": liq,
            "energy": energy,
            "war": war,
            "bottom": bottom,
        },
    }


def build_what_next_probability_table(engine: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for row in engine["rows"]:
        rows.append((row["Scenario"], f"{row['Probability']:.0%}", row["Path"], row["Driver"], row["What confirms"], row["What breaks it"], row["What to do"]))
    return pd.DataFrame(rows, columns=["Scenario", "Probability", "Path", "Driver", "What confirms", "What breaks it", "What to do"])


def build_what_next_cross_asset_table(engine: Dict[str, object], top_n: int = 6) -> pd.DataFrame:
    rows = []
    cols = ["US stocks", "Index futures", "Treasury futures", "Commodities", "Energy", "Emerging Mkts", "IHSG", "Crypto", "USD", "Gold"]
    for row in engine["rows"][:top_n]:
        out = {
            "Scenario": row["Scenario"],
            "Prob": f"{row['Probability']:.0%}",
            "Path": row["Path"],
        }
        for col in cols:
            out[col] = shock_word(float(row["Impacts"][col]))
        rows.append(out)
    return pd.DataFrame(rows)


def build_top_what_next_summary(engine: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for row in engine["rows"][:3]:
        impacts = row["Impacts"]
        winners = []
        losers = []
        for asset, score in sorted(impacts.items(), key=lambda kv: kv[1], reverse=True):
            if score >= 0.45 and len(winners) < 3:
                winners.append(asset)
        for asset, score in sorted(impacts.items(), key=lambda kv: kv[1]):
            if score <= -0.45 and len(losers) < 3:
                losers.append(asset)
        rows.append((row["Scenario"], f"{row['Probability']:.0%}", row["Path"], ", ".join(winners) if winners else "Mixed", ", ".join(losers) if losers else "None major", row["What to do"]))
    return pd.DataFrame(rows, columns=["Scenario", "Prob", "Path", "Big winners", "Main losers", "Action"])


def build_crash_meter(state: MacroState, prices: pd.DataFrame, news_items: List[Dict[str, str]], ihsg_overlay: Dict[str, object]) -> Dict[str, object]:
    front = classify_front_end_stress(state, news_items)
    pc = classify_private_credit_stress(state, prices, news_items)
    liq = classify_liquidity_event(state, prices)
    energy = classify_energy_shock(state, prices, news_items)
    war = classify_war_damage_persistence(state, prices, news_items)
    bottom = classify_bottom_quality(state, prices, news_items, next_likely_quad(state))

    dy2 = delta(state.y2_now, state.y2_prev)
    ddxy = delta(state.dxy_now, state.dxy_prev)
    dbreadth = delta(state.breadth_now, state.breadth_prev)
    dcredit = delta(state.credit_now, state.credit_prev)
    doil_pct = safe_pct_delta(state.oil_now, state.oil_prev)
    growth_delta = delta(state.growth_now, state.growth_prev)

    front_s = float(front.get("score", 0.0))
    pc_s = float(pc.get("score", 0.0))
    liq_s = float(liq.get("score", 0.0))
    energy_s = float(energy.get("score", 0.0))
    war_s = float(war.get("score", 0.0))
    bottom_s = float(bottom.get("score", 0.0))
    em_s = float(ihsg_overlay.get("em_stress", 0.0))

    deflation = max(0.05, 0.32 * liq_s + 0.16 * front_s + 0.12 * clamp01(-growth_delta / 0.55) + 0.12 * clamp01(-dbreadth / 4.0) + 0.10 * clamp01(ddxy / 2.0) + 0.10 * (1 - bottom_s) + 0.08 * clamp01(dcredit / 0.45))
    credit = max(0.05, 0.38 * pc_s + 0.18 * front_s + 0.14 * liq_s + 0.12 * clamp01(dcredit / 0.45) + 0.10 * clamp01(-dbreadth / 4.0) + 0.08 * em_s)
    growth = max(0.05, 0.24 * clamp01(-growth_delta / 0.55) + 0.18 * clamp01(-doil_pct / 8.0) + 0.16 * liq_s + 0.12 * clamp01(dcredit / 0.45) + 0.15 * clamp01(-dbreadth / 4.0) + 0.15 * (1 - bottom_s))
    inflation = max(0.05, 0.24 * energy_s + 0.20 * clamp01(doil_pct / 8.0) + 0.14 * war_s + 0.14 * clamp01(dy2 / 0.35) + 0.14 * clamp01(ddxy / 2.0) + 0.14 * clamp01(-dbreadth / 4.0))
    policy = max(0.05, 0.42 * front_s + 0.16 * clamp01(dy2 / 0.35) + 0.14 * clamp01(ddxy / 2.0) + 0.10 * liq_s + 0.10 * clamp01(-dbreadth / 4.0) + 0.08 * clamp01(dcredit / 0.45))
    geopolitical = max(0.05, 0.34 * war_s + 0.24 * energy_s + 0.14 * clamp01(doil_pct / 8.0) + 0.10 * clamp01(ddxy / 2.0) + 0.10 * clamp01(-dbreadth / 4.0) + 0.08 * em_s)
    systemic = max(0.05, 0.24 * liq_s + 0.22 * pc_s + 0.18 * front_s + 0.12 * clamp01(dcredit / 0.45) + 0.12 * clamp01(-dbreadth / 4.0) + 0.12 * clamp01(ddxy / 2.0))

    raw = {
        "Deflationary / liquidity crash": deflation,
        "Credit accident": credit,
        "Growth collapse": growth,
        "Inflation / stagflation crash": inflation,
        "Policy / rates shock": policy,
        "Geopolitical shock": geopolitical,
        "Systemic confidence shock": systemic,
    }
    total = sum(raw.values())
    family_probs = {k: float(v / total) for k, v in raw.items()}
    dominant_family = max(family_probs, key=family_probs.get)

    crash_score = 100.0 * clamp01(
        0.22 * front_s +
        0.20 * pc_s +
        0.18 * liq_s +
        0.10 * energy_s +
        0.08 * war_s +
        0.07 * em_s +
        0.07 * clamp01(dcredit / 0.45) +
        0.05 * clamp01(ddxy / 2.0) +
        0.03 * clamp01(-dbreadth / 4.0)
    )
    if crash_score >= 80:
        label = "Extreme"
    elif crash_score >= 65:
        label = "High"
    elif crash_score >= 45:
        label = "Elevated"
    elif crash_score >= 28:
        label = "Moderate"
    else:
        label = "Low"

    trigger_rows = [
        ("Front-end stress", front["label"], f"{front_s:.0%}"),
        ("Private credit", pc["label"], f"{pc_s:.0%}"),
        ("Everything-down tape", liq["label"], f"{liq_s:.0%}"),
        ("Energy / inflation shock", energy["label"], f"{energy_s:.0%}"),
        ("War persistence", war["label"], f"{war_s:.0%}"),
        ("Bottom quality", bottom["label"], f"{bottom_s:.0%}"),
        ("DXY pressure", shock_word(-clamp01(-ddxy / 2.0) + clamp01(ddxy / 2.0)), f"{clamp01(ddxy / 2.0):.0%}"),
        ("Breadth decay", "Narrowing" if dbreadth < 0 else "Not worsening", f"{clamp01(-dbreadth / 4.0):.0%}"),
    ]

    family_rows = [(name, f"{prob:.0%}") for name, prob in sorted(family_probs.items(), key=lambda kv: kv[1], reverse=True)]
    return {
        "score": crash_score,
        "label": label,
        "family": dominant_family,
        "family_prob": family_probs[dominant_family],
        "families": family_probs,
        "family_df": pd.DataFrame(family_rows, columns=["Crash family", "Probability"]),
        "trigger_df": pd.DataFrame(trigger_rows, columns=["Trigger", "State", "Weight"]),
        "read": f"Crash meter {crash_score:.0f}/100. Dominant family: {dominant_family} ({family_probs[dominant_family]:.0%}). Ini bukan prediksi crash pasti, tapi pengukur seberapa disorderly tape bisa berubah jika trigger berikutnya ikut confirm.",
    }

def build_action_map(state: MacroState, prices: pd.DataFrame, current_quad: str, next_quad: str, news_items: List[Dict[str, str]], ihsg_overlay: Dict[str, object]) -> pd.DataFrame:
    top = current_top_diagnosis(state, current_quad, news_items)
    nxt = next_phase_activation(state, next_quad, news_items)
    trans = transition_quality_score(state, news_items)
    front = classify_front_end_stress(state, news_items)
    pc = classify_private_credit_stress(state, prices, news_items)
    liq = classify_liquidity_event(state, prices)
    bottom = classify_bottom_quality(state, prices, news_items, next_quad)
    energy = classify_energy_shock(state, prices, news_items)
    gold = classify_gold_mode(state, prices, news_items)

    top_s = float(top.get('score', 0.0))
    nxt_s = float(nxt.get('score', 0.0))
    trans_s = float(trans.get('score', 0.0))
    front_s = float(front.get('score', 0.0))
    pc_s = float(pc.get('score', 0.0))
    liq_s = float(liq.get('score', 0.0))
    bottom_s = float(bottom.get('score', 0.0))
    em_override = float(ihsg_overlay.get('em_stress', 0.0))

    if liq_s >= 0.70 or pc_s >= 0.75 or front_s >= 0.75:
        stance = 'Reduce gross / selective only'
        chase = 'Jangan kejar broad beta. Kalau pun ambil, pilih quality atau sleeve yang paling bersih.'
        avoid = 'Junk beta, weak followers, false-bottom names, broad EM tanpa konfirmasi.'
        hedge = 'Cash / short duration / defensive quality. Gold hanya hedge kalau mode-nya bukan liquidity squeeze.'
    elif top_s >= 0.70 and bottom_s < 0.55:
        stance = 'Late current / trim chasing'
        chase = f'Current winners yang masih paling bersih di {current_quad}, tapi size lebih kecil.'
        avoid = 'Nama yang sudah crowded dan follower yang tertinggal.'
        hedge = 'Hedge via duration / quality defensives jika front-end mulai relief.'
    elif nxt_s >= 0.65 and trans_s >= 0.60 and bottom_s >= 0.60:
        stance = 'Transition building / rotate gradually'
        chase = f'Leader awal untuk {next_quad}; tambah bertahap saat breadth dan credit confirm.'
        avoid = 'Old winners yang sudah kehilangan konfirmasi kedua dan ketiga.'
        hedge = 'Sizing discipline lebih penting daripada hedge berat; tambah saat pullback ditahan.'
    else:
        stance = 'Wait for confirmation'
        chase = 'Selective only: fokus ke leader paling bersih, bukan seluruh tape.'
        avoid = 'Overtrading di tengah signal campur-aduk.'
        hedge = 'Pertahankan fleksibilitas dan tunggu trigger berikutnya.'

    if em_override >= 0.65:
        ihsg_read = 'IHSG broad jangan dibaca copy US. Fokus selective commodity atau bank quality, bukan broad confirm.'
    else:
        ihsg_read = 'IHSG bisa ikut lebih broad kalau DXY / USDIDR / EM relief ikut confirm.'

    if energy['label'] in ['Structural energy persistence', 'Mixed energy stress']:
        extra = 'Energy shock belum tentu selesai walau headline perang reda; bedakan crude premium vs scarcity theme.'
    else:
        extra = 'Energy premium lebih mudah pudar kalau headline reda dan chain tidak confirm.'

    if gold['label'] == 'Liquidity squeeze / forced selling':
        gold_read = 'Gold lagi tidak bersih sebagai hedge tunggal karena likuidasi / real-yield pressure.'
    else:
        gold_read = f'Gold mode: {gold["label"]}. Pakai sebagai filter, bukan asumsi otomatis.'

    rows = [
        ('Core stance', stance),
        ('What to chase now', chase),
        ('What to avoid now', avoid),
        ('What needs confirmation first', '2Y relief, credit stabilization, breadth repair, dan leader next phase tahan pullback.'),
        ('Best hedge / defense', hedge),
        ('IHSG read', ihsg_read),
        ('Extra macro note', extra),
        ('Gold note', gold_read),
        ('What invalidates this stance', 'Kalau breadth mendadak membaik, credit ikut stabil, dan 2Y/front-end benar-benar relief, stance defensif perlu dilonggarkan. Sebaliknya kalau private credit + everything-down makin lebar, stance harus makin ketat.'),
    ]
    return pd.DataFrame(rows, columns=['Action', 'Read'])


# --------------------------
# Sidebar
# --------------------------
with st.sidebar:
    st.header("Data")
    fred_key = st.text_input("FRED API Key", type="password", value=st.session_state.get("fred_api_key", ""))
    if fred_key:
        st.session_state["fred_api_key"] = fred_key
    start = st.text_input("Observation start", value=DEFAULT_START)
    custom_query = st.text_input("News query", value=DEFAULT_NEWS_QUERY)
    refresh = st.button("Refresh data", use_container_width=True)
    if refresh:
        fred_single.clear()
        fetch_news_rss.clear()

    st.divider()
    st.header("Main toggles")
    show_board = st.toggle("Show Regime Decision Board", value=True)
    show_crash = st.toggle("Show Crash / Black Swan Engine", value=False)
    show_appendix = st.toggle("Show Appendix", value=False)

    st.divider()
    st.header("Display mode")
    show_detail_tables = st.toggle("Show detailed tables", value=True)
    show_advanced = st.toggle("Show advanced diagnostics", value=True)
    show_raw = st.toggle("Show raw state in appendix", value=False)
    show_full_curves = st.toggle("Show full curves inside board", value=False)

st.title("Macro Scenario Matrix — Decision Engine Final V7")
st.caption(
    "Versi ini nge-merge blok yang mirip jadi satu board yang lebih rapat: summary -> current vs next core -> context/stress -> scenario probability -> action map -> advanced detail. Di luar board cuma crash engine dan appendix."
)

api_key = st.session_state.get("fred_api_key", "")
if not api_key:
    st.info("Masukkan FRED API key di sidebar.")
    st.stop()

try:
    with st.spinner("Loading live macro data..."):
        state, comp, dfs, ids, errs = build_live_macro_state(api_key, start)
except Exception as e:
    st.error(f"Gagal build macro state: {e}")
    st.stop()

try:
    news_items = fetch_news_rss(custom_query, max_items=6)
except Exception:
    news_items = []

quad = current_quad(state)
next_quad = next_likely_quad(state)
conf = phase_confidence(next_quad_probabilities(state))

maturity, maturity_score, current_confirms, current_failures = current_phase_maturity(state, quad, news_items)
top_diag = current_top_diagnosis(state, quad, news_items)
next_diag = next_phase_activation(state, next_quad, news_items)
transition_diag = transition_quality_score(state, news_items)
treasury_label, treasury_read = treasury_regime_text(state)
premium_label, premium_read = news_premium_status(state, news_items)
exec_df = build_executive_summary_table(state, quad, next_quad, news_items)
market_bundle = fetch_market_bundle()
market_prices = build_market_price_frame(market_bundle)
current_compare_df = build_compare_board_table_v2(state, market_prices, quad, next_quad, news_items, mode="current")
next_compare_df = build_compare_board_table_v2(state, market_prices, quad, next_quad, news_items, mode="next")
current_signal_df = build_phase_signal_story_table(state, quad, next_quad, news_items, mode="current")
next_signal_df = build_phase_signal_story_table(state, quad, next_quad, news_items, mode="next")
current_seq_df = build_top_sequence_table(state, quad, news_items)
next_seq_df = build_bottom_sequence_table(state, next_quad, news_items)
current_stack_df = build_integrated_stack_table(state, quad, next_quad, news_items, mode="current")
next_stack_df = build_integrated_stack_table(state, quad, next_quad, news_items, mode="next")
transition_table_df = build_transition_vertical_table(state, news_items)
curve_df_v2 = build_curve_table_v2(state, news_items)
structural_stress_df = build_structural_stress_table(state, market_prices, news_items, next_quad)
energy_diag = classify_energy_shock(state, market_prices, news_items)
war_diag = classify_war_damage_persistence(state, market_prices, news_items)
front_diag = classify_front_end_stress(state, news_items)
private_credit_diag = classify_private_credit_stress(state, market_prices, news_items)
gold_diag = classify_gold_mode(state, market_prices, news_items)
liquidity_diag = classify_liquidity_event(state, market_prices)
bottom_quality_diag = classify_bottom_quality(state, market_prices, news_items, next_quad)
energy_theme_df = build_energy_theme_split_table(market_prices)
private_credit_snapshot_df = build_private_credit_snapshot(market_prices)
liquidation_snapshot_df = build_liquidation_snapshot(market_prices, state)
ihsg_overlay = build_ihsg_overlay(state, market_prices, quad, next_quad)
ihsg_filter_df = build_ihsg_filter_table(state, ihsg_overlay, quad, next_quad)
ihsg_corr_df = build_ihsg_correlation_table(market_prices)
ihsg_scenario_df = build_ihsg_scenario_table(state, ihsg_overlay)
ihsg_order_df = build_ihsg_order_table(ihsg_overlay)
ihsg_snapshot_df = build_ihsg_snapshot_table(ihsg_overlay)
ihsg_fit_matrix_df = build_ihsg_quad_fit_matrix(ihsg_overlay)
scenario_engine = build_what_next_scenario_engine(state, market_prices, quad, next_quad, news_items, ihsg_overlay)
what_next_prob_df = build_what_next_probability_table(scenario_engine)
what_next_cross_asset_df = build_what_next_cross_asset_table(scenario_engine, top_n=6)
what_next_summary_df = build_top_what_next_summary(scenario_engine)
crash_meter = build_crash_meter(state, market_prices, news_items, ihsg_overlay)

if show_board:
    current_core_df = merge_field_tables(current_compare_df, current_stack_df)
    next_core_df = merge_field_tables(next_compare_df, next_stack_df)
    context_board_df = build_context_board_table(state, market_prices, news_items, next_quad, ihsg_overlay)
    scenario_prob_df = what_next_prob_df.copy()
    action_map_df = build_action_map(state, market_prices, quad, next_quad, news_items, ihsg_overlay)

    with st.container(border=True):
        st.subheader("1. Regime Decision Board")

        r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
        with r1c1:
            compact_kpi("Current Quad", quad, QUAD_THEMES[quad]["theme"])
        with r1c2:
            compact_kpi("Current maturity", maturity, "Umur / crowding current phase.")
        with r1c3:
            compact_kpi("Current top risk", top_diag["label"], "Apakah winners sekarang mulai capek / crowded.")
        with r1c4:
            compact_kpi("Next likely", next_quad, QUAD_THEMES[next_quad]["theme"])
        with r1c5:
            compact_kpi("Next activation", next_diag["label"], "Apakah next phase baru ide, starting, atau sudah confirm.")

        r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns(5)
        with r2c1:
            compact_kpi("Transition", transition_diag["label"], "Clean / fragile / false risk.")
        with r2c2:
            compact_kpi("Treasury", treasury_label, treasury_read)
        with r2c3:
            compact_kpi("Bottom quality", bottom_quality_diag["label"], bottom_quality_diag["read"])
        with r2c4:
            compact_kpi("IHSG filter", str(ihsg_overlay["dominant"]), str(ihsg_overlay["headline"]))
        with r2c5:
            action_headline = action_map_df.iloc[0]["Read"] if not action_map_df.empty else "Wait for confirmation"
            compact_kpi("Action state", action_headline, "Stance ringkas yang paling sesuai sekarang.")

        r3c1, r3c2, r3c3 = st.columns(3)
        with r3c1:
            compact_kpi("Crash meter", f"{crash_meter['score']:.0f}/100", crash_meter['read'])
        with r3c2:
            compact_kpi("Crash family", str(crash_meter['family']), f"Dominant share {crash_meter['family_prob']:.0%}")
        with r3c3:
            top_scenario = scenario_engine['top']['Scenario'] if scenario_engine.get('top') else '-'
            top_prob = scenario_engine['top']['Probability'] if scenario_engine.get('top') else 0.0
            compact_kpi("Top what-next", top_scenario, f"Probability {top_prob:.0%}")

        st.write(
            f"**Current winners:** {QUAD_THEMES[quad]['current_winners']}. "
            f"**Current top risk:** {top_diag['score']:.0%}. "
            f"**Next activation:** {next_diag['score']:.0%}. "
            f"**Transition quality:** {transition_diag['score']:.0%}. "
            f"**Top what-next:** {scenario_engine['top']['Scenario']} ({scenario_engine['top']['Probability']:.0%}) -> {scenario_engine['top']['Path']}. "
            f"**Crash meter:** {crash_meter['score']:.0f}/100, dominant family {crash_meter['family']} ({crash_meter['family_prob']:.0%}). "
            f"**Treasury regime:** {treasury_label}. {treasury_read} "
            f"**Dominant premium:** {premium_label}. {premium_read} "
            f"**Energy mode:** {energy_diag['label']}. {energy_diag['read']} "
            f"**Front-end mode:** {front_diag['label']}. {front_diag['read']} "
            f"**Bottom quality:** {bottom_quality_diag['label']}. {bottom_quality_diag['read']} "
            f"**IHSG read:** {ihsg_overlay['dominant']}. {ihsg_overlay['headline']}"
        )

        if show_detail_tables:
            with st.expander("Merged header summary", expanded=False):
                st.dataframe(exec_df, use_container_width=True, hide_index=True)

        st.markdown("### Current vs Next — merged core")
        st.write(
            "Blok yang isinya mirip sudah digabung. Jadi current/next sekarang langsung berisi compare core + strongest proxies + chain + correlation focus + cross-asset read, tanpa pecah ke section lain yang mengulang konteks yang sama."
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Current side — {quad}**")
            st.dataframe(current_core_df, use_container_width=True, hide_index=True)
            st.markdown("**Ordered top sequence**")
            st.dataframe(current_seq_df[["Step", "Sequence", "Now"]], use_container_width=True, hide_index=True)
            if show_detail_tables:
                with st.expander("Current signal / topping detail", expanded=False):
                    st.write(top_diag["interpretation"])
                    st.markdown("**What starts the top**")
                    render_list(top_diag["starting"])
                    st.markdown("**What confirms the top**")
                    render_list(top_diag.get("confirm", []))
                    st.markdown("**What invalidates the top**")
                    render_list(top_diag.get("invalidate", []))
                    st.markdown("**False-top traps**")
                    render_list(top_diag["traps"])
                    st.dataframe(current_signal_df, use_container_width=True, hide_index=True)
        with c2:
            st.markdown(f"**Next side — {next_quad}**")
            st.dataframe(next_core_df, use_container_width=True, hide_index=True)
            st.markdown("**Ordered bottom sequence**")
            st.dataframe(next_seq_df[["Step", "Sequence", "Now"]], use_container_width=True, hide_index=True)
            if show_detail_tables:
                with st.expander("Next signal / bottoming detail", expanded=False):
                    st.write(next_diag["interpretation"])
                    st.markdown("**What starts the bottom / activation**")
                    render_list(next_diag["starting"])
                    st.markdown("**Early stabilization**")
                    render_list(next_diag["stabilization"])
                    st.markdown("**What confirms the bottom**")
                    render_list(next_diag.get("confirmation", []))
                    st.markdown("**False-bottom traps**")
                    render_list(next_diag["traps"])
                    st.dataframe(next_signal_df, use_container_width=True, hide_index=True)

        st.markdown("### Context / stress / probability / action")
        x1, x2 = st.columns([1.1, 1.0])
        with x1:
            st.markdown("**Context strip**")
            st.dataframe(context_board_df, use_container_width=True, hide_index=True)
            if show_detail_tables:
                with st.expander("Transition / IHSG context detail", expanded=False):
                    st.dataframe(transition_table_df, use_container_width=True, hide_index=True)
                    st.markdown("**IHSG filter**")
                    st.dataframe(ihsg_filter_df, use_container_width=True, hide_index=True)
                    st.markdown("**IHSG fit by quad**")
                    st.dataframe(ihsg_fit_matrix_df, use_container_width=True, hide_index=True)
                    st.markdown("**Read order paling penting dulu**")
                    st.dataframe(ihsg_order_df, use_container_width=True, hide_index=True)
        with x2:
            st.markdown("**Scenario probability board**")
            st.dataframe(scenario_prob_df, use_container_width=True, hide_index=True)
            if show_detail_tables:
                with st.expander("What-next scenario tree / cross-asset map", expanded=False):
                    st.markdown("**Top 3 scenario summary**")
                    st.dataframe(what_next_summary_df, use_container_width=True, hide_index=True)
                    st.markdown("**Cross-asset impact map**")
                    st.dataframe(what_next_cross_asset_df, use_container_width=True, hide_index=True)
            st.markdown("**Action map**")
            st.dataframe(action_map_df, use_container_width=True, hide_index=True)

        if show_detail_tables:
            with st.expander("Advanced market detail", expanded=False):
                st.markdown("**Key numbers / curves**")
                st.dataframe(curve_df_v2, use_container_width=True, hide_index=True)
                if show_full_curves:
                    chart_df = comp.tail(180).set_index("date")[["growth_composite", "inflation_composite", "liquidity_composite", "credit_composite"]]
                    st.line_chart(chart_df)
                st.markdown("**Empirical IHSG correlation**")
                st.dataframe(ihsg_corr_df, use_container_width=True, hide_index=True)
                st.markdown("**IHSG live scenario map**")
                st.dataframe(ihsg_scenario_df, use_container_width=True, hide_index=True)
                st.markdown("**IHSG live market snapshot**")
                st.dataframe(ihsg_snapshot_df, use_container_width=True, hide_index=True)
                st.markdown("**IHSG notes / traps**")
                render_list(list(ihsg_overlay["notes"]))
                st.markdown("**Current proxy group detail**")
                st.dataframe(build_proxy_df(best_proxy_group_for_quad(quad)), use_container_width=True, hide_index=True)
                st.markdown("**Next proxy group detail**")
                st.dataframe(build_proxy_df(best_proxy_group_for_quad(next_quad)), use_container_width=True, hide_index=True)
                st.markdown("**Divergence / chain failure cases**")
                st.dataframe(build_divergence_df(), use_container_width=True, hide_index=True)

            with st.expander("Narrative / shock detail", expanded=False):
                st.markdown("**Dominant news read**")
                st.write(economy_news_positioning_text(state, news_items))
                st.markdown("**Structural stress stack**")
                st.dataframe(structural_stress_df, use_container_width=True, hide_index=True)
                st.markdown("**Energy theme split / scarcity map**")
                st.dataframe(energy_theme_df, use_container_width=True, hide_index=True)
                st.markdown("**Private credit / funding snapshot**")
                st.dataframe(private_credit_snapshot_df, use_container_width=True, hide_index=True)
                st.write(private_credit_diag["read"])
                st.markdown("**Everything-down / liquidation snapshot**")
                st.dataframe(liquidation_snapshot_df, use_container_width=True, hide_index=True)
                st.write(liquidity_diag["read"])
                if news_items:
                    st.markdown("**Headline classification**")
                    for item in news_items[:6]:
                        case, read, action = infer_news_case(item["title"], item["desc"])
                        with st.container(border=True):
                            st.markdown(f"**{item['title']}**")
                            st.caption(item["published"])
                            st.write(f"**Shock type:** {case}")
                            st.write(read)
                            st.write(f"**Implication:** {action}")
                            st.write(item["link"])
                else:
                    st.info("No news fetched right now.")

        if show_advanced:
            with st.expander("Model internals / explicit diagnostics", expanded=False):
                st.markdown("**Macro scenario matrix**")
                st.dataframe(build_scenario_df(), use_container_width=True, hide_index=True)
                st.markdown("**Explicit topping diagnostics**")
                st.dataframe(build_topping_signal_table(state), use_container_width=True, hide_index=True)
                st.markdown("**Explicit bottoming diagnostics**")
                st.dataframe(build_bottoming_signal_table(state), use_container_width=True, hide_index=True)
                st.markdown("**False transition map**")
                st.dataframe(build_false_transition_table(), use_container_width=True, hide_index=True)
                st.markdown("**Energy shock detail**")
                st.dataframe(pd.DataFrame(energy_diag["rows"], columns=["Field", "Value", "Read"]), use_container_width=True, hide_index=True)
                st.markdown("**Private credit detail**")
                st.dataframe(pd.DataFrame(private_credit_diag["rows"], columns=["Field", "Value", "Read"]), use_container_width=True, hide_index=True)


if show_crash:
    with st.container(border=True):
        st.subheader("2. Crash / Black Swan Engine")
        st.write(
            "Bagian ini sengaja tetap di luar Regime Decision Board karena ini tail-risk mode, bukan pembacaan base case harian. Pakai ini untuk bedain normal transition, forced deleveraging, dan black swan stress."
        )
        k1, k2, k3 = st.columns(3)
        with k1:
            compact_kpi("Crash meter", f"{crash_meter['score']:.0f}/100", crash_meter['read'])
        with k2:
            compact_kpi("Risk label", crash_meter['label'], "Low / moderate / elevated / high / extreme")
        with k3:
            compact_kpi("Dominant family", str(crash_meter['family']), f"Share {crash_meter['family_prob']:.0%}")
        st.write(crash_meter['read'])
        st.write("Early signs: breadth sempit, credit rusak, dollar dominan, 2Y/front-end stress naik, proxy chain putus.")
        st.write("Mid signs: broad beta menyerah, EM/crypto/small caps rapuh, forced deleveraging terasa.")
        st.write("Late signs: correlation naik ke satu, forced selling / capitulation, lalu market fokus ke survival dan policy response.")
        if show_detail_tables:
            c0, c1, c2 = st.columns([1.05, 1.0, 1.0])
            with c0:
                st.markdown("**Crash family probabilities**")
                st.dataframe(crash_meter['family_df'], use_container_width=True, hide_index=True)
                st.markdown("**Crash trigger stack**")
                st.dataframe(crash_meter['trigger_df'], use_container_width=True, hide_index=True)
            with c1:
                st.dataframe(build_crash_types_df(), use_container_width=True, hide_index=True)
            with c2:
                st.dataframe(build_crash_recovery_df(), use_container_width=True, hide_index=True)

if show_appendix:
    with st.container(border=True):
        st.subheader("3. Appendix")
        with st.expander("Terms", expanded=False):
            st.dataframe(build_terms_df(), use_container_width=True, hide_index=True)
        with st.expander("Regime switch archetypes", expanded=False):
            st.dataframe(build_regime_switch_df(), use_container_width=True, hide_index=True)
        if show_raw:
            with st.expander("Raw state", expanded=False):
                st.json(asdict(state))
