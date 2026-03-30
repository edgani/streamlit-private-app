import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import ccxt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf


# =========================================================
# APP CONFIG
# =========================================================
st.set_page_config(
    page_title="Quant Story Engine - Python",
    page_icon="📊",
    layout="wide",
)

EPS = 1e-9
SEC_HEADERS = {"User-Agent": "EdwardGani Quant Story Engine/1.0 email@example.com"}

INTERVAL_MAP = {
    "Daily": {"yf_interval": "1d", "yf_period": "5y", "ccxt_tf": "1d", "resample": None},
    "4H": {"yf_interval": "1h", "yf_period": "730d", "ccxt_tf": "1h", "resample": "4H"},
    "1H": {"yf_interval": "1h", "yf_period": "730d", "ccxt_tf": "1h", "resample": None},
}

FOREX_LIBRARY = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "USDCHF": "CHF=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "USDCAD": "CAD=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    "EURGBP": "EURGBP=X",
    "EURCHF": "EURCHF=X",
    "AUDJPY": "AUDJPY=X",
    "CHFJPY": "CHFJPY=X",
    "GBPCHF": "GBPCHF=X",
    "EURAUD": "EURAUD=X",
    "AUDCAD": "AUDCAD=X",
    "AUDCHF": "AUDCHF=X",
    "AUDNZD": "AUDNZD=X",
    "CADJPY": "CADJPY=X",
    "EURCAD": "EURCAD=X",
    "EURNZD": "EURNZD=X",
    "GBPAUD": "GBPAUD=X",
    "GBPCAD": "GBPCAD=X",
    "GBPNZD": "GBPNZD=X",
    "NZDCAD": "NZDCAD=X",
    "NZDJPY": "NZDJPY=X",
    "NZDCHF": "NZDCHF=X",
    "USDIDR": "USDIDR=X",
    "USDINR": "USDINR=X",
    "USDSGD": "USDSGD=X",
    "USDHKD": "HKD=X",
    "USDCNH": "CNH=X",
    "USDMYR": "USDMYR=X",
    "USDPHP": "USDPHP=X",
    "USDTHB": "USDTHB=X",
    "USDTWD": "USDTWD=X",
    "USDKRW": "USDKRW=X",
    "USDZAR": "USDZAR=X",
    "USDMXN": "MXN=X",
    "USDBRL": "USDBRL=X",
    "USDTRY": "USDTRY=X",
}

FUTURES_LIBRARY = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Crude Oil WTI": "CL=F",
    "Brent Oil": "BZ=F",
    "Natural Gas": "NG=F",
    "Copper": "HG=F",
    "Platinum": "PL=F",
    "Palladium": "PA=F",
    "Corn": "ZC=F",
    "Soybeans": "ZS=F",
    "Wheat": "ZW=F",
    "Coffee": "KC=F",
    "Cotton": "CT=F",
    "Sugar": "SB=F",
    "Lean Hogs": "HE=F",
    "Live Cattle": "LE=F",
    "S&P 500 E-mini": "ES=F",
    "Nasdaq 100 E-mini": "NQ=F",
    "Dow E-mini": "YM=F",
    "Russell 2000 E-mini": "RTY=F",
    "10Y Treasury Note": "ZN=F",
    "US Dollar Index": "DX=F",
    "Cocoa": "CC=F",
    "Orange Juice": "OJ=F",
}

POPULAR_US = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "JPM", "PLTR"]
POPULAR_IHSG = ["BBCA.JK", "BBRI.JK", "BMRI.JK", "ASII.JK", "TLKM.JK", "ICBP.JK", "UNVR.JK", "ADRO.JK"]
POPULAR_CRYPTO = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT"]


# =========================================================
# TEXT HELPERS
# =========================================================
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(np.clip(x, -8, 8))))


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def id_label(score: float, breaks: List[Tuple[float, str]]) -> str:
    for level, label in breaks:
        if score >= level:
            return label
    return breaks[-1][1]


def normalize_series(s: pd.Series) -> pd.Series:
    std = float(s.std())
    if std < EPS:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def safe_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


# =========================================================
# UNIVERSE LOADERS
# =========================================================
@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_us_universe() -> pd.DataFrame:
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = []
    for _, item in data.items():
        ticker = str(item.get("ticker", "")).strip().upper()
        title = str(item.get("title", "")).strip()
        if ticker:
            rows.append({"symbol": ticker, "name": title})
    df = pd.DataFrame(rows).drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_ihsg_universe() -> pd.DataFrame:
    url = "https://www.idx.co.id/en/market-data/stocks-data/stock-list"
    tables = pd.read_html(url)
    if not tables:
        raise ValueError("Tidak bisa mengambil daftar saham IHSG dari IDX.")

    best = None
    for tbl in tables:
        cols = [str(c).strip().lower() for c in tbl.columns]
        if any("code" in c for c in cols) or any("kode" in c for c in cols):
            best = tbl.copy()
            break
    if best is None:
        best = tables[0].copy()

    best.columns = [str(c).strip() for c in best.columns]
    code_col = next((c for c in best.columns if "Code" in c or "Kode" in c), best.columns[0])
    name_col = next((c for c in best.columns if "Company" in c or "Perusahaan" in c or "Nama" in c), best.columns[1])

    df = best[[code_col, name_col]].copy()
    df.columns = ["code", "name"]
    df["code"] = df["code"].astype(str).str.strip().str.upper()
    df["symbol"] = df["code"] + ".JK"
    df["name"] = df["name"].astype(str).str.strip()
    df = df[df["code"].str.len().between(3, 5)].drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
    return df[["symbol", "name"]]


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def load_crypto_universe() -> pd.DataFrame:
    ex = ccxt.binance({"enableRateLimit": True})
    markets = ex.load_markets()
    rows = []
    for symbol, meta in markets.items():
        if meta.get("spot") and meta.get("active"):
            base = meta.get("base", "")
            quote = meta.get("quote", "")
            rows.append({
                "symbol": symbol,
                "name": f"{base}/{quote}",
                "quote": quote,
            })
    df = pd.DataFrame(rows).drop_duplicates("symbol")
    if not df.empty:
        quote_order = pd.Categorical(df["quote"], categories=["USDT", "USDC", "BTC", "ETH"], ordered=True)
        df = df.assign(_quote_order=quote_order).sort_values(["_quote_order", "symbol"]).drop(columns=["_quote_order"])
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_forex_universe() -> pd.DataFrame:
    rows = [{"symbol": v, "name": k} for k, v in FOREX_LIBRARY.items()]
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_futures_universe() -> pd.DataFrame:
    rows = [{"symbol": v, "name": k} for k, v in FUTURES_LIBRARY.items()]
    return pd.DataFrame(rows)


# =========================================================
# DATA FETCHERS
# =========================================================
def _resample_ohlcv(df: pd.DataFrame, rule: Optional[str]) -> pd.DataFrame:
    if rule is None or df.empty:
        return df
    ohlc = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    out = df.resample(rule).agg(ohlc).dropna()
    return out


@st.cache_data(show_spinner=False, ttl=900)
def fetch_yf_history(symbol: str, tf_label: str) -> pd.DataFrame:
    cfg = INTERVAL_MAP[tf_label]
    df = yf.download(
        symbol,
        interval=cfg["yf_interval"],
        period=cfg["yf_period"],
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns=str.title)
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols].copy()
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df.index = pd.to_datetime(df.index)
    df = _resample_ohlcv(df, cfg["resample"])
    return df.dropna()


@st.cache_data(show_spinner=False, ttl=900)
def fetch_crypto_history(symbol: str, tf_label: str) -> pd.DataFrame:
    cfg = INTERVAL_MAP[tf_label]
    ex = ccxt.binance({"enableRateLimit": True})
    ex.load_markets()
    if symbol not in ex.symbols:
        raise ValueError(f"Symbol crypto {symbol} tidak ada di Binance spot.")

    timeframe = cfg["ccxt_tf"]
    limit = 1000
    rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not rows:
        raise ValueError("Data crypto kosong.")
    df = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("ts")
    df = _resample_ohlcv(df, cfg["resample"])
    return df.dropna()


@st.cache_data(show_spinner=False, ttl=900)
def fetch_history(category: str, symbol: str, tf_label: str) -> pd.DataFrame:
    if category == "Crypto":
        return fetch_crypto_history(symbol, tf_label)
    return fetch_yf_history(symbol, tf_label)


# =========================================================
# FEATURE ENGINEERING
# =========================================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    tp = (data["High"] + data["Low"] + data["Close"]) / 3.0
    data["VWAP"] = (tp * data["Volume"].replace(0, np.nan)).cumsum() / data["Volume"].replace(0, np.nan).cumsum()
    data["VWAP"] = data["VWAP"].fillna(data["Close"])

    for n in [20, 50, 100]:
        data[f"EMA{n}"] = data["Close"].ewm(span=n, adjust=False).mean()

    prev_close = data["Close"].shift(1)
    tr = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - prev_close).abs(),
            (data["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["ATR14"] = tr.rolling(14).mean()
    data["Range"] = data["High"] - data["Low"]
    data["RangeAvg20"] = data["Range"].rolling(20).mean()

    data["Ret"] = np.log(data["Close"] / data["Close"].shift(1))
    data["RetAbs"] = data["Close"].diff().abs()
    data["VolFast"] = data["Ret"].rolling(5).std()
    data["VolSlow"] = data["Ret"].rolling(20).std()
    data["VolRatio"] = data["VolFast"] / (data["VolSlow"] + EPS)

    data["CloseLoc"] = ((data["Close"] - data["Low"]) - (data["High"] - data["Close"])) / (data["High"] - data["Low"] + EPS)
    data["MoneyPressure"] = (data["Volume"] * data["CloseLoc"]).ewm(span=5, adjust=False).mean()

    data["Support20"] = data["Low"].rolling(20).min()
    data["Resistance20"] = data["High"].rolling(20).max()
    data["Support50"] = data["Low"].rolling(50).min()
    data["Resistance50"] = data["High"].rolling(50).max()

    mean_price = (data["EMA20"] + data["EMA50"] + data["EMA100"] + data["VWAP"]) / 4.0
    data["MeanPrice"] = mean_price
    data["DistSignedATR"] = (data["Close"] - mean_price) / (data["ATR14"] + EPS)
    data["DistMeanATR"] = data["DistSignedATR"].abs()

    data["SlopeFast"] = (data["EMA20"] - data["EMA20"].shift(5)) / (data["ATR14"] + EPS)
    data["SlopeMid"] = (data["EMA50"] - data["EMA50"].shift(5)) / (data["ATR14"] + EPS)
    data["LinSlope"] = (data["Close"].rolling(20).mean() - data["Close"].rolling(20).mean().shift(1)) / (data["ATR14"] + EPS)

    z_slope_fast = normalize_series(data["SlopeFast"].fillna(0.0))
    z_slope_mid = normalize_series(data["SlopeMid"].fillna(0.0))
    z_lin = normalize_series(data["LinSlope"].fillna(0.0))
    stack_fast = np.where(data["EMA20"] > data["EMA50"], 1.0, -1.0)
    stack_slow = np.where(data["EMA50"] > data["EMA100"], 1.0, -1.0)
    data["TrendRaw"] = 0.25 * z_slope_fast + 0.25 * z_slope_mid + 0.20 * z_lin + 0.15 * stack_fast + 0.15 * stack_slow
    data["TrendDir"] = np.where(data["TrendRaw"] > 0.15, 1, np.where(data["TrendRaw"] < -0.15, -1, 0))

    trend_flip = (data["TrendDir"] != data["TrendDir"].shift(1)).astype(float)
    data["Instability"] = trend_flip.rolling(12).mean().fillna(0.0)

    data["VolumeZ"] = normalize_series(np.log1p(data["Volume"].replace(0, np.nan).fillna(0.0)))
    data["EffortZ"] = normalize_series(data["MoneyPressure"].fillna(0.0))
    data["RangeZ"] = normalize_series(data["Range"] / (data["RangeAvg20"] + EPS))
    data["Compression"] = np.where(data["Range"] < data["RangeAvg20"] * 0.80, 1.0, 0.0)
    data["Expansion"] = np.where(data["Range"] > data["RangeAvg20"] * 1.20, 1.0, 0.0)

    mean_roll = data["Close"].rolling(20).mean()
    std_roll = data["Close"].rolling(20).std()
    data["MRZ"] = (data["Close"] - mean_roll) / (std_roll + EPS)

    data["BreakoutUp"] = (data["Close"] > data["High"].shift(1).rolling(10).max()).astype(float)
    data["BreakoutDn"] = (data["Close"] < data["Low"].shift(1).rolling(10).min()).astype(float)

    trend_score = 1 / (1 + np.exp(-data["TrendRaw"].fillna(0.0)))
    mr_pressure = np.tanh(data["MRZ"].abs().fillna(0.0) / 2.0)
    breakout_conflict = (data["BreakoutUp"] * (data["TrendDir"] < 0).astype(float)) + (data["BreakoutDn"] * (data["TrendDir"] > 0).astype(float))
    data["Conflict"] = np.clip(
        0.45 * (1 - (2 * np.abs(trend_score - 0.5))) +
        0.35 * mr_pressure +
        0.20 * breakout_conflict,
        0.0,
        1.0,
    )

    data = data.replace([np.inf, -np.inf], np.nan).dropna().copy()
    return data


# =========================================================
# ANALOG + TRIPLE BARRIER MODEL
# =========================================================
def _first_barrier_hit(window: pd.DataFrame, upper: float, lower: float) -> Tuple[int, bool]:
    hit_up = False
    hit_down = False
    for _, row in window.iterrows():
        up_now = row["High"] >= upper
        dn_now = row["Low"] <= lower
        if up_now and dn_now:
            return 0, True
        if up_now:
            hit_up = True
            return 1, hit_down
        if dn_now:
            hit_down = True
            return -1, hit_up
    return 0, hit_up and hit_down


def nearest_analog_model(data: pd.DataFrame, horizon: int, k_neighbors: int, target_atr: float, stop_atr: float) -> Dict:
    feature_cols = [
        "TrendRaw",
        "DistSignedATR",
        "DistMeanATR",
        "VolRatio",
        "VolumeZ",
        "EffortZ",
        "RangeZ",
        "MRZ",
        "Instability",
        "Conflict",
        "CloseLoc",
        "BreakoutUp",
        "BreakoutDn",
        "Compression",
        "Expansion",
    ]

    work = data.copy().dropna(subset=feature_cols + ["ATR14", "Close"])
    if len(work) < max(120, horizon + 30):
        raise ValueError("Data belum cukup panjang untuk analog model. Coba timeframe Daily atau history lebih panjang.")

    hist = work.iloc[:-horizon].copy()
    current = work.iloc[-1]

    z_hist = hist[feature_cols].copy()
    mu = z_hist.mean()
    sigma = z_hist.std().replace(0, 1.0)
    hist_mat = ((z_hist - mu) / sigma).to_numpy(dtype=float)
    cur_vec = ((current[feature_cols] - mu) / sigma).to_numpy(dtype=float)

    distances = np.sqrt(((hist_mat - cur_vec) ** 2).sum(axis=1))
    hist = hist.assign(_distance=distances)
    analogs = hist.nsmallest(min(k_neighbors, len(hist)), "_distance").copy()

    labels = []
    both_hits = []
    fwd_returns = []
    t_hit = []
    for idx in analogs.index:
        loc = work.index.get_loc(idx)
        row = work.loc[idx]
        upper = row["Close"] + row["ATR14"] * target_atr
        lower = row["Close"] - row["ATR14"] * stop_atr
        future = work.iloc[loc + 1 : loc + 1 + horizon]
        label, both_hit = _first_barrier_hit(future, upper, lower)
        labels.append(label)
        both_hits.append(1.0 if both_hit else 0.0)

        if not future.empty:
            horizon_ret = future["Close"].iloc[-1] / row["Close"] - 1.0
            fwd_returns.append(horizon_ret)
            hit_bar = np.nan
            for n, (_, frow) in enumerate(future.iterrows(), start=1):
                if frow["High"] >= upper or frow["Low"] <= lower:
                    hit_bar = n
                    break
            t_hit.append(hit_bar)
        else:
            fwd_returns.append(0.0)
            t_hit.append(np.nan)

    analogs["label"] = labels
    analogs["both_hit"] = both_hits
    analogs["future_ret"] = fwd_returns
    analogs["hit_bar"] = t_hit

    p_up = float((analogs["label"] == 1).mean())
    p_down = float((analogs["label"] == -1).mean())
    p_neutral = max(0.0, 1.0 - p_up - p_down)
    p_trap = float(analogs["both_hit"].mean())

    current_dir = int(current["TrendDir"])
    if current_dir > 0:
        p_continue = p_up
        p_revert = p_down
    elif current_dir < 0:
        p_continue = p_down
        p_revert = p_up
    else:
        p_continue = max(p_up, p_down)
        p_revert = min(p_up, p_down)

    analogs_sorted = analogs.sort_values("_distance").copy()
    quantiles = analogs_sorted["future_ret"].quantile([0.2, 0.5, 0.8]).to_dict()
    avg_hit = safe_float(pd.Series(t_hit).dropna().mean(), default=float(horizon))

    return {
        "p_up": p_up,
        "p_down": p_down,
        "p_neutral": p_neutral,
        "p_trap": p_trap,
        "p_continue": p_continue,
        "p_revert": p_revert,
        "ret_q20": float(quantiles.get(0.2, 0.0)),
        "ret_q50": float(quantiles.get(0.5, 0.0)),
        "ret_q80": float(quantiles.get(0.8, 0.0)),
        "avg_hit_bars": avg_hit,
        "analog_table": analogs_sorted[["_distance", "label", "both_hit", "future_ret", "hit_bar"]].copy(),
    }


# =========================================================
# DECISION ENGINE
# =========================================================
def explain_state(last: pd.Series, model: Dict) -> Tuple[str, str]:
    trend_raw = safe_float(last["TrendRaw"])
    dist = safe_float(last["DistMeanATR"])
    comp = safe_float(last["Compression"])
    conflict = safe_float(last["Conflict"])
    instability = safe_float(last["Instability"])

    if conflict > 0.68 or instability > 0.42:
        return "Chaos / No-Trade", "Sinyal saling bertabrakan. Model lebih menyarankan tunggu daripada memaksa entry."
    if abs(trend_raw) < 0.18 and comp > 0.5:
        return "Sideways / Menunggu Arah", "Harga sedang rapat dan belum punya arah yang jelas."
    if trend_raw > 0.45 and dist < 1.8 and model["p_continue"] > 0.55:
        return "Markup Sehat", "Arah naik masih rapi dan analog masa lalu lebih sering lanjut daripada gagal."
    if trend_raw < -0.45 and dist < 1.8 and model["p_continue"] > 0.55:
        return "Markdown Sehat", "Arah turun masih rapi dan analog masa lalu lebih sering lanjut daripada gagal."
    if trend_raw > 0.25 and dist >= 1.8:
        return "Bull Sudah Lari", "Arah naik masih ada, tapi posisi sudah cukup jauh dari harga wajar sehingga rawan telat masuk."
    if trend_raw < -0.25 and dist >= 1.8:
        return "Bear Sudah Lari", "Arah turun masih ada, tapi posisi sudah cukup jauh dari harga wajar sehingga rawan telat masuk."
    if trend_raw > 0 and model["p_revert"] > model["p_continue"]:
        return "Naik Tapi Rawan Balik", "Trend masih naik, tapi analog lama lebih banyak berakhir pullback daripada lanjut."
    if trend_raw < 0 and model["p_revert"] > model["p_continue"]:
        return "Turun Tapi Rawan Pantul", "Trend masih turun, tapi analog lama lebih banyak berakhir bounce daripada lanjut."
    return "Transisi", "Market sedang pindah fase. Bias ada, tapi belum bersih."




def infer_sleeve(category: str, symbol: str, last: pd.Series) -> str:
    sym = str(symbol).upper()
    if category == "Crypto":
        return "Crypto Momentum"
    if category == "Forex":
        return "FX Mean Reversion"
    if category == "IHSG":
        return "IHSG Beta"
    if category == "Futures & Commodities":
        if sym in {"GC=F", "SI=F", "DX=F", "ZN=F", "ZB=F"}:
            return "Defensive Macro"
        if sym in {"CL=F", "BZ=F", "NG=F", "HG=F", "ZC=F", "ZS=F", "ZW=F"}:
            return "Cyclical Commodities"
        return "Broad Futures"
    if category == "US Stocks":
        if sym in {"AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL", "TSLA", "AMD", "PLTR", "QQQ", "SPY"}:
            return "US Trend Leaders"
        if safe_float(last.get("TrendRaw", 0.0)) > 0.35 and safe_float(last.get("Conflict", 1.0)) < 0.45:
            return "US Trend Leaders"
        return "US General Equities"
    return "Generalist"


def sleeve_profile(sleeve: str) -> Dict:
    profiles = {
        "US Trend Leaders": {
            "t2_mult": 2.1, "min_conf_go": 64.0, "min_trade_go": 62.0, "min_conf_watch": 54.0,
            "min_trade_watch": 52.0, "late_soft": 0.52, "trap_mult": 0.90, "desc": "Fokus ke saham pemimpin tren. Boleh kasih ruang tren bernapas, tapi jangan kejar kalau sudah terlalu jauh."
        },
        "US General Equities": {
            "t2_mult": 1.8, "min_conf_go": 67.0, "min_trade_go": 66.0, "min_conf_watch": 57.0,
            "min_trade_watch": 56.0, "late_soft": 0.42, "trap_mult": 1.00, "desc": "Saham umum butuh filter lebih ketat daripada trend leaders."
        },
        "IHSG Beta": {
            "t2_mult": 1.7, "min_conf_go": 66.0, "min_trade_go": 64.0, "min_conf_watch": 56.0,
            "min_trade_watch": 54.0, "late_soft": 0.38, "trap_mult": 1.05, "desc": "IHSG beta cenderung sensitif ke sentimen umum, jadi hindari entry telat."
        },
        "Defensive Macro": {
            "t2_mult": 1.6, "min_conf_go": 61.0, "min_trade_go": 60.0, "min_conf_watch": 52.0,
            "min_trade_watch": 50.0, "late_soft": 0.34, "trap_mult": 1.00, "desc": "Gold dan aset defensif biasanya lebih rapi, tapi targetnya jangan terlalu agresif."
        },
        "Cyclical Commodities": {
            "t2_mult": 1.9, "min_conf_go": 67.0, "min_trade_go": 64.0, "min_conf_watch": 57.0,
            "min_trade_watch": 55.0, "late_soft": 0.36, "trap_mult": 1.08, "desc": "Komoditas siklikal lebih liar. Butuh kompresi yang pecah rapi dan trap risk rendah."
        },
        "Broad Futures": {
            "t2_mult": 1.7, "min_conf_go": 65.0, "min_trade_go": 62.0, "min_conf_watch": 55.0,
            "min_trade_watch": 53.0, "late_soft": 0.38, "trap_mult": 1.03, "desc": "Futures umum diperlakukan moderat: tidak terlalu longgar, tidak terlalu ketat."
        },
        "FX Mean Reversion": {
            "t2_mult": 1.4, "min_conf_go": 69.0, "min_trade_go": 68.0, "min_conf_watch": 59.0,
            "min_trade_watch": 58.0, "late_soft": 0.28, "trap_mult": 1.10, "desc": "FX mayor cenderung lebih cocok dibaca hati-hati. Entry telat cepat merusak expectancy."
        },
        "Crypto Momentum": {
            "t2_mult": 2.3, "min_conf_go": 71.0, "min_trade_go": 69.0, "min_conf_watch": 60.0,
            "min_trade_watch": 58.0, "late_soft": 0.26, "trap_mult": 1.20, "desc": "Crypto boleh target lebih lebar, tapi trap risk dan late entry harus dihukum keras."
        },
        "Generalist": {
            "t2_mult": 1.8, "min_conf_go": 66.0, "min_trade_go": 64.0, "min_conf_watch": 56.0,
            "min_trade_watch": 54.0, "late_soft": 0.40, "trap_mult": 1.00, "desc": "Profil default saat sleeve belum jelas."
        },
    }
    return profiles.get(sleeve, profiles["Generalist"])



def build_decision_frame(data: pd.DataFrame, model: Dict, category: str, symbol: str, horizon: int, target_atr: float, stop_atr: float) -> Dict:
    last = data.iloc[-1]
    close = safe_float(last["Close"])
    atr = safe_float(last["ATR14"])
    mean_price = safe_float(last["MeanPrice"])
    support = safe_float(last["Support20"])
    resistance = safe_float(last["Resistance20"])
    trend_dir = int(last["TrendDir"])

    state_name, state_text = explain_state(last, model)
    sleeve = infer_sleeve(category, symbol, last)
    cfg = sleeve_profile(sleeve)

    bull_prob = model["p_up"]
    bear_prob = model["p_down"]
    neutral_prob = model["p_neutral"]
    trap_prob = model["p_trap"]

    gap = 0.06 if sleeve in {"FX Mean Reversion", "IHSG Beta"} else 0.04
    if bull_prob > bear_prob + gap:
        direction = "Bullish"
        active_prob = bull_prob
        t1 = close + atr * target_atr
        t2 = close + atr * target_atr * cfg["t2_mult"]
        invalidation = min(mean_price, support) if support > 0 else mean_price
        risk = max(close - invalidation, atr * stop_atr * 0.8)
        reward = max(t2 - close, 0.0)
        ev = max(0.0, bull_prob - trap_prob * cfg["trap_mult"] * 0.15 - max(0.0, safe_float(last["DistMeanATR"]) - 2.0) * 0.03) * reward - (1.0 - bull_prob) * risk
    elif bear_prob > bull_prob + gap:
        direction = "Bearish"
        active_prob = bear_prob
        t1 = close - atr * target_atr
        t2 = close - atr * target_atr * cfg["t2_mult"]
        invalidation = max(mean_price, resistance) if resistance > 0 else mean_price
        risk = max(invalidation - close, atr * stop_atr * 0.8)
        reward = max(close - t2, 0.0)
        ev = max(0.0, bear_prob - trap_prob * cfg["trap_mult"] * 0.15 - max(0.0, safe_float(last["DistMeanATR"]) - 2.0) * 0.03) * reward - (1.0 - bear_prob) * risk
    else:
        direction = "Netral"
        active_prob = max(bull_prob, bear_prob, neutral_prob)
        t1 = mean_price
        t2 = mean_price
        invalidation = mean_price
        risk = atr * stop_atr
        reward = atr * target_atr * 0.7
        ev = 0.0

    rr = reward / (risk + EPS)

    late_penalty = clip01(max(0.0, safe_float(last["DistMeanATR"]) - 1.7) / 1.2 + 0.55 * trap_prob)
    direction_clarity = max(bull_prob, bear_prob, neutral_prob)
    trend_strength = clip01(abs(safe_float(last["TrendRaw"])) / 1.8)
    stability = clip01(1.0 - safe_float(last["Instability"]))
    cleanliness = clip01(1.0 - safe_float(last["Conflict"]))
    trap_safety = clip01(1.0 - trap_prob * cfg["trap_mult"])
    stretch_safety = clip01(1.0 - min(1.0, safe_float(last["DistMeanATR"]) / 3.0))

    confidence = 100.0 * clip01(
        0.27 * direction_clarity +
        0.22 * trend_strength +
        0.18 * stability +
        0.17 * cleanliness +
        0.16 * trap_safety
    )

    tradeability = 100.0 * clip01(
        0.28 * cleanliness +
        0.21 * stability +
        0.17 * trap_safety +
        0.14 * stretch_safety +
        0.12 * direction_clarity +
        0.08 * clip01(1.0 - late_penalty)
    )

    route_bias = "trend-following" if sleeve in {"US Trend Leaders", "Crypto Momentum", "Cyclical Commodities"} else ("balanced" if sleeve in {"US General Equities", "Broad Futures", "IHSG Beta"} else "more defensive")

    if direction == "Netral" or ev <= 0:
        sizing = "No Trade / Tunggu"
        meta_label = "SKIP"
    elif confidence >= cfg["min_conf_go"] and tradeability >= cfg["min_trade_go"] and late_penalty <= cfg["late_soft"]:
        meta_label = "GO"
        sizing = "Normal → Agresif" if confidence >= cfg["min_conf_go"] + 10 and tradeability >= cfg["min_trade_go"] + 8 and rr >= 1.6 else "Normal"
    elif confidence >= cfg["min_conf_watch"] and tradeability >= cfg["min_trade_watch"]:
        meta_label = "WATCH"
        sizing = "Reduced" if late_penalty <= cfg["late_soft"] + 0.12 else "Tiny Feeler"
    else:
        meta_label = "SKIP"
        sizing = "No Trade / Tunggu"

    confidence_label = id_label(confidence, [(72, "tinggi"), (45, "sedang"), (0, "rendah")])
    tradeability_label = id_label(tradeability, [(78, "sangat enak"), (58, "layak"), (40, "tipis"), (0, "jelek")])

    if direction == "Bullish":
        simple_read = (
            f"Sleeve aktif: {sleeve}. Saat ini model lebih condong ke NAIK. Peluang naik sekitar {bull_prob * 100:.0f}%, "
            f"risiko jebakan sekitar {trap_prob * 100:.0f}%, dan gaya baca yang dipakai lebih {route_bias}."
        )
    elif direction == "Bearish":
        simple_read = (
            f"Sleeve aktif: {sleeve}. Saat ini model lebih condong ke TURUN. Peluang turun sekitar {bear_prob * 100:.0f}%, "
            f"risiko jebakan sekitar {trap_prob * 100:.0f}%, dan gaya baca yang dipakai lebih {route_bias}."
        )
    else:
        simple_read = (
            f"Sleeve aktif: {sleeve}. Saat ini model belum punya arah tegas. Ini biasanya berarti routing masih defensif "
            f"atau market belum cukup bersih untuk dipaksa."
        )

    reasons = []
    if abs(safe_float(last["TrendRaw"])) > 0.45:
        reasons.append("trend cukup kuat")
    if safe_float(last["Conflict"]) < 0.40:
        reasons.append("sinyal internal cukup kompak")
    if safe_float(last["Compression"]) > 0.5:
        reasons.append("harga lagi rapat / kompres")
    if safe_float(last["Expansion"]) > 0.5:
        reasons.append("harga lagi ekspansi")
    if trap_prob > 0.35:
        reasons.append("risiko jebakan cukup tinggi")
    if late_penalty > 0.45:
        reasons.append("entry sudah agak telat")
    if sleeve in {"FX Mean Reversion", "Defensive Macro"}:
        reasons.append("routing lebih defensif")
    if sleeve in {"US Trend Leaders", "Crypto Momentum"}:
        reasons.append("routing kasih ruang ke continuation")
    if not reasons:
        reasons.append("kondisi campuran")

    move_low = close * (1.0 + model["ret_q20"])
    move_mid = close * (1.0 + model["ret_q50"])
    move_high = close * (1.0 + model["ret_q80"])

    return {
        "sleeve": sleeve,
        "sleeve_desc": cfg["desc"],
        "route_bias": route_bias,
        "meta_label": meta_label,
        "state_name": state_name,
        "state_text": state_text,
        "direction": direction,
        "simple_read": simple_read,
        "bull_prob": bull_prob,
        "bear_prob": bear_prob,
        "neutral_prob": neutral_prob,
        "continue_prob": model["p_continue"],
        "revert_prob": model["p_revert"],
        "trap_prob": trap_prob,
        "active_prob": active_prob,
        "confidence": confidence,
        "confidence_label": confidence_label,
        "tradeability": tradeability,
        "tradeability_label": tradeability_label,
        "late_penalty": late_penalty,
        "sizing": sizing,
        "t1": t1,
        "t2": t2,
        "invalidation": invalidation,
        "rr": rr,
        "ev": ev,
        "move_low": move_low,
        "move_mid": move_mid,
        "move_high": move_high,
        "avg_hit_bars": model["avg_hit_bars"],
        "reasons": reasons,
        "trend_dir": trend_dir,
        "last_row": last,
    }


# =========================================================
# VISUALS
# =========================================================
def plot_price(data: pd.DataFrame, decision: Dict, symbol: str) -> go.Figure:
    tail = data.tail(180).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=tail.index,
            open=tail["Open"],
            high=tail["High"],
            low=tail["Low"],
            close=tail["Close"],
            name="Harga",
        )
    )
    for col in ["EMA20", "EMA50", "EMA100", "VWAP"]:
        fig.add_trace(go.Scatter(x=tail.index, y=tail[col], mode="lines", name=col))

    last_ts = tail.index[-1]
    for y, label in [
        (decision["t1"], "T1"),
        (decision["t2"], "T2"),
        (decision["invalidation"], "Invalidation"),
    ]:
        fig.add_hline(y=y, line_dash="dot", annotation_text=label, annotation_position="right")

    fig.update_layout(
        title=f"{symbol} - Harga + Mean + Target",
        xaxis_title="Waktu",
        yaxis_title="Harga",
        height=640,
        xaxis_rangeslider_visible=False,
        legend_title="Layer",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# =========================================================
# SCANNER
# =========================================================

def summarize_symbol(category: str, symbol: str, tf_label: str, horizon: int, k_neighbors: int, target_atr: float, stop_atr: float) -> Optional[Dict]:
    try:
        raw = fetch_history(category, symbol, tf_label)
        if raw is None or raw.empty or len(raw) < max(120, horizon + 30):
            return None
        data = add_features(raw)
        model = nearest_analog_model(data, horizon=horizon, k_neighbors=k_neighbors, target_atr=target_atr, stop_atr=stop_atr)
        decision = build_decision_frame(data, model, category, symbol, horizon, target_atr, stop_atr)
        rank_score = 0.40 * decision["tradeability"] + 0.35 * decision["confidence"] + 25.0 * max(decision["ev"], 0.0) - 18.0 * decision["trap_prob"]
        return {
            "symbol": symbol,
            "sleeve": decision["sleeve"],
            "meta": decision["meta_label"],
            "state": decision["state_name"],
            "direction": decision["direction"],
            "bull_prob": round(decision["bull_prob"] * 100, 1),
            "bear_prob": round(decision["bear_prob"] * 100, 1),
            "trap_prob": round(decision["trap_prob"] * 100, 1),
            "confidence": round(decision["confidence"], 1),
            "tradeability": round(decision["tradeability"], 1),
            "ev": round(decision["ev"], 4),
            "rr": round(decision["rr"], 2),
            "rank_score": round(rank_score, 2),
            "sizing": decision["sizing"],
            "why": ", ".join(decision["reasons"][:3]),
        }
    except Exception:
        return None


# =========================================================
# UI
# =========================================================
st.title("📊 Quant Story Engine - Python")
st.caption(
    "Versi ini dibuat untuk bantu baca market dengan bahasa yang lebih gampang. "
    "Bukan ramalan sakti. Mesin ini membandingkan kondisi sekarang dengan kondisi historis yang mirip, "
    "lalu menghitung peluang lanjut, balik arah, atau jadi jebakan."
)

with st.expander("Cara baca paling sederhana", expanded=True):
    st.markdown(
        """
1. **Arah dominan** = model lebih condong naik, turun, atau netral.
2. **State pasar** = market lagi trending sehat, sideways, telat masuk, atau chaos.
3. **Tradeability** = seberapa enak kondisi market buat ditrade.
4. **Confidence** = seberapa yakin model terhadap bacaan arahnya.
5. **T1 / T2 / Invalidation** = target dekat, target lanjut, dan level salah.
6. **Trap risk** = risiko breakout palsu / jebakan dua arah.
        """
    )

with st.sidebar:
    st.header("Pengaturan")
    category = st.selectbox("Kategori market", ["US Stocks", "IHSG", "Futures & Commodities", "Forex", "Crypto", "Custom"])
    tf_label = st.selectbox("Timeframe", list(INTERVAL_MAP.keys()), index=0)
    horizon = st.slider("Berapa bar ke depan yang dihitung", 5, 30, 10)
    k_neighbors = st.slider("Jumlah analog historis", 10, 100, 40)
    target_atr = st.slider("Jarak target (ATR)", 0.5, 3.0, 1.0, 0.1)
    stop_atr = st.slider("Jarak stop (ATR)", 0.5, 3.0, 1.0, 0.1)
    scan_max = st.slider("Maks simbol saat scanner", 5, 50, 15)

    st.markdown("---")
    mode = st.radio("Cara pilih simbol", ["Ketik manual", "Pilih dari daftar market"], index=1)

    universe_df = pd.DataFrame(columns=["symbol", "name"])
    default_symbol = "AAPL"

    try:
        if category == "US Stocks":
            universe_df = load_us_universe()
            default_symbol = POPULAR_US[0]
        elif category == "IHSG":
            universe_df = load_ihsg_universe()
            default_symbol = POPULAR_IHSG[0]
        elif category == "Crypto":
            universe_df = load_crypto_universe()
            default_symbol = POPULAR_CRYPTO[0]
        elif category == "Forex":
            universe_df = load_forex_universe()
            default_symbol = "EURUSD=X"
        elif category == "Futures & Commodities":
            universe_df = load_futures_universe()
            default_symbol = "GC=F"
    except Exception as exc:
        st.warning(f"Universe list gagal di-load: {exc}")

    if category == "Custom":
        symbol = st.text_input("Masukkan simbol", value="AAPL")
    else:
        if mode == "Ketik manual":
            help_text = {
                "US Stocks": "Contoh: AAPL, MSFT, NVDA",
                "IHSG": "Contoh: BBCA.JK, BBRI.JK, TLKM.JK",
                "Forex": "Contoh: EURUSD=X, USDIDR=X",
                "Futures & Commodities": "Contoh: GC=F, CL=F, NQ=F",
                "Crypto": "Contoh: BTC/USDT, ETH/USDT, SOL/USDT",
            }
            symbol = st.text_input("Masukkan simbol", value=default_symbol, help=help_text.get(category, ""))
        else:
            search_text = st.text_input("Cari ticker / nama", value="")
            filtered = universe_df.copy()
            if not filtered.empty and search_text.strip():
                q = search_text.strip().lower()
                filtered = filtered[
                    filtered["symbol"].astype(str).str.lower().str.contains(q, na=False)
                    | filtered["name"].astype(str).str.lower().str.contains(q, na=False)
                ]
            if filtered.empty:
                symbol = default_symbol
                st.info("Hasil filter kosong. Pakai simbol default dulu.")
            else:
                filtered_display = (filtered["symbol"] + " - " + filtered["name"].fillna(""))
                selected = st.selectbox("Pilih simbol", filtered_display.tolist(), index=0)
                symbol = selected.split(" - ")[0].strip()

    st.markdown("---")
    custom_watchlist = st.text_area(
        "Custom watchlist untuk scanner (pisahkan dengan koma / baris baru)",
        value="",
        help="Contoh campur market: AAPL, BBCA.JK, EURUSD=X, GC=F, BTC/USDT",
    )


def parse_watchlist(text: str) -> List[str]:
    raw = text.replace("\n", ",").split(",")
    return [x.strip() for x in raw if x.strip()]


watchlist = parse_watchlist(custom_watchlist)


def infer_category_from_symbol(sym: str) -> str:
    sym = str(sym).strip()
    if "/" in sym:
        return "Crypto"
    if sym.endswith(".JK"):
        return "IHSG"
    if sym.endswith("=X"):
        return "Forex"
    if sym.endswith("=F"):
        return "Futures & Commodities"
    return "US Stocks"


active_category = category if category != "Custom" else infer_category_from_symbol(symbol)

# =========================================================
# MAIN ANALYSIS
# =========================================================
if not symbol:
    st.stop()

try:
    raw = fetch_history(active_category, symbol, tf_label)
    if raw is None or raw.empty:
        st.error("Data kosong. Cek lagi simbol atau provider data untuk market ini.")
        st.stop()
    data = add_features(raw)
    model = nearest_analog_model(data, horizon=horizon, k_neighbors=k_neighbors, target_atr=target_atr, stop_atr=stop_atr)
    decision = build_decision_frame(data, model, active_category, symbol, horizon, target_atr, stop_atr)
except Exception as exc:
    st.error(f"Analisis gagal: {exc}")
    st.stop()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Arah dominan", decision["direction"])
col2.metric("State pasar", decision["state_name"])
col3.metric("Sleeve", decision["sleeve"])
col4.metric("Tradeability", f"{decision['tradeability']:.0f}%", decision["tradeability_label"])
col5.metric("Confidence", f"{decision['confidence']:.0f}%", decision["confidence_label"])

lamp = "🟢" if decision["meta_label"] == "GO" else ("🟡" if decision["meta_label"] == "WATCH" else "🔴")
st.info(f"{lamp} {decision['simple_read']}")

with st.container(border=True):
    st.subheader("Bacaan cepat untuk orang awam")
    st.write(decision["state_text"])

    read1, read2, read3 = st.columns(3)
    read1.markdown(
        f"**Kalau mau cari arah**\n\n"
        f"- Bullish: **{decision['bull_prob'] * 100:.1f}%**\n"
        f"- Bearish: **{decision['bear_prob'] * 100:.1f}%**\n"
        f"- Netral: **{decision['neutral_prob'] * 100:.1f}%**"
    )
    read2.markdown(
        f"**Kalau mau cari kelanjutan**\n\n"
        f"- Lanjut arah sekarang: **{decision['continue_prob'] * 100:.1f}%**\n"
        f"- Balik arah: **{decision['revert_prob'] * 100:.1f}%**\n"
        f"- Jebakan / dua arah: **{decision['trap_prob'] * 100:.1f}%**"
    )
    read3.markdown(
        f"**Kalau mau eksekusi**\n\n"
        f"- Meta-label: **{decision['meta_label']}**\n"
        f"- Size saran: **{decision['sizing']}**\n"
        f"- Late entry penalty: **{decision['late_penalty'] * 100:.1f}%**\n"
        f"- RR kasar: **{decision['rr']:.2f}**"
    )


with st.container(border=True):
    st.subheader("Routing multi-sleeve")
    r1, r2, r3 = st.columns(3)
    r1.markdown(f"**Sleeve aktif**\n\n`{decision['sleeve']}`")
    r2.markdown(f"**Gaya mesin**\n\n`{decision['route_bias']}`")
    r3.markdown(f"**Keputusan cepat**\n\n`{decision['meta_label']}`")
    st.write(decision["sleeve_desc"])

st.plotly_chart(plot_price(data, decision, symbol), use_container_width=True)

line1, line2 = st.columns([1.2, 1])
with line1:
    with st.container(border=True):
        st.subheader("Level penting")
        lvl = pd.DataFrame(
            {
                "Item": ["Harga sekarang", "Harga wajar / mean", "Target 1", "Target 2", "Invalidation"],
                "Nilai": [
                    data["Close"].iloc[-1],
                    data["MeanPrice"].iloc[-1],
                    decision["t1"],
                    decision["t2"],
                    decision["invalidation"],
                ],
            }
        )
        st.dataframe(lvl.style.format({"Nilai": "{:.4f}"}), use_container_width=True, hide_index=True)

        st.markdown(
            f"**Expected move berdasar analog historis ({horizon} bar):**\n\n"
            f"- Skenario lemah: `{decision['move_low']:.4f}`\n"
            f"- Skenario dasar: `{decision['move_mid']:.4f}`\n"
            f"- Skenario kuat: `{decision['move_high']:.4f}`\n"
            f"- Rata-rata waktu barrier kena: `{decision['avg_hit_bars']:.1f}` bar"
        )

with line2:
    with st.container(border=True):
        st.subheader("Kenapa model ngomong begitu?")
        for reason in decision["reasons"]:
            st.write(f"- {reason}")
        st.markdown(
            f"\n**Expected value kasar:** `{decision['ev']:.4f}`  \n"
            f"**Probabilitas aktif:** `{decision['active_prob'] * 100:.1f}%`"
        )

# =========================================================
# DETAIL TAB
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs(["Ringkasan detail", "Analog historis", "Scanner", "Routing notes"])

with tab1:
    last = decision["last_row"]
    score_df = pd.DataFrame(
        {
            "Komponen": [
                "Trend raw",
                "Jarak dari mean (ATR)",
                "Instability",
                "Conflict",
                "Compression",
                "Expansion",
                "Effort",
                "Volume Z",
            ],
            "Nilai": [
                safe_float(last["TrendRaw"]),
                safe_float(last["DistMeanATR"]),
                safe_float(last["Instability"]),
                safe_float(last["Conflict"]),
                safe_float(last["Compression"]),
                safe_float(last["Expansion"]),
                safe_float(last["EffortZ"]),
                safe_float(last["VolumeZ"]),
            ],
            "Arti sederhana": [
                "Positif = dorongan naik, negatif = dorongan turun",
                "Semakin besar = makin jauh dari harga wajar",
                "Semakin tinggi = arah sering ganti",
                "Semakin tinggi = sinyal saling bertabrakan",
                "1 = harga lagi rapat / diam",
                "1 = harga lagi meledak / bergerak besar",
                "Positif = volume lebih mendukung arah naik",
                "Positif = volume di atas normal",
            ],
        }
    )
    st.dataframe(score_df.style.format({"Nilai": "{:.4f}"}), use_container_width=True, hide_index=True)

with tab2:
    analog_table = model["analog_table"].copy().reset_index().rename(columns={"index": "Tanggal analog"})
    analog_table["Label"] = analog_table["label"].map({1: "Naik duluan", -1: "Turun duluan", 0: "Tidak jelas"})
    analog_table["Trap"] = np.where(analog_table["both_hit"] > 0, "Ya", "Tidak")
    analog_table["Future Return"] = analog_table["future_ret"] * 100
    analog_table = analog_table[["Tanggal analog", "_distance", "Label", "Trap", "Future Return", "hit_bar"]]
    analog_table.columns = ["Tanggal analog", "Jarak kemiripan", "Hasil", "Trap", "Return %", "Bar sampai kena barrier"]
    st.write("Semakin kecil jarak kemiripan, berarti kondisi historis itu semakin mirip dengan kondisi sekarang.")
    st.dataframe(
        analog_table.style.format({"Jarak kemiripan": "{:.3f}", "Return %": "{:.2f}", "Bar sampai kena barrier": "{:.1f}"}),
        use_container_width=True,
        hide_index=True,
    )

with tab3:
    st.write(
        "Scanner berguna untuk mencari simbol yang paling rapi menurut model. "
        "Kalau watchlist custom diisi, scanner akan pakai isi watchlist itu. Kalau kosong, scanner pakai kategori aktif."
    )

    if watchlist:
        symbols_to_scan = watchlist[:scan_max]
        categories = []
        for s in symbols_to_scan:
            if "/" in s:
                categories.append("Crypto")
            elif s.endswith(".JK"):
                categories.append("IHSG")
            elif s.endswith("=X"):
                categories.append("Forex")
            elif s.endswith("=F"):
                categories.append("Futures & Commodities")
            else:
                categories.append("US Stocks")
        cat_map = dict(zip(symbols_to_scan, categories))
    else:
        if category == "US Stocks":
            symbols_to_scan = POPULAR_US[:scan_max]
        elif category == "IHSG":
            symbols_to_scan = POPULAR_IHSG[:scan_max]
        elif category == "Crypto":
            symbols_to_scan = POPULAR_CRYPTO[:scan_max]
        elif category == "Forex":
            symbols_to_scan = list(FOREX_LIBRARY.values())[:scan_max]
        elif category == "Futures & Commodities":
            symbols_to_scan = list(FUTURES_LIBRARY.values())[:scan_max]
        else:
            symbols_to_scan = [symbol]
        cat_map = {s: (category if category != "Custom" else infer_category_from_symbol(s)) for s in symbols_to_scan}

    if st.button("Jalankan scanner"):
        rows = []
        progress = st.progress(0)
        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = {
                ex.submit(
                    summarize_symbol,
                    cat_map[s],
                    s,
                    tf_label,
                    horizon,
                    k_neighbors,
                    target_atr,
                    stop_atr,
                ): s
                for s in symbols_to_scan
            }
            done = 0
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:
                    rows.append(result)
                done += 1
                progress.progress(done / max(len(futures), 1))

        if rows:
            scan_df = pd.DataFrame(rows).sort_values(["tradeability", "confidence", "ev"], ascending=False)
            st.dataframe(scan_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Tidak ada hasil scanner yang valid. Coba kurangi jumlah simbol atau ganti timeframe.")

st.markdown("---")
st.caption(
    "Catatan jujur: US stocks, IHSG, dan crypto diambil dari daftar simbol publik yang bisa di-load otomatis. "
    "Forex dan futures/commodities disediakan lewat library simbol luas + input manual. Kalau ada simbol yang belum muncul di daftar, "
    "tetap bisa diketik manual selama provider data mendukung."
)
