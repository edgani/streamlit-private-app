import math
from datetime import datetime, timezone
from typing import Dict, List, Tuple
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

st.set_page_config(page_title="Quad Impact Signal Dashboard", layout="wide")

# -------------------------------------------------
# STYLE
# -------------------------------------------------
st.markdown(
    """
<style>
:root {
  --bg:#07101f;
  --card:#0c1526;
  --line:#263246;
  --muted:#9fb0c8;
  --text:#f3f6fb;
  --green:#33d17a;
  --red:#ff5d5d;
  --amber:#ffbf47;
  --blue:#6ea8fe;
}
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg);
  color: var(--text);
}
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1680px;}
h1,h2,h3,h4,h5,h6,p,span,div,label {color: var(--text);}
.card {
  background: linear-gradient(180deg, rgba(16,24,41,0.98), rgba(10,17,30,0.98));
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 14px 16px;
  height: 100%;
}
.kpi {
  background: linear-gradient(180deg, rgba(19,29,50,1), rgba(12,20,35,1));
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 12px 14px;
  min-height: 94px;
}
.section-title {
  font-weight: 800;
  letter-spacing: .02em;
  margin-bottom: 8px;
  font-size: 1.0rem;
}
.metric-title {
  font-size: .76rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .05em;
}
.metric-value {
  font-size: 1.75rem;
  font-weight: 800;
  line-height: 1.1;
}
.metric-sub {
  font-size: .88rem;
  color:#c4d2e6;
  margin-top:4px;
}
.small-muted {color: var(--muted); font-size: .92rem;}
.note-box {
  border:1px solid #27476e;
  background: rgba(18,46,79,.65);
  border-radius:12px;
  padding:12px 14px;
}
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
.badge {
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
.bar-wrap {
  width:100%; background:#101b2d; border:1px solid #243147; border-radius:10px; overflow:hidden; height:16px;
}
.bar-fill {
  height:100%; background: linear-gradient(90deg, #2d6cdf, #6ea8fe);
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def pct(x: float) -> str:
    return f"{100*x:.1f}%"


def num1(x: float) -> str:
    return f"{x:.1f}"


def bucket(x: float, cuts: Tuple[float, float], labels: Tuple[str, str, str]) -> str:
    if x < cuts[0]:
        return labels[0]
    if x < cuts[1]:
        return labels[1]
    return labels[2]


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


def bar_html(value: float) -> str:
    width = int(round(100 * clamp01(value)))
    return f"<div class='bar-wrap'><div class='bar-fill' style='width:{width}%'></div></div>"


def pill(text: str) -> str:
    return f"<span class='badge'>{text}</span>"


def ladder_state(score: float, side: str) -> str:
    if side == "top":
        if score < 0.18:
            return "No clean top"
        if score < 0.35:
            return "Building top"
        if score < 0.55:
            return "Provisional top"
        return "Extended / blow-off risk"
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


def _norm_tanh(x: float, scale: float) -> float:
    if scale <= 0:
        return 0.5
    return clamp01((math.tanh(x / scale) + 1.0) / 2.0)


def _trend_score(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 80:
        return 0.5
    ema20 = s.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = s.ewm(span=50, adjust=False).mean().iloc[-1]
    ema100 = s.ewm(span=100, adjust=False).mean().iloc[-1]
    p = float(s.iloc[-1])
    score = 0.0
    score += 0.35 * float(p > ema20)
    score += 0.30 * float(ema20 > ema50)
    score += 0.20 * float(ema50 > ema100)
    score += 0.15 * _norm_tanh((p / ema50) - 1.0, 0.10)
    return clamp01(score)


def ret_n(s: pd.Series, n: int = 21) -> float:
    if s.empty or len(s) < n + 1:
        return 0.0
    return float(s.iloc[-n:].pct_change().add(1).prod() - 1)


def stretch_n(s: pd.Series, n: int = 63) -> float:
    if s.empty or len(s) < n:
        return 0.0
    return float((s.iloc[-1] / s.iloc[-n:].mean()) - 1)


def _clean_tickers(raw: str, suffix: str = "") -> List[str]:
    out = []
    for x in raw.split(','):
        t = x.strip().upper()
        if not t:
            continue
        if suffix and not t.endswith(suffix):
            t = f"{t}{suffix}"
        out.append(t)
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _coerce_series(obj, name: str = "value") -> pd.Series:
    if obj is None:
        return pd.Series(dtype=float, name=name)

    if isinstance(obj, pd.DataFrame):
        if obj.empty:
            return pd.Series(dtype=float, name=name)
        if name in obj.columns:
            sel = obj.loc[:, name]
            obj = sel.iloc[:, 0] if isinstance(sel, pd.DataFrame) else sel
        else:
            obj = obj.iloc[:, 0]

    if isinstance(obj, pd.Series):
        s = obj.copy()
    else:
        arr = np.asarray(obj)
        if arr.size == 0:
            return pd.Series(dtype=float, name=name)
        if arr.ndim == 0:
            s = pd.Series([arr.item()])
        elif arr.ndim == 1:
            s = pd.Series(arr)
        else:
            # Defensive fallback for duplicated columns / 2D slices from pandas.
            s = pd.Series(arr[:, 0])

    s = pd.to_numeric(s, errors="coerce").dropna()
    s.name = name
    return s


def _get_col_as_series(df: pd.DataFrame, col: str) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float, name=col)

    if isinstance(df.columns, pd.MultiIndex):
        last_level = df.columns.get_level_values(-1)
        if col in last_level:
            sel = df.loc[:, last_level == col]
            return _coerce_series(sel, col)
        first_level = df.columns.get_level_values(0)
        if col in first_level:
            sel = df.loc[:, first_level == col]
            return _coerce_series(sel, col)
        return pd.Series(dtype=float, name=col)

    if col not in df.columns:
        return pd.Series(dtype=float, name=col)

    sel = df.loc[:, col]
    return _coerce_series(sel, col)


def _period_to_range(period: str) -> str:
    mp = {
        "1mo": "1mo",
        "3mo": "3mo",
        "6mo": "6mo",
        "1y": "1y",
        "2y": "2y",
        "5y": "5y",
        "10y": "10y",
        "max": "max",
    }
    return mp.get(period, "1y")


# -------------------------------------------------
# DATA FETCH
# -------------------------------------------------
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
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


@st.cache_data(ttl=60 * 60 * 4, show_spinner=False)
def yahoo_close_http(ticker: str, period: str = "1y") -> pd.Series:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(ticker, safe='')}?range={_period_to_range(period)}&interval=1d&includeAdjustedClose=true"
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if not r.ok:
            return pd.Series(dtype=float, name=ticker)
        payload = r.json()
        result = (((payload or {}).get("chart") or {}).get("result") or [None])[0]
        if not result:
            return pd.Series(dtype=float, name=ticker)
        ts = result.get("timestamp") or []
        quote_block = (((result.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose"))
        if quote_block is None:
            quote_block = (((result.get("indicators") or {}).get("quote") or [{}])[0].get("close"))
        if not ts or quote_block is None:
            return pd.Series(dtype=float, name=ticker)
        idx = pd.to_datetime(ts, unit="s", utc=True).tz_localize(None)
        s = pd.Series(quote_block, index=idx, name=ticker)
        s = pd.to_numeric(s, errors="coerce").dropna()
        return s
    except Exception:
        return pd.Series(dtype=float, name=ticker)


@st.cache_data(ttl=60 * 60 * 4, show_spinner=False)
def yahoo_close(ticker: str, period: str = "1y") -> pd.Series:
    if yf is not None:
        try:
            data = yf.download(
                ticker,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if data is not None and len(data) > 0:
                if isinstance(data.columns, pd.MultiIndex):
                    if ("Close", ticker) in data.columns:
                        s = data[("Close", ticker)]
                    elif "Close" in data.columns.get_level_values(0):
                        s = data["Close"]
                    else:
                        s = data.iloc[:, 0]
                else:
                    s = data["Close"] if "Close" in data.columns else data.iloc[:, 0]
                s = _coerce_series(s, ticker)
                if not s.empty:
                    return s
        except Exception:
            pass
    return yahoo_close_http(ticker, period)


@st.cache_data(ttl=60 * 60 * 4, show_spinner=False)
def yahoo_close_batch(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    tickers = [t for t in tickers if t]
    if not tickers:
        return pd.DataFrame()

    def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = [str(c[-1]) if isinstance(c, tuple) else str(c) for c in out.columns]
        else:
            out.columns = [str(c) for c in out.columns]
        out = out.apply(pd.to_numeric, errors="coerce")
        out = out.groupby(level=0, axis=1).first()
        return out.sort_index()

    if yf is not None:
        for chunk_size in (max(1, min(40, len(tickers))), 15, 8, 1):
            frames = []
            ok = True
            for i in range(0, len(tickers), chunk_size):
                chunk = tickers[i : i + chunk_size]
                try:
                    data = yf.download(
                        chunk,
                        period=period,
                        interval="1d",
                        auto_adjust=True,
                        progress=False,
                        threads=False,
                    )
                except Exception:
                    ok = False
                    break
                if data is None or len(data) == 0:
                    continue
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        if "Close" in data.columns.get_level_values(0):
                            close = data["Close"].copy()
                        else:
                            close = data.iloc[:, : len(chunk)].copy()
                            close.columns = chunk[: close.shape[1]]
                    else:
                        col = "Close" if "Close" in data.columns else data.columns[0]
                        close = data[[col]].copy()
                        close.columns = [chunk[0]]
                    close = _flatten_cols(close)
                    if not close.empty:
                        frames.append(close)
                except Exception:
                    ok = False
                    break
            if ok and frames:
                merged = pd.concat(frames, axis=1)
                merged = _flatten_cols(merged)
                if not merged.empty:
                    return merged

    frames = []
    for ticker in tickers:
        s = yahoo_close_http(ticker, period)
        if not s.empty:
            frames.append(s.rename(ticker))
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, axis=1).sort_index()
    return _flatten_cols(merged)


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def get_market_caps(tickers: Tuple[str, ...]) -> Dict[str, float]:
    tickers = tuple([t for t in tickers if t])
    out: Dict[str, float] = {}
    if yf is not None:
        for ticker in tickers:
            try:
                fast = yf.Ticker(ticker).fast_info
                cap = None if fast is None else fast.get("market_cap")
                if cap is None:
                    info = yf.Ticker(ticker).info
                    cap = None if info is None else info.get("marketCap")
                if cap is not None and np.isfinite(cap) and float(cap) > 0:
                    out[ticker] = float(cap)
            except Exception:
                pass
    missing = [t for t in tickers if t not in out]
    for i in range(0, len(missing), 20):
        chunk = missing[i:i+20]
        try:
            url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={quote(','.join(chunk), safe=',=^.-')}"
            r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if not r.ok:
                continue
            results = (((r.json() or {}).get("quoteResponse") or {}).get("result") or [])
            for row in results:
                sym = row.get("symbol")
                cap = row.get("marketCap")
                if sym and cap is not None and np.isfinite(cap) and float(cap) > 0:
                    out[sym] = float(cap)
        except Exception:
            pass
    return out


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
}


# -------------------------------------------------
# QUAD ENGINE
# -------------------------------------------------
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


@st.cache_data(ttl=60 * 30, show_spinner=False)
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

    g_indpro = robust_z(_annualized_n(indpro, 3) - indpro.pct_change(12), 36)
    g_sales = robust_z(_annualized_n(rsafs, 3) - rsafs.pct_change(12), 36)
    g_jobs = robust_z((payems.diff(3) / 3.0) - (payems.diff(12) / 12.0), 36)
    g_unrate = -robust_z(unrate.diff(3), 36)
    g_claims = -robust_z(icsa.rolling(4).mean() - icsa.rolling(26).mean(), 52)
    growth_inputs = [g_indpro, g_sales, g_jobs, g_unrate, g_claims]
    growth_weights = [0.22, 0.22, 0.22, 0.18, 0.16]

    i_cpi = robust_z(_annualized_n(cpi, 3) - cpi.pct_change(12), 36)
    i_core = robust_z(_annualized_n(core_cpi, 3) - core_cpi.pct_change(12), 36)
    i_ppi = robust_z(_annualized_n(ppi, 3) - ppi.pct_change(12), 36)
    wti_m = _monthly_last(SER["WTI"])
    hy = SER["HY"].dropna()
    nfci = SER["NFCI"].dropna()
    sahm = SER["SAHM"].dropna()
    oil_3m = robust_z(_annualized_n(wti_m, 3), 36)
    oil_1m = robust_z(_annualized_n(wti_m, 1), 36)

    infl_inputs = [i_cpi, i_core, i_ppi, oil_3m, oil_1m]
    infl_official = [i_cpi, i_core, i_ppi]
    infl_official_weights = [0.36, 0.40, 0.24]
    infl_now_weights = [0.28, 0.32, 0.18, 0.14, 0.08]

    stress_inputs = [robust_z(sahm, 36), robust_z(nfci, 52), robust_z(hy, 52)]

    growth_neg_breadth = float(np.mean([x < 0 for x in growth_inputs]))
    growth_pos_breadth = float(np.mean([x > 0 for x in growth_inputs]))
    infl_pos_breadth = float(np.mean([x > 0 for x in infl_inputs]))
    infl_off_pos_breadth = float(np.mean([x > 0 for x in infl_official]))

    labor_weak = _weighted_mean(
        [max(0.0, -g_jobs), max(0.0, -g_unrate), max(0.0, -g_claims)],
        [0.45, 0.30, 0.25],
    )

    g_off_raw = _weighted_mean(growth_inputs, growth_weights)
    i_off_raw = _weighted_mean(infl_official, infl_official_weights)
    i_now_raw = _weighted_mean(infl_inputs, infl_now_weights)
    s_m = float(np.nanmean(stress_inputs))

    growth_drag = 0.24 * max(0.0, growth_neg_breadth - 0.35) + 0.28 * max(0.0, labor_weak - 0.06)
    g_off = g_off_raw - 0.12 * growth_drag
    g_now = g_off_raw - growth_drag
    i_off = i_off_raw + 0.10 * max(0.0, infl_off_pos_breadth - 0.50)
    i_now = i_now_raw + 0.20 * max(0.0, infl_pos_breadth - 0.50) + 0.08 * max(0.0, oil_1m)

    g_up_off = sigmoid(1.35 * (g_off - 0.05) - 0.08 * max(0.0, s_m))
    i_up_off = sigmoid(1.35 * (i_off + 0.02) + 0.06 * max(0.0, s_m))
    g_up_now = sigmoid(1.85 * (g_now - 0.12) - 0.18 * max(0.0, s_m) - 0.14 * growth_neg_breadth)
    i_up_now = sigmoid(1.65 * (i_now + 0.04) + 0.10 * max(0.0, s_m))

    official_probs = _quad_probs(g_up_off, i_up_off)
    directional_probs = _quad_probs(g_up_now, i_up_now)

    inflation_push = max(0.0, i_now)
    stag_pressure = clamp01(
        0.45 * sigmoid(2.2 * (labor_weak - 0.08))
        + 0.35 * sigmoid(2.0 * (inflation_push - 0.06))
        + 0.20 * sigmoid(4.0 * (growth_neg_breadth - 0.52))
    )

    q2_veto = stag_pressure > 0.46 and growth_neg_breadth >= 0.55 and inflation_push > -0.02
    if q2_veto and directional_probs["Q2"] >= directional_probs["Q3"]:
        shift = min(direction_probs := directional_probs["Q2"] * 0.78, directional_probs["Q2"] * (0.28 + 0.38 * stag_pressure))
        directional_probs["Q2"] -= shift
        directional_probs["Q3"] += shift * 0.92
        directional_probs["Q4"] += shift * 0.08 * float(inflation_push < 0.08)
        directional_probs = _renorm_probs(directional_probs)

    def _pair_score(asset_ticker: str, bench_ticker: str = "SPY") -> float:
        asset = yahoo_close(asset_ticker, "1y")
        bench = yahoo_close(bench_ticker, "1y")
        if asset.empty or bench.empty:
            return np.nan
        df = pd.concat([asset.rename("asset"), bench.rename("bench")], axis=1).sort_index().ffill().dropna()
        if len(df) < 40:
            return np.nan
        rel = (df["asset"] / df["bench"]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(rel) < 40:
            return np.nan
        sig = 0.55 * ret_n(rel, 21) + 0.30 * ret_n(rel, 63) + 0.15 * stretch_n(rel, 63)
        return float(math.tanh(sig / 0.08))

    def _abs_score(ticker: str) -> float:
        s = yahoo_close(ticker, "1y")
        if s.empty or len(s) < 40:
            return np.nan
        sig = 0.55 * ret_n(s, 21) + 0.30 * ret_n(s, 63) + 0.15 * stretch_n(s, 63)
        return float(math.tanh(sig / 0.08))

    growth_cross_inputs = [
        _pair_score("IWM", "SPY"),
        _pair_score("XLY", "XLP"),
        _pair_score("XLI", "XLU"),
        _pair_score("XLF", "XLU"),
        _pair_score("HYG", "IEF"),
    ]
    growth_cross_weights = [0.24, 0.24, 0.18, 0.16, 0.18]

    infl_cross_inputs = [
        _pair_score("XLE", "SPY"),
        _pair_score("GLD", "SPY"),
        _pair_score("DBC", "SPY"),
        _abs_score("UUP"),
        -_abs_score("TLT"),
    ]
    infl_cross_weights = [0.28, 0.22, 0.22, 0.12, 0.16]

    g_cross = _weighted_mean(growth_cross_inputs, growth_cross_weights)
    i_cross = _weighted_mean(infl_cross_inputs, infl_cross_weights)
    cross_growth_neg_breadth = float(np.mean([x < -0.02 for x in growth_cross_inputs if np.isfinite(x)])) if any(np.isfinite(x) for x in growth_cross_inputs) else 0.0
    cross_growth_pos_breadth = float(np.mean([x > 0.02 for x in growth_cross_inputs if np.isfinite(x)])) if any(np.isfinite(x) for x in growth_cross_inputs) else 0.0
    cross_infl_pos_breadth = float(np.mean([x > 0.02 for x in infl_cross_inputs if np.isfinite(x)])) if any(np.isfinite(x) for x in infl_cross_inputs) else 0.0

    g_live = 0.42 * g_now + 0.58 * g_cross - 0.10 * growth_drag
    i_live = 0.55 * i_now + 0.45 * i_cross + 0.04 * max(0.0, oil_1m)

    g_up_live = sigmoid(1.95 * (g_live - 0.04) - 0.10 * max(0.0, s_m) - 0.14 * cross_growth_neg_breadth)
    i_up_live = sigmoid(1.80 * (i_live + 0.01) + 0.06 * max(0.0, s_m) + 0.12 * cross_infl_pos_breadth)
    live_probs = _quad_probs(g_up_live, i_up_live)

    q3_live_pressure = clamp01(
        0.30 * sigmoid(2.4 * (stag_pressure - 0.40))
        + 0.30 * sigmoid(2.8 * (-g_cross - 0.02))
        + 0.25 * sigmoid(2.4 * (i_cross - 0.00))
        + 0.15 * sigmoid(4.0 * (cross_growth_neg_breadth - 0.45))
    )

    if q3_live_pressure > 0.42 and live_probs["Q2"] >= live_probs["Q3"]:
        shift = min(live_probs["Q2"] * 0.65, live_probs["Q2"] * (0.18 + 0.45 * q3_live_pressure))
        live_probs["Q2"] -= shift
        live_probs["Q3"] += shift * 0.94
        live_probs["Q4"] += shift * 0.06 * float(i_cross < 0)
        live_probs = _renorm_probs(live_probs)

    live_blend = {k: 0.18 * official_probs[k] + 0.30 * directional_probs[k] + 0.52 * live_probs[k] for k in official_probs}
    if q2_veto or q3_live_pressure > 0.45:
        bridge = min(live_blend["Q2"] * 0.55, 0.08 + 0.18 * max(stag_pressure, q3_live_pressure))
        live_blend["Q2"] -= bridge
        live_blend["Q3"] += bridge * 0.92
        live_blend["Q4"] += bridge * 0.08 * float(inflation_push < 0.06)
    live_blend = _renorm_probs(live_blend)

    ranked = sorted(live_blend.items(), key=lambda x: x[1], reverse=True)
    current_q, current_p = ranked[0]
    next_q, next_p = ranked[1]

    agreement = clamp01(
        1.0
        - 0.18 * sum(abs(official_probs[k] - directional_probs[k]) for k in official_probs)
        - 0.18 * sum(abs(official_probs[k] - live_probs[k]) for k in official_probs)
        - 0.24 * sum(abs(directional_probs[k] - live_probs[k]) for k in official_probs)
    )
    confidence = clamp01(0.52 * current_p + 0.22 * agreement + 0.26 * max(ranked[0][1], sorted(directional_probs.values(), reverse=True)[0]))

    phase_strength = clamp01(0.34 * abs(g_live) + 0.34 * abs(i_live) + 0.18 * max(0.0, current_p - 0.25) + 0.14 * q3_live_pressure)
    breadth = 0.34 * growth_pos_breadth + 0.26 * infl_off_pos_breadth + 0.20 * cross_growth_pos_breadth + 0.20 * cross_infl_pos_breadth
    regime_divergence = abs(g_live - g_off) + abs(i_live - i_off)
    fragility = clamp01(
        0.34 * (1 - current_p)
        + 0.26 * (1 - agreement)
        + 0.16 * max(0.0, s_m)
        + 0.14 * min(1.0, regime_divergence / 2.5)
        + 0.10 * abs(current_p - sorted(live_probs.values(), reverse=True)[0])
    )

    margin = current_p - next_p
    if margin < 0.03:
        fragility = clamp01(fragility + 0.08)
    if margin < 0.015:
        confidence = clamp01(confidence - 0.05)

    if current_q == "Q1":
        sub_phase = "Goldilocks / recovery" if fragility < 0.45 else "Recovery but fragile"
    elif current_q == "Q2":
        sub_phase = "Reflation / heating up" if q3_live_pressure < 0.40 else "Hot but cross-asset not fully confirmed"
    elif current_q == "Q3":
        sub_phase = "Live stagflation / cross-asset Q3" if q3_live_pressure > 0.55 else "Stagflation building"
    else:
        sub_phase = "Late Q4 / inflation trying to turn" if sorted(live_probs.items(), key=lambda x: x[1], reverse=True)[0][0] == "Q3" else "Deflationary slowdown / bottoming attempt"

    top_score = clamp01(0.32 * max(0.0, i_live) + 0.18 * phase_strength + 0.16 * fragility + 0.18 * float(current_q in ["Q2", "Q3"]) + 0.16 * q3_live_pressure)
    bottom_score = clamp01(0.34 * float(current_q == "Q4") + 0.18 * max(0.0, -g_live) + 0.18 * (1 - fragility) + 0.15 * float(sorted(official_probs.items(), key=lambda x: x[1], reverse=True)[0][0] == "Q4") + 0.15 * float(sorted(live_probs.items(), key=lambda x: x[1], reverse=True)[0][0] == "Q4"))
    transition_pressure = clamp01(0.32 * fragility + 0.24 * (1 - current_p) + 0.20 * min(1.0, regime_divergence / 2.5) + 0.14 * abs(g_live - i_live) / 3.0 + 0.10 * q3_live_pressure)
    transition_conviction = clamp01(0.52 * transition_pressure + 0.48 * next_p)

    return {
        "monthly": official_probs,
        "quarterly": directional_probs,
        "blend": live_blend,
        "current_q": current_q,
        "current_p": current_p,
        "next_q": next_q,
        "next_p": next_p,
        "official_date": official_dt.strftime("%Y-%m-%d") if official_dt is not None else "n/a",
        "agreement": agreement,
        "confidence": confidence,
        "phase_strength": phase_strength,
        "breadth": breadth,
        "fragility": fragility,
        "sub_phase": sub_phase,
        "top_score": top_score,
        "bottom_score": bottom_score,
        "stress_growth": clamp01(sigmoid(1.25 * (-g_live + 0.05))),
        "stress_infl": clamp01(sigmoid(1.25 * (i_live + 0.02))),
        "stress_liq": clamp01(sigmoid(1.25 * s_m)),
        "transition_prob": transition_conviction,
        "transition_pressure": transition_pressure,
        "margin": margin,
        "signal_quality": signal_quality_label(confidence, agreement, fragility),
        "g_live": g_live,
        "i_live": i_live,
        "growth_neg_breadth": growth_neg_breadth,
        "growth_pos_breadth": growth_pos_breadth,
        "infl_pos_breadth": infl_pos_breadth,
        "cross_growth_pos_breadth": cross_growth_pos_breadth,
        "cross_infl_pos_breadth": cross_infl_pos_breadth,
        "stag_pressure": stag_pressure,
    }


core = compute_core()


# -------------------------------------------------
# PLAYBOOK LAYER
# -------------------------------------------------
STAGE_GUIDE = {
    "Q1": {
        "Early": ("quality growth, semis, consumer beta", "utilities / staples", "duration still helps; breadth should widen"),
        "Mid": ("broad equities, software, discretionary", "bond proxies", "earnings upgrades and clean breadth"),
        "Late": ("cyclicals still okay, but froth rises", "defensives still lag", "watch Q2 heat or false-dawn failure"),
    },
    "Q2": {
        "Early": ("small caps, cyclicals, industrials, commodity FX", "duration, staples, utilities", "rates rise orderly and breadth broadens"),
        "Mid": ("financials, materials, broad beta, reflation", "REITs / bond proxies", "credit stable; USD not too brutal"),
        "Late": ("energy, value, nominal-growth trades", "long-duration beta if yields spike", "top risk rises; watch bad-Q2 / rollover"),
    },
    "Q3": {
        "Early": ("gold, miners, energy, USD", "small caps, EMFX, weak cyclicals", "inflation re-accelerates before growth fully breaks"),
        "Mid": ("gold, defensives, selective energy, quality", "broad beta, lower-quality cyclicals", "breadth stays narrow and USD/yields matter"),
        "Late": ("gold + duration barbell, quality defensives", "crowded beta, fragile reflation longs", "branch point: cleaner reflation or hard slide into Q4"),
    },
    "Q4": {
        "Early": ("duration, defensives, USD", "cyclicals, small caps, weak credit", "growth scare starts dominating"),
        "Mid": ("duration, quality, defensives", "broad beta / lower quality", "bottoming is not confirmed yet"),
        "Late": ("duration, selective gold, early cyclicals only if breadth repairs", "junky beta", "watch Q4→Q1 true bottom vs false dawn"),
    },
}

ASSET_SCORE_BY_QUAD = {
    "US cyclicals": {"Q1": 0.55, "Q2": 0.85, "Q3": -0.55, "Q4": -0.45},
    "US defensives": {"Q1": -0.25, "Q2": -0.45, "Q3": 0.70, "Q4": 0.75},
    "US small caps": {"Q1": 0.50, "Q2": 0.95, "Q3": -0.75, "Q4": -0.65},
    "EM equities": {"Q1": 0.45, "Q2": 0.75, "Q3": -0.40, "Q4": -0.25},
    "IHSG cyclicals": {"Q1": 0.35, "Q2": 0.70, "Q3": -0.20, "Q4": -0.20},
    "Gold / miners": {"Q1": -0.10, "Q2": 0.00, "Q3": 0.90, "Q4": 0.35},
    "Oil / energy": {"Q1": 0.05, "Q2": 0.55, "Q3": 0.80, "Q4": -0.15},
    "USD": {"Q1": -0.40, "Q2": 0.05, "Q3": 0.80, "Q4": 0.55},
    "Duration / bonds": {"Q1": -0.30, "Q2": -0.85, "Q3": 0.10, "Q4": 0.90},
    "BTC / crypto beta": {"Q1": 0.60, "Q2": 0.75, "Q3": -0.35, "Q4": -0.10},
}


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
]

US_THEME_TAGS = {
    "Quantum / speculative tech": {"IONQ","QBTS","RGTI","QUBT","QMCO","ARQQ"},
    "Semis / AI infra": {"NVDA","AVGO","AMD","MU","MRVL","ANET","ARM","TSM","SMCI","QCOM"},
    "Software / AI apps": {"MSFT","ORCL","PLTR","CRWD","PANW","NET","DDOG","SNOW","MDB","ZS"},
    "Energy": {"XOM","CVX","COP","SLB","EOG","FANG","MPC","VLO","OXY","HAL"},
    "Defense / industrial": {"RTX","LMT","NOC","GD","CAT","GE","GEV","DE","ETN","PH","AXON"},
    "Financials": {"JPM","GS","MS","BAC","WFC","BRK-B","V","MA","SPGI","KKR"},
    "Gold / metals": {"GLD","NEM","AEM","FCX","SCCO","NUE","STLD","CLF","AA","X"},
    "Consumer / quality": {"WMT","COST","PG","KO","PEP","MCD","CMG","HD","LOW","TJX","AAPL"},
    "Health care": {"LLY","ABBV","JNJ","UNH","ISRG","BSX","ABT","PFE","VRTX","MRK"},
}

IHSG_THEME_TAGS = {
    "Banks / large cap": {"BBCA.JK","BBRI.JK","BMRI.JK","BBNI.JK"},
    "Commodities / mining": {"ANTM.JK","MDKA.JK","INCO.JK","ADRO.JK","PTBA.JK","ITMG.JK","INDY.JK"},
    "Oil / gas / shipping": {"MEDC.JK","AKRA.JK","ESSA.JK","ELSA.JK","HUMI.JK","GTSI.JK","RAJA.JK"},
    "Property / beta": {"CTRA.JK","BSDE.JK","PWON.JK","SMRA.JK","MTLA.JK","DMAS.JK","TRIN.JK","TRUE.JK"},
    "Infra / industrial": {"AMMN.JK","BREN.JK","TPIA.JK","BRPT.JK","PGEO.JK"},
    "Consumer / telco": {"TLKM.JK","EXCL.JK","ISAT.JK","ICBP.JK","INDF.JK","CPIN.JK","AMRT.JK","MAPI.JK","ACES.JK","ERAA.JK"},
}

FX_UNIVERSE = [
    "UUP","FXE","FXY","FXF","FXA","FXC","FXB","CYB","CEW",
    "EURUSD=X","GBPUSD=X","AUDUSD=X","NZDUSD=X","JPY=X","CHF=X","CAD=X","IDR=X"
]

COMMODITY_UNIVERSE = [
    "GLD","SLV","GDX","USO","UNG","DBC","DBA","CPER","DBB","CORN","WEAT","SOYB"
]

CRYPTO_UNIVERSE = [
    "BTC-USD","ETH-USD","SOL-USD","XRP-USD","BNB-USD","ADA-USD","LINK-USD","AVAX-USD","DOGE-USD","LTC-USD"
]

FX_THEME_TAGS = {
    "USD / safety": {"UUP","JPY=X","CHF=X"},
    "Europe majors": {"FXE","FXB","EURUSD=X","GBPUSD=X","FXF"},
    "Commodity FX": {"FXA","FXC","AUDUSD=X","NZDUSD=X","CAD=X"},
    "Asia / EM FX": {"CYB","CEW","IDR=X"},
}

COMMODITY_THEME_TAGS = {
    "Precious metals": {"GLD","SLV","GDX"},
    "Energy": {"USO","UNG"},
    "Broad commodity beta": {"DBC","DBA","WEAT","CORN","SOYB"},
    "Base metals": {"CPER","DBB"},
}

CRYPTO_THEME_TAGS = {
    "BTC / majors": {"BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","LTC-USD"},
    "Alt beta": {"ADA-USD","LINK-USD","AVAX-USD","DOGE-USD"},
}

DEFAULT_BAG7 = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA"]
DEFAULT_OLD_WALL = ["XOM", "CVX", "JPM", "BAC", "ABBV", "LLY", "CAT", "GE", "WMT", "COST"]


def _theme_from_tags(ticker: str, mapping: Dict[str, set]) -> str:
    for theme, names in mapping.items():
        if ticker in names:
            return theme
    return "Other"


def infer_cycle_stage(q: str, engine: Dict[str, object]) -> tuple[str, float]:
    if q == "Q1":
        maturity = 0.45 * engine["phase_strength"] + 0.30 * engine["top_score"] + 0.25 * engine["transition_pressure"]
    elif q == "Q2":
        maturity = 0.42 * engine["top_score"] + 0.28 * engine["phase_strength"] + 0.20 * engine["transition_pressure"] + 0.10 * engine["fragility"]
    elif q == "Q3":
        maturity = 0.35 * engine["top_score"] + 0.25 * engine["phase_strength"] + 0.20 * engine["transition_pressure"] + 0.20 * engine["fragility"]
    else:
        maturity = 0.40 * engine["bottom_score"] + 0.22 * engine["transition_pressure"] + 0.18 * engine["phase_strength"] + 0.20 * engine["fragility"]
    if maturity < 0.36:
        return "Early", float(maturity)
    if maturity < 0.67:
        return "Mid", float(maturity)
    return "Late", float(maturity)


def transition_variant(engine: Dict[str, object]) -> str:
    cur, nxt = engine["current_q"], engine["next_q"]
    if cur == "Q3" and nxt == "Q2":
        bad = engine["stress_infl"] > 0.58 and engine["stress_liq"] > 0.45 and engine["top_score"] > 0.55
        return "Bad reflation" if bad else "Good reflation"
    if cur == "Q2":
        return "Crash-prone Q2" if (engine["top_score"] > 0.56 and engine["stress_infl"] > 0.56 and engine["fragility"] > 0.42) else "Healthy Q2"
    if cur == "Q4" and nxt == "Q1":
        return "False dawn" if engine["agreement"] < 0.55 or engine["breadth"] < 0.42 else "True bottoming"
    if cur == "Q2" and nxt == "Q3":
        return "Overheating rollover"
    if cur == "Q4" and nxt == "Q3":
        return "Inflation shock turn"
    return "Base path"


def current_vs_next_playbook(engine: Dict[str, object]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    family_score = {
        "Q1": {"duration": 0.0, "usd": -0.4, "gold": -0.2, "beta": 0.8, "cyclical": 0.7},
        "Q2": {"duration": -0.8, "usd": 0.0, "gold": 0.0, "beta": 0.6, "cyclical": 0.9},
        "Q3": {"duration": 0.2, "usd": 0.7, "gold": 0.9, "beta": -0.7, "cyclical": -0.8},
        "Q4": {"duration": 0.9, "usd": 0.6, "gold": 0.4, "beta": -0.5, "cyclical": -0.6},
    }
    family_to_assets = {
        "duration": {"US Stocks": ["duration-sensitive quality", "defensives"], "Impact": ["bond-sensitive quality"], "Crypto": ["BTC over alts"]},
        "usd": {"US Stocks": ["selective exporters"], "Impact": ["USD tailwind names"], "Crypto": ["pressure on weaker beta"]},
        "gold": {"US Stocks": ["gold miners"], "Impact": ["gold / hedge bucket"], "Crypto": ["less beta than alts"]},
        "beta": {"US Stocks": ["small caps / cyclicals"], "Impact": ["broad beta leaders"], "Crypto": ["alts"]},
        "cyclical": {"US Stocks": ["industrials", "materials"], "Impact": ["cyclical leadership"], "Crypto": ["risk-on rotation"]},
    }
    cur_scores = family_score[engine["current_q"]]
    nxt_scores = family_score[engine["next_q"]]
    cur_sorted = sorted(cur_scores, key=cur_scores.get, reverse=True)
    nxt_sorted = sorted(nxt_scores, key=nxt_scores.get, reverse=True)
    cur_out, nxt_out = {}, {}
    for bucket_name in ["US Stocks", "Impact", "Crypto"]:
        cur_assets, nxt_assets = [], []
        for fam in cur_sorted[:2]:
            cur_assets.extend(family_to_assets[fam][bucket_name][:1])
        for fam in nxt_sorted[:2]:
            nxt_assets.extend(family_to_assets[fam][bucket_name][:1])
        cur_out[bucket_name] = cur_assets
        nxt_out[bucket_name] = nxt_assets
    return cur_out, nxt_out


def current_crash_probability(engine: Dict[str, object]) -> float:
    base = {"Q1": 0.22, "Q2": 0.42, "Q3": 0.61, "Q4": 0.74}.get(engine["current_q"], 0.50)
    variant = transition_variant(engine)
    adj = 0.0
    if variant in ["Bad reflation", "Crash-prone Q2", "Overheating rollover", "Inflation shock turn"]:
        adj += 0.08
    elif variant in ["Good reflation", "Healthy Q2", "True bottoming"]:
        adj -= 0.06
    elif variant in ["False dawn"]:
        adj += 0.04
    stress_mix = (
        0.24 * engine["stress_liq"]
        + 0.20 * engine["stress_infl"]
        + 0.16 * (1 - engine["breadth"])
        + 0.16 * engine["top_score"]
        + 0.12 * engine["transition_pressure"]
        + 0.12 * engine["fragility"]
    )
    score = base + (stress_mix - 0.50) * 0.55 + adj
    return clamp01(score)


def compute_risk_meters(engine: Dict[str, object]) -> Dict[str, float]:
    big_crash = current_crash_probability(engine)
    risk_off = clamp01(
        0.26 * (1 - engine["breadth"])
        + 0.16 * engine["fragility"]
        + 0.14 * engine["transition_pressure"]
        + 0.12 * engine["stress_infl"]
        + 0.10 * engine["stress_liq"]
        + 0.12 * float(engine["current_q"] in ["Q3", "Q4"])
        + 0.10 * big_crash
    )
    risk_on = clamp01(
        0.28 * engine["breadth"]
        + 0.18 * (1 - engine["fragility"])
        + 0.16 * (1 - engine["transition_pressure"])
        + 0.14 * float(engine["current_q"] in ["Q1", "Q2"])
        + 0.12 * max(0.0, 1 - engine["stress_infl"])
        + 0.12 * max(0.0, 1 - big_crash)
    )
    return {"risk_on": risk_on, "risk_off": risk_off, "big_crash": big_crash}


# -------------------------------------------------
# LEADERSHIP / LONG-SHORT SCORE
# -------------------------------------------------
def _lead_state(alpha21: float, alpha63: float, alpha126: float, trend: float) -> str:
    if alpha21 < -0.06 and alpha63 < -0.10 and trend < 0.45:
        return "Weak"
    if alpha21 < 0 and alpha63 > 0.04 and trend >= 0.50:
        return "Fading"
    if alpha21 > 0.05 and (alpha63 > 0.02 or trend >= 0.62):
        return "Emerging"
    if alpha126 > 0.08 and alpha21 > -0.01:
        return "Leader"
    return "Neutral"


def _lead_comment(state: str, alpha21: float, alpha63: float, trend: float) -> str:
    if state == "Leader":
        return "clear relative leader" if alpha21 > 0.06 and alpha63 > 0.10 else "steady relative strength"
    if state == "Emerging":
        return "starting to outperform" if alpha21 > alpha63 + 0.04 else "early RS improvement"
    if state == "Fading":
        return "still above benchmark, but momentum fading"
    if state == "Weak":
        return "underperforming benchmark"
    return "mixed / no clean edge"


@st.cache_data(ttl=60 * 20, show_spinner=False)
def rank_market_leaders(tickers: Tuple[str, ...], benchmark_ticker: str, period: str = "1y", fallback_benchmark: str | None = None) -> pd.DataFrame:
    tickers = [t for t in tickers if t]
    if not tickers:
        return pd.DataFrame()

    fetch = [benchmark_ticker] + tickers
    if fallback_benchmark:
        fetch.append(fallback_benchmark)
    prices = yahoo_close_batch(fetch, period)

    enough_cols = set(prices.columns) if not prices.empty else set()
    if prices.empty or len(enough_cols.intersection(set(tickers))) < max(3, min(8, max(1, len(tickers) // 4))):
        frames = []
        for ticker in fetch:
            s = yahoo_close(ticker, period)
            if not s.empty:
                frames.append(s.rename(ticker))
        prices = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
    if prices.empty:
        return pd.DataFrame()

    prices = prices.sort_index().ffill().dropna(how="all")

    def _pick_benchmark(name: str | None) -> pd.Series:
        if not name:
            return pd.Series(dtype=float)
        s = _get_col_as_series(prices, name).dropna()
        return s if len(s) >= 25 else pd.Series(dtype=float)

    bench_name = benchmark_ticker
    bench = _pick_benchmark(benchmark_ticker)
    if bench.empty and fallback_benchmark:
        cand = _pick_benchmark(fallback_benchmark)
        if not cand.empty:
            bench = cand
            bench_name = fallback_benchmark

    if bench.empty:
        series = []
        for ticker in tickers:
            s = _get_col_as_series(prices, ticker).dropna()
            if len(s) < 25:
                continue
            series.append((s / float(s.iloc[0])).rename(ticker))
        if series:
            basket = pd.concat(series, axis=1).dropna(how="all").mean(axis=1).dropna()
            if not basket.empty:
                bench = basket.rename("bench")
                bench_name = "INTERNAL_BASKET"
    if bench.empty:
        return pd.DataFrame()
    bench = _coerce_series(bench, "bench")

    rows = []
    min_len = 45 if period == "1y" else 25
    for ticker in tickers:
        s = _get_col_as_series(prices, ticker).dropna()
        if s.empty:
            continue
        df = pd.concat([_coerce_series(s, "asset"), _coerce_series(bench, "bench")], axis=1).sort_index().ffill().dropna()
        if len(df) < min_len or "asset" not in df.columns or "bench" not in df.columns:
            continue
        asset = _coerce_series(df["asset"], "asset")
        bm = _coerce_series(df["bench"], "bench")
        if len(asset) < min_len or len(bm) < min_len:
            continue
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
        rows.append(
            {
                "Ticker": ticker,
                "Benchmark": bench_name,
                "State": state,
                "RSScore": rs_score,
                "StartScore": start_score,
                "Alpha21": alpha21,
                "Alpha63": alpha63,
                "Alpha126": alpha126,
                "Ret21": r21,
                "Ret63": r63,
                "Trend": trend,
                "Comment": comment,
                "Bars": len(df),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["RSScore", "StartScore", "Alpha21"], ascending=False).reset_index(drop=True)


def region_theme_mapping(region: str) -> Dict[str, set]:
    return {
        "US": US_THEME_TAGS,
        "IHSG": IHSG_THEME_TAGS,
        "Forex": FX_THEME_TAGS,
        "Commodities": COMMODITY_THEME_TAGS,
        "Crypto": CRYPTO_THEME_TAGS,
    }.get(region, {})


def macro_bucket_for_ticker(ticker: str, theme: str, region: str) -> str:
    if region == "IHSG":
        if theme in ["Commodities / mining", "Oil / gas / shipping"]:
            return "Oil / energy"
        if theme in ["Banks / large cap", "Property / beta", "Infra / industrial"]:
            return "IHSG cyclicals"
        return "EM equities"
    if region == "Forex":
        if theme in ["USD / safety"]:
            return "USD"
        if theme in ["Europe majors"]:
            return "Duration / bonds" if ticker in {"FXF", "CHF=X"} else "EM equities"
        if theme in ["Commodity FX"]:
            return "US small caps"
        return "EM equities"
    if region == "Commodities":
        if theme in ["Precious metals"]:
            return "Gold / miners"
        if theme in ["Energy"]:
            return "Oil / energy"
        if theme in ["Base metals", "Broad commodity beta"]:
            return "US cyclicals"
        return "US cyclicals"
    if region == "Crypto":
        return "BTC / crypto beta"
    if theme in ["Energy"]:
        return "Oil / energy"
    if theme in ["Gold / metals"]:
        return "Gold / miners"
    if theme in ["Health care", "Consumer / quality"]:
        return "US defensives"
    if theme in ["Financials", "Defense / industrial"]:
        return "US cyclicals"
    if theme in ["Semis / AI infra", "Software / AI apps", "Quantum / speculative tech"]:
        return "US small caps" if ticker not in {"AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA"} else "US cyclicals"
    return "US cyclicals"


def quad_fit_score(bucket_name: str, engine: Dict[str, object]) -> float:
    """Map a ticker macro bucket to a regime-fit score in [0, 1].

    This is intentionally dashboard-facing and coarse: it converts the hidden
    quad engine into tradeable family preferences without exposing raw internals.
    Higher means the current/near-next regime is supportive for longs.
    """
    current_q = str(engine.get("current_q", "Q3"))
    next_q = str(engine.get("next_q", current_q))
    transition_prob = float(engine.get("transition_prob", 0.0))
    variant = str(transition_variant(engine)) if isinstance(engine, dict) else "Base path"

    fit_map = {
        "Q1": {
            "Gold / miners": 0.42,
            "Oil / energy": 0.40,
            "US defensives": 0.50,
            "US cyclicals": 0.72,
            "US small caps": 0.86,
            "IHSG cyclicals": 0.72,
            "EM equities": 0.66,
            "USD": 0.28,
            "Duration / bonds": 0.70,
            "BTC / crypto beta": 0.82,
        },
        "Q2": {
            "Gold / miners": 0.38,
            "Oil / energy": 0.78,
            "US defensives": 0.34,
            "US cyclicals": 0.84,
            "US small caps": 0.90,
            "IHSG cyclicals": 0.80,
            "EM equities": 0.76,
            "USD": 0.40,
            "Duration / bonds": 0.18,
            "BTC / crypto beta": 0.76,
        },
        "Q3": {
            "Gold / miners": 0.92,
            "Oil / energy": 0.68,
            "US defensives": 0.80,
            "US cyclicals": 0.26,
            "US small caps": 0.18,
            "IHSG cyclicals": 0.28,
            "EM equities": 0.36,
            "USD": 0.84,
            "Duration / bonds": 0.42,
            "BTC / crypto beta": 0.24,
        },
        "Q4": {
            "Gold / miners": 0.56,
            "Oil / energy": 0.22,
            "US defensives": 0.72,
            "US cyclicals": 0.16,
            "US small caps": 0.12,
            "IHSG cyclicals": 0.18,
            "EM equities": 0.24,
            "USD": 0.76,
            "Duration / bonds": 0.94,
            "BTC / crypto beta": 0.14,
        },
    }

    current_fit = fit_map.get(current_q, {}).get(bucket_name, 0.50)
    next_fit = fit_map.get(next_q, {}).get(bucket_name, current_fit)
    blend_w = clamp01(0.18 + 0.55 * transition_prob)
    score = (1.0 - blend_w) * current_fit + blend_w * next_fit

    # Variant nudges for the transition path the dashboard is already surfacing.
    if variant == "Bad reflation":
        if bucket_name in {"US cyclicals", "US small caps", "IHSG cyclicals", "EM equities", "BTC / crypto beta"}:
            score -= 0.10
        if bucket_name in {"Gold / miners", "US defensives", "USD"}:
            score += 0.05
    elif variant in {"Good reflation", "Healthy Q2"}:
        if bucket_name in {"US cyclicals", "US small caps", "IHSG cyclicals", "EM equities", "BTC / crypto beta", "Oil / energy"}:
            score += 0.05
        if bucket_name in {"USD", "Duration / bonds"}:
            score -= 0.04
    elif variant == "True bottoming":
        if bucket_name in {"US small caps", "US cyclicals", "EM equities", "BTC / crypto beta"}:
            score += 0.04
    elif variant in {"False dawn", "Crash-prone Q2", "Overheating rollover", "Inflation shock turn"}:
        if bucket_name in {"US cyclicals", "US small caps", "IHSG cyclicals", "EM equities", "BTC / crypto beta"}:
            score -= 0.06
        if bucket_name in {"USD", "Duration / bonds", "US defensives"}:
            score += 0.04

    return clamp01(float(score))


def score_ticker_table(df: pd.DataFrame, region: str, engine: Dict[str, object]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    mapping = region_theme_mapping(region)
    work = df.copy()
    work["Theme"] = work["Ticker"].apply(lambda x: _theme_from_tags(x, mapping) if mapping else "Other")
    work["Cluster"] = work["Theme"]
    work["MacroBucket"] = work.apply(lambda r: macro_bucket_for_ticker(r["Ticker"], r["Theme"], region), axis=1)
    work["MacroFit"] = work["MacroBucket"].apply(lambda x: quad_fit_score(x, engine))
    work["Weakness"] = work["Alpha21"].apply(lambda x: clamp01(_norm_tanh(-x, 0.08)))
    work["Strength"] = work["Alpha21"].apply(lambda x: clamp01(_norm_tanh(x, 0.08)))
    work["LongScore"] = 100 * (
        0.40 * work["MacroFit"]
        + 0.35 * work["RSScore"]
        + 0.15 * work["StartScore"]
        + 0.10 * work["Trend"]
    ) * (0.88 + 0.12 * (1 - engine["fragility"]))
    work["ShortScore"] = 100 * (
        0.42 * (1 - work["MacroFit"])
        + 0.30 * work["Weakness"]
        + 0.18 * (1 - work["Trend"])
        + 0.10 * (1 - work["RSScore"])
    ) * (0.92 + 0.08 * engine["risk_pack"]["risk_off"])
    work["LongScore"] = work["LongScore"].clip(0, 100)
    work["ShortScore"] = work["ShortScore"].clip(0, 100)

    def _bias(row: pd.Series) -> str:
        if row["LongScore"] >= 72 and row["State"] in ["Leader", "Emerging"]:
            return "Long candidate"
        if row["ShortScore"] >= 70 and row["State"] in ["Weak", "Fading"]:
            return "Short candidate"
        if row["LongScore"] >= 62:
            return "Long watch"
        if row["ShortScore"] >= 62:
            return "Short watch"
        return "Neutral"

    work["Bias"] = work.apply(_bias, axis=1)
    return work.sort_values(["LongScore", "RSScore", "StartScore"], ascending=False).reset_index(drop=True)


# -------------------------------------------------
# IMPACT BOARD
# -------------------------------------------------
@st.cache_data(ttl=60 * 15, show_spinner=False)
def build_impact_frame(tickers: Tuple[str, ...], period: str = "3mo") -> pd.DataFrame:
    tickers = tuple([t for t in tickers if t])
    if not tickers:
        return pd.DataFrame()
    prices = yahoo_close_batch(list(tickers), period=period)
    if prices.empty:
        frames = []
        for t in tickers:
            s = yahoo_close(t, period=period)
            if not s.empty:
                frames.append(s.rename(t))
        prices = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    if prices.empty:
        return pd.DataFrame()
    caps = get_market_caps(tickers)
    rows = []
    for t in tickers:
        s = _get_col_as_series(prices, t).dropna()
        if len(s) < 2:
            continue
        prev_close = float(s.iloc[-2])
        close = float(s.iloc[-1])
        daily_ret = (close / prev_close) - 1.0 if prev_close else np.nan
        market_cap = caps.get(t)
        if market_cap is None or not np.isfinite(market_cap):
            continue
        rows.append(
            {
                "Ticker": t,
                "PrevClose": prev_close,
                "Close": close,
                "DailyRet": daily_ret,
                "MarketCap": float(market_cap),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    total_cap = float(df["MarketCap"].sum())
    if total_cap <= 0:
        return pd.DataFrame()
    df["Weight"] = df["MarketCap"] / total_cap
    df["ImpactB"] = (df["MarketCap"] * df["DailyRet"]) / 1e9
    df["IndexContributionPct"] = df["Weight"] * df["DailyRet"] * 100
    df["AbsImpactB"] = df["ImpactB"].abs()
    return df.sort_values("ImpactB", ascending=False).reset_index(drop=True)


def summarize_impact(df: pd.DataFrame, bag7: List[str], old_wall: List[str]) -> Dict[str, float]:
    if df is None or df.empty:
        return {}
    adv = float((df["DailyRet"] > 0).mean())
    eq_ret = float(df["DailyRet"].mean())
    cap_ret = float(df["IndexContributionPct"].sum())
    top3 = df["AbsImpactB"].nlargest(min(3, len(df))).sum()
    total_abs = df["AbsImpactB"].sum()
    concentration = float(top3 / total_abs) if total_abs > 0 else 0.0
    bag7_impact = float(df[df["Ticker"].isin(bag7)]["ImpactB"].sum())
    old_wall_impact = float(df[df["Ticker"].isin(old_wall)]["ImpactB"].sum())
    return {
        "advancers": adv,
        "equal_weight_return": eq_ret,
        "cap_weight_return": cap_ret,
        "concentration": concentration,
        "bag7_impact": bag7_impact,
        "old_wall_impact": old_wall_impact,
    }


def theme_impact_rows(df: pd.DataFrame, mapping: Dict[str, set], n: int = 8) -> List[List[str]]:
    if df is None or df.empty:
        return [["No data", "-", "-", "-"]]
    work = df.copy()
    work["Theme"] = work["Ticker"].apply(lambda x: _theme_from_tags(x, mapping))
    agg = (
        work.groupby("Theme", as_index=False)
        .agg(
            ImpactB=("ImpactB", "sum"),
            AvgRet=("DailyRet", "mean"),
            Weight=("Weight", "sum"),
        )
        .sort_values("ImpactB", ascending=False)
    )
    out = []
    for _, r in agg.head(n).iterrows():
        out.append([r["Theme"], f"{r['ImpactB']:+.1f}B", f"{100*r['AvgRet']:+.2f}%", pct(float(r["Weight"]))])
    return out


# -------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------
st.sidebar.markdown("## Dashboard Controls")
region_mode = st.sidebar.selectbox("Ticker score region", ["US", "IHSG", "Forex", "Commodities", "Crypto", "All markets"], index=5)

us_custom = st.sidebar.text_input("Extra US tickers", value="")
ihsg_custom = st.sidebar.text_input("Extra IHSG tickers", value="")
fx_custom = st.sidebar.text_input("Extra forex tickers", value="")
commod_custom = st.sidebar.text_input("Extra commodities tickers", value="")
crypto_custom = st.sidebar.text_input("Extra crypto tickers", value="")

watchlist_us_raw = st.sidebar.text_input("Exact US watchlist", value="AAPL,NVDA,MSFT,PLTR,XOM,GLD")
watchlist_ihsg_raw = st.sidebar.text_input("Exact IHSG watchlist", value="BBCA,BBRI,BREN,AMMN,MEDC,TLKM")
watchlist_fx_raw = st.sidebar.text_input("Exact forex watchlist", value="UUP,FXE,FXY,FXA,EURUSD=X,JPY=X")
watchlist_commod_raw = st.sidebar.text_input("Exact commodities watchlist", value="GLD,SLV,USO,UNG,DBC,CPER")
watchlist_crypto_raw = st.sidebar.text_input("Exact crypto watchlist", value="BTC-USD,ETH-USD,SOL-USD,XRP-USD")

impact_mode = st.sidebar.selectbox("Impact board universe", ["US major universe", "Bag7 + Old Wall", "Custom US list"], index=0)
impact_custom = st.sidebar.text_input("Custom impact tickers", value="AAPL,MSFT,NVDA,META,AMZN,GOOGL,TSLA,XOM,CVX,JPM,ABBV")
bag7_raw = st.sidebar.text_input("Bag7 basket", value=",".join(DEFAULT_BAG7))
old_wall_raw = st.sidebar.text_input("Old Wall basket", value=",".join(DEFAULT_OLD_WALL))
show_count = st.sidebar.slider("Rows per table", min_value=5, max_value=20, value=10)

bag7 = _clean_tickers(bag7_raw)
old_wall = _clean_tickers(old_wall_raw)
extra_us = _clean_tickers(us_custom)
extra_ihsg = _clean_tickers(ihsg_custom, ".JK")
extra_fx = _clean_tickers(fx_custom)
extra_commod = _clean_tickers(commod_custom)
extra_crypto = _clean_tickers(crypto_custom)

watchlist_us = list(dict.fromkeys(_clean_tickers(watchlist_us_raw) + extra_us))
watchlist_ihsg = list(dict.fromkeys(_clean_tickers(watchlist_ihsg_raw, ".JK") + extra_ihsg))
watchlist_fx = list(dict.fromkeys(_clean_tickers(watchlist_fx_raw) + extra_fx))
watchlist_commod = list(dict.fromkeys(_clean_tickers(watchlist_commod_raw) + extra_commod))
watchlist_crypto = list(dict.fromkeys(_clean_tickers(watchlist_crypto_raw) + extra_crypto))

us_universe = list(dict.fromkeys(US_UNIVERSE + extra_us + watchlist_us))
ihsg_universe = list(dict.fromkeys(IHSG_UNIVERSE + extra_ihsg + watchlist_ihsg))
fx_universe = list(dict.fromkeys(FX_UNIVERSE + extra_fx + watchlist_fx))
commod_universe = list(dict.fromkeys(COMMODITY_UNIVERSE + extra_commod + watchlist_commod))
crypto_universe = list(dict.fromkeys(CRYPTO_UNIVERSE + extra_crypto + watchlist_crypto))

if impact_mode == "US major universe":
    impact_universe = tuple(us_universe)
elif impact_mode == "Bag7 + Old Wall":
    impact_universe = tuple(list(dict.fromkeys(bag7 + old_wall)))
else:
    impact_universe = tuple(_clean_tickers(impact_custom))

risk_pack = compute_risk_meters(core)
core["risk_pack"] = risk_pack
cur_stage, cur_maturity = infer_cycle_stage(core["current_q"], core)
variant_now = transition_variant(core)
play_now, play_next = current_vs_next_playbook(core)

# -------------------------------------------------
# DATA PREP
# -------------------------------------------------
impact_df = build_impact_frame(impact_universe)
impact_summary = summarize_impact(impact_df, bag7, old_wall) if not impact_df.empty else {}

us_rank_df = rank_market_leaders(tuple(us_universe), benchmark_ticker="SPY", period="1y", fallback_benchmark="QQQ")
if us_rank_df.empty:
    us_rank_df = rank_market_leaders(tuple(us_universe), benchmark_ticker="SPY", period="6mo", fallback_benchmark="QQQ")
us_score_df = score_ticker_table(us_rank_df, "US", core) if not us_rank_df.empty else pd.DataFrame()

ihsg_rank_df = rank_market_leaders(tuple(ihsg_universe), benchmark_ticker="^JKSE", period="1y", fallback_benchmark="EIDO")
if ihsg_rank_df.empty:
    ihsg_rank_df = rank_market_leaders(tuple(ihsg_universe), benchmark_ticker="EIDO", period="6mo", fallback_benchmark="SPY")
ihsg_score_df = score_ticker_table(ihsg_rank_df, "IHSG", core) if not ihsg_rank_df.empty else pd.DataFrame()

fx_rank_df = rank_market_leaders(tuple(fx_universe), benchmark_ticker="UUP", period="1y", fallback_benchmark="CEW")
if fx_rank_df.empty:
    fx_rank_df = rank_market_leaders(tuple(fx_universe), benchmark_ticker="UUP", period="6mo", fallback_benchmark="CEW")
fx_score_df = score_ticker_table(fx_rank_df, "Forex", core) if not fx_rank_df.empty else pd.DataFrame()

commod_rank_df = rank_market_leaders(tuple(commod_universe), benchmark_ticker="DBC", period="1y", fallback_benchmark="GLD")
if commod_rank_df.empty:
    commod_rank_df = rank_market_leaders(tuple(commod_universe), benchmark_ticker="DBC", period="6mo", fallback_benchmark="GLD")
commod_score_df = score_ticker_table(commod_rank_df, "Commodities", core) if not commod_rank_df.empty else pd.DataFrame()

crypto_rank_df = rank_market_leaders(tuple(crypto_universe), benchmark_ticker="BTC-USD", period="1y", fallback_benchmark="ETH-USD")
if crypto_rank_df.empty:
    crypto_rank_df = rank_market_leaders(tuple(crypto_universe), benchmark_ticker="BTC-USD", period="6mo", fallback_benchmark="ETH-USD")
crypto_score_df = score_ticker_table(crypto_rank_df, "Crypto", core) if not crypto_rank_df.empty else pd.DataFrame()

def coverage_row(label: str, universe: List[str], score_df: pd.DataFrame) -> List[str]:
    scored = set(score_df["Ticker"].tolist()) if score_df is not None and not score_df.empty else set()
    missing = [t for t in universe if t not in scored]
    return [label, f"{len(scored)}/{len(universe)}", ", ".join([m.replace('.JK','') for m in missing[:5]]) if missing else "OK"]

coverage_rows = [
    coverage_row("US", us_universe, us_score_df),
    coverage_row("IHSG", ihsg_universe, ihsg_score_df),
    coverage_row("Forex", fx_universe, fx_score_df),
    coverage_row("Commodities", commod_universe, commod_score_df),
    coverage_row("Crypto", crypto_universe, crypto_score_df),
]

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("Quad • Impact • Signal • Ticker Score")
st.markdown("<div class='small-muted'>Decision-support dashboard: regime first, impact second, execution third. No raw engine internals shown.</div>", unsafe_allow_html=True)

hero = st.columns(7)
hero_items = [
    ("Current Quad", core["current_q"], pill(f"{cur_stage} • {core['sub_phase']}")),
    ("Next Likely", core["next_q"], pill(f"transition {pct(core['transition_prob'])}")),
    ("Confidence", pct(core["confidence"]), pill(f"agreement {pct(core['agreement'])}")),
    ("Risk-On", pct(risk_pack["risk_on"]), pill(bucket(risk_pack["risk_on"], (0.26, 0.56), ("Low", "Elevated", "High")))),
    ("Risk-Off", pct(risk_pack["risk_off"]), pill(bucket(risk_pack["risk_off"], (0.26, 0.56), ("Low", "Elevated", "High")))),
    ("Big Crash", pct(risk_pack["big_crash"]), pill(bucket(risk_pack["big_crash"], (0.25, 0.45), ("Low", "Watch", "Elevated+")))),
    ("Action Bias", "Selective" if core["current_q"] in ["Q3", "Q4"] else "Risk-on selective", pill(variant_now)),
]
for col, (title, value, sub) in zip(hero, hero_items):
    with col:
        st.markdown(
            f"""
            <div class='kpi'>
              <div class='metric-title'>{title}</div>
              <div class='metric-value'>{value}</div>
              <div class='metric-sub'>{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    f"<div class='note-box'><b>Quick read:</b> now {cur_stage} {core['current_q']} with {variant_now.lower()}. Base case favors {', '.join(play_now['US Stocks'])}; if {core['next_q']} takes over, watch for {', '.join(play_next['US Stocks'])}. Breadth {pct(core['breadth'])}, fragility {pct(core['fragility'])}, top risk state: {ladder_state(core['top_score'], 'top')}.</div>",
    unsafe_allow_html=True,
)

# -------------------------------------------------
# QUAD BOARD
# -------------------------------------------------
st.markdown("### Quad Board")
q_left, q_mid, q_right = st.columns([1.2, 1.1, 1.3])
with q_left:
    st.markdown("<div class='card'><div class='section-title'>Current vs Next</div>", unsafe_allow_html=True)
    current_rows = [
        ["Now", f"{cur_stage} {core['current_q']}", STAGE_GUIDE[core['current_q']][cur_stage][0], STAGE_GUIDE[core['current_q']][cur_stage][1]],
        ["If next wins", f"Early {core['next_q']}", STAGE_GUIDE[core['next_q']]['Early'][0], STAGE_GUIDE[core['next_q']]['Early'][1]],
    ]
    st.markdown(table_html(["Window", "Quad", "Usually strong", "Usually weak"], current_rows), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with q_mid:
    st.markdown("<div class='card'><div class='section-title'>Quad Probabilities</div>", unsafe_allow_html=True)
    for quad in ["Q1", "Q2", "Q3", "Q4"]:
        val = float(core["blend"].get(quad, 0.0))
        st.markdown(f"<div class='small-muted'><b>{quad}</b> {pct(val)}</div>", unsafe_allow_html=True)
        st.markdown(bar_html(val), unsafe_allow_html=True)
        st.write("")
    st.markdown("</div>", unsafe_allow_html=True)
with q_right:
    st.markdown("<div class='card'><div class='section-title'>Regime Read</div>", unsafe_allow_html=True)
    regime_rows = [
        ["Variant", variant_now],
        ["Signal quality", core["signal_quality"]],
        ["Maturity", pct(cur_maturity)],
        ["Official macro cutoff", core["official_date"]],
        ["Growth live", num1(core["g_live"])],
        ["Inflation live", num1(core["i_live"])],
    ]
    st.markdown(table_html(["Field", "Read"], regime_rows), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# IMPACT BOARD
# -------------------------------------------------
st.markdown("### Impact Board")
i_left, i_mid, i_right = st.columns([1.15, 1.15, 1.1])
with i_left:
    st.markdown("<div class='card'><div class='section-title'>Largest Positive Impact</div>", unsafe_allow_html=True)
    if impact_df.empty:
        st.info("No impact data available from Yahoo / market cap feed.")
    else:
        pos = impact_df.sort_values("ImpactB", ascending=False).head(show_count)
        rows = [[r["Ticker"], f"{100*r['DailyRet']:+.2f}%", f"{r['ImpactB']:+.1f}B", pct(float(r['Weight']))] for _, r in pos.iterrows()]
        st.markdown(table_html(["Ticker", "1D", "Δ MCap", "Weight"], rows), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with i_mid:
    st.markdown("<div class='card'><div class='section-title'>Largest Negative Impact</div>", unsafe_allow_html=True)
    if impact_df.empty:
        st.info("No impact data available from Yahoo / market cap feed.")
    else:
        neg = impact_df.sort_values("ImpactB", ascending=True).head(show_count)
        rows = [[r["Ticker"], f"{100*r['DailyRet']:+.2f}%", f"{r['ImpactB']:+.1f}B", pct(float(r['Weight']))] for _, r in neg.iterrows()]
        st.markdown(table_html(["Ticker", "1D", "Δ MCap", "Weight"], rows), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with i_right:
    st.markdown("<div class='card'><div class='section-title'>Breadth & Basket Read</div>", unsafe_allow_html=True)
    if not impact_summary:
        st.info("No summary available.")
    else:
        summary_rows = [
            ["Advancers", pct(impact_summary["advancers"])],
            ["Equal-weight return", f"{100*impact_summary['equal_weight_return']:+.2f}%"],
            ["Cap-weight return", f"{impact_summary['cap_weight_return']:+.2f}%"],
            ["Top-3 impact concentration", pct(impact_summary["concentration"])],
            ["Bag7 impact", f"{impact_summary['bag7_impact']:+.1f}B"],
            ["Old Wall impact", f"{impact_summary['old_wall_impact']:+.1f}B"],
        ]
        st.markdown(table_html(["Metric", "Read"], summary_rows), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if not impact_df.empty:
    st.markdown("<div class='card'><div class='section-title'>Theme Impact Rollup</div>", unsafe_allow_html=True)
    st.markdown(table_html(["Theme", "Δ MCap", "Avg 1D", "Weight"], theme_impact_rows(impact_df, US_THEME_TAGS)), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# SIGNAL BOARD
# -------------------------------------------------
st.markdown("### Signal Board")
s_left, s_mid, s_right = st.columns([1.0, 1.0, 1.2])
with s_left:
    st.markdown("<div class='card'><div class='section-title'>Risk Meters</div>", unsafe_allow_html=True)
    risk_rows = [
        ["Risk-On", pct(risk_pack["risk_on"]), bucket(risk_pack["risk_on"], (0.26, 0.56), ("Low", "Elevated", "High"))],
        ["Risk-Off", pct(risk_pack["risk_off"]), bucket(risk_pack["risk_off"], (0.26, 0.56), ("Low", "Elevated", "High"))],
        ["Big Crash", pct(risk_pack["big_crash"]), bucket(risk_pack["big_crash"], (0.25, 0.45), ("Low", "Watch", "Elevated+"))],
    ]
    st.markdown(table_html(["Meter", "Now", "State"], risk_rows), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with s_mid:
    st.markdown("<div class='card'><div class='section-title'>Turn Risk</div>", unsafe_allow_html=True)
    turn_rows = [
        ["Top risk", ladder_state(core["top_score"], "top")],
        ["Bottom risk", ladder_state(core["bottom_score"], "bottom")],
        ["Transition pressure", pct(core["transition_pressure"])],
        ["Fragility", pct(core["fragility"])],
        ["Breadth", pct(core["breadth"])],
        ["Stag pressure", pct(core["stag_pressure"])],
    ]
    st.markdown(table_html(["Field", "Read"], turn_rows), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with s_right:
    st.markdown("<div class='card'><div class='section-title'>Execution Posture</div>", unsafe_allow_html=True)
    posture_rows = [
        ["Base stance", "Keep longs selective" if core["current_q"] in ["Q3", "Q4"] else "Can press leaders selectively"],
        ["What confirms", "breadth improvement, stronger leaders, lower fragility"],
        ["What invalidates", "narrow breadth, rising crash meter, weak leader retention"],
        ["Use impact board for", "who is actually moving the tape and whether it is broad or narrow"],
        ["Use ticker score for", "candidate ranking, not certainty"],
    ]
    st.markdown(table_html(["Focus", "Read"], posture_rows), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# LONG/SHORT TICKER SCORE
# -------------------------------------------------
st.markdown("### Long / Short Ticker Score")

def score_rows_for_display(df: pd.DataFrame, mode: str, n: int = 10) -> List[List[str]]:
    if df is None or df.empty:
        return [["No data", "-", "-", "-", "-", "-"]]
    work = df.copy()
    if mode == "long":
        work = work.sort_values(["LongScore", "RSScore", "StartScore"], ascending=False)
    else:
        work = work.sort_values(["ShortScore", "Alpha21", "Trend"], ascending=[False, True, True])
    out = []
    for _, r in work.head(n).iterrows():
        out.append([
            r["Ticker"].replace(".JK", ""),
            r["Bias"],
            f"{r['LongScore']:.0f}",
            f"{r['ShortScore']:.0f}",
            r.get("Cluster", r.get("Theme", "Other")),
            r["Comment"],
        ])
    return out


def exact_rows_for_display(df: pd.DataFrame, watchlist: List[str], n: int = 12) -> List[List[str]]:
    if not watchlist:
        return [["No watchlist", "-", "-", "-", "-", "-", "-"]]
    if df is None or df.empty:
        return [[t.replace('.JK',''), "No data", "-", "-", "Missing", "-", "No price / benchmark data"] for t in watchlist[:n]]
    out = []
    work = df.set_index("Ticker", drop=False)
    for t in watchlist[:n]:
        if t in work.index:
            r = work.loc[t]
            out.append([
                t.replace('.JK',''), r["Bias"], f"{r['LongScore']:.0f}", f"{r['ShortScore']:.0f}", r["State"], r.get("Cluster", r.get("Theme", "Other")), r["Comment"],
            ])
        else:
            out.append([t.replace('.JK',''), "No data", "-", "-", "Missing", "-", "Ticker not scored / bad coverage"] )
    return out


def leadership_rows(df: pd.DataFrame, mode: str, n: int = 8) -> List[List[str]]:
    if df is None or df.empty:
        return [["No data", "-", "-", "-", "-"]]
    work = df.copy()
    if mode == "leaders":
        work = work[work["State"].isin(["Leader"])].sort_values(["RSScore", "Alpha21"], ascending=False)
    elif mode == "emerging":
        work = work[work["State"].isin(["Emerging"])].sort_values(["StartScore", "Alpha21"], ascending=False)
    else:
        work = work[work["State"].isin(["Fading", "Weak"])].sort_values(["ShortScore", "Alpha21"], ascending=[False, True]) if "ShortScore" in work.columns else work.sort_values(["Alpha21", "Alpha63"])
    if work.empty:
        return [["None", "-", "-", "-", "No clean names"]]
    out = []
    for _, r in work.head(n).iterrows():
        out.append([r["Ticker"].replace(".JK", ""), r["State"], pct(float(r["LongScore"]) / 100.0) if "LongScore" in r else "-", pct(float(r["ShortScore"]) / 100.0) if "ShortScore" in r else "-", r["Comment"]])
    return out


def cluster_rows(df: pd.DataFrame, n: int = 6) -> List[List[str]]:
    if df is None or df.empty:
        return [["No data", "-", "-", "-", "-"]]
    key = "Cluster" if "Cluster" in df.columns else ("Theme" if "Theme" in df.columns else None)
    if not key:
        return [["No cluster", "-", "-", "-", "-"]]
    work = df.copy()
    agg = (
        work.groupby(key, as_index=False)
        .agg(Names=("Ticker", "count"), AvgLong=("LongScore", "mean"), AvgShort=("ShortScore", "mean"), Examples=("Ticker", lambda s: ", ".join([x.replace('.JK','') for x in list(s.head(3))])))
        .sort_values(["AvgLong", "Names"], ascending=[False, False])
    )
    out = []
    for _, r in agg.head(n).iterrows():
        out.append([r[key], str(int(r["Names"])), f"{r['AvgLong']:.0f}", f"{r['AvgShort']:.0f}", r["Examples"]])
    return out


def top_rank_text(df: pd.DataFrame, side: str = "long", n: int = 3) -> str:
    if df is None or df.empty:
        return "No clean data"
    work = df.copy()
    if side == "long":
        work = work.sort_values(["LongScore", "RSScore", "StartScore"], ascending=False)
    else:
        work = work.sort_values(["ShortScore", "Alpha21", "Trend"], ascending=[False, True, True])
    names = [str(x).replace('.JK', '') for x in work["Ticker"].head(n).tolist()]
    return ", ".join(names) if names else "No clean data"


def merged_playbook_rows(core: Dict[str, object], play_now: Dict[str, List[str]], play_next: Dict[str, List[str]], us_df: pd.DataFrame, ihsg_df: pd.DataFrame, fx_df: pd.DataFrame, commod_df: pd.DataFrame, crypto_df: pd.DataFrame) -> List[List[str]]:
    return [
        ["US stocks", ", ".join(play_now.get("US Stocks", [])), ", ".join(play_next.get("US Stocks", [])), top_rank_text(us_df, "long"), top_rank_text(us_df, "short")],
        ["IHSG", "selective cyclicals / resource beta" if core["current_q"] in ["Q1", "Q2"] else "banks + resource barbell / selective defensives", "broader beta if breadth improves" if core["next_q"] in ["Q1", "Q2"] else "stay selective and quality", top_rank_text(ihsg_df, "long"), top_rank_text(ihsg_df, "short")],
        ["Forex", "favor USD / safety" if core["current_q"] in ["Q3", "Q4"] else "favor commodity FX / weaker USD", "watch cross-over in commodity FX and Europe majors", top_rank_text(fx_df, "long"), top_rank_text(fx_df, "short")],
        ["Commodities", "hard-asset hedge + selective energy" if core["current_q"] == "Q3" else ("reflation basket" if core["current_q"] == "Q2" else "duration-sensitive / mixed"), "more cyclicals / base metals" if core["next_q"] == "Q2" else ("more duration / defensives" if core["next_q"] == "Q4" else "stay selective"), top_rank_text(commod_df, "long"), top_rank_text(commod_df, "short")],
        ["Crypto", ", ".join(play_now.get("Crypto", [])), ", ".join(play_next.get("Crypto", [])), top_rank_text(crypto_df, "long"), top_rank_text(crypto_df, "short")],
    ]


def merged_risk_rows(core: Dict[str, object], risk_pack: Dict[str, float]) -> List[List[str]]:
    return [
        ["Base stance", "Stay selective" if core["current_q"] in ["Q3", "Q4"] else "Can lean risk-on selectively", "Use ticker score for ranking; do not treat it as certainty"],
        ["Crash watch", bucket(risk_pack["big_crash"], (0.25, 0.45), ("Low", "Watch", "Elevated+")), "Watch breadth, leader retention, credit/liquidity proxies"],
        ["Top / bottom risk", f"{ladder_state(core['top_score'], 'top')} / {ladder_state(core['bottom_score'], 'bottom')}", "Separates stretched tape from true washout"],
        ["Transition pressure", pct(core["transition_pressure"]), "Higher means current quad is less stable"],
        ["Confirmation", "breadth improvement + stronger leaders + lower fragility", "Needed before pressing broad beta"],
        ["Invalidation", "narrow breadth + rising crash meter + weak leader retention", "Means keep sizing smaller and avoid chasing"],
    ]


def macro_framework_rows() -> List[List[str]]:
    return [
        ["Inflation prints", "CPI / PPI / PCE", "Shifts Q2↔Q3 probability and hard-asset vs duration balance"],
        ["Labor / growth", "NFP, jobless claims, retail sales, ISM, industrial production", "Moves growth branch and breadth durability"],
        ["Policy / liquidity", "FOMC, QT/QE, funding stress, real rates", "Changes fragility and risk-on / risk-off posture"],
        ["Cross-asset confirms", "DXY, yields, oil, credit", "Confirms whether the move is broad, inflationary, or defensive"],
        ["Market structure", "breadth, leaders, basket concentration", "Tells you whether impact is healthy or too narrow"],
    ]


def render_market_panels(label: str, df: pd.DataFrame, watchlist: List[str], tabs_container=None):
    cols = st.columns(2)
    with cols[0]:
        st.markdown(f"<div class='card'><div class='section-title'>Top Long Candidates • {label}</div>", unsafe_allow_html=True)
        st.markdown(table_html(["Ticker", "Bias", "Long", "Short", "Cluster", "Comment"], score_rows_for_display(df, "long", show_count)), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"<div class='card'><div class='section-title'>Top Short Candidates • {label}</div>", unsafe_allow_html=True)
        st.markdown(table_html(["Ticker", "Bias", "Long", "Short", "Cluster", "Comment"], score_rows_for_display(df, "short", show_count)), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='card'><div class='section-title'>Exact Watchlist • {label}</div>", unsafe_allow_html=True)
    st.markdown(table_html(["Ticker", "Bias", "Long", "Short", "State", "Cluster", "Comment"], exact_rows_for_display(df, watchlist, max(show_count, len(watchlist)))), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


market_payloads = [
    ("US", us_score_df, watchlist_us),
    ("IHSG", ihsg_score_df, watchlist_ihsg),
    ("Forex", fx_score_df, watchlist_fx),
    ("Commodities", commod_score_df, watchlist_commod),
    ("Crypto", crypto_score_df, watchlist_crypto),
]

if region_mode == "All markets":
    tabs = st.tabs([m[0] for m in market_payloads])
    for tab, (label, df, watchlist) in zip(tabs, market_payloads):
        with tab:
            render_market_panels(label, df, watchlist)
else:
    payload = {m[0]: m for m in market_payloads}[region_mode]
    render_market_panels(payload[0], payload[1], payload[2])

with st.expander("Show supporting tables that still matter"):
    support_tabs = st.tabs([m[0] for m in market_payloads])
    for tab, (label, df, _watchlist) in zip(support_tabs, market_payloads):
        with tab:
            a, b = st.columns(2)
            with a:
                st.markdown(f"<div class='card'><div class='section-title'>{label} Leadership Diagnostics</div>", unsafe_allow_html=True)
                st.markdown(table_html(["Ticker", "State", "Long", "Short", "Comment"], leadership_rows(df, "leaders", show_count)), unsafe_allow_html=True)
                st.write("")
                st.markdown(table_html(["Ticker", "State", "Long", "Short", "Comment"], leadership_rows(df, "emerging", show_count)), unsafe_allow_html=True)
                st.write("")
                st.markdown(table_html(["Ticker", "State", "Long", "Short", "Comment"], leadership_rows(df, "fading", show_count)), unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with b:
                st.markdown(f"<div class='card'><div class='section-title'>{label} Cluster Summary</div>", unsafe_allow_html=True)
                st.markdown(table_html(["Cluster", "Names", "AvgLong", "AvgShort", "Examples"], cluster_rows(df, show_count)), unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

with st.expander("Show merged engine details that still matter"):
    meta_tabs = st.tabs(["Merged playbook", "Risk + catalyst map", "Coverage / diagnostics"])
    with meta_tabs[0]:
        st.markdown("<div class='small-muted'>Ini bagian yang sebelumnya hilang dari engine lama: decision snapshot, cross-asset bias, FX / commodity map, dan winners-losers per market. Sekarang gue merge biar nggak numpuk.</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'><div class='section-title'>Merged Cross-Asset Playbook</div>", unsafe_allow_html=True)
        st.markdown(table_html(["Area", "Bias now", "If next wins", "Best ranked now", "Avoid now"], merged_playbook_rows(core, play_now, play_next, us_score_df, ihsg_score_df, fx_score_df, commod_score_df, crypto_score_df)), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with meta_tabs[1]:
        a, b = st.columns(2)
        with a:
            st.markdown("<div class='card'><div class='section-title'>Merged Risk Architecture</div>", unsafe_allow_html=True)
            st.markdown(table_html(["Lens", "Read now", "How to use"], merged_risk_rows(core, risk_pack)), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with b:
            st.markdown("<div class='card'><div class='section-title'>Macro Catalyst Framework</div>", unsafe_allow_html=True)
            st.markdown(table_html(["Bucket", "Watch", "Why it matters"], macro_framework_rows()), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    with meta_tabs[2]:
        st.markdown("<div class='small-muted'>Dashboard Control Check gunanya buat ngecek universe mana yang kebaca dan mana yang miss. Itu berguna buat debug input ticker, tapi bukan board utama, jadi gue taruh di bawah dan bisa di-hide.</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'><div class='section-title'>Dashboard Control Check</div>", unsafe_allow_html=True)
        st.markdown(table_html(["Market", "Coverage", "Missing examples"], coverage_rows), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='small-muted'>Correlated names/themes are merged into tighter clusters where it improves readability. Ticker scores rank candidates by macro fit + relative strength + trend. They are watchlists, not guarantees. Impact board is an attribution lens, not a certainty machine.</div>",
    unsafe_allow_html=True,
)
