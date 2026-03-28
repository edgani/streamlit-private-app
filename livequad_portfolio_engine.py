from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class PortfolioConfig:
    annual_vol_target: float = 0.12
    max_gross: float = 1.30
    max_net: float = 1.00
    max_single_weight: float = 0.18
    max_bucket_weight: float = 0.35
    correlation_cap: float = 0.75
    rebalance_threshold: float = 0.02
    commission_bps: float = 3.0
    market_impact_bps_per_10pct_adv: float = 8.0


def default_bucket_scores(decision_regime: str, next_regime: Optional[str] = None) -> Dict[str, float]:
    q = (decision_regime or "Q3").upper()
    nxt = (next_regime or q).upper()
    if q == "Q3":
        scores = {
            "Gold": 0.75,
            "Energy": 0.70,
            "Defensives": 0.62,
            "US_Cyclicals": 0.32,
            "Small_Caps": 0.22,
            "EM_IHSG": 0.28,
            "Crypto_Beta": 0.24,
            "Duration": 0.38,
            "Cash": 0.55,
        }
    elif q == "Q2":
        scores = {
            "Gold": 0.28,
            "Energy": 0.55,
            "Defensives": 0.26,
            "US_Cyclicals": 0.74,
            "Small_Caps": 0.70,
            "EM_IHSG": 0.62,
            "Crypto_Beta": 0.60,
            "Duration": 0.18,
            "Cash": 0.26,
        }
    elif q == "Q4":
        scores = {
            "Gold": 0.34,
            "Energy": 0.20,
            "Defensives": 0.48,
            "US_Cyclicals": 0.18,
            "Small_Caps": 0.10,
            "EM_IHSG": 0.12,
            "Crypto_Beta": 0.08,
            "Duration": 0.68,
            "Cash": 0.74,
        }
    else:
        scores = {
            "Gold": 0.32,
            "Energy": 0.34,
            "Defensives": 0.20,
            "US_Cyclicals": 0.62,
            "Small_Caps": 0.50,
            "EM_IHSG": 0.45,
            "Crypto_Beta": 0.42,
            "Duration": 0.24,
            "Cash": 0.30,
        }
    if q == "Q3" and nxt == "Q2":
        scores["US_Cyclicals"] += 0.10
        scores["Small_Caps"] += 0.08
        scores["EM_IHSG"] += 0.07
        scores["Gold"] -= 0.06
        scores["Cash"] -= 0.05
    return {k: float(max(0.0, min(1.0, v))) for k, v in scores.items()}


def score_to_target_weights(
    bucket_scores: Mapping[str, float],
    *,
    confidence: float,
    crash_meter: float,
    fragility: float,
    config: Optional[PortfolioConfig] = None,
) -> pd.DataFrame:
    cfg = config or PortfolioConfig()
    gross_budget = max(0.20, min(cfg.max_gross, 0.45 + 0.75 * confidence - 0.45 * crash_meter - 0.20 * fragility))
    raw = pd.Series(bucket_scores, dtype=float).clip(lower=0.0)
    if raw.sum() <= 0:
        raw = pd.Series({"Cash": 1.0})
    tgt = raw / raw.sum() * gross_budget
    tgt = tgt.clip(upper=cfg.max_bucket_weight)
    if "Cash" in tgt:
        tgt["Cash"] = max(tgt.get("Cash", 0.0), 0.08 + 0.20 * crash_meter)
    else:
        tgt.loc["Cash"] = 0.08 + 0.20 * crash_meter
    tgt = tgt / tgt.sum() * min(cfg.max_gross, max(tgt.sum(), gross_budget))
    out = pd.DataFrame({"bucket": tgt.index, "target_weight": tgt.values})
    out["conviction"] = out["bucket"].map(lambda b: bucket_scores.get(b, 0.0))
    return out.sort_values("target_weight", ascending=False).reset_index(drop=True)


def correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    if returns_df.empty:
        return pd.DataFrame()
    work = returns_df.apply(pd.to_numeric, errors="coerce")
    work = work.dropna(axis=1, how="all")
    if work.shape[1] < 2:
        return pd.DataFrame()
    return work.corr()


def apply_correlation_caps(weights: pd.DataFrame, corr: pd.DataFrame, config: Optional[PortfolioConfig] = None) -> pd.DataFrame:
    cfg = config or PortfolioConfig()
    if weights.empty or corr.empty:
        return weights.copy()
    out = weights.copy()
    capped = []
    for i, row in out.iterrows():
        b = row["bucket"]
        if b not in corr.columns:
            capped.append(row["target_weight"])
            continue
        highly_corr = corr[b].drop(index=b, errors="ignore")
        overlap = highly_corr[highly_corr.abs() >= cfg.correlation_cap].index.tolist()
        penalty = 1.0 - 0.10 * len(overlap)
        capped.append(max(0.0, row["target_weight"] * max(0.55, penalty)))
    out["target_weight_corr_capped"] = capped
    total = out["target_weight_corr_capped"].sum()
    if total > 0:
        out["target_weight_corr_capped"] = out["target_weight_corr_capped"] / total * min(cfg.max_gross, total)
    return out


def estimate_rebalance(
    current_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
    *,
    adv_participation: Optional[Mapping[str, float]] = None,
    config: Optional[PortfolioConfig] = None,
) -> pd.DataFrame:
    cfg = config or PortfolioConfig()
    names = sorted(set(current_weights) | set(target_weights))
    rows: List[Dict[str, float]] = []
    for name in names:
        cur = float(current_weights.get(name, 0.0))
        tgt = float(target_weights.get(name, 0.0))
        trade = tgt - cur
        abs_trade = abs(trade)
        adv_pct = float((adv_participation or {}).get(name, 0.05))
        impact_bps = cfg.market_impact_bps_per_10pct_adv * (adv_pct / 0.10) * max(abs_trade, 1e-6) / max(adv_pct, 1e-6)
        total_cost_bps = cfg.commission_bps + impact_bps
        action = "Hold"
        if trade > cfg.rebalance_threshold:
            action = "Buy"
        elif trade < -cfg.rebalance_threshold:
            action = "Sell"
        rows.append({
            "bucket": name,
            "current_weight": cur,
            "target_weight": tgt,
            "trade": trade,
            "action": action,
            "est_cost_bps": total_cost_bps,
        })
    out = pd.DataFrame(rows).sort_values("trade", key=lambda s: s.abs(), ascending=False)
    out["turnover"] = out["trade"].abs()
    return out.reset_index(drop=True)


def build_risk_report(
    decision_regime: str,
    next_regime: str,
    *,
    confidence: float,
    crash_meter: float,
    fragility: float,
    returns_df: Optional[pd.DataFrame] = None,
    current_weights: Optional[Mapping[str, float]] = None,
    config: Optional[PortfolioConfig] = None,
) -> Dict[str, object]:
    cfg = config or PortfolioConfig()
    scores = default_bucket_scores(decision_regime, next_regime)
    targets = score_to_target_weights(scores, confidence=confidence, crash_meter=crash_meter, fragility=fragility, config=cfg)
    corr = correlation_matrix(returns_df if returns_df is not None else pd.DataFrame())
    capped = apply_correlation_caps(targets, corr, config=cfg) if not corr.empty else targets.copy()
    tgt_map_col = "target_weight_corr_capped" if "target_weight_corr_capped" in capped.columns else "target_weight"
    rebalance = estimate_rebalance(current_weights or {}, dict(zip(capped["bucket"], capped[tgt_map_col])), config=cfg)
    posture = "Defensive-selective"
    if decision_regime == "Q2" and confidence >= 0.6 and crash_meter < 0.35:
        posture = "Risk-on but disciplined"
    elif decision_regime == "Q4":
        posture = "Capital preservation / duration bias"
    elif decision_regime == "Q3" and next_regime == "Q2":
        posture = "Defensive core with selective reflation probes"
    controls = pd.DataFrame([
        ["Gross cap", cfg.max_gross],
        ["Net cap", cfg.max_net],
        ["Single-name cap", cfg.max_single_weight],
        ["Bucket cap", cfg.max_bucket_weight],
        ["Correlation cap", cfg.correlation_cap],
        ["Rebalance threshold", cfg.rebalance_threshold],
    ], columns=["control", "value"])
    return {
        "posture": posture,
        "scores": pd.DataFrame({"bucket": list(scores.keys()), "score": list(scores.values())}).sort_values("score", ascending=False),
        "targets": capped,
        "rebalance": rebalance,
        "controls": controls,
        "correlation": corr,
    }
