from __future__ import annotations

from typing import Dict, List


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def recommended_risk_budget(core: Dict[str, object], crash_now: float) -> List[List[str]]:
    q = str(core.get("current_q", "Q3"))
    confidence = float(core.get("confidence", 0.5))
    fragility = float(core.get("fragility", 0.5))
    q_next = str(core.get("next_q", q))

    if q == "Q3":
        base = {
            "Defensives / quality": 0.22,
            "Gold / gold miners": 0.18,
            "Energy / inflation hedges": 0.16,
            "US cyclicals": 0.08,
            "Small caps": 0.05,
            "EM / IHSG beta": 0.07,
            "Crypto beta": 0.05,
            "Cash / dry powder": 0.19,
        }
    elif q == "Q2":
        base = {
            "Defensives / quality": 0.10,
            "Gold / gold miners": 0.08,
            "Energy / inflation hedges": 0.12,
            "US cyclicals": 0.20,
            "Small caps": 0.16,
            "EM / IHSG beta": 0.12,
            "Crypto beta": 0.10,
            "Cash / dry powder": 0.12,
        }
    elif q == "Q4":
        base = {
            "Defensives / quality": 0.18,
            "Gold / gold miners": 0.10,
            "Energy / inflation hedges": 0.06,
            "US cyclicals": 0.06,
            "Small caps": 0.04,
            "EM / IHSG beta": 0.05,
            "Crypto beta": 0.03,
            "Cash / dry powder": 0.48,
        }
    else:
        base = {
            "Defensives / quality": 0.12,
            "Gold / gold miners": 0.08,
            "Energy / inflation hedges": 0.08,
            "US cyclicals": 0.16,
            "Small caps": 0.10,
            "EM / IHSG beta": 0.10,
            "Crypto beta": 0.08,
            "Cash / dry powder": 0.28,
        }

    # transition tilt
    if q == "Q3" and q_next == "Q2":
        base["US cyclicals"] += 0.03
        base["Small caps"] += 0.02
        base["Cash / dry powder"] -= 0.03
        base["Gold / gold miners"] -= 0.02

    gross_cap = _clamp01(0.35 + 0.55 * confidence - 0.35 * fragility - 0.30 * crash_now)
    gross_cap = max(0.15, gross_cap)
    cash_floor = 0.10 + 0.20 * crash_now + 0.10 * fragility

    rows: List[List[str]] = []
    for bucket, weight in base.items():
        scaled = weight * gross_cap
        if bucket == "Cash / dry powder":
            scaled = max(scaled, cash_floor)
        use = "Core"
        if bucket in ["Small caps", "Crypto beta", "EM / IHSG beta"] and crash_now > 0.45:
            use = "Tactical only"
        elif bucket in ["Gold / gold miners", "Energy / inflation hedges"] and q == "Q3":
            use = "Prefer"
        elif bucket == "US cyclicals" and q == "Q3":
            use = "Probe only"
        rows.append([bucket, f"{scaled*100:.1f}%", use])
    rows.sort(key=lambda r: float(r[1].strip('%')), reverse=True)
    return rows


def risk_controls_rows(core: Dict[str, object], crash_now: float) -> List[List[str]]:
    confidence = float(core.get("confidence", 0.5))
    fragility = float(core.get("fragility", 0.5))
    top = float(core.get("top_score", 0.5))
    max_single = max(0.03, 0.12 * confidence * (1 - crash_now))
    max_theme = max(0.08, 0.25 * confidence * (1 - 0.6 * crash_now))
    gross = max(0.15, 0.45 * confidence + 0.20 * (1 - fragility) - 0.20 * crash_now)
    rows = [
        ["Max single-name risk", f"{max_single*100:.1f}%", "Cut lower if catalyst risk is near"],
        ["Max theme exposure", f"{max_theme*100:.1f}%", "Avoid stacking highly correlated longs"],
        ["Gross exposure guide", f"{gross*100:.1f}%", "Use as ceiling, not target"],
        ["Beta kill-switch", ">55% crash meter", "Cut beta / high-duration trades by 25–40%"],
        ["Top-risk de-chase", f"{top*100:.1f}% top score", "Do not add to stretched winners blindly"],
    ]
    return rows
