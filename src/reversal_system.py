from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .indicators import moving_average


@dataclass
class ReversalParams:
    rps50_threshold: float = 85.0
    near_120d_high_threshold: float = 0.9


def evaluate_reversal_stock(stock_df: pd.DataFrame, rps50_value: float, params: ReversalParams) -> tuple[dict[str, bool | int | float], pd.DataFrame]:
    x = stock_df.copy()
    x = x.sort_values("date").reset_index(drop=True)

    close = x["close"].astype(float)
    high = x["high"].astype(float)

    ma10 = moving_average(close, 10)
    ma20 = moving_average(close, 20)
    ma50 = moving_average(close, 50)
    ma120 = moving_average(close, 120)
    ma250 = moving_average(close, 250)

    a = close > ma250

    hhv50 = high.rolling(50, min_periods=50).max()
    nh = high >= hhv50
    b = nh.rolling(30, min_periods=1).sum() >= 1

    nn = a
    aa_days = nn.rolling(30, min_periods=1).sum()
    aa = (aa_days > 2) & (aa_days < 30)

    hhv120 = high.rolling(120, min_periods=120).max()
    ab = (high / hhv120) > params.near_120d_high_threshold

    d = pd.Series([bool(rps50_value > params.rps50_threshold)] * len(x), index=x.index)

    reversal_signal_series = a & b & d & aa & ab
    prev29_any_true = reversal_signal_series.shift(1).rolling(29, min_periods=1).sum().fillna(0) > 0
    reversal_first_in_30d_series = reversal_signal_series & (~prev29_any_true)

    tail_debug = pd.DataFrame(
        {
            "date": x["date"],
            "close": close,
            "high": high,
            "reversal_A__close_above_ma250": a,
            "reversal_B__new_high_50d_in_30d": b,
            "reversal_D__rps50_gt_85": d,
            "reversal_AA__days_above_ma250_in_30d": aa_days.astype("Int64"),
            "reversal_AB__high_near_120d_high": ab,
            "reversal_signal": reversal_signal_series,
            "reversal_first_in_30d": reversal_first_in_30d_series,
            "ma10": ma10,
            "ma20": ma20,
            "ma50": ma50,
            "ma120": ma120,
            "ma250": ma250,
        }
    )

    last = tail_debug.iloc[-1]
    out = {
        "reversal_signal": bool(last["reversal_signal"]),
        "reversal_first_in_30d": bool(last["reversal_first_in_30d"]),
        "reversal_rps50": float(rps50_value),
        "reversal_A__close_above_ma250": bool(last["reversal_A__close_above_ma250"]),
        "reversal_B__new_high_50d_in_30d": bool(last["reversal_B__new_high_50d_in_30d"]),
        "reversal_D__rps50_gt_85": bool(last["reversal_D__rps50_gt_85"]),
        "reversal_AA__days_above_ma250_in_30d": int(last["reversal_AA__days_above_ma250_in_30d"])
        if pd.notna(last["reversal_AA__days_above_ma250_in_30d"])
        else 0,
        "reversal_AB__high_near_120d_high": bool(last["reversal_AB__high_near_120d_high"]),
        "reversal_system_signal": bool(last["reversal_signal"]),
    }
    return out, tail_debug.tail(5).copy()


def signal_label(trend_signal: bool, reversal_signal: bool) -> str:
    if trend_signal and reversal_signal:
        return "BOTH"
    if trend_signal:
        return "TREND_ONLY"
    if reversal_signal:
        return "REVERSAL_ONLY"
    return "NONE"
