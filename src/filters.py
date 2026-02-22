from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd

from .indicators import count_true_last_n, every_true_last_n, hhv_ratio, moving_average, rolling_max_drawdown


@dataclass
class TrendConfig:
    hhv_ratio_threshold: float
    rps_sum_threshold: float


@dataclass
class RiskConfig:
    reduce_on_close_below_ma20_days: int
    exit_on_close_below_ma60: bool
    exit_on_drawdown: float
    take_profit_protect_half: bool


def market_regime_pass(index_df: pd.DataFrame) -> Tuple[bool, Dict[str, bool]]:
    c = index_df["close"]
    ma20 = moving_average(c, 20)
    ma60 = moving_average(c, 60)
    ma250 = moving_average(c, 250)
    cond_close_ma250 = bool(c.iloc[-1] > ma250.iloc[-1]) if len(c) >= 250 else False
    cond_ma20_ma60 = bool(ma20.iloc[-1] > ma60.iloc[-1]) if len(c) >= 60 else False
    passed = cond_close_ma250 or cond_ma20_ma60
    return passed, {
        "market_close_gt_ma250": cond_close_ma250,
        "market_ma20_gt_ma60": cond_ma20_ma60,
    }


def top_n_per_industry(spot_df: pd.DataFrame, n: int, min_turnover: float) -> pd.DataFrame:
    x = spot_df.copy()
    x = x[~x["is_st"]]
    x = x[x["turnover"].fillna(0) >= min_turnover]
    x = x.dropna(subset=["industry", "total_mv"])
    x = x.sort_values(["industry", "total_mv"], ascending=[True, False])
    x["industry_rank_by_mv"] = x.groupby("industry").cumcount() + 1
    return x[x["industry_rank_by_mv"] <= n].copy()


def industry_strength_by_proxy(
    candidates: pd.DataFrame,
    stock_hist: Dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
) -> pd.DataFrame:
    index_c = index_df["close"].dropna()
    hs300_60d = index_c.iloc[-1] / index_c.iloc[-61] - 1 if len(index_c) > 60 else float("nan")

    rows = []
    for industry, part in candidates.groupby("industry"):
        returns = []
        for s in part["symbol"].tolist():
            hist = stock_hist.get(s)
            if hist is None or hist.empty:
                continue
            c = hist["close"].dropna()
            if len(c) <= 60:
                continue
            returns.append(c.iloc[-1] / c.iloc[-61] - 1)
        if not returns:
            continue
        eq_ret60 = float(pd.Series(returns).mean())
        rows.append({"industry": industry, "industry_ret60_proxy": eq_ret60, "industry_excess60_proxy": eq_ret60 - hs300_60d})

    return pd.DataFrame(rows)


def industry_strength_by_index(
    candidates: pd.DataFrame,
    industry_hist: Dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
) -> pd.DataFrame:
    index_c = index_df["close"].dropna()
    hs300_60d = index_c.iloc[-1] / index_c.iloc[-61] - 1 if len(index_c) > 60 else float("nan")

    rows = []
    for industry in sorted(set(candidates["industry"].dropna().astype(str).tolist())):
        hist = industry_hist.get(industry)
        if hist is None or hist.empty:
            continue
        c = hist["close"].dropna()
        if len(c) < 121:
            continue
        ma120 = moving_average(c, 120)
        close_gt_ma120 = bool(c.iloc[-1] > ma120.iloc[-1])
        ret60 = c.iloc[-1] / c.iloc[-61] - 1 if len(c) > 60 else float("nan")
        excess60 = ret60 - hs300_60d
        rows.append(
            {
                "industry": industry,
                "industry_close_gt_ma120": close_gt_ma120,
                "industry_ret60": float(ret60),
                "industry_excess60": float(excess60),
                "industry_strength_pass": bool(close_gt_ma120 or excess60 > 0),
            }
        )
    return pd.DataFrame(rows)


def trend_train_track_flags(stock_df: pd.DataFrame, trend_cfg: TrendConfig, rps120: float, rps250: float) -> Dict[str, bool | float]:
    c = stock_df["close"].dropna()
    ma10 = moving_average(c, 10)
    ma20 = moving_average(c, 20)
    ma200 = moving_average(c, 200)
    ma250 = moving_average(c, 250)

    cond_c_ma250 = count_true_last_n(c > ma250, 30) >= 25
    cond_c_ma200 = count_true_last_n(c > ma200, 30) >= 25
    cond_c_ma20 = count_true_last_n(c > ma20, 10) >= 9
    cond_hhv = hhv_ratio(c, 250) > trend_cfg.hhv_ratio_threshold
    cond_ma20_up = every_true_last_n(ma20 >= ma20.shift(1), 5)
    cond_ma10_over_ma20 = every_true_last_n(ma10 >= ma20, 5)
    cond_rps = (rps120 + rps250) > trend_cfg.rps_sum_threshold

    final_signal = all(
        [
            cond_c_ma250,
            cond_c_ma200,
            cond_c_ma20,
            cond_hhv,
            cond_ma20_up,
            cond_ma10_over_ma20,
            cond_rps,
        ]
    )
    return {
        "cond_c_ma250": cond_c_ma250,
        "cond_c_ma200": cond_c_ma200,
        "cond_c_ma20": cond_c_ma20,
        "cond_hhv": cond_hhv,
        "cond_ma20_up": cond_ma20_up,
        "cond_ma10_over_ma20": cond_ma10_over_ma20,
        "cond_rps": cond_rps,
        "rps120": float(rps120),
        "rps250": float(rps250),
        "rps_sum": float(rps120 + rps250),
        "final_signal": final_signal,
    }


def risk_exit_flags(stock_df: pd.DataFrame, risk_cfg: RiskConfig) -> Dict[str, bool | float]:
    c = stock_df["close"].dropna()
    ma20 = moving_average(c, 20)
    ma60 = moving_average(c, 60)
    below_ma20_days = count_true_last_n(c < ma20, risk_cfg.reduce_on_close_below_ma20_days)
    reduce_position = below_ma20_days >= risk_cfg.reduce_on_close_below_ma20_days
    exit_ma60 = bool(c.iloc[-1] < ma60.iloc[-1]) if risk_cfg.exit_on_close_below_ma60 and len(c) >= 60 else False
    drawdown = rolling_max_drawdown(c)
    exit_drawdown = bool(drawdown > risk_cfg.exit_on_drawdown)
    return {
        "risk_reduce": reduce_position,
        "risk_exit_ma60": exit_ma60,
        "risk_drawdown": float(drawdown),
        "risk_exit_drawdown": exit_drawdown,
    }
