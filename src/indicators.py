from __future__ import annotations

import numpy as np
import pandas as pd


def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def count_true_last_n(condition: pd.Series, n: int) -> int:
    if len(condition) < n:
        return 0
    return int(condition.tail(n).sum())


def every_true_last_n(condition: pd.Series, n: int) -> bool:
    if len(condition) < n:
        return False
    return bool(condition.tail(n).all())


def hhv_ratio(close: pd.Series, lookback: int) -> float:
    if len(close) < lookback:
        return float("nan")
    peak = close.tail(lookback).max()
    if peak <= 0:
        return float("nan")
    return float(close.iloc[-1] / peak)


def rolling_max_drawdown(close: pd.Series) -> float:
    if close.empty:
        return float("nan")
    roll_max = close.cummax()
    dd = (roll_max - close) / roll_max.replace(0, np.nan)
    return float(dd.max())


def pct_rank(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True) * 100
