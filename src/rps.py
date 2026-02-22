from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, cast

import pandas as pd

from .indicators import pct_rank


@dataclass
class RpsResult:
    rps120: pd.Series
    rps250: pd.Series


@dataclass
class Rps50Result:
    rps50: pd.Series


def compute_fast_proxy_rps(
    latest_returns: pd.DataFrame,
    hs300_ret120: float,
    hs300_ret250: float,
) -> RpsResult:
    df = latest_returns.copy()
    df["excess120"] = df["ret120"] - hs300_ret120
    df["excess250"] = df["ret250"] - hs300_ret250
    r120 = pct_rank(cast(pd.Series, df["excess120"].fillna(-9e9)))
    r250 = pct_rank(cast(pd.Series, df["excess250"].fillna(-9e9)))
    return RpsResult(rps120=r120, rps250=r250)


def compute_strict_market_rps_latest_day(stock_close_map: Dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for symbol, close in stock_close_map.items():
        if len(close) < 251:
            continue
        c = close.dropna()
        if len(c) < 251:
            continue
        ret120 = c.iloc[-1] / c.iloc[-121] - 1
        ret250 = c.iloc[-1] / c.iloc[-251] - 1
        rows.append((symbol, ret120, ret250))

    if not rows:
        return pd.DataFrame({"symbol": pd.Series(dtype="string"), "rps120": pd.Series(dtype="float"), "rps250": pd.Series(dtype="float")})
    df = pd.DataFrame(rows, columns=["symbol", "ret120", "ret250"])

    df["rps120"] = pct_rank(cast(pd.Series, df["ret120"]))
    df["rps250"] = pct_rank(cast(pd.Series, df["ret250"]))
    return df.loc[:, ["symbol", "rps120", "rps250"]].copy()


def compute_fast_proxy_rps50(latest_returns: pd.DataFrame, hs300_ret50: float) -> Rps50Result:
    df = latest_returns.copy()
    df["excess50"] = df["ret50"] - hs300_ret50
    r50 = pct_rank(cast(pd.Series, df["excess50"].fillna(-9e9)))
    return Rps50Result(rps50=r50)


def compute_strict_market_rps50_latest_day(stock_close_map: Dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for symbol, close in stock_close_map.items():
        c = close.dropna()
        if len(c) < 51:
            continue
        ret50 = c.iloc[-1] / c.iloc[-51] - 1
        rows.append((symbol, ret50))

    if not rows:
        return pd.DataFrame({"symbol": pd.Series(dtype="string"), "rps50": pd.Series(dtype="float")})
    df = pd.DataFrame(rows, columns=["symbol", "ret50"])

    df["rps50"] = pct_rank(cast(pd.Series, df["ret50"]))
    return df.loc[:, ["symbol", "rps50"]].copy()
