from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, cast

import akshare as ak
import pandas as pd


@dataclass
class FetchConfig:
    cache_dir: Path
    max_retries: int = 3
    retry_sleep_seconds: float = 1.5
    request_sleep_seconds: float = 0.2


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _retry_call(func: Callable[[], pd.DataFrame], cfg: FetchConfig, logger: logging.Logger) -> pd.DataFrame:
    last_exc: Optional[Exception] = None
    for i in range(cfg.max_retries):
        try:
            time.sleep(cfg.request_sleep_seconds)
            data = func()
            if data is None or data.empty:
                raise ValueError("empty dataframe")
            return data
        except Exception as exc:  # noqa: PERF203
            last_exc = exc
            logger.warning("fetch attempt %s/%s failed: %s", i + 1, cfg.max_retries, exc)
            time.sleep(cfg.retry_sleep_seconds)
    raise RuntimeError(f"fetch failed after retries: {last_exc}")


def _normalize_hist_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }
    for c in list(df.columns):
        if c in col_map:
            df = df.rename(columns={c: col_map[c]})
    if "date" not in df.columns and "日期" in df.columns:
        df = df.rename(columns={"日期": "date"})
    df["date"] = pd.to_datetime(df["date"])  # type: ignore[index]
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _normalize_spot_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "代码": "symbol",
        "名称": "name",
        "总市值": "total_mv",
        "成交额": "turnover",
        "成交额(元)": "turnover",
        "行业": "industry",
        "所属行业": "industry",
    }
    matched = {k: v for k, v in rename_map.items() if k in df.columns}
    out = df.rename(columns=matched).copy()
    for key in ["symbol", "name", "total_mv", "turnover"]:
        if key not in out.columns:
            out[key] = pd.NA
    if "industry" not in out.columns:
        out["industry"] = pd.NA
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    out["name"] = out["name"].astype(str)
    out["is_st"] = out["name"].str.contains("ST", case=False, na=False)
    out["total_mv"] = pd.to_numeric(out["total_mv"], errors="coerce")
    out["turnover"] = pd.to_numeric(out["turnover"], errors="coerce")
    return out.loc[:, ["symbol", "name", "industry", "total_mv", "turnover", "is_st"]].copy()


def fetch_spot(cfg: FetchConfig, logger: logging.Logger) -> pd.DataFrame:
    def _call() -> pd.DataFrame:
        return ak.stock_zh_a_spot_em()

    df = _retry_call(_call, cfg, logger)
    return _normalize_spot_columns(df)


def fetch_hs300_daily(cfg: FetchConfig, logger: logging.Logger) -> pd.DataFrame:
    cache = cfg.cache_dir / "index" / "sh000300.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        return _normalize_hist_columns(df)

    def _call_primary() -> pd.DataFrame:
        return ak.stock_zh_index_daily_em(symbol="sh000300")

    def _call_fallback() -> pd.DataFrame:
        return ak.index_zh_a_hist(symbol="000300", period="daily")

    try:
        raw = _retry_call(_call_primary, cfg, logger)
    except Exception:
        logger.warning("primary index interface failed, switching fallback")
        raw = _retry_call(_call_fallback, cfg, logger)

    df = _normalize_hist_columns(raw)
    _ensure_parent(cache)
    df.to_parquet(cache, index=False)
    return df


def fetch_stock_daily(symbol: str, cfg: FetchConfig, logger: logging.Logger) -> pd.DataFrame:
    symbol6 = str(symbol).zfill(6)
    cache = cfg.cache_dir / "stocks" / f"{symbol6}.parquet"
    if cache.exists():
        cached = cast(pd.DataFrame, pd.read_parquet(cache))
        return _normalize_hist_columns(cached)

    def _call() -> pd.DataFrame:
        return ak.stock_zh_a_hist(symbol=symbol6, period="daily", adjust="qfq")

    raw = _retry_call(_call, cfg, logger)
    df = _normalize_hist_columns(raw)
    _ensure_parent(cache)
    df.to_parquet(cache, index=False)
    return df


def fetch_industry_board_hist(industry_name: str, cfg: FetchConfig, logger: logging.Logger) -> pd.DataFrame:
    safe = (
        str(industry_name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("*", "_")
    )
    cache = cfg.cache_dir / "industry" / f"{safe}.parquet"
    if cache.exists():
        cached = cast(pd.DataFrame, pd.read_parquet(cache))
        return _normalize_hist_columns(cached)

    def _call() -> pd.DataFrame:
        return ak.stock_board_industry_hist_em(symbol=industry_name, adjust="")

    raw = _retry_call(_call, cfg, logger)
    df = _normalize_hist_columns(raw)
    _ensure_parent(cache)
    df.to_parquet(cache, index=False)
    return df


def load_industry_map(industry_csv: Optional[Path]) -> pd.DataFrame:
    if industry_csv is None or not industry_csv.exists():
        return pd.DataFrame(
            {
                "industry": pd.Series(dtype="string"),
                "symbol": pd.Series(dtype="string"),
            }
        )
    df = pd.read_csv(industry_csv)
    need = {"industry", "symbol"}
    if not need.issubset(df.columns):
        raise ValueError("industry csv must contain columns: industry, symbol")
    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    return df.loc[:, ["industry", "symbol"]].copy()


def build_industry_map_from_boards(cfg: FetchConfig, logger: logging.Logger) -> pd.DataFrame:
    cache = cfg.cache_dir / "industry" / "industry_map_from_boards.parquet"
    if cache.exists():
        cached = cast(pd.DataFrame, pd.read_parquet(cache))
        if not cached.empty and {"industry", "symbol"}.issubset(cached.columns):
            out = cached.loc[:, ["industry", "symbol"]].copy()
            symbol_series = cast(pd.Series, out["symbol"])
            out["symbol"] = symbol_series.astype(str).str.zfill(6)
            return out.loc[:, ["industry", "symbol"]].copy()

    def _get_board_names() -> pd.DataFrame:
        return ak.stock_board_industry_name_em()

    boards = _retry_call(_get_board_names, cfg, logger)
    name_col = "板块名称" if "板块名称" in boards.columns else boards.columns[0]
    board_names = boards[name_col].dropna().astype(str).tolist()

    rows = []
    for industry in board_names:
        try:
            cons = _retry_call(lambda: ak.stock_board_industry_cons_em(symbol=industry), cfg, logger)
        except Exception as exc:
            logger.warning("skip board %s due to fetch error: %s", industry, exc)
            continue

        if "代码" not in cons.columns:
            continue
        symbols = cons["代码"].dropna().astype(str).str.zfill(6)
        for symbol in symbols.tolist():
            rows.append({"industry": industry, "symbol": symbol})

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.drop_duplicates(subset=["industry", "symbol"]).reset_index(drop=True)
    _ensure_parent(cache)
    out.to_parquet(cache, index=False)
    return out
