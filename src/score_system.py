from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from .filters import RiskConfig, risk_exit_flags
from .indicators import moving_average, pct_rank


class ScoreDataProvider(Protocol):
    def get_stock_history(self, symbol: str) -> pd.DataFrame:
        ...

    def get_benchmark_history(self) -> pd.DataFrame:
        ...


@dataclass
class DictDataProvider:
    stock_hist: dict[str, pd.DataFrame]
    benchmark_hist: pd.DataFrame

    def get_stock_history(self, symbol: str) -> pd.DataFrame:
        return self.stock_hist[symbol]

    def get_benchmark_history(self) -> pd.DataFrame:
        return self.benchmark_hist


@dataclass
class ScoreConfig:
    top_n_output: int = 30
    slope_lookback: int = 10
    structure_lookback: int = 60
    swing_order: int = 4
    breakout_check_window: int = 20
    breakout_high_lookback: int = 60
    volume_sma_window: int = 20
    breakout_volume_mult: float = 1.5
    pullback_trigger_ratio: float = 0.95
    pullback_drawdown_limit: float = 0.20
    rs_long_window: int = 63
    rs_short_window: int = 21
    risk_churn_vol_mult: float = 2.5
    risk_breakdown_vol_mult: float = 1.5
    risk_window: int = 5
    strength_full_score_percentile: float = 80.0
    pullback_near_high_ratio: float = 0.985
    pullback_overextension_ratio: float = 0.18
    enable_extension_filter: bool = True
    extension_ratio_threshold: float = 1.25
    extension_pullback_penalty: float = 5.0


def score_output_columns() -> list[str]:
    return [
        "symbol",
        "name",
        "industry",
        "score_total",
        "score_trend",
        "score_structure",
        "score_volume",
        "score_strength",
        "score_pullback",
        "score_strength_pct63",
        "score_strength_pct21",
        "risk_penalty",
        "risk_flags",
        "risk_reason",
        "score_reason_top3",
        "pullback_note",
        "last_date",
        "close",
        "ma50",
        "ma250",
        "risk_reduce",
        "risk_exit_ma60",
        "risk_drawdown",
        "risk_exit_drawdown",
    ]


def _ret_n(close: pd.Series, n: int) -> float:
    c = close.dropna()
    if len(c) <= n:
        return float("nan")
    return float(c.iloc[-1] / c.iloc[-(n + 1)] - 1)


def _local_extrema_indices(series: pd.Series, order: int, find_high: bool) -> list[int]:
    x = series.dropna().reset_index(drop=True)
    n = len(x)
    if n < (2 * order + 1):
        return []

    idxs: list[int] = []
    for i in range(order, n - order):
        center = float(x.iloc[i])
        left = x.iloc[i - order : i]
        right = x.iloc[i + 1 : i + order + 1]
        if find_high:
            if bool((left < center).all() and (right < center).all()):
                idxs.append(i)
        else:
            if bool((left > center).all() and (right > center).all()):
                idxs.append(i)
    return idxs


def _score_single_stock(
    stock_df: pd.DataFrame,
    cfg: ScoreConfig,
    risk_cfg: RiskConfig,
    strength_long_pct: float,
    strength_short_pct: float,
) -> dict[str, object]:
    x = stock_df.sort_values("date").reset_index(drop=True).copy()
    close = x["close"].astype(float)
    open_ = x["open"].astype(float)
    high = x["high"].astype(float)
    low = x["low"].astype(float)
    volume = x["volume"].astype(float)

    ma50 = moving_average(close, 50)
    ma250 = moving_average(close, 250)
    vol_sma20 = moving_average(volume, cfg.volume_sma_window)

    score_trend = 0
    slope_lb = cfg.slope_lookback
    if len(ma50.dropna()) > slope_lb:
        ma50_pos = bool(close.iloc[-1] > ma50.iloc[-1] and (ma50.iloc[-1] - ma50.iloc[-(slope_lb + 1)]) > 0)
        score_trend += 10 if ma50_pos else 0
    if len(ma250.dropna()) > slope_lb:
        ma250_pos = bool(close.iloc[-1] > ma250.iloc[-1] and (ma250.iloc[-1] - ma250.iloc[-(slope_lb + 1)]) > 0)
        score_trend += 10 if ma250_pos else 0

    score_structure = 0
    tail = x.tail(cfg.structure_lookback).reset_index(drop=True)
    swing_lows = _local_extrema_indices(tail["low"].astype(float), cfg.swing_order, find_high=False)
    swing_highs = _local_extrema_indices(tail["high"].astype(float), cfg.swing_order, find_high=True)
    if len(swing_lows) >= 2:
        low1 = float(tail["low"].iloc[swing_lows[-2]])
        low2 = float(tail["low"].iloc[swing_lows[-1]])
        if low2 > low1:
            score_structure += 10
    if len(swing_highs) >= 2:
        high1 = float(tail["high"].iloc[swing_highs[-2]])
        high2 = float(tail["high"].iloc[swing_highs[-1]])
        if high2 > high1:
            score_structure += 10

    score_volume = 0
    rolling_hh = close.rolling(cfg.breakout_high_lookback, min_periods=cfg.breakout_high_lookback).max()
    breakout_cond = (close >= rolling_hh) & (volume >= cfg.breakout_volume_mult * vol_sma20)
    recent_break = breakout_cond.tail(cfg.breakout_check_window)
    if bool(recent_break.fillna(False).any()):
        score_volume += 10
        brk_idx = int(recent_break[recent_break.fillna(False)].index[-1])
        forward = close.iloc[brk_idx + 1 : brk_idx + 4]
        if len(forward) > 0 and bool((forward >= close.iloc[brk_idx]).all()):
            score_volume += 5

    full_pct = max(1e-9, float(cfg.strength_full_score_percentile))
    long_score = 8.0 if strength_long_pct >= full_pct else 8.0 * max(0.0, strength_long_pct) / full_pct
    short_score = 7.0 if strength_short_pct >= full_pct else 7.0 * max(0.0, strength_short_pct) / full_pct
    score_strength = float(long_score + short_score)

    score_pullback = 0
    pullback_note = ""
    hh20 = close.rolling(20, min_periods=20).max()
    peak20 = float(hh20.iloc[-1]) if pd.notna(hh20.iloc[-1]) else float("nan")
    in_pullback = bool(pd.notna(peak20) and close.iloc[-1] < peak20 * cfg.pullback_trigger_ratio)
    if not in_pullback:
        near_high = bool(pd.notna(peak20) and close.iloc[-1] >= peak20 * cfg.pullback_near_high_ratio)
        ext = float(close.iloc[-1] / ma50.iloc[-1] - 1) if pd.notna(ma50.iloc[-1]) and ma50.iloc[-1] > 0 else 0.0
        if ext > cfg.pullback_overextension_ratio:
            score_pullback = 2
            pullback_note = "no_pullback_overextended"
        elif near_high:
            score_pullback = 5
            pullback_note = "no_pullback_near_high"
        else:
            score_pullback = 10
            pullback_note = "no_pullback"
    else:
        diff = close.diff()
        tail_close = close.tail(cfg.breakout_check_window)
        tail_vol = volume.tail(cfg.breakout_check_window)
        tail_diff = diff.tail(cfg.breakout_check_window)
        up_vol = tail_vol[tail_diff > 0]
        down_vol = tail_vol[tail_diff < 0]
        if len(up_vol) > 0 and len(down_vol) > 0 and float(down_vol.mean()) < float(up_vol.mean()):
            score_pullback += 5
        if pd.notna(ma50.iloc[-1]) and bool(close.iloc[-1] >= ma50.iloc[-1]):
            score_pullback += 5
        hh60 = tail_close.max()
        drawdown = 1 - float(close.iloc[-1] / hh60) if hh60 > 0 else 1.0
        if drawdown <= cfg.pullback_drawdown_limit:
            score_pullback += 5

    if cfg.enable_extension_filter and pd.notna(ma50.iloc[-1]) and ma50.iloc[-1] > 0:
        if float(close.iloc[-1] / ma50.iloc[-1]) > float(cfg.extension_ratio_threshold):
            score_pullback = float(score_pullback) - float(cfg.extension_pullback_penalty)

    flags: list[str] = []
    risk_penalty = 0
    w = cfg.risk_window

    churn_cond = (
        (volume.tail(w) >= cfg.risk_churn_vol_mult * vol_sma20.tail(w))
        & (((close.tail(w) - open_.tail(w)) / open_.tail(w)) < 0.01)
    )
    churn_mask = churn_cond.fillna(False).reindex(x.index, fill_value=False).astype(bool)
    if bool(churn_mask.any()):
        flags.append("churn")
        risk_penalty -= 5
        churn_dates = x.loc[churn_mask, "date"]
        churn_date = pd.to_datetime(churn_dates.iloc[-1]).strftime("%Y-%m-%d") if len(churn_dates) > 0 else ""
    else:
        churn_date = ""

    body_top = pd.concat([open_, close], axis=1).max(axis=1)
    upper_shadow = high - body_top
    candle_range = (high - low).replace(0, pd.NA)
    upper_ratio = upper_shadow / candle_range
    upper_cnt = int((upper_ratio.tail(w) > 0.6).fillna(False).sum())
    if upper_cnt >= 2:
        flags.append("upper_shadow")
        risk_penalty -= 5
        upper_idx = upper_ratio.tail(w)[upper_ratio.tail(w) > 0.6].index
        upper_date = pd.to_datetime(x.loc[upper_idx[-1], "date"]).strftime("%Y-%m-%d") if len(upper_idx) > 0 else ""
    else:
        upper_date = ""

    if pd.notna(ma50.iloc[-1]) and pd.notna(vol_sma20.iloc[-1]):
        if bool(close.iloc[-1] < ma50.iloc[-1] and volume.iloc[-1] >= cfg.risk_breakdown_vol_mult * vol_sma20.iloc[-1]):
            flags.append("breakdown")
            risk_penalty -= 5
            breakdown_date = pd.to_datetime(x["date"].iloc[-1]).strftime("%Y-%m-%d")
        else:
            breakdown_date = ""
    else:
        breakdown_date = ""

    risk_detail = risk_exit_flags(x, risk_cfg)
    risk_flags = ",".join(flags) if flags else "none"
    risk_reason_parts: list[str] = []
    if churn_date:
        risk_reason_parts.append(f"churn@{churn_date}")
    if upper_date:
        risk_reason_parts.append(f"upper_shadow@{upper_date}")
    if breakdown_date:
        risk_reason_parts.append(f"breakdown@{breakdown_date}")
    risk_reason = ",".join(risk_reason_parts) if risk_reason_parts else "none"

    comp = {
        "trend": float(score_trend),
        "structure": float(score_structure),
        "volume": float(score_volume),
        "strength": float(score_strength),
        "pullback": float(score_pullback),
    }
    top3 = sorted(comp.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:3]
    score_reason_top3 = "+".join([k for k, _ in top3])

    score_total_raw = score_trend + score_structure + score_volume + score_strength + score_pullback + risk_penalty
    score_total = float(max(0, min(100, score_total_raw)))

    return {
        "score_total": score_total,
        "score_trend": float(score_trend),
        "score_structure": float(score_structure),
        "score_volume": float(score_volume),
        "score_strength": float(score_strength),
        "score_pullback": float(score_pullback),
        "score_strength_pct63": float(strength_long_pct),
        "score_strength_pct21": float(strength_short_pct),
        "risk_penalty": float(risk_penalty),
        "risk_flags": risk_flags,
        "risk_reason": risk_reason,
        "score_reason_top3": score_reason_top3,
        "pullback_note": pullback_note,
        "last_date": pd.to_datetime(x["date"].iloc[-1]).strftime("%Y-%m-%d"),
        "close": float(close.iloc[-1]),
        "ma50": float(ma50.iloc[-1]) if pd.notna(ma50.iloc[-1]) else float("nan"),
        "ma250": float(ma250.iloc[-1]) if pd.notna(ma250.iloc[-1]) else float("nan"),
        "risk_reduce": bool(risk_detail["risk_reduce"]),
        "risk_exit_ma60": bool(risk_detail["risk_exit_ma60"]),
        "risk_drawdown": float(risk_detail["risk_drawdown"]),
        "risk_exit_drawdown": bool(risk_detail["risk_exit_drawdown"]),
    }


def _series_stats(series: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
        }
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p10": float(s.quantile(0.10)),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
    }


def run_score(
    date: str,
    universe: pd.DataFrame,
    data_provider: ScoreDataProvider,
    config: ScoreConfig,
    logger: logging.Logger,
    risk_config: RiskConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    risk_cfg = risk_config or RiskConfig(
        reduce_on_close_below_ma20_days=2,
        exit_on_close_below_ma60=True,
        exit_on_drawdown=0.12,
        take_profit_protect_half=False,
    )

    rows: list[dict[str, object]] = []
    benchmark = data_provider.get_benchmark_history()
    bm_close = benchmark.sort_values("date").reset_index(drop=True)["close"].astype(float)
    bm_ret63 = _ret_n(bm_close, config.rs_long_window)
    bm_ret21 = _ret_n(bm_close, config.rs_short_window)

    history_ready: list[dict[str, object]] = []
    skipped_history_error = 0
    skipped_empty = 0
    skipped_short = 0
    skipped_indicator_na = 0

    for _, row in universe.iterrows():
        symbol = str(row["symbol"])
        try:
            hist = data_provider.get_stock_history(symbol)
        except Exception as exc:  # noqa: PERF203
            logger.warning("score skip %s due to history error: %s", symbol, exc)
            skipped_history_error += 1
            continue
        if hist is None or hist.empty:
            skipped_empty += 1
            continue
        if len(hist.dropna(subset=["close", "open", "high", "low", "volume"])) < 260:
            skipped_short += 1
            continue
        vol_sma20 = moving_average(hist.sort_values("date").reset_index(drop=True)["volume"].astype(float), config.volume_sma_window)
        if pd.isna(vol_sma20.iloc[-1]):
            skipped_indicator_na += 1
            continue

        history_ready.append({"row": row.to_dict(), "hist": hist, "symbol": symbol})

    if not history_ready:
        empty = pd.DataFrame(columns=score_output_columns())
        summary = {
            "run_date": date,
            "adjust": "qfq",
            "data_last_date": pd.to_datetime(benchmark["date"]).max().strftime("%Y-%m-%d") if not benchmark.empty else "",
            "input_universe_count": int(len(universe)),
            "scored_count": 0,
            "top_n_output": int(config.top_n_output),
            "top_symbols": [],
            "score_distributions": {},
            "risk_flag_counts": {},
            "coverage": {
                "history_ready": 0,
                "skipped_history_error": skipped_history_error,
                "skipped_empty": skipped_empty,
                "skipped_short": skipped_short,
                "skipped_indicator_na": skipped_indicator_na,
            },
        }
        return empty, summary

    strength_rows: list[dict[str, object]] = []
    for item in history_ready:
        hist = item["hist"]
        symbol = str(item["symbol"])
        close = hist.sort_values("date").reset_index(drop=True)["close"].astype(float)
        ex63 = _ret_n(close, config.rs_long_window) - bm_ret63
        ex21 = _ret_n(close, config.rs_short_window) - bm_ret21
        strength_rows.append({"symbol": symbol, "ex63": ex63, "ex21": ex21})

    strength_df = pd.DataFrame(strength_rows)
    strength_df["pct63"] = pct_rank(pd.to_numeric(strength_df["ex63"], errors="coerce").fillna(-9e9))
    strength_df["pct21"] = pct_rank(pd.to_numeric(strength_df["ex21"], errors="coerce").fillna(-9e9))
    pct63_map = {str(r["symbol"]): float(r["pct63"]) for _, r in strength_df.iterrows()}
    pct21_map = {str(r["symbol"]): float(r["pct21"]) for _, r in strength_df.iterrows()}

    for item in history_ready:
        symbol = str(item["symbol"])
        hist = item["hist"]
        row_dict = item["row"]
        score = _score_single_stock(
            hist,
            config,
            risk_cfg,
            strength_long_pct=float(pct63_map.get(symbol, 0.0)),
            strength_short_pct=float(pct21_map.get(symbol, 0.0)),
        )
        out = row_dict
        out.update(score)
        rows.append(out)

    scored = pd.DataFrame(rows)
    scored = scored.sort_values(["score_total", "score_strength", "score_trend"], ascending=[False, False, False]).reset_index(drop=True)
    if "symbol" in scored.columns:
        scored = scored.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)
    top_n = int(config.top_n_output)
    out_df = scored.head(top_n).copy()

    risk_counts: dict[str, int] = {}
    for item in scored["risk_flags"].fillna("none").astype(str):
        for flag in item.split(","):
            key = flag.strip()
            if not key:
                continue
            risk_counts[key] = risk_counts.get(key, 0) + 1

    dist_cols = [
        "score_total",
        "score_trend",
        "score_structure",
        "score_volume",
        "score_strength",
        "score_pullback",
        "risk_penalty",
    ]
    score_dist = {col: _series_stats(scored[col]) for col in dist_cols if col in scored.columns}

    top_symbols = [
        {"symbol": str(r["symbol"]), "score_total": float(r["score_total"])}
        for _, r in out_df.loc[:, ["symbol", "score_total"]].iterrows()
    ]

    summary = {
        "run_date": date,
        "adjust": "qfq",
        "data_last_date": pd.to_datetime(benchmark["date"]).max().strftime("%Y-%m-%d") if not benchmark.empty else "",
        "input_universe_count": int(len(universe)),
        "scored_count": int(len(scored)),
        "top_n_output": top_n,
        "top_symbols": top_symbols,
        "score_distributions": score_dist,
        "risk_flag_counts": risk_counts,
        "coverage": {
            "history_ready": int(len(history_ready)),
            "skipped_history_error": skipped_history_error,
            "skipped_empty": skipped_empty,
            "skipped_short": skipped_short,
            "skipped_indicator_na": skipped_indicator_na,
        },
    }

    return out_df, summary
