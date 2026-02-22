from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml
from tqdm import tqdm

from .data_fetch import (
    FetchConfig,
    build_industry_map_from_boards,
    fetch_hs300_daily,
    fetch_industry_board_hist,
    fetch_spot,
    fetch_stock_daily,
    load_industry_map,
)
from .filters import (
    RiskConfig,
    TrendConfig,
    industry_strength_by_index,
    industry_strength_by_proxy,
    market_regime_pass,
    risk_exit_flags,
    top_n_per_industry,
    trend_train_track_flags,
)
from .reversal_system import ReversalParams, evaluate_reversal_stock, signal_label
from .rps import (
    compute_fast_proxy_rps,
    compute_fast_proxy_rps50,
    compute_strict_market_rps50_latest_day,
    compute_strict_market_rps_latest_day,
)


def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("akshare_screener")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Industry top-3 + trend/reversal screener")
    p.add_argument("--date", default="today", help="Run date: today or YYYY-MM-DD")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--industry-map", default="", help="Optional CSV with columns: industry,symbol")
    p.add_argument("--self-check", action="store_true", help="Fetch HS300 and compute MAs")
    p.add_argument("--system", choices=["trend", "reversal", "both"], default="both")
    return p.parse_args()


def _norm_run_date(date_str: str) -> str:
    if date_str == "today":
        return datetime.now().strftime("%Y-%m-%d")
    datetime.strptime(date_str, "%Y-%m-%d")
    return date_str


def _ret_n(close: pd.Series, n: int) -> float:
    c = close.dropna()
    if len(c) <= n:
        return float("nan")
    return float(c.iloc[-1] / c.iloc[-(n + 1)] - 1)


def run_self_check(fetch_cfg: FetchConfig, logger: logging.Logger) -> int:
    logger.info("running self-check: fetch HS300 and compute MAs")
    hs300 = fetch_hs300_daily(fetch_cfg, logger)
    c = hs300["close"].dropna()
    if len(c) < 300:
        logger.error("self-check failed: hs300 rows < 300")
        return 1
    ma20 = c.rolling(20).mean().iloc[-1]
    ma60 = c.rolling(60).mean().iloc[-1]
    ma250 = c.rolling(250).mean().iloc[-1]
    logger.info("self-check passed: ma20=%.4f ma60=%.4f ma250=%.4f", ma20, ma60, ma250)
    return 0


def _write_outputs(result_df: pd.DataFrame, summary_obj: Dict[str, object], output_dir: Path, run_date: str, logger: logging.Logger) -> None:
    out_csv = output_dir / f"{run_date}_candidates.csv"
    out_json = output_dir / f"{run_date}_summary.json"
    result_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    out_json.write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("done: %s | %s", out_csv, out_json)


def _load_reversal_rps50_strict(
    spot: pd.DataFrame,
    stock_hist: Dict[str, pd.DataFrame],
    hs300: pd.DataFrame,
    fetch_cfg: FetchConfig,
    logger: logging.Logger,
    universe_limit: int,
) -> pd.DataFrame:
    cache_path = fetch_cfg.cache_dir / "rps50_latest.parquet"
    trade_date = pd.to_datetime(hs300["date"]).max().strftime("%Y-%m-%d")

    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if not cached.empty and "trade_date" in cached.columns:
            if str(cached["trade_date"].iloc[0]) == trade_date:
                return cached[["symbol", "rps50"]].copy()

    universe = spot.sort_values("total_mv", ascending=False).head(universe_limit)
    close_map: Dict[str, pd.Series] = {}
    for symbol in tqdm(universe["symbol"].tolist(), desc="strict rps50 universe", unit="stock"):
        if symbol in stock_hist:
            close_map[symbol] = stock_hist[symbol]["close"]
            continue
        try:
            hist = fetch_stock_daily(symbol, fetch_cfg, logger)
            stock_hist[symbol] = hist
            close_map[symbol] = hist["close"]
        except Exception:
            continue

    strict_df = compute_strict_market_rps50_latest_day(close_map)
    strict_df["trade_date"] = trade_date
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    strict_df.to_parquet(cache_path, index=False)
    return strict_df[["symbol", "rps50"]].copy()


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    cfg.setdefault("enable_reversal_system", True)
    cfg.setdefault("reversal_universe_mode", "industry_top3")
    cfg.setdefault("reversal_rps_mode", "fast_proxy")
    cfg.setdefault(
        "reversal_params",
        {
            "rps50_threshold": 85,
            "near_120d_high_threshold": 0.9,
            "near_250d_high_ratio": 0.8,
        },
    )

    io_cfg = cfg["io"]
    output_dir = Path(io_cfg["output_dir"])
    cache_dir = Path(io_cfg["cache_dir"])
    log_dir = Path(io_cfg["log_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_dir)

    fetch_cfg = FetchConfig(
        cache_dir=cache_dir,
        max_retries=int(cfg["fetch"]["max_retries"]),
        retry_sleep_seconds=float(cfg["fetch"]["retry_sleep_seconds"]),
        request_sleep_seconds=float(cfg["fetch"]["request_sleep_seconds"]),
    )

    if args.self_check:
        return run_self_check(fetch_cfg, logger)

    run_date = _norm_run_date(args.date)
    trend_cfg = TrendConfig(
        hhv_ratio_threshold=float(cfg["trend"]["hhv_ratio_threshold"]),
        rps_sum_threshold=float(cfg["trend"]["rps_sum_threshold"]),
    )
    risk_cfg = RiskConfig(
        reduce_on_close_below_ma20_days=int(cfg["risk"]["reduce_on_close_below_ma20_days"]),
        exit_on_close_below_ma60=bool(cfg["risk"]["exit_on_close_below_ma60"]),
        exit_on_drawdown=float(cfg["risk"]["exit_on_drawdown"]),
        take_profit_protect_half=bool(cfg["risk"]["take_profit_protect_half"]),
    )

    logger.info("step1: fetch hs300")
    hs300 = fetch_hs300_daily(fetch_cfg, logger)
    market_ok, market_flags = market_regime_pass(hs300)

    summary: Dict[str, object] = {
        "run_date": run_date,
        "market_regime_pass": market_ok,
        "market_flags": market_flags,
        "system_mode": args.system,
    }

    if not market_ok:
        summary["message"] = "market regime filter failed, no candidates"
        _write_outputs(pd.DataFrame(), summary, output_dir, run_date, logger)
        return 0

    logger.info("step2: fetch spot list")
    spot = fetch_spot(fetch_cfg, logger)
    industry_map_path = Path(args.industry_map) if args.industry_map else None
    if spot["industry"].isna().all():
        user_map = load_industry_map(industry_map_path)
        if user_map.empty:
            logger.info("spot industry missing, building mapping from industry boards")
            user_map = build_industry_map_from_boards(fetch_cfg, logger)
        if user_map.empty:
            raise RuntimeError("spot has no industry field and no valid industry mapping source")
        spot = spot.drop(columns=["industry"]).merge(user_map, on="symbol", how="left")

    summary["spot_count"] = int(len(spot))

    min_turnover = float(cfg["run"]["min_turnover"])
    top_n = int(cfg["run"]["top_n_per_industry"])
    top3 = top_n_per_industry(spot, top_n, min_turnover)
    summary["after_top_n"] = int(len(top3))
    if top3.empty:
        summary["message"] = "no symbols left after top-n industry filter"
        _write_outputs(pd.DataFrame(), summary, output_dir, run_date, logger)
        return 0

    logger.info("step3: fetch stock history for top candidates")
    stock_hist: Dict[str, pd.DataFrame] = {}
    for symbol in tqdm(top3["symbol"].tolist(), desc="download candidates", unit="stock"):
        try:
            stock_hist[symbol] = fetch_stock_daily(symbol, fetch_cfg, logger)
        except Exception as exc:
            logger.warning("skip %s due to hist error: %s", symbol, exc)

    avail_symbols = set(stock_hist.keys())
    top3 = top3[top3["symbol"].isin(avail_symbols)].copy()
    summary["after_history_available"] = int(len(top3))
    if top3.empty:
        summary["message"] = "no symbols left after history fetch"
        _write_outputs(pd.DataFrame(), summary, output_dir, run_date, logger)
        return 0

    top3_for_reversal = top3.copy()

    trend_result = pd.DataFrame()
    if args.system in {"trend", "both"}:
        logger.info("step4: industry strength")
        ind_mode = str(cfg["industry_strength"].get("mode", "auto"))
        ind_strength = pd.DataFrame()
        if ind_mode in {"auto", "index"}:
            industry_hist: Dict[str, pd.DataFrame] = {}
            industries = sorted(set(top3["industry"].dropna().astype(str).tolist()))
            for industry in tqdm(industries, desc="industry index", unit="industry"):
                try:
                    industry_hist[industry] = fetch_industry_board_hist(industry, fetch_cfg, logger)
                except Exception as exc:
                    logger.warning("industry index unavailable for %s: %s", industry, exc)
            if industry_hist:
                ind_strength = industry_strength_by_index(top3, industry_hist, hs300)
                summary["industry_strength_mode_used"] = "index"

        if ind_strength.empty:
            ind_strength = industry_strength_by_proxy(top3, stock_hist, hs300)
            if ind_strength.empty:
                raise RuntimeError("industry strength computation yielded empty result")
            ind_strength["industry_strength_pass"] = ind_strength["industry_excess60_proxy"] > 0
            summary["industry_strength_mode_used"] = "proxy"

        strong_inds = set(ind_strength[ind_strength["industry_strength_pass"]]["industry"].tolist())
        top3 = top3[top3["industry"].isin(strong_inds)].copy()
        summary["after_industry_strength"] = int(len(top3))
        if top3.empty:
            summary["message"] = "no symbols left after industry strength filter"
            _write_outputs(pd.DataFrame(), summary, output_dir, run_date, logger)
            return 0

        logger.info("step5: compute RPS")
        hs300_ret120 = _ret_n(hs300["close"], 120)
        hs300_ret250 = _ret_n(hs300["close"], 250)
        latest_ret_rows = []
        for symbol in top3["symbol"].tolist():
            close = stock_hist[symbol]["close"]
            latest_ret_rows.append({"symbol": symbol, "ret120": _ret_n(close, 120), "ret250": _ret_n(close, 250)})
        latest_returns = pd.DataFrame(latest_ret_rows)

        rps_mode = cfg["rps"]["mode"]
        if rps_mode == "strict_market":
            universe_limit = int(cfg["run"].get("strict_market_universe_limit", 1200))
            universe = spot.sort_values("total_mv", ascending=False).head(universe_limit)
            universe_map: Dict[str, pd.Series] = {}
            for symbol in tqdm(universe["symbol"].tolist(), desc="strict rps universe", unit="stock"):
                try:
                    universe_map[symbol] = fetch_stock_daily(symbol, fetch_cfg, logger)["close"]
                except Exception:
                    continue
            strict_df = compute_strict_market_rps_latest_day(universe_map)
            rps_df = latest_returns[["symbol"]].merge(strict_df, on="symbol", how="left")
            summary["rps_mode"] = "strict_market"
            summary["strict_universe_size"] = int(len(universe_map))
        else:
            proxy = compute_fast_proxy_rps(latest_returns, hs300_ret120, hs300_ret250)
            rps_df = latest_returns[["symbol"]].copy()
            rps_df["rps120"] = proxy.rps120.values
            rps_df["rps250"] = proxy.rps250.values
            summary["rps_mode"] = "fast_proxy"

        top3 = top3.merge(rps_df, on="symbol", how="left")

        logger.info("step6: trend core + risk flags")
        result_rows = []
        for _, row in tqdm(top3.iterrows(), total=len(top3), desc="evaluate trend", unit="stock"):
            symbol = row["symbol"]
            hist = stock_hist[symbol]
            trend_flags = trend_train_track_flags(hist, trend_cfg, float(row["rps120"]), float(row["rps250"]))
            risk_flags = risk_exit_flags(hist, risk_cfg)
            out = row.to_dict()
            out.update(trend_flags)
            out.update(risk_flags)
            result_rows.append(out)

        trend_result = pd.DataFrame(result_rows)
        summary["final_signals"] = int(trend_result["final_signal"].sum()) if not trend_result.empty else 0
    else:
        summary["after_industry_strength"] = 0
        summary["final_signals"] = 0

    reversal_enabled = bool(cfg.get("enable_reversal_system", True)) and args.system in {"reversal", "both"}
    reversal_result = pd.DataFrame()
    if reversal_enabled:
        logger.info("step7: reversal system")
        reversal_mode = str(cfg.get("reversal_universe_mode", "industry_top3"))
        if reversal_mode == "all_a":
            rev_base = spot[~spot["is_st"]].copy()
            rev_base = rev_base[rev_base["turnover"].fillna(0) >= min_turnover]
            rev_base = rev_base.dropna(subset=["symbol", "total_mv"])
            rev_limit = int(cfg["run"].get("strict_market_universe_limit", 1200))
            rev_base = rev_base.sort_values("total_mv", ascending=False).head(rev_limit)
        else:
            rev_base = top3_for_reversal.copy()

        rev_symbols = rev_base["symbol"].dropna().astype(str).tolist()
        missing = [s for s in rev_symbols if s not in stock_hist]
        for symbol in tqdm(missing, desc="download reversal missing", unit="stock"):
            try:
                stock_hist[symbol] = fetch_stock_daily(symbol, fetch_cfg, logger)
            except Exception:
                continue

        rev_base = rev_base[rev_base["symbol"].isin(stock_hist.keys())].copy()

        hs300_ret50 = _ret_n(hs300["close"], 50)
        rev_ret_rows = []
        for symbol in rev_base["symbol"].tolist():
            close = stock_hist[symbol]["close"]
            rev_ret_rows.append({"symbol": symbol, "ret50": _ret_n(close, 50)})
        rev_returns = pd.DataFrame(rev_ret_rows)

        reversal_rps_mode = str(cfg.get("reversal_rps_mode", "fast_proxy"))
        if reversal_rps_mode == "strict_market":
            strict_rps50 = _load_reversal_rps50_strict(
                spot=spot,
                stock_hist=stock_hist,
                hs300=hs300,
                fetch_cfg=fetch_cfg,
                logger=logger,
                universe_limit=int(cfg["run"].get("strict_market_universe_limit", 1200)),
            )
            rev_rps_df = rev_returns[["symbol"]].merge(strict_rps50, on="symbol", how="left")
            summary["reversal_rps_mode"] = "strict_market"
        else:
            rps50 = compute_fast_proxy_rps50(rev_returns, hs300_ret50)
            rev_rps_df = rev_returns[["symbol"]].copy()
            rev_rps_df["rps50"] = rps50.rps50.values
            summary["reversal_rps_mode"] = "fast_proxy"

        rev_base = rev_base.merge(rev_rps_df, on="symbol", how="left")
        params_cfg = cfg.get("reversal_params", {})
        params = ReversalParams(
            rps50_threshold=float(params_cfg.get("rps50_threshold", 85)),
            near_120d_high_threshold=float(params_cfg.get("near_120d_high_threshold", 0.9)),
        )

        rev_rows = []
        debug_tails: Dict[str, pd.DataFrame] = {}
        for _, row in tqdm(rev_base.iterrows(), total=len(rev_base), desc="evaluate reversal", unit="stock"):
            symbol = str(row["symbol"])
            hist = stock_hist[symbol]
            rps50_value = float(row["rps50"]) if pd.notna(row["rps50"]) else 0.0
            flags, tail_dbg = evaluate_reversal_stock(hist, rps50_value, params)
            out = row.to_dict()
            out.update(flags)
            rev_rows.append(out)
            debug_tails[symbol] = tail_dbg

        reversal_result = pd.DataFrame(rev_rows)
        summary["reversal_signals"] = int(reversal_result["reversal_signal"].sum()) if not reversal_result.empty else 0

        if not reversal_result.empty:
            sample_symbols = random.sample(reversal_result["symbol"].astype(str).unique().tolist(), k=min(3, reversal_result["symbol"].nunique()))
            for symbol in sample_symbols:
                logger.info("reversal_sanity %s\n%s", symbol, debug_tails[symbol].to_string(index=False))
    else:
        summary["reversal_signals"] = 0

    if trend_result.empty and reversal_result.empty:
        _write_outputs(pd.DataFrame(), summary, output_dir, run_date, logger)
        return 0

    if trend_result.empty:
        result = reversal_result.copy()
        result["trend_signal"] = False
    elif reversal_result.empty:
        result = trend_result.copy()
        result["trend_signal"] = result["final_signal"].astype(bool)
        result["reversal_signal"] = False
        result["reversal_first_in_30d"] = False
        result["reversal_rps50"] = 0.0
        result["reversal_A__close_above_ma250"] = False
        result["reversal_B__new_high_50d_in_30d"] = False
        result["reversal_D__rps50_gt_85"] = False
        result["reversal_AA__days_above_ma250_in_30d"] = 0
        result["reversal_AB__high_near_120d_high"] = False
        result["reversal_system_signal"] = False
    else:
        base = trend_result.copy()
        base["trend_signal"] = base["final_signal"].astype(bool)
        rev_cols = [
            "symbol",
            "reversal_signal",
            "reversal_first_in_30d",
            "reversal_rps50",
            "reversal_A__close_above_ma250",
            "reversal_B__new_high_50d_in_30d",
            "reversal_D__rps50_gt_85",
            "reversal_AA__days_above_ma250_in_30d",
            "reversal_AB__high_near_120d_high",
            "reversal_system_signal",
        ]
        rev_merge = reversal_result[rev_cols].copy()
        rev_merge = rev_merge.sort_values(["symbol", "reversal_signal", "reversal_rps50"], ascending=[True, False, False])
        rev_merge = rev_merge.drop_duplicates(subset=["symbol"], keep="first")
        result = base.merge(rev_merge, on="symbol", how="left")
        fill_map = {
            "reversal_signal": False,
            "reversal_first_in_30d": False,
            "reversal_rps50": 0.0,
            "reversal_A__close_above_ma250": False,
            "reversal_B__new_high_50d_in_30d": False,
            "reversal_D__rps50_gt_85": False,
            "reversal_AA__days_above_ma250_in_30d": 0,
            "reversal_AB__high_near_120d_high": False,
            "reversal_system_signal": False,
        }
        for key, val in fill_map.items():
            result[key] = result[key].fillna(val)

    if "trend_signal" not in result.columns:
        result["trend_signal"] = False
    if "trend_system_signal" not in result.columns:
        result["trend_system_signal"] = result["trend_signal"].astype(bool)
    else:
        result["trend_system_signal"] = result["trend_signal"].astype(bool)

    result["signal_label"] = [
        signal_label(bool(t), bool(r))
        for t, r in zip(result["trend_signal"].astype(bool), result["reversal_signal"].astype(bool))
    ]

    label_counts = result["signal_label"].value_counts().to_dict()
    summary["signal_label_counts"] = {str(k): int(v) for k, v in label_counts.items()}
    logger.info("signal_label distribution: %s", summary["signal_label_counts"])

    _write_outputs(result, summary, output_dir, run_date, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
