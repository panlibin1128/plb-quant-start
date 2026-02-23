from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Protocol

import pandas as pd

from .indicators import pct_rank


class ValueDataProvider(Protocol):
    def get_financial_snapshot(self, symbol: str) -> dict[str, float] | None:
        ...


@dataclass
class ValueConfig:
    industry_whitelist: list[str] = field(
        default_factory=lambda: ["有色金属", "煤炭", "石油石化", "化工", "农业", "银行", "电力"]
    )
    exclude_high_pe: bool = True
    max_pe: float = 30.0
    min_dividend_yield: float = 0.03
    max_pb: float = 3.0
    min_roe: float = 0.10
    min_operating_cf_ratio: float = 0.8
    top_n_output: int = 30
    degrade_level1_enabled: bool = False
    degrade_level2_on_empty: bool = False
    roe_proxy_max_pe: float = 80.0
    ocf_debt_ratio_guard: float = 0.60
    dividend_missing_level1_max_pe: float = 20.0
    dividend_missing_level1_max_pb: float = 2.0
    dividend_missing_level2_max_pe: float = 15.0
    dividend_missing_level2_max_pb: float = 1.8


def value_output_columns() -> list[str]:
    return [
        "symbol",
        "name",
        "industry",
        "close",
        "pe",
        "pb",
        "roe",
        "dividend_yield",
        "operating_cf_ratio",
        "debt_ratio",
        "net_profit_yoy",
        "value_score_total",
        "value_score_dividend",
        "value_score_pe",
        "value_score_pb",
        "value_score_roe",
        "degrade_level",
        "missing_fields",
        "replacement_tags",
        "risk_flags_value",
    ]


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


def _in_whitelist(industry: object, whitelist: list[str]) -> bool:
    if pd.isna(industry):
        return False
    text = str(industry)
    return any(key in text for key in whitelist)


def _empty_summary(date: str, input_count: int, top_n: int) -> dict[str, object]:
    return {
        "run_date": date,
        "input_universe_count": int(input_count),
        "after_industry_filter": 0,
        "after_financial_filter": 0,
        "scored_count": 0,
        "top_n_output": int(top_n),
        "value_score_distribution": {},
        "coverage": {
            "skipped_financial_missing": 0,
            "skipped_financial_filter": 0,
        },
    }


def run_value(
    date: str,
    universe: pd.DataFrame,
    data_provider: ValueDataProvider,
    config: ValueConfig,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, object]]:
    logger.debug("run_value start: date=%s universe=%s", date, len(universe))
    if universe.empty:
        return pd.DataFrame(columns=value_output_columns()), _empty_summary(date, 0, config.top_n_output)

    base = universe.copy()
    summary: dict[str, object] = {
        "run_date": date,
        "input_universe_count": int(len(base)),
    }

    industry_filtered = base[base["industry"].apply(lambda x: _in_whitelist(x, config.industry_whitelist))].copy()
    summary["after_industry_filter"] = int(len(industry_filtered))
    if industry_filtered.empty:
        s = _empty_summary(date, len(base), config.top_n_output)
        s["after_industry_filter"] = 0
        return pd.DataFrame(columns=value_output_columns()), s

    skipped_financial_missing = 0
    skipped_financial_filter = 0
    replacement_counts: dict[str, int] = {
        "missing_roe_used_proxy": 0,
        "missing_ocf_ratio_used_debt_guard": 0,
        "missing_dividend": 0,
        "earnings_data_missing": 0,
    }

    def _run_level(level: int) -> tuple[pd.DataFrame, int, int, dict[str, int]]:
        level_rows: list[dict[str, object]] = []
        level_missing = 0
        level_filter = 0
        level_repl: dict[str, int] = {
            "missing_roe_used_proxy": 0,
            "missing_ocf_ratio_used_debt_guard": 0,
            "missing_dividend": 0,
            "earnings_data_missing": 0,
        }

        for _, row in industry_filtered.iterrows():
            symbol = str(row["symbol"])
            snap = data_provider.get_financial_snapshot(symbol)
            if snap is None:
                level_missing += 1
                continue

            pe = float(pd.to_numeric(snap.get("pe"), errors="coerce"))
            pb = float(pd.to_numeric(snap.get("pb"), errors="coerce"))
            if pd.isna(pe) or pd.isna(pb):
                level_missing += 1
                continue

            roe = float(pd.to_numeric(snap.get("roe"), errors="coerce"))
            dividend_yield = float(pd.to_numeric(snap.get("dividend_yield"), errors="coerce"))
            operating_cf_ratio = float(pd.to_numeric(snap.get("operating_cf_ratio"), errors="coerce"))
            debt_ratio = float(pd.to_numeric(snap.get("debt_ratio"), errors="coerce"))
            net_profit_yoy = float(pd.to_numeric(snap.get("net_profit_yoy"), errors="coerce"))
            net_profit_ttm = float(pd.to_numeric(snap.get("net_profit_ttm"), errors="coerce"))

            missing_fields: list[str] = []
            replacement_tags: list[str] = []

            if pd.isna(roe):
                missing_fields.append("roe")
                if level >= 1 and pe > 0 and pb > 0 and pe < config.roe_proxy_max_pe:
                    roe = pb / pe
                    replacement_tags.append("missing_roe_used_proxy")
                else:
                    level_missing += 1
                    continue

            if pd.isna(operating_cf_ratio):
                missing_fields.append("operating_cf_ratio")
                if level >= 1 and pd.notna(debt_ratio) and debt_ratio <= config.ocf_debt_ratio_guard and pd.notna(net_profit_ttm) and net_profit_ttm > 0:
                    operating_cf_ratio = config.min_operating_cf_ratio
                    replacement_tags.append("missing_ocf_ratio_used_debt_guard")
                else:
                    level_missing += 1
                    continue

            dividend_replaced = False
            if pd.isna(dividend_yield):
                missing_fields.append("dividend_yield")
                allow_missing_dividend = False
                if level == 1:
                    allow_missing_dividend = bool(pe <= config.dividend_missing_level1_max_pe or pb <= config.dividend_missing_level1_max_pb)
                elif level == 2:
                    allow_missing_dividend = bool(
                        pe <= config.dividend_missing_level2_max_pe
                        and pb <= config.dividend_missing_level2_max_pb
                        and pd.notna(net_profit_ttm)
                        and net_profit_ttm > 0
                    )

                if allow_missing_dividend:
                    dividend_yield = 0.0
                    dividend_replaced = True
                    replacement_tags.append("missing_dividend")
                else:
                    level_missing += 1
                    continue

            if config.exclude_high_pe and pe > config.max_pe:
                level_filter += 1
                continue

            dividend_pass = (dividend_yield >= config.min_dividend_yield) or dividend_replaced
            if pb > config.max_pb or roe < config.min_roe or operating_cf_ratio < config.min_operating_cf_ratio or not dividend_pass:
                level_filter += 1
                continue

            flags: list[str] = []
            if dividend_yield < 0.04:
                flags.append("low_dividend_warning")
            if pd.notna(debt_ratio) and debt_ratio > 0.70:
                flags.append("high_debt_warning")
            if pd.notna(net_profit_yoy):
                if net_profit_yoy < 0:
                    flags.append("earnings_decline_warning")
            else:
                replacement_tags.append("earnings_data_missing")
                if pd.notna(net_profit_ttm) and net_profit_ttm <= 0:
                    flags.append("earnings_decline_warning")

            for tag in replacement_tags:
                if tag in level_repl:
                    level_repl[tag] += 1

            level_rows.append(
                {
                    "symbol": symbol,
                    "name": row.get("name", ""),
                    "industry": row.get("industry", ""),
                    "close": float(row.get("close", float("nan"))),
                    "pe": pe,
                    "pb": pb,
                    "roe": roe,
                    "dividend_yield": dividend_yield,
                    "operating_cf_ratio": operating_cf_ratio,
                    "debt_ratio": debt_ratio,
                    "net_profit_yoy": net_profit_yoy,
                    "degrade_level": int(level),
                    "missing_fields": ",".join(sorted(set(missing_fields))) if missing_fields else "none",
                    "replacement_tags": ",".join(sorted(set(replacement_tags))) if replacement_tags else "none",
                    "risk_flags_value": ",".join(flags) if flags else "none",
                }
            )

        return pd.DataFrame(level_rows), level_missing, level_filter, level_repl

    levels: list[int] = [0]
    if config.degrade_level1_enabled:
        levels.append(1)
    if config.degrade_level2_on_empty:
        levels.append(2)

    level_used = 0
    filtered = pd.DataFrame()
    for level in levels:
        cand, miss_cnt, filter_cnt, repl_cnt = _run_level(level)
        level_used = level
        filtered = cand
        skipped_financial_missing = miss_cnt
        skipped_financial_filter = filter_cnt
        replacement_counts = repl_cnt
        if not filtered.empty:
            break

    summary["after_financial_filter"] = int(len(filtered))
    summary["coverage"] = {
        "skipped_financial_missing": int(skipped_financial_missing),
        "skipped_financial_filter": int(skipped_financial_filter),
        "replacement_counts": replacement_counts,
    }
    summary["degrade_level_used"] = int(level_used)

    if filtered.empty:
        summary["scored_count"] = 0
        summary["value_score_distribution"] = {}
        summary["top_n_output"] = int(config.top_n_output)
        return pd.DataFrame(columns=value_output_columns()), summary

    div_pct = pct_rank(filtered["dividend_yield"]) / 100.0
    pe_pct = pct_rank(filtered["pe"]) / 100.0
    pb_pct = pct_rank(filtered["pb"]) / 100.0
    roe_pct = pct_rank(filtered["roe"]) / 100.0

    filtered["value_score_dividend"] = 0.4 * div_pct
    filtered["value_score_pe"] = 0.3 * (1 - pe_pct)
    filtered["value_score_pb"] = 0.2 * (1 - pb_pct)
    filtered["value_score_roe"] = 0.1 * roe_pct
    filtered["value_score_total"] = (
        filtered["value_score_dividend"]
        + filtered["value_score_pe"]
        + filtered["value_score_pb"]
        + filtered["value_score_roe"]
    )

    filtered = filtered.sort_values("value_score_total", ascending=False).reset_index(drop=True)
    top_n = int(config.top_n_output)
    out_df = filtered.head(top_n).copy()
    out_df = out_df.reindex(columns=value_output_columns())

    summary["scored_count"] = int(len(filtered))
    summary["top_n_output"] = top_n
    summary["value_score_distribution"] = _series_stats(filtered["value_score_total"])
    return out_df, summary


@dataclass
class AkValueDataProvider:
    spot_df: pd.DataFrame
    fetch_financial_abstract: Callable[[str], pd.DataFrame]

    def _extract_from_abstract(self, df: pd.DataFrame, keywords: list[str]) -> float:
        if df is None or df.empty or "指标" not in df.columns:
            return float("nan")
        date_cols = [c for c in df.columns if str(c).isdigit()]
        if not date_cols:
            return float("nan")
        latest_col = sorted(date_cols, reverse=True)[0]
        mask = df["指标"].astype(str).apply(lambda x: any(k in x for k in keywords))
        part = df.loc[mask]
        if part.empty:
            return float("nan")
        val = pd.to_numeric(part[latest_col], errors="coerce").dropna()
        if val.empty:
            return float("nan")
        return float(val.iloc[0])

    def _extract_yoy_from_abstract(self, df: pd.DataFrame, keywords: list[str]) -> float:
        if df is None or df.empty or "指标" not in df.columns:
            return float("nan")
        date_cols = [c for c in df.columns if str(c).isdigit()]
        if len(date_cols) < 2:
            return float("nan")
        desc_cols = sorted(date_cols, reverse=True)
        mask = df["指标"].astype(str).apply(lambda x: any(k in x for k in keywords))
        part = df.loc[mask]
        if part.empty:
            return float("nan")

        first_row = part.iloc[0]
        nums: list[float] = []
        for col in desc_cols:
            val = pd.to_numeric(first_row.get(col), errors="coerce")
            if pd.notna(val):
                nums.append(float(val))
            if len(nums) >= 2:
                break
        if len(nums) < 2:
            return float("nan")

        latest, previous = nums[0], nums[1]
        if abs(previous) <= 1e-12:
            return float("nan")
        return float((latest - previous) / abs(previous))

    def get_financial_snapshot(self, symbol: str) -> dict[str, float] | None:
        symbol6 = str(symbol).zfill(6)
        part = self.spot_df[self.spot_df["symbol"] == symbol6]
        if part.empty:
            return None
        row = part.iloc[0]
        pe = float(pd.to_numeric(row.get("pe", pd.NA), errors="coerce"))
        pb = float(pd.to_numeric(row.get("pb", pd.NA), errors="coerce"))
        abstract = self.fetch_financial_abstract(symbol6)

        roe = self._extract_from_abstract(abstract, ["净资产收益率(ROE)"])
        operating_cf = self._extract_from_abstract(abstract, ["经营现金流量净额", "经营活动产生的现金流量净额"])
        net_profit = self._extract_from_abstract(abstract, ["归母净利润", "净利润"])
        debt_ratio = self._extract_from_abstract(abstract, ["资产负债率"]) / 100.0
        net_profit_yoy_row = self._extract_from_abstract(abstract, ["净利润同比", "净利润增长率"])
        if pd.notna(net_profit_yoy_row):
            net_profit_yoy = float(net_profit_yoy_row / 100.0)
        else:
            net_profit_yoy = self._extract_yoy_from_abstract(abstract, ["归母净利润", "净利润"])

        operating_cf_ratio = float("nan")
        if pd.notna(operating_cf) and pd.notna(net_profit) and abs(net_profit) > 1e-12:
            operating_cf_ratio = float(operating_cf / net_profit)

        dividend_yield = float(pd.to_numeric(row.get("dividend_yield", pd.NA), errors="coerce"))

        return {
            "pe": pe,
            "pb": pb,
            "roe": roe / 100.0 if pd.notna(roe) else float("nan"),
            "dividend_yield": dividend_yield,
            "operating_cf_ratio": operating_cf_ratio,
            "debt_ratio": debt_ratio,
            "net_profit_yoy": net_profit_yoy,
            "net_profit_ttm": net_profit,
        }
