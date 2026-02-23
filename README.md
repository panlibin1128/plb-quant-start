# A-Share Industry Top3 + Trend + Reversal Screener

This project implements a daily A-share stock screener with the fixed workflow:

1. Market regime filter
2. Industry strength filter
3. Top-3 leaders per industry (by total market cap)
4. Trend train-track rules
5. Monthly reversal 5.0 rules (parallel subsystem)
6. Risk exit flags

Data source is AkShare, with local cache and retry logic for stability.

## 1) Setup (WSL2 Ubuntu)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2) Quick self-check

This verifies index fetch + MA calculation.

```bash
python -m src.main --self-check
```

## 3) Run screener

```bash
python -m src.main --date today
```

Run only one subsystem:

```bash
python -m src.main --date today --system trend
python -m src.main --date today --system reversal
python -m src.main --date today --system score
python -m src.main --date today --system value
python -m src.main --date today --system both
```

Value presets:

```bash
python -m src.main --date today --config config.value_v1.yaml --system value
python -m src.main --date today --config config.value_v2.yaml --system value
```

- `config.value_v1.yaml`: conservative value preset (V1)
- `config.value_v2.yaml`: more aggressive value preset (V2)

Validated snapshot (2026-02-23, same market/cache context):

- V1: `scored_count=6` (`outputs/2026-02-23_value_summary.json`)
- V2: `scored_count=13` (`outputs/2026-02-23_value_summary.json` after V2 run)

Or run for a specific date label in output filenames:

```bash
python -m src.main --date 2026-02-21
```

If spot data has no industry field, provide fallback industry mapping:

```bash
python -m src.main --date today --industry-map industry_map.csv
```

`industry_map.csv` must have columns:

- `industry`
- `symbol`

## 4) Output files

- `outputs/YYYY-MM-DD_candidates.csv`
- `outputs/YYYY-MM-DD_summary.json`
- `outputs/YYYY-MM-DD_score_candidates.csv`
- `outputs/YYYY-MM-DD_score_summary.json`
- `outputs/YYYY-MM-DD_value_candidates.csv`
- `outputs/YYYY-MM-DD_value_summary.json`
- `logs/run.log`

## 5) Key config (`config.yaml`)

- `rps.mode`: `fast_proxy` (default) or `strict_market`
- `trend.rps_sum_threshold`: default `185`
- `trend.hhv_ratio_threshold`: default `0.8`
- `run.min_turnover`: liquidity filter
- `run.top_n_per_industry`: default `3`
- `enable_reversal_system`: `true` / `false`
- `reversal_universe_mode`: `industry_top3` or `all_a`
- `reversal_rps_mode`: `fast_proxy` or `strict_market`
- `reversal_params.rps50_threshold`: default `85`
- `reversal_params.near_120d_high_threshold`: default `0.9`
- `score.top_n_output`: score output top N, default `30`
- `score.swing_order`: local extrema order, default `4`
- `score.structure_lookback`: structure lookback days, default `60`
- `score.breakout_check_window`: breakout detection window, default `20`
- `score.pullback_trigger_ratio`: pullback trigger ratio, default `0.95`
- `score.strength_full_score_percentile`: percentile threshold for full strength score, default `80`
- `score.pullback_near_high_ratio`: near-high ratio used in no-pullback branch, default `0.985`
- `score.pullback_overextension_ratio`: overextension ratio vs MA50, default `0.18`
- `score.enable_extension_filter`: whether to penalize overextended price vs MA50, default `true`
- `score.extension_ratio_threshold`: extension threshold for `close/ma50`, default `1.25`
- `score.extension_pullback_penalty`: pullback penalty when extension filter is triggered, default `5`
- `value.industry_whitelist`: production-material industries used by value system
- `value.max_pe`: max PE filter, default `30`
- `value.max_pb`: max PB filter, default `3`
- `value.min_roe`: min ROE filter, default `0.10`
- `value.min_dividend_yield`: min dividend yield filter, default `0.03`
- `value.min_operating_cf_ratio`: min operating cashflow/net-profit ratio, default `0.8`
- `value.top_n_output`: value output top N, default `30`
- `value.degrade_level1_enabled`: enable conservative fallback replacements, default `false`
- `value.degrade_level2_on_empty`: enable stricter empty-result fallback, default `false`
- `value.roe_proxy_max_pe`: ROE proxy guard (`roe_proxy=pb/pe`) max PE, default `80`
- `value.ocf_debt_ratio_guard`: OCF fallback debt guard threshold, default `0.60`
- `value.dividend_missing_level1_max_pe`/`value.dividend_missing_level1_max_pb`: Level1 dividend-missing valuation guard
- `value.dividend_missing_level2_max_pe`/`value.dividend_missing_level2_max_pb`: Level2 stricter guard

Score output key columns:

- `symbol`, `name`, `industry`
- `score_total` (0-100, after risk penalty)
- `score_trend`, `score_structure`, `score_volume`, `score_strength`, `score_pullback`
- `score_strength_pct63`, `score_strength_pct21` (universe percentile strength)
- `risk_penalty`, `risk_flags` (e.g. `churn,upper_shadow`)
- `score_reason_top3` (top scoring component tags), `risk_reason` (flag+date details)
- `last_date`, `close`, `ma50`, `ma250`
- `risk_reduce`, `risk_exit_ma60`, `risk_drawdown`, `risk_exit_drawdown`

Score summary includes:

- `input_universe_count`, `scored_count`, `top_symbols`
- `score_distributions` (`mean`, `median`, `p10`, `p25`, `p75`, `p90`)
- `risk_flag_counts`
- `adjust`, `data_last_date`, `coverage` (history/indicator readiness and skip stats)

Value output key columns:

- `symbol`, `name`, `industry`, `close`
- `pe`, `pb`, `roe`, `dividend_yield`, `operating_cf_ratio`
- `value_score_total`, `value_score_dividend`, `value_score_pe`, `value_score_pb`, `value_score_roe`
- `degrade_level`, `missing_fields`, `replacement_tags`
- `risk_flags_value`

Value summary includes:

- `input_universe_count`, `after_industry_filter`, `after_financial_filter`, `scored_count`
- `value_score_distribution`
- `coverage` (including `skipped_financial_missing`, `replacement_counts`)

Reversal output columns:

- `reversal_signal`
- `reversal_first_in_30d`
- `reversal_rps50`
- `reversal_A__close_above_ma250`
- `reversal_B__new_high_50d_in_30d`
- `reversal_D__rps50_gt_85`
- `reversal_AA__days_above_ma250_in_30d`
- `reversal_AB__high_near_120d_high`
- `trend_signal`
- `signal_label` (`NONE`, `TREND_ONLY`, `REVERSAL_ONLY`, `BOTH`)

Switch to strict RPS:

```yaml
rps:
  mode: strict_market
```

For strict mode speed control:

```yaml
run:
  strict_market_universe_limit: 1200
```

## 6) Notes

- AkShare interfaces may occasionally fail due to upstream rate limit; this project retries and logs fallback behavior.
- Historical data is cached under `data_cache/` as parquet to avoid repeat downloads.
- Main trend rules are preserved as requested:
  - `COUNT(C > MA250, 30) >= 25`
  - `COUNT(C > MA200, 30) >= 25`
  - `COUNT(C > MA20, 10) >= 9`
  - `C / HHV(C, 250) > threshold`
  - `EVERY(MA20 >= REF(MA20,1), 5)`
  - `EVERY(MA10 >= MA20, 5)`
  - `RPS120 + RPS250 > threshold`
- Reversal 5.0 rules are evaluated in parallel and labeled independently.

## 7) Runbook

- Detailed Chinese runbook (logic + next-run commands):
  - `docs/运行手册-双系统.md`

## 8) Current Stable Baselines

- Trend/Reversal/Score existing behavior remains unchanged by value presets.
- Value V1 (`config.value_v1.yaml`) is the accepted conservative baseline.
- Value V2 (`config.value_v2.yaml`) is the accepted more-aggressive baseline.
- Value outputs include explicit fallback traceability columns:
  - `degrade_level`
  - `missing_fields`
  - `replacement_tags`
