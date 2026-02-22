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
python -m src.main --date today --system both
```

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
