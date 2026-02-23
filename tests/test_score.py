import logging
import unittest
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.score_system import ScoreConfig, run_score


@dataclass
class _FakeProvider:
    stock_map: dict[str, pd.DataFrame]
    benchmark: pd.DataFrame

    def get_stock_history(self, symbol: str) -> pd.DataFrame:
        return self.stock_map[symbol]

    def get_benchmark_history(self) -> pd.DataFrame:
        return self.benchmark


def _mk_hist(seed: int, rows: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    drift = np.linspace(0, 25, rows)
    noise = rng.normal(0, 0.6, size=rows).cumsum()
    close = 50 + drift + noise
    open_ = close - rng.normal(0, 0.4, size=rows)
    high = np.maximum(open_, close) + rng.uniform(0.1, 0.9, size=rows)
    low = np.minimum(open_, close) - rng.uniform(0.1, 0.9, size=rows)
    volume = 1.2e6 + rng.uniform(0, 2.5e5, size=rows)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


class ScoreSystemTests(unittest.TestCase):
    def test_extension_filter_switch_reduces_pullback_score(self) -> None:
        universe = pd.DataFrame(
            {
                "symbol": ["000001"],
                "name": ["A"],
                "industry": ["I1"],
            }
        )
        hist = _mk_hist(21)
        hist.loc[len(hist) - 1, "close"] = float(hist["close"].tail(60).mean() * 1.45)
        hist.loc[len(hist) - 1, "open"] = float(hist.loc[len(hist) - 1, "close"] * 0.997)
        hist.loc[len(hist) - 1, "high"] = float(hist.loc[len(hist) - 1, "close"] * 1.01)
        hist.loc[len(hist) - 1, "low"] = float(hist.loc[len(hist) - 1, "close"] * 0.99)

        provider = _FakeProvider(stock_map={"000001": hist}, benchmark=_mk_hist(99))
        cfg_on = ScoreConfig(top_n_output=1, enable_extension_filter=True, extension_ratio_threshold=1.25, extension_pullback_penalty=5)
        cfg_off = ScoreConfig(top_n_output=1, enable_extension_filter=False, extension_ratio_threshold=1.25, extension_pullback_penalty=5)

        c_on, _ = run_score(
            date="2026-02-22",
            universe=universe,
            data_provider=provider,
            config=cfg_on,
            logger=logging.getLogger("test-score"),
        )
        c_off, _ = run_score(
            date="2026-02-22",
            universe=universe,
            data_provider=provider,
            config=cfg_off,
            logger=logging.getLogger("test-score"),
        )
        self.assertEqual(len(c_on), 1)
        self.assertEqual(len(c_off), 1)
        self.assertLess(float(c_on.iloc[0]["score_pullback"]), float(c_off.iloc[0]["score_pullback"]))

    def test_empty_universe_still_returns_schema_columns(self) -> None:
        provider = _FakeProvider(stock_map={}, benchmark=_mk_hist(99))
        candidates, summary = run_score(
            date="2026-02-22",
            universe=pd.DataFrame(columns=["symbol", "name", "industry"]),
            data_provider=provider,
            config=ScoreConfig(top_n_output=5),
            logger=logging.getLogger("test-score"),
        )
        self.assertEqual(len(candidates), 0)
        self.assertIn("score_total", set(candidates.columns))
        self.assertEqual(summary["scored_count"], 0)

    def test_run_score_outputs_required_fields_and_summary(self) -> None:
        universe = pd.DataFrame(
            {
                "symbol": ["000001", "000002", "000003"],
                "name": ["A", "B", "C"],
                "industry": ["I1", "I1", "I2"],
            }
        )
        stock_map = {
            "000001": _mk_hist(1),
            "000002": _mk_hist(2),
            "000003": _mk_hist(3),
        }
        benchmark = _mk_hist(99)
        provider = _FakeProvider(stock_map=stock_map, benchmark=benchmark)

        cfg = ScoreConfig(top_n_output=2)
        candidates, summary = run_score(
            date="2026-02-22",
            universe=universe,
            data_provider=provider,
            config=cfg,
            logger=logging.getLogger("test-score"),
        )

        required_cols = {
            "symbol",
            "name",
            "industry",
            "score_total",
            "score_trend",
            "score_structure",
            "score_volume",
            "score_strength",
            "score_pullback",
            "risk_flags",
            "score_reason_top3",
            "risk_reason",
            "last_date",
            "close",
            "ma50",
            "ma250",
        }
        self.assertTrue(required_cols.issubset(set(candidates.columns)))
        self.assertEqual(len(candidates), 2)
        self.assertTrue(candidates["score_total"].between(0, 100).all())

        self.assertEqual(summary["run_date"], "2026-02-22")
        self.assertEqual(summary["input_universe_count"], 3)
        self.assertEqual(summary["scored_count"], 3)
        self.assertIn("top_symbols", summary)
        self.assertIn("score_distributions", summary)
        self.assertIn("risk_flag_counts", summary)
        self.assertIn("adjust", summary)
        self.assertIn("data_last_date", summary)
        self.assertIn("coverage", summary)

    def test_strength_uses_percentile_ranking(self) -> None:
        universe = pd.DataFrame(
            {
                "symbol": ["000001", "000002", "000003"],
                "name": ["A", "B", "C"],
                "industry": ["I1", "I1", "I2"],
            }
        )

        strong = _mk_hist(11)
        strong["close"] = np.linspace(20, 120, len(strong))
        strong["open"] = strong["close"] - 0.5
        strong["high"] = strong["close"] + 1.0
        strong["low"] = strong["close"] - 1.0

        weak = _mk_hist(12)
        weak["close"] = np.linspace(120, 40, len(weak))
        weak["open"] = weak["close"] + 0.5
        weak["high"] = weak["close"] + 1.0
        weak["low"] = weak["close"] - 1.0

        mid = _mk_hist(13)

        provider = _FakeProvider(
            stock_map={"000001": strong, "000002": weak, "000003": mid},
            benchmark=_mk_hist(99),
        )
        candidates, _ = run_score(
            date="2026-02-22",
            universe=universe,
            data_provider=provider,
            config=ScoreConfig(top_n_output=3),
            logger=logging.getLogger("test-score"),
        )
        score_map = {str(r["symbol"]): float(r["score_strength"]) for _, r in candidates.iterrows()}
        self.assertGreater(score_map["000001"], score_map["000002"])

    def test_run_score_handles_churn_reason_index_alignment(self) -> None:
        universe = pd.DataFrame(
            {
                "symbol": ["000001"],
                "name": ["A"],
                "industry": ["I1"],
            }
        )
        hist = _mk_hist(7)
        for i in range(1, 6):
            hist.loc[len(hist) - i, "volume"] = float(hist["volume"].mean() * 4)
            hist.loc[len(hist) - i, "open"] = float(hist.loc[len(hist) - i, "close"] * 0.999)

        provider = _FakeProvider(stock_map={"000001": hist}, benchmark=_mk_hist(99))
        candidates, _ = run_score(
            date="2026-02-22",
            universe=universe,
            data_provider=provider,
            config=ScoreConfig(top_n_output=1),
            logger=logging.getLogger("test-score"),
        )
        self.assertEqual(len(candidates), 1)
        self.assertIn("risk_reason", candidates.columns)


if __name__ == "__main__":
    unittest.main()
