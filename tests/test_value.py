import logging
import unittest

import pandas as pd

from src.value_system import AkValueDataProvider, ValueConfig, run_value, value_output_columns


class _FakeValueProvider:
    def __init__(self, snap_map: dict[str, dict[str, float] | None]):
        self.snap_map = snap_map

    def get_financial_snapshot(self, symbol: str) -> dict[str, float] | None:
        return self.snap_map.get(symbol)


class ValueSystemTests(unittest.TestCase):
    def test_industry_filter(self) -> None:
        universe = pd.DataFrame(
            {
                "symbol": ["000001", "000002"],
                "name": ["A", "B"],
                "industry": ["银行", "传媒"],
                "close": [10.0, 11.0],
            }
        )
        provider = _FakeValueProvider(
            {
                "000001": {
                    "pe": 10,
                    "pb": 1.0,
                    "roe": 0.12,
                    "dividend_yield": 0.05,
                    "operating_cf_ratio": 1.0,
                    "debt_ratio": 0.5,
                    "net_profit_yoy": 0.1,
                },
                "000002": {
                    "pe": 10,
                    "pb": 1.0,
                    "roe": 0.12,
                    "dividend_yield": 0.05,
                    "operating_cf_ratio": 1.0,
                    "debt_ratio": 0.5,
                    "net_profit_yoy": 0.1,
                },
            }
        )
        cfg = ValueConfig(industry_whitelist=["银行"], top_n_output=10)
        out, summary = run_value("2026-02-22", universe, provider, cfg, logging.getLogger("test-value"))
        self.assertEqual(summary["after_industry_filter"], 1)
        self.assertEqual(len(out), 1)
        self.assertEqual(str(out.iloc[0]["symbol"]), "000001")

    def test_financial_missing_coverage(self) -> None:
        universe = pd.DataFrame(
            {
                "symbol": ["000001", "000002"],
                "name": ["A", "B"],
                "industry": ["银行", "银行"],
                "close": [10.0, 11.0],
            }
        )
        provider = _FakeValueProvider(
            {
                "000001": None,
                "000002": {
                    "pe": 10,
                    "pb": 1.0,
                    "roe": 0.12,
                    "dividend_yield": 0.05,
                    "operating_cf_ratio": 1.0,
                    "debt_ratio": 0.5,
                    "net_profit_yoy": 0.1,
                },
            }
        )
        out, summary = run_value("2026-02-22", universe, provider, ValueConfig(top_n_output=10), logging.getLogger("test-value"))
        self.assertEqual(len(out), 1)
        cov = summary["coverage"]
        self.assertEqual(cov["skipped_financial_missing"], 1)

    def test_value_score_calculation(self) -> None:
        universe = pd.DataFrame(
            {
                "symbol": ["000001", "000002", "000003"],
                "name": ["A", "B", "C"],
                "industry": ["银行", "银行", "银行"],
                "close": [10.0, 11.0, 12.0],
            }
        )
        provider = _FakeValueProvider(
            {
                "000001": {"pe": 8, "pb": 0.9, "roe": 0.15, "dividend_yield": 0.06, "operating_cf_ratio": 1.2, "debt_ratio": 0.6, "net_profit_yoy": 0.1},
                "000002": {"pe": 15, "pb": 1.5, "roe": 0.12, "dividend_yield": 0.04, "operating_cf_ratio": 1.0, "debt_ratio": 0.6, "net_profit_yoy": 0.1},
                "000003": {"pe": 25, "pb": 2.2, "roe": 0.11, "dividend_yield": 0.031, "operating_cf_ratio": 0.9, "debt_ratio": 0.8, "net_profit_yoy": -0.1},
            }
        )
        out, _ = run_value("2026-02-22", universe, provider, ValueConfig(top_n_output=3), logging.getLogger("test-value"))
        self.assertEqual(len(out), 3)
        self.assertTrue(out["value_score_total"].iloc[0] >= out["value_score_total"].iloc[1] >= out["value_score_total"].iloc[2])
        self.assertIn("value_score_dividend", out.columns)
        self.assertIn("value_score_pe", out.columns)
        self.assertIn("value_score_pb", out.columns)
        self.assertIn("value_score_roe", out.columns)

    def test_risk_flags_value_content(self) -> None:
        universe = pd.DataFrame(
            {
                "symbol": ["000003"],
                "name": ["C"],
                "industry": ["银行"],
                "close": [12.0],
            }
        )
        provider = _FakeValueProvider(
            {
                "000003": {
                    "pe": 12,
                    "pb": 1.2,
                    "roe": 0.12,
                    "dividend_yield": 0.035,
                    "operating_cf_ratio": 1.1,
                    "debt_ratio": 0.75,
                    "net_profit_yoy": -0.2,
                }
            }
        )
        out, _ = run_value("2026-02-22", universe, provider, ValueConfig(top_n_output=5), logging.getLogger("test-value"))
        self.assertEqual(len(out), 1)
        flags = str(out.iloc[0]["risk_flags_value"]).split(",")
        self.assertIn("low_dividend_warning", flags)
        self.assertIn("high_debt_warning", flags)
        self.assertIn("earnings_decline_warning", flags)

    def test_financial_abstract_yoy_fallback(self) -> None:
        spot = pd.DataFrame(
            {
                "symbol": ["000001"],
                "pe": [10.0],
                "pb": [1.0],
                "dividend_yield": [0.05],
            }
        )
        abstract = pd.DataFrame(
            {
                "指标": ["归母净利润", "净资产收益率(ROE)", "资产负债率", "经营现金流量净额"],
                "2024": [100.0, 12.0, 50.0, 120.0],
                "2023": [120.0, 11.0, 49.0, 100.0],
            }
        )
        provider = AkValueDataProvider(spot_df=spot, fetch_financial_abstract=lambda _s: abstract)
        snap = provider.get_financial_snapshot("000001")
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertAlmostEqual(float(snap["net_profit_yoy"]), -1.0 / 6.0, places=6)

    def test_strict_mode_rejects_missing_dividend(self) -> None:
        universe = pd.DataFrame(
            {
                "symbol": ["000001"],
                "name": ["A"],
                "industry": ["银行"],
                "close": [10.0],
            }
        )
        provider = _FakeValueProvider(
            {
                "000001": {
                    "pe": 12,
                    "pb": 1.1,
                    "roe": 0.12,
                    "dividend_yield": float("nan"),
                    "operating_cf_ratio": 1.0,
                    "debt_ratio": 0.5,
                    "net_profit_yoy": 0.1,
                    "net_profit_ttm": 100.0,
                }
            }
        )
        out, _ = run_value("2026-02-22", universe, provider, ValueConfig(top_n_output=5), logging.getLogger("test-value"))
        self.assertEqual(len(out), 0)

    def test_level1_allows_missing_dividend_with_explicit_tags(self) -> None:
        universe = pd.DataFrame(
            {
                "symbol": ["000001"],
                "name": ["A"],
                "industry": ["银行"],
                "close": [10.0],
            }
        )
        provider = _FakeValueProvider(
            {
                "000001": {
                    "pe": 12,
                    "pb": 1.1,
                    "roe": 0.12,
                    "dividend_yield": float("nan"),
                    "operating_cf_ratio": 1.0,
                    "debt_ratio": 0.5,
                    "net_profit_yoy": 0.1,
                    "net_profit_ttm": 100.0,
                }
            }
        )
        cfg = ValueConfig(
            top_n_output=5,
            degrade_level1_enabled=True,
            dividend_missing_level1_max_pe=20.0,
            dividend_missing_level1_max_pb=2.0,
        )
        out, _ = run_value("2026-02-22", universe, provider, cfg, logging.getLogger("test-value"))
        self.assertEqual(len(out), 1)
        self.assertEqual(int(out.iloc[0]["degrade_level"]), 1)
        self.assertIn("dividend_yield", str(out.iloc[0]["missing_fields"]))
        self.assertIn("missing_dividend", str(out.iloc[0]["replacement_tags"]))

    def test_level2_on_empty_works_when_level1_disabled(self) -> None:
        universe = pd.DataFrame(
            {
                "symbol": ["000001"],
                "name": ["A"],
                "industry": ["银行"],
                "close": [10.0],
            }
        )
        provider = _FakeValueProvider(
            {
                "000001": {
                    "pe": 14,
                    "pb": 1.7,
                    "roe": 0.12,
                    "dividend_yield": float("nan"),
                    "operating_cf_ratio": 1.0,
                    "debt_ratio": 0.5,
                    "net_profit_yoy": 0.1,
                    "net_profit_ttm": 100.0,
                }
            }
        )
        cfg = ValueConfig(
            top_n_output=5,
            degrade_level1_enabled=False,
            degrade_level2_on_empty=True,
            dividend_missing_level2_max_pe=15.0,
            dividend_missing_level2_max_pb=1.8,
        )
        out, summary = run_value("2026-02-22", universe, provider, cfg, logging.getLogger("test-value"))
        self.assertEqual(len(out), 1)
        self.assertEqual(int(out.iloc[0]["degrade_level"]), 2)
        self.assertEqual(summary.get("degrade_level_used"), 2)

    def test_empty_output_schema(self) -> None:
        universe = pd.DataFrame(columns=["symbol", "name", "industry", "close"])
        out, summary = run_value("2026-02-22", universe, _FakeValueProvider({}), ValueConfig(), logging.getLogger("test-value"))
        self.assertEqual(len(out), 0)
        self.assertEqual(set(out.columns), set(value_output_columns()))
        self.assertEqual(summary["scored_count"], 0)


if __name__ == "__main__":
    unittest.main()
