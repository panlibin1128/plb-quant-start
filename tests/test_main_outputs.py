import logging
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.main import _dedupe_universe_by_symbol, _write_reversal_outputs, _write_trend_outputs


class MainOutputTests(unittest.TestCase):
    def test_dedupe_universe_by_symbol_adds_industry_tags(self) -> None:
        df = pd.DataFrame(
            {
                "symbol": ["000001", "000001", "000002"],
                "name": ["A", "A", "B"],
                "industry": ["银行", "金融", "煤炭"],
                "turnover": [100, 200, 50],
            }
        )
        out = _dedupe_universe_by_symbol(df)
        self.assertEqual(len(out), 2)
        row = out[out["symbol"] == "000001"].iloc[0]
        self.assertIn("银行", str(row["industry_tags"]))
        self.assertIn("金融", str(row["industry_tags"]))

    def test_write_trend_outputs_keeps_positive_unique_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            df = pd.DataFrame(
                {
                    "symbol": ["000001", "000001", "000002"],
                    "final_signal": [True, True, False],
                    "industry": ["银行", "金融", "煤炭"],
                }
            )
            _write_trend_outputs(df, {"run_date": "2026-02-23"}, out_dir, "2026-02-23", logging.getLogger("test-main"))
            out_df = pd.read_csv(out_dir / "2026-02-23_trend_candidates.csv")
            self.assertEqual(len(out_df), 1)
            self.assertEqual(out_df["symbol"].astype(str).nunique(), 1)

    def test_write_reversal_outputs_keeps_positive_unique_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            df = pd.DataFrame(
                {
                    "symbol": ["000001", "000001", "000002"],
                    "reversal_signal": [True, True, False],
                    "industry": ["银行", "金融", "煤炭"],
                }
            )
            _write_reversal_outputs(df, {"run_date": "2026-02-23"}, out_dir, "2026-02-23", logging.getLogger("test-main"))
            out_df = pd.read_csv(out_dir / "2026-02-23_reversal_candidates.csv")
            self.assertEqual(len(out_df), 1)
            self.assertEqual(out_df["symbol"].astype(str).nunique(), 1)


if __name__ == "__main__":
    unittest.main()
