import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.publish_table import generate_publish_simple_table


class PublishTableTests(unittest.TestCase):
    def test_generate_publish_simple_table(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            run_date = "2026-02-23"

            pd.DataFrame(
                {
                    "symbol": ["000001", "000002"],
                    "name": ["A", "B"],
                    "industry": ["银行", "煤炭"],
                }
            ).to_csv(out / f"{run_date}_trend_candidates.csv", index=False)

            pd.DataFrame(
                {
                    "symbol": ["000001"],
                    "name": ["A"],
                    "industry": ["银行"],
                }
            ).to_csv(out / f"{run_date}_reversal_candidates.csv", index=False)

            pd.DataFrame(
                {
                    "symbol": ["000003", "000002"],
                    "name": ["C", "B"],
                    "industry": ["化工", "煤炭"],
                }
            ).to_csv(out / f"{run_date}_score_candidates.csv", index=False)

            pd.DataFrame(
                {
                    "symbol": ["000003"],
                    "name": ["C"],
                    "industry": ["化工"],
                }
            ).to_csv(out / f"{run_date}_value_candidates.csv", index=False)

            csv_path, md_path, rows = generate_publish_simple_table(out, run_date)

            self.assertEqual(rows, 3)
            self.assertTrue(csv_path.exists())
            self.assertTrue(md_path.exists())

            final_df = pd.read_csv(csv_path)
            self.assertEqual(list(final_df.columns), ["symbol", "name", "industry", "picked_by"])
            self.assertEqual(len(final_df), 3)
            one = final_df[final_df["symbol"].astype(str).str.zfill(6) == "000001"].iloc[0]
            two = final_df[final_df["symbol"].astype(str).str.zfill(6) == "000002"].iloc[0]
            three = final_df[final_df["symbol"].astype(str).str.zfill(6) == "000003"].iloc[0]
            self.assertEqual(one["picked_by"], "trend|reversal")
            self.assertEqual(two["picked_by"], "trend|score")
            self.assertEqual(three["picked_by"], "score|value")


if __name__ == "__main__":
    unittest.main()
