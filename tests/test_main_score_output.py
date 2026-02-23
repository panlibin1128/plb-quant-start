import logging
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.main import _write_score_outputs


class MainScoreOutputTests(unittest.TestCase):
    def test_write_score_outputs_empty_dataframe_has_header(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            _write_score_outputs(
                result_df=pd.DataFrame(),
                summary_obj={"run_date": "2026-02-22"},
                output_dir=out_dir,
                run_date="2026-02-22",
                logger=logging.getLogger("test-main"),
            )
            csv_path = out_dir / "2026-02-22_score_candidates.csv"
            df = pd.read_csv(csv_path)
            self.assertIn("score_total", set(df.columns))


if __name__ == "__main__":
    unittest.main()
