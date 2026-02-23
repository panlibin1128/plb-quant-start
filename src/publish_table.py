from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    return df


def _picked_by(row: pd.Series) -> str:
    picked: list[str] = []
    if bool(row.get("trend_hit", False)):
        picked.append("trend")
    if bool(row.get("reversal_hit", False)):
        picked.append("reversal")
    if bool(row.get("score_hit", False)):
        picked.append("score")
    if bool(row.get("value_hit", False)):
        picked.append("value")
    return "|".join(picked) if picked else "none"


def generate_publish_simple_table(output_dir: Path, run_date: str) -> tuple[Path, Path, int]:
    files = {
        "trend": output_dir / f"{run_date}_trend_candidates.csv",
        "reversal": output_dir / f"{run_date}_reversal_candidates.csv",
        "score": output_dir / f"{run_date}_score_candidates.csv",
        "value": output_dir / f"{run_date}_value_candidates.csv",
    }
    for key, file_path in files.items():
        if not file_path.exists():
            raise FileNotFoundError(f"missing {key} candidates file: {file_path}")

    trend = _read_table(files["trend"])
    reversal = _read_table(files["reversal"])
    score = _read_table(files["score"])
    value = _read_table(files["value"])

    base_cols = [c for c in ["symbol", "name", "industry"] if c in trend.columns]
    if not base_cols:
        base_cols = [c for c in ["symbol", "name", "industry"] if c in score.columns]
    if "symbol" not in base_cols:
        raise ValueError("symbol column is required in strategy outputs")

    base = pd.concat(
        [
            trend[[c for c in ["symbol", "name", "industry"] if c in trend.columns]],
            reversal[[c for c in ["symbol", "name", "industry"] if c in reversal.columns]],
            score[[c for c in ["symbol", "name", "industry"] if c in score.columns]],
            value[[c for c in ["symbol", "name", "industry"] if c in value.columns]],
        ],
        ignore_index=True,
    )
    base = base.drop_duplicates(subset=["symbol"], keep="first")

    trend_s = trend[["symbol"]].drop_duplicates().copy()
    trend_s["trend_hit"] = True
    reversal_s = reversal[["symbol"]].drop_duplicates().copy()
    reversal_s["reversal_hit"] = True
    score_s = score[["symbol"]].drop_duplicates().copy()
    score_s["score_hit"] = True
    value_s = value[["symbol"]].drop_duplicates().copy()
    value_s["value_hit"] = True

    merged = (
        base.merge(trend_s, on="symbol", how="left")
        .merge(reversal_s, on="symbol", how="left")
        .merge(score_s, on="symbol", how="left")
        .merge(value_s, on="symbol", how="left")
    )
    for col in ["trend_hit", "reversal_hit", "score_hit", "value_hit"]:
        merged[col] = merged[col].fillna(False).astype(bool)
    merged["picked_by"] = merged.apply(_picked_by, axis=1)
    merged["pick_count"] = merged[["trend_hit", "reversal_hit", "score_hit", "value_hit"]].astype(int).sum(axis=1)
    merged = merged.sort_values(["pick_count", "symbol"], ascending=[False, True]).reset_index(drop=True)

    final_df = merged[["symbol", "name", "industry", "picked_by"]].copy()

    csv_path = output_dir / f"{run_date}_4systems_publish_simple.csv"
    md_path = output_dir / f"{run_date}_4systems_publish_simple.md"
    final_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text("# 四策略简约展示表\n\n" + final_df.to_markdown(index=False), encoding="utf-8")
    return csv_path, md_path, int(len(final_df))


def _build_parser() -> ArgumentParser:
    p = ArgumentParser(description="Generate simplified publish table for 4 strategies")
    p.add_argument("--date", default="today", help="run date, format YYYY-MM-DD or 'today'")
    p.add_argument("--output-dir", default="outputs", help="output directory path")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    run_date = datetime.now().strftime("%Y-%m-%d") if args.date == "today" else args.date
    output_dir = Path(args.output_dir)
    csv_path, md_path, rows = generate_publish_simple_table(output_dir, run_date)
    print(f"rows={rows}")
    print(f"csv={csv_path}")
    print(f"md={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
