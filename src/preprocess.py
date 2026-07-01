"""Preprocess Rossmann Store Sales competition data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS_TO_DROP = [
    "Date",
    "Customers",
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "Promo2SinceWeek",
    "Promo2SinceYear",
    "PromoInterval",
]


def _months_since(year: pd.Series, month: pd.Series, since_year: pd.Series, since_month: pd.Series) -> pd.Series:
    duration = 12 * (year - since_year) + (month - since_month)
    return duration.clip(lower=0)


def _promo2_open_months(
    year: pd.Series,
    week_of_year: pd.Series,
    since_year: pd.Series,
    since_week: pd.Series,
) -> pd.Series:
    duration = 12 * (year - since_year) + (week_of_year - since_week) / 4.0
    return duration.clip(lower=0)


def preprocess_store(store_data: pd.DataFrame) -> pd.DataFrame:
    store = store_data.copy()
    store["CompetitionDistance"] = store["CompetitionDistance"].fillna(store["CompetitionDistance"].median())
    for column in ("CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2SinceWeek", "Promo2SinceYear"):
        store[column] = store[column].fillna(0)
    store["PromoInterval"] = store["PromoInterval"].fillna("")
    return store


def preprocess_sales_frame(frame: pd.DataFrame, store_data: pd.DataFrame, *, is_test: bool = False) -> pd.DataFrame:
    data = frame.copy()
    if is_test:
        data["Open"] = data["Open"].fillna(1)

    data["Date"] = pd.to_datetime(data["Date"], format="%d/%m/%Y")
    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Day"] = data["Date"].dt.day
    data["WeekOfYear"] = data["Date"].dt.isocalendar().week.astype(int)
    data["DayOfYear"] = data["Date"].dt.dayofyear

    data = data.merge(store_data, how="left", on="Store")
    data["CompetitionOpen"] = _months_since(
        data["Year"],
        data["Month"],
        data["CompetitionOpenSinceYear"],
        data["CompetitionOpenSinceMonth"],
    )
    data["Promo2Open"] = _promo2_open_months(
        data["Year"],
        data["WeekOfYear"],
        data["Promo2SinceYear"],
        data["Promo2SinceWeek"],
    )

    return data.drop(columns=[col for col in COLUMNS_TO_DROP if col in data.columns])


def run_preprocessing(raw_dir: Path, processed_dir: Path) -> None:
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    store_path = raw_dir / "store.csv"
    train_path = raw_dir / "train.csv"
    test_path = raw_dir / "test.csv"

    for path in (store_path, train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name}. Download the Kaggle dataset and place train.csv, test.csv, "
                "and store.csv in data/raw/. See README.md for instructions."
            )

    store_data = preprocess_store(pd.read_csv(store_path))
    train_data = preprocess_sales_frame(pd.read_csv(train_path), store_data)
    test_data = preprocess_sales_frame(pd.read_csv(test_path), store_data, is_test=True)

    train_data.to_csv(processed_dir / "train_preprocessed.csv", index=False)
    test_data.to_csv(processed_dir / "test_preprocessed.csv", index=False)
    store_data.to_csv(processed_dir / "store_preprocessed.csv", index=False)

    print(f"Preprocessing complete. Files saved to {processed_dir}")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Preprocess Rossmann Store Sales data.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=project_root / "data" / "raw",
        help="Directory containing store.csv, train.csv, and test.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "processed",
        help="Directory where preprocessed CSV files are written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_preprocessing(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
