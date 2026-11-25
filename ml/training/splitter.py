"""
Chronological train/validation/test splitting.

Splits are made on unique dates, not random rows, so future dates never
enter training. The test set is kept completely separate from optimization.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..config import (
    TRAIN_DATE_RATIO,
    VALIDATION_DATE_RATIO,
    MIN_TRAIN_SAMPLES,
    MIN_VALIDATION_SAMPLES,
    MIN_TEST_SAMPLES,
)


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_dates: tuple
    validation_dates: tuple
    test_dates: tuple

    @property
    def train_start(self):
        return self.train_dates[0] if self.train_dates else None

    @property
    def train_end(self):
        return self.train_dates[-1] if self.train_dates else None

    @property
    def validation_start(self):
        return self.validation_dates[0] if self.validation_dates else None

    @property
    def validation_end(self):
        return self.validation_dates[-1] if self.validation_dates else None

    @property
    def test_start(self):
        return self.test_dates[0] if self.test_dates else None

    @property
    def test_end(self):
        return self.test_dates[-1] if self.test_dates else None


class TemporalSplitter:
    """Chronological splitter using unique calendar dates."""

    def __init__(
        self,
        train_ratio=TRAIN_DATE_RATIO,
        validation_ratio=VALIDATION_DATE_RATIO,
    ):
        if not 0 < train_ratio < validation_ratio < 1:
            raise ValueError(
                "Expected 0 < train_ratio < validation_ratio < 1."
            )
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio

    def split(self, dataframe, date_column="date"):
        if date_column not in dataframe.columns:
            raise ValueError(f"Missing date column: {date_column}")

        df = dataframe.copy()
        df[date_column] = pd.to_datetime(
            df[date_column], errors="raise"
        )

        if df[date_column].isna().any():
            raise ValueError("Date column contains missing dates.")

        sort_columns = [date_column]
        if "mountain" in df.columns:
            sort_columns.append("mountain")
        df = df.sort_values(sort_columns).reset_index(drop=True)

        dates = np.array(
            sorted(df[date_column].dt.normalize().unique())
        )

        if len(dates) < 10:
            raise ValueError(
                f"Only {len(dates)} unique dates are available."
            )

        train_end = max(1, int(np.floor(len(dates) * self.train_ratio)))
        valid_end = max(
            train_end + 1,
            int(np.floor(len(dates) * self.validation_ratio)),
        )
        valid_end = min(valid_end, len(dates) - 1)

        train_dates = dates[:train_end]
        valid_dates = dates[train_end:valid_end]
        test_dates = dates[valid_end:]

        normalized = df[date_column].dt.normalize()

        train = df.loc[normalized.isin(train_dates)].copy()
        validation = df.loc[normalized.isin(valid_dates)].copy()
        test = df.loc[normalized.isin(test_dates)].copy()

        if len(train) < MIN_TRAIN_SAMPLES:
            raise ValueError(f"Training set has only {len(train)} rows.")
        if len(validation) < MIN_VALIDATION_SAMPLES:
            raise ValueError(
                f"Validation set has only {len(validation)} rows."
            )
        if len(test) < MIN_TEST_SAMPLES:
            raise ValueError(f"Test set has only {len(test)} rows.")

        result = TemporalSplit(
            train=train,
            validation=validation,
            test=test,
            train_dates=tuple(train_dates),
            validation_dates=tuple(valid_dates),
            test_dates=tuple(test_dates),
        )

        self._print_summary(result)
        return result
    
    def rolling_split(
        self,
        dataframe,
        season_column="season",
        train_seasons=3,
    ):
        """
        Expanding-window rolling validation by season.

        The final season is reserved for the completely
        untouched test set.

        Example with 6 seasons:

            Fold 1:
                Train:      seasons 1-3
                Validation: season 4

            Fold 2:
                Train:      seasons 1-4
                Validation: season 5

            Final test:
                season 6
        """

        if season_column not in dataframe.columns:
            raise ValueError(
                f"Missing season column: {season_column}"
            )

        df = dataframe.copy()

        # Get seasons in chronological order.
        seasons = sorted(
            df[season_column].dropna().unique()
        )

        if len(seasons) < train_seasons + 2:
            raise ValueError(
                f"Need at least {train_seasons + 2} seasons "
                f"to create rolling validation and reserve "
                f"a final test season, but only found "
                f"{len(seasons)} seasons."
            )

        folds = []

        # Leave the final season untouched for testing.
        for i in range(
            train_seasons,
            len(seasons) - 1,
        ):
            train_seasons_list = seasons[:i]
            validation_season = seasons[i]

            train = df[
                df[season_column].isin(train_seasons_list)
            ].copy()

            validation = df[
                df[season_column] == validation_season
            ].copy()

            if len(train) < MIN_TRAIN_SAMPLES:
                continue

            if len(validation) < MIN_VALIDATION_SAMPLES:
                continue

            fold = TemporalSplit(
                train=train,
                validation=validation,
                test=pd.DataFrame(),
                train_dates=tuple(train_seasons_list),
                validation_dates=(validation_season,),
                test_dates=tuple(),
            )

            folds.append(fold)

        if not folds:
            raise ValueError(
                "No valid rolling validation folds were created."
            )

        print("\n" + "=" * 70)
        print("ROLLING-ORIGIN VALIDATION")
        print("=" * 70)

        for i, fold in enumerate(folds, start=1):
            print(f"\nFold {i}")
            print(
                f"Train:      {fold.train_start} -> "
                f"{fold.train_end}"
            )
            print(
                f"Validation: {fold.validation_start} -> "
                f"{fold.validation_end}"
            )
            print(
                f"Samples:    {len(fold.train):,} train | "
                f"{len(fold.validation):,} validation"
            )

        print(
            f"\nFinal test season reserved: {seasons[-1]}"
        )

        return folds

    @staticmethod
    def _print_summary(split):
        print("\n" + "=" * 70)
        print("TEMPORAL DATA SPLIT")
        print("=" * 70)
        print(
            f"Train:      {len(split.train):>8,} | "
            f"{split.train_start} -> {split.train_end}"
        )
        print(
            f"Validation: {len(split.validation):>8,} | "
            f"{split.validation_start} -> {split.validation_end}"
        )
        print(
            f"Test:       {len(split.test):>8,} | "
            f"{split.test_start} -> {split.test_end}"
        )

    def final_test_split(
        self,
        dataframe,
        season_column="season",
    ):
        if season_column not in dataframe.columns:
            raise ValueError(
                f"Missing season column: {season_column}"
            )

        df = dataframe.copy()

        seasons = sorted(
            df[season_column].dropna().unique()
        )

        if len(seasons) < 2:
            raise ValueError(
                "Need at least 2 seasons for a final test split."
            )

        test_season = seasons[-1]

        test = df[
            df[season_column] == test_season
        ].copy()

        train = df[
            df[season_column] != test_season
        ].copy()

        if len(test) < MIN_TEST_SAMPLES:
            raise ValueError(
                f"Test season {test_season} has only "
                f"{len(test)} rows."
            )

        return train, test

def split_target(split, feature_matrix, target):
    """Align X/y by index and return train/validation/test matrices."""
    common = feature_matrix.index.intersection(target.index)

    X = feature_matrix.loc[common]
    y = target.loc[common]

    train_idx = split.train.index.intersection(common)
    valid_idx = split.validation.index.intersection(common)
    test_idx = split.test.index.intersection(common)

    return (
        X.loc[train_idx], y.loc[train_idx],
        X.loc[valid_idx], y.loc[valid_idx],
        X.loc[test_idx], y.loc[test_idx],
    )