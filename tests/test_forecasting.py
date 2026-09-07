import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features import create_features
from forecasting import train_and_forecast


def make_train_test_split():
    """Creates sample train and test data with features already applied."""
    dates = pd.date_range(start="2026-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "branch_id": ["B001"] * 30,
        "sales": [100 + i * 3 for i in range(30)]
    })
    df_feat = create_features(df)

    train_df = df_feat.iloc[:25].copy()
    test_df = df_feat.iloc[25:].copy()
    return train_df, test_df


def test_output_has_required_columns():
    """The forecast output must have exactly date, branch_id, predicted_sales."""
    train_df, test_df = make_train_test_split()
    result = train_and_forecast(train_df, test_df)

    assert list(result.columns) == ["date", "branch_id", "predicted_sales"]


def test_output_row_count_matches_test_set():
    """Output should have one prediction per row in the test set."""
    train_df, test_df = make_train_test_split()
    result = train_and_forecast(train_df, test_df)

    assert len(result) == len(test_df)


def test_predicted_sales_are_numeric():
    """Predicted sales values must be numbers, not text or null."""
    train_df, test_df = make_train_test_split()
    result = train_and_forecast(train_df, test_df)

    assert pd.api.types.is_numeric_dtype(result["predicted_sales"])
    assert result["predicted_sales"].isna().sum() == 0


def test_date_format_is_string():
    """Dates in the output should be formatted as YYYY-MM-DD strings."""
    train_df, test_df = make_train_test_split()
    result = train_and_forecast(train_df, test_df)

    sample_date = result["date"].iloc[0]
    assert isinstance(sample_date, str)
    assert len(sample_date) == 10  # YYYY-MM-DD is 10 characters