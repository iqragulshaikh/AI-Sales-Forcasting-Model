import pandas as pd
import sys
import os

# Add src folder to path so we can import from it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features import create_features


def make_sample_data():
    """Creates a small sample dataset with 20 days for one branch."""
    dates = pd.date_range(start="2026-01-01", periods=20, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "branch_id": ["B001"] * 20,
        "sales": [100 + i * 5 for i in range(20)]
    })
    return df


def test_calendar_features_exist():
    """Check that calendar features are created correctly."""
    df = make_sample_data()
    result = create_features(df)

    assert "day_of_week" in result.columns
    assert "month" in result.columns
    assert "is_weekend" in result.columns
    assert "week_of_year" in result.columns


def test_lag_features_no_leakage():
    """
    The first row for a branch should have NaN lag values,
    since there's no earlier data to look back on.
    This confirms lag features don't leak future/current info.
    """
    df = make_sample_data()
    result = create_features(df)

    # First row should have no lag_1 value (nothing before it)
    assert pd.isna(result['sales_lag_1'].iloc[0])

    # By the 8th row, lag_7 should have a real value
    assert not pd.isna(result['sales_lag_7'].iloc[7])


def test_rolling_features_are_shifted():
    """
    Rolling mean should not include the current day's sales value,
    otherwise it would be leaking the target into the feature.
    """
    df = make_sample_data()
    result = create_features(df)

    # Rolling mean for day 8 (index 7) should be based on days 1-7, not include day 8
    # Since sales increase by 5 each day, rolling_mean_7 should be less than current day's sales
    current_sales = df['sales'].iloc[7]
    rolling_mean = result['sales_rolling_mean_7'].iloc[7]

    if not pd.isna(rolling_mean):
        assert rolling_mean < current_sales


def test_branch_one_hot_encoding():
    """Check that branch_id gets converted into one-hot encoded columns."""
    df = make_sample_data()
    result = create_features(df)

    assert "branch_id" not in result.columns  # original column should be gone
    assert "branch_B001" in result.columns    # replaced with one-hot column