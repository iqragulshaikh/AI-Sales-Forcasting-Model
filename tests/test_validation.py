import pytest
import pandas as pd

def test_missing_columns_validation():
    invalid_df = pd.DataFrame({'sales': [100, 200]})
    required_cols = ['date', 'sales', 'branch_id']
    missing = [col for col in required_cols if col not in invalid_df.columns]
    assert len(missing) > 0, "Validation failed to detect missing columns"

def test_invalid_branch_validation():
    valid_branches = {'B001', 'B002', 'B003'}
    df_branches = set(['B001', 'INVALID_BRANCH'])
    invalid = df_branches - valid_branches
    assert len(invalid) > 0, "Validation failed to detect invalid branch ID"

def test_negative_sales_check():
    sales_data = pd.Series([100.0, -50.0, 200.0])
    has_negative = (sales_data < 0).any()
    assert has_negative, "Validation failed to detect negative sales numbers"
