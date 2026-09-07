import pandas as pd

def prepare_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans types, parses dates, and sorts data chronologically by branch."""
    df_clean = df.copy()
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean = df_clean.sort_values(by=['branch_id', 'date']).reset_index(drop=True)
    return df_clean
