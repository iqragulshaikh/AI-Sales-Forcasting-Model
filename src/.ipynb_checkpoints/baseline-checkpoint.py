import pandas as pd

def make_baseline_forecast(df: pd.DataFrame, branch_id: str, horizon: int = 7) -> pd.DataFrame:
    """Generates baseline forecast using the last 7 days average sales."""
    df_branch = df[df['branch_id'] == branch_id].sort_values('date').copy()
    
    if df_branch.empty:
        raise ValueError(f"Branch '{branch_id}' not found in dataset.")
        
    last_date = df_branch['date'].max()
    last_7_days_avg = df_branch.tail(7)['sales'].mean()
    
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon + 1)]
    
    return pd.DataFrame({
        'date': future_dates,
        'branch_id': branch_id,
        'predicted_sales': [last_7_days_avg] * horizon
    })
