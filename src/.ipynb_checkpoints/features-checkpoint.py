import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generates leakage-safe calendar, lag, rolling, and branch encoding features."""
    df_feat = df.copy()
    
    # Calendar features
    df_feat['day_of_week'] = df_feat['date'].dt.dayofweek
    df_feat['month'] = df_feat['date'].dt.month
    df_feat['day'] = df_feat['date'].dt.day
    df_feat['week_of_year'] = df_feat['date'].dt.isocalendar().week.astype(int)
    df_feat['is_weekend'] = df_feat['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Lag features (grouped by branch_id to avoid cross-branch leakage)
    df_feat['sales_lag_1'] = df_feat.groupby('branch_id')['sales'].shift(1)
    df_feat['sales_lag_7'] = df_feat.groupby('branch_id')['sales'].shift(7)
    df_feat['sales_lag_14'] = df_feat.groupby('branch_id')['sales'].shift(14)
    
    # Rolling features (shifted by 1 to exclude target day)
    df_feat['sales_rolling_mean_7'] = df_feat.groupby('branch_id')['sales'].transform(
        lambda x: x.shift(1).rolling(window=7).mean()
    )
    df_feat['sales_rolling_mean_14'] = df_feat.groupby('branch_id')['sales'].transform(
        lambda x: x.shift(1).rolling(window=14).mean()
    )
    
    # One-hot encoding for branches
    df_feat = pd.get_dummies(df_feat, columns=['branch_id'], prefix='branch', drop_first=False)
    
    return df_feat
