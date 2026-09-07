import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def train_and_forecast(train_df, test_df):
    """
    Trains the Random Forest model and returns the standardized forecast table.
    """
    # 1. Identify feature columns
    ignore_cols = ['date', 'sales', 'branch_id']
    features = [c for c in train_df.columns if c not in ignore_cols]
    
    X_train = train_df[features]
    y_train = train_df['sales']
    X_test = test_df[features]
    
    # 2. Train Random Forest Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 3. Predict
    test_df_out = test_df.copy()
    test_df_out['predicted_sales'] = model.predict(X_test)
    
    # 4. Standardize branch_id (B001, B002, B003) for team contract
    if 'branch_id' not in test_df_out.columns:
        branch_cols = [c for c in test_df_out.columns if c.startswith('branch_')]
        if branch_cols:
            test_df_out['branch_id'] = test_df_out[branch_cols].idxmax(axis=1).str.replace('branch_', '')
        else:
            test_df_out['branch_id'] = 'B001'
            
    # 5. Extract and format output table
    forecast_output = test_df_out[['date', 'branch_id', 'predicted_sales']].copy()
    
    if pd.api.types.is_datetime64_any_dtype(forecast_output['date']):
        forecast_output['date'] = forecast_output['date'].dt.strftime('%Y-%m-%d')
        
    forecast_output['predicted_sales'] = forecast_output['predicted_sales'].round(2)
    
    return forecast_output[['date', 'branch_id', 'predicted_sales']]
