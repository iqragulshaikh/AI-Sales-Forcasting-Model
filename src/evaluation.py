import numpy as np
import pandas as pd


def evaluate_predictions(actual, predicted):
    """
    Compares actual sales values with predicted sales values.
    Returns a dictionary with MAE, RMSE, and sMAPE.
    
    actual: list or array of real sales numbers
    predicted: list or array of predicted sales numbers (same length as actual)
    """
    # Convert to numpy arrays so we can do math operations easily
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)

    # Basic check: both must be the same length
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")

    # MAE (Mean Absolute Error) - average of how far off each prediction was
    mae = np.mean(np.abs(actual - predicted))

    # RMSE (Root Mean Squared Error) - similar to MAE but punishes big errors more
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    # sMAPE (Symmetric Mean Absolute Percentage Error) - error as a percentage
    # We handle the case where actual + predicted = 0 to avoid dividing by zero
    denominator = (np.abs(actual) + np.abs(predicted))
    smape = np.mean(
        np.where(
            denominator == 0,
            0,  # if both actual and predicted are 0, treat error as 0
            2 * np.abs(actual - predicted) / denominator
        )
    ) * 100  # convert to percentage

    return {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "sMAPE": round(float(smape), 2)
    }


def evaluate_by_branch(df, actual_col="sales", predicted_col="predicted_sales", branch_col="branch_id"):
    """
    Runs evaluate_predictions() separately for each branch in a DataFrame.
    Returns a DataFrame with one row per branch showing its metrics.
    
    df: a DataFrame that has actual sales, predicted sales, and branch_id columns
    """
    results = []

    # Loop through each unique branch and evaluate it separately
    for branch in df[branch_col].unique():
        branch_data = df[df[branch_col] == branch]
        metrics = evaluate_predictions(branch_data[actual_col], branch_data[predicted_col])
        metrics["branch_id"] = branch
        results.append(metrics)

    # Return as a clean DataFrame, branch_id column first
    result_df = pd.DataFrame(results)
    result_df = result_df[["branch_id", "MAE", "RMSE", "sMAPE"]]
    return result_df


# Quick manual test - only runs if you execute this file directly
if __name__ == "__main__":
    # Sample data to test with
    actual_sales = [100, 150, 200, 250, 300]
    predicted_sales = [110, 145, 190, 260, 295]

    print("Testing evaluate_predictions()...")
    metrics = evaluate_predictions(actual_sales, predicted_sales)
    print(metrics)

    print("\nTesting evaluate_by_branch()...")
    test_df = pd.DataFrame({
        "branch_id": ["B001", "B001", "B002", "B002"],
        "sales": [100, 150, 200, 250],
        "predicted_sales": [110, 145, 190, 260]
    })
    branch_results = evaluate_by_branch(test_df)
    print(branch_results)