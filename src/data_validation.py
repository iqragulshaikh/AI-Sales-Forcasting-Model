import pandas as pd

REQUIRED_COLUMNS = [
    "date",
    "branch_id",
    "sales",
    "orders",
    "promotion",
    "holiday",
    "store_open",
    "temperature"
]

VALID_BRANCHES = ["B001", "B002", "B003"]

MIN_HORIZON = 1
MAX_HORIZON = 30


def validate_input(df, branch_id, horizon):

    # Check required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Validate branch
    if branch_id not in VALID_BRANCHES:
        raise ValueError(
            f"Unknown branch_id: {branch_id}"
        )

    # Validate horizon
    if horizon < MIN_HORIZON:
        raise ValueError(
            "Forecast horizon must be greater than 0."
        )

    if horizon > MAX_HORIZON:
        raise ValueError(
            f"Forecast horizon cannot exceed {MAX_HORIZON} days."
        )

    data = df.copy()

    # Validate dates
    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    if data["date"].isna().any():
        raise ValueError(
            "Invalid or missing date values found."
        )

    # Validate sales
    data["sales"] = pd.to_numeric(
        data["sales"],
        errors="coerce"
    )

    if data["sales"].isna().any():
        raise ValueError(
            "Missing or invalid sales values found."
        )

    if (data["sales"] < 0).any():
        raise ValueError(
            "Sales cannot contain negative values."
        )

    # Filter branch
    filtered = data[
        data["branch_id"] == branch_id
    ].copy()

    if filtered.empty:
        raise ValueError(
            f"No data available for branch: {branch_id}"
        )

    # Sort
    filtered = filtered.sort_values(
        "date"
    ).reset_index(drop=True)

    return filtered