# Data Dictionary

## Dataset
`data/sample_sales.csv`

This dataset is an adapted daily sales dataset created from the provided
source retail sales dataset. The source data was aggregated by date and
assigned to three branch identifiers for the restaurant forecasting
prototype.

## Columns

| Column | Type | Required | Unit | Allowed Values | Missing Values | Meaning |
|---|---|---|---|---|---|---|
| date | date | Yes | YYYY-MM-DD | Valid dates | Not allowed | Business date |
| branch_id | string | Yes | N/A | B001, B002, B003 | Not allowed | Restaurant branch identifier |
| sales | float | Yes | Currency units | >= 0 | Not allowed | Daily sales target |
| orders | integer | Yes | Count | >= 0 | Not allowed | Daily order/quantity count |
| promotion | integer | Yes | Binary | 0 or 1 | Not allowed | Promotion indicator |
| holiday | integer | Yes | Binary | 0 or 1 | Not allowed | Holiday indicator |
| store_open | integer | Yes | Binary | 0 or 1 | Not allowed | Whether the branch was open |
| temperature | float | Optional | Degrees | Numeric | Not allowed | Weather-related temperature feature |

## Data Limitations

The source dataset is a retail sales dataset rather than actual restaurant
transaction data. Therefore, the final dataset is an adapted prototype
dataset and must not be presented as real restaurant performance data.

The branch IDs are created for the forecasting prototype.

The holiday indicator and temperature values are synthetic/documented
features because equivalent source fields were not available.

## Validation Rules

- Required columns must exist.
- Dates must be valid.
- Sales cannot be negative.
- Branch ID must be valid.
- Forecast horizon must be greater than zero.
- Forecast horizon must not exceed the documented maximum.
- Empty branch selections must be rejected.
- Invalid sales values must not silently be converted to zero.