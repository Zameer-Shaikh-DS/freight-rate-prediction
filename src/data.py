"""Input loading and validation for the assessment files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = {
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
}
PREDICTION_COLUMNS = ["load_id", "predicted_rate"]
DECEMBER_COLUMNS = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "predicted_rate",
]


def read_csv(path: Path, label: str) -> pd.DataFrame:
    """Read a CSV and add context if parsing fails.

    Args:
        path: CSV file path.
        label: Human-readable input name for errors.

    Returns:
        The loaded table.

    Raises:
        FileNotFoundError: If the CSV file is missing.
        ValueError: If pandas cannot parse the CSV.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not read {label}: {exc}") from exc


def validate_training_data(frame: pd.DataFrame) -> None:
    """Validate required columns, unique IDs, dates, and target values.

    Args:
        frame: Labeled development data.

    Raises:
        ValueError: If required columns or data constraints are not satisfied.
    """
    required = FEATURE_COLUMNS | {"load_id", "posted_rate"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Training data is missing columns: {sorted(missing)}")
    if frame["load_id"].isna().any() or frame["load_id"].duplicated().any():
        raise ValueError("Training load_id values must be present and unique")
    target = pd.to_numeric(frame["posted_rate"], errors="coerce")
    if target.isna().any() or not np.isfinite(target.to_numpy()).all() or (target <= 0).any():
        raise ValueError("posted_rate must contain finite positive values")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Training dates contain invalid values")


def validate_prediction_inputs(
    validation: pd.DataFrame, template: pd.DataFrame, december: pd.DataFrame
) -> None:
    """Check prediction inputs, template IDs, and fixed December rows.

    Args:
        validation: Unlabeled loads to score.
        template: Required load ID output layout.
        december: Fixed December chart inputs.

    Raises:
        ValueError: If any schema, ID, row-count, date, or fixed-value check fails.
    """
    missing = (FEATURE_COLUMNS | {"load_id"}) - set(validation.columns)
    if missing:
        raise ValueError(f"Validation data is missing columns: {sorted(missing)}")
    if len(validation) != 12000 or validation["load_id"].isna().any() or validation["load_id"].duplicated().any():
        raise ValueError("Validation input must contain 12,000 unique, present load IDs")
    if list(template.columns) != PREDICTION_COLUMNS:
        raise ValueError("Prediction template columns must be load_id,predicted_rate")
    if len(template) != 12000:
        raise ValueError("Prediction template must contain 12,000 rows")
    if template["load_id"].isna().any() or template["load_id"].duplicated().any():
        raise ValueError("Prediction template IDs must be present and unique")
    if set(template["load_id"].astype(str)) != set(validation["load_id"].astype(str)):
        raise ValueError("Prediction template IDs do not match validation IDs")
    if list(december.columns) != DECEMBER_COLUMNS or len(december) != 31:
        raise ValueError("December inputs must keep seven expected columns and 31 rows")
    dates = pd.to_datetime(december["date"], errors="coerce")
    expected_dates = pd.date_range("2025-12-01", "2025-12-31", freq="D")
    if dates.isna().any() or set(dates) != set(expected_dates):
        raise ValueError("December inputs must cover every date in December 2025")
    fixed = {
        "pickup": "Lexington",
        "delivery": "Fort Wayne",
        "distance": 360.0,
        "equipment": "Dry Van",
        "weight": 32000.0,
    }
    for column, expected in fixed.items():
        values = december[column]
        matches = np.isclose(values.astype(float), expected) if column in {"distance", "weight"} else values.eq(expected)
        if not matches.all():
            raise ValueError(f"December input {column} does not match the fixed assessment value")
