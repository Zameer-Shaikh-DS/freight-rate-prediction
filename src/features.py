"""Fold-safe deterministic features for freight-rate modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

CATEGORICAL_FEATURES = ["pickup", "delivery", "equipment", "lane"]
NUMERIC_FEATURES = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "log_distance",
    "weight_per_1000_miles",
    "weight_distance_interaction",
    "latitude_delta",
    "longitude_delta",
    "straight_line_distance",
    "day_of_week",
    "month",
    "week_of_year",
    "day_of_year",
    "days_since_start",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "is_weekend",
    "is_us_federal_holiday",
    "weight_missing",
    "market_index_missing",
    "quote_signal_missing",
    "pickup_lat_missing",
    "pickup_lon_missing",
    "delivery_lat_missing",
    "delivery_lon_missing",
]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
US_FEDERAL_HOLIDAYS = USFederalHolidayCalendar().holidays(
    start="1900-01-01", end="2100-12-31"
)


def coordinate_lookup(training_frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Build a training-only coordinate map for cities without supplied coordinates.

    Args:
        training_frame: Labeled training rows with pickup and delivery coordinates.

    Returns:
        A mapping from city name to its median observed latitude and longitude.
    """
    points = pd.concat(
        [
            training_frame[["pickup", "pickup_lat", "pickup_lon"]].rename(
                columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}
            ),
            training_frame[["delivery", "delivery_lat", "delivery_lon"]].rename(
                columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}
            ),
        ],
        ignore_index=True,
    ).dropna(subset=["city", "lat", "lon"])
    grouped = points.groupby("city", observed=True)[["lat", "lon"]].agg("median")
    return {str(city): (float(row.lat), float(row.lon)) for city, row in grouped.iterrows()}


def _fill_missing_coordinates(
    frame: pd.DataFrame, lookup: dict[str, tuple[float, float]] | None
) -> None:
    """Fill absent coordinate columns from a training-only lookup.

    Args:
        frame: Rows to update in place.
        lookup: City coordinates computed from training rows, if available.
    """
    for role, city_column in (("pickup", "pickup"), ("delivery", "delivery")):
        lat_column, lon_column = f"{role}_lat", f"{role}_lon"
        if lat_column not in frame:
            frame[lat_column] = np.nan
        if lon_column not in frame:
            frame[lon_column] = np.nan
        if lookup is None:
            continue
        missing_lat = frame[lat_column].isna()
        missing_lon = frame[lon_column].isna()
        coordinates = frame[city_column].astype(str).map(lookup)
        if missing_lat.any():
            frame.loc[missing_lat, lat_column] = coordinates.loc[missing_lat].map(
                lambda value: value[0] if isinstance(value, (tuple, list)) else np.nan
            )
        if missing_lon.any():
            frame.loc[missing_lon, lon_column] = coordinates.loc[missing_lon].map(
                lambda value: value[1] if isinstance(value, (tuple, list)) else np.nan
            )


def build_features(
    raw_frame: pd.DataFrame,
    coordinate_reference: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Create row-level features without using targets or fold-wide statistics.

    Args:
        raw_frame: Raw loads with the required lane, date, weight, and distance fields.
        coordinate_reference: Optional city coordinates learned from training rows.

    Returns:
        A model-ready DataFrame with categorical and numeric features.

    Raises:
        ValueError: If required fields are absent or a date/distance is invalid.
    """
    missing = {"pickup", "delivery", "equipment", "distance", "weight", "date"} - set(
        raw_frame.columns
    )
    if missing:
        raise ValueError(f"Cannot build features; missing columns: {sorted(missing)}")
    frame = raw_frame.copy()
    for column in CATEGORICAL_FEATURES[:3]:
        frame[column] = frame[column].astype("string").str.strip().fillna("<MISSING>").astype(str)
    _fill_missing_coordinates(frame, coordinate_reference)
    for column in ("market_index", "quote_signal"):
        if column not in frame:
            frame[column] = np.nan
    date = pd.to_datetime(frame["date"], errors="coerce")
    if date.isna().any():
        raise ValueError("Feature rows contain invalid dates")
    frame["date"] = date

    numeric_columns = [
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "distance",
        "weight",
        "market_index",
        "quote_signal",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if (frame["distance"] <= 0).any() or frame["distance"].isna().any():
        raise ValueError("Distance must be present and positive")
    frame.loc[frame["weight"] <= 0, "weight"] = np.nan

    frame["lane"] = frame["pickup"] + " -> " + frame["delivery"]
    frame["log_distance"] = np.log1p(frame["distance"])
    frame["weight_per_1000_miles"] = frame["weight"] / frame["distance"] * 1000.0
    frame["weight_distance_interaction"] = frame["weight"] * frame["distance"] / 1_000_000.0
    frame["latitude_delta"] = frame["delivery_lat"] - frame["pickup_lat"]
    frame["longitude_delta"] = frame["delivery_lon"] - frame["pickup_lon"]
    lat1 = np.radians(frame["pickup_lat"])
    lat2 = np.radians(frame["delivery_lat"])
    dlat = lat2 - lat1
    dlon = np.radians(frame["delivery_lon"] - frame["pickup_lon"])
    haversine = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    frame["straight_line_distance"] = 3958.8 * 2.0 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))

    day_of_week = frame["date"].dt.dayofweek
    month = frame["date"].dt.month
    day_of_year = frame["date"].dt.dayofyear
    frame["day_of_week"] = day_of_week
    frame["month"] = month
    frame["week_of_year"] = frame["date"].dt.isocalendar().week.astype(int)
    frame["day_of_year"] = day_of_year
    frame["days_since_start"] = (frame["date"] - pd.Timestamp("2025-01-01")).dt.days
    frame["day_of_week_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    frame["day_of_week_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)
    frame["month_sin"] = np.sin(2.0 * np.pi * (month - 1) / 12.0)
    frame["month_cos"] = np.cos(2.0 * np.pi * (month - 1) / 12.0)
    frame["day_of_year_sin"] = np.sin(2.0 * np.pi * (day_of_year - 1) / 365.25)
    frame["day_of_year_cos"] = np.cos(2.0 * np.pi * (day_of_year - 1) / 365.25)
    frame["is_weekend"] = day_of_week.ge(5).astype(int)
    frame["is_us_federal_holiday"] = (
        frame["date"].dt.normalize().isin(US_FEDERAL_HOLIDAYS).astype(int)
    )

    for column in (
        "weight",
        "market_index",
        "quote_signal",
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
    ):
        frame[f"{column}_missing"] = frame[column].isna().astype(int)
    result = frame[MODEL_FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        result[column] = result[column].astype(str)
    return result
