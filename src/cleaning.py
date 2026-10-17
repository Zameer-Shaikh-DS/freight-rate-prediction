"""Fold-local, leave-one-out target-ratio screening for training rows only."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _loo_group_medians(values: np.ndarray) -> np.ndarray:
    """Return each value's group median after removing that value itself."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    result = np.full(n, np.nan, dtype=float)
    if n <= 1:
        return result
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    remaining = n - 1
    if remaining % 2:
        midpoint = remaining // 2
        source = midpoint + (rank <= midpoint)
        result = sorted_values[source]
    else:
        lower, upper = remaining // 2 - 1, remaining // 2
        source_lower = lower + (rank <= lower)
        source_upper = upper + (rank <= upper)
        result = (sorted_values[source_lower] + sorted_values[source_upper]) / 2.0
    return result


def compute_training_flags(
    training_rows: pd.DataFrame,
    cleaning_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute target-ratio flags using only the passed labeled training rows.

    Expected RPM is the leave-one-out median within equipment-by-distance bucket.
    Sparse buckets fall back to the same equipment's leave-one-out median, then
    to the full passed training frame's leave-one-out median. A row's own target
    never contributes to its expected RPM, including either fallback.
    """
    cfg = cleaning_config or {}
    if not cfg.get("enabled", False):
        return pd.DataFrame(
            {"expected_rpm": np.nan, "expected_rate": np.nan, "ratio": np.nan, "flagged": False},
            index=training_rows.index,
        )
    required = {"equipment", "distance", "posted_rate"}
    missing = required - set(training_rows.columns)
    if missing:
        raise ValueError(f"Cannot flag training rows; missing columns: {sorted(missing)}")
    if training_rows.empty:
        raise ValueError("Cannot compute training flags for an empty frame")

    low_ratio = float(cfg.get("low_ratio", 0.4))
    high_ratio = float(cfg.get("high_ratio", 2.5))
    if not 0 < low_ratio < 1 < high_ratio:
        raise ValueError("Cleaning thresholds must satisfy 0 < low_ratio < 1 < high_ratio")
    edges = [float(v) for v in cfg.get("distance_edges", [0, 250, 500, 1000, 2000])]
    if sorted(edges) != edges or len(edges) < 2:
        raise ValueError("distance_edges must be a strictly increasing list")
    bins = edges + [np.inf]

    data = training_rows[["equipment", "distance", "posted_rate"]].copy()
    data["equipment"] = data["equipment"].astype("string").fillna("<MISSING>")
    data["distance"] = pd.to_numeric(data["distance"], errors="coerce")
    data["posted_rate"] = pd.to_numeric(data["posted_rate"], errors="coerce")
    if data["distance"].isna().any() or (data["distance"] <= 0).any():
        raise ValueError("Cleaning requires finite positive distances")
    if data["posted_rate"].isna().any() or (data["posted_rate"] <= 0).any():
        raise ValueError("Cleaning requires finite positive posted_rate targets")
    data["rpm"] = data["posted_rate"] / data["distance"]
    data["distance_bucket"] = pd.cut(data["distance"], bins=bins, right=False, include_lowest=True)

    expected = pd.Series(np.nan, index=training_rows.index, dtype=float)
    for _, group_index in data.groupby(["equipment", "distance_bucket"], observed=True, sort=False).groups.items():
        ids = np.asarray(list(group_index))
        expected.loc[ids] = _loo_group_medians(data.loc[ids, "rpm"].to_numpy())

    missing_expected = expected.isna()
    if missing_expected.any():
        for _, group_index in data.groupby("equipment", observed=True, sort=False).groups.items():
            ids = np.asarray(list(group_index))
            group_expected = _loo_group_medians(data.loc[ids, "rpm"].to_numpy())
            expected.loc[ids] = expected.loc[ids].fillna(pd.Series(group_expected, index=ids))
    missing_expected = expected.isna()
    if missing_expected.any():
        global_expected = _loo_group_medians(data["rpm"].to_numpy())
        expected.loc[missing_expected] = global_expected[missing_expected.to_numpy()]
    # A single-row training frame has no leave-one-out reference; retain it.
    no_reference = expected.isna()
    expected.loc[no_reference] = data.loc[no_reference, "rpm"]

    expected_rate = expected * data["distance"]
    ratio = data["posted_rate"] / expected_rate
    flagged = ratio.gt(high_ratio) | ratio.lt(low_ratio)
    return pd.DataFrame(
        {"expected_rpm": expected, "expected_rate": expected_rate, "ratio": ratio, "flagged": flagged.astype(bool)},
        index=training_rows.index,
    )
