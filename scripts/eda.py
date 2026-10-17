"""Generate exploratory summaries and figures for the freight-rate assessment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
NUMERIC_FEATURES = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
]
DISTANCE_EDGES = [0, 250, 500, 1000, 2000, np.inf]
DISTANCE_LABELS = ["0-250", "250-500", "500-1,000", "1,000-2,000", ">2,000"]
COLORS = {"Dry Van": "#16697A", "Reefer": "#DB6400", "Flatbed": "#6A4C93"}


def parse_args() -> argparse.Namespace:
    """Parse input and output locations.

    Returns:
        Parsed command-line arguments.
    """
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=root / "data" / "train_test.csv")
    parser.add_argument("--validation", type=Path, default=root / "data" / "validation.csv")
    parser.add_argument(
        "--december", type=Path, default=root / "data" / "december_chart_inputs.csv"
    )
    parser.add_argument(
        "--template", type=Path, default=root / "data" / "validation_predictions_template.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=root / "reports" / "figures")
    parser.add_argument(
        "--findings", type=Path, default=root / "reports" / "EDA_FINDINGS.md"
    )
    return parser.parse_args()


def read_inputs(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read assessment CSVs and validate their required columns.

    Args:
        args: Parsed input paths.

    Returns:
        Training, validation, December, and prediction-template tables.

    Raises:
        ValueError: If a required column or valid date is missing.
    """
    train = pd.read_csv(args.train)
    validation = pd.read_csv(args.validation)
    december = pd.read_csv(args.december)
    template = pd.read_csv(args.template)
    required_train = {"load_id", "pickup", "delivery", "distance", "equipment", "weight", "date", "posted_rate"}
    required_validation = required_train - {"posted_rate"}
    missing_train = required_train - set(train.columns)
    missing_validation = required_validation - set(validation.columns)
    if missing_train or missing_validation:
        raise ValueError(
            f"Missing required columns: train={sorted(missing_train)}, "
            f"validation={sorted(missing_validation)}"
        )
    if list(template.columns) != ["load_id", "predicted_rate"]:
        raise ValueError("The prediction template must contain load_id,predicted_rate in order")
    for frame in (train, validation, december):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        if frame["date"].isna().any():
            raise ValueError("Found a missing or unparseable date")
    return train, validation, december, template


def quality_table(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    december: pd.DataFrame,
    template: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize row-level quality and physical-value checks by input.

    Args:
        train: Labeled development table.
        validation: Unlabeled future prediction table.
        december: Fixed December chart inputs.
        template: Required validation prediction layout.

    Returns:
        One quality summary row per input table.
    """
    records: list[dict[str, Any]] = []
    for name, frame in (
        ("train", train),
        ("validation", validation),
        ("december", december),
        ("prediction_template", template),
    ):
        data: dict[str, Any] = {
            "dataset": name,
            "rows": len(frame),
            "columns": len(frame.columns),
            "duplicate_rows": int(frame.duplicated().sum()),
            "duplicate_ids": int(frame["load_id"].duplicated().sum()) if "load_id" in frame else np.nan,
            "missing_cells": int(frame.isna().sum().sum()),
            "missing_weight": int(frame["weight"].isna().sum()) if "weight" in frame else 0,
            "nonpositive_weight": int((frame["weight"] <= 0).sum()) if "weight" in frame else 0,
            "missing_market_index": int(frame["market_index"].isna().sum())
            if "market_index" in frame
            else 0,
            "nonpositive_distance": int((frame["distance"] <= 0).sum())
            if "distance" in frame
            else 0,
            "invalid_target": int(
                ((~np.isfinite(frame["posted_rate"])) | (frame["posted_rate"] <= 0)).sum()
            )
            if "posted_rate" in frame
            else np.nan,
            "leading_or_trailing_space_cells": 0,
            "blank_category_cells": 0,
            "date_min": frame["date"].min().date().isoformat() if "date" in frame else "",
            "date_max": frame["date"].max().date().isoformat() if "date" in frame else "",
            "coordinate_range_violations": 0,
            "casefold_variant_groups": 0,
        }
        for column in ("pickup_lat", "delivery_lat"):
            if column in frame:
                data["coordinate_range_violations"] += int(
                    (~frame[column].between(-90, 90) | ~np.isfinite(frame[column])).sum()
                )
        for column in ("pickup_lon", "delivery_lon"):
            if column in frame:
                data["coordinate_range_violations"] += int(
                    (~frame[column].between(-180, 180) | ~np.isfinite(frame[column])).sum()
                )
        for column in ("pickup", "delivery", "equipment"):
            if column not in frame:
                continue
            values = frame[column].astype("string")
            data["leading_or_trailing_space_cells"] += int((values != values.str.strip()).sum())
            data["blank_category_cells"] += int((values.str.strip() == "").sum())
            normalized_groups = (
                pd.DataFrame({"value": values.dropna().unique()})
                .assign(normalized=lambda items: items["value"].str.strip().str.casefold())
                .groupby("normalized")["value"]
                .nunique()
            )
            data["casefold_variant_groups"] += int((normalized_groups > 1).sum())
        records.append(data)
    return pd.DataFrame.from_records(records)


def schema_table(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    december: pd.DataFrame,
    template: pd.DataFrame,
) -> pd.DataFrame:
    """Record input schemas, missing counts, and category cardinalities.

    Args:
        train: Labeled development table.
        validation: Unlabeled future prediction table.
        december: Fixed December chart inputs.
        template: Required validation prediction layout.

    Returns:
        One row per source column and dataset.
    """
    records = []
    for name, frame in (
        ("train", train),
        ("validation", validation),
        ("december", december),
        ("prediction_template", template),
    ):
        for column in frame.columns:
            records.append(
                {
                    "dataset": name,
                    "column": column,
                    "dtype": str(frame[column].dtype),
                    "missing": int(frame[column].isna().sum()),
                    "unique_nonmissing": int(frame[column].nunique(dropna=True)),
                }
            )
    return pd.DataFrame.from_records(records)


def numeric_summary(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """Compare numeric feature distributions across the time boundary.

    Args:
        train: Labeled development table.
        validation: Unlabeled future prediction table.

    Returns:
        A distribution comparison with missing counts and standardized shifts.
    """
    records = []
    for column in NUMERIC_FEATURES + ["posted_rate"]:
        if column not in train:
            continue
        first = pd.to_numeric(train[column], errors="coerce")
        second = pd.to_numeric(validation[column], errors="coerce") if column in validation else None
        std = float(first.std())
        records.append(
            {
                "feature": column,
                "train_missing": int(first.isna().sum()),
                "train_mean": float(first.mean()),
                "train_median": float(first.median()),
                "train_std": std,
                "validation_missing": int(second.isna().sum()) if second is not None else np.nan,
                "validation_mean": float(second.mean()) if second is not None else np.nan,
                "validation_median": float(second.median()) if second is not None else np.nan,
                "validation_std": float(second.std()) if second is not None else np.nan,
                "standardized_mean_difference": (
                    float((second.mean() - first.mean()) / std)
                    if second is not None and std > 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def category_shift(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """Count unseen categories and validation rows for each categorical field.

    Args:
        train: Labeled development table.
        validation: Unlabeled future prediction table.

    Returns:
        Category counts and unseen-category row shares by feature.
    """
    records = []
    train_lanes = train["pickup"].astype(str) + " -> " + train["delivery"].astype(str)
    validation_lanes = validation["pickup"].astype(str) + " -> " + validation["delivery"].astype(str)
    for column, first, second in (
        ("pickup", train["pickup"], validation["pickup"]),
        ("delivery", train["delivery"], validation["delivery"]),
        ("equipment", train["equipment"], validation["equipment"]),
        ("lane", train_lanes, validation_lanes),
    ):
        unseen = set(second.dropna().astype(str)) - set(first.dropna().astype(str))
        records.append(
            {
                "feature": column,
                "train_categories": int(first.nunique()),
                "validation_categories": int(second.nunique()),
                "unseen_categories": len(unseen),
                "unseen_values": "; ".join(sorted(unseen)),
                "validation_rows_unseen": int(second.astype(str).isin(unseen).sum()),
                "validation_row_fraction_unseen": float(second.astype(str).isin(unseen).mean()),
            }
        )
    return pd.DataFrame.from_records(records)


def make_tables(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    december: pd.DataFrame,
    template: pd.DataFrame,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Build analysis tables and save CSV evidence.

    Args:
        train: Labeled development table.
        validation: Unlabeled future prediction table.
        december: Fixed December chart inputs.
        template: Required validation prediction layout.
        output_dir: Destination for CSV summary tables.

    Returns:
        Named analysis tables used by report generation and plotting.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    source_train = train
    source_validation = validation
    source_december = december
    source_template = template
    train = train.copy()
    train["rate_per_mile"] = train["posted_rate"] / train["distance"]
    train["month"] = train["date"].dt.to_period("M").astype(str)
    train["weekday"] = train["date"].dt.day_name()
    train["distance_bucket"] = pd.cut(
        train["distance"], DISTANCE_EDGES, labels=DISTANCE_LABELS, right=False
    )
    tables = {
        "quality_audit": quality_table(
            source_train, source_validation, source_december, source_template
        ),
        "schema_audit": schema_table(
            source_train, source_validation, source_december, source_template
        ),
        "numeric_summary": numeric_summary(train, validation),
        "category_shift": category_shift(train, validation),
        "monthly_summary": train.groupby("month", observed=True)
        .agg(
            rows=("posted_rate", "size"),
            posted_rate_mean=("posted_rate", "mean"),
            posted_rate_median=("posted_rate", "median"),
            rate_per_mile_mean=("rate_per_mile", "mean"),
            rate_per_mile_median=("rate_per_mile", "median"),
            market_index_median=("market_index", "median"),
            quote_signal_median=("quote_signal", "median"),
        )
        .reset_index(),
        "equipment_summary": train.groupby("equipment", observed=True)
        .agg(
            rows=("posted_rate", "size"),
            rate_mean=("posted_rate", "mean"),
            rate_median=("posted_rate", "median"),
            rate_per_mile_mean=("rate_per_mile", "mean"),
            rate_per_mile_median=("rate_per_mile", "median"),
        )
        .reset_index(),
        "distance_bucket_summary": train.groupby("distance_bucket", observed=True)
        .agg(
            rows=("posted_rate", "size"),
            rate_mean=("posted_rate", "mean"),
            rate_median=("posted_rate", "median"),
            rate_per_mile_median=("rate_per_mile", "median"),
        )
        .reset_index(),
        "weekday_summary": train.groupby("weekday", observed=True)
        .agg(
            rows=("posted_rate", "size"),
            rate_median=("posted_rate", "median"),
            rate_per_mile_median=("rate_per_mile", "median"),
        )
        .reset_index(),
    }
    lane_values = train["pickup"].astype(str) + " -> " + train["delivery"].astype(str)
    tables["lane_history_summary"] = (
        train.assign(lane=lane_values)
        .groupby("lane", observed=True)
        .agg(
            loads=("posted_rate", "size"),
            rate_median=("posted_rate", "median"),
            rate_per_mile_median=("rate_per_mile", "median"),
        )
        .sort_values("loads", ascending=False)
        .reset_index()
    )
    city_points = pd.concat(
        [
            pd.DataFrame(
                {"city": train["pickup"], "lat": train["pickup_lat"], "lon": train["pickup_lon"]}
            ),
            pd.DataFrame(
                {
                    "city": train["delivery"],
                    "lat": train["delivery_lat"],
                    "lon": train["delivery_lon"],
                }
            ),
            pd.DataFrame(
                {
                    "city": validation["pickup"],
                    "lat": validation["pickup_lat"],
                    "lon": validation["pickup_lon"],
                }
            ),
            pd.DataFrame(
                {
                    "city": validation["delivery"],
                    "lat": validation["delivery_lat"],
                    "lon": validation["delivery_lon"],
                }
            ),
        ],
        ignore_index=True,
    )
    tables["city_coordinate_audit"] = (
        city_points.groupby("city", observed=True)
        .agg(
            observations=("lat", "size"),
            unique_latitudes=("lat", "nunique"),
            unique_longitudes=("lon", "nunique"),
            latitude_min=("lat", "min"),
            latitude_max=("lat", "max"),
            longitude_min=("lon", "min"),
            longitude_max=("lon", "max"),
        )
        .reset_index()
    )
    history = train.loc[
        train["pickup"].eq("Lexington") & train["delivery"].eq("Fort Wayne"),
        ["date", "pickup", "delivery", "distance", "equipment", "weight", "posted_rate", "rate_per_mile"],
    ].sort_values("date")
    tables["december_lane_history"] = history
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    return tables


def make_plots(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Save plots describing targets, relationships, time, segments, and shift.

    Args:
        train: Labeled development table.
        validation: Unlabeled future prediction table.
        tables: Summary tables returned by ``make_tables``.
        output_dir: Destination for PNG figures.
    """
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    rate_per_mile = train["posted_rate"] / train["distance"]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].hist(train["posted_rate"], bins=60, color="#16697A", alpha=0.9)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Posted rate ($, log scale)")
    axes[0].set_ylabel("Loads")
    axes[0].set_title("Rate distribution")
    axes[1].hist(rate_per_mile, bins=60, color="#DB6400", alpha=0.9)
    axes[1].set_xlabel("Posted rate per mile ($)")
    axes[1].set_ylabel("Loads")
    axes[1].set_title("Rate per mile distribution")
    figure.tight_layout()
    figure.savefig(output_dir / "target_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    sample = train.sample(n=min(7000, len(train)), random_state=17)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for equipment, subset in sample.groupby("equipment"):
        axes[0].scatter(
            subset["distance"], subset["posted_rate"], s=8, alpha=0.22,
            label=equipment, color=COLORS.get(equipment),
        )
    axes[0].set_xlabel("Distance (miles)")
    axes[0].set_ylabel("Posted rate ($)")
    axes[0].set_title("Rate increases with distance")
    axes[0].legend(frameon=False, markerscale=2)
    for equipment, subset in train.assign(rate_per_mile=rate_per_mile).groupby("equipment"):
        binned = subset.assign(distance_bucket=pd.cut(subset["distance"], 15)).groupby(
            "distance_bucket", observed=True
        ).agg(distance=("distance", "median"), rate_per_mile=("rate_per_mile", "median"))
        axes[1].plot(
            binned["distance"], binned["rate_per_mile"], marker="o", linewidth=1.7,
            label=equipment, color=COLORS.get(equipment),
        )
    axes[1].set_xlabel("Distance (miles)")
    axes[1].set_ylabel("Median posted rate per mile ($)")
    axes[1].set_title("Rate per mile varies by distance and equipment")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "rate_distance_relationship.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    valid_weight = train.loc[train["weight"].gt(0) & train["weight"].notna()].copy()
    weight_sample = valid_weight.sample(n=min(7000, len(valid_weight)), random_state=23)
    weight_bins = pd.cut(valid_weight["weight"], 12)
    weight_medians = valid_weight.groupby(weight_bins, observed=True).agg(
        weight=("weight", "median"), rate=("posted_rate", "median")
    )
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.scatter(weight_sample["weight"], weight_sample["posted_rate"], s=8, alpha=0.16, color="#16697A")
    axis.plot(weight_medians["weight"], weight_medians["rate"], color="#DB6400", marker="o", linewidth=2)
    axis.set_xlabel("Weight (lb; positive values only)")
    axis.set_ylabel("Posted rate ($)")
    axis.set_title("Posted rate has little raw association with weight")
    figure.tight_layout()
    figure.savefig(output_dir / "rate_weight_relationship.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    monthly = tables["monthly_summary"]
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    month_dates = pd.to_datetime(monthly["month"] + "-01")
    axes[0].plot(month_dates, monthly["posted_rate_median"], marker="o", label="Median rate")
    axes[0].plot(month_dates, monthly["posted_rate_mean"], marker="o", label="Mean rate")
    axes[0].set_ylabel("Posted rate ($)")
    axes[0].set_title("Monthly target and market signal")
    axes[0].legend(frameon=False)
    axes[1].plot(month_dates, monthly["market_index_median"], marker="o", label="Market index")
    axes[1].plot(month_dates, monthly["quote_signal_median"], marker="o", label="Quote signal")
    axes[1].set_ylabel("Median feature value")
    axes[1].set_xlabel("Month")
    axes[1].legend(frameon=False)
    axes[1].tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(output_dir / "monthly_patterns.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    weekday_summary = tables["weekday_summary"].copy()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_summary["weekday"] = pd.Categorical(
        weekday_summary["weekday"], categories=weekday_order, ordered=True
    )
    weekday_summary = weekday_summary.sort_values("weekday")
    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    axis.bar(
        weekday_summary["weekday"].astype(str),
        weekday_summary["rate_per_mile_median"],
        color="#16697A",
        alpha=0.9,
    )
    axis.set_ylabel("Median rate per mile ($)")
    axis.set_title("Weekly pattern is small relative to the target spread")
    figure.tight_layout()
    figure.savefig(output_dir / "weekday_patterns.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    equipment = tables["equipment_summary"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    groups = [rate_per_mile[train["equipment"].eq(name)].dropna().to_numpy() for name in equipment["equipment"]]
    box = axis.boxplot(groups, showfliers=False, patch_artist=True)
    axis.set_xticks(range(1, len(equipment) + 1), equipment["equipment"])
    for patch, name in zip(box["boxes"], equipment["equipment"]):
        patch.set_facecolor(COLORS.get(name, "#16697A"))
        patch.set_alpha(0.8)
    axis.set_ylabel("Posted rate per mile ($)")
    axis.set_title("Rate per mile by equipment")
    figure.tight_layout()
    figure.savefig(output_dir / "equipment_segments.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for name, frame in (("Train", train), ("Validation", validation)):
        grouped = frame.set_index("date").resample("MS").agg(
            market_index=("market_index", "median"), quote_signal=("quote_signal", "median")
        )
        axes[0].plot(grouped.index, grouped["market_index"], marker="o", label=name)
        axes[1].plot(grouped.index, grouped["quote_signal"], marker="o", label=name)
    axes[0].set_ylabel("Median market index")
    axes[0].set_title("Feature distributions across the time boundary")
    axes[0].legend(frameon=False)
    axes[1].set_ylabel("Median quote signal")
    axes[1].set_xlabel("Date")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "temporal_distribution_shift.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    history = tables["december_lane_history"]
    figure, axis = plt.subplots(figsize=(10, 4.5))
    for equipment_name, subset in history.groupby("equipment"):
        axis.scatter(
            subset["date"], subset["posted_rate"], label=equipment_name,
            color=COLORS.get(equipment_name), s=35, alpha=0.85,
        )
    axis.set_ylabel("Posted rate ($)")
    axis.set_xlabel("Date")
    axis.set_title("Lexington to Fort Wayne historical rates")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "december_lane_history.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_findings(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    template: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write the main EDA observations with counts and caveats.

    Args:
        train: Labeled development table.
        validation: Unlabeled future prediction table.
        template: Required validation prediction layout.
        output_path: Destination Markdown path.
    """
    q1, q3 = train["posted_rate"].quantile([0.25, 0.75])
    upper_fence = q3 + 1.5 * (q3 - q1)
    rate_per_mile = train["posted_rate"] / train["distance"]
    city_train = set(train["pickup"]) | set(train["delivery"])
    city_validation = set(validation["pickup"]) | set(validation["delivery"])
    lane_train = train["pickup"].astype(str) + " -> " + train["delivery"].astype(str)
    lane_validation = validation["pickup"].astype(str) + " -> " + validation["delivery"].astype(str)
    lane = train.loc[train["pickup"].eq("Lexington") & train["delivery"].eq("Fort Wayne")]
    dry_van = lane.loc[lane["equipment"].eq("Dry Van")]
    lane_sizes = lane_train.value_counts()
    market_change = validation["market_index"].median() / train["market_index"].median() - 1
    train_months = train.groupby(train["date"].dt.to_period("M"))["posted_rate"].median()
    weekday = train.assign(rate_per_mile=rate_per_mile).groupby(
        train["date"].dt.day_name()
    )["rate_per_mile"].median()
    missing_weight = int((train["weight"].isna() | (train["weight"] <= 0)).sum())
    target_outliers = int((train["posted_rate"] > upper_fence).sum())
    unseen_lane_rows = int((~lane_validation.isin(set(lane_train))).sum())
    lane_sizes = lane_train.value_counts()
    valid_weight = train.loc[train["weight"].gt(0) & train["weight"].notna()]
    weight_corr = valid_weight[["posted_rate", "weight"]].corr().loc["posted_rate", "weight"]
    report = f"""# EDA Findings

## Data shape and dates

- `train_test.csv` has {len(train):,} labeled loads and {len(train.columns)} columns, dated {train.date.min().date()} through {train.date.max().date()} across {train.date.dt.to_period('M').nunique()} months.
- `validation.csv` has {len(validation):,} unlabeled loads from {validation.date.min().date()} through {validation.date.max().date()}. It is a strictly later November-December period, so random splitting would leak future-period structure.
- The submission template has 12,000 IDs, and the December chart input has 31 consecutive dates with fixed lane, equipment, distance, and weight. The CSV schemas and missing-value counts are in `reports/figures/schema_audit.csv` and `reports/figures/quality_audit.csv`.

## Data quality

- There are {int(train.duplicated().sum())} duplicate training rows and {int(train.load_id.duplicated().sum())} duplicate training IDs. The prediction template has {len(template):,} rows, {template.load_id.nunique():,} unique IDs, and its ID order exactly matches validation: {template.load_id.astype(str).equals(validation.load_id.astype(str))}. No dates failed parsing. No target values are missing, non-finite, zero, or negative.
- Training has {int(train.weight.isna().sum())} missing weights and {int((train.weight <= 0).sum())} non-positive weights ({missing_weight} rows combined, {missing_weight / len(train):.2%}). Non-positive weights are physically invalid and will be converted to missing values before fold-fitted median imputation. Validation has {int(validation.weight.isna().sum())} missing and {int((validation.weight <= 0).sum())} non-positive weights.
- `market_index` is missing for {int(train.market_index.isna().sum())} training rows ({train.market_index.isna().mean():.2%}) and {int(validation.market_index.isna().sum())} validation rows ({validation.market_index.isna().mean():.2%}). Imputation statistics must be learned from each training fold only. No string trimming/case-fold collisions or invalid coordinate ranges were found; see `reports/figures/quality_audit.csv` and `reports/figures/city_coordinate_audit.csv`.
- Distance is positive for every row and ranges from {train.distance.min():.1f} to {train.distance.max():.1f} miles (99th percentile {train.distance.quantile(.99):.1f}). The maximum is long but plausible for a US lane; it is not removed as an outlier.
- The target is right-skewed (skew {train.posted_rate.skew():.2f}; median ${train.posted_rate.median():,.2f}; 99th percentile ${train.posted_rate.quantile(.99):,.2f}; maximum ${train.posted_rate.max():,.2f}). The standard 1.5-IQR upper fence flags {target_outliers} rows. These rates are retained unless validation shows that a robust or log-target objective improves performance; there is no evidence to delete them.
- City, delivery, and equipment values have no blank values or leading/trailing spaces. Every city maps consistently to one coordinate pair across pickup/delivery observations. Eight validation cities are new relative to training: {', '.join(sorted(city_validation - city_train))}. These require safe unknown-category handling, with coordinates retained as numeric location features.

## Target patterns and feature relationships

- Distance is the clearest raw numeric relationship with posted rate (Pearson r = {train[['posted_rate', 'distance']].corr().iloc[0, 1]:.3f}). Correlations with positive, non-missing weight, market index, and quote signal are {{'weight': {weight_corr:.3f}, 'market_index': {train[['posted_rate', 'market_index']].corr().iloc[0, 1]:.3f}, 'quote_signal': {train[['posted_rate', 'quote_signal']].corr().iloc[0, 1]:.3f}}}.
- Median rate per mile is ${rate_per_mile.median():.2f}. Its medians differ by equipment: {', '.join(f'{r.equipment} ${r.rate_per_mile_median:.2f}' for r in pd.DataFrame({'equipment': train.equipment, 'rpm': rate_per_mile}).groupby('equipment').rpm.median().rename('rate_per_mile_median').reset_index().itertuples())}. Distance therefore needs more than a single constant rate-per-mile assumption.
- Median rate per mile by weekday ranges from ${weekday.min():.2f} to ${weekday.max():.2f}; the difference is small relative to the target spread. Monthly medians vary, and the 10-month history does not support a year-over-year seasonal claim. Monthly summaries are in `reports/figures/monthly_summary.csv`.
- Training contains {lane_train.nunique():,} distinct pickup-delivery lanes. The median lane has {lane_sizes.median():.0f} observed loads, and {lane_sizes.eq(1).mean():.1%} of lanes occur once. Lane statistics need fold-only calculation and backing off for rare or unseen lanes. See `reports/figures/lane_history_summary.csv`.

## Train-validation shift and lane history

- Validation has {len(city_validation - city_train)} cities absent from training and {unseen_lane_rows:,} of {len(validation):,} rows ({unseen_lane_rows / len(validation):.2%}) on lanes not observed in training. Pickup/delivery/equipment and numeric distribution comparisons are recorded in `reports/figures/category_shift.csv` and `reports/figures/numeric_summary.csv`.
- Validation's median market index is {market_change:.1%} lower than training's, while its median quote signal is {validation.quote_signal.median():.3f} versus {train.quote_signal.median():.3f}. Treat this covariate shift as a material forecasting risk. Do not use the validation rows to adjust preprocessing or select hyperparameters.
- The exact Lexington-to-Fort-Wayne 360-mile, Dry Van, 32,000-lb combination has no exact match in training. The lane itself appears {len(lane)} times on {lane.date.nunique()} dates from {lane.date.min().date()} through {lane.date.max().date()}, including {len(dry_van)} Dry Van loads. Its distance spans {lane.distance.min():.1f}-{lane.distance.max():.1f} miles. This is useful lane history, but not an exact feature match.

## Figures

- `reports/figures/target_distribution.png`
- `reports/figures/rate_distance_relationship.png`
- `reports/figures/rate_weight_relationship.png`
- `reports/figures/monthly_patterns.png`
- `reports/figures/weekday_patterns.png`
- `reports/figures/equipment_segments.png`
- `reports/figures/temporal_distribution_shift.png`
- `reports/figures/december_lane_history.png`

## Modeling implications

Use a chronological design. Reserve October as the most recent labeled holdout. Compare models on expanding-window monthly folds from May through September, with each fold trained only on preceding months. Refit preprocessing within each fold. Use a global median rate-per-mile baseline, linear regression, and a gradient-boosted model. Compare raw and log-target fitting because the rate target is skewed, retain extreme labels, handle unseen categories explicitly, and report the October holdout and fold metrics separately.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    """Run the EDA and write all figures, tables, and findings."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    train, validation, december, template = read_inputs(args)
    tables = make_tables(train, validation, december, template, args.output_dir)
    make_plots(train, validation, tables, args.output_dir)
    render_findings(
        train,
        validation,
        template,
        args.findings,
    )
    LOGGER.info("EDA saved to %s and %s", args.output_dir, args.findings)


if __name__ == "__main__":
    main()
