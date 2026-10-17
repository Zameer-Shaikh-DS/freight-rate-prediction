"""Run chronological validation, compare models, refit, and predict submissions."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import load_config, resolve_path
from src.cleaning import compute_training_flags
from src.data import read_csv, validate_prediction_inputs, validate_training_data
from src.features import build_features
from src.models import ModelBundle, fit_model
from src.predict import predict_files

LOGGER = logging.getLogger(__name__)


def regression_metrics(actual: pd.Series | np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compute regression metrics on the original dollar scale.

    Args:
        actual: Observed positive rates in dollars.
        predicted: Finite positive rate predictions in dollars.

    Returns:
        MAE, RMSE, MAPE, sMAPE, and R2 values.

    Raises:
        ValueError: If arrays are empty, have different shapes, or predictions are invalid.
    """
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("Actual and predicted values must have matching non-empty shapes")
    if not np.isfinite(y_pred).all() or (y_pred <= 0).any():
        raise ValueError("Predictions must be finite and positive")
    absolute_error = np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape_pct": float(np.mean(absolute_error / np.maximum(np.abs(y_true), 1e-12)) * 100.0),
        "smape_pct": float(
            np.mean(2.0 * absolute_error / np.maximum(denominator, 1e-12)) * 100.0
        ),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _date_mask(dates: pd.Series, start: Any | None, end: Any) -> pd.Series:
    """Return a date-range mask with inclusive boundaries.

    Args:
        dates: Date values to evaluate.
        start: Inclusive lower date, or ``None`` for no lower bound.
        end: Inclusive upper date.

    Returns:
        Boolean mask aligned to ``dates``.
    """
    end_date = pd.Timestamp(end)
    mask = dates.le(end_date)
    if start is not None:
        mask &= dates.ge(pd.Timestamp(start))
    return mask


def _make_models(
    features: pd.DataFrame,
    target: pd.Series,
    model_config: dict[str, Any],
) -> dict[str, ModelBundle]:
    """Fit the linear comparator and configured CatBoost candidates.

    Args:
        features: Training-fold feature rows.
        target: Training-fold targets in dollars.
        model_config: Configuration for the estimator candidates.

    Returns:
        Fitted model bundles keyed by their stable model names.
    """
    models = {
        "linear_regression": fit_model(
            "linear_regression", "linear", features, target, model_config
        )
    }
    for candidate in model_config["catboost"]["candidates"]:
        name = str(candidate["name"])
        models[name] = fit_model(name, "catboost", features, target, model_config, candidate)
    return models


def _fit_fold_models(
    fit_rows: pd.DataFrame,
    scored_rows: pd.DataFrame,
    fit_features: pd.DataFrame,
    fit_target: pd.Series,
    model_config: dict[str, Any],
    cleaning_config: dict[str, Any],
) -> tuple[dict[str, ModelBundle], dict[str, Any], pd.DataFrame]:
    """Flag/filter only fit rows, then fit estimators without scored-row access."""
    overlap = fit_rows.index.intersection(scored_rows.index)
    if len(overlap):
        raise ValueError("Fit and scored rows overlap")
    screen = compute_training_flags(fit_rows, cleaning_config)
    keep = ~screen["flagged"]
    if not keep.any():
        raise ValueError("Cleaning removed every row in a training fold")
    used_features = fit_features.loc[keep]
    used_target = fit_target.loc[keep]
    models = _make_models(used_features, used_target, model_config)
    audit = {
        "fit_rows_before_cleaning": int(len(fit_rows)),
        "fit_rows_flagged": int(screen["flagged"].sum()),
        "fit_rows_used": int(keep.sum()),
        "scored_index_overlap_with_flagger": int(len(screen.index.intersection(scored_rows.index))),
        "scored_index_overlap_with_fit": int(len(used_features.index.intersection(scored_rows.index))),
    }
    return models, audit, fit_rows.loc[keep]


def _append_metric(
    records: list[dict[str, Any]],
    model_name: str,
    scope: str,
    actual: pd.Series | np.ndarray,
    predicted: np.ndarray,
    fold: str = "",
) -> None:
    """Append a consistent model metric record.

    Args:
        records: Mutable destination for metric rows.
        model_name: Stable model identifier.
        scope: CV-fold or holdout scope label.
        actual: Observed positive rates in dollars.
        predicted: Predicted rates in dollars.
        fold: Fold name, if the metric is for one CV fold.
    """
    metrics = regression_metrics(actual, predicted)
    records.append(
        {
            "model": model_name,
            "scope": scope,
            "fold": fold,
            "rows": len(actual),
            **metrics,
        }
    )


def _segment_metrics(
    holdout: pd.DataFrame, selected_prediction: np.ndarray
) -> pd.DataFrame:
    """Measure October errors by equipment, distance bucket, and month.

    Args:
        holdout: October holdout records with targets and segment fields.
        selected_prediction: Predictions from the CV-selected model.

    Returns:
        One metric row for each equipment, distance, and month segment.
    """
    data = holdout[["equipment", "distance", "date", "posted_rate"]].copy()
    data["predicted_rate"] = selected_prediction
    data["distance_bucket"] = pd.cut(
        data["distance"],
        [0, 250, 500, 1000, 2000, np.inf],
        labels=["0-250", "250-500", "500-1,000", "1,000-2,000", ">2,000"],
        right=False,
    )
    data["month"] = data["date"].dt.to_period("M").astype(str)
    records = []
    for segment in ("equipment", "distance_bucket", "month"):
        for key, group in data.groupby(segment, observed=True):
            records.append(
                {
                    "segment": segment,
                    "value": str(key),
                    "rows": len(group),
                    **regression_metrics(group["posted_rate"], group["predicted_rate"].to_numpy()),
                }
            )
    return pd.DataFrame.from_records(records)


def _validate_splits(
    train: pd.DataFrame, split_config: dict[str, Any]
) -> list[tuple[dict[str, Any], pd.Series, pd.Series]]:
    """Build fold masks and verify strict chronological ordering.

    Args:
        train: Labeled development data with parsed dates.
        split_config: Expanding-window fold and October holdout settings.

    Returns:
        Fold settings with separate training and scored-row masks.

    Raises:
        ValueError: If a fold or holdout is empty or violates time ordering.
    """
    folds = []
    dates = train["date"]
    for fold in split_config["cv_folds"]:
        train_mask = dates.le(pd.Timestamp(fold["train_end"]))
        scored_mask = _date_mask(
            dates, fold["validation_start"], fold["validation_end"]
        )
        if not train_mask.any() or not scored_mask.any():
            raise ValueError(f"Configured fold has no training or validation rows: {fold['name']}")
        if dates.loc[train_mask].max() >= dates.loc[scored_mask].min():
            raise ValueError(f"Configured fold is not strictly chronological: {fold['name']}")
        folds.append((fold, train_mask, scored_mask))
    holdout_mask = _date_mask(
        dates, split_config["holdout_start"], split_config["holdout_end"]
    )
    if not holdout_mask.any() or dates.loc[~holdout_mask].max() >= dates.loc[holdout_mask].min():
        raise ValueError("Configured October holdout is empty or chronologically invalid")
    return folds


def _baseline_predictions(
    train_rows: pd.DataFrame, validation_rows: pd.DataFrame
) -> np.ndarray:
    """Fit the median rate-per-mile baseline using training rows only.

    Args:
        train_rows: Training rows and targets for the current fold.
        validation_rows: Scored rows containing distances.

    Returns:
        Positive rate predictions in dollars.
    """
    median_per_mile = float((train_rows["posted_rate"] / train_rows["distance"]).median())
    return np.maximum(median_per_mile * validation_rows["distance"].to_numpy(dtype=float), 0.01)


def run_training(config_path: Path) -> None:
    """Run chronological CV, compare estimators, refit, and predict outputs.

    Args:
        config_path: YAML configuration path for inputs, folds, models, and outputs.

    Raises:
        FileNotFoundError: If a configured input is absent.
        ValueError: If input validation or the chronological split checks fail.
    """
    config, root = load_config(config_path)
    paths = {key: resolve_path(root, value) for key, value in config["paths"].items()}
    reports_dir = resolve_path(root, config["paths"]["reports_dir"])
    train = read_csv(paths["train"], "training")
    validation = read_csv(paths["validation"], "validation")
    template = read_csv(paths["template"], "prediction template")
    december = read_csv(paths["december"], "December inputs")
    validate_training_data(train)
    validate_prediction_inputs(validation, template, december)

    train = train.copy()
    train["date"] = pd.to_datetime(train["date"], errors="raise")
    train["posted_rate"] = pd.to_numeric(train["posted_rate"], errors="raise")
    features = build_features(train)
    target = train["posted_rate"].astype(float)
    folds = _validate_splits(train, config["split"])
    model_config = config["model"]
    cleaning_config = config.get("cleaning", {"enabled": False})
    fold_records: list[dict[str, Any]] = []
    cleaning_records: list[dict[str, Any]] = []
    oof_records: list[pd.DataFrame] = []
    holdout_records: list[pd.DataFrame] = []
    holdout_metrics: list[dict[str, Any]] = []

    for fold, train_mask, scored_mask in folds:
        fold_train = train.loc[train_mask]
        fold_scored = train.loc[scored_mask]
        fold_features = features.loc[train_mask]
        scored_features = features.loc[scored_mask]
        fold_target = target.loc[train_mask]
        scored_target = target.loc[scored_mask]
        LOGGER.info(
            "Fold %s: %s training rows, %s scored rows",
            fold["name"],
            len(fold_train),
            len(fold_scored),
        )

        fold_models, clean_audit, cleaned_fold_train = _fit_fold_models(
            fold_train, fold_scored, fold_features, fold_target, model_config, cleaning_config
        )
        baseline = _baseline_predictions(cleaned_fold_train, fold_scored)
        _append_metric(fold_records, "median_rate_per_mile", "cv_fold", scored_target, baseline, fold["name"])
        oof_records.append(
            pd.DataFrame(
                {
                    "load_id": fold_scored["load_id"].to_numpy(),
                    "date": fold_scored["date"].to_numpy(),
                    "actual_rate": scored_target.to_numpy(),
                    "model": "median_rate_per_mile",
                    "fold": fold["name"],
                    "predicted_rate": baseline,
                }
            )
        )
        clean_audit["scope"] = str(fold["name"])
        cleaning_records.append(clean_audit)
        for name, bundle in fold_models.items():
            prediction = bundle.predict(scored_features)
            _append_metric(fold_records, name, "cv_fold", scored_target, prediction, fold["name"])
            oof_records.append(
                pd.DataFrame(
                    {
                        "load_id": fold_scored["load_id"].to_numpy(),
                        "date": fold_scored["date"].to_numpy(),
                        "actual_rate": scored_target.to_numpy(),
                        "model": name,
                        "fold": fold["name"],
                        "predicted_rate": prediction,
                    }
                )
            )

    holdout_mask = _date_mask(
        train["date"], config["split"]["holdout_start"], config["split"]["holdout_end"]
    )
    holdout_train_mask = train["date"].lt(pd.Timestamp(config["split"]["holdout_start"]))
    holdout_train = train.loc[holdout_train_mask]
    holdout = train.loc[holdout_mask]
    holdout_models, holdout_clean_audit, cleaned_holdout_train = _fit_fold_models(
        holdout_train,
        holdout,
        features.loc[holdout_train_mask],
        target.loc[holdout_train_mask],
        model_config,
        cleaning_config,
    )
    holdout_clean_audit["scope"] = "october_holdout"
    cleaning_records.append(holdout_clean_audit)
    all_holdout_predictions: dict[str, np.ndarray] = {
        "median_rate_per_mile": _baseline_predictions(cleaned_holdout_train, holdout)
    }
    for name, bundle in holdout_models.items():
        all_holdout_predictions[name] = bundle.predict(features.loc[holdout_mask])
    for name, prediction in all_holdout_predictions.items():
        _append_metric(holdout_metrics, name, "october_holdout", holdout["posted_rate"], prediction)
        holdout_records.append(
            pd.DataFrame(
                {
                    "load_id": holdout["load_id"].to_numpy(),
                    "date": holdout["date"].to_numpy(),
                    "actual_rate": holdout["posted_rate"].to_numpy(),
                    "model": name,
                    "predicted_rate": prediction,
                }
            )
        )

    oof = pd.concat(oof_records, ignore_index=True)
    cv_summary = (
        oof.groupby("model", observed=True)
        .apply(
            lambda group: pd.Series(regression_metrics(group["actual_rate"], group["predicted_rate"].to_numpy())),
            include_groups=False,
        )
        .reset_index()
    )
    holdout_summary = pd.DataFrame(holdout_metrics).drop(columns=["scope", "fold", "rows"])
    comparison = cv_summary.merge(holdout_summary, on="model", suffixes=("_cv", "_october"))
    candidate_names = {str(item["name"]) for item in model_config["catboost"]["candidates"]}
    cv_rank = comparison.loc[comparison["model"].isin(candidate_names)].sort_values(
        [f"{config['split']['selection_metric']}_cv", "rmse_cv"], ascending=True
    )
    selected_name = str(cv_rank.iloc[0]["model"])
    comparison["selected_by_cv_mae"] = comparison["model"].eq(selected_name)
    LOGGER.info("Selected %s by mean out-of-fold MAE", selected_name)

    final_config = next(
        candidate for candidate in model_config["catboost"]["candidates"]
        if candidate["name"] == selected_name
    )
    final_screen = compute_training_flags(train, cleaning_config)
    final_keep = ~final_screen["flagged"]
    if not final_keep.any():
        raise ValueError("Cleaning removed every row from final training data")
    final_fit_rows = train.loc[final_keep]
    final_bundle = fit_model(
        selected_name,
        "catboost",
        features.loc[final_keep],
        target.loc[final_keep],
        model_config,
        final_config,
    )
    if config["run"].get("write_model_artifact", True):
        paths["model"].parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_bundle, paths["model"])

    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)
    pd.DataFrame(fold_records).to_csv(reports_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(cleaning_records).to_csv(reports_dir / "cleaning_fold_audit.csv", index=False)
    if config["run"].get("write_oof_predictions", True):
        oof.to_csv(reports_dir / "cv_predictions.csv", index=False)
    pd.concat(holdout_records, ignore_index=True).to_csv(
        reports_dir / "october_holdout_predictions.csv", index=False
    )
    selected_holdout = all_holdout_predictions[selected_name]
    _segment_metrics(holdout, selected_holdout).to_csv(
        reports_dir / "segment_metrics.csv", index=False
    )
    final_bundle.feature_importance().head(25).to_csv(
        reports_dir / "feature_importance.csv", index=False
    )
    validation_ids = set(validation["load_id"].astype(str))
    development_ids = set(train["load_id"].astype(str))
    metadata = {
        "selected_model": selected_name,
        "selection_metric": config["split"]["selection_metric"],
        "cv_fold_names": [fold["name"] for fold, _, _ in folds],
        "holdout_period": [str(config["split"]["holdout_start"]), str(config["split"]["holdout_end"])],
        "final_training_rows": len(train),
        "final_training_flagged_rows": int(final_screen["flagged"].sum()),
        "final_training_rows_fitted": int(final_keep.sum()),
        "cleaning_enabled": bool(cleaning_config.get("enabled", False)),
        "validation_rows_used_for_fit_or_tuning": int(len(validation_ids & development_ids)),
        "validation_ids_disjoint_from_fit_or_tuning_data": bool(
            validation_ids.isdisjoint(development_ids)
        ),
        "baseline_training_rows_cleaned": bool(cleaning_config.get("enabled", False)),
    }
    (reports_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    predict_files(config_path)
    LOGGER.info("Training and final prediction pipeline completed")


def main() -> None:
    """Run the end-to-end training and prediction command-line entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
