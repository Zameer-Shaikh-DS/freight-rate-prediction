"""Model constructors and fitted model bundles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES


@dataclass
class ModelBundle:
    """Fitted estimator plus the transform metadata needed at inference."""

    name: str
    kind: str
    target_transform: str
    estimator: Any
    numeric_medians: dict[str, float] | None = None

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict positive rates in dollars from engineered input features.

        Args:
            features: Feature rows with the columns used during fitting.

        Returns:
            Finite positive rate estimates in original dollar units.
        """
        model_features = features[MODEL_FEATURES].copy()
        if self.kind == "catboost":
            medians = self.numeric_medians or {}
            model_features[NUMERIC_FEATURES] = model_features[NUMERIC_FEATURES].fillna(medians)
        prediction = np.asarray(self.estimator.predict(model_features), dtype=float).reshape(-1)
        if self.target_transform == "log1p":
            prediction = np.expm1(prediction)
        return np.maximum(prediction, 0.01)

    def feature_importance(self) -> pd.DataFrame:
        """Return native CatBoost feature importances when available.

        Returns:
            A feature and importance table, or an empty table for linear models.
        """
        if self.kind != "catboost":
            return pd.DataFrame(columns=["feature", "importance"])
        values = self.estimator.get_feature_importance()
        return (
            pd.DataFrame({"feature": MODEL_FEATURES, "importance": values})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


def _transform_target(target: pd.Series, transform: str) -> np.ndarray:
    """Apply a configured target transformation.

    Args:
        target: Original positive target values.
        transform: Either ``identity`` or ``log1p``.

    Returns:
        Transformed target values as a NumPy array.

    Raises:
        ValueError: If the transformation name is unsupported.
    """
    values = target.to_numpy(dtype=float)
    if transform == "identity":
        return values
    if transform == "log1p":
        return np.log1p(values)
    raise ValueError(f"Unsupported target transform: {transform}")


def _linear_estimator(minimum_category_frequency: int) -> Pipeline:
    """Build a fold-fitted one-hot linear regression pipeline.

    Args:
        minimum_category_frequency: Minimum category count before grouping rare values.

    Returns:
        A scikit-learn pipeline with fold-fitted imputation, encoding, and regression.
    """
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=minimum_category_frequency,
                ),
            ),
        ]
    )
    transformer = ColumnTransformer(
        transformers=[
            ("categorical", categorical, CATEGORICAL_FEATURES),
            ("numeric", numeric, NUMERIC_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return Pipeline(
        steps=[("features", transformer), ("regressor", LinearRegression())]
    )


def fit_model(
    name: str,
    kind: str,
    features: pd.DataFrame,
    target: pd.Series,
    model_config: dict[str, Any],
    candidate_config: dict[str, Any] | None = None,
) -> ModelBundle:
    """Fit one estimator and its preprocessing on a single training fold.

    Args:
        name: Stable model name for reports.
        kind: Estimator family, currently ``linear`` or ``catboost``.
        features: Engineered training-fold features.
        target: Training-fold targets in dollars.
        model_config: Model and preprocessing configuration.
        candidate_config: Candidate-specific CatBoost settings, when applicable.

    Returns:
        A fitted model bundle with the inference transforms needed later.

    Raises:
        ValueError: If the model family or target transform is unsupported.
    """
    if kind == "linear":
        transform = "identity"
        estimator = _linear_estimator(
            int(model_config["linear"]["minimum_category_frequency"])
        )
        estimator.fit(features[MODEL_FEATURES], target.to_numpy(dtype=float))
        return ModelBundle(name, kind, transform, estimator)
    if kind != "catboost" or candidate_config is None:
        raise ValueError(f"Unsupported model type or missing configuration: {kind}")

    transform = str(candidate_config["target_transform"])
    numeric_medians = features[NUMERIC_FEATURES].median(numeric_only=True).to_dict()
    missing_medians = [key for key, value in numeric_medians.items() if not np.isfinite(value)]
    if missing_medians:
        raise ValueError(f"Cannot impute all-missing numeric features: {missing_medians}")
    model_features = features[MODEL_FEATURES].copy()
    model_features[NUMERIC_FEATURES] = model_features[NUMERIC_FEATURES].fillna(numeric_medians)
    parameters = model_config["catboost"]
    estimator = CatBoostRegressor(
        iterations=int(candidate_config.get("iterations", parameters["iterations"])),
        depth=int(candidate_config.get("depth", parameters["depth"])),
        learning_rate=float(candidate_config.get("learning_rate", parameters["learning_rate"])),
        l2_leaf_reg=float(candidate_config.get("l2_leaf_reg", parameters["l2_leaf_reg"])),
        loss_function=str(candidate_config["loss_function"]),
        random_seed=int(model_config["random_seed"]),
        thread_count=int(parameters["thread_count"]),
        allow_writing_files=False,
        verbose=False,
    )
    estimator.fit(
        model_features,
        _transform_target(target, transform),
        cat_features=CATEGORICAL_FEATURES,
    )
    # CatBoost embeds a wall-clock finish time and random model UUID in its
    # metadata. Normalize those non-model fields so fixed-seed artifacts hash
    # identically across fresh runs; the learned trees and predictions are unchanged.
    metadata = estimator.get_metadata()
    if "train_finish_time" in metadata:
        metadata["train_finish_time"] = "fixed-seed-reproducible"
    if "model_guid" in metadata:
        metadata["model_guid"] = "fixed-seed-reproducible"
    return ModelBundle(name, kind, transform, estimator, numeric_medians)

