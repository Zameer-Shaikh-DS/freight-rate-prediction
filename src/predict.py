"""Load a saved model and write the assessment prediction files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.data import read_csv, validate_prediction_inputs
from src.features import build_features, coordinate_lookup
from src.models import ModelBundle

LOGGER = logging.getLogger(__name__)


def predict_files(config_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create validation and December prediction files from a saved model.

    Args:
        config_path: YAML configuration path for inputs, artifact, and outputs.

    Returns:
        The validation prediction table and completed December input table.

    Raises:
        FileNotFoundError: If an input file or trained model artifact is missing.
        ValueError: If IDs, schemas, or predicted rates are invalid.
    """
    config, root = load_config(config_path)
    paths = {key: resolve_path(root, value) for key, value in config["paths"].items()}
    december_output_path = paths.get("december_predictions", paths["december"])
    training = read_csv(paths["train"], "training")
    validation = read_csv(paths["validation"], "validation")
    template = read_csv(paths["template"], "prediction template")
    december = read_csv(paths["december"], "December inputs")
    validate_prediction_inputs(validation, template, december)
    model_path = paths["model"]
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}. Run src.train first.")
    bundle = joblib.load(model_path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError("Model artifact has an unexpected type")

    coordinates = coordinate_lookup(training)
    validation_features = build_features(validation, coordinates)
    validation_values = bundle.predict(validation_features)
    template_ids = template["load_id"].astype(str)
    prediction_by_id = pd.Series(validation_values, index=validation["load_id"].astype(str))
    prediction_frame = pd.DataFrame(
        {"load_id": template_ids, "predicted_rate": template_ids.map(prediction_by_id)}
    )
    if prediction_frame["predicted_rate"].isna().any():
        raise ValueError("Some template IDs do not have predictions")
    if not np.isfinite(prediction_frame["predicted_rate"]).all() or (
        prediction_frame["predicted_rate"] <= 0
    ).any():
        raise ValueError("Final rates must be finite and positive")
    prediction_frame.to_csv(paths["predictions"], index=False)

    december_features = build_features(december, coordinates)
    december_result = december.copy()
    december_result["predicted_rate"] = bundle.predict(december_features)
    december_output_path.parent.mkdir(parents=True, exist_ok=True)
    december_result.to_csv(december_output_path, index=False)
    LOGGER.info("Saved %s validation predictions and %s December predictions", len(prediction_frame), len(december_result))
    return prediction_frame, december_result


def main() -> None:
    """Run prediction using the artifact produced by training."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args()
    predict_files(args.config)


if __name__ == "__main__":
    main()
