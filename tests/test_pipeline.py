import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features import build_features, coordinate_lookup
from src.train import _validate_splits


class PipelineTests(unittest.TestCase):
    def test_feature_construction_cleans_weight_and_builds_lane(self):
        row = pd.DataFrame(
            {
                "pickup": [" Lexington "],
                "delivery": ["Fort Wayne"],
                "equipment": ["Dry Van"],
                "distance": [360.0],
                "weight": [-500.0],
                "date": ["2025-12-01"],
            }
        )
        result = build_features(row)
        self.assertEqual(result.loc[0, "pickup"], "Lexington")
        self.assertEqual(result.loc[0, "lane"], "Lexington -> Fort Wayne")
        self.assertTrue(np.isnan(result.loc[0, "weight"]))
        self.assertEqual(result.loc[0, "weight_missing"], 1)

    def test_coordinates_come_from_training_lookup(self):
        training = pd.DataFrame(
            {
                "pickup": ["Lexington"],
                "pickup_lat": [38.0],
                "pickup_lon": [-85.0],
                "delivery": ["Fort Wayne"],
                "delivery_lat": [41.0],
                "delivery_lon": [-85.0],
            }
        )
        scored = pd.DataFrame(
            {
                "pickup": ["Lexington"],
                "delivery": ["Fort Wayne"],
                "equipment": ["Dry Van"],
                "distance": [360.0],
                "weight": [32000.0],
                "date": ["2025-12-01"],
            }
        )
        result = build_features(scored, coordinate_lookup(training))
        self.assertEqual(result.loc[0, "pickup_lat"], 38.0)
        self.assertEqual(result.loc[0, "delivery_lon"], -85.0)
        self.assertTrue(np.isfinite(result.loc[0, "straight_line_distance"]))

    def test_each_configured_fold_is_strictly_chronological(self):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root / "configs" / "config.yaml").read_text())
        dates = pd.date_range("2025-01-01", "2025-10-31", freq="D")
        folds = _validate_splits(pd.DataFrame({"date": dates}), config["split"])
        self.assertEqual(len(folds), 5)
        for fold, train_mask, scored_mask in folds:
            with self.subTest(fold=fold["name"]):
                self.assertLess(dates[train_mask].max(), dates[scored_mask].min())


if __name__ == "__main__":
    unittest.main()
