import unittest
from unittest.mock import patch

import pandas as pd

from src.cleaning import compute_training_flags
from src.train import _fit_fold_models


class CleaningTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "enabled": True,
            "low_ratio": 0.4,
            "high_ratio": 2.5,
            "distance_edges": [0, 250, 500, 1000, 2000],
        }

    @staticmethod
    def _rows(rates, indices):
        return pd.DataFrame(
            {
                "equipment": ["Dry Van"] * len(rates),
                "distance": [100.0] * len(rates),
                "posted_rate": rates,
            },
            index=indices,
        )

    def test_loo_reference_does_not_use_row_target(self):
        rows = self._rows([100, 200, 300, 400, 100_000], [0, 1, 2, 3, 4])
        changed = rows.copy()
        changed.loc[4, "posted_rate"] = 1_000_000
        first = compute_training_flags(rows, self.config)
        second = compute_training_flags(changed, self.config)
        self.assertEqual(first.loc[4, "expected_rate"], 250)
        self.assertEqual(second.loc[4, "expected_rate"], 250)
        self.assertTrue(first.loc[4, "flagged"])

    def test_flagger_and_fit_use_only_training_indices(self):
        fit_rows = self._rows([100, 100, 100, 100, 1_000], [0, 1, 2, 3, 4])
        scored_rows = self._rows([50_000, 60_000], [100, 101])
        features = pd.DataFrame({"signal": range(5)}, index=fit_rows.index)
        target = fit_rows["posted_rate"]
        captured = {"flagger": None, "fit": None}
        real_flagger = compute_training_flags

        def flagger_spy(rows, config):
            captured["flagger"] = set(rows.index)
            return real_flagger(rows, config)

        def model_spy(fit_features, fit_target, model_config):
            captured["fit"] = (set(fit_features.index), set(fit_target.index))
            return {"spy": object()}

        with (
            patch("src.train.compute_training_flags", side_effect=flagger_spy),
            patch("src.train._make_models", side_effect=model_spy),
        ):
            _, audit, kept_rows = _fit_fold_models(
                fit_rows, scored_rows, features, target, {}, self.config
            )

        training_ids = set(fit_rows.index)
        scored_ids = set(scored_rows.index)
        self.assertTrue(captured["flagger"].issubset(training_ids))
        self.assertTrue(captured["flagger"].isdisjoint(scored_ids))
        fit_features, fit_target = captured["fit"]
        self.assertTrue(fit_features.issubset(training_ids))
        self.assertTrue(fit_features.isdisjoint(scored_ids))
        self.assertEqual(fit_features, fit_target)
        self.assertEqual(set(kept_rows.index), fit_features)
        self.assertEqual(audit["scored_index_overlap_with_flagger"], 0)
        self.assertEqual(audit["scored_index_overlap_with_fit"], 0)


if __name__ == "__main__":
    unittest.main()
