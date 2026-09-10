import unittest
from unittest.mock import patch

import pandas as pd

from scripts.summarize_multiseed import aggregate, collect, normalized, run_keys


class MultiSeedTests(unittest.TestCase):
    def test_plan_has_fifteen_unique_method_seed_pairs(self):
        keys = run_keys()
        self.assertEqual(len(keys), 15)
        self.assertEqual(len({(m, s) for m, s, _ in keys}), 15)
        self.assertIn(("augmix", 42, "augmix_resnet18/seed_42/validation-v2"), keys)

    def test_sample_standard_deviation(self):
        frame = pd.DataFrame(dict(method=["baseline"] * 3, seed=[42, 43, 44],
                                  clean_accuracy=[90., 92., 94.], mca=[70., 73., 76.]))
        row = aggregate(frame).loc["baseline"]
        self.assertEqual(row.clean_mean, 92)
        self.assertEqual(row.clean_std, 2)
        self.assertEqual(row.mca_std, 3)

    def test_configuration_comparison_preserves_hyperparameters(self):
        a = dict(seed=42, run={}, experiment={}, training={"epochs": 100})
        b = dict(seed=43, run={"key": "new"}, training={"epochs": 100})
        self.assertEqual(normalized(a), normalized(b))
        b["training"]["epochs"] = 99
        self.assertNotEqual(normalized(a), normalized(b))

    def test_missing_runs_fail_instead_of_partial_aggregation(self):
        with patch("pathlib.Path.exists", return_value=False):
            with self.assertRaisesRegex(ValueError, "Incomplete runs"):
                collect()
