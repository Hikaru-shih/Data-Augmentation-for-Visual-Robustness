import unittest
import pandas as pd
from scripts.analyze_corruptions import summarize, validate_detail


class AnalysisTests(unittest.TestCase):
    def test_paired_sd_not_independent_sd(self):
        rows = []
        for seed, value in zip([42, 43, 44], [50, 60, 70]):
            for method, delta in [("baseline", 0), ("mixup", 5)]:
                for accuracy in [value + delta - 2, value + delta + 2]:
                    rows.append(dict(seed=seed, method=method, category="noise", accuracy=accuracy))
        paired, stats = summarize(pd.DataFrame(rows), ["category"])
        row = stats[stats.method == "mixup"].iloc[0]
        self.assertEqual(row.accuracy_std, 10)
        self.assertEqual(row.delta_mean, 5)
        self.assertEqual(row.delta_std, 0)
        self.assertEqual(len(paired), 6)

    def test_incomplete_detail_rejected(self):
        with self.assertRaises(ValueError):
            validate_detail(pd.DataFrame(dict(corruption=["fog"] * 75, severity=[1] * 75, accuracy=[50] * 75)))
