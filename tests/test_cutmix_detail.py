import unittest
import pandas as pd
from scripts.cutmix_detail_analysis import paired_statistics


class DetailTests(unittest.TestCase):
    def data(self):
        return pd.DataFrame([dict(method=m, seed=s, corruption="fog", severity=v, accuracy=a+d+v)
                             for m,d in [("baseline",0),("cutmix_p1",-5),("cutmix_p05",2)]
                             for s,a in [(42,50),(43,60),(44,70)] for v in (1,2)])

    def test_paired_seed_sd_and_within_seed_average(self):
        per_seed, accuracy, paired, stats = paired_statistics(self.data(), ["corruption"])
        self.assertEqual(len(per_seed), 9)
        row = stats[stats.control == "cutmix_p1"].iloc[0]
        self.assertEqual(row["mean"], 7)
        self.assertEqual(row["std"], 0)
        self.assertEqual(accuracy.iloc[0]["std"], 10)

    def test_missing_pair_rejected(self):
        data = self.data()
        data = data[~((data.method == "baseline") & (data.seed == 43))]
        with self.assertRaises(ValueError):
            paired_statistics(data, ["corruption"])
