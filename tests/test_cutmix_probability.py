import unittest
from unittest.mock import patch
import torch
from src.training.cutmix_trainer import train_one_epoch_cutmix


class ProbabilityTests(unittest.TestCase):
    def run_epoch(self, probability):
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
        batch = (torch.ones(2, 1, 2, 2), torch.tensor([0, 1]))
        stats = {}
        train_one_epoch_cutmix(model, [batch, batch], torch.nn.CrossEntropyLoss(),
                               torch.optim.SGD(model.parameters(), lr=.01), "cpu",
                               probability=probability, stats=stats)
        return stats

    def test_probability_zero_skips_mixing_and_rng(self):
        with patch("src.training.cutmix_trainer.cutmix_data") as mix, patch("numpy.random.random") as rng:
            self.assertEqual(self.run_epoch(0), {"batches": 2, "applied_batches": 0})
            mix.assert_not_called()
            rng.assert_not_called()

    def test_probability_one_preserves_legacy_rng(self):
        with patch("numpy.random.random") as rng:
            self.assertEqual(self.run_epoch(1)["applied_batches"], 2)
            rng.assert_not_called()

    def test_probability_half_gates_per_batch(self):
        with patch("numpy.random.random", side_effect=[.2, .8]):
            self.assertEqual(self.run_epoch(.5)["applied_batches"], 1)

    def test_invalid_probability(self):
        with self.assertRaises(ValueError):
            self.run_epoch(1.1)
