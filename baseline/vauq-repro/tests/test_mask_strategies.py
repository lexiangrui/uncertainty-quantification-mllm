from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from vauq.masking import choose_masked_offsets
from vauq.scoring import (
    compute_mask_comparison_scores,
    compute_multi_seed_comparison_scores,
)


class MaskStrategyTests(unittest.TestCase):
    def test_blank_masks_all_visual_tokens(self):
        offsets = choose_masked_offsets(576, 0.3, "blank")
        self.assertEqual(offsets.tolist(), list(range(576)))

    def test_random_masks_same_count_as_core(self):
        scores = torch.arange(576, dtype=torch.float32)
        core = choose_masked_offsets(576, 0.3, "core", attention_scores=scores)
        random = choose_masked_offsets(576, 0.3, "random", seed=42)
        self.assertEqual(len(core), int(576 * 0.3))
        self.assertEqual(len(random), len(core))
        self.assertEqual(len(set(random.tolist())), len(random))

    def test_random_is_reproducible_and_seeded(self):
        first = choose_masked_offsets(576, 0.4, "random", seed=7)
        repeated = choose_masked_offsets(576, 0.4, "random", seed=7)
        different = choose_masked_offsets(576, 0.4, "random", seed=8)
        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.equal(first, different))

    def test_ratio_validation(self):
        with self.assertRaises(ValueError):
            choose_masked_offsets(576, 0.0, "random", seed=1)

    def test_joint_comparison_reuses_one_base_entropy(self):
        masked_entropy = {"core": 1.5, "blank": 1.7, "random": 1.4}
        with (
            patch("vauq.scoring.compute_entropy", return_value=1.0) as base,
            patch(
                "vauq.scoring.compute_entropy_core_masked",
                side_effect=lambda *args, mask_strategy, **kwargs: masked_entropy[mask_strategy],
            ) as masked,
        ):
            results = compute_mask_comparison_scores(
                object(), None, "question", object(), alpha=0.5, mask_seed=49
            )

        base.assert_called_once()
        self.assertEqual(masked.call_count, 3)
        self.assertEqual(list(results), ["core", "blank", "random"])
        self.assertEqual(results["core"].vauq, 0.75)
        self.assertEqual(results["blank"].vauq, 0.65)
        self.assertEqual(results["random"].vauq, 0.8)

    def test_multi_seed_comparison_reuses_one_base_entropy(self):
        values = {"core": 1.5, "blank": 1.7, 42: 1.4, 43: 1.6}
        with (
            patch("vauq.scoring.compute_entropy", return_value=1.0) as base,
            patch(
                "vauq.scoring.compute_entropy_core_masked",
                side_effect=lambda *args, mask_strategy, mask_seed, **kwargs: (
                    values[mask_strategy] if mask_strategy != "random"
                    else values[mask_seed - 7]
                ),
            ) as masked,
        ):
            results = compute_multi_seed_comparison_scores(
                object(), None, "question", object(), [42, 43],
                alpha=0.5, sample_index=7,
            )

        base.assert_called_once()
        self.assertEqual(masked.call_count, 4)
        self.assertEqual(
            list(results), ["core", "blank", "random_seed42", "random_seed43"]
        )
        self.assertEqual(results["random_seed42"].vauq, 0.8)
        self.assertEqual(results["random_seed43"].vauq, 0.7)


if __name__ == "__main__":
    unittest.main()
