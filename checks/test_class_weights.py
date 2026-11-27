"""클래스 가중치 회귀 테스트. 원본 이미지·학습·Test 평가 없이 실행한다.

실행: uv run python -m unittest discover -s checks -p test_class_weights.py -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset import balanced_class_weights  # noqa: E402


class BalancedClassWeightsTest(unittest.TestCase):
    def test_all_classes_present_preserves_formula(self):
        frame = pd.DataFrame({"target_id": [0, 1, 1, 2, 2, 2]})
        with patch("dataset.load_split", return_value=frame) as load:
            actual = balanced_class_weights(train_pct=25, num_classes=3)
        load.assert_called_once_with("train", 25)
        expected = torch.tensor([2.0, 1.0, 2.0 / 3.0], dtype=torch.float32)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.isfinite(actual).all().item())

    def test_missing_class_raises_value_error(self):
        frame = pd.DataFrame({"target_id": [0, 0, 2]})
        with patch("dataset.load_split", return_value=frame):
            with self.assertRaisesRegex(ValueError, r"target_id.*\[1\]"):
                balanced_class_weights(num_classes=3)

    def test_empty_training_split_raises_value_error(self):
        frame = pd.DataFrame({"target_id": pd.Series(dtype=int)})
        with patch("dataset.load_split", return_value=frame):
            with self.assertRaisesRegex(ValueError, r"target_id.*\[0, 1, 2\]"):
                balanced_class_weights(num_classes=3)


if __name__ == "__main__":
    unittest.main()
