import tempfile
import unittest
from pathlib import Path

from qanta_bench.adversarialness import build_adversarialness_tables, load_ai_traces

ROOT = Path(__file__).resolve().parents[1]


class AdversarialnessTest(unittest.TestCase):
    def test_scoring_regimes_use_the_same_submitted_field(self) -> None:
        strict, strict_models = load_ai_traces(ROOT, "strict")
        pedant, pedant_models = load_ai_traces(ROOT, "pedant")
        self.assertEqual(strict_models, pedant_models)
        self.assertEqual(len(strict_models), 16)
        self.assertEqual(set(strict), set(pedant))
        self.assertEqual(len({item for traces in strict.values() for item in traces}), 100)
        self.assertEqual(len({item for traces in pedant.values() for item in traces}), 100)

    def test_pedant_analysis_rebuilds_expected_submitted_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = build_adversarialness_tables(ROOT, "pedant", Path(temp_dir))
            self.assertEqual(summary.models, 16)
            self.assertEqual(summary.ai_items, 100)
            self.assertEqual(summary.human_items, 90)
            self.assertEqual(summary.adversarial_items, 66)
            self.assertEqual(summary.human_outsolve_items, 90)
            self.assertAlmostEqual(summary.human_theta, 0.92164244, places=6)
            self.assertAlmostEqual(summary.best_ai_theta, 0.53821261, places=6)
            self.assertAlmostEqual(summary.organizer_correlation, 0.27374919, places=6)


if __name__ == "__main__":
    unittest.main()
