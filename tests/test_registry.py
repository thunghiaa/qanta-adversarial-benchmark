import unittest
from pathlib import Path

from qanta_bench.registry import load_registry, models_for

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "models.json"


class RegistryTest(unittest.TestCase):
    def test_submitted_and_benchmark_cohorts_are_disjoint(self) -> None:
        models = load_registry(REGISTRY)
        submitted = {model.id for model in models if model.cohort == "qanta_submitted"}
        benchmark = {model.id for model in models if model.cohort == "additional_benchmark"}
        self.assertTrue(submitted)
        self.assertTrue(benchmark)
        self.assertTrue(submitted.isdisjoint(benchmark))

    def test_qanta_tossup_field_has_sixteen_models(self) -> None:
        models = models_for("qanta_submitted", "tossup", REGISTRY)
        self.assertEqual(len(models), 16)

    def test_qanta_bonus_field_has_five_models(self) -> None:
        models = models_for("qanta_submitted", "bonus", REGISTRY)
        self.assertEqual(len(models), 5)


if __name__ == "__main__":
    unittest.main()
