import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("frontier_gpu", ROOT / "scripts" / "frontier_gpu.py")
assert SPEC is not None and SPEC.loader is not None
frontier_gpu = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frontier_gpu)


class FrontierGpuRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = frontier_gpu.load_config()

    def test_all_ten_inventory_models_have_gpu_recipes(self) -> None:
        inventory = json.loads(
            (ROOT / "configs" / "frontier_models.json").read_text(encoding="utf-8")
        )
        expected = {model["id"] for model in inventory["models"]}
        actual = {model["canonical_id"] for model in self.config["models"]}
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 10)

    def test_revisions_and_sources_are_pinned(self) -> None:
        for model in self.config["models"]:
            with self.subTest(model=model["slug"]):
                self.assertEqual(len(model["revision"]), 40)
                int(model["revision"], 16)
                self.assertTrue(model["recipe_url"].startswith("https://"))
                self.assertGreater(model["checkpoint_weight_bytes"], 0)
                self.assertGreater(model["minimum_total_vram_gb"], 0)
                self.assertGreater(model["minimum_gpus"], 0)
                self.assertEqual(model["canonical_id"].count("/"), 1)

    def test_rendered_command_contains_exact_checkpoint_and_revision(self) -> None:
        for model in self.config["models"]:
            command = frontier_gpu.render_command(
                model, tp=model["default_tp"], port=8123, max_model_len=8192
            )
            with self.subTest(model=model["slug"]):
                self.assertIn(model["served_checkpoint"], command)
                self.assertIn(model["revision"], command)
                self.assertIn("8123", command)
                self.assertNotIn("{checkpoint}", command)

    def test_cache_inside_checkout_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            frontier_gpu.ensure_external_cache(ROOT / "model_artifacts")

    def test_kimi_checks_vram_in_tp_group_not_all_visible_gpus(self) -> None:
        model = frontier_gpu.resolve_model(self.config, "kimi-k3")
        sixteen_h200 = [
            {"index": index, "name": "NVIDIA H200", "memory_gb": 141}
            for index in range(16)
        ]
        eight_b300 = [
            {"index": index, "name": "NVIDIA B300", "memory_gb": 268}
            for index in range(8)
        ]
        self.assertFalse(frontier_gpu.hardware_check(model, sixteen_h200)[0])
        self.assertTrue(frontier_gpu.hardware_check(model, eight_b300)[0])


if __name__ == "__main__":
    unittest.main()
