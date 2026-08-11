import tempfile
import unittest
from pathlib import Path

import numpy as np

from qanta_bench.plots import _zero_aligned_bins, generate_adversarialness_figures

ROOT = Path(__file__).resolve().parents[1]


class PlotTest(unittest.TestCase):
    def test_histogram_bins_have_zero_edge(self) -> None:
        bins = _zero_aligned_bins(np.array([-0.31, 0.92]))
        self.assertTrue(np.isclose(bins, 0).any())

    def test_paper_figures_render(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = generate_adversarialness_figures(ROOT / "analysis_inputs", Path(temp_dir))
            self.assertEqual(
                [path.name for path in outputs],
                [
                    "qanta_fig1_theta_ladder_paper.png",
                    "qanta_fig2_difficulty_curves_paper.png",
                    "qanta_fig3_delta_hist_paper.png",
                ],
            )
            self.assertTrue(all(path.stat().st_size > 10_000 for path in outputs))

    def test_pedant_paper_figures_render(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = generate_adversarialness_figures(
                ROOT / "analysis_inputs" / "pedant", Path(temp_dir), "pedant"
            )
            self.assertEqual(
                [path.name for path in outputs],
                [
                    "qanta_fig1_theta_ladder_pedant_paper.png",
                    "qanta_fig2_difficulty_curves_pedant_paper.png",
                    "qanta_fig3_delta_hist_pedant_paper.png",
                ],
            )
            self.assertTrue(all(path.stat().st_size > 10_000 for path in outputs))


if __name__ == "__main__":
    unittest.main()
