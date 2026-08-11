import unittest

from qanta_bench.frontier import (
    image_before_text_tokens,
    inject_packet_qid,
    parse_answer_json,
    run_indices_from_tokens,
    strict_correct,
    text_fragment,
)


class FrontierBenchmarkTest(unittest.TestCase):
    def test_legacy_slide_is_reordered_like_submission_runner(self) -> None:
        tokens = [
            {"type": "text", "content": "One", "position": 0},
            {"type": "text", "content": "clue.", "position": 1},
            {"type": "image", "hash_key": "x", "position": 2},
            {"type": "delay", "content": None, "position": 3},
            {"type": "text", "content": "Next", "position": 4},
        ]
        ordered = image_before_text_tokens(tokens)
        self.assertEqual([token["type"] for token in ordered], ["image", "delay", "text", "text", "text"])
        self.assertEqual(run_indices_from_tokens(ordered), [1, 3, 4])
        self.assertIn("IMAGE OMITTED", text_fragment(ordered, 1))

    def test_historical_text_first_schedule_is_preserved_for_comparison(self) -> None:
        tokens = [
            {"type": "text", "content": word, "position": index}
            for index, word in enumerate("one two three four five six seven clue.".split())
        ]
        tokens.extend(
            [
                {"type": "image", "position": 8},
                {"type": "delay", "position": 9},
            ]
        )
        self.assertEqual(run_indices_from_tokens(tokens), [6, 7, 9])

    def test_json_response_and_scoring(self) -> None:
        answer, confidence = parse_answer_json('```json\n{"answer":"Apollo missions","confidence":0.9}\n```')
        self.assertEqual(answer, "Apollo missions")
        self.assertEqual(confidence, 0.9)
        self.assertEqual(strict_correct(answer, ["_Apollo_ Mission"]), 1)

    def test_packet_qids_are_stable(self) -> None:
        expected = "advvqa-packet3-t-abc"
        self.assertEqual(inject_packet_qid("advvqa-t-abc", 3, "tossup"), expected)
        self.assertEqual(inject_packet_qid(expected, 3, "tossup"), expected)


if __name__ == "__main__":
    unittest.main()
