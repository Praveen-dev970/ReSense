"""Unit tests for the documented Q-CHAT-10 scoring implementation.

Run from the repository root:
    python -m unittest test_questionnaire -v
"""

import unittest

from textutils.questionnaire import (
    RESPONSE_OPTIONS,
    calculate_qchat10_score,
    exceeds_screening_threshold,
)


class QChat10ScoringTests(unittest.TestCase):
    def test_all_always_scores_one(self):
        answers = ["Always"] * 10
        self.assertEqual(calculate_qchat10_score(answers), 1)

    def test_a1_to_a9_scoring_direction(self):
        answers = ["Never"] * 9 + ["Never"]
        self.assertEqual(calculate_qchat10_score(answers), 9)

    def test_a10_reverse_scoring(self):
        answers = ["Always"] * 9 + ["Always"]
        self.assertEqual(calculate_qchat10_score(answers), 1)
        answers[-1] = "Never"
        self.assertEqual(calculate_qchat10_score(answers), 0)

    def test_maximum_score_is_ten(self):
        answers = ["Never"] * 9 + ["Always"]
        self.assertEqual(calculate_qchat10_score(answers), 10)

    def test_minimum_score_is_zero(self):
        answers = ["Always"] * 9 + ["Never"]
        self.assertEqual(calculate_qchat10_score(answers), 0)

    def test_score_three_does_not_exceed_threshold(self):
        answers = ["Sometimes"] * 3 + ["Always"] * 6 + ["Never"]
        score = calculate_qchat10_score(answers)
        self.assertEqual(score, 3)
        self.assertFalse(exceeds_screening_threshold(score))

    def test_score_four_exceeds_threshold(self):
        answers = ["Sometimes"] * 4 + ["Always"] * 5 + ["Never"]
        score = calculate_qchat10_score(answers)
        self.assertEqual(score, 4)
        self.assertTrue(exceeds_screening_threshold(score))

    def test_exactly_ten_answers_are_required(self):
        with self.assertRaises(ValueError):
            calculate_qchat10_score(["Always"] * 9)
        with self.assertRaises(ValueError):
            calculate_qchat10_score(["Always"] * 11)

    def test_invalid_response_is_rejected(self):
        answers = ["Always"] * 9 + ["1"]
        with self.assertRaises(ValueError):
            calculate_qchat10_score(answers)

    def test_each_documented_response_option_is_accepted(self):
        for option in RESPONSE_OPTIONS:
            with self.subTest(option=option):
                self.assertIsInstance(
                    calculate_qchat10_score([option] * 10), int
                )

    def test_threshold_rejects_out_of_range_or_non_integer_scores(self):
        for score in (-1, 11, 3.5, True):
            with self.subTest(score=score):
                with self.assertRaises(ValueError):
                    exceeds_screening_threshold(score)


if __name__ == "__main__":
    unittest.main()
