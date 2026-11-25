"""Pure Q-CHAT-10 scoring logic used by the ReSense questionnaire.

Scoring convention documented in the supplied toddler dataset description:
- A1-A9: Sometimes, Rarely, Never = 1
- A10: Always, Usually, Sometimes = 1

This is a questionnaire screening score, not a clinical diagnostic result.
"""

RESPONSE_OPTIONS = ("Always", "Usually", "Sometimes", "Rarely", "Never")
SCORED_A1_A9 = frozenset(("Sometimes", "Rarely", "Never"))
SCORED_A10 = frozenset(("Always", "Usually", "Sometimes"))
SCREENING_THRESHOLD = 3


def calculate_qchat10_score(answers):
    """Return the documented Q-CHAT-10 score for exactly ten responses.

    Raises ValueError if the answer count or any response is invalid.
    """
    answers = list(answers)

    if len(answers) != 10:
        raise ValueError("Exactly 10 questionnaire responses are required.")

    invalid = [answer for answer in answers if answer not in RESPONSE_OPTIONS]
    if invalid:
        raise ValueError(
            "Each response must be one of: {}.".format(
                ", ".join(RESPONSE_OPTIONS)
            )
        )

    score = sum(answer in SCORED_A1_A9 for answer in answers[:9])
    score += answers[9] in SCORED_A10
    return int(score)


def exceeds_screening_threshold(score):
    """Return True when the score is greater than the documented threshold."""
    if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 10:
        raise ValueError("Score must be an integer from 0 to 10.")
    return score > SCREENING_THRESHOLD
