"""Tools the Quiz Tutor Agent can choose from.

Each tool does one job and returns plain Python data. If something goes
wrong, a tool raises ToolError so the agent can react (for example, switch
to another tool) instead of crashing.
"""
import json
import random
from pathlib import Path

from agent import memory

BANK_PATH = Path(__file__).resolve().parent.parent / "data" / "question_bank.json"
DIFFICULTIES = ["easy", "medium", "hard"]
REQUIRED_FIELDS = [
    "id", "topic", "difficulty", "question", "options",
    "answer", "hint", "explanation",
]


class ToolError(Exception):
    """Raised when a tool cannot do its job."""


def is_valid_question(q):
    """Guardrail: only well-formed questions may reach the student."""
    if not isinstance(q, dict) or any(f not in q for f in REQUIRED_FIELDS):
        return False
    options = q["options"]
    if not isinstance(options, list) or len(options) < 2:
        return False
    if len(set(map(str, options))) != len(options):
        return False
    if q["difficulty"] not in DIFFICULTIES:
        return False
    return isinstance(q["answer"], int) and 0 <= q["answer"] < len(options)


def load_bank():
    """Read the question bank file and keep only valid questions."""
    try:
        with open(BANK_PATH, encoding="utf-8-sig") as f:
            bank = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ToolError(f"Question bank could not be loaded: {e}")
    if not isinstance(bank, list):
        raise ToolError("Question bank must be a list of questions.")
    valid = [q for q in bank if is_valid_question(q)]
    if not valid:
        raise ToolError("Question bank has no valid questions.")
    return valid


def list_topics():
    """All topics that have questions."""
    return sorted({q["topic"] for q in load_bank()})


def retrieve_questions(topic, difficulty, n, exclude_ids=()):
    """TOOL 1: pick n questions for a topic.

    Order of preference: questions not seen before, then the closest
    difficulty to the one requested. Ties are broken at random.
    May return fewer than n if the topic is small.
    """
    if difficulty not in DIFFICULTIES:
        raise ToolError(f"Unknown difficulty: {difficulty}")
    if n < 1:
        raise ToolError("Number of questions must be at least 1.")
    pool = [q for q in load_bank() if q["topic"] == topic]
    if not pool:
        raise ToolError(f"No questions found for topic: {topic}")
    target = DIFFICULTIES.index(difficulty)
    random.shuffle(pool)
    pool.sort(
        key=lambda q: (
            q["id"] in exclude_ids,
            abs(DIFFICULTIES.index(q["difficulty"]) - target),
        )
    )
    chosen = pool[:n]
    random.shuffle(chosen)
    return chosen


def evaluate_answer(question, chosen):
    """TOOL 2: mark one answer. Done by code, never by an LLM."""
    options = question["options"]
    if not isinstance(chosen, int) or not 0 <= chosen < len(options):
        raise ToolError("Chosen answer is not a valid option.")
    correct = question["answer"]
    return {
        "is_correct": chosen == correct,
        "correct_index": correct,
        "correct_text": options[correct],
        "explanation": question["explanation"],
    }


def analyse_progress():
    """TOOL 3: summarise the student's history from memory."""
    history = memory.get_history()
    stats = memory.get_topic_stats()
    recent = memory.get_recent_scores(3)
    weakest = stats.iloc[0] if not stats.empty else None
    return {
        "quizzes_taken": len(history),
        "recent_scores": recent,
        "average_recent": sum(recent) / len(recent) if recent else None,
        "weakest_topic": weakest["topic"] if weakest is not None else None,
        "weakest_accuracy": float(weakest["accuracy"]) if weakest is not None else None,
        "topic_stats": stats,
    }
