"""Guardrails: checks that keep the quiz safe, reliable and private."""
import re
import sqlite3

from agent import memory, tools

MAX_QUESTION_CHARS = 600
MAX_OPTION_CHARS = 200
MAX_TEXT_CHARS = 800


def check_question(q):
    """Return (ok, reason). Stricter than tools.is_valid_question."""
    if not tools.is_valid_question(q):
        return False, "malformed question"
    if not str(q["question"]).strip() or len(str(q["question"])) > MAX_QUESTION_CHARS:
        return False, "question text is empty or too long"
    options = [str(o) for o in q["options"]]
    if any(not o.strip() or len(o) > MAX_OPTION_CHARS for o in options):
        return False, "an option is empty or too long"
    if len({o.strip().lower() for o in options}) != len(options):
        return False, "two options are the same"
    for field in ("hint", "explanation"):
        text = str(q[field])
        if not text.strip() or len(text) > MAX_TEXT_CHARS:
            return False, f"{field} is empty or too long"
    if "source" in q:  # questions made from the student's notes
        answer = options[q["answer"]]
        if re.search(rf"\b{re.escape(answer)}\b", str(q["question"]), re.IGNORECASE):
            return False, "the question gives away the answer"
    return True, ""


def filter_questions(questions):
    """Keep only the questions that pass every check."""
    return [q for q in questions if check_question(q)[0]]


def erase_all_data():
    """Privacy: delete every quiz, answer and note stored on this laptop."""
    with memory.db() as conn:
        for table in ("attempts", "quizzes", "notes"):
            try:
                conn.execute(f"DELETE FROM {table}")
            except sqlite3.OperationalError:
                pass  # this table was never created
