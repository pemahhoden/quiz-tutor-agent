"""Long-term memory for the Quiz Tutor Agent, stored in SQLite."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "quiz_agent.db"


@contextmanager
def db():
    """Open the database, save changes, and always close it."""
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create the tables if they do not exist yet."""
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS quizzes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                topic TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                tool_used TEXT NOT NULL,
                score REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                quiz_id INTEGER NOT NULL,
                question_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                chosen INTEGER NOT NULL,
                is_correct INTEGER NOT NULL,
                hint_used INTEGER NOT NULL
            );
            """
        )


def save_quiz(topic, difficulty, tool_used, attempts):
    """Save one finished quiz and all its answers. Returns the quiz id."""
    if not attempts:
        raise ValueError("Cannot save a quiz with no answers.")
    score = 100.0 * sum(a["is_correct"] for a in attempts) / len(attempts)
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO quizzes (created_at, topic, difficulty, tool_used, score) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), topic, difficulty, tool_used, score),
        )
        quiz_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO attempts (quiz_id, question_id, topic, difficulty, "
            "chosen, is_correct, hint_used) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (quiz_id, a["question_id"], a["topic"], a["difficulty"],
                 a["chosen"], a["is_correct"], a["hint_used"])
                for a in attempts
            ],
        )
    return quiz_id


def get_history():
    """All past quizzes as a table (oldest first)."""
    with db() as conn:
        return pd.read_sql_query("SELECT * FROM quizzes ORDER BY id", conn)


def get_attempts():
    """All answered questions as a table."""
    with db() as conn:
        return pd.read_sql_query("SELECT * FROM attempts ORDER BY id", conn)


def get_topic_stats():
    """Accuracy per topic, weakest topic first."""
    with db() as conn:
        return pd.read_sql_query(
            "SELECT topic, COUNT(*) AS answered, "
            "ROUND(100.0 * SUM(is_correct) / COUNT(*), 1) AS accuracy "
            "FROM attempts GROUP BY topic ORDER BY accuracy",
            conn,
        )


def get_recent_scores(n=3):
    """Scores of the last n quizzes, oldest first."""
    with db() as conn:
        rows = conn.execute(
            "SELECT score FROM quizzes ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
    return [r[0] for r in reversed(rows)]


def get_seen_question_ids(topic):
    """Ids of questions already answered in this topic."""
    with db() as conn:
        rows = conn.execute(
            "SELECT DISTINCT question_id FROM attempts WHERE topic = ?", (topic,)
        ).fetchall()
    return [r[0] for r in rows]
