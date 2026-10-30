import json

import pytest

from agent import memory, tools


def make_q(qid, topic="T", difficulty="easy"):
    return {"id": qid, "topic": topic, "difficulty": difficulty,
            "question": "Q?", "options": ["a", "b", "c"], "answer": 1,
            "hint": "h", "explanation": "e"}


@pytest.fixture
def small_bank(tmp_path, monkeypatch):
    bank = [make_q("a1", difficulty="easy"), make_q("a2", difficulty="medium"),
            make_q("a3", difficulty="hard"), make_q("b1", topic="Other"),
            {"id": "bad"}]
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(bank))
    monkeypatch.setattr(tools, "BANK_PATH", path)


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "test.db")
    memory.init_db()


def test_invalid_question_is_skipped(small_bank):
    assert len(tools.load_bank()) == 4


def test_real_bank_has_no_invalid_questions():
    with open(tools.BANK_PATH, encoding="utf-8-sig") as f:
        raw = json.load(f)
    assert len(tools.load_bank()) == len(raw)


def test_retrieve_prefers_requested_difficulty(small_bank):
    assert tools.retrieve_questions("T", "medium", 1)[0]["id"] == "a2"


def test_retrieve_avoids_seen_questions(small_bank):
    got = tools.retrieve_questions("T", "medium", 1, exclude_ids={"a2"})
    assert got[0]["id"] != "a2"


def test_retrieve_returns_fewer_when_topic_is_small(small_bank):
    assert len(tools.retrieve_questions("T", "easy", 10)) == 3


def test_retrieve_unknown_topic_and_difficulty(small_bank):
    with pytest.raises(tools.ToolError):
        tools.retrieve_questions("Nope", "easy", 1)
    with pytest.raises(tools.ToolError):
        tools.retrieve_questions("T", "impossible", 1)


def test_missing_bank_raises_tool_error(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "BANK_PATH", tmp_path / "missing.json")
    with pytest.raises(tools.ToolError):
        tools.load_bank()


def test_evaluate_answer():
    q = make_q("x")
    assert tools.evaluate_answer(q, 1)["is_correct"] is True
    assert tools.evaluate_answer(q, 0)["is_correct"] is False
    with pytest.raises(tools.ToolError):
        tools.evaluate_answer(q, 9)


def test_analyse_progress_empty_and_with_data(fresh_db):
    empty = tools.analyse_progress()
    assert empty["quizzes_taken"] == 0 and empty["weakest_topic"] is None
    attempts = [
        {"question_id": "p1", "topic": "Python basics", "difficulty": "easy",
         "chosen": 1, "is_correct": 1, "hint_used": 0},
        {"question_id": "p2", "topic": "Python basics", "difficulty": "easy",
         "chosen": 0, "is_correct": 0, "hint_used": 1},
    ]
    memory.save_quiz("Python basics", "easy", "question_bank", attempts)
    result = tools.analyse_progress()
    assert result["recent_scores"] == [50.0]
    assert result["weakest_topic"] == "Python basics"
