import pytest

from agent import memory


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "test.db")
    memory.init_db()


def make_attempt(qid, correct, hint=0):
    return {"question_id": qid, "topic": "Python basics", "difficulty": "easy",
            "chosen": 1, "is_correct": correct, "hint_used": hint}


def test_save_and_read(fresh_db):
    attempts = [make_attempt("py-001", 1), make_attempt("py-002", 0, hint=1)]
    quiz_id = memory.save_quiz("Python basics", "easy", "question_bank", attempts)
    assert quiz_id == 1
    assert memory.get_recent_scores(3) == [50.0]
    assert memory.get_topic_stats().loc[0, "accuracy"] == 50.0
    assert set(memory.get_seen_question_ids("Python basics")) == {"py-001", "py-002"}


def test_empty_quiz_is_rejected(fresh_db):
    with pytest.raises(ValueError):
        memory.save_quiz("Python basics", "easy", "question_bank", [])
