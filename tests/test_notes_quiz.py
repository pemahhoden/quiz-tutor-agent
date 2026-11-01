import re

import pytest

from agent import memory, notes, notes_quiz, tools

SAMPLE = """Overfitting is when a model memorises the training data and fails on new data.
Underfitting is when a model is too simple to learn the pattern in the data.
Regularisation is a method that adds a penalty to reduce overfitting in a model.
Cross-validation is a method that tests a model on several different splits of the data.
A good model should learn the pattern in the training data and also work on new data.
The test data is kept apart so that the model is judged on data it has never seen.
Overfitting often happens when the model is very complex and the training data is small."""

TERMS = {"Overfitting", "Underfitting", "Regularisation", "Cross-validation"}


def answer_text(q):
    return q["options"][q["answer"]]


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "t.db")
    memory.init_db()


def test_easy_questions_come_from_definitions():
    note = {"id": 1, "title": "ML notes", "content": SAMPLE}
    questions = notes_quiz.generate_from_note(note, "easy", 4)
    assert len(questions) == 4
    for q in questions:
        ans = answer_text(q)
        assert tools.is_valid_question(q)
        assert q["topic"] == "My notes: ML notes"
        assert q["id"].startswith("note1-")
        assert notes_quiz.BLANK in q["question"]
        assert re.search(rf"\b{re.escape(ans)}\b", q["question"], re.I) is None
        assert ans.lower() in q["explanation"].lower()
        assert ans in TERMS


def test_medium_and_hard_questions_are_valid():
    note = {"id": 1, "title": "ML notes", "content": SAMPLE}
    for level in ["medium", "hard"]:
        questions = notes_quiz.generate_from_note(note, level, 3)
        assert 1 <= len(questions) <= 3
        for q in questions:
            assert tools.is_valid_question(q)
            assert q["difficulty"] in tools.DIFFICULTIES


def test_falls_back_to_other_levels_for_small_notes():
    content = "\n".join(SAMPLE.splitlines()[:4])
    note = {"id": 2, "title": "Defs", "content": content}
    questions = notes_quiz.generate_from_note(note, "hard", 3)
    assert len(questions) == 3
    assert all(q["difficulty"] == "easy" for q in questions)


def test_notes_without_clear_sentences_raise_an_error():
    note = {"id": 3, "title": "Bad", "content": "Ok. Yes. No way. Fine."}
    with pytest.raises(tools.ToolError):
        notes_quiz.generate_from_note(note, "easy", 3)


def test_generate_for_topic_uses_the_saved_note(fresh_db):
    notes.add_note("ML notes", SAMPLE)
    questions = notes_quiz.generate_for_topic("My notes: ML notes", "easy", 3)
    assert len(questions) == 3
    with pytest.raises(tools.ToolError):
        notes_quiz.generate_for_topic("My notes: Missing", "easy", 3)
    with pytest.raises(tools.ToolError):
        notes_quiz.generate_for_topic("Statistics", "easy", 3)


def test_ids_are_stable_for_the_same_note():
    note = {"id": 1, "title": "ML notes", "content": SAMPLE}
    first = {q["id"] for q in notes_quiz.generate_from_note(note, "easy", 4)}
    second = {q["id"] for q in notes_quiz.generate_from_note(note, "easy", 4)}
    assert first == second


def test_unknown_difficulty_is_rejected():
    note = {"id": 1, "title": "ML notes", "content": SAMPLE}
    with pytest.raises(tools.ToolError):
        notes_quiz.generate_from_note(note, "impossible", 3)


def test_small_notes_always_give_questions():
    content = "\n".join(SAMPLE.splitlines()[:4])
    note = {"id": 4, "title": "Defs", "content": content}
    for _ in range(30):
        assert len(notes_quiz.generate_from_note(note, "easy", 4)) == 4
