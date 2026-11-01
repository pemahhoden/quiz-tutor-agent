import pytest

from agent import memory, notes, planner
from agent.agent import QuizAgent

SAMPLE = """Overfitting is when a model memorises the training data and fails on new data.
Underfitting is when a model is too simple to learn the pattern in the data.
Regularisation is a method that adds a penalty to reduce overfitting in a model.
Cross-validation is a method that tests a model on several different splits of the data.
A good model should learn the pattern in the training data and also work on new data."""

TOPIC = "My notes: ML notes"


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "t.db")
    memory.init_db()


def test_planner_uses_the_notes_tool_for_notes(fresh_db):
    notes.add_note("ML notes", SAMPLE)
    plan = planner.plan_quiz(topic=TOPIC, n=5)
    assert plan["tool"] == "notes_generator"
    assert plan["n"] == 5


def test_agent_runs_a_whole_quiz_from_notes(fresh_db):
    notes.add_note("ML notes", SAMPLE)
    agent = QuizAgent()
    session = agent.start_quiz(topic=TOPIC, n=4)
    assert session["tool_used"] == "notes_generator"
    assert all(q["topic"] == TOPIC for q in session["questions"])
    for q in session["questions"]:
        agent.submit_answer(session, q["id"], q["answer"])
    result = agent.finish_quiz(session)
    assert result["score"] == 100.0


def test_deleted_notes_are_no_longer_planned(fresh_db):
    note_id = notes.add_note("ML notes", SAMPLE)
    agent = QuizAgent()
    session = agent.start_quiz(topic=TOPIC, n=3)
    for q in session["questions"]:
        agent.submit_answer(session, q["id"], 0)
    agent.finish_quiz(session)
    notes.delete_note(note_id)
    plan = planner.plan_quiz()
    assert not plan["topic"].startswith("My notes")
