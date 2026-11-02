import json

import pytest

from agent import generators, guardrails, memory, notes, tools
from agent.agent import QuizAgent


def good(qid="x1", topic="T"):
    return {"id": qid, "topic": topic, "difficulty": "easy",
            "question": "What is 2 + 2?", "options": ["3", "4", "5"], "answer": 1,
            "hint": "Add them.", "explanation": "2 + 2 = 4."}


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "t.db")
    memory.init_db()


def use_bank(monkeypatch, tmp_path, bank):
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(bank))
    monkeypatch.setattr(tools, "BANK_PATH", path)


def mark_seen(topic, ids):
    attempts = [{"question_id": i, "topic": topic, "difficulty": "easy",
                 "chosen": 1, "is_correct": 1, "hint_used": 0} for i in ids]
    memory.save_quiz(topic, "easy", "question_bank", attempts)


def test_real_bank_passes_all_checks():
    for q in tools.load_bank():
        ok, reason = guardrails.check_question(q)
        assert ok, (q["id"], reason)


def test_generated_questions_pass_all_checks():
    for topic in generators.TEMPLATES:
        for level in tools.DIFFICULTIES:
            for q in generators.generate_questions(topic, level, 10):
                ok, reason = guardrails.check_question(q)
                assert ok, (q["question"], reason)


def test_bad_questions_are_caught():
    assert guardrails.check_question(good())[0]
    changes = [
        {"question": "   "},
        {"question": "x" * 601},
        {"options": ["3", "4" * 201, "5"]},
        {"options": ["Yes", "yes", "No"], "answer": 2},
        {"hint": ""},
        {"explanation": " "},
        {"question": "The answer is 4 so what is it?", "source": "s"},
    ]
    for change in changes:
        assert guardrails.check_question({**good(), **change})[0] is False, change


def test_filter_keeps_only_good_questions():
    bad = {**good("b"), "hint": ""}
    assert guardrails.filter_questions([good("a"), bad]) == [good("a")]


def test_erase_all_data(fresh_db):
    guardrails.erase_all_data()  # works even before the notes table exists
    notes.add_note("Temp", "Overfitting is when a model memorises the training data. " * 5)
    mark_seen("T", ["x1"])
    guardrails.erase_all_data()
    assert memory.get_history().empty
    assert memory.get_attempts().empty
    assert notes.list_notes() == []


def test_agent_drops_bad_generated_questions(fresh_db, tmp_path, monkeypatch):
    use_bank(monkeypatch, tmp_path, [good(f"b{i}", "B") for i in range(3)])
    mark_seen("B", ["b0", "b1", "b2"])

    def mixed(topic, difficulty, n):
        return [good("g0", "B"), good("g1", "B"), {**good("g2", "B"), "hint": ""}]

    session = QuizAgent(generator=mixed).start_quiz(topic="B", n=3)
    assert session["tool_used"] == "question_generator"
    assert {q["id"] for q in session["questions"]} == {"g0", "g1"}


def test_all_bad_generated_questions_fall_back_to_the_bank(fresh_db, tmp_path, monkeypatch):
    use_bank(monkeypatch, tmp_path, [good(f"b{i}", "B") for i in range(3)])
    mark_seen("B", ["b0", "b1", "b2"])

    def broken(topic, difficulty, n):
        return [{**good("g0", "B"), "hint": ""}]

    session = QuizAgent(generator=broken).start_quiz(topic="B", n=3)
    assert session["tool_used"] == "question_bank"
    assert session["notices"]


def test_bad_bank_question_is_filtered(fresh_db, tmp_path, monkeypatch):
    use_bank(monkeypatch, tmp_path,
             [good("a0", "A"), {**good("a1", "A"), "explanation": ""}])
    session = QuizAgent().start_quiz(topic="A", n=2)
    assert [q["id"] for q in session["questions"]] == ["a0"]


def test_quiz_with_only_bad_questions_raises_an_error(fresh_db, tmp_path, monkeypatch):
    use_bank(monkeypatch, tmp_path, [{**good("a0", "A"), "explanation": ""}])
    with pytest.raises(tools.ToolError):
        QuizAgent().start_quiz(topic="A", n=1)
