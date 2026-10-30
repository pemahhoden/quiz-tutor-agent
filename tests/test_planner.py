import json

import pytest

from agent import memory, planner, tools


def make_q(qid, topic, difficulty="easy"):
    return {"id": qid, "topic": topic, "difficulty": difficulty,
            "question": "Q?", "options": ["a", "b", "c"], "answer": 1,
            "hint": "h", "explanation": "e"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    levels = ["easy"] * 3 + ["medium"] * 3
    bank = [make_q(f"a{i}", "A", d) for i, d in enumerate(levels)]
    bank += [make_q(f"b{i}", "B") for i in range(3)]
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(bank))
    monkeypatch.setattr(tools, "BANK_PATH", path)
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "t.db")
    memory.init_db()


def add_quiz(topic, difficulty, correct, ids=None, total=5):
    ids = ids or [f"{topic}{i}{difficulty}" for i in range(total)]
    attempts = [
        {"question_id": qid, "topic": topic, "difficulty": difficulty,
         "chosen": 0, "is_correct": 1 if i < correct else 0, "hint_used": 0}
        for i, qid in enumerate(ids)
    ]
    memory.save_quiz(topic, difficulty, "question_bank", attempts)


def test_new_student_starts_easy(env):
    plan = planner.plan_quiz()
    assert plan["difficulty"] == "easy"
    assert plan["topic"] == "A"
    assert plan["n"] == 5
    assert plan["tool"] == "question_bank"


def test_difficulty_goes_up_after_three_high_scores(env):
    for _ in range(3):
        add_quiz("A", "easy", correct=5)
    plan = planner.plan_quiz()
    assert plan["difficulty"] == "medium"
    assert plan["topic"] == "B"


def test_difficulty_goes_down_after_a_low_score(env):
    add_quiz("A", "medium", correct=1)
    plan = planner.plan_quiz()
    assert plan["difficulty"] == "easy"
    assert plan["topic"] == "A"
    assert plan["hint_mode"] == "proactive"


def test_weak_topic_is_chosen_for_revision(env):
    add_quiz("A", "easy", correct=1)
    add_quiz("B", "easy", correct=5)
    assert planner.plan_quiz()["topic"] == "A"


def test_student_choice_is_respected(env):
    plan = planner.plan_quiz(topic="B", difficulty="hard", n=2)
    assert (plan["topic"], plan["difficulty"], plan["n"]) == ("B", "hard", 2)


def test_bad_input_is_rejected(env):
    for kwargs in [{"topic": "Nope"}, {"difficulty": "x"}, {"n": 0}, {"n": 99}]:
        with pytest.raises(tools.ToolError):
            planner.plan_quiz(**kwargs)


def test_number_of_questions_is_capped(env):
    assert planner.plan_quiz(topic="B", n=10)["n"] == 3


def test_tool_choice_depends_on_unseen_questions_and_llm(env):
    add_quiz("B", "easy", correct=3, ids=["b0", "b1", "b2"], total=3)
    assert planner.plan_quiz(topic="B", n=3)["tool"] == "question_bank"
    assert planner.plan_quiz(topic="B", n=3, llm_available=True)["tool"] == "llm_generator"


def test_next_difficulty_edges():
    assert planner.next_difficulty("hard", [90, 90, 90])[0] == "hard"
    assert planner.next_difficulty("easy", [10])[0] == "easy"
    assert planner.next_difficulty("medium", [70, 60])[0] == "medium"
    assert planner.next_difficulty("easy", [90, 90, 90])[0] == "medium"


def test_recommend_next_has_a_message(env):
    assert "Next:" in planner.recommend_next()["message"]
