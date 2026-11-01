import json

import pytest

from agent import memory, tools
from agent.agent import QuizAgent


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


def answer_all(agent, session, chosen):
    for q in session["questions"]:
        agent.submit_answer(session, q["id"], chosen)


def mark_b_as_seen():
    attempts = [
        {"question_id": f"b{i}", "topic": "B", "difficulty": "easy",
         "chosen": 1, "is_correct": 1, "hint_used": 0}
        for i in range(3)
    ]
    memory.save_quiz("B", "easy", "question_bank", attempts)


def test_full_flow_saves_to_memory(env):
    agent = QuizAgent()
    session = agent.start_quiz(topic="A", n=3)
    answer_all(agent, session, chosen=1)
    result = agent.finish_quiz(session)
    assert result["score"] == 100.0
    assert result["correct"] == 3 and result["total"] == 3
    assert len(memory.get_history()) == 1
    steps = [t["step"] for t in session["trace"]]
    for step in ["Perceive", "Plan", "Select tool", "Act", "Evaluate",
                 "Update memory", "Adapt"]:
        assert step in steps


def test_agent_adapts_after_a_bad_quiz(env):
    agent = QuizAgent()
    session = agent.start_quiz(topic="A", difficulty="medium", n=3)
    answer_all(agent, session, chosen=0)
    result = agent.finish_quiz(session)
    assert result["score"] == 0.0
    assert result["recommendation"]["difficulty"] == "easy"
    assert result["recommendation"]["topic"] == "A"


def test_hint_use_is_recorded(env):
    agent = QuizAgent()
    session = agent.start_quiz(topic="A", n=2)
    first = session["questions"][0]
    assert agent.get_hint(session, first["id"]) == "h"
    answer_all(agent, session, chosen=1)
    agent.finish_quiz(session)
    assert memory.get_attempts()["hint_used"].sum() == 1


def test_unknown_question_is_rejected(env):
    agent = QuizAgent()
    session = agent.start_quiz(topic="A", n=2)
    with pytest.raises(tools.ToolError):
        agent.submit_answer(session, "nope", 0)


def test_same_question_cannot_be_answered_twice(env):
    agent = QuizAgent()
    session = agent.start_quiz(topic="A", n=2)
    qid = session["questions"][0]["id"]
    agent.submit_answer(session, qid, 1)
    with pytest.raises(tools.ToolError):
        agent.submit_answer(session, qid, 1)


def test_cannot_finish_before_answering_everything(env):
    agent = QuizAgent()
    session = agent.start_quiz(topic="A", n=2)
    with pytest.raises(tools.ToolError):
        agent.finish_quiz(session)


def test_llm_failure_falls_back_to_the_bank(env):
    mark_b_as_seen()

    def broken(topic, difficulty, n):
        raise RuntimeError("LLM is down")

    agent = QuizAgent(generator=broken)
    session = agent.start_quiz(topic="B", n=3)
    assert session["plan"]["tool"] == "question_generator"
    assert session["tool_used"] == "question_bank"
    assert session["notices"]
    assert len(session["questions"]) == 3


def test_valid_generated_questions_are_used(env):
    mark_b_as_seen()

    def good(topic, difficulty, n):
        return [make_q(f"g{i}", "B") for i in range(n)]

    agent = QuizAgent(generator=good)
    session = agent.start_quiz(topic="B", n=3)
    assert session["tool_used"] == "question_generator"
    assert all(q["id"].startswith("g") for q in session["questions"])


def test_invalid_generated_questions_are_rejected(env):
    mark_b_as_seen()

    def bad(topic, difficulty, n):
        return [{"id": "broken"}]

    agent = QuizAgent(generator=bad)
    session = agent.start_quiz(topic="B", n=3)
    assert session["tool_used"] == "question_bank"
    assert session["notices"]
