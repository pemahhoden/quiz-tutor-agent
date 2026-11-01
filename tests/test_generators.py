import ast
import re
import statistics

import pytest

from agent import generators, memory, planner, tools
from agent.agent import QuizAgent


def answer_text(q):
    return q["options"][q["answer"]]


def build(template, level="easy"):
    q = generators._build(template, level)
    assert q is not None
    return q


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "t.db")
    memory.init_db()


def test_every_topic_and_level_gives_valid_questions():
    for topic, levels in generators.TEMPLATES.items():
        for level in tools.DIFFICULTIES:
            assert level in levels
            questions = generators.generate_questions(topic, level, 10)
            assert len(questions) == 10
            for q in questions:
                assert tools.is_valid_question(q)
                assert q["topic"] == topic
                assert q["difficulty"] == level
                assert q["id"].startswith("gen-")


def test_unknown_topic_is_rejected():
    with pytest.raises(tools.ToolError):
        generators.generate_questions("Cooking", "easy", 3)


def test_python_arithmetic_answers_are_correct():
    for _ in range(40):
        q = build(generators._py_arith)
        a, op, b = re.search(r"value of (\d+) (//|%) (\d+)", q["question"]).groups()
        expected = int(a) // int(b) if op == "//" else int(a) % int(b)
        assert answer_text(q) == str(expected)


def test_python_slice_answers_are_correct():
    for _ in range(40):
        q = build(generators._py_slice, "medium")
        m = re.search(r"lst = (\[.*?\])\. What does lst\[(\d+):(\d+)\]", q["question"])
        lst, i, j = ast.literal_eval(m.group(1)), int(m.group(2)), int(m.group(3))
        assert answer_text(q) == str(lst[i:j])


def test_mean_answers_are_correct():
    for _ in range(40):
        q = build(generators._st_mean)
        nums = ast.literal_eval(re.search(r"mean of (\[.*?\])\?", q["question"]).group(1))
        assert answer_text(q) == generators._fmt(sum(nums) / len(nums))


def test_median_answers_are_correct():
    for _ in range(40):
        q = build(generators._st_median)
        nums = ast.literal_eval(re.search(r"median of (\[.*?\])\?", q["question"]).group(1))
        assert answer_text(q) == generators._fmt(statistics.median(nums))


def test_accuracy_answers_are_correct():
    for _ in range(40):
        q = build(generators._ml_accuracy)
        n, c = re.search(r"made (\d+) predictions and (\d+) were", q["question"]).groups()
        assert answer_text(q) == generators._fmt(100 * int(c) / int(n)) + "%"


def test_split_answers_are_correct():
    for _ in range(40):
        q = build(generators._ml_split, "medium")
        n, t = re.search(
            r"has (\d+) rows and is split \d+% for training and (\d+)% for testing",
            q["question"]).groups()
        assert answer_text(q) == str(int(n) * int(t) // 100)


def test_planner_allows_more_questions_than_the_bank_when_a_generator_exists(fresh_db):
    plan = planner.plan_quiz(topic="Python basics", n=10, llm_available=True)
    assert plan["n"] == 10
    assert plan["tool"] == "question_generator"


def test_agent_mixes_bank_and_generated_questions(fresh_db):
    agent = QuizAgent(generator=generators.generate_questions)
    session = agent.start_quiz(topic="Python basics", difficulty="easy", n=10)
    ids = [q["id"] for q in session["questions"]]
    assert len(ids) == 10 and len(set(ids)) == 10
    assert session["tool_used"] == "question_generator"
    assert any(i.startswith("gen-") for i in ids)
    assert any(not i.startswith("gen-") for i in ids)
    for q in session["questions"]:
        agent.submit_answer(session, q["id"], 0)
    agent.finish_quiz(session)
    assert len(memory.get_attempts()) == 10
