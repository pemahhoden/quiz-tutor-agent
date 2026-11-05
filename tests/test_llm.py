import json
import urllib.error

import pytest

from agent import llm, memory, tools
from agent.agent import QuizAgent

QUESTION = {"question": "What is the mean of [2, 4, 6, 8]?"}
CHOSEN, CORRECT = "4", "5"
STORED = "The sum is 20 and there are 4 numbers, so the mean is 20 / 4 = 5."


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class GoodLLM:
    def explain(self, q, chosen, correct, stored):
        return "Your answer does not match the explanation, and the correct one does."

    def advice(self, progress, message):
        return "Practise a little every day and review your mistakes."


class BrokenLLM:
    def explain(self, *args):
        raise tools.ToolError("the AI tutor did not respond")

    def advice(self, *args):
        raise tools.ToolError("the AI tutor did not respond")


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "t.db")
    memory.init_db()


def wrong_answer_session(llm_object):
    agent = QuizAgent(llm=llm_object)
    session = agent.start_quiz(topic="Statistics", difficulty="easy", n=2)
    q = session["questions"][0]
    agent.submit_answer(session, q["id"], (q["answer"] + 1) % len(q["options"]))
    return agent, session, q


def test_clean_reply_rules():
    assert llm.clean_reply("`code` is   fine and long enough") == "code is fine and long enough"
    with pytest.raises(tools.ToolError):
        llm.clean_reply("  ")
    long_text = llm.clean_reply("A sentence here. " * 100)
    assert len(long_text) <= llm.MAX_CHARS and long_text.endswith(".")
    assert len(llm.clean_reply("x" * 800)) <= llm.MAX_CHARS + 3


def test_numbers_grounded():
    assert llm.numbers_grounded("a 5 b", "5 is here")
    assert llm.numbers_grounded("no numbers at all", "5 is here")
    assert not llm.numbers_grounded("7 apples", "5 is here")


def test_explain_accepts_a_grounded_reply(monkeypatch):
    monkeypatch.setattr(
        llm, "ask",
        lambda prompt: "You picked 4, but the sum is 20 and 20 / 4 is 5, so 5 is right.")
    assert "5" in llm.explain(QUESTION, CHOSEN, CORRECT, STORED)


def test_explain_rejects_made_up_numbers(monkeypatch):
    monkeypatch.setattr(
        llm, "ask", lambda prompt: "The answer is 7 because the numbers add up to 28.")
    with pytest.raises(tools.ToolError):
        llm.explain(QUESTION, CHOSEN, CORRECT, STORED)


def test_explain_rejects_an_empty_reply(monkeypatch):
    monkeypatch.setattr(llm, "ask", lambda prompt: "  ")
    with pytest.raises(tools.ToolError):
        llm.explain(QUESTION, CHOSEN, CORRECT, STORED)


def test_ask_handles_a_missing_model_server(monkeypatch):
    def offline(*args, **kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(llm.urllib.request, "urlopen", offline)
    with pytest.raises(tools.ToolError):
        llm.ask("hello")
    monkeypatch.setattr(
        llm.urllib.request, "urlopen",
        lambda *a, **k: FakeResponse({"response": "Hello there friend."}))
    assert llm.ask("hello") == "Hello there friend."


def test_available_checks_the_server_and_caches(monkeypatch):
    def offline(*args, **kwargs):
        raise urllib.error.URLError("offline")

    llm._cache["time"] = 0.0
    monkeypatch.setattr(llm.urllib.request, "urlopen", offline)
    assert llm.available() is False
    llm._cache["time"] = 0.0
    monkeypatch.setattr(
        llm.urllib.request, "urlopen",
        lambda *a, **k: FakeResponse({"models": [{"name": llm.MODEL}]}))
    assert llm.available() is True
    llm._cache["time"] = 0.0


def test_agent_uses_the_llm_for_explanations(fresh_db):
    agent, session, q = wrong_answer_session(GoodLLM())
    out = agent.explain_answer(session, q["id"])
    assert out["source"] == "llm" and "does not match" in out["text"]


def test_agent_falls_back_when_the_llm_fails(fresh_db):
    agent, session, q = wrong_answer_session(BrokenLLM())
    out = agent.explain_answer(session, q["id"])
    assert out["source"] == "stored"
    assert out["text"] == q["explanation"] and out["note"]


def test_agent_works_without_an_llm(fresh_db):
    agent, session, q = wrong_answer_session(None)
    assert agent.explain_answer(session, q["id"])["source"] == "stored"


def test_cannot_explain_before_answering(fresh_db):
    agent, session, q = wrong_answer_session(GoodLLM())
    other = session["questions"][1]["id"]
    with pytest.raises(tools.ToolError):
        agent.explain_answer(session, other)


def test_study_advice_uses_the_llm_or_falls_back(fresh_db):
    assert QuizAgent(llm=GoodLLM()).study_advice()["source"] == "llm"
    out = QuizAgent(llm=BrokenLLM()).study_advice()
    assert out["source"] == "rules" and out["text"].startswith("Next:")


def test_cut_off_sentences_are_trimmed():
    cut = "This is the first full sentence about the topic. And this one is cut o"
    assert llm.clean_reply(cut) == "This is the first full sentence about the topic."
