import pytest

from agent import tools
from evaluation import llm_eval


def test_summarise_computes_the_numbers():
    rows = [
        {"outcome": "ok", "seconds": 10.0, "names_correct_answer": True, "reply": "x"},
        {"outcome": "ok", "seconds": 20.0, "names_correct_answer": False, "reply": "y"},
        {"outcome": "rejected_numbers", "seconds": 30.0,
         "names_correct_answer": False, "reply": ""},
        {"outcome": "no_response", "seconds": 40.0,
         "names_correct_answer": False, "reply": ""},
    ]
    s = llm_eval.summarise(rows, warmup=50.0)
    assert s["calls"] == 4
    assert s["reply_ok_percent"] == 50.0
    assert s["rejected_by_safety_check_percent"] == 25.0
    assert s["no_response_percent"] == 25.0
    assert s["names_correct_answer_percent"] == 50.0
    assert s["median_seconds"] == 25.0
    assert s["slowest_seconds"] == 40.0
    assert s["warmup_seconds"] == 50.0


def test_run_config_labels_the_outcomes(monkeypatch):
    def fake(q, chosen, correct, stored):
        raise tools.ToolError("the reply contained numbers that are not in the lesson")

    monkeypatch.setattr(llm_eval.llm, "OPTIONS", dict(llm_eval.llm.OPTIONS))
    monkeypatch.setattr(llm_eval.llm, "explain", fake)
    warmup, rows = llm_eval.run_config({"num_predict": 10}, llm_eval.make_cases(2))
    assert [r["outcome"] for r in rows] == ["rejected_numbers"] * 2
    assert llm_eval.llm.OPTIONS == {"num_predict": 10}
