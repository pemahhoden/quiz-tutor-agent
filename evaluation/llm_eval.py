"""Measures the local LLM: speed, rejected replies and basic reply quality.

Run it from the Evaluation page, or with:  python -m evaluation.llm_eval
Ollama must be running with the model. It compares two model settings.
"""
import json
import random
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from agent import generators, llm, tools

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CALLS = 4
CONFIGS = {
    "Default settings": {"temperature": 0.2, "num_predict": 80},
    "Small context + 3 threads": {"temperature": 0.2, "num_predict": 80,
                                  "num_ctx": 1024, "num_thread": 3},
}


def make_cases(n):
    """Wrong-answer cases made from generated Statistics questions."""
    random.seed(5)
    cases = []
    for q in generators.generate_questions("Statistics", "easy", n):
        chosen = (q["answer"] + 1) % len(q["options"])
        cases.append((q, q["options"][chosen], q["options"][q["answer"]],
                      q["explanation"]))
    return cases


def run_config(options, cases):
    """One warm-up call (loads the model), then one timed call per case."""
    llm.OPTIONS = options
    q, chosen, correct, stored = cases[0]
    start = time.perf_counter()
    try:
        llm.explain(q, chosen, correct, stored)
    except Exception:
        pass
    warmup = time.perf_counter() - start
    rows = []
    for q, chosen, correct, stored in cases:
        start = time.perf_counter()
        outcome, text = "ok", ""
        try:
            text = llm.explain(q, chosen, correct, stored)
        except tools.ToolError as e:
            message = str(e)
            if "numbers" in message:
                outcome = "rejected_numbers"
            elif "empty" in message:
                outcome = "rejected_empty"
            elif "respond" in message:
                outcome = "no_response"
            else:
                outcome = "other_error"
        seconds = round(time.perf_counter() - start, 1)
        rows.append({
            "outcome": outcome, "seconds": seconds, "reply": text,
            "names_correct_answer": outcome == "ok" and correct.lower() in text.lower(),
        })
    return warmup, rows


def percent(part, whole):
    return round(100 * part / whole, 1) if whole else 0.0


def summarise(rows, warmup):
    seconds = [r["seconds"] for r in rows]
    ok = [r for r in rows if r["outcome"] == "ok"]
    rejected = sum(r["outcome"] in ("rejected_numbers", "rejected_empty") for r in rows)
    return {
        "calls": len(rows),
        "reply_ok_percent": percent(len(ok), len(rows)),
        "rejected_by_safety_check_percent": percent(rejected, len(rows)),
        "no_response_percent": percent(
            sum(r["outcome"] == "no_response" for r in rows), len(rows)),
        "names_correct_answer_percent": percent(
            sum(r["names_correct_answer"] for r in ok), len(ok)),
        "median_seconds": round(statistics.median(seconds), 1),
        "slowest_seconds": max(seconds),
        "warmup_seconds": round(warmup, 1),
    }


def make_chart(all_rows):
    charts = RESULTS / "charts"
    charts.mkdir(exist_ok=True)
    groups = {}
    for r in all_rows:
        groups.setdefault(r["setting"], []).append(r["seconds"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(list(groups.values()))
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(list(groups))
    ax.set_ylabel("Seconds per explanation")
    ax.set_title("AI tutor speed by model setting")
    fig.tight_layout()
    fig.savefig(charts / "llm_speed.png", dpi=150)
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "llm_results.json"
    if not llm.available():
        message = "Ollama is not running, or the model is not downloaded."
        out.write_text(json.dumps({"available": False, "message": message}, indent=2),
                       encoding="utf-8")
        print(message)
        return
    cases = make_cases(CALLS)
    configs, all_rows = {}, []
    for name, options in CONFIGS.items():
        print(f"Testing: {name} ...", flush=True)
        warmup, rows = run_config(options, cases)
        configs[name] = summarise(rows, warmup)
        all_rows.extend({"setting": name, **r} for r in rows)
    summary = {
        "available": True,
        "model": llm.MODEL,
        "calls_per_config": CALLS,
        "configs": configs,
        "faster_setting": min(configs, key=lambda k: configs[k]["median_seconds"]),
        "note": "Speed depends on free memory and on what else is running. "
                "'Names the correct answer' is a simple check, not a full quality score.",
    }
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(all_rows).to_csv(RESULTS / "llm_calls.csv", index=False)
    make_chart(all_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
