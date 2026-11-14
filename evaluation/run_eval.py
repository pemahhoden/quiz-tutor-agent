"""Evaluation of the Quiz Tutor Agent.

Run from the project folder:  python -m evaluation.run_eval

Everything runs on a temporary database, so your real quizzes and notes are
not touched. Every number is measured by this script, none is typed by hand.
"""
import csv
import json
import os
import random
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from agent import generators, guardrails, memory, notes, notes_quiz, planner, tools
from agent.agent import QuizAgent
from evaluation import hard_tests

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SEED = 42

SAMPLE_NOTE = """Overfitting is when a model memorises the training data and fails on new data.
Underfitting is when a model is too simple to learn the pattern in the data.
Regularisation is a method that adds a penalty to reduce overfitting in a model.
Cross-validation is a method that tests a model on several different splits of the data.
A good model should learn the pattern in the training data and also work on new data.
The test data is kept apart so that the model is judged on data it has never seen.
Overfitting often happens when the model is very complex and the training data is small."""

_TMP = tempfile.TemporaryDirectory()
_COUNT = [0]


def fresh_db():
    """Switch to a new, empty temporary database."""
    _COUNT[0] += 1
    memory.DB_PATH = Path(_TMP.name) / f"eval_{_COUNT[0]}.db"
    memory.init_db()


fresh_db()  # from here on, the real database is never touched


def pct(x):
    return round(float(x) * 100, 1)


def mark_seen(topic, unseen):
    """Pretend the student has seen all bank questions except `unseen` of them."""
    if unseen == "all":
        return
    ids = [q["id"] for q in tools.load_bank() if q["topic"] == topic]
    seen = ids[: max(len(ids) - int(unseen), 0)]
    if seen:
        attempts = [{"question_id": i, "topic": topic, "difficulty": "easy",
                     "chosen": 1, "is_correct": 1, "hint_used": 0} for i in seen]
        memory.save_quiz(topic, "easy", "question_bank", attempts)


# ---------- 1. generation, completion, answer marking, latency ----------
def run_generation_experiment(repeats=5):
    rows = []
    for topic in tools.list_topics():
        for level in tools.DIFFICULTIES:
            for n in (3, 5, 10):
                for _ in range(repeats):
                    fresh_db()
                    agent = QuizAgent(generator=generators.generate_questions)
                    row = {"topic": topic, "difficulty": level, "n": n,
                           "start_ms": None, "finish_ms": None, "returned": 0,
                           "tool": "error", "generated": False, "completed": False,
                           "eval_checks": 0, "eval_correct": 0, "error": ""}
                    t0 = time.perf_counter()
                    try:
                        session = agent.start_quiz(topic, level, n)
                    except Exception as e:
                        row["error"] = repr(e)
                        rows.append(row)
                        continue
                    row["start_ms"] = (time.perf_counter() - t0) * 1000
                    row["returned"] = len(session["questions"])
                    row["tool"] = session["tool_used"]
                    row["generated"] = row["returned"] == n
                    t1 = time.perf_counter()
                    try:
                        for q in session["questions"]:
                            if random.random() < 0.5:
                                chosen = q["answer"]
                            else:
                                chosen = (q["answer"] + 1) % len(q["options"])
                            result = agent.submit_answer(session, q["id"], chosen)
                            row["eval_checks"] += 1
                            row["eval_correct"] += int(
                                result["is_correct"] == (chosen == q["answer"]))
                        agent.finish_quiz(session)
                        row["completed"] = True
                    except Exception as e:
                        row["error"] = repr(e)
                    row["finish_ms"] = (time.perf_counter() - t1) * 1000
                    rows.append(row)
    return pd.DataFrame(rows)


# ---------- 2. tool selection ----------
def run_tool_selection(cases):
    rows = []
    for c in cases:
        fresh_db()
        topic = c["topic"]
        if topic == "NOTES":
            notes.add_note("Eval notes", SAMPLE_NOTE)
            topic = notes_quiz.topic_name("Eval notes")
        else:
            mark_seen(topic, c["unseen"])
        try:
            actual = planner.plan_quiz(topic, None, c["n"],
                                       llm_available=c["generator"])["tool"]
        except Exception as e:
            actual = "error: " + type(e).__name__
        rows.append({"kind": "tool", "id": c["id"], "description": c["description"],
                     "expected": c["expected"], "actual": actual,
                     "correct": actual == c["expected"]})
    return pd.DataFrame(rows)


# ---------- 3. difficulty and topic decisions ----------
def run_decisions(cases):
    rows = []
    for c in cases["difficulty"]:
        actual = planner.next_difficulty(c["last"], c["scores"])[0]
        rows.append({"kind": "difficulty", "id": c["id"], "description": c["description"],
                     "expected": c["expected"], "actual": actual,
                     "correct": actual == c["expected"]})
    for c in cases["topic"]:
        stats = pd.DataFrame(c["stats"], columns=["topic", "answered", "accuracy"])
        stats = stats.sort_values("accuracy")
        actual = planner.choose_topic(c["topics"], stats, c["requested"])[0]
        rows.append({"kind": "topic", "id": c["id"], "description": c["description"],
                     "expected": c["expected"], "actual": actual,
                     "correct": actual == c["expected"]})
    return pd.DataFrame(rows)


# ---------- 4. fault injection ----------
def _start(topic="Statistics", difficulty="easy", n=3, generator=None, seen=None):
    fresh_db()
    if seen is not None:
        mark_seen(topic, seen)
    return QuizAgent(generator=generator).start_quiz(topic, difficulty, n)


def _with_bank(path, fn):
    old = tools.BANK_PATH
    tools.BANK_PATH = path
    try:
        return fn()
    finally:
        tools.BANK_PATH = old


def _write_bad_bank():
    path = Path(_TMP.name) / "bad_bank.json"
    path.write_text("this is { not json", encoding="utf-8")
    return path


def _session():
    fresh_db()
    agent = QuizAgent()
    return agent, agent.start_quiz("Statistics", "easy", 3)


def _answer_unknown():
    agent, s = _session()
    agent.submit_answer(s, "nope", 0)


def _answer_out_of_range():
    agent, s = _session()
    agent.submit_answer(s, s["questions"][0]["id"], 999)


def _finish_early():
    agent, s = _session()
    agent.finish_quiz(s)


def _answer_twice():
    agent, s = _session()
    qid = s["questions"][0]["id"]
    agent.submit_answer(s, qid, 0)
    agent.submit_answer(s, qid, 0)


def _boom(topic, difficulty, n):
    raise RuntimeError("generator is down")


def _junk(topic, difficulty, n):
    return [{"id": "junk"}]


def _bad_quality(topic, difficulty, n):
    questions = generators.generate_questions(topic, difficulty, n)
    for q in questions:
        q["hint"] = ""
    return questions


ERROR_FAULTS = [
    ("R01", "Unknown topic", lambda: _start(topic="Cooking")),
    ("R02", "Invalid difficulty", lambda: _start(difficulty="extreme")),
    ("R03", "Zero questions requested", lambda: _start(n=0)),
    ("R04", "Too many questions requested", lambda: _start(n=99)),
    ("R05", "Question bank file missing",
     lambda: _with_bank(Path(_TMP.name) / "missing.json", _start)),
    ("R06", "Question bank is not valid JSON",
     lambda: _with_bank(_write_bad_bank(), _start)),
    ("R07", "Notes too short",
     lambda: (fresh_db(), notes.add_note("Short", "Too short."))),
    ("R08", "Unsupported file type", lambda: notes.decode_upload("virus.exe", b"x")),
    ("R09", "Damaged PDF upload", lambda: notes.decode_upload("broken.pdf", b"not a pdf")),
    ("R10", "Notes without usable sentences",
     lambda: (fresh_db(), notes.add_note("Junk", "Ok. Yes. No way. " * 20),
              QuizAgent().start_quiz("My notes: Junk", "easy", 3))),
    ("R11", "Answer for an unknown question", _answer_unknown),
    ("R12", "Answer option out of range", _answer_out_of_range),
    ("R13", "Finish quiz before answering", _finish_early),
    ("R14", "Same question answered twice", _answer_twice),
]
FALLBACK_FAULTS = [
    ("F01", "Generator crashes", lambda: _start(generator=_boom, seen=0)),
    ("F02", "Generator returns junk", lambda: _start(generator=_junk, seen=0)),
    ("F03", "Generator returns low-quality questions",
     lambda: _start(generator=_bad_quality, seen=0)),
]


def handled_error(fn):
    try:
        fn()
    except tools.ToolError:
        return True
    except Exception:
        return False
    return False


def handled_fallback(fn):
    try:
        s = fn()
    except Exception:
        return False
    return (s["tool_used"] == "question_bank" and bool(s["notices"])
            and len(s["questions"]) > 0)


def run_faults():
    rows = []
    for fid, desc, fn in ERROR_FAULTS:
        rows.append({"id": fid, "description": desc,
                     "expected": "clear error message",
                     "handled": handled_error(fn)})
    for fid, desc, fn in FALLBACK_FAULTS:
        rows.append({"id": fid, "description": desc,
                     "expected": "fallback to the question bank with a notice",
                     "handled": handled_fallback(fn)})
    return pd.DataFrame(rows)


# ---------- 5. automatic question quality ----------
def run_template_quality(calls=30, n=10):
    rows = []
    for topic in generators.TEMPLATES:
        for level in tools.DIFFICULTIES:
            total = passed = 0
            ids = set()
            for _ in range(calls):
                for q in generators.generate_questions(topic, level, n):
                    total += 1
                    passed += int(guardrails.check_question(q)[0])
                    ids.add(q["id"])
            rows.append({"topic": topic, "difficulty": level, "questions": total,
                         "passed_checks": passed, "unique": len(ids)})
    return pd.DataFrame(rows)


def load_sample_notes():
    found = [{"id": 1, "title": "Built-in sample", "content": SAMPLE_NOTE}]
    folder = Path(os.environ.get("EVAL_NOTES_DIR", HERE / "sample_notes"))
    folder.mkdir(exist_ok=True)
    for i, path in enumerate(sorted(folder.iterdir()), start=2):
        if path.suffix.lower() in notes.ALLOWED_FILES:
            try:
                text = notes.decode_upload(path.name, path.read_bytes())
            except tools.ToolError as e:
                print(f"  Skipped {path.name}: {e}")
                continue
            text = notes.clean_text(text)[: notes.MAX_CHARS]
            found.append({"id": i, "title": path.stem, "content": text})
    return found


def run_notes_quality(note_list, calls=10, n=10):
    rows = []
    for note in note_list:
        for level in tools.DIFFICULTIES:
            total = passed = matched = failures = 0
            ids = set()
            for _ in range(calls):
                try:
                    questions = notes_quiz.generate_from_note(note, level, n)
                except tools.ToolError:
                    failures += 1
                    continue
                for q in questions:
                    total += 1
                    passed += int(guardrails.check_question(q)[0])
                    matched += int(q["difficulty"] == level)
                    ids.add(q["id"])
            rows.append({"note": note["title"], "difficulty": level, "calls": calls,
                         "failed_calls": failures, "questions": total,
                         "passed_checks": passed, "level_matched": matched,
                         "unique": len(ids)})
    return pd.DataFrame(rows)


def export_rating_sheet(note_list, how_many=30):
    """A sheet for YOU to rate notes questions from 1 to 5 (kept if it exists)."""
    path = RESULTS / "notes_questions_to_rate.csv"
    if path.exists():
        return path
    pool, seen = [], set()
    for note in note_list:
        for level in tools.DIFFICULTIES:
            try:
                questions = notes_quiz.generate_from_note(note, level, 10)
            except tools.ToolError:
                continue
            for q in questions:
                if q["question"] not in seen:
                    seen.add(q["question"])
                    pool.append((note["title"], q))
    random.shuffle(pool)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["note", "difficulty", "question", "options",
                         "correct_answer", "source_sentence", "rating_1_to_5"])
        for title, q in pool[:how_many]:
            writer.writerow([title, q["difficulty"], q["question"],
                             " | ".join(q["options"]), q["options"][q["answer"]],
                             q["source"], ""])
    return path


def read_ratings(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    ratings = pd.to_numeric(df["rating_1_to_5"], errors="coerce").dropna()
    return ratings[(ratings >= 1) & (ratings <= 5)]


# ---------- metrics and charts ----------
def ratio(a, b):
    return pct(a / b) if b else None


def build_metrics(gen, sel, dec, faults, tq, nq, rating_path):
    start = gen["start_ms"].dropna()
    finish = gen["finish_ms"].dropna()
    metrics = {
        "seed": SEED,
        "quiz_runs": int(len(gen)),
        "rates": {
            "Quiz generated as requested": pct(gen["generated"].mean()),
            "Quiz completed end to end": pct(gen["completed"].mean()),
            "Answers marked correctly":
                ratio(gen["eval_correct"].sum(), gen["eval_checks"].sum()),
            "Tool chosen correctly": pct(sel["correct"].mean()),
            "Difficulty decided correctly":
                pct(dec[dec["kind"] == "difficulty"]["correct"].mean()),
            "Topic decided correctly":
                pct(dec[dec["kind"] == "topic"]["correct"].mean()),
            "Faults handled as designed": pct(faults["handled"].mean()),
            "Generated questions passing checks":
                ratio(tq["passed_checks"].sum(), tq["questions"].sum()),
            "Notes questions passing checks":
                ratio(nq["passed_checks"].sum(), nq["questions"].sum()),
        },
        "latency_ms": {
            "start_quiz_median": round(float(start.median()), 1) if len(start) else None,
            "start_quiz_p95": round(float(start.quantile(0.95)), 1) if len(start) else None,
            "answer_and_finish_median": round(float(finish.median()), 1) if len(finish) else None,
            "answer_and_finish_p95": round(float(finish.quantile(0.95)), 1) if len(finish) else None,
        },
        "tool_usage": {k: int(v) for k, v in gen["tool"].value_counts().items()},
        "generated_unique_share_percent":
            ratio(tq["unique"].sum(), tq["questions"].sum()),
        "notes_level_match_percent":
            ratio(nq["level_matched"].sum(), nq["questions"].sum()),
        "notes_failed_calls": int(nq["failed_calls"].sum()),
        "notes_calls": int(nq["calls"].sum()),
    }
    if rating_path.exists():
        ratings = read_ratings(rating_path)
        if len(ratings):
            metrics["notes_human_rating"] = {
                "rated_questions": int(len(ratings)),
                "mean_1_to_5": round(float(ratings.mean()), 2),
            }
    return metrics


def make_charts(metrics, gen, tq):
    charts = RESULTS / "charts"
    charts.mkdir(exist_ok=True)

    rates = {k: v for k, v in metrics["rates"].items() if v is not None}
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(list(rates), list(rates.values()), color="#4F46E5")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percent")
    ax.set_title("Key success rates")
    ax.invert_yaxis()
    for i, v in enumerate(rates.values()):
        ax.text(min(v + 1, 90), i, f"{v:.1f}%", va="center")
    fig.tight_layout()
    fig.savefig(charts / "success_rates.png", dpi=150)
    plt.close(fig)

    data = gen.dropna(subset=["start_ms"])
    groups = [(name, g["start_ms"]) for name, g in data.groupby("tool")]
    if groups:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot([g for _, g in groups])
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels([name for name, _ in groups])
        ax.set_ylabel("Milliseconds")
        ax.set_title("Time to prepare a quiz, by tool")
        fig.tight_layout()
        fig.savefig(charts / "latency.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    gen["tool"].value_counts().plot.bar(ax=ax, color="#059669")
    ax.set_ylabel("Number of quizzes")
    ax.set_title("Tools used in the evaluation runs")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(charts / "tool_usage.png", dpi=150)
    plt.close(fig)

    by_topic = tq.groupby("topic")[["unique", "questions"]].sum()
    share = (by_topic["unique"] / by_topic["questions"] * 100)
    fig, ax = plt.subplots(figsize=(7, 4))
    share.plot.bar(ax=ax, color="#7C3AED")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Unique questions (%)")
    ax.set_title("Variety of generated questions (900 per topic)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(charts / "question_variety.png", dpi=150)
    plt.close(fig)


def main():
    random.seed(SEED)
    RESULTS.mkdir(exist_ok=True)
    cases = json.loads((HERE / "test_cases.json").read_text(encoding="utf-8-sig"))

    print("1/5 Running 135 quizzes (this takes a minute)...")
    gen = run_generation_experiment()
    print("2/5 Checking tool selection and decisions...")
    sel = run_tool_selection(cases["tool_selection"])
    dec = run_decisions(cases)
    print("3/5 Injecting faults...")
    faults = run_faults()
    print("4/5 Measuring question quality...")
    tq = run_template_quality()
    note_list = load_sample_notes()
    nq = run_notes_quality(note_list)
    rating_path = export_rating_sheet(note_list)
    print("5/5 Saving results and charts...")

    gen.to_csv(RESULTS / "quiz_runs.csv", index=False)
    pd.concat([sel, dec]).to_csv(RESULTS / "decision_tests.csv", index=False)
    faults.to_csv(RESULTS / "fault_tests.csv", index=False)
    tq.to_csv(RESULTS / "template_quality.csv", index=False)
    nq.to_csv(RESULTS / "notes_quality.csv", index=False)
    metrics = build_metrics(gen, sel, dec, faults, tq, nq, rating_path)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    make_charts(metrics, gen, tq)
    print("Running the harder tests (a few more minutes)...")
    hard_tests.run_hard_tests(RESULTS, fresh_db)

    print("\nRESULTS (also saved in evaluation/results/metrics.json)")
    print(json.dumps(metrics, indent=2))
    wrong = pd.concat([sel, dec])
    wrong = wrong[~wrong["correct"]]
    if len(wrong):
        print("\nDecision cases that FAILED:")
        print(wrong[["id", "description", "expected", "actual"]].to_string(index=False))
    missed = faults[~faults["handled"]]
    if len(missed):
        print("\nFaults NOT handled as designed:")
        print(missed[["id", "description"]].to_string(index=False))
    errors = gen[gen["error"] != ""]
    if len(errors):
        print("\nExample quiz-run errors:")
        print(errors[["topic", "difficulty", "n", "error"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
