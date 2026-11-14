"""Harder evaluation tests that are designed so the agent CAN fail.

Called from evaluation/run_eval.py. Everything runs on temporary databases.
"""
import ast
import json
import math
import random
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from agent import generators, guardrails, notes_quiz, tools
from agent.agent import QuizAgent

LEVEL_VALUE = {"easy": 0.2, "medium": 0.5, "hard": 0.8}
ERRORS = {"baseline": 0, "simulation": 0}

CLEAN = """Overfitting is when a model memorises the training data and fails on new data.
Underfitting is when a model is too simple to learn the pattern in the data.
Regularisation is a method that adds a penalty to reduce overfitting in a model.
Cross-validation is a method that tests a model on several different splits of the data.
A good model should learn the pattern in the training data and also work on new data.
The test data is kept apart so that the model is judged on data it has never seen.
Overfitting often happens when the model is very complex and the training data is small."""


def pct(a, b):
    return round(100 * a / b, 1) if b else None


# ---------- 1. independent answer check ----------
def _nums(text):
    return ast.literal_eval(re.search(r"(\[.*?\])", text).group(1))


def _check_py_arith(t):
    a, op, b = re.search(r"value of (\d+) (//|%) (\d+)", t).groups()
    return str(int(a) // int(b) if op == "//" else int(a) % int(b))


def _check_py_slice(t):
    m = re.search(r"lst = (\[.*?\])\. What does lst\[(\d+):(\d+)\]", t)
    lst = ast.literal_eval(m.group(1))
    return str(lst[int(m.group(2)):int(m.group(3))])


def _check_py_floor(t):
    a, b = re.search(r"\(-(\d+)\) // (\d+)", t).groups()
    return str((-int(a)) // int(b))


def _check_mean(t):
    nums = _nums(t)
    return generators._fmt(sum(nums) / len(nums))


def _check_median(t):
    return generators._fmt(statistics.median(_nums(t)))


def _check_zscore(t):
    mu, sigma, x = map(int, re.search(
        r"mean of (\d+) and a standard deviation of (\d+)\. "
        r"What is the z-score of a mark of (\d+)", t).groups())
    return generators._fmt((x - mu) / sigma)


def _check_bayes(t):
    p, s, f = map(int, re.search(
        r"(\d+)% of people have a disease\. A test detects (\d+)% of people "
        r"who have it, and wrongly flags (\d+)% of healthy", t).groups())
    p, s, f = p / 100, s / 100, f / 100
    return f"About {round(100 * s * p / (s * p + f * (1 - p)))}%"


def _check_se(t):
    sigma, n = map(int, re.search(
        r"standard deviation of (\d+)\. A random sample of n = (\d+)", t).groups())
    return generators._fmt(sigma / math.sqrt(n))


def _check_accuracy(t):
    n, c = map(int, re.search(r"made (\d+) predictions and (\d+) were", t).groups())
    return generators._fmt(100 * c / n) + "%"


def _check_split(t):
    n, pct_test = map(int, re.search(
        r"has (\d+) rows and is split \d+% for training and (\d+)% for testing",
        t).groups())
    return str(n * pct_test // 100)


def _check_prf(t):
    tp, fp, fn, tn, metric = re.search(
        r"TP = (\d+), FP = (\d+), FN = (\d+), TN = (\d+)\. What is the (\w+)\?",
        t).groups()
    tp, fp, fn, tn = map(int, (tp, fp, fn, tn))
    value = {"precision": tp / (tp + fp), "recall": tp / (tp + fn),
             "accuracy": (tp + tn) / (tp + fp + fn + tn)}[metric]
    return generators._fmt(value)


def _check_f1(t):
    tp, fp, fn = map(int, re.search(
        r"TP = (\d+), FP = (\d+) and FN = (\d+)", t).groups())
    return generators._fmt(2 * tp / (2 * tp + fp + fn))


CHECKERS = [
    ("Python: // and %", generators._py_arith, _check_py_arith),
    ("Python: list slice", generators._py_slice, _check_py_slice),
    ("Python: floor division of a negative", generators._py_floor_neg, _check_py_floor),
    ("Statistics: mean", generators._st_mean, _check_mean),
    ("Statistics: median", generators._st_median, _check_median),
    ("Statistics: z-score", generators._st_zscore, _check_zscore),
    ("Statistics: Bayes", generators._st_bayes, _check_bayes),
    ("Statistics: standard error", generators._st_se, _check_se),
    ("ML: accuracy", generators._ml_accuracy, _check_accuracy),
    ("ML: train/test split", generators._ml_split, _check_split),
    ("ML: precision, recall, accuracy", generators._ml_prf, _check_prf),
    ("ML: F1-score", generators._ml_f1, _check_f1),
]


def run_independent_checks(per_template=100):
    rows = []
    for name, template, checker in CHECKERS:
        checked = matched = 0
        example = ""
        for _ in range(per_template):
            q = generators._build(template, "easy")
            if q is None:
                continue
            checked += 1
            try:
                expected = checker(q["question"])
            except Exception:
                expected = None
            if expected == q["options"][q["answer"]]:
                matched += 1
            elif not example:
                example = q["question"]
        rows.append({"template": name, "checked": checked, "matched": matched,
                     "mismatched": checked - matched, "example_mismatch": example})
    return pd.DataFrame(rows)


# ---------- 2. agent compared with a plain random quiz ----------
def run_baseline(fresh_db, trials=20, quizzes=10, n=5, topic="Statistics"):
    bank = [q["id"] for q in tools.load_bank() if q["topic"] == topic]
    a_total = a_repeat = a_generated = b_total = b_repeat = 0
    for _ in range(trials):
        fresh_db()
        agent = QuizAgent(generator=generators.generate_questions)
        seen = set()
        for _quiz in range(quizzes):
            try:
                session = agent.start_quiz(topic, None, n)
                for q in session["questions"]:
                    a_total += 1
                    a_repeat += int(q["id"] in seen)
                    a_generated += int(q["id"].startswith("gen-"))
                    seen.add(q["id"])
                    ok = random.random() < 0.6
                    chosen = q["answer"] if ok else (q["answer"] + 1) % len(q["options"])
                    agent.submit_answer(session, q["id"], chosen)
                agent.finish_quiz(session)
            except Exception:
                ERRORS["baseline"] += 1
        seen = set()
        for _quiz in range(quizzes):
            for qid in random.sample(bank, min(n, len(bank))):
                b_total += 1
                b_repeat += int(qid in seen)
                seen.add(qid)
    return {
        "students": trials, "quizzes_each": quizzes, "questions_per_quiz": n,
        "bank_size_for_topic": len(bank),
        "agent_repeat_percent": pct(a_repeat, a_total),
        "plain_random_repeat_percent": pct(b_repeat, b_total),
        "agent_generated_share_percent": pct(a_generated, a_total),
        "errors": ERRORS["baseline"],
    }


# ---------- 3. simulated students ----------
def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def _p_correct(skill, level):
    """Chance of a right answer: 25% guessing floor, rising with skill."""
    return 0.25 + 0.7 * _sigmoid(6 * (skill - LEVEL_VALUE[level]))


def run_simulation(fresh_db, students=20, quizzes=12, n=5, topic="Statistics"):
    policies = {"Adaptive agent": None, "Always easy": "easy",
                "Always hard": "hard", "Random level": "random"}
    starts = [0.2 + 0.4 * i / (students - 1) for i in range(students)]
    rng = random.Random(7)
    rows = []
    for policy, fixed in policies.items():
        for sid, skill in enumerate(starts):
            fresh_db()
            agent = QuizAgent(generator=generators.generate_questions)
            for number in range(1, quizzes + 1):
                level = rng.choice(tools.DIFFICULTIES) if fixed == "random" else fixed
                try:
                    session = agent.start_quiz(topic, level, n)
                    correct, match, levels = 0, 0.0, []
                    for q in session["questions"]:
                        d = q["difficulty"]
                        levels.append(LEVEL_VALUE[d])
                        ok = rng.random() < _p_correct(skill, d)
                        chosen = q["answer"] if ok else (q["answer"] + 1) % len(q["options"])
                        agent.submit_answer(session, q["id"], chosen)
                        correct += int(ok)
                        match += max(0.0, 1 - 2 * abs(skill - LEVEL_VALUE[d]))
                    agent.finish_quiz(session)
                except Exception:
                    ERRORS["simulation"] += 1
                    continue
                count = len(session["questions"])
                skill = min(1.0, skill + 0.05 * match / count)
                rows.append({"policy": policy, "student": sid, "quiz": number,
                             "skill": skill, "score": 100 * correct / count,
                             "difficulty": sum(levels) / count})
    return pd.DataFrame(rows)


# ---------- 4. messy notes ----------
def _messy_cases():
    sentence = "Overfitting is when a model memorises the training data and fails on new data."
    return [
        ("Clean definitions (control)", CLEAN),
        ("Bullet fragments only", "\n".join(
            f"- {w}" for w in ["gradient descent", "learning rate", "loss function",
                              "batch size", "activation function", "weights and bias"] * 4)),
        ("Slide style 'Term: meaning'", "\n".join([
            "Overfitting: model memorises the training data and fails on new data",
            "Underfitting: model is too simple to learn the pattern in the data",
            "Regularisation: adds a penalty to reduce overfitting in a model",
            "Cross-validation: tests a model on several different splits of the data",
            "Test data: kept apart so the model is judged on unseen data",
            "Training data: the examples the model learns its pattern from"])),
        ("Mostly code", "\n".join([
            "for i in range(10):", "    print(i)", "def add(a, b):",
            "    return a + b", "x = [1, 2, 3]", "y = x[1:3]", "print(len(x))",
            "A list keeps the order of its items in Python.",
            "A dictionary stores values under keys in Python."])),
        ("One sentence repeated 20 times", "\n".join([sentence] * 20)),
        ("Numbers and tables", "\n".join(["Model A 0.91 0.88 0.90"] * 20)),
        ("Long run-on text", " ".join([
            "the model learns patterns from data and then uses those patterns "
            "to make predictions on new data and when the data changes the model "
            "may need more training"] * 3)),
        ("ALL CAPS definitions", CLEAN.upper()),
    ]


def run_messy_notes():
    rows = []
    for i, (name, text) in enumerate(_messy_cases(), start=1):
        note = {"id": 900 + i, "title": name, "content": text}
        try:
            questions = notes_quiz.generate_from_note(note, "easy", 5)
            message = ""
        except tools.ToolError as e:
            questions, message = [], str(e)[:90]
        passed = all(guardrails.check_question(q)[0] for q in questions)
        rows.append({"case": name, "questions": len(questions),
                     "all_pass_checks": passed,
                     "usable": len(questions) >= 3 and passed,
                     "message": message})
    return pd.DataFrame(rows)


# ---------- 5. stress test ----------
def run_stress(fresh_db, rounds=400, seed=11):
    rng = random.Random(seed)
    topics = tools.list_topics() + ["Cooking", "", None, 5, "My notes: Ghost"]
    levels = list(tools.DIFFICULTIES) + ["extreme", None, 3, ""]
    sizes = [None, 1, 3, 5, 10, 0, -2, 11, 99, 3.5, "5", True]
    choices = [0, 1, 2, 3, 4, -1, 99, None, "a", 1.5]
    fresh_db()
    agent = QuizAgent(generator=generators.generate_questions)
    session = None
    ok = handled = 0
    crashes = []
    for step in range(rounds):
        action = rng.choice(["start", "start", "answer", "answer", "hint", "finish"])
        try:
            if action == "start":
                session = agent.start_quiz(
                    rng.choice(topics), rng.choice(levels), rng.choice(sizes))
            elif session is None:
                continue
            elif action == "answer":
                ids = [q["id"] for q in session["questions"]] + ["nope", None]
                agent.submit_answer(session, rng.choice(ids), rng.choice(choices))
            elif action == "hint":
                ids = [q["id"] for q in session["questions"]] + ["nope"]
                agent.get_hint(session, rng.choice(ids))
            else:
                agent.finish_quiz(session)
            ok += 1
        except tools.ToolError:
            handled += 1
        except Exception as e:
            crashes.append({"step": step, "action": action,
                            "error": f"{type(e).__name__}: {e}"[:200]})
    return {"rounds": rounds, "successful_actions": ok,
            "handled_errors": handled, "unexpected_crashes": len(crashes),
            "crashes": crashes}


# ---------- charts and saving ----------
def make_charts(results_dir, base, sim, messy):
    charts = results_dir / "charts"
    charts.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Agent (with memory)", "Plain random quiz"],
           [base["agent_repeat_percent"], base["plain_random_repeat_percent"]],
           color=["#4F46E5", "#9CA3AF"])
    ax.set_ylabel("Repeated questions (%)")
    ax.set_title("Repeats over 10 quizzes")
    fig.tight_layout()
    fig.savefig(charts / "baseline_repeats.png", dpi=150)
    plt.close(fig)

    for column, title, name in [("skill", "Simulated skill after each quiz", "simulation_skill"),
                                ("score", "Simulated average score (%)", "simulation_score")]:
        fig, ax = plt.subplots(figsize=(7, 4))
        sim.groupby(["policy", "quiz"])[column].mean().unstack(0).plot(ax=ax)
        ax.set_xlabel("Quiz number")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(charts / f"{name}.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#059669" if u else "#DC2626" for u in messy["usable"]]
    ax.barh(messy["case"], messy["questions"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Questions made (green = usable quiz, red = not usable)")
    ax.set_title("Quiz from messy notes")
    fig.tight_layout()
    fig.savefig(charts / "messy_notes.png", dpi=150)
    plt.close(fig)


def run_hard_tests(results_dir, fresh_db):
    results_dir = Path(results_dir)
    print("  independent answer checks...")
    ind = run_independent_checks()
    print("  agent compared with a plain random quiz...")
    base = run_baseline(fresh_db)
    print("  simulated students (about a minute)...")
    sim = run_simulation(fresh_db)
    print("  messy notes...")
    messy = run_messy_notes()
    print("  stress test...")
    stress = run_stress(fresh_db)

    ind.to_csv(results_dir / "independent_checks.csv", index=False)
    sim.to_csv(results_dir / "simulation_runs.csv", index=False)
    messy.to_csv(results_dir / "messy_notes.csv", index=False)
    pd.DataFrame(stress["crashes"], columns=["step", "action", "error"]).to_csv(
        results_dir / "stress_crashes.csv", index=False)

    last = sim[sim["quiz"] == sim["quiz"].max()]
    summary = {
        "independent_check": {
            "templates": int(len(ind)),
            "questions_checked": int(ind["checked"].sum()),
            "mismatches": int(ind["mismatched"].sum()),
            "match_rate": pct(ind["matched"].sum(), ind["checked"].sum()),
        },
        "baseline": base,
        "simulation": {
            "students_per_strategy": int(sim["student"].nunique()),
            "quizzes_each": int(sim["quiz"].max()),
            "start_skill_mean": 0.4,
            "final_skill": last.groupby("policy")["skill"].mean().round(3).to_dict(),
            "mean_score": sim.groupby("policy")["score"].mean().round(1).to_dict(),
            "errors": ERRORS["simulation"],
            "assumption": "Virtual students learn fastest when the question level "
                          "matches their skill. This is an assumption, so the result "
                          "shows the agent tracks a student's level. It does not "
                          "prove that real students learn more.",
        },
        "messy_notes": {
            "cases": int(len(messy)),
            "usable": int(messy["usable"].sum()),
            "usable_percent": pct(messy["usable"].sum(), len(messy)),
        },
        "stress_test": {
            "rounds": stress["rounds"],
            "successful_actions": stress["successful_actions"],
            "handled_errors": stress["handled_errors"],
            "unexpected_crashes": stress["unexpected_crashes"],
            "crash_examples": stress["crashes"][:3],
        },
    }
    (results_dir / "hard_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    make_charts(results_dir, base, sim, messy)
    return summary
