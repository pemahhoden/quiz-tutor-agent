"""Question generator: makes new questions from templates with random numbers.

The correct answer is calculated by Python, so it is always right.
No LLM and no API key are needed. The agent uses this tool when the
question bank has too few new questions for a topic.
"""
import hashlib
import math
import random
import statistics

from agent import tools

PY, ST, ML = "Python basics", "Statistics", "Machine learning basics"


def _fmt(x):
    """Show whole numbers without .0, other numbers with up to 2 decimals."""
    x = round(float(x), 2)
    return str(int(x)) if x == int(x) else f"{x:g}"


def _make(topic, difficulty, text, correct, wrong, hint, explanation):
    """Build one question. Needs 3 different wrong answers, else returns None."""
    correct = str(correct)
    distractors = list(dict.fromkeys(str(w) for w in wrong if str(w) != correct))[:3]
    if len(distractors) < 3:
        return None
    options = distractors + [correct]
    random.shuffle(options)
    return {
        "id": "gen-" + hashlib.md5(text.encode("utf-8")).hexdigest()[:10],
        "topic": topic, "difficulty": difficulty, "question": text,
        "options": options, "answer": options.index(correct),
        "hint": hint, "explanation": explanation,
    }


# ---------------- Python basics ----------------
def _py_arith(d):
    a, b = random.randint(11, 60), random.randint(3, 9)
    if random.random() < 0.5:
        op, ans = "//", a // b
        wrong = [ans + 1, ans - 1, a % b, ans + 2, round(a / b)]
        hint = "// divides and drops everything after the decimal point."
        why = f"{a} // {b} is the whole-number part of {a} / {b}, which is {ans}."
    else:
        op, ans = "%", a % b
        wrong = [ans + 1, ans + 2, a // b, ans + 3, b]
        hint = "% gives the remainder left over after dividing."
        why = f"{a} divided by {b} leaves a remainder of {ans}."
    return _make(PY, d, f"What is the value of {a} {op} {b} in Python?",
                 ans, wrong, hint, why)


def _py_slice(d):
    lst = random.sample(range(1, 30), 6)
    i = random.randint(1, 2)
    j = random.randint(i + 2, 5)
    ans = lst[i:j]
    wrong = [lst[i:j + 1], lst[i - 1:j], lst[i + 1:j], lst[i:j - 1]]
    return _make(PY, d, f"lst = {lst}. What does lst[{i}:{j}] return?", ans, wrong,
                 "The start index is included and the end index is not.",
                 f"Positions start at 0. lst[{i}:{j}] takes positions {i} to "
                 f"{j - 1}, which gives {ans}.")


def _py_floor_neg(d):
    b = random.randint(2, 5)
    a = random.randint(7, 40)
    while a % b == 0:
        a += 1
    ans = (-a) // b
    wrong = [-(a // b), a // b, ans - 1, ans + 2, a // b + 1]
    return _make(PY, d, f"What does (-{a}) // {b} return in Python?", ans, wrong,
                 "Floor division rounds down, toward the smaller number.",
                 f"-{a} / {b} is about {-a / b:.2f}. Rounding down (not toward "
                 f"zero) gives {ans}.")


# ---------------- Statistics ----------------
def _st_mean(d):
    k = random.choice([4, 5])
    nums = [random.randint(2, 20) for _ in range(k)]
    mean = sum(nums) / k
    wrong = [statistics.median(nums), mean + 1, mean - 1, sum(nums), max(nums), mean + 2]
    return _make(ST, d, f"What is the mean of {nums}?", _fmt(mean),
                 [_fmt(w) for w in wrong],
                 "Add all the numbers, then divide by how many there are.",
                 f"The sum is {sum(nums)} and there are {k} numbers, so the mean is "
                 f"{sum(nums)} / {k} = {_fmt(mean)}.")


def _st_median(d):
    nums = random.sample(range(1, 50), 5)
    ordered = sorted(nums)
    ans = ordered[2]
    wrong = [nums[2], sum(nums) / 5, max(nums), min(nums), ordered[1]]
    return _make(ST, d, f"What is the median of {nums}?", _fmt(ans),
                 [_fmt(w) for w in wrong],
                 "Sort the numbers first, then take the middle one.",
                 f"In order the numbers are {ordered}. The middle value is {ans}.")


def _st_zscore(d):
    mu = random.choice([50, 60, 70, 80])
    sigma = random.choice([5, 10])
    z = random.choice([-2, -1, 1, 2])
    x = mu + z * sigma
    wrong = [_fmt(w) for w in (-z, 2 * z, z / 2, 0, z + 1, z - 1)]
    return _make(ST, d,
                 f"Exam marks have a mean of {mu} and a standard deviation of {sigma}. "
                 f"What is the z-score of a mark of {x}?", _fmt(z), wrong,
                 "z = (value - mean) / standard deviation.",
                 f"z = ({x} - {mu}) / {sigma} = {z}.")


def _st_bayes(d):
    p = random.choice([0.01, 0.02, 0.05])
    s = random.choice([0.9, 0.95])
    f = random.choice([0.05, 0.1])
    post = s * p / (s * p + f * (1 - p))
    ans = round(post * 100)
    cands = [round(s * 100), 50, ans + 20, ans + 10, max(ans - 5, 1)]
    sick = round(10000 * p)
    tp = round(sick * s)
    fp = round((10000 - sick) * f)
    return _make(ST, d,
                 f"{round(p * 100)}% of people have a disease. A test detects "
                 f"{round(s * 100)}% of people who have it, and wrongly flags "
                 f"{round(f * 100)}% of healthy people. A person tests positive. "
                 "About how likely is it that they truly have the disease?",
                 f"About {ans}%", [f"About {c}%" for c in cands],
                 "Imagine 10,000 people and count the real and false positives.",
                 f"Of 10,000 people, {sick} are sick and {tp} of them test positive. "
                 f"Of the {10000 - sick} healthy people, {fp} also test positive. "
                 f"So {tp} / ({tp} + {fp}) is about {ans}%.")


def _st_se(d):
    sigma = random.choice([10, 20, 30, 40])
    n = random.choice([4, 9, 16, 25, 36, 100])
    ans = sigma / math.sqrt(n)
    wrong = [sigma / n, sigma * math.sqrt(n), sigma, ans * 2, ans / 2]
    return _make(ST, d,
                 f"A population has a standard deviation of {sigma}. A random sample "
                 f"of n = {n} is taken. What is the standard error of the sample mean?",
                 _fmt(ans), [_fmt(w) for w in wrong],
                 "Divide the standard deviation by the square root of n.",
                 f"Standard error = {sigma} / sqrt({n}) = {_fmt(ans)}.")


# ---------------- Machine learning basics ----------------
def _ml_accuracy(d):
    n = random.choice([20, 40, 50, 80, 100, 200])
    c = random.randint(n // 2, n - 2)
    acc = 100 * c / n
    wrong = [f"{_fmt(x)}%" for x in (100 - acc, acc + 5, acc - 5, acc + 10, acc - 10)
             if 0 <= x <= 100]
    return _make(ML, d, f"A model made {n} predictions and {c} were correct. "
                 "What is its accuracy?", f"{_fmt(acc)}%", wrong,
                 "Accuracy = correct predictions / all predictions.",
                 f"{c} / {n} = {_fmt(acc / 100)}, which is {_fmt(acc)}%.")


def _ml_split(d):
    n = random.choice([100, 200, 400, 500, 1000])
    t = random.choice([20, 25, 30])
    test = n * t // 100
    wrong = [n - test, test * 2, n * (t + 10) // 100, n * (t - 10) // 100, n // 10]
    return _make(ML, d,
                 f"A data set has {n} rows and is split {100 - t}% for training and "
                 f"{t}% for testing. How many rows are in the test set?", test, wrong,
                 "Take the testing percentage of the total number of rows.",
                 f"{t}% of {n} is {test} rows.")


def _ml_prf(d):
    tp, fp, fn, tn = (random.randint(10, 60) for _ in range(4))
    formulas = {
        "precision": ("TP / (TP + FP)", tp / (tp + fp)),
        "recall": ("TP / (TP + FN)", tp / (tp + fn)),
        "accuracy": ("(TP + TN) / (TP + FP + FN + TN)", (tp + tn) / (tp + fp + fn + tn)),
        "specificity": ("TN / (TN + FP)", tn / (tn + fp)),
    }
    metric = random.choice(["precision", "recall", "accuracy"])
    formula, ans = formulas[metric]
    wrong = [_fmt(v) for k, (_, v) in formulas.items() if k != metric]
    return _make(ML, d,
                 f"A model's results are TP = {tp}, FP = {fp}, FN = {fn}, TN = {tn}. "
                 f"What is the {metric}?", _fmt(ans), wrong,
                 f"{metric.capitalize()} = {formula}.",
                 f"{metric.capitalize()} = {formula} = {_fmt(ans)}.")


def _ml_f1(d):
    tp, fp, fn = random.randint(20, 80), random.randint(5, 40), random.randint(5, 40)
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    cands = [precision, recall, (precision + recall) / 2, f1 + 0.1, f1 - 0.1]
    return _make(ML, d, f"A model has TP = {tp}, FP = {fp} and FN = {fn}. "
                 "What is its F1-score?", _fmt(f1),
                 [_fmt(x) for x in cands if 0 <= x <= 1],
                 "F1 = 2TP / (2TP + FP + FN).",
                 f"Precision is {_fmt(precision)} and recall is {_fmt(recall)}. "
                 f"F1 = {2 * tp} / {2 * tp + fp + fn} = {_fmt(f1)}.")


TEMPLATES = {
    PY: {"easy": [_py_arith], "medium": [_py_slice], "hard": [_py_floor_neg]},
    ST: {"easy": [_st_mean, _st_median], "medium": [_st_zscore],
         "hard": [_st_bayes, _st_se]},
    ML: {"easy": [_ml_accuracy], "medium": [_ml_split, _ml_prf], "hard": [_ml_f1]},
}


def _build(template, difficulty):
    """Try a template a few times (some random draws give too few wrong answers)."""
    for _ in range(20):
        q = template(difficulty)
        if q is not None:
            return q
    return None


def generate_questions(topic, difficulty, n):
    """The generator tool: returns up to n new, different, valid questions."""
    if topic not in TEMPLATES:
        raise tools.ToolError(f"No question templates for topic: {topic}")
    if difficulty not in TEMPLATES[topic]:
        raise tools.ToolError(f"Unknown difficulty: {difficulty}")
    questions, texts = [], set()
    for _ in range(n * 30):
        if len(questions) == n:
            break
        q = _build(random.choice(TEMPLATES[topic][difficulty]), difficulty)
        if q and q["question"] not in texts and tools.is_valid_question(q):
            texts.add(q["question"])
            questions.append(q)
    return questions
