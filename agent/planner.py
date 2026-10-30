"""The planner: decision rules of the Quiz Tutor Agent.

These are plain rules (no LLM), so every decision is predictable, testable
and easy to explain. Each decision also returns a short reason, which the
app shows to the student as "Why this quiz?".
"""
from agent import memory, tools

HIGH_SCORE = 80   # last 3 scores all at or above this -> harder questions
LOW_SCORE = 50    # last score below this -> easier questions
WEAK_TOPIC = 70   # topic accuracy below this -> revision quiz
DEFAULT_N = 5
MAX_N = 10


def next_difficulty(last_difficulty, recent_scores):
    """Decide the next difficulty from the recent scores."""
    levels = tools.DIFFICULTIES
    i = levels.index(last_difficulty)
    if len(recent_scores) >= 3 and all(s >= HIGH_SCORE for s in recent_scores[-3:]):
        new = levels[min(i + 1, len(levels) - 1)]
        if new == last_difficulty:
            return new, "Your last 3 scores were high, and you are already at the hardest level."
        return new, f"Your last 3 scores were all {HIGH_SCORE}% or more, so the difficulty goes up."
    if recent_scores and recent_scores[-1] < LOW_SCORE:
        new = levels[max(i - 1, 0)]
        if new == last_difficulty:
            return new, "Your last score was low, and you are already at the easiest level."
        return new, f"Your last score was below {LOW_SCORE}%, so the difficulty goes down."
    return last_difficulty, "Your recent scores are steady, so the difficulty stays the same."


def choose_topic(topics, stats, requested=None):
    """Use the student's choice, or let the agent pick a topic."""
    if requested is not None:
        if requested not in topics:
            raise tools.ToolError(f"Unknown topic: {requested}")
        return requested, f"You chose {requested}."
    if stats.empty:
        return topics[0], "No history yet, so we start with the first topic."
    weakest = stats.iloc[0]
    if weakest["accuracy"] < WEAK_TOPIC:
        return weakest["topic"], (
            f"Your accuracy in {weakest['topic']} is {weakest['accuracy']:.0f}%, "
            "so this is a revision quiz."
        )
    answered = {row.topic: row.answered for row in stats.itertuples()}
    topic = min(topics, key=lambda t: answered.get(t, 0))
    return topic, f"Your accuracy is good everywhere, so we practise {topic}, the topic you answered least."


def choose_tool(topic, n, llm_available):
    """Pick where the questions come from."""
    seen = set(memory.get_seen_question_ids(topic))
    unseen = [q for q in tools.load_bank()
              if q["topic"] == topic and q["id"] not in seen]
    if len(unseen) >= n:
        return "question_bank", "The question bank has enough new questions for this topic."
    if llm_available:
        return "llm_generator", "The bank has too few new questions, so the LLM will write new ones."
    return "question_bank", (
        "The bank has too few new questions and the LLM is not available, "
        "so some questions may repeat."
    )


def plan_quiz(topic=None, difficulty=None, n=None, llm_available=False):
    """Plan the next quiz. Choices given by the student are respected."""
    if difficulty is not None and difficulty not in tools.DIFFICULTIES:
        raise tools.ToolError(f"Unknown difficulty: {difficulty}")
    if n is None:
        n = DEFAULT_N
    if not isinstance(n, int) or not 1 <= n <= MAX_N:
        raise tools.ToolError(f"Number of questions must be between 1 and {MAX_N}.")

    topics = tools.list_topics()
    stats = memory.get_topic_stats()
    history = memory.get_history()
    recent = memory.get_recent_scores(3)
    reasons = []

    topic, why = choose_topic(topics, stats, topic)
    reasons.append(why)

    if difficulty is not None:
        reasons.append(f"You chose {difficulty} difficulty.")
    elif history.empty:
        difficulty = "easy"
        reasons.append("No history yet, so we start with easy questions.")
    else:
        difficulty, why = next_difficulty(history.iloc[-1]["difficulty"], recent)
        reasons.append(why)

    available = len([q for q in tools.load_bank() if q["topic"] == topic])
    if n > available:
        reasons.append(f"Only {available} questions exist for {topic}, so the quiz has {available}.")
        n = available

    tool, why = choose_tool(topic, n, llm_available)
    reasons.append(why)

    if recent and sum(recent) / len(recent) < LOW_SCORE:
        hint_mode = "proactive"
        reasons.append("Your recent average is low, so hints are offered up front.")
    else:
        hint_mode = "on_request"

    return {"topic": topic, "difficulty": difficulty, "n": n,
            "tool": tool, "hint_mode": hint_mode, "reasons": reasons}


def recommend_next(llm_available=False):
    """After a quiz: let the agent plan the next one by itself."""
    plan = plan_quiz(llm_available=llm_available)
    plan["message"] = (
        f"Next: a {plan['difficulty']} {plan['topic']} quiz. "
        + " ".join(plan["reasons"][:2])
    )
    return plan
