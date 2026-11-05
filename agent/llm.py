"""Small local LLM (Ollama) used for explanations and study advice.

No API key and no cloud: the model runs on this laptop. The LLM never decides
anything. It only rewrites facts that the agent already has, and every reply
goes through guardrails before the student sees it.
"""
import json
import re
import time
import urllib.error
import urllib.request

from agent import tools

HOST = "http://localhost:11434"
MODEL = "qwen2.5:0.5b"
TIMEOUT_SECONDS = 90
MAX_CHARS = 500
# Model settings. The AI tutor test compares two settings before one is kept.
OPTIONS = {"temperature": 0.2, "num_predict": 80}
NUMBER = re.compile(r"\d+(?:\.\d+)?")
_cache = {"time": 0.0, "value": False}


def available():
    """True if Ollama is running and the model is downloaded (checked every 15 s)."""
    now = time.time()
    if now - _cache["time"] < 15:
        return _cache["value"]
    ok = False
    try:
        with urllib.request.urlopen(HOST + "/api/tags", timeout=2) as response:
            names = [m.get("name", "") for m in json.load(response).get("models", [])]
        ok = MODEL in names
    except Exception:
        ok = False
    _cache["time"], _cache["value"] = now, ok
    return ok


def ask(prompt, timeout=TIMEOUT_SECONDS):
    """Send one prompt to the local model and return its text."""
    body = json.dumps({
        "model": MODEL, "prompt": prompt, "stream": False, "keep_alive": "10m",
        "options": OPTIONS,
    }).encode("utf-8")
    request = urllib.request.Request(
        HOST + "/api/generate", data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
    except (urllib.error.URLError, OSError, ValueError):
        raise tools.ToolError("the AI tutor did not respond")
    return data.get("response", "")


def clean_reply(text):
    """Guardrail: tidy the reply and reject empty or unusable ones."""
    text = re.sub(r"\s+", " ", text or "").strip().replace("`", "")
    if len(text) < 15:
        raise tools.ToolError("the AI reply was empty")
    if text and text[-1] not in ".!?":
        end = max(text.rfind(". "), text.rfind("! "), text.rfind("? "))
        if end > 40:
            text = text[: end + 1]
    if len(text) > MAX_CHARS:
        cut = text[:MAX_CHARS]
        end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        text = cut[: end + 1] if end > 100 else cut.rstrip() + "..."
    return text


def numbers_grounded(reply, context):
    """Guardrail against made-up facts: every number must appear in the context."""
    allowed = set(NUMBER.findall(context))
    return all(n in allowed for n in NUMBER.findall(reply))


def explain(question, chosen_text, correct_text, stored_explanation):
    """Explain a wrong answer in simple words, using only the stored facts."""
    context = (
        f"Question: {question['question']}\n"
        f"Student's answer: {chosen_text}\n"
        f"Correct answer: {correct_text}\n"
        f"Explanation: {stored_explanation}"
    )
    prompt = (
        "You are a kind tutor for a university student. Use ONLY the facts below. "
        "In at most 2 short sentences, explain in simple words why the student's "
        "answer is wrong and why the correct answer is right. "
        "Do not add any new facts or numbers.\n\n" + context
    )
    reply = clean_reply(ask(prompt))
    if not numbers_grounded(reply, context):
        raise tools.ToolError("the reply contained numbers that are not in the lesson")
    return reply


def advice(progress, plan_message):
    """Turn the student's real results into two short study tips."""
    lines = [
        f"- {r.topic}: {r.accuracy:.0f}% correct over {r.answered} questions"
        for r in progress["topic_stats"].itertuples()
    ]
    recent = ", ".join(f"{s:.0f}%" for s in progress["recent_scores"]) or "none yet"
    context = (
        f"Quizzes taken: {progress['quizzes_taken']}\n"
        f"Recent scores: {recent}\n"
        + ("\n".join(lines) if lines else "No topic results yet.")
        + f"\nPlanned next step: {plan_message}"
    )
    prompt = (
        "You are a kind study coach. Use ONLY the facts below. Give 2 short, "
        "friendly study tips for this student. Do not invent numbers.\n\n" + context
    )
    reply = clean_reply(ask(prompt))
    if not numbers_grounded(reply, context):
        raise tools.ToolError("the reply contained numbers that are not in the results")
    return reply
