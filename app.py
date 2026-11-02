"""Streamlit UI for the Quiz Tutor Agent. Run with: streamlit run app.py"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from agent import guardrails, memory, notes, notes_quiz, planner, tools
from agent.agent import QuizAgent
from agent.generators import generate_questions

st.set_page_config(page_title="Quiz Tutor Agent", page_icon="🎓")

HOME = "🏠 Home"

CSS = """
<style>
footer {visibility: hidden;}
.hero {background: linear-gradient(135deg, #4F46E5, #7C3AED); color: white;
       padding: 28px 24px; border-radius: 16px; margin-bottom: 20px;}
.hero h1 {color: white; margin: 0 0 6px 0; font-size: 2rem;}
.hero p {margin: 0; font-size: 1.05rem; opacity: 0.95;}
[data-testid="stMetric"] {background: white; border: 1px solid #E2E8F0;
       border-radius: 12px; padding: 12px 16px;}
.stButton > button {border-radius: 10px;}
.step-card {background: white; border: 1px solid #E2E8F0; border-radius: 12px;
       padding: 14px; font-size: 0.95rem;}
.topic-head {color: white; padding: 10px 14px; border-radius: 10px;
       font-weight: 700; font-size: 1.05rem; margin-bottom: 8px;}
.chip {display: inline-block; padding: 6px 12px; border-radius: 999px;
       margin: 4px 6px 4px 0; font-size: 0.9rem; font-weight: 600;}
.chip-on {background: #EEF2FF; color: #4338CA; border: 1px solid #C7D2FE;}
.chip-off {background: #F1F5F9; color: #94A3B8; border: 1px dashed #CBD5E1;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def get_agent():
    return QuizAgent(generator=generate_questions)


agent = get_agent()

for key, default in [("session", None), ("result", None), ("quiz_no", 0),
                     ("hint_text", {}), ("celebrated", 0), ("menu", HOME)]:
    if key not in st.session_state:
        st.session_state[key] = default

BADGES = {
    "easy": ":green-badge[easy]",
    "medium": ":orange-badge[medium]",
    "hard": ":red-badge[hard]",
}
TOPIC_ICONS = {"Python basics": "🐍", "Statistics": "📊",
               "Machine learning basics": "🤖"}
TOPIC_COLORS = {"Python basics": ("#2563EB", "#06B6D4"),
                "Statistics": ("#7C3AED", "#EC4899"),
                "Machine learning basics": ("#059669", "#65A30D")}
AUTO = "🤖 Let the agent decide"
DIFFICULTY_LABELS = {"easy": "🟢 easy", "medium": "🟠 medium", "hard": "🔴 hard"}


# ---------- small helpers and button actions ----------
def go_home():
    st.session_state.menu = HOME


def reset_quiz():
    st.session_state.session = None
    st.session_state.result = None
    st.session_state.hint_text = {}


def finish_and_go_home():
    reset_quiz()
    go_home()


def hero(title, subtitle):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def score_message(score):
    if score >= 80:
        return "🌟 Excellent work!"
    if score >= 50:
        return "👍 Good effort. Read the explanations below."
    return "💪 Keep practising. The agent will adjust your next quiz."


def score_ring(score):
    color = "#16A34A" if score >= 80 else "#F59E0B" if score >= 50 else "#DC2626"
    deg = int(score * 3.6)
    return (
        f'<div style="width:130px;height:130px;border-radius:50%;'
        f'background:conic-gradient({color} {deg}deg,#E2E8F0 0);'
        f'display:flex;align-items:center;justify-content:center;">'
        f'<div style="width:100px;height:100px;border-radius:50%;background:white;'
        f'display:flex;align-items:center;justify-content:center;'
        f'font-size:1.7rem;font-weight:700;color:{color};">{score:.0f}%</div></div>'
    )


def achievements(history, n_topics):
    scores = list(history["score"]) if not history.empty else []
    tried = len(memory.get_topic_stats())
    return [
        ("🥇", "First quiz", len(scores) >= 1),
        ("🔥", "3 quizzes done", len(scores) >= 3),
        ("💯", "Perfect score", any(s == 100 for s in scores)),
        ("📈", "Improving", len(scores) >= 2 and scores[-1] > scores[0]),
        ("🌍", "Tried every topic", n_topics > 0 and tried >= n_topics),
    ]


def show_plan(session):
    plan = session["plan"]
    with st.expander("🧠 Why this quiz? (the agent's plan)", expanded=True):
        st.write(
            f"**Topic:** {plan['topic']}  |  **Difficulty:** {plan['difficulty']}  "
            f"|  **Questions:** {len(session['questions'])}"
        )
        st.write(f"**Question source:** {session['tool_used']}")
        for reason in plan["reasons"]:
            st.write("- " + reason)


# ---------- pages ----------
def page_home():
    hero(
        "🎓 Quiz Tutor Agent",
        "An agent that plans your quizzes, gives hints, marks your answers "
        "and adapts to how you are doing.",
    )
    try:
        progress = tools.analyse_progress()
        rec = planner.recommend_next(llm_available=agent.generator is not None)
        bank = tools.load_bank()
    except tools.ToolError as e:
        st.error(f"The question bank has a problem: {e}")
        return
    history = memory.get_history()
    topics = {q["topic"] for q in bank}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quizzes taken", progress["quizzes_taken"])
    avg = progress["average_recent"]
    c2.metric("Recent average", f"{avg:.0f}%" if avg is not None else "-")
    c3.metric("Weakest topic", progress["weakest_topic"] or "-")
    c4.metric("Level", 1 + progress["quizzes_taken"] // 3)

    with st.container(border=True):
        st.markdown("#### 🎯 Recommended next quiz")
        st.write(rec["message"])
        st.caption("Open the Quiz page from the menu and let the agent decide.")

    st.markdown("#### 🏅 Achievements")
    chips = "".join(
        f'<span class="chip {"chip-on" if earned else "chip-off"}">'
        f'{icon} {label}</span>'
        for icon, label, earned in achievements(history, len(topics))
    )
    st.markdown(chips, unsafe_allow_html=True)

    st.markdown("#### How the agent works")
    steps = [
        ("1. Perceive", "Reads your past scores and mistakes."),
        ("2. Plan", "Chooses the topic, difficulty and tool."),
        ("3. Act", "Gives questions and hints, then marks your answers."),
        ("4. Adapt", "Saves the results and plans your next quiz."),
    ]
    for col, (title, text) in zip(st.columns(4), steps):
        col.markdown(
            f'<div class="step-card"><b>{title}</b><br>{text}</div>',
            unsafe_allow_html=True,
        )
    st.caption(f"Question bank: {len(bank)} questions across {len(topics)} topics.")


def topic_cards(topics):
    """One card per topic: size, questions already seen, and accuracy."""
    bank = tools.load_bank()
    stats = memory.get_topic_stats().set_index("topic")
    for col, topic in zip(st.columns(len(topics)), topics):
        total = len([q for q in bank if q["topic"] == topic])
        seen = len(memory.get_seen_question_ids(topic))
        c1, c2 = TOPIC_COLORS.get(topic, ("#4F46E5", "#7C3AED"))
        with col.container(border=True):
            st.markdown(
                f'<div class="topic-head" style="background:linear-gradient('
                f'135deg,{c1},{c2});">{TOPIC_ICONS.get(topic, "📚")} {topic}</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"{total} questions · {seen} already seen")
            if topic in stats.index:
                acc = float(stats.loc[topic, "accuracy"])
                st.progress(int(acc), text=f"Your accuracy: {acc:.0f}%")
            else:
                st.caption("Not started yet")


def show_setup():
    st.button("← Home", on_click=go_home, key="setup_home")
    st.header("📝 Start a quiz")
    try:
        topics = tools.list_topics()
        topic_cards(topics)
        my_topics = [notes_quiz.topic_name(n["title"]) for n in notes.list_notes()]
    except tools.ToolError as e:
        st.error(f"Could not load the question bank: {e}")
        return
    with st.container(border=True):
        st.markdown("**1. Topic**")
        topic_pick = st.pills(
            "Topic", [AUTO] + topics + my_topics, default=AUTO, selection_mode="single",
            format_func=lambda t: t if t == AUTO else f"{TOPIC_ICONS.get(t, '📚')} {t}",
            label_visibility="collapsed", key="topic_pick",
        )
        st.markdown("**2. Difficulty**")
        diff_pick = st.pills(
            "Difficulty", [AUTO] + tools.DIFFICULTIES, default=AUTO,
            selection_mode="single",
            format_func=lambda d: d if d == AUTO else DIFFICULTY_LABELS[d],
            label_visibility="collapsed", key="diff_pick",
        )
        st.markdown("**3. Number of questions**")
        n = st.slider("Number of questions", 1, planner.MAX_N, planner.DEFAULT_N,
                      label_visibility="collapsed")
        topic = None if topic_pick in (None, AUTO) else topic_pick
        difficulty = None if diff_pick in (None, AUTO) else diff_pick
        if topic is None or difficulty is None:
            st.caption("Anything set to 'Let the agent decide' is chosen by the "
                       "agent from your history, and it will explain why.")
        start = st.button("Start quiz", type="primary")
    if start:
        started = False
        try:
            with st.spinner("The agent is planning your quiz..."):
                st.session_state.session = agent.start_quiz(topic, difficulty, n)
            st.session_state.quiz_no += 1
            st.session_state.hint_text = {}
            st.session_state.result = None
            started = True
        except tools.ToolError as e:
            st.error(f"Could not start the quiz: {e}")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
        if started:
            st.rerun()


def show_quiz():
    session = st.session_state.session
    quiz_no = st.session_state.quiz_no
    questions = session["questions"]
    st.button("← Cancel quiz", on_click=reset_quiz, key="cancel_quiz")
    st.caption("Cancelling does not save this quiz.")
    st.header(f"📝 {TOPIC_ICONS.get(session['plan']['topic'], '')} "
              f"{session['plan']['topic']}")
    for notice in session["notices"]:
        st.warning(notice)
    show_plan(session)
    if session["plan"]["hint_mode"] == "proactive":
        st.info("Your recent average is low, so the agent suggests using the Hint buttons.")

    answered = sum(
        st.session_state.get(f"ans_{quiz_no}_{q['id']}") is not None
        for q in questions
    )
    st.progress(answered / len(questions),
                text=f"{answered} of {len(questions)} answered")

    choices = {}
    for i, q in enumerate(questions, start=1):
        with st.container(border=True):
            st.markdown(f"**Question {i}** {BADGES.get(q['difficulty'], '')}")
            st.write(q["question"])
            choices[q["id"]] = st.radio(
                "Your answer",
                range(len(q["options"])),
                format_func=lambda k, opts=q["options"]: opts[k],
                index=None,
                key=f"ans_{quiz_no}_{q['id']}",
                label_visibility="collapsed",
            )
            if st.button("💡 Hint", key=f"hint_{quiz_no}_{q['id']}"):
                st.session_state.hint_text[q["id"]] = agent.get_hint(session, q["id"])
            if q["id"] in st.session_state.hint_text:
                st.info("Hint: " + st.session_state.hint_text[q["id"]])

    if st.button("Submit quiz", type="primary"):
        if any(v is None for v in choices.values()):
            st.warning("Please answer every question before submitting.")
            return
        try:
            with st.spinner("Marking your answers..."):
                for q in questions:
                    if q["id"] not in session["attempts"]:
                        agent.submit_answer(session, q["id"], choices[q["id"]])
                st.session_state.result = agent.finish_quiz(session)
        except tools.ToolError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"Something went wrong while saving your quiz: {e}")
            return
        st.rerun()


def show_results():
    result = st.session_state.result
    session = st.session_state.session
    st.header("🏁 Results")
    if result["score"] >= 80 and st.session_state.celebrated != st.session_state.quiz_no:
        st.balloons()
        st.session_state.celebrated = st.session_state.quiz_no
    with st.container(border=True):
        c1, c2 = st.columns([1, 2])
        c1.markdown(score_ring(result["score"]), unsafe_allow_html=True)
        c2.markdown(f"### {score_message(result['score'])}")
        c2.write(f"{result['correct']} of {result['total']} correct.")

    st.subheader("Your answers")
    for i, r in enumerate(result["review"], start=1):
        icon = "✅" if r["is_correct"] else "❌"
        with st.expander(f"{icon} Question {i}", expanded=not r["is_correct"]):
            st.write(r["question"])
            if r["is_correct"]:
                st.success(f"Your answer: {r['options'][r['chosen']]}")
            else:
                st.error(f"Your answer: {r['options'][r['chosen']]}")
                st.success(f"Correct answer: {r['options'][r['correct_index']]}")
            if r["hint_used"]:
                st.caption("You used a hint on this question.")
            st.info(r["explanation"])

    st.subheader("What the agent recommends next")
    st.success(result["recommendation"]["message"])
    with st.expander("🔍 What did the agent do? (trace)"):
        for step in session["trace"]:
            st.write(f"**{step['step']}:** {step['detail']}")
    b1, b2 = st.columns(2)
    b1.button("Start a new quiz", type="primary", on_click=reset_quiz,
              key="new_quiz", use_container_width=True)
    b2.button("🏠 Back to Home", on_click=finish_and_go_home,
              key="results_home", use_container_width=True)


def page_quiz():
    if st.session_state.result is not None:
        show_results()
    elif st.session_state.session is not None:
        show_quiz()
    else:
        show_setup()


def page_progress():
    st.header("📈 Progress and History")
    history = memory.get_history()
    if history.empty:
        st.info("No quizzes yet. Take a quiz and your progress will appear here.")
        return
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Topics", "Mistakes", "History"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Quizzes taken", len(history))
        c2.metric("Average score", f"{history['score'].mean():.0f}%")
        c3.metric("Best score", f"{history['score'].max():.0f}%")
        st.subheader("Score over time")
        scores = history[["score"]].copy()
        scores.index = range(1, len(scores) + 1)
        scores.index.name = "Quiz number"
        st.line_chart(scores, color="#4F46E5")

    with tab2:
        stats = memory.get_topic_stats()
        st.subheader("Accuracy by topic (%)")
        st.bar_chart(stats.set_index("topic")["accuracy"], color="#7C3AED")
        weakest = stats.iloc[0]
        st.write(f"Area to improve: **{weakest['topic']}** "
                 f"({weakest['accuracy']:.0f}% correct).")
        st.subheader("Tools used by the agent")
        st.bar_chart(history["tool_used"].value_counts(), color="#059669")

    with tab3:
        attempts = memory.get_attempts()
        wrong = attempts[attempts["is_correct"] == 0]
        if wrong.empty:
            st.write("No mistakes yet.")
        else:
            try:
                texts = {q["id"]: q["question"] for q in tools.load_bank()}
            except tools.ToolError:
                texts = {}
            counts = wrong.groupby("question_id").size().sort_values(ascending=False)
            table = counts.rename("times wrong").reset_index()
            table["question"] = table["question_id"].map(lambda i: texts.get(i, "(generated question)"))
            st.dataframe(table[["question", "times wrong"]], hide_index=True)

    with tab4:
        shown = history.sort_values("id", ascending=False)[
            ["created_at", "topic", "difficulty", "tool_used", "score"]
        ].copy()
        shown["score"] = shown["score"].round(0)
        st.dataframe(shown, hide_index=True)


def page_notes():
    st.header("My Notes")
    st.write("Add your own notes and the agent will make quiz questions from them. "
             "Your notes stay on this laptop and you can delete them at any time.")
    with st.container(border=True):
        title = st.text_input("Title", max_chars=notes.MAX_TITLE)
        upload = st.file_uploader("Upload a file (optional): .txt, .md, .pdf, .docx or .pptx",
                                  type=["txt", "md", "pdf", "docx", "pptx"])
        pasted = st.text_area("Or paste your notes here", height=200)
        st.caption("If you do both, the uploaded file is used. Notes work best when "
                   "they are written in full sentences, for example "
                   "'Overfitting is when ...'.")
        if st.button("Save notes", type="primary"):
            try:
                text = (notes.decode_upload(upload.name, upload.getvalue())
                        if upload else pasted)
                notes.add_note(title, text)
            except tools.ToolError as e:
                st.error(str(e))
            else:
                st.success("Notes saved. Open the Quiz page and choose them as a topic.")
    st.subheader("Your saved notes")
    saved = notes.list_notes()
    if not saved:
        st.info("No notes yet.")
    for note in saved:
        with st.container(border=True):
            st.markdown(f"**{note['title']}**")
            st.caption(f"{len(note['content'])} characters, saved {note['created_at']}")
            with st.expander("Preview"):
                st.write(note["content"][:500]
                         + ("..." if len(note["content"]) > 500 else ""))
            if st.button("Delete", key=f"del_{note['id']}"):
                notes.delete_note(note["id"])
                st.rerun()


EVAL_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_ROOT / "evaluation" / "results"
EVAL_TABLES = {
    "Decision tests": "decision_tests.csv",
    "Fault tests": "fault_tests.csv",
    "Quiz runs": "quiz_runs.csv",
    "Generated question quality": "template_quality.csv",
    "Notes question quality": "notes_quality.csv",
}


def run_evaluation(include_notes):
    """Run the evaluation in a separate process, so the app's own data is safe."""
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ)
        if include_notes:
            for note in notes.list_notes():
                name = re.sub(r"[^A-Za-z0-9_-]+", "_", note["title"])[:40]
                path = Path(tmp) / f"{name}_{note['id']}.txt"
                path.write_text(note["content"], encoding="utf-8")
            env["EVAL_NOTES_DIR"] = tmp
        return subprocess.run(
            [sys.executable, "-m", "evaluation.run_eval"], cwd=EVAL_ROOT, env=env,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=900,
        )


def show_rating_tab():
    sheet = RESULTS_DIR / "notes_questions_to_rate.csv"
    if not sheet.exists():
        st.info("No rating sheet yet. Click 'Run evaluation' first.")
        return
    st.write("Rate each question from 1 to 5: 1 = confusing or wrong, "
             "5 = clear and relevant to the notes.")
    df = pd.read_csv(sheet, encoding="utf-8-sig")
    df["rating_1_to_5"] = pd.to_numeric(df["rating_1_to_5"], errors="coerce")
    edited = st.data_editor(
        df, hide_index=True, key="rating_editor",
        disabled=[c for c in df.columns if c != "rating_1_to_5"],
        column_config={"rating_1_to_5": st.column_config.NumberColumn(
            "Your rating (1-5)", min_value=1, max_value=5, step=1)},
    )
    if st.button("Save ratings"):
        edited.to_csv(sheet, index=False, encoding="utf-8-sig")
        st.success("Ratings saved. Run the evaluation again to add them to metrics.json.")
    rated = pd.to_numeric(edited["rating_1_to_5"], errors="coerce").dropna()
    if len(rated):
        st.metric("Average rating", f"{rated.mean():.2f} / 5",
                  help=f"{len(rated)} questions rated")
    if st.button("Start a new rating sheet"):
        sheet.unlink()
        st.info("Deleted. Click 'Run evaluation' to make a new sheet.")


def show_evaluation_results():
    metrics_path = RESULTS_DIR / "metrics.json"
    if not metrics_path.exists():
        st.info("No results yet. Click 'Run evaluation'.")
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    when = time.strftime("%d %b %Y, %H:%M",
                         time.localtime(metrics_path.stat().st_mtime))
    st.caption(f"Last run: {when}. {metrics['quiz_runs']} quiz runs, "
               f"random seed {metrics['seed']}.")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Summary", "Charts", "Test details", "Rate questions"])

    with tab1:
        cols = st.columns(3)
        for i, (name, value) in enumerate(metrics["rates"].items()):
            cols[i % 3].metric(name, "n/a" if value is None else f"{value:.1f}%")
        st.subheader("Speed (milliseconds)")
        st.dataframe(pd.DataFrame({"value": metrics["latency_ms"]}))
        st.subheader("Tools used")
        st.bar_chart(pd.Series(metrics["tool_usage"]))
        st.write(f"Unique generated questions: "
                 f"{metrics['generated_unique_share_percent']}%")
        st.write(f"Notes questions at the requested level: "
                 f"{metrics['notes_level_match_percent']}%")
        st.write(f"Notes quizzes that failed: {metrics['notes_failed_calls']} "
                 f"of {metrics['notes_calls']}")
        if "notes_human_rating" in metrics:
            r = metrics["notes_human_rating"]
            st.write(f"Your rating of notes questions: {r['mean_1_to_5']} / 5 "
                     f"({r['rated_questions']} rated)")

    with tab2:
        for name in ["success_rates", "latency", "tool_usage", "question_variety"]:
            path = RESULTS_DIR / "charts" / f"{name}.png"
            if path.exists():
                st.image(str(path))

    with tab3:
        choice = st.radio("Show", list(EVAL_TABLES), horizontal=True)
        path = RESULTS_DIR / EVAL_TABLES[choice]
        if path.exists():
            table = pd.read_csv(path)
            if st.checkbox("Show only problems (failed or unhandled)"):
                if "correct" in table.columns:
                    table = table[~table["correct"]]
                if "handled" in table.columns:
                    table = table[~table["handled"]]
            st.dataframe(table, hide_index=True)

    with tab4:
        show_rating_tab()


def page_evaluation():
    st.header("Evaluation")
    st.write("This page measures the agent on test cases and shows the real "
             "results. It uses a temporary database, so your own quizzes and "
             "notes are not changed.")
    my_notes = notes.list_notes()
    include = st.checkbox("Also use my saved notes (from the My Notes page)",
                          value=False, disabled=not my_notes)
    st.caption("The results then contain sentences from your notes. They stay on "
               "this laptop, and the rating sheet is kept out of GitHub.")
    if st.button("Run evaluation", type="primary"):
        with st.spinner("Running the evaluation. This takes one to three minutes..."):
            try:
                proc = run_evaluation(include)
            except subprocess.TimeoutExpired:
                st.error("The evaluation took too long and was stopped.")
                return
        if proc.returncode != 0:
            st.error("The evaluation failed. Details below.")
            st.code((proc.stderr or proc.stdout)[-2000:])
            return
        st.success("Evaluation finished.")
    show_evaluation_results()


# ---------- navigation ----------
PAGES = {HOME: page_home, "📝 Quiz": page_quiz, "📈 Progress": page_progress, "📓 My Notes": page_notes, "📊 Evaluation": page_evaluation}

st.sidebar.title("🎓 Quiz Tutor Agent")
choice = st.sidebar.radio("Menu", list(PAGES), label_visibility="collapsed",
                          key="menu")
st.sidebar.divider()
st.sidebar.caption(
    "Questions: question bank + rules" + (" + generated questions" if agent.generator else "")
)
st.sidebar.caption("Memory: local SQLite database")


def erase_everything():
    guardrails.erase_all_data()
    reset_quiz()
    st.session_state.confirm_erase = False
    st.session_state.menu = HOME


with st.sidebar.expander("Privacy"):
    st.caption("Your quizzes and notes are stored only on this laptop.")
    confirm = st.checkbox("I want to delete all my data", key="confirm_erase")
    st.button("Delete all my data", disabled=not confirm, on_click=erase_everything)

PAGES[choice]()
