"""Streamlit UI for the Quiz Tutor Agent. Run with: streamlit run app.py"""
import streamlit as st

from agent import memory, planner, tools
from agent.agent import QuizAgent

st.set_page_config(page_title="Quiz Tutor Agent", page_icon="🎓")

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
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def get_agent():
    return QuizAgent()


agent = get_agent()

for key, default in [("session", None), ("result", None),
                     ("quiz_no", 0), ("hint_text", {})]:
    if key not in st.session_state:
        st.session_state[key] = default

BADGES = {
    "easy": ":green-badge[easy]",
    "medium": ":orange-badge[medium]",
    "hard": ":red-badge[hard]",
}


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
    c1, c2, c3 = st.columns(3)
    c1.metric("Quizzes taken", progress["quizzes_taken"])
    avg = progress["average_recent"]
    c2.metric("Recent average", f"{avg:.0f}%" if avg is not None else "-")
    c3.metric("Weakest topic", progress["weakest_topic"] or "-")

    with st.container(border=True):
        st.markdown("#### 🎯 Recommended next quiz")
        st.write(rec["message"])
        st.caption("Open the Quiz page from the menu and let the agent decide.")

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
    topics = {q["topic"] for q in bank}
    st.caption(f"Question bank: {len(bank)} questions across {len(topics)} topics.")


TOPIC_ICONS = {"Python basics": "🐍", "Statistics": "📊",
               "Machine learning basics": "🤖"}
AUTO = "🤖 Let the agent decide"
DIFFICULTY_LABELS = {"easy": "🟢 easy", "medium": "🟠 medium", "hard": "🔴 hard"}


def topic_cards(topics):
    """One card per topic: size, questions already seen, and accuracy."""
    bank = tools.load_bank()
    stats = memory.get_topic_stats().set_index("topic")
    for col, topic in zip(st.columns(len(topics)), topics):
        total = len([q for q in bank if q["topic"] == topic])
        seen = len(memory.get_seen_question_ids(topic))
        with col.container(border=True):
            st.markdown(f"### {TOPIC_ICONS.get(topic, '📚')} {topic}")
            st.caption(f"{total} questions · {seen} already seen")
            if topic in stats.index:
                acc = float(stats.loc[topic, "accuracy"])
                st.progress(int(acc), text=f"Your accuracy: {acc:.0f}%")
            else:
                st.caption("Not started yet")


def show_setup():
    st.header("📝 Start a quiz")
    try:
        topics = tools.list_topics()
        topic_cards(topics)
    except tools.ToolError as e:
        st.error(f"Could not load the question bank: {e}")
        return
    with st.container(border=True):
        st.markdown("**1. Topic**")
        topic_pick = st.pills(
            "Topic", [AUTO] + topics, default=AUTO, selection_mode="single",
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
    st.header("📝 Quiz")
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
    with st.container(border=True):
        c1, c2 = st.columns([1, 2])
        c1.metric("Score", f"{result['score']:.0f}%")
        c2.markdown(f"### {score_message(result['score'])}")
        c2.write(f"{result['correct']} of {result['total']} correct.")
        st.progress(int(result["score"]))

    st.subheader("Your answers")
    for i, r in enumerate(result["review"], start=1):
        icon = "✅" if r["is_correct"] else "❌"
        with st.expander(f"{icon} Question {i}", expanded=not r["is_correct"]):
            st.write(r["question"])
            st.write(f"Your answer: **{r['options'][r['chosen']]}**")
            st.write(f"Correct answer: **{r['options'][r['correct_index']]}**")
            if r["hint_used"]:
                st.caption("You used a hint on this question.")
            st.info(r["explanation"])

    st.subheader("What the agent recommends next")
    st.success(result["recommendation"]["message"])
    with st.expander("🔍 What did the agent do? (trace)"):
        for step in session["trace"]:
            st.write(f"**{step['step']}:** {step['detail']}")
    if st.button("Start a new quiz", type="primary"):
        st.session_state.session = None
        st.session_state.result = None
        st.rerun()


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
        st.line_chart(scores)

    with tab2:
        stats = memory.get_topic_stats()
        st.subheader("Accuracy by topic (%)")
        st.bar_chart(stats.set_index("topic")["accuracy"])
        weakest = stats.iloc[0]
        st.write(f"Area to improve: **{weakest['topic']}** "
                 f"({weakest['accuracy']:.0f}% correct).")
        st.subheader("Tools used by the agent")
        st.bar_chart(history["tool_used"].value_counts())

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
            table["question"] = table["question_id"].map(lambda i: texts.get(i, i))
            st.dataframe(table[["question", "times wrong"]], hide_index=True)

    with tab4:
        shown = history.sort_values("id", ascending=False)[
            ["created_at", "topic", "difficulty", "tool_used", "score"]
        ].copy()
        shown["score"] = shown["score"].round(0)
        st.dataframe(shown, hide_index=True)


PAGES = {"🏠 Home": page_home, "📝 Quiz": page_quiz, "📈 Progress": page_progress}

st.sidebar.title("🎓 Quiz Tutor Agent")
choice = st.sidebar.radio("Menu", list(PAGES), label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption(
    "Questions: question bank + rules" + (" + LLM" if agent.generator else "")
)
st.sidebar.caption("Memory: local SQLite database")

PAGES[choice]()
