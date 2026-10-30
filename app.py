"""Streamlit UI for the Quiz Tutor Agent. Run with: streamlit run app.py"""
import streamlit as st

from agent import memory, planner, tools
from agent.agent import QuizAgent

st.set_page_config(page_title="Quiz Tutor Agent", page_icon="🎓")


@st.cache_resource
def get_agent():
    return QuizAgent()


agent = get_agent()

for key, default in [("session", None), ("result", None),
                     ("quiz_no", 0), ("hint_text", {})]:
    if key not in st.session_state:
        st.session_state[key] = default


def show_plan(session):
    plan = session["plan"]
    with st.expander("Why this quiz? (the agent's plan)", expanded=True):
        st.write(
            f"**Topic:** {plan['topic']}  |  **Difficulty:** {plan['difficulty']}  "
            f"|  **Questions:** {len(session['questions'])}"
        )
        st.write(f"**Question source:** {session['tool_used']}")
        for reason in plan["reasons"]:
            st.write("- " + reason)


def page_home():
    st.title("Quiz Tutor Agent")
    st.write(
        "Welcome! This agent plans each quiz from your past results, gives hints, "
        "marks your answers, and adapts the next quiz."
    )
    try:
        progress = tools.analyse_progress()
        rec = planner.recommend_next(llm_available=agent.generator is not None)
    except tools.ToolError as e:
        st.error(f"The question bank has a problem: {e}")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Quizzes taken", progress["quizzes_taken"])
    avg = progress["average_recent"]
    c2.metric("Recent average", f"{avg:.0f}%" if avg is not None else "-")
    c3.metric("Weakest topic", progress["weakest_topic"] or "-")
    st.subheader("Recommended next quiz")
    st.info(rec["message"])
    st.caption("Open 'Start Quiz' in the menu and let the agent decide.")


def show_setup():
    st.title("Start Quiz")
    try:
        topics = tools.list_topics()
    except tools.ToolError as e:
        st.error(f"Could not load the question bank: {e}")
        return
    auto = st.checkbox("Let the agent choose the topic and difficulty", value=True)
    topic = difficulty = None
    if not auto:
        topic = st.selectbox("Topic", topics)
        difficulty = st.selectbox("Difficulty", tools.DIFFICULTIES)
    n = st.slider("Number of questions", 1, planner.MAX_N, planner.DEFAULT_N)
    if st.button("Start quiz", type="primary"):
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
    st.title("Quiz")
    for notice in session["notices"]:
        st.warning(notice)
    show_plan(session)
    if session["plan"]["hint_mode"] == "proactive":
        st.info("Your recent average is low, so the agent suggests using the Hint buttons.")

    quiz_no = st.session_state.quiz_no
    choices = {}
    for i, q in enumerate(session["questions"], start=1):
        st.markdown(f"**Question {i}.** {q['question']}")
        choices[q["id"]] = st.radio(
            "Your answer",
            range(len(q["options"])),
            format_func=lambda k, opts=q["options"]: opts[k],
            index=None,
            key=f"ans_{quiz_no}_{q['id']}",
            label_visibility="collapsed",
        )
        if st.button("Hint", key=f"hint_{quiz_no}_{q['id']}"):
            st.session_state.hint_text[q["id"]] = agent.get_hint(session, q["id"])
        if q["id"] in st.session_state.hint_text:
            st.info("Hint: " + st.session_state.hint_text[q["id"]])
        st.divider()

    if st.button("Submit quiz", type="primary"):
        if any(v is None for v in choices.values()):
            st.warning("Please answer every question before submitting.")
            return
        try:
            with st.spinner("Marking your answers..."):
                for q in session["questions"]:
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
    st.title("Results")
    st.metric("Score", f"{result['score']:.0f}%")
    st.progress(int(result["score"]))
    st.write(f"{result['correct']} of {result['total']} correct.")
    for i, r in enumerate(result["review"], start=1):
        label = "Correct" if r["is_correct"] else "Wrong"
        with st.expander(f"Question {i}: {label}"):
            st.write(r["question"])
            st.write(f"Your answer: {r['options'][r['chosen']]}")
            st.write(f"Correct answer: {r['options'][r['correct_index']]}")
            if r["hint_used"]:
                st.caption("You used a hint on this question.")
            st.info(r["explanation"])
    st.subheader("What the agent recommends next")
    st.success(result["recommendation"]["message"])
    with st.expander("What did the agent do? (trace)"):
        for step in session["trace"]:
            st.write(f"**{step['step']}:** {step['detail']}")
    if st.button("Start a new quiz"):
        st.session_state.session = None
        st.session_state.result = None
        st.rerun()


def page_progress():
    st.title("Progress and History")
    history = memory.get_history()
    if history.empty:
        st.info("No quizzes yet. Take a quiz and your progress will appear here.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Quizzes taken", len(history))
    c2.metric("Average score", f"{history['score'].mean():.0f}%")
    c3.metric("Best score", f"{history['score'].max():.0f}%")

    st.subheader("Score over time")
    scores = history[["score"]].copy()
    scores.index = range(1, len(scores) + 1)
    scores.index.name = "Quiz number"
    st.line_chart(scores)

    st.subheader("Accuracy by topic (%)")
    stats = memory.get_topic_stats()
    st.bar_chart(stats.set_index("topic")["accuracy"])
    weakest = stats.iloc[0]
    st.write(f"Area to improve: **{weakest['topic']}** ({weakest['accuracy']:.0f}% correct).")

    st.subheader("Tools used by the agent")
    st.bar_chart(history["tool_used"].value_counts())

    st.subheader("Mistakes")
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

    st.subheader("Quiz history")
    shown = history.sort_values("id", ascending=False)[
        ["created_at", "topic", "difficulty", "tool_used", "score"]
    ].copy()
    shown["score"] = shown["score"].round(0)
    st.dataframe(shown, hide_index=True)


page = st.sidebar.radio("Go to", ["Home", "Start Quiz", "Progress and History"])

if page == "Home":
    page_home()
elif page == "Start Quiz":
    if st.session_state.result is not None:
        show_results()
    elif st.session_state.session is not None:
        show_quiz()
    else:
        show_setup()
else:
    page_progress()
