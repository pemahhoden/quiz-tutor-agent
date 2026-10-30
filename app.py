import streamlit as st

st.set_page_config(page_title="Quiz Tutor Agent", page_icon="🎓")

page = st.sidebar.radio(
    "Go to", ["Home", "Start Quiz", "Progress and History"]
)

if page == "Home":
    st.title("Quiz Tutor Agent")
    st.write("Welcome! The agent will plan quizzes based on how you are doing.")
elif page == "Start Quiz":
    st.title("Start Quiz")
    st.info("Quiz setup will go here.")
else:
    st.title("Progress and History")
    st.info("Charts and past attempts will go here.")
