"""The Quiz Tutor Agent: joins the planner, tools and memory into one loop.

Loop: perceive -> plan -> select tool -> act -> evaluate -> give feedback
      -> update memory -> adapt (recommend the next quiz).

A "session" is a plain dictionary holding one quiz in progress. It also
keeps a trace of the agent's steps, which the app can show to the student.
"""
import random
import time

from agent import guardrails, memory, notes_quiz, planner, tools


class QuizAgent:
    def __init__(self, generator=None, llm=None):
        """generator: optional function (topic, difficulty, n) -> list of questions."""
        memory.init_db()
        self.generator = generator
        self.llm = llm

    @staticmethod
    def _log(session, step, detail):
        session["trace"].append({"step": step, "detail": detail})

    @staticmethod
    def _find(session, question_id):
        for q in session["questions"]:
            if q["id"] == question_id:
                return q
        raise tools.ToolError("This question is not part of the current quiz.")

    def _generate(self, plan, count):
        """Ask the optional LLM generator for questions and validate them."""
        raw = self.generator(plan["topic"], plan["difficulty"], count)
        questions = [
            q for q in raw
            if guardrails.check_question(q)[0] and q["topic"] == plan["topic"]
        ]
        if not questions:
            raise tools.ToolError("The generator produced no valid questions.")
        return questions[:count]

    def start_quiz(self, topic=None, difficulty=None, n=None):
        """Perceive, plan, choose a tool and prepare the questions."""
        session = {"trace": [], "notices": [], "attempts": {}, "hints": {}}

        # 1. PERCEIVE: read what memory knows about the student
        progress = tools.analyse_progress()
        self._log(
            session, "Perceive",
            f"{progress['quizzes_taken']} past quizzes. "
            f"Recent scores: {progress['recent_scores']}. "
            f"Weakest topic: {progress['weakest_topic']}.",
        )

        # 2. PLAN: decide topic, difficulty, number, tool and hint mode
        plan = planner.plan_quiz(
            topic, difficulty, n, llm_available=self.generator is not None
        )
        session["plan"] = plan
        self._log(session, "Plan", " ".join(plan["reasons"]))

        # 3. SELECT TOOL and 4. ACT
        tool_used = plan["tool"]
        self._log(session, "Select tool", tool_used)
        questions = None
        if tool_used == "notes_generator":
            questions = notes_quiz.generate_for_topic(
                plan["topic"], plan["difficulty"], plan["n"]
            )
        elif tool_used == "question_generator":
            try:
                seen_ids = set(memory.get_seen_question_ids(plan["topic"]))
                fresh = [q for q in tools.load_bank()
                         if q["topic"] == plan["topic"]
                         and q["difficulty"] == plan["difficulty"]
                         and q["id"] not in seen_ids]
                random.shuffle(fresh)
                questions = fresh[: plan["n"]]
                need = plan["n"] - len(questions)
                if need > 0:
                    questions += self._generate(plan, need)
                random.shuffle(questions)
            except Exception as e:  # any LLM failure must not crash the app
                session["notices"].append(
                    "The question generator failed, so the question bank is used instead."
                )
                self._log(session, "Error handled",
                          f"Question generator failed ({e}). Falling back to question_bank.")
                tool_used = "question_bank"
                questions = None
        if questions is None:
            seen = memory.get_seen_question_ids(plan["topic"])
            questions = tools.retrieve_questions(
                plan["topic"], plan["difficulty"], plan["n"], exclude_ids=seen
            )
        questions = guardrails.filter_questions(questions)
        if not questions:
            raise tools.ToolError(
                "No valid questions could be prepared. Please try again."
            )
        session["tool_used"] = tool_used
        session["questions"] = questions
        self._log(session, "Act", f"{len(questions)} questions ready from {tool_used}.")
        return session

    def get_hint(self, session, question_id):
        """Give the stored hint and remember that a hint was used."""
        q = self._find(session, question_id)
        session["hints"][question_id] = True
        self._log(session, "Hint", f"Hint given for {question_id}.")
        return q["hint"]

    def submit_answer(self, session, question_id, chosen):
        """5. EVALUATE one answer and 6. give feedback."""
        q = self._find(session, question_id)
        if question_id in session["attempts"]:
            raise tools.ToolError("This question was already answered.")
        result = tools.evaluate_answer(q, chosen)
        session["attempts"][question_id] = {
            "question_id": question_id,
            "topic": q["topic"],
            "difficulty": q["difficulty"],
            "chosen": chosen,
            "is_correct": int(result["is_correct"]),
            "hint_used": int(question_id in session["hints"]),
            "result": result,
        }
        self._log(session, "Evaluate",
                  f"{question_id}: {'correct' if result['is_correct'] else 'wrong'}.")
        return result

    def explain_answer(self, session, question_id):
        """LLM tool: explain a wrong answer in simple words.

        The LLM only gets the stored explanation to work from. If it is slow,
        offline or fails the safety check, the stored explanation is used.
        """
        q = self._find(session, question_id)
        attempt = session["attempts"].get(question_id)
        if attempt is None:
            raise tools.ToolError("Answer this question first.")
        result = attempt["result"]
        stored = result["explanation"]
        text, source = stored, "stored"
        note = "The AI tutor is not available, so the saved explanation is shown."
        start = time.perf_counter()
        if self.llm is not None:
            try:
                text = self.llm.explain(
                    q, q["options"][attempt["chosen"]], result["correct_text"], stored
                )
                source, note = "llm", ""
            except Exception as e:  # any LLM problem falls back to the stored text
                note = f"The AI tutor could not help ({e}), so the saved explanation is shown."
        seconds = round(time.perf_counter() - start, 1)
        self._log(session, "LLM tool",
                  f"Explanation for {question_id}: {source} in {seconds}s.")
        return {"text": text, "source": source, "seconds": seconds, "note": note}

    def study_advice(self):
        """LLM tool: short study tips written from the student's real results."""
        progress = tools.analyse_progress()
        plan = planner.recommend_next(llm_available=self.generator is not None)
        text, source = plan["message"], "rules"
        note = "The AI tutor is not available, so the rule-based advice is shown."
        start = time.perf_counter()
        if self.llm is not None:
            try:
                text = self.llm.advice(progress, plan["message"])
                source, note = "llm", ""
            except Exception as e:
                note = f"The AI tutor could not help ({e}), so the rule-based advice is shown."
        seconds = round(time.perf_counter() - start, 1)
        return {"text": text, "source": source, "seconds": seconds, "note": note}

    def finish_quiz(self, session):
        """7. UPDATE MEMORY and 8. ADAPT (recommend the next quiz)."""
        if session.get("finished"):
            raise tools.ToolError("This quiz is already finished.")
        questions = session["questions"]
        if len(session["attempts"]) < len(questions):
            raise tools.ToolError("Please answer every question before finishing.")

        attempts = [session["attempts"][q["id"]] for q in questions]
        plan = session["plan"]
        quiz_id = memory.save_quiz(
            plan["topic"], plan["difficulty"], session["tool_used"], attempts
        )
        correct = sum(a["is_correct"] for a in attempts)
        score = 100.0 * correct / len(attempts)
        self._log(session, "Update memory",
                  f"Quiz {quiz_id} saved with score {score:.0f}%.")

        recommendation = planner.recommend_next(
            llm_available=self.generator is not None
        )
        self._log(session, "Adapt", recommendation["message"])
        session["finished"] = True

        review = []
        for q in questions:
            a = session["attempts"][q["id"]]
            review.append({
                "id": q["id"],
                "question": q["question"],
                "options": q["options"],
                "chosen": a["chosen"],
                "correct_index": a["result"]["correct_index"],
                "is_correct": bool(a["is_correct"]),
                "hint_used": bool(a["hint_used"]),
                "explanation": a["result"]["explanation"],
            })
        return {
            "quiz_id": quiz_id,
            "score": score,
            "correct": correct,
            "total": len(attempts),
            "review": review,
            "recommendation": recommendation,
        }
