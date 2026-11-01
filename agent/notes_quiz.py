"""Make fill-in-the-blank questions from the student's own notes.

Rule-based: no LLM and no API key. The correct answer is always a term taken
from the notes, and every question shows its source sentence as the
explanation, so the student can check it.
"""
import random
import re
from collections import Counter

from agent import generators, notes, tools

TOPIC_PREFIX = "My notes: "
BLANK = "_____"
WORD = re.compile(r"[A-Za-z][A-Za-z-]{3,}")
DEFINITION = re.compile(
    r"^([A-Za-z][A-Za-z-]*(?: [A-Za-z][A-Za-z-]*){0,2}) "
    r"(?:is|are|refers to|means) (.+)$"
)
STOP = set(
    "a an it in on of to if is as at by or so no we he she his her the and that "
    "this these those with from have has had were was are been being which who "
    "whom what when where while than then also can could may might will would "
    "should into over under between about such more most other some any each "
    "many much very only because used use using called known often not but for "
    "its their there they them your you our out one two".split()
)


def topic_name(title):
    return TOPIC_PREFIX + title


def split_sentences(text):
    """Split notes into sentences of a reasonable length."""
    sentences = []
    for line in text.splitlines():
        line = line.strip(" -*#\t")
        for s in re.split(r"(?<=[.!?])\s+", line):
            s = s.strip()
            if 5 <= len(s.split()) <= 35:
                sentences.append(s)
    return sentences


def _match_case(term, reference):
    if reference[:1].isupper():
        return term[:1].upper() + term[1:]
    return term[:1].lower() + term[1:]


def _good_term(term):
    return len(term) >= 4 and term.split()[0].lower() not in STOP


def _blank(sentence, term):
    return re.sub(rf"\b{re.escape(term)}\b", BLANK, sentence, flags=re.IGNORECASE)


def _distractors(answer, pool, sentence, reference):
    """Pick 3 different wrong options: not in the sentence, similar length."""
    low = sentence.lower()
    seen, options = set(), []
    for t in pool:
        key = t.lower()
        if key == answer.lower() or key in low or key in seen:
            continue
        seen.add(key)
        options.append(t)
    options.sort(key=lambda t: (abs(len(t) - len(answer)), random.random()))
    return [_match_case(t, reference) for t in options[:3]]


def _question(note, level, sentence, term, pool):
    wrong = _distractors(term, pool, sentence, term)
    if len(wrong) < 3:
        return None
    q = generators._make(
        topic_name(note["title"]), level,
        f'Fill in the blank using your notes: "{_blank(sentence, term)}"',
        term, wrong,
        f'The missing term has {len(term)} characters and starts with "{term[0]}".',
        f'From your notes: "{sentence}"',
    )
    if q is None:
        return None
    q["id"] = f"note{note['id']}-" + q["id"][4:]
    q["source"] = sentence
    return q


def _cloze_term(sentence, level, freq):
    """Choose which word to blank in a general (non-definition) sentence."""
    words = [w for w in dict.fromkeys(x.lower() for x in WORD.findall(sentence))
             if w not in STOP]
    if level == "medium":
        words = [w for w in words if freq[w] >= 2]
        return max(words, key=lambda w: freq[w]) if words else None
    words = [w for w in words if freq[w] == 1 and len(w) >= 6]
    return max(words, key=len) if words else None


def build_pools(note):
    """Make every question the notes can give, grouped by difficulty."""
    text = note["content"]
    freq = Counter(w.lower() for w in WORD.findall(text) if w.lower() not in STOP)
    frequent = [w for w, c in freq.most_common(40) if c >= 2]
    common = [w for w, _ in freq.most_common(80)]
    defs, others = [], []
    for s in split_sentences(text):
        m = DEFINITION.match(s)
        if m and _good_term(m.group(1)):
            defs.append((s, m.group(1)))
        else:
            others.append(s)
    def_terms = [t for _, t in defs]
    pools = {"easy": [], "medium": [], "hard": []}
    for s, term in defs:
        q = _question(note, "easy", s, term, def_terms + frequent + common)
        if q:
            pools["easy"].append(q)
    for s in others:
        for level in ("medium", "hard"):
            term = _cloze_term(s, level, freq)
            if term:
                q = _question(note, level, s, term, frequent + common)
                if q:
                    pools[level].append(q)
    return pools


def generate_from_note(note, difficulty, n):
    """The notes tool: up to n questions, using other levels if notes are small."""
    if difficulty not in tools.DIFFICULTIES:
        raise tools.ToolError(f"Unknown difficulty: {difficulty}")
    pools = build_pools(note)
    order = {"easy": ["easy", "medium", "hard"],
             "medium": ["medium", "easy", "hard"],
             "hard": ["hard", "medium", "easy"]}[difficulty]
    questions, used = [], set()
    for level in order:
        random.shuffle(pools[level])
        for q in pools[level]:
            if len(questions) == n:
                break
            if q["source"] not in used:
                used.add(q["source"])
                questions.append(q)
    if not questions:
        raise tools.ToolError(
            "Your notes do not have enough clear sentences to make questions. "
            "Try notes written in full sentences, for example definitions such as "
            "'Overfitting is when ...'."
        )
    return questions


def generate_for_topic(topic, difficulty, n):
    """Find the saved note that matches a topic name and make questions from it."""
    if not topic.startswith(TOPIC_PREFIX):
        raise tools.ToolError("This topic is not one of your notes.")
    title = topic[len(TOPIC_PREFIX):]
    for note in notes.list_notes():
        if note["title"] == title:
            return generate_from_note(note, difficulty, n)
    raise tools.ToolError("Those notes were not found.")
