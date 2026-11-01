"""Student notes: saved locally in SQLite and checked before use.

Privacy: notes never leave the laptop. They live in the local database
(which is excluded from GitHub) and can be deleted at any time.
"""
from datetime import datetime

from agent import memory, tools

MIN_CHARS = 200
MAX_CHARS = 20000
MAX_TITLE = 60
MAX_NOTES = 10
MAX_FILE_BYTES = 100_000
ALLOWED_FILES = (".txt", ".md")


def init_notes():
    """Create the notes table if it does not exist yet."""
    with memory.db() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS notes ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "title TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL)"
        )


def clean_text(text):
    """Remove control characters and extra blank space."""
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def decode_upload(filename, data):
    """Turn an uploaded file into text, or raise a clear error."""
    if not filename.lower().endswith(ALLOWED_FILES):
        raise tools.ToolError("Only .txt and .md files are supported.")
    if len(data) > MAX_FILE_BYTES:
        raise tools.ToolError("The file is too large (maximum 100 KB).")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise tools.ToolError(
            "The file could not be read as text. Save it as UTF-8 and try again."
        )


def check_note(title, text):
    """Guardrail: validate and clean a note before it is saved."""
    title = (title or "").strip()
    if not title:
        raise tools.ToolError("Please give your notes a title.")
    if len(title) > MAX_TITLE:
        raise tools.ToolError(f"The title is too long (maximum {MAX_TITLE} characters).")
    text = clean_text(text or "")
    if len(text) < MIN_CHARS:
        raise tools.ToolError(
            f"Your notes are too short. Please add at least {MIN_CHARS} characters."
        )
    if len(text) > MAX_CHARS:
        raise tools.ToolError(
            f"Your notes are too long (maximum {MAX_CHARS} characters)."
        )
    return title, text


def add_note(title, text):
    """Save a note and return its id."""
    title, text = check_note(title, text)
    init_notes()
    with memory.db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        if count >= MAX_NOTES:
            raise tools.ToolError(
                f"You can keep at most {MAX_NOTES} notes. Delete one first."
            )
        if conn.execute("SELECT 1 FROM notes WHERE title = ?", (title,)).fetchone():
            raise tools.ToolError("A note with this title already exists.")
        cur = conn.execute(
            "INSERT INTO notes (title, content, created_at) VALUES (?, ?, ?)",
            (title, text, datetime.now().isoformat(timespec="seconds")),
        )
        return cur.lastrowid


def list_notes():
    """All saved notes, oldest first."""
    init_notes()
    with memory.db() as conn:
        rows = conn.execute(
            "SELECT id, title, content, created_at FROM notes ORDER BY id"
        ).fetchall()
    return [{"id": r[0], "title": r[1], "content": r[2], "created_at": r[3]}
            for r in rows]


def get_note(note_id):
    for note in list_notes():
        if note["id"] == note_id:
            return note
    raise tools.ToolError("That note was not found.")


def delete_note(note_id):
    """Delete a note. Returns True if something was deleted."""
    init_notes()
    with memory.db() as conn:
        cur = conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        return cur.rowcount > 0
