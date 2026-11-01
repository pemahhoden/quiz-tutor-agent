"""Student notes: saved locally in SQLite and checked before use.

Privacy: notes never leave the laptop. They live in the local database
(which is excluded from GitHub) and can be deleted at any time.
"""
import io
from datetime import datetime

from agent import memory, tools

MIN_CHARS = 200
MAX_CHARS = 50000
MAX_TITLE = 60
MAX_NOTES = 10
MAX_FILE_BYTES = 100_000
ALLOWED_FILES = (".txt", ".md", ".pdf", ".docx", ".pptx")
MAX_DOC_BYTES = 10_000_000
MAX_PDF_PAGES = 100


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


def _join_wrapped(text):
    """Join lines that were wrapped in the middle of a sentence (common in PDFs)."""
    paragraphs, buffer = [], ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            if buffer:
                paragraphs.append(buffer)
                buffer = ""
            continue
        buffer = f"{buffer} {line}" if buffer else line
        if line.endswith((".", "!", "?", ":")):
            paragraphs.append(buffer)
            buffer = ""
    if buffer:
        paragraphs.append(buffer)
    return "\n".join(paragraphs)


def _read_pdf(data):
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        raise tools.ToolError("This PDF is password-protected.")
    if len(reader.pages) > MAX_PDF_PAGES:
        raise tools.ToolError(
            f"This PDF has more than {MAX_PDF_PAGES} pages. Upload a smaller part."
        )
    pages = [page.extract_text() or "" for page in reader.pages]
    return _join_wrapped("\n".join(pages))


def _read_docx(data):
    from docx import Document

    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" ".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def _read_pptx(data):
    from pptx import Presentation

    prs = Presentation(io.BytesIO(data))
    parts = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                parts.extend(p.text for p in shape.text_frame.paragraphs)
    return "\n".join(parts)


READERS = {".pdf": _read_pdf, ".docx": _read_docx, ".pptx": _read_pptx}


def decode_upload(filename, data):
    """Turn an uploaded file into text, or raise a clear error."""
    name = filename.lower()
    if not name.endswith(ALLOWED_FILES):
        raise tools.ToolError("Supported files are .txt, .md, .pdf, .docx and .pptx.")
    if name.endswith((".txt", ".md")):
        if len(data) > MAX_FILE_BYTES:
            raise tools.ToolError("The file is too large (maximum 100 KB).")
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise tools.ToolError(
                "The file could not be read as text. Save it as UTF-8 and try again."
            )
    if len(data) > MAX_DOC_BYTES:
        raise tools.ToolError("The file is too large (maximum 10 MB).")
    reader = READERS[name[name.rindex("."):]]
    try:
        text = reader(data)
    except tools.ToolError:
        raise
    except ImportError:
        raise tools.ToolError(
            "Reading this file type needs an extra package. "
            "Run: pip install pypdf python-docx python-pptx"
        )
    except Exception:
        raise tools.ToolError(
            "The file could not be read. It may be damaged or password-protected."
        )
    if not text.strip():
        raise tools.ToolError(
            "No text was found in this file. Scanned pages and images are not supported."
        )
    return text


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
