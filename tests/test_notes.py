import pytest

from agent import memory, notes, tools

GOOD = "Overfitting is when a model memorises the training data. " * 5


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "t.db")
    memory.init_db()


def test_add_and_list_note(fresh_db):
    note_id = notes.add_note("  ML notes  ", GOOD)
    saved = notes.list_notes()
    assert len(saved) == 1
    assert saved[0]["id"] == note_id and saved[0]["title"] == "ML notes"
    assert notes.get_note(note_id)["content"].startswith("Overfitting")


def test_short_note_is_rejected(fresh_db):
    with pytest.raises(tools.ToolError):
        notes.add_note("Short", "Too short.")


def test_long_note_is_rejected(fresh_db):
    with pytest.raises(tools.ToolError):
        notes.add_note("Long", "word " * 12000)


def test_title_is_required_and_limited(fresh_db):
    with pytest.raises(tools.ToolError):
        notes.add_note("   ", GOOD)
    with pytest.raises(tools.ToolError):
        notes.add_note("x" * 61, GOOD)


def test_duplicate_title_is_rejected(fresh_db):
    notes.add_note("Same", GOOD)
    with pytest.raises(tools.ToolError):
        notes.add_note("Same", GOOD)


def test_note_limit(fresh_db):
    for i in range(notes.MAX_NOTES):
        notes.add_note(f"Note {i}", GOOD)
    with pytest.raises(tools.ToolError):
        notes.add_note("One too many", GOOD)


def test_delete_note(fresh_db):
    note_id = notes.add_note("Temp", GOOD)
    assert notes.delete_note(note_id) is True
    assert notes.delete_note(note_id) is False
    assert notes.list_notes() == []


def test_clean_text_removes_control_characters():
    assert notes.clean_text("a\x00b   c\n\n\n d") == "ab c\nd"


def test_decode_upload_checks():
    assert notes.decode_upload("n.md", "hello".encode("utf-8")) == "hello"
    with pytest.raises(tools.ToolError):
        notes.decode_upload("n.exe", b"hello")
    with pytest.raises(tools.ToolError):
        notes.decode_upload("n.txt", b"x" * (notes.MAX_FILE_BYTES + 1))
    with pytest.raises(tools.ToolError):
        notes.decode_upload("n.txt", b"\xff\xfe\xfa")
