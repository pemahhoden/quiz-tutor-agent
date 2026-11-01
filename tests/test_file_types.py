import io

import pytest

from agent import memory, notes, notes_quiz, tools

LINES = [
    "Overfitting is when a model memorises the training data and fails on new data.",
    "Underfitting is when a model is too simple to learn the pattern in the data.",
    "Regularisation is a method that adds a penalty to reduce overfitting in a model.",
    "Cross-validation is a method that tests a model on several different splits of the data.",
]


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "t.db")
    memory.init_db()


def make_docx(lines):
    from docx import Document

    doc = Document()
    for line in lines:
        doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def make_pptx(lines):
    from pptx import Presentation

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    frame = slide.shapes.add_textbox(0, 0, 5000000, 3000000).text_frame
    frame.text = lines[0]
    for line in lines[1:]:
        frame.add_paragraph().text = line
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def make_pdf(text):
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for i, obj in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (
        len(objs) + 1, xref)
    return out


def test_docx_text_is_read():
    text = notes.decode_upload("n.docx", make_docx(LINES))
    assert "Overfitting is when" in text and "Cross-validation" in text


def test_pptx_text_is_read():
    text = notes.decode_upload("n.pptx", make_pptx(LINES))
    assert "Underfitting is when" in text


def test_pdf_text_is_read():
    text = notes.decode_upload("n.pdf", make_pdf("Overfitting is when a model memorises data."))
    assert "Overfitting" in text


def test_damaged_files_give_clear_errors():
    for name in ["bad.pdf", "bad.docx", "bad.pptx"]:
        with pytest.raises(tools.ToolError):
            notes.decode_upload(name, b"this is not a real file")


def test_documents_without_text_are_rejected():
    with pytest.raises(tools.ToolError):
        notes.decode_upload("empty.docx", make_docx([]))


def test_big_documents_are_rejected():
    with pytest.raises(tools.ToolError):
        notes.decode_upload("big.pdf", b"x" * (notes.MAX_DOC_BYTES + 1))


def test_wrapped_pdf_lines_are_joined():
    wrapped = "Overfitting is when a model\nmemorises the data.\n\nNext idea"
    assert notes._join_wrapped(wrapped) == (
        "Overfitting is when a model memorises the data.\nNext idea"
    )


def test_uploaded_docx_becomes_a_quiz(fresh_db):
    text = notes.decode_upload("n.docx", make_docx(LINES))
    notes.add_note("ML notes", text)
    questions = notes_quiz.generate_for_topic("My notes: ML notes", "easy", 3)
    assert len(questions) == 3
