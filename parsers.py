import os
from dataclasses import dataclass


@dataclass
class Section:
    text: str
    locator: str = ""   # where in the file this came from, e.g. "page 3"


def read_txt(path):
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return [Section(raw.decode(enc))]
        except UnicodeDecodeError:
            pass
    return [Section(raw.decode("utf-8", errors="replace"))]


def read_pdf(path):
    from pypdf import PdfReader

    sections, missing = [], []
    for i, page in enumerate(PdfReader(path).pages, 1):
        text = page.extract_text() or ""
        if text.strip():
            sections.append(Section(text, f"page {i}"))
        else:
            missing.append(i)  # no text layer: probably a scan
    if missing:
        from ocr import ocr_pages

        for n, text in ocr_pages(path, missing):
            sections.append(Section(text, f"page {n} (OCR)"))
        sections.sort(key=lambda s: int(s.locator.split()[1]))
    return sections

def read_docx(path):
    from docx import Document
    doc = Document(path)
    lines = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = []
            for cell in row.cells:
                value = cell.text.strip().replace("\n", " ")
                if value and (not cells or cells[-1] != value):
                    cells.append(value)
            if cells:
                lines.append(" | ".join(cells))
    return [Section("\n".join(lines))] if lines else []

 
def read_csv(path):
    import csv
    sections, buf, header, start, n = [], [], None, 1, 0
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        for row in csv.reader(f):
            if not any(c.strip() for c in row):
                continue
            if header is None:
                header = row
                continue
            n += 1
            pairs = [f"{h}: {v}" for h, v in zip(header, row) if v.strip()]
            buf.append("; ".join(pairs))
            if len(buf) == 25:
                sections.append(Section("\n".join(buf), f"rows {start}-{n}"))
                buf, start = [], n + 1
    if buf:
        sections.append(Section("\n".join(buf), f"rows {start}-{n}"))
    return sections

def _blocks(rows, label=""):
    """Turn table rows into 'column: value' lines, 25 rows per section."""
    sections, buf, header, start, n = [], [], None, 1, 0
    for row in rows:
        cells = ["" if c is None else str(c).strip() for c in row]
        if not any(cells):
            continue
        if header is None:
            header = cells
            continue
        n += 1
        buf.append("; ".join(f"{h}: {v}" for h, v in zip(header, cells) if v))
        if len(buf) == 25:
            sections.append(Section("\n".join(buf), f"{label} rows {start}-{n}".strip()))
            buf, start = [], n + 1
    if buf:
        sections.append(Section("\n".join(buf), f"{label} rows {start}-{n}".strip()))
    return sections


def read_xlsx(path):
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        out = []
        for ws in wb.worksheets:
            out.extend(_blocks(ws.iter_rows(values_only=True), ws.title))
        return out
    finally:
        wb.close()


def read_pptx(path):
    from pptx import Presentation

    out = []
    for i, slide in enumerate(Presentation(path).slides, 1):
        parts = []
        for shape in slide.shapes:
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                    if any(cells):
                        parts.append(" | ".join(cells))
            elif getattr(shape, "has_text_frame", False) and shape.text_frame.text.strip():
                parts.append(shape.text_frame.text.strip())
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                parts.append("Speaker notes: " + notes)
        if parts:
            out.append(Section("\n".join(parts), f"slide {i}"))
    return out


def list_documents(root="documents"):
    """Every supported file under root, as paths relative to root (with / separators)."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in filenames:
            if name.startswith((".", "~$")):
                continue
            if os.path.splitext(name)[1].lower() in READERS:
                rel = os.path.relpath(os.path.join(dirpath, name), root)
                found.append(rel.replace(os.sep, "/"))
    return sorted(found)

READERS = {
    ".txt": read_txt,
    ".md": read_txt,
    ".pdf": read_pdf,
    ".docx": read_docx,
    ".csv": read_csv,
    ".pptx": read_pptx,
    ".xlsx": read_xlsx,
}


def read_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext not in READERS:
        raise ValueError(f"unsupported file type: {ext}")
    return READERS[ext](path)

def group_of(rel_path):
    """The folder decides who may see a file. Unknown folders are treated as confidential."""
    if "/" not in rel_path:
        return "everyone"
    return "everyone" if rel_path.split("/")[0].lower() == "everyone" else "confidential"