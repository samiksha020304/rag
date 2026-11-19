import hashlib
import io
import os
import time

from google.genai import types

from answer import MODEL, _get_client

CACHE = "ocr_cache"
MAX_PAGES = 30  # per file, to limit Gemini calls; raise it if you need more

PROMPT = (
    "Transcribe all the text on this page exactly as written, including handwriting, "
    "code and table contents. Keep headings and bullet structure. Return only the "
    "transcription, or nothing if the page has no text."
)


def _read_image(png):
    for attempt in range(4):
        try:
            resp = _get_client().models.generate_content(
                model=MODEL,
                contents=[types.Part.from_bytes(data=png, mime_type="image/png"), PROMPT],
            )
            return resp.text or ""
        except Exception as e:
            busy = any(c in str(e) for c in ("503", "429", "UNAVAILABLE"))
            if not busy or attempt == 3:
                raise
            time.sleep(2 * 2 ** attempt)


def ocr_pages(path, pages):
    """pages are 1-based page numbers. Returns a list of (page_number, text)."""
    import pypdfium2 as pdfium

    os.makedirs(CACHE, exist_ok=True)
    pdf = pdfium.PdfDocument(path)
    out = []
    for n in pages[:MAX_PAGES]:
        buf = io.BytesIO()
        pdf[n - 1].render(scale=2).to_pil().save(buf, format="PNG")
        png = buf.getvalue()
        cache_file = os.path.join(CACHE, hashlib.sha1(png).hexdigest() + ".txt")
        if os.path.exists(cache_file):
            with open(cache_file, encoding="utf-8") as f:
                text = f.read()
        else:
            text = _read_image(png)
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(text)
        if text.strip():
            out.append((n, text))
    return out