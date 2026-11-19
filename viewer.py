import html
import os
import re

import streamlit as st
import streamlit.components.v1 as components

from parsers import read_file

MAX_CHARS = 400_000  # above this, only sections around the highlights are shown

_TOKEN = re.compile(r"\w+")
_STOP = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "its", "has", "have",
    "shall", "will", "not", "but", "any", "each", "what", "which", "who", "how", "per",
}

PAGE = """<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;height:100%}
#doc{position:relative;height:100%;overflow:auto;box-sizing:border-box;padding:20px 24px;
background:#fff;color:#1d1d1b;font:15px/1.65 Georgia,serif}
.sec{font:600 12px system-ui,sans-serif;color:#6b6b66;margin:22px 0 8px;padding-top:10px;
border-top:1px solid #e4e4de;text-transform:uppercase;letter-spacing:.04em}
.sec.firstsec{margin-top:0;border-top:0;padding-top:0}
mark.chunk{background:#fff3b0;color:inherit}
mark.key{background:#ffb74d;color:inherit}
</style></head><body><div id="doc">__BODY__</div>
<script>
var d=document.getElementById('doc'), f=document.getElementById('first');
if(f){ d.scrollTop = Math.max(0, f.offsetTop - 90); }
</script></body></html>"""


def _norm(text):
    """Same whitespace clean-up the chunker applies, so chunks can be found in the text."""
    return re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()


def _tokens(text):
    return {
        t for t in _TOKEN.findall(text.lower())
        if t not in _STOP and (len(t) >= 3 or any(c.isdigit() for c in t))
    }


def _sentence_spans(text):
    """(start, end) of each line or sentence in text."""
    spans = []
    for line in re.finditer(r"[^\n]+", text):
        for s in re.finditer(r".+?(?:(?<=[.!?])\s+|$)", line.group(0)):
            a = line.start() + s.start()
            piece = text[a:line.start() + s.end()]
            stripped = piece.strip()
            if stripped:
                a += len(piece) - len(piece.lstrip())
                spans.append((a, a + len(stripped)))
    return spans


def _key_spans(chunk, target, n=2):
    """The lines of a chunk that share the most words and numbers with the answer."""
    scored = []
    for a, b in _sentence_spans(chunk):
        common = _tokens(chunk[a:b]) & target
        score = sum(2 if any(c.isdigit() for c in t) else 1 for t in common)
        if score > 0:
            scored.append((score, a, b))
    scored.sort(reverse=True)
    if not scored:
        return []
    cutoff = max(2, scored[0][0] * 0.6)  # only lines nearly as good as the best one
    return sorted((a, b) for s, a, b in scored[:n] if s >= cutoff)


def _locate(chunk, sections, norms, locator):
    order = [i for i, (loc, _) in enumerate(sections) if loc == locator]
    order += [i for i in range(len(sections)) if i not in order]
    for probe in (chunk, chunk[:80]):
        for i in order:
            pos = norms[i].find(probe)
            if pos >= 0:
                return i, pos, min(len(norms[i]), pos + len(chunk))
    return None


def _highlight(text, chunk_spans, key_spans, first_done):
    cuts = sorted({0, len(text), *[x for s in chunk_spans + key_spans for x in s]})
    out = []
    for a, b in zip(cuts, cuts[1:]):
        seg = html.escape(text[a:b]).replace("\n", "<br>")
        if any(s <= a and b <= e for s, e in key_spans):
            kind = "key"
        elif any(s <= a and b <= e for s, e in chunk_spans):
            kind = "chunk"
        else:
            out.append(seg)
            continue
        anchor = ""
        if not first_done[0]:
            anchor, first_done[0] = ' id="first"', True
        out.append(f'<mark{anchor} class="{kind}">{seg}</mark>')
    return "".join(out)


@st.cache_data(show_spinner="Opening document...")
def load_sections(path, mtime):
    return [(s.locator, s.text) for s in read_file(path)]


def build_page(sections, passages, target_text):
    """Returns (html_page, number_of_passages_located)."""
    norms = [_norm(t) for _, t in sections]
    target = _tokens(target_text)
    marks, located = {}, 0
    for loc, chunk in passages:
        hit = _locate(chunk, sections, norms, loc)
        if not hit:
            continue
        located += 1
        i, a, b = hit
        c_spans, k_spans = marks.setdefault(i, ([], []))
        c_spans.append((a, b))
        k_spans.extend((a + x, a + y) for x, y in _key_spans(norms[i][a:b], target))

    show = list(range(len(sections)))
    if sum(len(n) for n in norms) > MAX_CHARS:
        keep = set()
        for i in marks:
            keep |= {i - 1, i, i + 1}
        show = [i for i in show if i in keep]

    first_done, body = [False], []
    for n, i in enumerate(show):
        loc = sections[i][0]
        if loc:
            cls = "sec firstsec" if n == 0 else "sec"
            body.append(f'<div class="{cls}">{html.escape(loc)}</div>')
        c_spans, k_spans = marks.get(i, ([], []))
        body.append(_highlight(norms[i], c_spans, k_spans, first_done))
    return PAGE.replace("__BODY__", "\n".join(body)), located


def show_document(path, passages, target_text, height=720):
    """Shows the whole document read-only, with the given passages highlighted.
    passages is a list of (locator, chunk_text). Returns how many were located."""
    sections = load_sections(path, os.path.getmtime(path))
    page, located = build_page(sections, passages, target_text)
    components.html(page, height=height, scrolling=False)
    return located