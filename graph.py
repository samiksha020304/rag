import json
import os
import time
import re
from google.genai import types

from answer import MODEL, _get_client
from chunking import split_text
from parsers import READERS, group_of, list_documents, read_file

DOCS = "documents"
GRAPH = "graph.json"
SKIP = {".csv",".xlsx"}

PROMPT = (
    "Extract the factual relationships stated in the text below as a JSON list of "
    'objects with keys "s" (subject), "p" (relationship, a short verb phrase) and '
    '"o" (object). Use short, canonical names for people, companies, tools, skills, '
    "places and concepts, and use the same name every time the same thing appears. "
    "Only include facts the text states. Return at most 20 items, or [] if there are none.\n\n"
    "TEXT:\n"
)


def load_graph():
    if os.path.exists(GRAPH):
        with open(GRAPH, encoding="utf-8") as f:
            return json.load(f)
    return {"files": {}, "triples": []}


def _save(data):
    with open(GRAPH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def _extract(text):
    for attempt in range(4):
        try:
            resp = _get_client().models.generate_content(
                model=MODEL,
                contents=PROMPT + text,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            items = json.loads(resp.text)
            return [
                {"s": i["s"].strip(), "p": i["p"].strip(), "o": i["o"].strip()}
                for i in items
                if isinstance(i, dict)
                and all(isinstance(i.get(k), str) and i[k].strip() for k in ("s", "p", "o"))
            ]
        except (json.JSONDecodeError, TypeError):
            return []  # model returned something unusable for this chunk
        except Exception as e:
            busy = any(c in str(e) for c in ("503", "429", "UNAVAILABLE"))
            if not busy or attempt == 3:
                raise
            time.sleep(2 * 2 ** attempt)


def build_graph(progress=None):
    data = load_graph()
    files = [f for f in list_documents(DOCS) if os.path.splitext(f)[1].lower() not in SKIP]
    report = {"built": [], "skipped": [], "failed": [], "removed": []}

    for i, name in enumerate(files, 1):
        if progress:
            progress(i, len(files), name)
        st_ = os.stat(os.path.join(DOCS, name))
        sig = [st_.st_size, st_.st_mtime_ns]
        if data["files"].get(name) == sig:
            report["skipped"].append(name)
            continue
        try:
            new = []
            for s in read_file(os.path.join(DOCS, name)):
                for chunk in split_text(s.text, 3000, 200):
                    for t in _extract(chunk):
                        t.update(source=name, locator=s.locator)
                        new.append(t)
            data["triples"] = [t for t in data["triples"] if t["source"] != name] + new
            data["files"][name] = sig
            report["built"].append((name, len(new)))
        except Exception as e:
            report["failed"].append((name, str(e)))
        _save(data)

    for name in list(data["files"]):  # files deleted from the folder
        if name not in files:
            data["triples"] = [t for t in data["triples"] if t["source"] != name]
            del data["files"][name]
            report["removed"].append(name)
    _save(data)
    return report


def to_dot(triples):
    def q(x):
        return '"' + x.replace("\\", "").replace('"', "'") + '"'

    lines = ["digraph G {", "rankdir=LR;", "node [shape=box];"]
    for t in triples:
        lines.append(f'{q(t["s"])} -> {q(t["o"])} [label={q(t["p"])}];')
    lines.append("}")
    return "\n".join(lines)

STOP = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "does", "did",
    "with", "from", "that", "this", "these", "those", "there", "their", "about", "have",
    "has", "had", "were", "was", "are", "the", "and", "for", "into", "over", "tell",
    "list", "give", "show", "person", "describe",
}


def _words(text):
    return {w for w in re.findall(r"[a-z0-9+#.]+", text.lower()) if len(w) >= 4 and w not in STOP}

def visible_triples(groups):
    """Graph facts from files this user may see."""
    groups = set(groups) if groups else {"everyone"}
    return [t for t in load_graph()["triples"] if group_of(t["source"]) in groups]

def related_facts(question, groups=None, limit=30):
    """Facts connected to the entities the question mentions."""
    triples = visible_triples(groups)
    if not triples:
        return []
    q = question.lower()
    qwords = _words(question)
    scores = {}
    for e in {t["s"] for t in triples} | {t["o"] for t in triples}:
        el = e.lower()
        if len(el) >= 3 and el in q:
            scores[e] = 10  # the entity's full name appears in the question
        else:
            n = len(_words(el) & qwords)
            if n:
                scores[e] = n  # some of its words appear
    top = set(sorted(scores, key=scores.get, reverse=True)[:5])
    facts = [t for t in triples if t["s"] in top or t["o"] in top]
    return facts[:limit]    