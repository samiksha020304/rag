import json
import os

from answer import describe_document
from chunking import split_text
from parsers import READERS, group_of, list_documents, read_file
from store import add_chunks, delete_source

DOCS = "documents"
STATE = "index_state.json"


def _load():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {}


def _save(state):
    with open(STATE, "w") as f:
        json.dump(state, f)


def ingest(progress=None):
    state = _load()
    files = list_documents(DOCS)
    report = {"indexed": [], "skipped": [], "failed": [], "removed": []}

    for i, name in enumerate(files, 1):
        if progress:
            progress(i, len(files), name)
        st = os.stat(os.path.join(DOCS, name))
        signature = [st.st_size, st.st_mtime_ns]
        if state.get(name) == signature:  # unchanged since last time
            report["skipped"].append(name)
            continue
        try:
            sections = read_file(os.path.join(DOCS, name))
            chunks = [(s.locator, c) for s in sections for c in split_text(s.text)]
            if not chunks:
                raise ValueError("no extractable text (scanned PDF? needs OCR)")
            context = describe_document(name, sections)
            add_chunks(name, chunks, group_of(name), context)
            state[name] = signature
            report["indexed"].append((name, len(chunks)))
        except Exception as e:
            report["failed"].append((name, str(e)))
        _save(state)

    for name in list(state):  # files that were deleted from the folder
        if name not in files:
            delete_source(name)
            del state[name]
            report["removed"].append(name)
    _save(state)
    return report