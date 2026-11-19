import re


def _split(text, size, seps):
    """Break text into pieces that are each at most `size` characters,
    cutting at the biggest natural boundary available."""
    if len(text) <= size:
        return [text]
    if not seps:                      # nothing left to cut on: hard cut
        return [text[i:i + size] for i in range(0, len(text), size)]
    sep, rest = seps[0], seps[1:]
    parts = text.split(sep)
    out = []
    for i, part in enumerate(parts):
        piece = part + (sep if i < len(parts) - 1 else "")
        if len(piece) <= size:
            out.append(piece)
        else:
            out.extend(_split(piece, size, rest))
    return out


def split_text(text, size=1200, overlap=200):
    text = re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()
    if not text:
        return []

    chunks, cur, cur_len = [], [], 0
    for piece in _split(text, size, ["\n\n", "\n", ". ", " "]):
        if cur and cur_len + len(piece) > size:
            chunks.append("".join(cur).strip())
            # start the next chunk with the tail of this one (the overlap)
            tail, tail_len = [], 0
            for prev in reversed(cur):
                if tail_len + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev)
            while tail and tail_len + len(piece) > size:
                tail_len -= len(tail.pop(0))
            cur, cur_len = tail, tail_len
        cur.append(piece)
        cur_len += len(piece)
    if cur:
        chunks.append("".join(cur).strip())
    return [c for c in chunks if c]