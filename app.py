import os
import re
import uuid

import streamlit as st

from answer import answer, rewrite_question
from auth import check_login
from chunking import split_text
from graph import build_graph, related_facts, to_dot, visible_triples
from ingest import ingest
from parsers import READERS, group_of, list_documents, read_file
from store import count_chunks, search
from viewer import show_document

DOCS = "documents"
os.makedirs(DOCS, exist_ok=True)

st.set_page_config(page_title="Ask your documents", layout="wide")
st.title("Ask your documents")

# ---- login: nothing below runs until someone has signed in
user = st.session_state.get("user")
if not user:
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in"):
            found = check_login(username, password)
            if found:
                st.session_state["user"] = found
                st.rerun()
            else:
                st.error("Wrong username or password.")
    st.stop()

groups = user["groups"]
is_admin = user["admin"]


# ---- chats (kept in this session only)
def new_chat():
    chats = st.session_state.setdefault("chats", {})
    cid = uuid.uuid4().hex
    chats[cid] = {"title": "New chat", "messages": []}
    st.session_state["current"] = cid
    while len(chats) > 20:
        del chats[next(iter(chats))]  # drop the oldest chat


def select_chat(cid):
    st.session_state["current"] = cid


def delete_chat():
    chats = st.session_state["chats"]
    chats.pop(st.session_state["current"], None)
    if chats:
        st.session_state["current"] = next(reversed(chats))
    else:
        new_chat()


if st.session_state.get("current") not in st.session_state.get("chats", {}):
    new_chat()


def turns_of(messages):
    """(question, answer) pairs from a chat, oldest first."""
    out, q = [], None
    for m in messages:
        if m["role"] == "user":
            q = m["content"]
        elif q is not None:
            out.append((q, m["content"]))
            q = None
    return out


def cited_numbers(text):
    nums = set()
    for m in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text):
        nums.update(int(x) for x in re.findall(r"\d+", m))
    return nums


def clean(text):
    """The answer without its [1] / [G1] citation markers."""
    return re.sub(r"\s*\[(?:G?\d{1,2}(?:\s*,\s*G?\d{1,2})*)\]", "", text)


def render(m):
    with st.chat_message(m["role"]):
        if m["role"] == "assistant" and m.get("confidential"):
            st.caption("🔒 Uses confidential documents. Don't paste outside this app.")
        st.markdown(clean(m["content"]) if m["role"] == "assistant" else m["content"])


# ---- sidebar
with st.sidebar:
    st.caption(f"Signed in as **{user['name']}**")
    if st.button("Sign out"):
        st.session_state.clear()
        st.rerun()

    st.header("Chats")
    st.button("＋ New chat", on_click=new_chat, use_container_width=True)
    current = st.session_state["current"]
    for cid, c in reversed(list(st.session_state["chats"].items())):
        st.button(
            c["title"], key=f"chat_{cid}", on_click=select_chat, args=(cid,),
            use_container_width=True, type="primary" if cid == current else "secondary",
        )
    st.button("Delete this chat", on_click=delete_chat)

    st.header("Documents")
    target = st.selectbox("Upload to group", groups)
    uploads = st.file_uploader(
        "Add files", type=[e.strip(".") for e in READERS], accept_multiple_files=True
    )
    if uploads and st.button("Save uploads to folder"):
        folder = os.path.join(DOCS, target)
        os.makedirs(folder, exist_ok=True)
        for up in uploads:
            with open(os.path.join(folder, os.path.basename(up.name)), "wb") as f:
                f.write(up.getbuffer())
        st.success(f"Saved {len(uploads)} file(s) to {target}. An admin must index them.")

    files = [f for f in list_documents(DOCS) if group_of(f) in groups]
    st.caption(f"{len(files)} documents you can access")

    if is_admin:
        if st.button("Index documents", type="primary", use_container_width=True):
            bar = st.progress(0.0, text="Starting...")
            st.session_state["report"] = ingest(
                lambda i, n, name: bar.progress(i / n, text=f"{i}/{n}: {name}")
            )
            st.rerun()
        report = st.session_state.get("report")
        if report:
            st.success(
                f"Indexed {len(report['indexed'])}, unchanged {len(report['skipped'])}, "
                f"removed {len(report['removed'])}"
            )
            for name, err in report["failed"]:
                st.warning(f"{name}: {err}")

    st.caption(f"{count_chunks(groups)} chunks you can search")

    with st.expander("Search settings"):
        k = st.slider("Passages to send to the model", 1, 10, 5)
        use_graph = st.checkbox("Use the knowledge graph", value=True)
        rerank = st.checkbox("Rerank results", value=True)
        picked = st.multiselect("Only search these files", files)

view = st.radio("View", ["Chat", "Preview", "Graph"], horizontal=True, label_visibility="collapsed")

# ================================= Chat =================================
if view == "Chat":
    chat = st.session_state["chats"][st.session_state["current"]]
    messages = chat["messages"]

    # the documents the latest answer cited (only ones this user may open)
    last = messages[-1] if messages and messages[-1]["role"] == "assistant" else None
    panel_files, cited = [], set()
    if last:
        cited = cited_numbers(last["content"])
        panel_files = [
            f for f in dict.fromkeys(s["file"] for s in last.get("sources", []) if s["n"] in cited)
            if f in files
        ]

    if panel_files:
        st.markdown(
            "<style>[data-testid='stColumn']:last-child, [data-testid='column']:last-child"
            "{position:sticky;top:3.5rem;align-self:flex-start;}</style>",
            unsafe_allow_html=True,
        )
        left, right = st.columns([2, 3])
    else:
        left, right = st.container(), None

    with left:
        if not messages:
            st.info("Ask a question about your documents. Use ＋ New chat for an unrelated conversation.")
        for m in messages:
            render(m)

    if right is not None:
        with right:
            if len(panel_files) > 1:
                file = st.radio("Document", panel_files, horizontal=True,
                                key=f"docsel_{st.session_state['current']}_{len(messages)}")
            else:
                file = panel_files[0]
            passages = [(s["loc"], s["text"]) for s in last["sources"]
                        if s["file"] == file and s["n"] in cited]
            question_text = messages[-2]["content"] if len(messages) >= 2 else ""
            st.markdown(f"**{file}**")
            found = show_document(os.path.join(DOCS, file), passages,
                                  question_text + " " + clean(last["content"]))
            if found < len(passages):
                st.caption(f"{len(passages) - found} passage(s) could not be located in the document text.")

    prompt = st.chat_input("Ask about your documents")
    if prompt and prompt.strip():
        prompt = prompt.strip()
        user_msg = {"role": "user", "content": prompt}
        with left:
            render(user_msg)
            if count_chunks(groups) == 0:
                reply = {"role": "assistant", "content": "There is nothing indexed that you can access yet.",
                         "sources": [], "facts": [], "confidential": False}
            else:
                with st.spinner("Thinking..."):
                    history = turns_of(messages)
                    standalone, related = rewrite_question(prompt, history)
                    results = search(standalone, k, rerank=rerank, groups=groups, sources=picked or None)
                    facts = related_facts(standalone, groups) if use_graph else []
                    if picked:
                        facts = [t for t in facts if t["source"] in picked]
                    try:
                        text, used = answer(standalone, results, facts, history if related else None)
                    except Exception as e:
                        text, used = f"Could not get an answer: {e}", results
                reply = {
                    "role": "assistant",
                    "content": text,
                    "sources": [
                        {
                            "n": i,
                            "file": p.payload["source"],
                            "loc": p.payload["locator"],
                            "text": p.payload["text"],
                            "ctx": p.payload.get("context", ""),
                        }
                        for i, p in enumerate(used, 1)
                    ],
                    "facts": facts,
                    "confidential": any(p.payload.get("group") == "confidential" for p in used)
                    or any(group_of(t["source"]) == "confidential" for t in facts),
                }
        messages += [user_msg, reply]
        del messages[:-40]
        if chat["title"] == "New chat":
            chat["title"] = prompt[:40]
        st.rerun()  # redraw with the new answer and its document

# ================================ Preview ================================
elif view == "Preview":
    chosen = st.selectbox("Preview a file", files) if files else None
    if chosen:
        c1, c2 = st.columns(2)
        size = c1.slider("Chunk size (characters)", 300, 3000, 1200, step=100)
        overlap = c2.slider("Overlap (characters)", 0, 600, 200, step=50)
        try:
            sections = read_file(os.path.join(DOCS, chosen))
            chunks = [(s.locator, c) for s in sections for c in split_text(s.text, size, overlap)]
            st.write(f"**{chosen}**: {len(sections)} section(s), {len(chunks)} chunks")
            for i, (loc, text) in enumerate(chunks[:30], 1):
                with st.expander(f"Chunk {i} · {loc or 'text'} · {len(text)} chars"):
                    st.text(text)
        except Exception as e:
            st.error(f"Could not read {chosen}: {e}")
    else:
        st.write("No documents you can access yet.")

# ================================= Graph =================================
else:
    if is_admin and st.button("Build / update graph"):
        bar = st.progress(0.0, text="Starting...")
        rep = build_graph(lambda i, n, name: bar.progress(i / n, text=f"{i}/{n}: {name}"))
        st.success(f"Built {len(rep['built'])}, unchanged {len(rep['skipped'])}, removed {len(rep['removed'])}")
        for name, err in rep["failed"]:
            st.warning(f"{name}: {err}")

    triples = visible_triples(groups)
    entities = {t["s"] for t in triples} | {t["o"] for t in triples}
    st.write(f"{len(triples)} relationships between {len(entities)} entities (CSV and Excel files are skipped)")
    focus = st.text_input("Show relationships around (leave empty for a sample)")
    shown = [
        t for t in triples
        if not focus or focus.lower() in t["s"].lower() or focus.lower() in t["o"].lower()
    ][:60]
    if shown:
        st.graphviz_chart(to_dot(shown), use_container_width=True)
        st.dataframe(shown, use_container_width=True)