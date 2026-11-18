"""Local chat UI for the RAG prototype.

Start:  streamlit run app.py   (or double-click "Start chat.bat")
"""
import itertools
import time

import ollama
import streamlit as st

import ingest
import rag_config as cfg
from ask import Index, build_messages, stream_answer

HISTORY_TURNS = 2  # previous Q/A pairs sent to the model (kept small: CPU is slow on long prompts)

st.set_page_config(page_title=cfg.APP_TITLE, page_icon="💬", layout="centered")


@st.cache_resource
def get_client() -> ollama.Client:
    return ollama.Client(host=cfg.OLLAMA_HOST)


@st.cache_resource
def get_index(index_mtime: float) -> Index:  # mtime in the key -> reloads after re-ingest
    return Index(get_client())


def chat_models() -> list[str]:
    try:
        names = [m.model for m in get_client().list().models]
    except Exception:
        return [cfg.CHAT_MODEL]
    names = [n for n in names if "embed" not in n]
    return sorted(set(names) | {cfg.CHAT_MODEL}, key=lambda n: (n != cfg.CHAT_MODEL, n))


# ---------- sidebar ----------
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", chat_models(),
                         help="Small instruct models are fastest on CPU. ':cloud' models require `ollama signin`.")
    top_k = st.slider("Retrieved excerpts (top-k)", 2, 10, cfg.TOP_K)
    show_sources = st.toggle("Show sources", value=True)
    if st.button("New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption(f"Documents: `{cfg.DATA_DIR}`")
    if st.button("Rebuild index", use_container_width=True,
                 help="Use after adding or changing files in the data/ folder"):
        with st.spinner("Indexing documents..."):
            try:
                ingest.main()
                st.success("Index updated.")
            except SystemExit as e:
                st.error(str(e))

# ---------- index ----------
if not cfg.INDEX_PATH.exists():
    st.error("Index not found. Click **Rebuild index** in the sidebar.")
    st.stop()
try:
    index = get_index(cfg.INDEX_PATH.stat().st_mtime)
except SystemExit as e:
    st.error(str(e))
    st.stop()

# ---------- chat ----------
st.title(f"💬 {cfg.APP_TITLE}")
st.caption(f"Answers questions about {cfg.KB_DESCRIPTION} · {len(index.texts)} excerpts indexed · "
           "runs locally with Ollama.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(hits) -> None:
    with st.expander(f"Sources ({len(hits)})"):
        for score, meta, text in hits:
            st.markdown(f"**{meta['source']}** · excerpt {meta['chunk']} · similarity {score:.2f}")
            st.text(text[:600] + ("…" if len(text) > 600 else ""))


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("hits"):
            render_sources(msg["hits"])

if question := st.chat_input("Ask a question about your documents..."):
    with st.chat_message("user"):
        st.markdown(question)

    # Previous turns as plain Q/A (without their retrieved context) to keep the prompt short.
    history = [{"role": m["role"], "content": m["content"]}
               for m in st.session_state.messages[-2 * HISTORY_TURNS:]]
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        t0 = time.time()
        hits = index.search(question, top_k)
        try:
            with st.spinner("Searching the documents and writing the answer... (on CPU this can take up to a minute)"):
                stream = stream_answer(get_client(), build_messages(question, hits, history), model=model)
                first = next(stream, "")  # keep the spinner until the first token arrives
            reply = st.write_stream(itertools.chain([first], stream))
        except (ollama.ResponseError, ConnectionError) as e:
            reply = f"⚠️ Error from model `{model}`: {e}"
            st.error(reply)
        st.caption(f"{model} · {time.time() - t0:.0f}s")
        if show_sources:
            render_sources(hits)

    st.session_state.messages.append({"role": "assistant", "content": reply, "hits": hits})
