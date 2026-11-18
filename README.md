# Local RAG prototype (Ollama + NumPy + Streamlit)

A small, dependency-light **Retrieval-Augmented Generation** pipeline that runs entirely on a laptop CPU.
It answers questions about a folder of Markdown/text documents with citations to the source files.

No vector database and no LangChain: about 340 lines of plain Python, so every step of RAG is visible.

```
            ┌──────────── ingest.py (offline) ────────────┐
 data/*.md ─► split by heading/paragraph ─► embed (Ollama) ─► index/index.npz
            └─────────────────────────────────────────────┘
                                                                  │
            ┌──────────── ask.py / app.py (online) ────────────────┼──────────────┐
 question ──► embed ─► cosine similarity (NumPy dot product) ─► top-k chunks
                                                       └─► prompt + chat model ─► answer [source.md]
            └──────────────────────────────────────────────────────────────────────┘
```

I built it originally to answer questions about my team's internal data-curation guidelines.
Those documents are private, so they are **not** in this repo. It ships with a small sample
knowledge base in `sample_data/` that explains RAG itself, so the bot can describe how it works.

## Features

- **Structure-aware chunking:** splits Markdown on headings, packs paragraphs up to ~1,200 characters
  with overlap, and repeats the section heading in each chunk.
- **Local embeddings** with [`embeddinggemma`](https://ollama.com/library/embeddinggemma) (multilingual,
  768-dim) using its document and query task prefixes.
- **Brute-force cosine search** in NumPy. With a few thousand chunks it takes milliseconds, so there's
  no vector DB to run.
- **Grounded answers:** the system prompt tells the model to answer only from the retrieved excerpts,
  to cite them as `[file.md]`, and to say when the context doesn't contain the answer.
- **Streaming chat UI** (Streamlit) with short conversation memory, adjustable top-k, a model picker,
  a source viewer with similarity scores, and a one-click "Rebuild index".
- **Backend-agnostic:** the same code runs fully locally, on a GPU server (`OLLAMA_HOST`) or with an
  Ollama cloud model (`CHAT_MODEL`). Every setting is an environment variable.
- **Safety check:** the index records which embedding model built it, and refuses to run if
  `EMBED_MODEL` has changed since.

## Project layout

```
sample_data/     demo knowledge base (Markdown) used when data/ does not exist
data/            your own documents (git-ignored)
index/           vector index built by ingest.py (git-ignored)
rag_config.py    models, paths, chunking, top-k (all overridable via env vars)
ingest.py        chunk -> embed -> save index
ask.py           CLI: retrieve -> prompt -> streamed answer with citations
app.py           Streamlit chat UI (reuses ask.py)
Start chat.bat   Windows double-click launcher (starts Ollama + the chat)
```

## Quick start

1. Install [Ollama](https://ollama.com) and pull the two models:

   ```bash
   ollama pull embeddinggemma
   ollama pull qwen3:4b-instruct
   ```

2. Install the Python dependencies (Python 3.10+):

   ```bash
   pip install -r requirements.txt
   ```

3. Build the index and ask a question:

   ```bash
   python ingest.py
   python ask.py "Why do chunks overlap?"
   python ask.py --retrieve-only "cosine similarity"   # inspect retrieval only
   python ask.py                                       # interactive mode
   ```

4. Or start the chat UI at http://localhost:8501:

   ```bash
   streamlit run app.py --server.address 127.0.0.1
   ```

   On Windows you can also double-click `Start chat.bat`.

## Using your own documents

Put `.md`, `.txt`, `.json` or `.csv` files in a `data/` folder (subfolders are fine), then run
`python ingest.py` again or click **Rebuild index** in the UI. `data/` and `index/` are git-ignored,
so private documents and their embeddings aren't committed.

## Configuration

| Variable              | Default                   | Meaning |
|-----------------------|---------------------------|---------|
| `OLLAMA_HOST`         | `http://127.0.0.1:11434`  | Ollama server (local, GPU server…) |
| `EMBED_MODEL`         | `embeddinggemma`          | Embedding model (re-run `ingest.py` after changing it) |
| `CHAT_MODEL`          | `qwen3:4b-instruct`       | Chat model, e.g. `gpt-oss:20b-cloud` after `ollama signin` |
| `RAG_DATA_DIR`        | `data/`, else `sample_data/` | Documents folder |
| `RAG_INDEX_PATH`      | `index/index.npz`         | Index file |
| `RAG_CHUNK_CHARS`     | `1200`                    | Target chunk size in characters |
| `RAG_CHUNK_OVERLAP`   | `200`                     | Overlap between consecutive chunks |
| `RAG_TOP_K`           | `5`                       | Chunks put in the prompt |
| `RAG_NUM_CTX`         | `8192`                    | Model context window |
| `RAG_ANSWER_LANGUAGE` | `English`                 | Answer language; `auto` = same as the question |
| `RAG_APP_TITLE`       | `Local RAG Chat`          | Title shown in the UI |
| `RAG_KB_DESCRIPTION`  | `the documents in the knowledge base` | What the assistant is about (used in the system prompt) |

## Performance

On a laptop CPU (Intel Core Ultra 7 165H, 16 GB RAM, no discrete GPU):

| Step                         | Time |
|------------------------------|------|
| Embedding ~250 chunks        | ~1 min |
| Retrieval                    | < 0.1 s (the query embedding dominates) |
| Answer with `qwen3:4b-instruct` | ~5–7 tokens/s, so 50–100 s per answer |

Pointing `OLLAMA_HOST` at a GPU machine, or using a cloud model, makes answers much faster
without code changes.

## Possible next steps

- Hybrid retrieval (BM25 + vectors) and a cross-encoder re-ranker
- Rewrite follow-up questions into standalone queries before retrieval
- A small evaluation set to measure recall@k and answer faithfulness

## License

MIT
