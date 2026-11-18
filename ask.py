"""Ask a question over the indexed documents.

Usage:
  python ask.py "Why do we split documents into overlapping chunks?"
  python ask.py --retrieve-only "cosine similarity"         # just show the retrieved chunks
  python ask.py                                              # interactive mode
Env: CHAT_MODEL, EMBED_MODEL, OLLAMA_HOST, RAG_TOP_K (see rag_config.py)
"""
import argparse
import json
import sys
import time

import numpy as np
import ollama

import rag_config as cfg

SYSTEM = (
    f"You are an assistant that answers questions about {cfg.KB_DESCRIPTION}. "
    "Answer ONLY from the context excerpts provided. "
    "If the context does not contain the answer, say so clearly. Cite the sources you used in square "
    "brackets, e.g. [folder/file.md]. "
    + ("Answer in the same language as the question." if cfg.ANSWER_LANGUAGE.lower() == "auto"
       else f"Always answer in {cfg.ANSWER_LANGUAGE}, whatever the language of the question.")
)


class Index:
    def __init__(self, client: ollama.Client):
        if not cfg.INDEX_PATH.exists():
            raise SystemExit(f"Index not found at {cfg.INDEX_PATH}. Run: python ingest.py")
        d = np.load(cfg.INDEX_PATH)
        self.emb = d["emb"]
        self.texts = json.loads(str(d["texts"]))
        self.metas = json.loads(str(d["metas"]))
        if str(d["model"]) != cfg.EMBED_MODEL:
            raise SystemExit(f"Index built with {d['model']}, but EMBED_MODEL={cfg.EMBED_MODEL}. Re-run ingest.py")
        self.client = client

    def search(self, query: str, k: int):
        q = self.client.embed(model=cfg.EMBED_MODEL, input=f"task: search result | query: {query}")["embeddings"][0]
        scores = self.emb @ np.asarray(q, dtype=np.float32)
        top = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.metas[i], self.texts[i]) for i in top]


def build_messages(question: str, hits, history: list[dict] | None = None) -> list[dict]:
    """System prompt + optional previous turns (plain Q/A, no context) + current question with retrieved context."""
    context = "\n\n".join(f"[{m['source']}]\n{t}" for _, m, t in hits)
    # Small models follow the language rule better when it sits right next to the question.
    lang = "" if cfg.ANSWER_LANGUAGE.lower() == "auto" else f"\n\n(Write your answer in {cfg.ANSWER_LANGUAGE}.)"
    return [
        {"role": "system", "content": SYSTEM},
        *(history or []),
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}{lang}"},
    ]


def stream_answer(client: ollama.Client, messages: list[dict], model: str = cfg.CHAT_MODEL):
    """Yield answer text pieces as the model generates them."""
    stream = client.chat(model=model, messages=messages, stream=True,
                         think=False, options={"num_ctx": cfg.NUM_CTX, "temperature": 0.1})
    for part in stream:
        yield part["message"]["content"]


def answer(index: Index, question: str, k: int, retrieve_only: bool) -> None:
    t0 = time.time()
    hits = index.search(question, k)
    print(f"\n--- Retrieved {len(hits)} chunks in {time.time() - t0:.2f}s")
    for s, m, t in hits:
        print(f"  {s:.3f}  {m['source']}#{m['chunk']}  {t[:90].replace(chr(10), ' ')}...")
    if retrieve_only:
        return

    print(f"--- Answer ({cfg.CHAT_MODEL}):\n")
    t0, n = time.time(), 0
    for piece in stream_answer(index.client, build_messages(question, hits)):
        n += 1
        print(piece, end="", flush=True)
    dt = time.time() - t0
    print(f"\n\n--- {dt:.1f}s, ~{n / dt:.1f} tokens/s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--retrieve-only", action="store_true")
    ap.add_argument("-k", type=int, default=cfg.TOP_K)
    args = ap.parse_args()

    index = Index(ollama.Client(host=cfg.OLLAMA_HOST))
    if args.question:
        answer(index, " ".join(args.question), args.k, args.retrieve_only)
        return
    print("Interactive mode. Empty line to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if not q:
            break
        answer(index, q, args.k, args.retrieve_only)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
