"""Build the vector index: read files in data/, split into chunks, embed with Ollama, save to index/index.npz.

Usage:  python ingest.py
"""
import json
import re
import time

import numpy as np
import ollama

import rag_config as cfg

EXTS = {".md", ".txt", ".json", ".csv"}


def split_markdown(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split on headings first, then pack paragraphs up to max_chars with a small overlap."""
    sections = re.split(r"(?m)^(?=#{1,4} )", text)
    chunks: list[str] = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        heading = sec.splitlines()[0] if sec.startswith("#") else ""
        if len(sec) <= max_chars:
            chunks.append(sec)
            continue
        buf = ""
        for para in re.split(r"\n\s*\n", sec):
            if buf and len(buf) + len(para) + 2 > max_chars:
                chunks.append(buf.strip())
                tail = buf[-overlap:] if overlap else ""
                buf = (heading + "\n" if heading and not tail.startswith(heading) else "") + tail + "\n\n" + para
            else:
                buf = f"{buf}\n\n{para}" if buf else para
            while len(buf) > max_chars * 1.5:  # very long paragraph / table: hard split
                chunks.append(buf[:max_chars].strip())
                buf = buf[max_chars - overlap:]
        if buf.strip():
            chunks.append(buf.strip())
    return chunks


def main() -> None:
    client = ollama.Client(host=cfg.OLLAMA_HOST)
    files = sorted(p for p in cfg.DATA_DIR.rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"No files found in {cfg.DATA_DIR}")

    texts, metas = [], []
    for f in files:
        rel = f.relative_to(cfg.DATA_DIR).as_posix()
        for i, chunk in enumerate(split_markdown(f.read_text(encoding="utf-8", errors="ignore"),
                                                 cfg.CHUNK_CHARS, cfg.CHUNK_OVERLAP)):
            texts.append(chunk)
            metas.append({"source": rel, "chunk": i})
    print(f"{len(files)} files -> {len(texts)} chunks. Embedding with {cfg.EMBED_MODEL} ...")

    t0 = time.time()
    vecs = []
    batch = 16
    for start in range(0, len(texts), batch):
        # Prefix the source name so the embedding knows which document the chunk belongs to.
        inp = [f"title: {metas[j]['source']} | text: {texts[j]}" for j in range(start, min(start + batch, len(texts)))]
        resp = client.embed(model=cfg.EMBED_MODEL, input=inp)
        vecs.extend(resp["embeddings"])
        print(f"  {min(start + batch, len(texts))}/{len(texts)}", end="\r")
    emb = np.asarray(vecs, dtype=np.float32)  # Ollama returns L2-normalised vectors -> dot product = cosine
    print(f"\nEmbedded {emb.shape[0]} chunks, dim {emb.shape[1]}, in {time.time() - t0:.1f}s")

    cfg.INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.INDEX_PATH, emb=emb,
                        texts=np.asarray(json.dumps(texts)), metas=np.asarray(json.dumps(metas)),
                        model=np.asarray(cfg.EMBED_MODEL))
    print(f"Index saved to {cfg.INDEX_PATH}")


if __name__ == "__main__":
    main()
