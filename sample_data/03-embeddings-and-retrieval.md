# Embeddings and retrieval

## What is an embedding?

An embedding model maps a piece of text to a fixed-length vector of numbers (768 numbers for
`embeddinggemma`). Texts with similar meaning end up close to each other in that vector space, even
when they use different words or different languages. This is what allows *semantic* search: the
question "how big should a chunk be?" can match a passage about "chunk size" without sharing keywords.

## Cosine similarity

Closeness between two vectors is measured with **cosine similarity**: 1.0 means same direction (very
similar), 0 means unrelated. When vectors are L2-normalised (length 1), cosine similarity is simply the
dot product, so scoring every chunk against the question is a single matrix-vector multiplication:

```python
scores = chunk_embeddings @ question_embedding   # one score per chunk
top = numpy.argsort(-scores)[:k]                 # indices of the k best chunks
```

For a few thousand chunks this brute-force search takes milliseconds on a laptop, so no vector database
is needed. Dedicated vector databases (FAISS, pgvector, Qdrant, Chroma...) become useful with millions
of chunks, metadata filters or many concurrent users.

## Top-k

`k` is the number of chunks pasted into the prompt (5 by default, `RAG_TOP_K`). A larger k raises the
chance that the right chunk is included but makes the prompt longer, slower on CPU, and noisier for a
small model.

## Query and document prompts

Some embedding models are trained with task prefixes. `embeddinggemma` expects documents as
`title: ... | text: ...` and queries as `task: search result | query: ...`. Using the expected prefixes
noticeably improves retrieval.

## Keep the same model

The question must be embedded with the **same model** used to build the index, otherwise the vectors are
not comparable. This prototype stores the model name inside the index file and refuses to answer if
`EMBED_MODEL` has changed since the last ingest.

## Ideas to improve retrieval

- **Hybrid search:** combine vector similarity with keyword search (BM25) to catch exact names and codes.
- **Re-ranking:** retrieve 20–50 candidates, then re-order them with a cross-encoder model.
- **Query rewriting:** let the chat model rewrite a follow-up question ("and for debt?") into a
  standalone question before retrieval.
