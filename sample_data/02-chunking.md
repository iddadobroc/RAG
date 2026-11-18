# Chunking documents

Embedding models and prompts both have a limited size, so documents are split into **chunks** before
indexing. Chunking is one of the settings with the biggest effect on answer quality.

## Chunk size

- **Too large:** one chunk mixes several topics, its embedding becomes a blurry average, and fewer
  chunks fit in the prompt.
- **Too small:** a chunk loses the context needed to understand it (for example a table row without its
  header, or a rule without the section that says when it applies).

This prototype uses about **1,200 characters** per chunk (`RAG_CHUNK_CHARS`), which is roughly
250–300 tokens of English text.

## Overlap

Consecutive chunks share a small **overlap** (200 characters by default, `RAG_CHUNK_OVERLAP`). If a
sentence or an idea falls on a boundary, it still appears complete in at least one chunk.

## Structure-aware splitting

Splitting blindly every N characters cuts through sentences and sections. A better approach for
Markdown is:

1. Split on headings first (`#`, `##`, `###`, `####`), so each section is a natural unit.
2. If a section is still too long, pack whole paragraphs into chunks up to the size limit.
3. Repeat the section heading at the start of every chunk that comes from it, so each chunk carries
   its own context.
4. Only hard-split when a single paragraph (often a long table) is far over the limit.

## Adding the source to the chunk

Before embedding, each chunk is prefixed with its file name (`title: folder/file.md | text: ...`). This
helps the embedding model place the chunk in the right topic, and the metadata is kept so answers can
cite their source.
