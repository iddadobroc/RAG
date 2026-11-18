"""Shared settings for the RAG prototype.

Everything can be overridden with environment variables, so the same code can run:
  - fully local  (default: Ollama on this machine)
  - hybrid       (local embeddings + index, generation on an Ollama cloud model, e.g. CHAT_MODEL=gpt-oss:20b-cloud
                  after `ollama signin`)
  - on a shared GPU server (OLLAMA_HOST=http://<server>:11434)
"""
import os
from pathlib import Path

ROOT = Path(__file__).parent
# Your own documents go in data/ (git-ignored). If it does not exist, the bundled sample_data/ is used.
_default_data = ROOT / "data" if (ROOT / "data").is_dir() else ROOT / "sample_data"
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", _default_data))
INDEX_PATH = Path(os.getenv("RAG_INDEX_PATH", ROOT / "index" / "index.npz"))

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3:4b-instruct")  # non-thinking variant: much faster on CPU

CHUNK_CHARS = int(os.getenv("RAG_CHUNK_CHARS", "1200"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
NUM_CTX = int(os.getenv("RAG_NUM_CTX", "8192"))
# "English" (default) = always answer in English; "auto" = answer in the language of the question
ANSWER_LANGUAGE = os.getenv("RAG_ANSWER_LANGUAGE", "English")

# Cosmetic: name shown in the chat UI and a one-line description of the knowledge base for the system prompt.
APP_TITLE = os.getenv("RAG_APP_TITLE", "Local RAG Chat")
KB_DESCRIPTION = os.getenv("RAG_KB_DESCRIPTION", "the documents in the knowledge base")
