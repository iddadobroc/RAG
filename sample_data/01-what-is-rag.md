# What is Retrieval-Augmented Generation (RAG)?

Retrieval-Augmented Generation (RAG) is a pattern for making a language model answer questions from a
specific body of documents instead of relying only on what it memorised during training.

A RAG system has two phases:

1. **Indexing (offline).** Documents are split into small chunks, each chunk is turned into a vector
   (an *embedding*) by an embedding model, and the vectors are stored in an index.
2. **Querying (online).** The user's question is embedded with the same model, the most similar chunks
   are retrieved from the index, and they are pasted into the prompt of a chat model together with the
   question. The model is instructed to answer only from those excerpts and to cite them.

## Why use RAG instead of fine-tuning?

- **Fresh knowledge.** Updating the knowledge base means re-indexing documents, not re-training a model.
- **Citations.** Because the answer is built from retrieved excerpts, the system can show its sources,
  which makes answers verifiable.
- **Less hallucination.** Telling the model "if the context does not contain the answer, say so" gives it
  a legitimate way to decline instead of inventing facts.
- **Cheap.** Small local models can give useful answers when the right context is in the prompt.
- **Access control.** Private documents stay in your own index; the model weights never contain them.

## Limits of RAG

- The answer can only be as good as the retrieval: if the right chunk is not in the top-k, the model
  cannot use it.
- Questions that need information spread across many documents (e.g. "count all X") are hard, because
  only a few chunks fit in the prompt.
- Small models sometimes ignore instructions such as the answer language or the citation format.
