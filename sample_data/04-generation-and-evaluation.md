# Generation, prompting and evaluation

## The prompt

The chat model receives:

1. A **system prompt** that sets the rules: answer only from the context, say so when the answer is not
   there, cite sources in square brackets, and answer in a given language.
2. Optionally, the **last few turns** of the conversation as plain question/answer pairs (without their
   old retrieved context, to keep the prompt short).
3. The **current question**, preceded by the retrieved excerpts, each labelled with its source file.

Small models follow instructions better when the most important ones sit right next to the question,
so the answer-language reminder is repeated at the end of the user message.

## Model settings

- **Temperature 0.1:** answers should be faithful to the documents, not creative.
- **Context window (`num_ctx`) 8,192 tokens:** enough for the system prompt, history and 5 chunks.
- **Streaming:** tokens are shown as soon as they are generated, so a slow CPU model still feels responsive.
- **Thinking disabled:** reasoning ("thinking") models are far slower on CPU; an instruct model is used.

## Running locally with Ollama

Ollama serves both the embedding model and the chat model through a local HTTP API on port 11434. On a
laptop without a usable GPU, a 4B-parameter model generates about 5–7 tokens per second, so a full answer
takes roughly one minute. The same code can point at a GPU server (`OLLAMA_HOST`) or at a cloud model
(`CHAT_MODEL=gpt-oss:20b-cloud`) without changes.

## How to evaluate a RAG system

Evaluate the two halves separately:

- **Retrieval:** for a set of test questions with a known source document, measure how often the right
  chunk appears in the top-k (*hit rate / recall@k*). `python ask.py --retrieve-only "..."` shows exactly
  what was retrieved and with what score.
- **Generation:** check that answers are *faithful* (every claim is supported by a retrieved excerpt),
  *relevant* to the question, and *cite* the right sources. This can be done by hand on a small set, or
  with an LLM acting as a judge.

When an answer is wrong, first look at the retrieved chunks: most failures are retrieval failures, not
model failures.
