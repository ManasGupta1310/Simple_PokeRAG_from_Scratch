# Simple PokeRAG from Scratch

A minimal Retrieval-Augmented Generation (RAG) pipeline built without any RAG framework (no LangChain/LlamaIndex), to understand how each stage works. Ask questions about Pokémon and get answers grounded in a local Pokédex text file.

## How it works

```
question ──► TF-IDF vectorize ──► cosine similarity vs. chunks ──► top-k chunks
                                                                      │
                     answer ◄── Groq LLM ◄── prompt (question + context)
```

1. **Chunking**: each Pokémon CSV record in `pokedex.txt` is parsed as one entry and rendered as readable text.
2. **Embedding**: chunks and queries are vectorized with scikit-learn's `TfidfVectorizer` (English stop words, unigrams + bigrams).
3. **Retrieval**: cosine similarity between the query vector and every entry; top 5 are returned. If the best score is below `SIMILARITY_THRESHOLD`, the request is declined without calling the language model.
4. **Augmentation**: retrieved chunks are injected into the prompt as context.
5. **Generation**: a Groq-hosted LLM answers using only that context.

`SIMILARITY_THRESHOLD` is an optional value from `0` to `1` in `.env`; it defaults to `0.02`. The score is TF-IDF cosine similarity, not a probability, so tune the threshold against representative in-scope and out-of-scope questions. Queries with no matching terms are declined even if the configured threshold is `0`.

## Project structure

| File | Purpose |
|------|---------|
| `rag.py` | Chunking, vectorizing, similarity search |
| `main.py` | FastAPI app and LLM call |
| `pokedex.txt` | Knowledge base |

## Setup

```bash
git clone https://github.com/ManasGupta1310/Simple_PokeRAG_from_Scratch.git
cd Simple_PokeRAG_from_Scratch
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install fastapi uvicorn scikit-learn groq python-dotenv
```

Create a `.env` file:

```
API_KEY=your_groq_api_key
MODEL=openai/gpt-oss-120b        # optional
SIMILARITY_THRESHOLD=0.02        # optional; TF-IDF cosine score cutoff
```

## Run

```bash
uvicorn main:app --reload
```

Then query it:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What type is Charizard?"}'
```

Interactive docs: http://127.0.0.1:8000/docs

## Limitations / next steps

- [ ] Build the index once at startup instead of per request
- [ ] Replace TF-IDF with dense embeddings (e.g. `sentence-transformers`)
- [ ] Source metadata for citations
- [ ] Small eval set to measure retrieval quality

## Acknowledgements

Learning resources: LangChain's *RAG From Scratch* series.
