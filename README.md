# Simple PokeRAG from Scratch

A minimal Retrieval-Augmented Generation (RAG) pipeline built without any RAG framework (no LangChain/LlamaIndex), to understand how each stage works. Ask questions about Pokémon and get answers grounded in a local Pokédex text file.

## How it works

```
question ──► TF-IDF vectorize ──► cosine similarity vs. chunks ──► top-k chunks
                                                                      │
                     answer ◄── Groq LLM ◄── prompt (question + context)
```

1. **Chunking**: `pokedex.txt` is split into fixed-size chunks (1000 chars, 50% overlap).
2. **Embedding**: chunks and queries are vectorized with scikit-learn's `TfidfVectorizer` (English stop words, unigrams + bigrams).
3. **Retrieval**: cosine similarity between the query vector and every chunk; top 5 returned.
4. **Augmentation**: retrieved chunks are injected into the prompt as context.
5. **Generation**: a Groq-hosted LLM answers using only that context.

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
GROQ_MODEL=openai/gpt-oss-120b   # optional
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
- [ ] Chunk by Pokédex entry instead of fixed size
- [ ] Replace TF-IDF with dense embeddings (e.g. `sentence-transformers`)
- [ ] Similarity threshold for out-of-scope questions
- [ ] Source metadata for citations
- [ ] Small eval set to measure retrieval quality

## Acknowledgements

Learning resources: LangChain's *RAG From Scratch* series.
