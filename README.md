# Simple PokeRAG from Scratch

A minimal Retrieval-Augmented Generation (RAG) pipeline built without any RAG framework (no LangChain/LlamaIndex), to understand how each stage works. Ask questions about Pokémon and get answers grounded in a local Pokédex text file.

## How it works

```
question ──► TF-IDF vectorize ──► cosine similarity vs. chunks ──► top-k chunks
                                                                      │
                     answer ◄── Groq LLM ◄── prompt (question + context)
```

1. **Chunking**: each Pokémon CSV record in `pokedex.txt` is parsed as one entry and rendered as readable text.
2. **Embedding**: chunks and queries are vectorized with scikit-learn's `TfidfVectorizer` (English stop words, unigrams + bigrams). A separate character n-gram vectorizer matches likely misspelled Pokémon names before retrieval.
3. **Retrieval**: cosine similarity between the query vector and every entry; top 5 are returned. If the best score is below `SIMILARITY_THRESHOLD`, the request is declined without calling the language model.
4. **Augmentation**: retrieved chunks are injected into the prompt as context.
5. **Generation**: a Groq-hosted LLM answers using only that context.

`SIMILARITY_THRESHOLD` is an optional value from `0` to `1` in `.env`; it defaults to `0.10`. The score is TF-IDF cosine similarity, not a probability. The default is based on the small evaluation set below and should be revisited as that set grows. Queries with no matching terms are declined even if the configured threshold is `0`.

## Project structure

| File | Purpose |
|------|---------|
| `rag.py` | Chunking, vectorizing, similarity search |
| `main.py` | FastAPI app and LLM call |
| `pokedex.txt` | Knowledge base |
| `eval_set.json` | Retrieval-only in-scope, off-topic, and in-domain unanswerable examples |
| `evaluate_retrieval.py` | Reports hit@k and similarity score separation without calling an LLM |
| `requirements.txt` | Python package dependencies |

## Evaluate retrieval

The in-scope hit@1, hit@3, and hit@5 metrics count a query as a hit when an expected Pokémon entry appears in the top-k retrieved entries. Off-topic questions are reported separately from in-domain questions whose answers are absent from this dataset. The off-topic score gap is `minimum in-scope top-1 - maximum off-topic top-1`. No model/API call is made.

Run the benchmark from the project directory. The first command disables typo correction; the second enables it:

```bash
python evaluate_retrieval.py
python evaluate_retrieval.py --correct-name-typos
```

Current results on the included small eval set:

| Name typo correction | Hit@1 | Hit@3 | Hit@5 | Off-topic mean top-1 | Off-topic max top-1 | Score gap |
|---|---:|---:|---:|---:|---:|---:|
| Off | 8/10 (80%) | 8/10 (80%) | 8/10 (80%) | 0.0416 | 0.0913 | -0.0244 |
| On | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) | 0.0416 | 0.0913 | 0.1661 |

The `0.10` threshold is above the maximum off-topic score (`0.0913`) and below the minimum in-scope score with typo correction (`0.2574`) in this eval. This margin is preliminary, not a guarantee of generalization. In-domain unanswerable questions still scored as high as `0.3094` because mentioning a Pokémon retrieves its entry; a similarity threshold cannot tell whether that entry contains the requested fact, so the prompt/model must handle that case.

## Setup

```bash
git clone https://github.com/ManasGupta1310/Simple_PokeRAG_from_Scratch.git
cd Simple_PokeRAG_from_Scratch
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:

```
API_KEY=your_groq_api_key
MODEL=openai/gpt-oss-120b        # optional
SIMILARITY_THRESHOLD=0.10        # optional; TF-IDF cosine score cutoff
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

The response includes the generated `answer` and a `sources` list with each retrieved Pokémon name and its cosine similarity score. On declined queries, `sources` shows the retrieved candidates and their scores for debugging.

Interactive docs: http://127.0.0.1:8000/docs

## Limitations and next steps

Retrieval only provides the top five entries to the model. Questions requiring aggregation across the entire Pokédex—such as “fastest Pokémon” or “list all Fire types”—may be incomplete because the answer may require entries outside that top five.

## Acknowledgements

Learning resources: LangChain's *RAG From Scratch* series.
