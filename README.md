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

`SIMILARITY_THRESHOLD` is an optional value from `0` to `1` in `.env`; it defaults to `0.1299`, selected on the calibration split of the small eval set below by maximizing balanced accuracy. The score is TF-IDF cosine similarity, not a probability, and this cutoff is only a starting point. The held-out scores overlap, so the threshold is not a reliable general-purpose scope classifier.

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

The in-scope hit@1, hit@3, and hit@5 metrics count a query as a hit when an expected Pokémon entry appears in the top-k retrieved entries. Five in-scope cases omit the Pokémon name and ask about types or stats. Off-topic questions are reported separately from in-domain questions whose answers are absent from this dataset. The threshold is chosen using only rows marked `calibration`; rows marked `holdout` are used only to report held-out acceptance/rejection. No model/API call is made.

Run the benchmark from the project directory. Typo correction is enabled by default to match the API; use the flag to compare without it:

```bash
python evaluate_retrieval.py
python evaluate_retrieval.py --no-typo-correction
```

Current results on the included small eval set:

| Name typo correction | Hit@1 | Hit@3 | Hit@5 | Off-topic mean top-1 | Off-topic max top-1 |
|---|---:|---:|---:|---:|---:|
| Off | 11/15 (73.3%) | 12/15 (80.0%) | 12/15 (80.0%) | 0.0416 | 0.0913 |
| On (API default) | 13/15 (86.7%) | 14/15 (93.3%) | 14/15 (93.3%) | 0.0416 | 0.0913 |

| Name typo correction | Calibration-selected threshold | Calibration balanced accuracy | Held-out in-scope accepted | Held-out off-topic rejected | Held-out score gap |
|---|---:|---:|---:|---:|---:|
| Off | 0.1299 | 90.0% | 3/5 | 2/2 | -0.0457 |
| On (API default) | 0.1299 | 95.0% | 4/5 | 2/2 | -0.0457 |

These are small, hand-written examples, not a statistically reliable estimate. The held-out score gap is negative (`-0.0457`): “Who has the highest HP?” scores lower than an off-topic negative, and retrieval does not place the expected Blissey entry in the top five. Similarly, the Fire-type query retrieves a relevant entry at rank five but its top score is only `0.0669`. Thus the threshold can reject some valid nameless questions; the eval exposes that weakness rather than showing a clean cutoff. In-domain unanswerable questions mentioning a Pokémon still score as high as `0.3094`, so the prompt must explicitly abstain when the entries lack the requested fact.

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
SIMILARITY_THRESHOLD=0.1299      # optional; TF-IDF cosine score cutoff
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
