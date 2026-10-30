import json
import argparse
from pathlib import Path

from rag import (
    build_name_matcher,
    build_vectorizer,
    chunk_by_entry,
    correct_name_typos,
    embed_chunks,
    embed_query,
    retrieve_similar_chunks,
)

parser = argparse.ArgumentParser(description="Evaluate retrieval without calling an LLM.")
parser.add_argument(
    "--correct-name-typos",
    action="store_true",
    help="Apply char n-gram matching to likely misspelled Pokémon names before retrieval.",
)


def main():
    args = parser.parse_args()
    eval_path = Path(__file__).with_name("eval_set.json")
    with eval_path.open(encoding="utf-8") as f:
        eval_set = json.load(f)

    chunks = chunk_by_entry()
    pokemon_names, name_vectorizer, name_embeddings = build_name_matcher(chunks)
    vectorizer = build_vectorizer()
    chunk_embeddings = embed_chunks(chunks, vectorizer)
    hit_counts = {k: 0 for k in (1, 3, 5)}
    in_scope_scores = []
    out_of_scope_scores = []

    for case in eval_set["in_scope"]:
        query = case["query"]
        expected_names = case["expected_names"]
        retrieval_query = query
        if args.correct_name_typos:
            retrieval_query = correct_name_typos(
                query, pokemon_names, name_vectorizer, name_embeddings
            )
        query_embedding = embed_query(retrieval_query, vectorizer)
        top_indices, scores = retrieve_similar_chunks(
            query_embedding, chunk_embeddings, top_k=max(hit_counts)
        )
        retrieved_chunks = [chunks[index] for index in top_indices]
        top_name = next(
            (
                name
                for name in expected_names
                if retrieved_chunks and f"Name: {name}" in retrieved_chunks[0]
            ),
            None,
        )
        in_scope_scores.append(float(scores[0]) if len(scores) else 0.0)

        for k in hit_counts:
            if any(
                f"Name: {name}" in chunk
                for chunk in retrieved_chunks[:k]
                for name in expected_names
            ):
                hit_counts[k] += 1

        print(
            f"IN  top1={float(scores[0]) if len(scores) else 0.0:.4f} "
            f"hit@1={'yes' if top_name else 'no'} "
            f"query={query!r} "
            f"{f'retrieval_query={retrieval_query!r} ' if retrieval_query != query else ''}"
            f"retrieved={top_name or 'no expected match'}"
        )

    for query in eval_set["out_of_scope"]:
        retrieval_query = query
        if args.correct_name_typos:
            retrieval_query = correct_name_typos(
                query, pokemon_names, name_vectorizer, name_embeddings
            )
        query_embedding = embed_query(retrieval_query, vectorizer)
        _, scores = retrieve_similar_chunks(query_embedding, chunk_embeddings, top_k=1)
        top_score = float(scores[0]) if len(scores) else 0.0
        out_of_scope_scores.append(top_score)
        print(f"OUT top1={top_score:.4f} query={query!r}")

    total = len(eval_set["in_scope"])
    print("\nIn-scope retrieval:")
    for k, hits in hit_counts.items():
        print(f"  hit@{k}: {hits}/{total} ({hits / total:.1%})")

    if out_of_scope_scores:
        mean_out_score = sum(out_of_scope_scores) / len(out_of_scope_scores)
        max_out_score = max(out_of_scope_scores)
        min_in_score = min(in_scope_scores) if in_scope_scores else 0.0
        print("\nOut-of-scope score summary:")
        print(f"  mean top-1 score: {mean_out_score:.4f}")
        print(f"  max top-1 score:  {max_out_score:.4f}")
        print(
            "  conservative score gap (min in-scope top-1 - max out-of-scope top-1): "
            f"{min_in_score - max_out_score:.4f}"
        )


if __name__ == "__main__":
    main()
