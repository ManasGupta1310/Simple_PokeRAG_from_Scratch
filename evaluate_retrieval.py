import argparse
import json
from pathlib import Path

from rag import (
    build_name_matcher,
    build_vectorizer,
    chunk_by_entry,
    embed_chunks,
    name_from_chunk,
    retrieve_query,
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

    for case in eval_set["in_scope"]:
        query = case["query"]
        expected_names = case["expected_names"]
        retrieval_query, top_indices, scores = retrieve_query(
            query,
            vectorizer,
            chunk_embeddings,
            pokemon_names,
            name_vectorizer,
            name_embeddings,
            top_k=max(hit_counts),
            correct_typos=args.correct_name_typos,
        )
        retrieved_chunks = [chunks[index] for index in top_indices]
        retrieved_names = [name_from_chunk(chunk) for chunk in retrieved_chunks]
        top_name = retrieved_names[0] if retrieved_names else None
        top_expected_name = top_name if top_name in expected_names else None
        in_scope_scores.append(float(scores[0]) if len(scores) else 0.0)

        for k in hit_counts:
            if any(name in expected_names for name in retrieved_names[:k]):
                hit_counts[k] += 1

        print(
            f"IN  top1={float(scores[0]) if len(scores) else 0.0:.4f} "
            f"hit@1={'yes' if top_expected_name else 'no'} "
            f"query={query!r} "
            f"{f'retrieval_query={retrieval_query!r} ' if retrieval_query != query else ''}"
            f"retrieved={top_name or 'no named entry'}"
        )

    category_scores = {}
    for category in ("off_topic", "in_domain_unanswerable"):
        scores_for_category = []
        label = "OFF" if category == "off_topic" else "IN-DOMAIN-UNANSWERABLE"
        for query in eval_set[category]:
            retrieval_query, _, scores = retrieve_query(
                query,
                vectorizer,
                chunk_embeddings,
                pokemon_names,
                name_vectorizer,
                name_embeddings,
                top_k=1,
                correct_typos=args.correct_name_typos,
            )
            top_score = float(scores[0]) if len(scores) else 0.0
            scores_for_category.append(top_score)
            print(
                f"{label} top1={top_score:.4f} query={query!r} "
                f"{f'retrieval_query={retrieval_query!r}' if retrieval_query != query else ''}"
            )
        category_scores[category] = scores_for_category

    total = len(eval_set["in_scope"])
    print("\nIn-scope retrieval:")
    for k, hits in hit_counts.items():
        print(f"  hit@{k}: {hits}/{total} ({hits / total:.1%})")

    off_topic_scores = category_scores["off_topic"]
    if off_topic_scores:
        mean_off_topic_score = sum(off_topic_scores) / len(off_topic_scores)
        max_off_topic_score = max(off_topic_scores)
        min_in_score = min(in_scope_scores) if in_scope_scores else 0.0
        print("\nOff-topic score summary:")
        print(f"  mean top-1 score: {mean_off_topic_score:.4f}")
        print(f"  max top-1 score:  {max_off_topic_score:.4f}")
        print(
            "  conservative score gap (min in-scope top-1 - max off-topic top-1): "
            f"{min_in_score - max_off_topic_score:.4f}"
        )
    in_domain_scores = category_scores["in_domain_unanswerable"]
    if in_domain_scores:
        print("\nIn-domain unanswerable score summary:")
        print(f"  mean top-1 score: {sum(in_domain_scores) / len(in_domain_scores):.4f}")
        print(f"  max top-1 score:  {max(in_domain_scores):.4f}")


if __name__ == "__main__":
    main()
