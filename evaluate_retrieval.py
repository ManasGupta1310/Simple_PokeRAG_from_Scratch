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
    "--no-typo-correction",
    dest="correct_typos",
    action="store_false",
    default=True,
    help="Disable character n-gram Pokémon-name typo correction.",
)


def _select_threshold(positive_scores, negative_scores):
    unique_scores = sorted(set(positive_scores + negative_scores))
    candidates = [0.0, 1.0]
    candidates.extend(
        (left + right) / 2
        for left, right in zip(unique_scores, unique_scores[1:])
    )

    best_threshold = 0.0
    best_balanced_accuracy = -1.0
    for threshold in candidates:
        true_positive_rate = sum(
            score >= threshold and score > 0 for score in positive_scores
        ) / len(positive_scores)
        true_negative_rate = sum(
            score < threshold or score == 0 for score in negative_scores
        ) / len(negative_scores)
        balanced_accuracy = (true_positive_rate + true_negative_rate) / 2
        if (balanced_accuracy, threshold) > (best_balanced_accuracy, best_threshold):
            best_threshold = threshold
            best_balanced_accuracy = balanced_accuracy

    return best_threshold, best_balanced_accuracy


def main():
    args = parser.parse_args()
    eval_path = Path(__file__).with_name("eval_set.json")
    with eval_path.open(encoding="utf-8") as f:
        eval_set = json.load(f)

    chunks = chunk_by_entry()
    names, known_terms, name_vectorizer, name_embeddings = build_name_matcher(chunks)
    vectorizer = build_vectorizer()
    chunk_embeddings = embed_chunks(chunks, vectorizer)

    in_scope_results = []
    hit_counts = {k: 0 for k in (1, 3, 5)}
    for case in eval_set["in_scope"]:
        query = case["query"]
        retrieval_query, top_indices, scores = retrieve_query(
            query,
            vectorizer,
            chunk_embeddings,
            names,
            known_terms,
            name_vectorizer,
            name_embeddings,
            top_k=max(hit_counts),
            correct_typos=args.correct_typos,
        )
        retrieved_names = [name_from_chunk(chunks[index]) for index in top_indices]
        expected_names = case["expected_names"]
        for k in hit_counts:
            if any(name in expected_names for name in retrieved_names[:k]):
                hit_counts[k] += 1

        top_score = float(scores[0]) if len(scores) else 0.0
        in_scope_results.append((case, top_score))
        top_name = retrieved_names[0] if retrieved_names else "no named entry"
        corrected_query = (
            f"retrieval_query={retrieval_query!r} "
            if retrieval_query != query
            else ""
        )
        print(
            f"IN  split={case['split']} top1={top_score:.4f} "
            f"hit@1={'yes' if top_name in expected_names else 'no'} "
            f"query={query!r} {corrected_query}retrieved={top_name}"
        )

    off_topic_results = []
    for case in eval_set["off_topic"]:
        query = case["query"]
        retrieval_query, _, scores = retrieve_query(
            query,
            vectorizer,
            chunk_embeddings,
            names,
            known_terms,
            name_vectorizer,
            name_embeddings,
            top_k=1,
            correct_typos=args.correct_typos,
        )
        score = float(scores[0]) if len(scores) else 0.0
        off_topic_results.append((case, score))
        corrected_query = (
            f"retrieval_query={retrieval_query!r} "
            if retrieval_query != query
            else ""
        )
        print(
            f"OFF split={case['split']} top1={score:.4f} "
            f"query={query!r} {corrected_query}"
        )

    unanswerable_scores = []
    for query in eval_set["in_domain_unanswerable"]:
        _, _, scores = retrieve_query(
            query,
            vectorizer,
            chunk_embeddings,
            names,
            known_terms,
            name_vectorizer,
            name_embeddings,
            top_k=1,
            correct_typos=args.correct_typos,
        )
        score = float(scores[0]) if len(scores) else 0.0
        unanswerable_scores.append(score)
        print(f"IN-DOMAIN-UNANSWERABLE top1={score:.4f} query={query!r}")

    calibration_positive_scores = [
        score for case, score in in_scope_results if case["split"] == "calibration"
    ]
    calibration_negative_scores = [
        score for case, score in off_topic_results if case["split"] == "calibration"
    ]
    threshold, calibration_balanced_accuracy = _select_threshold(
        calibration_positive_scores, calibration_negative_scores
    )

    holdout_positive_scores = [
        score for case, score in in_scope_results if case["split"] == "holdout"
    ]
    holdout_negative_scores = [
        score for case, score in off_topic_results if case["split"] == "holdout"
    ]
    holdout_accepted = sum(
        score >= threshold and score > 0 for score in holdout_positive_scores
    )
    holdout_rejected = sum(
        score < threshold or score == 0 for score in holdout_negative_scores
    )
    holdout_gap = (
        min(holdout_positive_scores) - max(holdout_negative_scores)
        if holdout_positive_scores and holdout_negative_scores
        else None
    )

    total = len(in_scope_results)
    print(f"\nName typo correction: {'on' if args.correct_typos else 'off'}")
    print("In-scope retrieval:")
    for k, hits in hit_counts.items():
        print(f"  hit@{k}: {hits}/{total} ({hits / total:.1%})")
    print("\nCalibration (threshold selected using these rows only):")
    print(f"  selected threshold: {threshold:.4f}")
    print(f"  balanced accuracy:  {calibration_balanced_accuracy:.1%}")
    print("\nHeld-out threshold check:")
    print(f"  in-scope accepted: {holdout_accepted}/{len(holdout_positive_scores)}")
    print(f"  off-topic rejected: {holdout_rejected}/{len(holdout_negative_scores)}")
    if holdout_gap is not None:
        print(f"  score gap: {holdout_gap:.4f}")

    if off_topic_results:
        off_topic_scores = [score for _, score in off_topic_results]
        print("\nOff-topic score summary:")
        print(f"  mean top-1 score: {sum(off_topic_scores) / len(off_topic_scores):.4f}")
        print(f"  max top-1 score:  {max(off_topic_scores):.4f}")
    if unanswerable_scores:
        print("\nIn-domain unanswerable score summary:")
        print(f"  mean top-1 score: {sum(unanswerable_scores) / len(unanswerable_scores):.4f}")
        print(f"  max top-1 score:  {max(unanswerable_scores):.4f}")


if __name__ == "__main__":
    main()
