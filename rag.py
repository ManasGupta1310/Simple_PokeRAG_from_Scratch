# A RAG model works by combining retrieval, augmenting and generating
# Retrieval: Fetching relevant documents or data from a knowledge base.
# Augmenting: Combining the retrieved information with the user's query.
# Generating: Producing a response based on the augmented information.

# For document storage, we split the document into specific chunks and store them in vector dataabase using embeddings
# This allows for efficient similarity search and retrieval of relevant chunks based on the user's query.

# For retrieval, we embed the query and then search across vector database to find the most similar chunks using similarity 
# search algorithms like cosine similarity.

# The retrieved chunks are then combined with the original query to form the augmented input for the generation step.

# For generation, we use a language model to produce a response based on the augmented input, combining the original query 
# and the retrieved chunks as the context for the prompt to the language model.

import csv
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NAME_TYPO_MATCH_THRESHOLD = 0.6
_WORD_PATTERN = re.compile(r"[^\W_]+(?:[♀♂])?", re.UNICODE)


def _format_pokemon_entry(row):
    name = (row.get("Name") or "").strip()
    number = (row.get("Number") or "").strip()
    type_1 = (row.get("Type 1") or "").strip()
    type_2 = (row.get("Type 2") or "").strip()
    stats = {
        "HP": (row.get("HP") or "").strip(),
        "Attack": (row.get("Attack") or "").strip(),
        "Defense": (row.get("Defense") or "").strip(),
        "Sp. Attack": (row.get("Sp.Attack") or "").strip(),
        "Sp. Defense": (row.get("Sp.Defense") or "").strip(),
        "Speed": (row.get("Speed") or "").strip(),
    }

    parts = []
    if number:
        parts.append(f"Entry: {number}")
    if name:
        parts.append(f"Name: {name}")
    if type_1 or type_2:
        types = ", ".join(part for part in [type_1, type_2] if part)
        parts.append(f"Type: {types}")
    for key, value in stats.items():
        if value:
            parts.append(f"{key}: {value}")

    return " | ".join(parts)


def chunk_by_entry():
    """Chunk the Pokédex by full Pokémon record using CSV parsing."""
    chunks = []
    data_path = Path(__file__).parent / "pokedex.txt"
    with data_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row is None or not any((value or "").strip() for value in row.values()):
                continue
            if not (row.get("Name") or "").strip():
                continue
            chunks.append(_format_pokemon_entry(row))
    return chunks


def name_from_chunk(chunk):
    for part in chunk.split(" | "):
        if part.startswith("Name: "):
            return part.removeprefix("Name: ")
    raise ValueError(f"Pokémon entry is missing its name: {chunk}")


def build_name_matcher(chunks):
    names = [name_from_chunk(chunk) for chunk in chunks]
    known_terms = {
        part.casefold()
        for chunk in chunks
        for part in _WORD_PATTERN.findall(chunk)
    }
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    name_embeddings = vectorizer.fit_transform(names)
    return names, known_terms, vectorizer, name_embeddings


def correct_name_typos(query, names, known_terms, vectorizer, name_embeddings):
    matches = list(_WORD_PATTERN.finditer(query))
    corrected_query = query

    for match in reversed(matches):
        token = match.group()
        if len(token) < 4 or token.casefold() in known_terms:
            continue

        token_embedding = vectorizer.transform([token])
        similarities = cosine_similarity(token_embedding, name_embeddings)[0]
        best_index = int(similarities.argmax())
        if similarities[best_index] >= NAME_TYPO_MATCH_THRESHOLD:
            corrected_query = (
                corrected_query[:match.start()]
                + names[best_index]
                + corrected_query[match.end():]
            )

    return corrected_query


def retrieve_query(
    query,
    vectorizer,
    chunk_embeddings,
    names,
    known_terms,
    name_vectorizer,
    name_embeddings,
    top_k=5,
    correct_typos=True,
):
    retrieval_query = query
    if correct_typos:
        retrieval_query = correct_name_typos(
            query, names, known_terms, name_vectorizer, name_embeddings
        )
    query_embedding = embed_query(retrieval_query, vectorizer)
    top_indices, similarities = retrieve_similar_chunks(
        query_embedding, chunk_embeddings, top_k=top_k
    )
    return retrieval_query, top_indices, similarities


def build_vectorizer():
    return TfidfVectorizer(stop_words="english", ngram_range=(1, 2))


def embed_chunks(chunks, vectorizer=None):
    if vectorizer is None:
        vectorizer = build_vectorizer()
    return vectorizer.fit_transform(chunks)


def embed_query(query, vectorizer):
    return vectorizer.transform([query])


def retrieve_similar_chunks(query_embedding, chunk_embeddings, top_k=5):
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return top_indices, similarities[top_indices]
