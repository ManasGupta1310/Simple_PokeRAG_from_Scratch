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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
    with open("pokedex.txt", "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row is None or not any((value or "").strip() for value in row.values()):
                continue
            chunks.append(_format_pokemon_entry(row))
    return chunks


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
