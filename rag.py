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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def fixed_size_chunking():
    """Chunk the Pokédex by Pokémon entry instead of slicing raw text arbitrarily.

    The source file is a CSV with one Pokémon per row, so splitting by entry preserves
    the full record for each Pokémon and keeps retrieval context grounded in a whole
    entity rather than cutting through it mid-entry.
    """
    with open("pokedex.txt", "r", encoding="utf-8") as f:
        rows = [line.strip() for line in f if line.strip()]

    if not rows:
        return []

    header = rows[0]
    chunks = []
    for row in rows[1:]:
        chunks.append(f"{header}\n{row}")

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
