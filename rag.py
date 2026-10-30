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

def fixed_size_chunking():
    chunk_size = 1000  # Define the chunk size here
    with open("pokedex.txt", "r", encoding="utf-8") as f:
        document = f.read()
    chunks = []
    # We split the document in chunks of chunk_size with overlap to ensure context continuity
    overlap = chunk_size // 2
    for i in range(0, len(document), chunk_size - overlap):
        chunks.append(document[i:i + chunk_size])
    return chunks


def build_vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(stop_words="english", ngram_range=(1, 2))


def embed_chunks(chunks, vectorizer=None):
    if vectorizer is None:
        vectorizer = build_vectorizer()
    return vectorizer.fit_transform(chunks)


def embed_query(query, vectorizer):
    return vectorizer.transform([query])


def retrieve_similar_chunks(query_embedding, chunk_embeddings, top_k=5):
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return top_indices, similarities[top_indices]
