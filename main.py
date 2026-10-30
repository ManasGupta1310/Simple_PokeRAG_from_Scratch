
import os
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from groq import Groq
from rag import (
    build_name_matcher,
    build_vectorizer,
    chunk_by_entry,
    embed_chunks,
    name_from_chunk,
    retrieve_query,
)

logger = logging.getLogger(__name__)

load_dotenv()

try:
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.10"))
except ValueError as exc:
    raise RuntimeError("SIMILARITY_THRESHOLD must be a number between 0 and 1.") from exc

if not 0 <= SIMILARITY_THRESHOLD <= 1:
    raise RuntimeError("SIMILARITY_THRESHOLD must be between 0 and 1.")

# Init the FastAPI application
app = FastAPI()

# Load the chunks and build the vectorizer and embeddings for the RAG system
chunks = chunk_by_entry()
pokemon_names, name_vectorizer, name_embeddings = build_name_matcher(chunks)
vectorizer = build_vectorizer()
chunk_embeddings = embed_chunks(chunks, vectorizer)

# Load the environment variables and initialize the client with the API key
api_key = os.getenv("API_KEY")
if not api_key:
    raise RuntimeError("Set API_KEY in .env")
client = Groq(api_key=api_key)

# Define the request model for the search endpoint
class SearchRequest(BaseModel):
    query: str = Field(min_length=1)

# Define the root endpoint for the FastAPI application
@app.get("/")
def read_root():
    return {"data": "Welcome to a simple RAG project from scratch for pokemon"}

# Define the search endpoint for the FastAPI application
@app.post("/search")
def search(payload: SearchRequest):
    user_query = payload.query
    if not user_query.strip():
        raise HTTPException(
            status_code=400,
            detail="A non-empty search query is required.",
        )

    # query is the question given by the user
    system_prompt = (
        "You are a helpful assistant that provides accurate and concise answers based on the retrieved documents. "
        "If the query is not related to the retrieved documents, politely cite that the information was not available "
        "on the documents and hence you cant help"
    )

    _, top_indices, similarities = retrieve_query(
        user_query,
        vectorizer,
        chunk_embeddings,
        pokemon_names,
        name_vectorizer,
        name_embeddings,
    )
    best_similarity = float(similarities[0]) if len(similarities) else 0.0
    sources = []
    for index, similarity in zip(top_indices, similarities):
        sources.append(
            {
                "name": name_from_chunk(chunks[index]),
                "similarity": float(similarity),
            }
        )

    if best_similarity == 0 or best_similarity < SIMILARITY_THRESHOLD:
        return {
            "answer": (
                "I couldn't find relevant information in the Pokédex data, "
                "so I can't answer that from the available content."
            ),
            "sources": sources,
        }

    retrieved_chunks = [chunks[i] for i in top_indices]
    context = "\n\n---\n\n".join(retrieved_chunks)

    model_name = os.getenv("MODEL", "openai/gpt-oss-120b")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Question: {user_query}\n\nRelevant context:\n{context}",
                },
            ],
            temperature=0.2,
            max_tokens=2048,
            stream=False,
        )
    except Exception as exc:
        logger.exception("Groq completion failed")
        raise HTTPException(
            status_code=502,
            detail="Failed to generate a response from the configured model.",
        ) from exc

    return {"answer": response.choices[0].message.content, "sources": sources}
