from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag import (
        build_vectorizer,
    chunk_by_entry,
    embed_chunks,
    embed_query,
        retrieve_similar_chunks,
    )
from groq import Groq
from dotenv import load_dotenv
import os

app = FastAPI()
chunks = chunk_by_entry()
vectorizer = build_vectorizer()
chunk_embeddings = embed_chunks(chunks, vectorizer)
load_dotenv()
client = Groq(api_key=os.getenv("API_KEY"))
class SearchRequest(BaseModel):
    query: str


@app.get("/")
def read_root():
    return {"data": "Welcome to a simple RAG project from scratch for pokemon"}


@app.post("/search")
def search(payload: Optional[SearchRequest] = None, query: Optional[str] = None):
    user_query = query
    if payload is not None:
        user_query = payload.query

    if user_query is None or not user_query.strip():
        raise HTTPException(status_code=400, detail="A non-empty search query is required.")

    # query is the question given by the user
    system_prompt = (
        "You are a helpful assistant that provides accurate and concise answers based on the retrieved documents. "
        "If the query is not related to the retrieved documents, politely cite that the information was not available "
        "on the documents and hence you cant help"
    )

    query_embedding = embed_query(user_query, vectorizer)
    top_indices, _ = retrieve_similar_chunks(query_embedding, chunk_embeddings)
    retrieved_chunks = [chunks[i] for i in top_indices]
    context = "\n\n---\n\n".join(retrieved_chunks)

    model_candidates = [os.getenv("MODEL", "openai/gpt-oss-120b")]
    last_error = None

    for model_name in dict.fromkeys(model_candidates):
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
            return {"answer": response.choices[0].message.content}
        except Exception as exc:
            last_error = exc

    raise HTTPException(
        status_code=502,
        detail=f"Failed to generate a response from the configured model. Last error: {last_error}",
    )
