from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Any
from rerankers import Reranker, Document
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

ranker = Reranker("flashrank")
# ranker = Reranker('ms-marco-MiniLM-L-12-v2', model_type='flashrank')
# ranker = Reranker('rank_zephyr_7b_v1_full', model_type='flashrank')

class RerankRequest(BaseModel):
    query: str
    documents: List[dict]

class RerankResponse(BaseModel):
    reranked_documents: List[dict]

@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    query = request.query
    documents = [
        Document(
            text=doc["text"] if doc.get("text") else "",
            doc_id=doc["doc_id"],
            metadata=doc.get("metadata", {})
        )
        for doc in request.documents
    ]

    results = ranker.rank(query=query, docs=documents)

    reranked_documents = [
        {
            "doc_id": result.document.doc_id,
            "text": result.document.text,
            "metadata": result.document.metadata,
            "score": result.score,
        }
        for result in results
    ]

    return {"reranked_documents": reranked_documents}
