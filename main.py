import logging
from fastapi import FastAPI, Request, HTTPException, Response
from pydantic import BaseModel, Field
from typing import List
import json
from pprint import pprint
from rerankers import Reranker
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastapi")

# Retrieve the API keys from environment variables
cohere_api_key = os.getenv('COHERE_API_KEY')
mixedbread_api_key = os.getenv('MIXEDBREAD_API_KEY')
jina_api_key = os.getenv('JINA_API_KEY')

# Map reranker names to their corresponding API keys
reranker_api_keys = {
    'jina': jina_api_key,
    'cohere': cohere_api_key,
    'mixedbread.ai': mixedbread_api_key
}
ranker = Reranker("jina", model_type="api", api_key=jina_api_key)

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
