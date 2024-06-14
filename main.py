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

app = FastAPI()

# RankLLM with default GPT (GPT-4o)
ranker = Reranker("rankllm", model_type="rankllm", api_key = os.getenv('OPENAI_API_KEY'))

# # RankLLM with specified GPT models
# ranker = Reranker('gpt-4-turbo', model_type="rankllm", api_key = os.getenv('OPENAI_API_KEY'))

# # EXPERIMENTAL: RankLLM with RankZephyr
# ranker = Reranker("rankllm", api_key = os.getenv('OPENAI_API_KEY'))

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
