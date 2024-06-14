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

# ranker = Reranker("cohere", lang='en', api_key='n1ytpDT5S9jVqY1abqvqoD6flMgo8M25UJce9fLy')
# ranker = Reranker('mixedbread-ai/mxbai-rerank-large-v1', model_type='cross-encoder')
ranker = Reranker('cross-encoder', model_type='cross-encoder')

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


class Document(BaseModel):
    doc_id: int = Field(..., description="The unique ID of the document")
    text: str = Field(..., description="The text of the document")

class RerankRequest(BaseModel):
    query: str = Field(..., description="The query to rank the documents against")
    documents: List[Document] = Field(..., description="The documents to be reranked")

class RerankResponse(BaseModel):
    reranked_documents: List[Document] = Field(..., description="The reranked documents")

@app.post("/rerank", response_model=RerankResponse)
async def rerank_documents(rerank_request: RerankRequest):
    docs = [doc.text for doc in rerank_request.documents]
    doc_ids = [doc.doc_id for doc in rerank_request.documents]
    reranked_results = ranker.rank(
        query=rerank_request.query,
        docs=docs,
        doc_ids=doc_ids
    )
    reranked_documents = [
        Document(doc_id=result.doc_id, text=result.text)
        for result in reranked_results.results
    ]
    return RerankResponse(reranked_documents=reranked_documents)
