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
voyage_api_key = os.getenv('VOYAGEAI_API_KEY')

# Map reranker names to their corresponding API keys
reranker_api_keys = {
    'jina': jina_api_key,
    'cohere': cohere_api_key,
    'mixedbread.ai': mixedbread_api_key,
    'voyage': voyage_api_key
}

# Initialize the ranker with the desired API key
# ranker = Reranker("cohere", model_type="api", api_key=cohere_api_key)
# ranker = Reranker("jina", model_type="api", api_key=jina_api_key)
# ranker = Reranker("mixedbread.ai", model_type="api", api_key=mixedbread_api_key)
ranker = Reranker("voyage", model_type="api", api_key=voyage_api_key)

class Document(BaseModel):
    text: str
    doc_id: str
    metadata: dict = Field(default_factory=dict)

class RerankRequest(BaseModel):
    query: str
    documents: List[dict]

class RerankResponse(BaseModel):
    reranked_documents: List[dict]

@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    try:
        query = request.query
        documents = [
            Document(
                text=doc.get("text", ""),
                doc_id=doc["doc_id"],
                metadata=doc.get("metadata", {})
            )
            for doc in request.documents
        ]

        results = ranker.rank(query=query, docs=documents)

        # Log the structure of the results for debugging
        logger.info(f"Ranker results structure: {type(results)}")
        logger.info(f"Ranker results content: {results}")

        # Handle different possible result structures
        if isinstance(results, list):
            reranked_documents = [
                {
                    "doc_id": getattr(result.document, 'doc_id', None) or result.get('doc_id'),
                    "text": getattr(result.document, 'text', None) or result.get('text', ''),
                    "metadata": getattr(result.document, 'metadata', None) or result.get('metadata', {}),
                    "score": getattr(result, 'score', None) or result.get('score', 0),
                }
                for result in results
            ]
        elif isinstance(results, dict) and 'results' in results:
            reranked_documents = [
                {
                    "doc_id": doc.get('doc_id'),
                    "text": doc.get('text', ''),
                    "metadata": doc.get('metadata', {}),
                    "score": doc.get('score', 0),
                }
                for doc in results['results']
            ]
        else:
            raise ValueError("Unexpected result structure from ranker")

        return {"reranked_documents": reranked_documents}

    except Exception as e:
        logger.error(f"Error in reranking: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
