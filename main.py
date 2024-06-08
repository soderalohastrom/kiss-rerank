import logging
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List
import json
from pprint import pprint
from pinecone import Pinecone
from rerankers import Reranker
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = FastAPI(response_class=JSONResponse)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastapi")


# Retrieve the API keys from environment variables
cohere_api_key = os.getenv('COHERE_API_KEY')
mixedbread_api_key = os.getenv('MIXEDBREAD_API_KEY')
jina_api_key = os.getenv('JINA_API_KEY')

# Log the retrieved API keys
logger.debug(f"Cohere API Key: {cohere_api_key}")
logger.debug(f"Mixedbread API Key: {mixedbread_api_key}")
logger.debug(f"Jina API Key: {jina_api_key}")

# Check if any API key is missing
if not cohere_api_key:
    logger.debug("Cohere API Key is missing")
if not mixedbread_api_key:
    logger.debug("Mixedbread API Key is missing")
if not jina_api_key:
    logger.debug("Jina API Key is missing")

# Map reranker names to their corresponding API keys
reranker_api_keys = {
    'jina': jina_api_key,
    'cohere': cohere_api_key,
    'mixedbread.ai': mixedbread_api_key
}

class Document(BaseModel):
    doc_id: str
    text: str

class RerankResponse(BaseModel):
    reranked_documents: List[Document]

class SearchParams(BaseModel):
    profile_id: str = Field(..., description="The profile ID to fetch the vector for")
    index_name: str = Field(..., description="The name of the Pinecone index")
    query_namespace: str = Field(..., description="The namespace for the query vector")
    search_namespace: str = Field(..., description="The namespace for the search vectors")
    alpha: float = Field(..., description="The weight for the dense vector in the hybrid score")
    reranker: dict = Field(..., description="The reranker configuration")
    similarity_top_k: int = Field(..., description="The number of top results to retrieve from similarity search")
    rerank_top_k: int = Field(..., description="The number of top results to return after reranking")
    embedding_model: str = Field(..., description="The embedding model used for similarity search")


def calculate_hybrid_score(match, alpha):
    """Calculates the hybrid score for a match."""
    if match.sparse_values:
        sparse_score = sum(match.sparse_values.values())
        hybrid_score = alpha * match.score + (1 - alpha) * sparse_score
    else:
        hybrid_score = match.score
    return hybrid_score


@app.post("/rerank", response_model=RerankResponse)
def rerank(search_params: SearchParams):
    logger.debug(f"Received search parameters: {search_params}")
    
    # Extract the search parameters from the search_params object
    profile_id = search_params.profile_id
    index_name = search_params.index_name
    query_namespace = search_params.query_namespace
    search_namespace = search_params.search_namespace
    alpha = search_params.alpha
    reranker_config = search_params.reranker
    similarity_top_k = search_params.similarity_top_k
    rerank_top_k = search_params.rerank_top_k
    embedding_model = search_params.embedding_model

    logger.debug(f"profile_id: {profile_id}")
    logger.debug(f"index_name: {index_name}")
    logger.debug(f"query_namespace: {query_namespace}")
    logger.debug(f"search_namespace: {search_namespace}")
    logger.debug(f"alpha: {alpha}")
    logger.debug(f"reranker_config: {reranker_config}")
    logger.debug(f"similarity_top_k: {similarity_top_k}")
    logger.debug(f"rerank_top_k: {rerank_top_k}")
    logger.debug(f"embedding_model: {embedding_model}")

    # Initialize Pinecone client
    pinecone = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

    # Get the Pinecone index
    index = pinecone.Index(index_name)

    # Validate profile_id
    if not profile_id:
        logger.debug("profile_id is missing!")
        raise HTTPException(status_code=422, detail="profile_id is required")


    reranker_name = reranker_config.get("name")
    
    if reranker_name in reranker_api_keys:
        reranker_api_key = reranker_api_keys[reranker_name]
        ranker = Reranker(reranker_name, api_key=reranker_api_key)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported reranker: {reranker_name}")

    # Fetch the query vector from Pinecone
    query_response = index.fetch(
        ids=[profile_id],
        namespace=query_namespace
    )
    query_vector = query_response['vectors'][profile_id]['values']
    query_metadata = query_response['vectors'][profile_id]['metadata']

    logger.debug(f"Fetched query vector for profile_id: {profile_id}")
    logger.debug(f"Query vector length: {len(query_vector)}")
    logger.debug(f"Query metadata: {query_metadata}")

    # Create the rerank_chunk from the query metadata
    rerank_chunk = query_metadata['bio'] + query_metadata['nuance_chunk'] + query_metadata['psych_eval']

    logger.debug(f"Rerank chunk: {rerank_chunk}")

    # Perform the similarity search
    search_response = index.query(
        vector=query_vector,
        top_k=similarity_top_k,
        include_metadata=True,
        namespace=search_namespace
    )

    # Calculate hybrid scores for matches
    matches_with_hybrid_scores = []
    for match in search_response.matches:
        hybrid_score = calculate_hybrid_score(match, alpha)
        profile_id = match.id
        matches_with_hybrid_scores.append({
            'id': match.id,
            'profile_id': match.metadata['profile_id'],
            'first_name': match.metadata['first_name'],
            'rerank_chunk': match.metadata['bio'] + match.metadata['nuance_chunk'] + match.metadata['psych_eval'],
            'hybrid_score': hybrid_score
        })

    # Sort matches by hybrid score
    matches_with_hybrid_scores.sort(key=lambda x: x['hybrid_score'], reverse=True)

    documents = [Document(
        doc_id=match['profile_id'],
        text=match['rerank_chunk']
    ) for match in matches_with_hybrid_scores]

    logger.debug("Documents to rerank:")
    for doc in documents:
        logger.debug(f"  - {doc.doc_id}: {doc.text[:50]}...")
    
    # Perform the reranking
    reranked_results = ranker.rank(
        query=rerank_chunk,
        docs=documents,
        top_k=rerank_top_k
    )

    reranked_documents = [Document(doc_id=doc.doc_id, text=doc.text)
                          for doc in reranked_results.results]

    logger.debug("Reranked documents:")
    for doc in reranked_documents:
        logger.debug(f"  - {doc.doc_id}: {doc.text[:50]}...")

    return RerankResponse(reranked_documents=reranked_documents)
