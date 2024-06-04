from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from typing import List
import json
from pprint import pprint
from pinecone import Pinecone
from rerankers import Reranker
import logging
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Retrieve the API keys from environment variables
cohere_api_key = os.getenv('COHERE_API_KEY')
mixedbread_api_key = os.getenv('MIXEDBREAD_API_KEY')
jina_api_key = os.getenv('JINA_API_KEY')

# Map reranker names to their corresponding API keys
reranker_api_keys = {
    'GPT-4': jina_api_key,
    'Jina Rank': jina_api_key,
    'Cohere': cohere_api_key,
    'VoyageAI': cohere_api_key,
    'Mixedbread': mixedbread_api_key,
    'ColbertV2': mixedbread_api_key,
    'Opus 3': mixedbread_api_key
}
# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

def hybrid_score_norm(dense, sparse, alpha: float):
    """Hybrid score using a convex combination

    alpha * dense + (1 - alpha) * sparse

    Args:
        dense: Array of floats representing
        sparse: a dict of `indices` and `values`
        alpha: scale between 0 and 1
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    hs = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    return [v * alpha for v in dense], hs

class Document(BaseModel):
    doc_id: int = Field(..., description="The unique ID of the document")
    text: str = Field(..., description="The text of the document")

class RerankRequest(BaseModel):
    query: str = Field(..., description="The query for reranking")

class RerankResponse(BaseModel):
    reranked_documents: List[Document] = Field(..., description="The reranked documents")

class SearchParams(BaseModel):
    profile_id: str = Field(..., description="The profile ID to fetch the vector for")
    index_name: str = Field(..., description="The name of the Pinecone index")
    query_namespace: str = Field(..., description="The namespace for the query vector")
    search_namespace: str = Field(..., description="The namespace for the search vectors")
    alpha: float = Field(..., description="The weight for the dense vector in the hybrid score")
    reranker: str = Field(..., description="The JSON-encoded reranker configuration")
    top_k: int = Field(..., description="The number of top results to retrieve")
    embedding_model: str = Field(..., description="The embedding model used for similarity search")

class CustomJSONEncoder(json.JSONEncoder):
    def encode(self, o):
        encoded = super().encode(o)
        return encoded.replace(',', '\054')

@app.post("/rerank", response_model=RerankResponse)
async def rerank(rerank_request: RerankRequest, search_params: SearchParams, response: Response):
    # Initialize Pinecone client
    pc = Pinecone(api_key="bb2dea00-df61-404e-9f29-5e40faee47c4")

    # Extract the search parameters from the request body
    profile_id = search_params.profile_id
    index_name = search_params.index_name
    query_namespace = search_params.query_namespace
    search_namespace = search_params.search_namespace
    alpha = search_params.alpha
    reranker_name = search_params.reranker

    # Initialize the reranker based on the reranker name
    if reranker_name == "GPT-4":
        ranker = Reranker("jina", api_key=reranker_api_keys["GPT-4"])
    elif reranker_name == "Jina Rank":
        ranker = Reranker("jina", api_key=reranker_api_keys["Jina Rank"])
    elif reranker_name == "Cohere":
        ranker = Reranker("cohere", api_key=reranker_api_keys["Cohere"])
    elif reranker_name == "VoyageAI":
        ranker = Reranker("voyage", api_key=reranker_api_keys["VoyageAI"])
    elif reranker_name == "Mixedbread":
        ranker = Reranker("mixedbread.ai", api_key=reranker_api_keys["Mixedbread"])
    elif reranker_name == "ColbertV2":
        ranker = Reranker("mixedbread.ai", api_key=reranker_api_keys["ColbertV2"])
    elif reranker_name == "Opus 3":
        ranker = Reranker("mixedbread.ai", api_key=reranker_api_keys["Opus 3"])
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported reranker: {reranker_name}")

    # Connect to the index
    index = pc.Index(index_name)

    # Fetch the vector with the specified profile_id from the men's namespace
    query_response = index.fetch(
        ids=[profile_id],
        namespace=query_namespace
    )

    # Check if the query response contains the requested vector
    if profile_id not in query_response.vectors:
        raise HTTPException(status_code=404, detail=f"No vector found with profile_id {profile_id} in namespace {query_namespace}")

    # Extract the dense and sparse vectors from the query response
    query_vector = query_response.vectors[profile_id].values
    query_sparse_vector = {
        'indices': query_response.vectors[profile_id].sparse_values.indices,
        'values': query_response.vectors[profile_id].sparse_values.values
    }

    # Apply hybrid scoring to the query vectors
    query_vector_hybrid, query_sparse_vector_hybrid = hybrid_score_norm(query_vector, query_sparse_vector, alpha)

    # Perform a similarity search in the women's namespace using the hybrid vectors
    search_response = index.query(
        namespace=search_namespace,
        top_k=30,
        vector=query_vector_hybrid,
        sparse_vector=query_sparse_vector_hybrid,
        include_values=True,
        include_metadata=True
    )

    # Create a list to store the matches with hybrid scores and metadata
    matches_with_hybrid_scores = []

    for match in search_response.matches:
        # Check if sparse_values are present in the match
        if match.sparse_values:
            # Calculate the sparse score manually
            sparse_score = sum(match.sparse_values.values)
            
            # Calculate the hybrid score
            hybrid_score = alpha * match.score + (1 - alpha) * sparse_score
            
            # Create a dictionary with the required metadata fields
            match_data = {
                'profile_id': match.metadata['profile_id'],
                'first_name': match.metadata['first_name'],
                'rerank_chunk': match.metadata['bio'] + match.metadata['nuance_chunk'] + match.metadata['psych_eval'],
                'hybrid_score': hybrid_score
            }
            
            # Add the match data to the list
            matches_with_hybrid_scores.append(match_data)
    
    # Sort the matches based on the hybrid scores in descending order
    matches_with_hybrid_scores.sort(key=lambda x: x['hybrid_score'], reverse=True)

    # Prepare the documents for reranking
    documents = [
        Document(
            doc_id=match['profile_id'],
            text=match['rerank_chunk']
        )
        for match in matches_with_hybrid_scores
    ]

    # Rerank the documents using the provided query
    reranked_results = ranker.rank(
        query=rerank_request.query,
        docs=[doc.text for doc in documents],
        doc_ids=[doc.doc_id for doc in documents]
    )

    # Get the top-k reranked results
    top_k = 20  # Set the desired value for top-k
    top_reranked_results = reranked_results.top_k(top_k)

     # Prepare the top-k reranked documents for the response
    top_reranked_documents = [
        Document(
            doc_id=result.doc_id,
            text=result.text
        )
        for result in top_reranked_results
    ]

    # Set the cookie in the response headers
    cookie_value = json.dumps({
        'similarity-net-size': search_params.top_k,
        'embedding-model': search_params.embedding_model,
        'hybrid-alpha-mix': search_params.alpha,
        'reranker-model': search_params.reranker,
        'final-results-size': len(top_reranked_documents)
    })

    response.set_cookie(
        key="kiss_settings",
        value=cookie_value,
        max_age=3600,
        path="/",
        domain="kiss-qa.kelleher-international.com",
        secure=False,
        httponly=False
    )

    # Log the cookie value
    logger.debug(f"Cookie value: {cookie_value}")

    # Log the response headers
    logger.debug(f"Response headers: {response.headers}")

    return RerankResponse(reranked_documents=top_reranked_documents)
