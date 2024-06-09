from fastapi import FastAPI
from pydantic import BaseModel, Field
from pinecone import Pinecone
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

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

@app.post("/search")
def search(search_params: SearchParams):
    # Initialize Pinecone client
    pinecone = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

    # Connect to the index
    index = pinecone.Index(search_params.index_name)

    # Fetch the vector and metadata based on the given ID and namespace
    fetch_response = index.fetch(
        ids=[search_params.profile_id],
        namespace=search_params.query_namespace
    )

    # Extract the vector and metadata from the fetch response
    vector = fetch_response['vectors'][search_params.profile_id]['values']
    metadata = fetch_response['vectors'][search_params.profile_id].get('metadata', {})

    # Add the vector and metadata to the search_params
    search_params_dict = search_params.dict()
    search_params_dict['vector'] = vector
    search_params_dict['metadata'] = metadata

    return search_params_dict
