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
    index_host: str = Field(..., description="The host of the Pinecone index")
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

    # Find your index host by calling describe_index through the Pinecone web console
    index = pinecone.Index(host=search_params.index_host)

    # Query the index based on the given vector ID and return both metadata and vectors
    query_response = index.query(
        id=search_params.profile_id,
        namespace=search_params.query_namespace,
        include_metadata=True,
        top_k=search_params.similarity_top_k,
        include_values=True
    )

    # Extract the query vector and metadata from the query response
    query_vector = query_response['matches'][0]['values']
    query_metadata = query_response['matches'][0]['metadata']

    # Add the query vector and metadata to the search_params
    search_params_dict = search_params.dict()
    search_params_dict['query_vector'] = query_vector
    search_params_dict['query_metadata'] = query_metadata

    return search_params_dict
