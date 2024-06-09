from fastapi import FastAPI, HTTPException
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
    top_k: int = Field(..., description="The number of top results to retrieve from similarity search")
    embedding_model: str = Field(..., description="The embedding model used for similarity search")

class FetchedVector(BaseModel):
    id: str
    metadata: dict
    values: list

@app.post("/search")
def search_documents(search_params: SearchParams):
    try:
        # Initialize Pinecone client
        pinecone = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

        # Connect to the index
        index = pinecone.Index(search_params.index_name)

        fetched_vector = FetchedVector(
            id=search_params.profile_id,
            metadata=fetch_response.vectors[search_params.profile_id].metadata,
            values=fetch_response.vectors[search_params.profile_id].values,
        )

        # Perform similarity search using the fetched vector
        search_response = index.query(
            vector=fetched_vector.values,
            top_k=search_params.top_k,
            include_metadata=True,
            namespace=search_params.search_namespace,
        )

        # Prepare the search results
        search_results = []
        for result in search_response.matches:
            search_results.append({
                'id': result.id,
                'score': result.score,
                'metadata': result.metadata,
            })

        return {
            'fetched_vector': fetched_vector,
            'search_results': search_results,
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
