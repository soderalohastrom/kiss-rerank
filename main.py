from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pinecone import Pinecone
from fastapi import Response

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

class SearchResponse(BaseModel):
    search_params: dict
    vector: list
    metadata: dict

@app.post("/search", response_model=SearchResponse)
async def search_documents(search_params: SearchParams, response: Response):
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key="bb2dea00-df61-404e-9f29-5e40faee47c4")

        # Extract the search parameters from the request body
        profile_id = search_params.profile_id
        index_name = search_params.index_name
        query_namespace = search_params.query_namespace
        search_namespace = search_params.search_namespace

        # Connect to the index
        index = pc.Index(index_name)

        # Fetch the vector with the specified profile_id from the query namespace
        query_response = index.fetch(
            ids=[profile_id],
            namespace=query_namespace
        )

        # Check if the query response contains the requested vector
        if profile_id not in query_response.vectors:
            raise HTTPException(status_code=404, detail=f"No vector found with profile_id {profile_id} in namespace {query_namespace}")

        # Extract the vector and metadata from the query response
        vector = query_response.vectors[profile_id].values
        metadata = query_response.vectors[profile_id].metadata

        # Add the vector and metadata to the search_params
        search_params_dict = search_params.dict()
        search_params_dict['vector'] = vector
        search_params_dict['metadata'] = metadata

        return SearchResponse(search_params=search_params_dict, vector=vector, metadata=metadata)
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
