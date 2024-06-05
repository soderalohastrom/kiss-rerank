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

class Document(BaseModel):
    doc_id: int = Field(..., description="The unique ID of the document")
    text: str = Field(..., description="The text of the document")

class RerankResponse(BaseModel):
    reranked_documents: List[Document] = Field(..., description="The reranked documents")

class SearchParams(BaseModel):
    profile_id: str = Field(..., description="The profile ID to fetch the vector for")
    index_name: str = Field(..., description="The name of the Pinecone index")
    query_namespace: str = Field(..., description="The namespace for the query vector")
    search_namespace: str = Field(..., description="The namespace for the search vectors")
    alpha: float = Field(..., description="The weight for the dense vector in the hybrid score")
    reranker: str = Field(..., description="The JSON-encoded reranker configuration")
    similarity_top_k: int = Field(..., description="The number of top results to retrieve from similarity search")
    rerank_top_k: int = Field(..., description="The number of top results to return after reranking")
    embedding_model: str = Field(..., description="The embedding model used for similarity search")

@app.post("/rerank", response_model=RerankResponse)
async def rerank(search_params: SearchParams, response: Response):
    # Initialize Pinecone client
    pc = Pinecone(api_key="bb2dea00-df61-404e-9f29-5e40faee47c4")

    # Extract the search parameters from the request body
    profile_id = search_params.profile_id
    index_name = search_params.index_name
    query_namespace = search_params.query_namespace
    search_namespace = search_params.search_namespace
    alpha = search_params.alpha
    reranker_name = search_params.reranker
    similarity_top_k = search_params.similarity_top_k
    rerank_top_k = search_params.rerank_top_k
    embedding_model = search_params.embedding_model

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

    # Fetch the vector and metadata for the given profile_id
    query_result = index.fetch(
        ids=[profile_id],
        namespace=query_namespace
    )

    if query_result.vectors:
        query_vector = query_result.vectors[0]
        query_metadata = query_result.metadata[0]
        rerank_chunk = query_metadata['bio'] + query_metadata['nuance_chunk'] + query_metadata['psych_eval']
    else:
        raise HTTPException(status_code=404, detail=f"Profile with ID {profile_id} not found")

    # Perform the similarity search
    search_results = index.query(
        vector=query_vector,
        top_k=similarity_top_k,
        include_metadata=True,
        namespace=search_namespace
    )

    # Prepare the documents for reranking
    documents = [
        Document(
            doc_id=match.id,
            text=match.metadata['bio'] + match.metadata['nuance_chunk'] + match.metadata['psych_eval']
        )
        for match in search_results.matches
    ]

    # Perform the reranking
    reranked_results = ranker.rerank(
        query=rerank_chunk,
        documents=documents,
        top_k=rerank_top_k
    )

    # Prepare the response
    reranked_documents = [
        Document(
            doc_id=doc.doc_id,
            text=doc.text
        )
        for doc in reranked_results
    ]

    return RerankResponse(reranked_documents=reranked_documents)
    
