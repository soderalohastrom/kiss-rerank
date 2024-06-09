from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

class FetchParams(BaseModel):
    ids: str
    index: str
    namespace: str

class FetchedVector(BaseModel):
    id: str
    metadata: dict
    values: list

@app.post("/fetch")
def fetch_vectors(fetch_params: FetchParams):
    try:
        # Initialize Pinecone client
        pinecone = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

        # Connect to the index
        index = pinecone.Index(fetch_params.index)

        # Fetch the vector with the specified ID from the given namespace
        fetch_response = index.fetch(
            ids=[fetch_params.ids],
            namespace=fetch_params.namespace
        )

        # Check if the fetch response contains the requested vector
        if fetch_params.ids not in fetch_response.vectors:
            raise HTTPException(status_code=404, detail=f"No vector found with ID {fetch_params.ids} in namespace {fetch_params.namespace}")

        # Create a FetchedVector instance with the necessary data
        fetched_vector = FetchedVector(
            id=fetch_params.ids,
            metadata=fetch_response.vectors[fetch_params.ids].metadata,
            values=fetch_response.vectors[fetch_params.ids].values,
        )

        return fetched_vector
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
