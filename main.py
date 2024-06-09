from fastapi import FastAPI
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

        return fetch_response
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
