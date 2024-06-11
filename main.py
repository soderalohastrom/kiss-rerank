from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Any
from rerankers import Reranker, Document


ranker = Reranker("jina", api_key = 'jina_bf5ea4fa09d94000b6ac739ac8c03e6abDa7EGRVPAcmEmUI4CV1rv9efZnk')

app = FastAPI()

class RerankRequest(BaseModel):
    query: str
    documents: List[dict]

class RerankResponse(BaseModel):
    reranked_documents: List[dict]

@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    query = request.query
    documents = [
        Document(
            text=doc["text"],
            doc_id=doc["doc_id"],
            metadata=doc.get("metadata", {})
        )
        for doc in request.documents
    ]

    ranker = Reranker()
    results = ranker.rank(query=query, docs=documents)

    reranked_documents = [
        {
            "doc_id": result.document.doc_id,
            "text": result.document.text,
            "metadata": result.document.metadata,
            "score": result.score,
        }
        for result in results
    ]

    return {"reranked_documents": reranked_documents}
