from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Any
from rerankers import Reranker, Document

app = FastAPI()

ranker = Reranker("unicamp-dl/InRanker-base", model_type = "t5")

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
            text=doc["text"] if doc.get("text") else "",
            doc_id=doc["doc_id"],
            metadata=doc.get("metadata", {})
        )
        for doc in request.documents
    ]

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
