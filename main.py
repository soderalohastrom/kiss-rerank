    # Get the top-k reranked results
    top_k = 5  # Set the desired value for top-k
    top_reranked_results = reranked_results.top_k(top_k)

    # Prepare the top-k reranked documents for the response
    top_reranked_documents = [
        Document(
            doc_id=result.doc_id,
            text=result.text,
            index=result.index,
            relevance_score=result.score
        )
        for result in top_reranked_results
    ]

    return RerankResponse(reranked_documents=top_reranked_documents)
