from app.services.vector_store import VectorStore
from app.services.embedding_service import EmbeddingService
from app.models import RetrievedChunk, ChunkMetadata
from app.config import get_settings
from loguru import logger


class RetrievalService:
    def __init__(self):
        self.settings = get_settings()
        self.vector_store = VectorStore()
        self.embedding_service = EmbeddingService()
    
    def retrieve(
        self,
        query: str,
        top_k: int = None
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks for query"""
        
        if top_k is None:
            top_k = self.settings.top_k_results
        
        # Generate query embedding
        query_vector = self.embedding_service.embed_text(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k
        )
        
        # Convert to RetrievedChunk models
        retrieved_chunks = []
        for result in results:
            chunk = RetrievedChunk(
                content=result["content"],
                metadata=ChunkMetadata(**result["metadata"]),
                score=result["score"]
            )
            retrieved_chunks.append(chunk)
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query")
        return retrieved_chunks
