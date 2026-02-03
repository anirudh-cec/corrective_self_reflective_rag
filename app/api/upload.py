from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models import UploadResponse
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStore
from app.services.embedding_service import EmbeddingService
from app.config import get_settings
from pathlib import Path
import shutil
from loguru import logger
from uuid import uuid4

router = APIRouter(prefix="/upload", tags=["upload"])

document_processor = DocumentProcessor()
vector_store = VectorStore()
embedding_service = EmbeddingService()
settings = get_settings()


@router.post("/", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document (PDF, MD, TXT, JSON)"""
    
    # Validate file type
    allowed_extensions = {".pdf", ".md", ".txt", ".json"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not supported. Allowed: {allowed_extensions}"
        )
    
    # Save uploaded file
    file_id = str(uuid4())
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / f"{file_id}_{file.filename}"
    
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Saved file: {file_path}")
        
        # Process document
        chunks, metadatas = document_processor.process_document(
            str(file_path),
            file_ext[1:]  # Remove dot
        )
        
        # Update total chunks
        metadatas = document_processor.update_total_chunks(metadatas)
        
        # Generate embeddings
        embeddings = embedding_service.embed_batch(chunks)
        
        # Store in vector database
        chunk_ids = vector_store.upsert_chunks(chunks, embeddings, metadatas)
        
        logger.info(f"Successfully processed {len(chunks)} chunks")
        
        return UploadResponse(
            file_id=file_id,
            filename=file.filename,
            file_type=file_ext[1:],
            chunks_created=len(chunks),
            status="success",
            message=f"Document processed successfully with {len(chunks)} chunks"
        )
        
    except Exception as e:
        logger.error(f"Upload processing error: {e}")
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))
