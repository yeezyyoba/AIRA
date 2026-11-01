"""Papers API Endpoint.

Provides endpoints for academic paper search, metadata cross-validation,
and indexed paper repository management.
"""
from pathlib import Path
import re
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.core.config import settings
from app.schemas.papers import (
    ContextRetrievalRequest,
    ContextRetrievalResponse,
    DeletePaperIndexResponse,
    PaperExtractionResponse,
    PaperIndexRequest,
    PaperIndexResponse,
    PaperIndexingStatusResponse,
    PaperReadRequest,
    PaperSearchRequest,
    PaperSearchResponse,
    PaperValidationRequest,
    PaperValidationResponse,
)
from memory.vector_store import IncompatibleEmbeddingModelError, vector_store
from services.embedding_provider import EmbeddingProviderUnavailableError
from services.paper_repository import paper_repository
from services.pdf_downloader import (
    InvalidPDFError,
    PDFDownloadError,
    PDFSizeExceededError,
    SSRFProtectionError,
)
from tools.academic_search import AllProvidersFailedError, search_academic_papers
from tools.metadata_validator import validate_paper_metadata
from tools.paper_indexer import (
    PaperNotFoundError,
    UnusablePaperTextError,
    index_research_paper,
)
from tools.paper_reader import read_research_paper
from tools.rag_retriever import retrieve_research_context

router = APIRouter()


@router.get("/", tags=["Papers"])
def list_papers() -> dict:
    """List extracted and indexed research papers."""
    extracted = paper_repository.list_papers()
    indexed = vector_store.get_indexed_paper_ids()
    return {
        "papers": indexed,
        "extracted_papers": extracted,
        "indexed_papers": indexed,
        "total_extracted": len(extracted),
        "total_indexed": len(indexed),
        "total_collection_chunks": vector_store.count(),
    }


@router.post("/search", response_model=PaperSearchResponse, tags=["Papers"])
async def search_papers_endpoint(request: PaperSearchRequest) -> PaperSearchResponse:
    """Search academic papers across OpenAlex, Semantic Scholar, and Crossref."""
    try:
        results = await search_academic_papers(
            query=request.query,
            year_from=request.year_from,
            year_to=request.year_to,
            sources=request.sources,
            max_results=request.max_results,
        )
        return PaperSearchResponse(**results)
    except AllProvidersFailedError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "message": str(e),
                "errors": e.errors,
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Academic search encountered an internal error: {e}",
        )


@router.post("/validate", response_model=PaperValidationResponse, tags=["Papers"])
async def validate_paper_endpoint(request: PaperValidationRequest) -> PaperValidationResponse:
    """Cross-verify paper metadata against Crossref, OpenAlex, and Semantic Scholar."""
    try:
        validation_result = await validate_paper_metadata(
            paper=request.paper,
            paper_id=request.paper_id,
        )
        return PaperValidationResponse(**validation_result.model_dump())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metadata validation encountered an internal error: {e}",
        )


@router.post("/read", response_model=PaperExtractionResponse, tags=["Papers"])
async def read_paper_endpoint(request: PaperReadRequest) -> PaperExtractionResponse:
    """Download and extract research paper dimensions from a remote URL."""
    try:
        result = await read_research_paper(
            paper_url_or_file=request.url,
            extraction_method=request.extraction_method,
            paper_id=request.paper_id,
        )
        if result.status != "FAILED":
            try:
                paper_repository.save_paper(result)
            except Exception:
                pass

        return PaperExtractionResponse(
            paper_id=result.paper_id,
            status=result.status.value,
            extraction_method=result.extraction_method,
            page_count=result.page_count,
            document_metadata=result.document_metadata,
            extracted_data=result.extracted_data.model_dump() if result.extracted_data else None,
            is_scanned=result.is_scanned,
            ocr_required=result.ocr_required,
            warnings=result.warnings,
            error_message=result.error_message,
        )
    except SSRFProtectionError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"SSRF Protection blocked URL: {e}",
        )
    except PDFSizeExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(e),
        )
    except InvalidPDFError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except PDFDownloadError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to download remote PDF: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Paper extraction error: {e}",
        )


@router.post("/upload", response_model=PaperExtractionResponse, tags=["Papers"])
async def upload_paper_endpoint(
    file: UploadFile = File(..., description="Academic PDF file to upload and extract"),
    extraction_method: str = Form("auto", description="Extraction method: auto, pymupdf, pdfplumber"),
    paper_id: Optional[str] = Form(None, description="Optional paper identifier"),
) -> PaperExtractionResponse:
    """Upload a local PDF file and extract structured academic dimensions.

    Enforces streaming chunk validation to abort oversized files during transmission
    and strictly sanitizes uploaded filenames against path traversal attacks.
    """
    raw_filename = file.filename or "uploaded_paper.pdf"
    # 1. Sanitize filename to prevent directory traversal
    clean_filename = Path(raw_filename).name
    clean_filename = re.sub(r"[^\w\.\-]", "_", clean_filename)

    if not clean_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a .pdf extension",
        )

    # 2. Stream chunk validation to prevent memory DOS
    try:
        content_chunks = []
        total_size = 0
        chunk_size = 64 * 1024  # 64 KB

        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > settings.MAX_PDF_SIZE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=f"Uploaded file exceeds limit of {settings.MAX_PDF_SIZE_BYTES} bytes",
                )
            content_chunks.append(chunk)

        content = b"".join(content_chunks)

        eff_id = paper_id or clean_filename
        result = await read_research_paper(
            paper_url_or_file=content,
            extraction_method=extraction_method,
            paper_id=eff_id,
        )

        if result.status != "FAILED":
            try:
                paper_repository.save_paper(result)
            except Exception as save_err:
                pass

        return PaperExtractionResponse(
            paper_id=result.paper_id,
            status=result.status.value,
            extraction_method=result.extraction_method,
            page_count=result.page_count,
            document_metadata=result.document_metadata,
            extracted_data=result.extracted_data.model_dump() if result.extracted_data else None,
            is_scanned=result.is_scanned,
            ocr_required=result.ocr_required,
            warnings=result.warnings,
            error_message=result.error_message,
        )
    except HTTPException:
        raise
    except PDFSizeExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(e),
        )
    except TimeoutError as e:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(e),
        )
    except InvalidPDFError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract uploaded PDF: {e}",
        )
    finally:
        await file.close()


@router.post("/{paper_id}/index", response_model=PaperIndexResponse, tags=["Papers"])
async def index_paper_endpoint(
    paper_id: str,
    request: Optional[PaperIndexRequest] = None,
) -> PaperIndexResponse:
    """Index a previously extracted academic research paper into ChromaDB."""
    chunk_size = request.chunk_size if request else None
    chunk_overlap = request.chunk_overlap if request else None

    try:
        summary = await index_research_paper(
            paper_id=paper_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return PaperIndexResponse(
            paper_id=summary.paper_id,
            chunk_count=summary.chunk_count,
            embedding_model=summary.embedding_model,
            dimension=summary.dimension,
            status=summary.status.value,
            message=summary.message,
        )
    except PaperNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except UnusablePaperTextError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except IncompatibleEmbeddingModelError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    except EmbeddingProviderUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Paper indexing failed: {e}",
        )


@router.post("/retrieve", response_model=ContextRetrievalResponse, tags=["Papers"])
async def retrieve_context_endpoint(request: ContextRetrievalRequest) -> ContextRetrievalResponse:
    """Retrieve relevant academic research passages using semantic vector search."""
    try:
        result = await retrieve_research_context(
            question=request.question,
            paper_ids=request.paper_ids,
            top_k=request.top_k,
        )
        return ContextRetrievalResponse(
            question=result.question,
            results_count=result.results_count,
            top_k=result.top_k,
            embedding_model=result.embedding_model,
            items=[item.model_dump() for item in result.items],
            filters_applied=result.filters_applied,
            is_degraded=result.is_degraded,
            is_synthetic=result.is_synthetic,
            warning=result.warning,
        )
    except IncompatibleEmbeddingModelError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic context retrieval failed: {e}",
        )


@router.get("/{paper_id}/indexing-status", response_model=PaperIndexingStatusResponse, tags=["Papers"])
def paper_indexing_status_endpoint(paper_id: str) -> PaperIndexingStatusResponse:
    """Inspect the vector indexing status of a paper in ChromaDB."""
    indexed_papers = vector_store.get_indexed_paper_ids()
    is_indexed = paper_id in indexed_papers

    chunk_count = 0
    if is_indexed:
        try:
            res = vector_store.collection.get(where={"paper_id": paper_id})
            chunk_count = len(res.get("ids") or [])
        except Exception:
            chunk_count = 0

    return PaperIndexingStatusResponse(
        paper_id=paper_id,
        is_indexed=is_indexed,
        chunk_count=chunk_count,
        total_collection_chunks=vector_store.count(),
        embedding_model=vector_store.embedding_model,
        dimension=vector_store.dimension,
    )


@router.delete("/{paper_id}/index", response_model=DeletePaperIndexResponse, tags=["Papers"])
def delete_paper_index_endpoint(paper_id: str) -> DeletePaperIndexResponse:
    """Delete all indexed vector chunks belonging to a paper from ChromaDB."""
    try:
        deleted = vector_store.delete_paper(paper_id)
        return DeletePaperIndexResponse(
            paper_id=paper_id,
            deleted_chunks=deleted,
            remaining_collection_chunks=vector_store.count(),
            message=f"Deleted {deleted} chunks for paper '{paper_id}' from vector store.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete paper index: {e}",
        )

