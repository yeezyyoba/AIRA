"""Tool 5: Research Methodology Analyzer.

Analyzes research design, data collection, preprocessing, architecture,
training, evaluation, and artifact/code availability for academic research papers.
Strictly distinguishes direct extracts from analytical inferences.
"""
import re
from typing import Any, Dict, List, Optional

from llm.model_router import ModelRouter, model_router as default_model_router
from logging_config.logger import get_logger
from memory.vector_store import ChromaVectorStore, vector_store as default_vector_store
from services.paper_repository import PaperRepository, paper_repository as default_paper_repo
from validation.schemas import (
    EvidenceItemSchema,
    EvidenceType,
    MethodologyAnalysisSchema,
    MethodologyDimensionSchema,
    PaperExtractionResultSchema,
)

logger = get_logger("methodology_analyzer")


def _extract_evidence_passage(paper: PaperExtractionResultSchema, heading_keyword: str) -> Optional[EvidenceItemSchema]:
    """Find first matching evidence item from paper by heading keyword."""
    if not paper.extracted_data or not paper.extracted_data.evidence:
        return None
    for ev in paper.extracted_data.evidence:
        if ev.section_heading and heading_keyword.lower() in ev.section_heading.lower():
            return ev
    return None


def _resolve_text(*candidates: Any, default: str = "Not reported") -> str:
    """Safely resolve the first non-empty string candidate, guaranteeing a valid string."""
    for c in candidates:
        if c is not None and isinstance(c, str) and c.strip():
            return c.strip()
    return default


def assess_artifact_and_code_availability(paper: PaperExtractionResultSchema) -> Dict[str, Any]:
    """Assess code, dataset, and hyperparameter availability indicators without claiming proof of reproducibility.
    
    Distinguishes author-associated repositories from third-party citations.
    """
    texts = list((paper.raw_text_by_page or {}).values())
    data = paper.extracted_data
    if data:
        for val in (data.methodology, data.experimental_setup, data.dataset, data.abstract, data.conclusion):
            if val:
                texts.append(val)
    full_text = " ".join(texts).lower()
    setup_text = (data.experimental_setup if data else "") or ""
    dataset_text = (data.dataset if data else "") or ""

    # 1. Code repository links
    repo_urls = re.findall(
        r"https?://(?:www\.)?(?:github\.com|gitlab\.com|huggingface\.co)/[a-zA-Z0-9\-_./]+",
        full_text,
        re.IGNORECASE,
    )

    code_status = "NOT_REPORTED"
    code_url = None
    if repo_urls:
        clean_url = repo_urls[0].rstrip(".,;)")
        # Check if author-associated or merely citing a third-party baseline
        author_cues = [
            "our code", "source code is available", "code is available",
            "we release", "released at", "code at http", "available at http",
            "implementation can be found", "github.com/my", "we open-source",
            "publicly available at",
        ]
        is_author = any(cue in full_text for cue in author_cues)
        if is_author:
            code_status = "AUTHOR_REPOSITORY_AVAILABLE"
            code_url = clean_url
        else:
            code_status = "THIRD_PARTY_OR_BASELINE_ONLY"
            code_url = clean_url

    # 2. Dataset availability
    dataset_status = "NOT_REPORTED"
    dataset_details = None
    if dataset_text and len(dataset_text.strip()) > 3:
        dataset_details = dataset_text.strip()[:100]
        if any(w in dataset_text.lower() for w in ("proprietary", "internal", "private", "nda", "confidential")):
            dataset_status = "PROPRIETARY_OR_RESTRICTED"
        elif any(w in dataset_text.lower() for w in ("zenodo", "huggingface.co/datasets", "openslr", "kaggle", "common voice", "librispeech", "imagenet", "glue", "squad")):
            dataset_status = "PUBLIC_BENCHMARK_OR_OPEN_DATASET"
        else:
            dataset_status = "CUSTOM_DATASET_REPORTED"

    # 3. Hyperparameters & Environment
    comb_setup = f"{setup_text} {full_text}".lower()
    has_hyperparams = any(w in comb_setup for w in ("learning rate", "lr=", "lr =", "batch size", "optimizer", "adam", "sgd", "epochs"))
    has_seeds = any(w in comb_setup for w in ("random seed", "seed=", "seed ", "random_state"))
    has_hardware = any(w in comb_setup for w in ("gpu", "tpu", "a100", "v100", "rtx", "cuda"))

    # Summary indicator
    parts = []
    if code_status == "AUTHOR_REPOSITORY_AVAILABLE":
        parts.append(f"Author code repository reported ({code_url})")
    elif code_status == "THIRD_PARTY_OR_BASELINE_ONLY":
        parts.append(f"Third-party baseline/library link noted ({code_url}); author code not confirmed")
    else:
        parts.append("No source code repository link identified in text")

    if dataset_status == "PUBLIC_BENCHMARK_OR_OPEN_DATASET":
        parts.append(f"Public dataset ({dataset_details})")
    elif dataset_status == "PROPRIETARY_OR_RESTRICTED":
        parts.append("Restricted/proprietary dataset")
    elif dataset_status == "CUSTOM_DATASET_REPORTED":
        parts.append(f"Dataset reported ({dataset_details})")
    else:
        parts.append("Dataset availability details not reported")

    if has_hyperparams:
        parts.append("Training hyperparameters documented")
    else:
        parts.append("Key hyperparameters not detailed")

    if has_seeds:
        parts.append("Random seed documented")
    else:
        parts.append("Random seed omitted")

    summary_str = "; ".join(parts) + ". (Indicator measures reported availability; does not verify independent execution)."

    return {
        "code_status": code_status,
        "code_url": code_url,
        "repository_url": code_url,
        "dataset_status": dataset_status,
        "dataset_details": dataset_details,
        "hyperparameters_reported": has_hyperparams,
        "hyperparameters_status": "REPORTED" if has_hyperparams else "NOT_REPORTED",
        "random_seeds_reported": has_seeds,
        "random_seeds_status": "REPORTED" if has_seeds else "NOT_REPORTED",
        "hardware_reported": has_hardware,
        "indicator_summary": summary_str,
    }


async def analyze_research_methodology(
    paper_id: str,
    repository: Optional[PaperRepository] = None,
    vector_db: Optional[ChromaVectorStore] = None,
    router: Optional[ModelRouter] = None,
) -> MethodologyAnalysisSchema:
    """Perform rigorous methodological breakdown for an extracted academic paper.
    
    Examines:
    - Research design
    - Data collection & dataset characteristics
    - Preprocessing methods
    - Model architecture & algorithms
    - Training strategy
    - Evaluation methodology
    - Artifact & code availability indicators
    - Reported limitations
    
    Preserves exact page references and marks evidence as DIRECT_EXTRACT or INFERENCE.
    Never fabricates fallback page numbers when unknown.
    """
    if not paper_id or not isinstance(paper_id, str):
        raise ValueError("paper_id must be a non-empty string")

    clean_id = paper_id.strip()
    repo = repository or default_paper_repo
    store = vector_db or default_vector_store
    llm = router or default_model_router

    paper = repo.get_paper(clean_id)
    if not paper:
        raise ValueError(f"Paper '{clean_id}' was not found in the repository. Please extract paper first.")

    doc_data = paper.extracted_data
    title = (doc_data.title if doc_data else None) or paper.document_metadata.get("title") or clean_id

    dimensions: List[MethodologyDimensionSchema] = []
    limitations: List[EvidenceItemSchema] = []

    # 1. Research Design (from Objectives / Abstract)
    ev_obj = _extract_evidence_passage(paper, "objective") or _extract_evidence_passage(paper, "abstract")
    design_desc = _resolve_text(
        doc_data.research_objectives if doc_data else None,
        doc_data.abstract if doc_data else None,
        default="Research objectives not explicitly documented in extracted text.",
    )
    dimensions.append(
        MethodologyDimensionSchema(
            aspect="Research Design",
            description=design_desc[:500],
            evidence_type=EvidenceType.DIRECT_EXTRACT if ev_obj else EvidenceType.INFERENCE,
            confidence="HIGH" if ev_obj else "MEDIUM",
            paper_id=clean_id,
            page_number=ev_obj.page_number if ev_obj else None,
            evidence_passage=ev_obj.passage if ev_obj else None,
        )
    )

    # 2. Data Collection & Characteristics (from Dataset)
    ev_data = _extract_evidence_passage(paper, "dataset") or _extract_evidence_passage(paper, "data")
    dataset_desc = _resolve_text(
        doc_data.dataset if doc_data else None,
        default="Dataset characteristics and collection methodology not detailed in paper text.",
    )
    dimensions.append(
        MethodologyDimensionSchema(
            aspect="Data Collection & Characteristics",
            description=dataset_desc[:500],
            evidence_type=EvidenceType.DIRECT_EXTRACT if ev_data else EvidenceType.INFERENCE,
            confidence="HIGH" if ev_data else "MEDIUM",
            paper_id=clean_id,
            page_number=ev_data.page_number if ev_data else None,
            evidence_passage=ev_data.passage if ev_data else None,
        )
    )

    # 3. Model Architecture & Algorithms
    ev_model = _extract_evidence_passage(paper, "model") or _extract_evidence_passage(paper, "method")
    model_desc = _resolve_text(
        doc_data.models_and_algorithms if doc_data else None,
        doc_data.methodology if doc_data else None,
        default="Algorithmic models and architectures not explicitly broken down in dedicated section.",
    )
    dimensions.append(
        MethodologyDimensionSchema(
            aspect="Model Architecture & Algorithms",
            description=model_desc[:500],
            evidence_type=EvidenceType.DIRECT_EXTRACT if ev_model else EvidenceType.INFERENCE,
            confidence="HIGH" if ev_model else "MEDIUM",
            paper_id=clean_id,
            page_number=ev_model.page_number if ev_model else None,
            evidence_passage=ev_model.passage if ev_model else None,
        )
    )

    # 4. Experimental Setup & Training Strategy
    ev_setup = _extract_evidence_passage(paper, "setup") or _extract_evidence_passage(paper, "experiment")
    setup_desc = _resolve_text(
        doc_data.experimental_setup if doc_data else None,
        default="Experimental setup, hyperparameters, and compute budget not explicitly reported.",
    )
    dimensions.append(
        MethodologyDimensionSchema(
            aspect="Experimental Setup & Training Strategy",
            description=setup_desc[:500],
            evidence_type=EvidenceType.DIRECT_EXTRACT if ev_setup else EvidenceType.INFERENCE,
            confidence="HIGH" if ev_setup else "LOW",
            paper_id=clean_id,
            page_number=ev_setup.page_number if ev_setup else None,
            evidence_passage=ev_setup.passage if ev_setup else None,
        )
    )

    # 5. Evaluation Methodology
    ev_metrics = _extract_evidence_passage(paper, "eval") or _extract_evidence_passage(paper, "result")
    metrics_desc = _resolve_text(
        doc_data.evaluation_metrics if doc_data else None,
        doc_data.results if doc_data else None,
        default="Evaluation protocol and metrics not explicitly documented in dedicated section.",
    )
    dimensions.append(
        MethodologyDimensionSchema(
            aspect="Evaluation Methodology",
            description=metrics_desc[:500],
            evidence_type=EvidenceType.DIRECT_EXTRACT if ev_metrics else EvidenceType.INFERENCE,
            confidence="HIGH" if ev_metrics else "MEDIUM",
            paper_id=clean_id,
            page_number=ev_metrics.page_number if ev_metrics else None,
            evidence_passage=ev_metrics.passage if ev_metrics else None,
        )
    )

    # 6. Artifact & Code Availability Assessment
    artifact_assessment = assess_artifact_and_code_availability(paper)
    dimensions.append(
        MethodologyDimensionSchema(
            aspect="Artifact & Code Availability",
            description=artifact_assessment["indicator_summary"][:500],
            evidence_type=EvidenceType.INFERENCE,
            confidence="HIGH" if artifact_assessment["code_status"] != "NOT_REPORTED" else "MEDIUM",
            paper_id=clean_id,
            page_number=None,
            evidence_passage=None,
        )
    )

    # 7. Collect Reported Limitations
    if doc_data and doc_data.limitations and doc_data.limitations.strip():
        ev_lim = _extract_evidence_passage(paper, "limitation")
        limitations.append(
            EvidenceItemSchema(
                paper_id=clean_id,
                page_number=ev_lim.page_number if ev_lim else None,
                section_heading="Limitations",
                passage=doc_data.limitations.strip()[:400],
                evidence_type=EvidenceType.DIRECT_EXTRACT if ev_lim else EvidenceType.INFERENCE,
            )
        )

    is_degraded = llm.is_degraded_mode() if llm else False

    return MethodologyAnalysisSchema(
        paper_id=clean_id,
        document_title=title,
        dimensions=dimensions,
        artifact_availability=artifact_assessment,
        reproducibility_score=artifact_assessment["indicator_summary"],
        reported_limitations=limitations,
        is_degraded=is_degraded,
    )
