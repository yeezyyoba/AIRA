"""AIRA Phase 9 Mathematical Metrics and Scoring Engine.

Defines standardized Information Retrieval (IR), extraction, RAG, and agentic metrics:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG@K)
- Field Extraction Accuracy
- Correct-Paper & Correct-Page Retrieval Rates
- Faithfulness & Unsupported Claim Rate
- Agentic Workflow Metrics (Task Success, Retry Rate, Redundancy, Stopping Accuracy)
"""
import math
from typing import Any, Dict, List, Optional, Set, Tuple


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int = 5) -> float:
    """Compute Precision@K.

    Formula: |Retrieved[:k] ∩ Relevant| / k
    Denominator: k (or min(k, len(retrieved)) if non-empty, standard IR uses k).
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    rel_hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return float(rel_hits / k)


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int = 5) -> float:
    """Compute Recall@K.

    Formula: |Retrieved[:k] ∩ Relevant| / |Relevant|
    Denominator: Total relevant documents.
    """
    if not relevant:
        return 1.0
    top_k = retrieved[:k]
    rel_hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return float(rel_hits / len(relevant))


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """Compute Reciprocal Rank (RR) for a single query.

    Formula: 1 / rank of first relevant item in retrieved (1-indexed).
    """
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    queries_retrieved: List[List[str]],
    queries_relevant: List[Set[str]],
) -> float:
    """Compute Mean Reciprocal Rank (MRR) across all queries.

    Formula: (1 / |Q|) * sum_{i=1}^{|Q|} (1 / rank_i)
    """
    if not queries_retrieved:
        return 0.0
    total_rr = sum(
        reciprocal_rank(ret, rel)
        for ret, rel in zip(queries_retrieved, queries_relevant)
    )
    return float(total_rr / len(queries_retrieved))


def dcg_at_k(relevance_scores: List[float], k: int = 5) -> float:
    """Compute Discounted Cumulative Gain at rank K.

    Formula: sum_{i=1}^K (2^{rel_i} - 1) / log2(i + 1)
    """
    dcg = 0.0
    for idx, score in enumerate(relevance_scores[:k], start=1):
        gain = (2.0 ** score) - 1.0
        discount = math.log2(idx + 1)
        dcg += gain / discount
    return dcg


def ndcg_at_k(
    retrieved: List[str],
    relevance_map: Dict[str, float],
    k: int = 5,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank K.

    Formula: DCG@K / IDCG@K
    """
    if k <= 0 or not relevance_map:
        return 0.0

    actual_scores = [relevance_map.get(doc_id, 0.0) for doc_id in retrieved[:k]]
    actual_dcg = dcg_at_k(actual_scores, k=k)

    ideal_scores = sorted(list(relevance_map.values()), reverse=True)[:k]
    ideal_dcg = dcg_at_k(ideal_scores, k=k)

    if ideal_dcg == 0.0:
        return 1.0 if actual_dcg == 0.0 else 0.0

    return float(actual_dcg / ideal_dcg)


def text_token_overlap(predicted: str, expected: str) -> float:
    """Compute normalized Jaccard word token overlap between two strings (0.0 to 1.0)."""
    if not predicted or not expected:
        return 0.0
    p_tokens = set(predicted.lower().split())
    e_tokens = set(expected.lower().split())
    if not p_tokens or not e_tokens:
        return 0.0
    intersection = p_tokens.intersection(e_tokens)
    union = p_tokens.union(e_tokens)
    return float(len(intersection) / len(union))


def field_extraction_accuracy(
    predicted_field: Any,
    expected_field: Any,
    threshold: float = 0.6,
) -> bool:
    """Determine whether an extracted field accurately reflects ground truth."""
    if predicted_field is None and expected_field is None:
        return True
    if predicted_field is None or expected_field is None:
        return False

    if isinstance(expected_field, (int, float)):
        try:
            val = float(predicted_field)
            exp = float(expected_field)
            return abs(val - exp) <= 0.01 * exp or abs(val - exp) <= 0.5
        except (ValueError, TypeError):
            return False

    if isinstance(expected_field, list):
        if not isinstance(predicted_field, list):
            return False
        exp_set = {str(x).lower().strip() for x in expected_field}
        pred_set = {str(x).lower().strip() for x in predicted_field}
        if not exp_set:
            return True
        overlap = len(exp_set.intersection(pred_set)) / len(exp_set)
        return overlap >= threshold

    # Text fields
    pred_str = str(predicted_field).strip()
    exp_str = str(expected_field).strip()
    if exp_str.lower() in pred_str.lower():
        return True
    overlap = text_token_overlap(pred_str, exp_str)
    return overlap >= threshold


def evidence_grounding_eval(
    claims: List[str],
    evidence_passages: List[str],
    overlap_threshold: float = 0.25,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Evaluate faithfulness and unsupported claim rate against evidence passages.

    Returns:
        (faithfulness_score, unsupported_claim_rate, claim_audit_log)
    """
    if not claims:
        return 1.0, 0.0, []

    combined_evidence = " ".join(evidence_passages).lower()
    audit = []
    supported_count = 0

    for claim in claims:
        claim_words = [w for w in claim.lower().split() if len(w) > 3]
        if not claim_words:
            is_supported = True
        else:
            matches = sum(1 for w in claim_words if w in combined_evidence)
            ratio = matches / len(claim_words)
            is_supported = ratio >= overlap_threshold

        if is_supported:
            supported_count += 1

        audit.append({
            "claim": claim,
            "supported": is_supported,
        })

    faithfulness = float(supported_count / len(claims))
    unsupported_rate = float(1.0 - faithfulness)
    return faithfulness, unsupported_rate, audit
