
"""AIRA Phase 9 Mathematical Metrics and Scoring Engine.

Defines deterministic evaluation metrics for information retrieval,
extraction, RAG, and agentic workflows.

Metrics:
- Precision@K and Recall@K
- Reciprocal Rank (RR) and Mean Reciprocal Rank (MRR)
- Discounted Cumulative Gain (DCG@K) and Normalized DCG (nDCG@K)
- Field Extraction Accuracy
- Lexical Evidence Overlap Proxy
- Lexical Jaccard Token Overlap

IMPORTANT:
The evidence-grounding evaluator is a lexical heuristic. It is not a
semantic entailment model and cannot establish factual correctness.
"""

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple


PUNCT_STR = """.,;:!?'"()[]{}<>«»፡።፣፤፦—–"""

COMMON_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "shall", "should", "can", "could", "may", "might", "must",
    "this", "that", "these", "those", "it", "its", "we", "our", "their",
}

NEGATION_WORDS = {
    "not", "never", "no", "neither", "nor", "none",
    "without", "failed", "cannot", "didn't", "doesn't",
    "isn't", "wasn't", "weren't", "won't",
}

MEASUREMENT_TERMS = {
    "accuracy": {"accuracy", "accurate"},
    "error": {"error", "errors", "erroneous"},
    "precision": {"precision"},
    "recall": {"recall"},
    "f1": {"f1", "f1-score", "f1_score"},
    "wer": {"wer", "word-error-rate"},
    "cer": {"cer", "character-error-rate"},
    "bleu": {"bleu"},
    "rouge": {"rouge"},
    "perplexity": {"perplexity"},
    "loss": {"loss"},
}

CONFLICTING_MEASUREMENT_PAIRS = {
    frozenset({"accuracy", "error"}),
    frozenset({"precision", "error"}),
    frozenset({"recall", "error"}),
    frozenset({"f1", "error"}),
}


def _validate_k(k: int) -> None:
    """Reject invalid retrieval cutoffs."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")


def _zero_for_nonpositive_k(k: int) -> bool:
    """Preserve the project's established k=0 behavior.

    A non-positive integer cutoff produces 0.0 in the public metrics.
    Invalid types, including bool, are still rejected.
    """
    if not isinstance(k, int) or isinstance(k, bool):
        raise ValueError(f"k must be an integer, got {k!r}")
    return k <= 0


def _validate_unit_interval(value: float, name: str) -> None:
    """Validate a parameter expected to be within [0, 1]."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number between 0 and 1.")

    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value!r}.")


def _clean_token(t: str) -> str:
    """Normalize a token by stripping whitespace and punctuation."""
    return t.strip().strip(PUNCT_STR).strip().lower()


def _extract_numbers(text: str) -> Set[str]:
    """Extract normalized numerical tokens, including percentages."""
    tokens = re.findall(r"(?<!\w)\d+(?:[.,]\d+)?%?(?!\w)", text)
    cleaned = set()

    for token in tokens:
        normalized = token.replace(",", "").strip()
        cleaned.add(normalized)

        if normalized.endswith("%"):
            cleaned.add(normalized[:-1])

    return cleaned


def _split_sentences(text: str) -> List[str]:
    """Split evidence into approximate sentences."""
    parts = re.split(r"(?<=[.!?።፨])\s+|[\n\r]+", text)
    return [part.strip() for part in parts if part.strip()]


def _measurement_terms(text: str) -> Set[str]:
    """Return recognized measurement concepts present in text."""
    tokens = {
        _clean_token(token)
        for token in re.findall(r"[A-Za-z0-9_-]+", text.lower())
    }

    found = set()
    for concept, variants in MEASUREMENT_TERMS.items():
        if tokens.intersection(variants):
            found.add(concept)

    return found


def _has_measurement_conflict(claim: str, evidence: str) -> bool:
    """Detect obvious conflicts between measurement concepts."""
    claim_terms = _measurement_terms(claim)
    evidence_terms = _measurement_terms(evidence)

    for pair in CONFLICTING_MEASUREMENT_PAIRS:
        if claim_terms.intersection(pair) and evidence_terms.intersection(pair):
            if claim_terms.intersection(pair) != evidence_terms.intersection(pair):
                return True

    return False


def _negation_state(text: str) -> bool:
    """Return whether a text contains a recognized negation token."""
    tokens = {
        _clean_token(token)
        for token in re.findall(r"[A-Za-z']+", text.lower())
    }
    return bool(tokens.intersection(NEGATION_WORDS))


def precision_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int = 5,
) -> float:
    """Compute Precision@K using distinct retrieved document IDs.

    The denominator is fixed at k. Missing positions count as non-relevant.
    For compatibility, k=0 returns 0.0.
    """
    if _zero_for_nonpositive_k(k):
        return 0.0

    _validate_k(k)

    seen = set()
    relevant_hits = 0

    for doc_id in retrieved[:k]:
        if doc_id in relevant and doc_id not in seen:
            relevant_hits += 1
            seen.add(doc_id)

    return float(relevant_hits / k)


def recall_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int = 5,
) -> Optional[float]:
    """Compute Recall@K.

    Returns None when the relevance set is empty because recall is undefined.
    For compatibility, k=0 returns 0.0.
    """
    if _zero_for_nonpositive_k(k):
        return 0.0

    _validate_k(k)

    if not relevant:
        return None

    seen = set()
    relevant_hits = 0

    for doc_id in retrieved[:k]:
        if doc_id in relevant and doc_id not in seen:
            relevant_hits += 1
            seen.add(doc_id)

    return float(relevant_hits / len(relevant))


def reciprocal_rank(
    retrieved: List[str],
    relevant: Set[str],
) -> float:
    """Compute reciprocal rank for one query."""
    if not relevant:
        return 0.0

    seen = set()

    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant and doc_id not in seen:
            return 1.0 / rank
        seen.add(doc_id)

    return 0.0


def mean_reciprocal_rank(
    queries_retrieved: List[List[str]],
    queries_relevant: List[Set[str]],
) -> float:
    """Compute MRR with strict input-length validation."""
    if len(queries_retrieved) != len(queries_relevant):
        raise ValueError(
            f"Mismatched query counts: "
            f"{len(queries_retrieved)} retrieved lists vs "
            f"{len(queries_relevant)} relevance sets."
        )

    if not queries_retrieved:
        return 0.0

    total = sum(
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(queries_retrieved, queries_relevant)
    )

    return float(total / len(queries_retrieved))


def dcg_at_k(
    relevance_scores: List[float],
    k: int = 5,
) -> float:
    """Compute DCG@K using exponential gain.

    Negative, non-finite, and non-numeric relevance scores are rejected.
    For compatibility, k=0 returns 0.0.
    """
    if _zero_for_nonpositive_k(k):
        return 0.0

    _validate_k(k)

    dcg = 0.0

    for index, score in enumerate(relevance_scores[:k], start=1):
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
        ):
            raise ValueError(
                f"Relevance scores must be finite numbers, got {score!r}."
            )

        if score < 0:
           raise ValueError(
    f"Relevance scores must be non-negative: {score!r}."
)

        try:
            gain = (2.0 ** float(score)) - 1.0
        except OverflowError as exc:
            raise ValueError(
                f"Relevance score is too large: {score!r}."
            ) from exc

        dcg += gain / math.log2(index + 1)

    if not math.isfinite(dcg):
        raise ValueError("DCG calculation produced a non-finite result.")

    return float(dcg)


def ndcg_at_k(
    retrieved: List[str],
    relevance_map: Dict[str, float],
    k: int = 5,
) -> float:
    """Compute nDCG@K.

    Empty relevance judgments or zero ideal gain return 0.0.
    Duplicate retrieved IDs receive zero gain after their first occurrence.
    For compatibility, k=0 returns 0.0.
    """
    if _zero_for_nonpositive_k(k):
        return 0.0

    _validate_k(k)

    if not relevance_map:
        return 0.0

    for doc_id, score in relevance_map.items():
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
        ):
            raise ValueError(
                f"Invalid relevance score for {doc_id!r}: {score!r}."
            )

        if score < 0:
          raise ValueError(
    f"Invalid negative score for {doc_id!r}: {score!r}. "
    "Relevance scores must be non-negative."
)

    seen = set()
    actual_scores = []

    for doc_id in retrieved[:k]:
        if doc_id in seen:
            actual_scores.append(0.0)
        else:
            seen.add(doc_id)
            actual_scores.append(float(relevance_map.get(doc_id, 0.0)))

    actual_dcg = dcg_at_k(actual_scores, k)

    ideal_scores = sorted(
        (float(score) for score in relevance_map.values()),
        reverse=True,
    )[:k]

    ideal_dcg = dcg_at_k(ideal_scores, k)

    if ideal_dcg == 0.0:
        return 0.0

    ratio = actual_dcg / ideal_dcg

    if not math.isfinite(ratio):
        raise ValueError("nDCG calculation produced a non-finite result.")

    if ratio > 1.0 + 1e-9:
        raise ValueError(f"Invalid nDCG result greater than 1.0: {ratio}.")

    if ratio > 1.0:
        ratio = 1.0

    if ratio < 0.0:
        raise ValueError(f"Invalid negative nDCG result: {ratio}.")

    return float(ratio)


def text_token_overlap(
    predicted: str,
    expected: str,
) -> float:
    """Compute lexical Jaccard token overlap, not semantic similarity."""
    if not predicted or not expected:
        return 0.0

    predicted_tokens = {
        _clean_token(token)
        for token in predicted.split()
        if _clean_token(token)
    }

    expected_tokens = {
        _clean_token(token)
        for token in expected.split()
        if _clean_token(token)
    }

    if not predicted_tokens or not expected_tokens:
        return 0.0

    intersection = predicted_tokens.intersection(expected_tokens)
    union = predicted_tokens.union(expected_tokens)

    return float(len(intersection) / len(union))


def field_extraction_accuracy(
    predicted_field: Any,
    expected_field: Any,
    threshold: float = 0.6,
    rel_tol: float = 0.01,
    abs_tol: float = 1e-4,
    allow_both_none: bool = False,
) -> bool:
    """Compare an extracted field against its ground truth."""
    _validate_unit_interval(threshold, "threshold")

    for value, name in ((rel_tol, "rel_tol"), (abs_tol, "abs_tol")):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise ValueError(f"{name} must be a finite non-negative number.")

    if predicted_field is None and expected_field is None:
        return bool(allow_both_none)

    if predicted_field is None or expected_field is None:
        return False

    if isinstance(expected_field, (int, float)):
        if isinstance(expected_field, bool):
            return predicted_field == expected_field

        try:
            predicted_number = float(predicted_field)
            expected_number = float(expected_field)

            if not (
                math.isfinite(predicted_number)
                and math.isfinite(expected_number)
            ):
                return False

            return math.isclose(
                predicted_number,
                expected_number,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
        except (ValueError, TypeError, OverflowError):
            return False

    if isinstance(expected_field, list):
        if not isinstance(predicted_field, list):
            return False

        def flatten_items(items: list) -> Set[str]:
            result = set()
            for item in items:
                for part in str(item).replace(" and ", ",").split(","):
                    cleaned = _clean_token(part)
                    if cleaned:
                        result.add(cleaned)
            return result

        expected_set = flatten_items(expected_field)
        predicted_set = flatten_items(predicted_field)

        if not expected_set and not predicted_set:
            return bool(allow_both_none)

        return expected_set == predicted_set

    predicted_text = str(predicted_field).strip()
    expected_text = str(expected_field).strip()

    if not expected_text:
        return not predicted_text

    if expected_text.lower() in predicted_text.lower():
        return True

    if (
        predicted_text.lower() in expected_text.lower()
        and len(predicted_text) >= 0.5 * len(expected_text)
    ):
        return True

    return text_token_overlap(
        predicted_text,
        expected_text,
    ) >= threshold


def evidence_grounding_eval(
    claims: List[str],
    evidence_passages: List[str],
    overlap_threshold: float = 0.25,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Evaluate lexical evidence overlap as a grounding proxy.

    Each claim is evaluated against individual evidence sentences.
    Numerical mismatches, measurement conflicts, and detected negation
    mismatches are flagged for manual review.

    This is not semantic entailment and cannot establish factual correctness.
    """
    _validate_unit_interval(overlap_threshold, "overlap_threshold")

    if not claims:
        return 1.0, 0.0, []

    usable_passages = [
        passage for passage in evidence_passages
        if isinstance(passage, str) and passage.strip()
    ]

    if not usable_passages:
        audit = [
            {
                "claim": claim,
                "supported": False,
                "requires_manual_review": False,
                "best_passage_index": None,
                "best_sentence": None,
                "overlap_ratio": 0.0,
                "reason": "No evidence passages provided.",
            }
            for claim in claims
        ]
        return 0.0, 1.0, audit

    evidence_units = []
    for passage_index, passage in enumerate(evidence_passages):
        if not isinstance(passage, str) or not passage.strip():
            continue

        for sentence in _split_sentences(passage):
            evidence_units.append({
                "passage_index": passage_index,
                "sentence": sentence,
            })

    audit = []
    supported_count = 0

    for claim in claims:
        claim_tokens = [
            _clean_token(token)
            for token in claim.split()
        ]

        content_words = [
            word for word in claim_tokens
            if len(word) > 2 and word not in COMMON_STOPWORDS
        ]

        if not content_words:
            audit.append({
                "claim": claim,
                "supported": False,
                "requires_manual_review": True,
                "best_passage_index": None,
                "best_sentence": None,
                "overlap_ratio": 0.0,
                "reason": "Claim has insufficient substantive content for evaluation.",
            })
            continue

        claim_numbers = _extract_numbers(claim)
        claim_negated = _negation_state(claim)

        best_match = None
        best_ratio = -1.0

        for unit in evidence_units:
            sentence = unit["sentence"]
            evidence_words = {
                _clean_token(word)
                for word in sentence.split()
            }

            matched = [
                word for word in content_words
                if word in evidence_words
            ]

            ratio = len(matched) / len(content_words)

            if ratio > best_ratio:
                best_ratio = ratio
                best_match = {
                    "passage_index": unit["passage_index"],
                    "sentence": sentence,
                    "matched_words": matched,
                    "unmatched_words": [
                        word for word in content_words
                        if word not in evidence_words
                    ],
                    "overlap_ratio": ratio,
                }

        if best_match is None:
            audit.append({
                "claim": claim,
                "supported": False,
                "requires_manual_review": False,
                "best_passage_index": None,
                "best_sentence": None,
                "overlap_ratio": 0.0,
                "reason": "No usable evidence sentences were available.",
            })
            continue

        sentence = best_match["sentence"]
        evidence_numbers = _extract_numbers(sentence)
        manual_review_reason = None

        if claim_numbers and not claim_numbers.issubset(evidence_numbers):
            manual_review_reason = (
                "Numerical values in the claim are not all present "
                "in the best-matching evidence sentence."
            )

        if _has_measurement_conflict(claim, sentence):
            manual_review_reason = (
                "Potentially conflicting measurement concepts were found."
            )

        evidence_negated = _negation_state(sentence)
        if claim_negated != evidence_negated:
            manual_review_reason = (
                "Claim and best-matching evidence sentence have different "
                "detected negation polarity. Negation scope requires review."
            )

        if manual_review_reason:
            audit.append({
                "claim": claim,
                "supported": False,
                "requires_manual_review": True,
                "best_passage_index": best_match["passage_index"],
                "best_sentence": sentence,
                "overlap_ratio": round(best_ratio, 4),
                "matched_words": best_match["matched_words"],
                "unmatched_words": best_match["unmatched_words"],
                "reason": manual_review_reason,
            })
            continue

        if len(content_words) >= 4:
            is_supported = (
                len(best_match["matched_words"]) >= 2
                and best_ratio >= overlap_threshold
            )
        else:
            is_supported = (
                len(best_match["matched_words"]) == len(content_words)
                and best_ratio >= overlap_threshold
            )

        if is_supported:
            supported_count += 1
            reason = (
                "Lexical overlap threshold satisfied. "
                "This is not semantic verification."
            )
        else:
            reason = (
                "Insufficient lexical overlap with any individual "
                "evidence sentence."
            )

        audit.append({
            "claim": claim,
            "supported": is_supported,
            "requires_manual_review": False,
            "best_passage_index": best_match["passage_index"],
            "best_sentence": sentence,
            "overlap_ratio": round(best_ratio, 4),
            "matched_words": best_match["matched_words"],
            "unmatched_words": best_match["unmatched_words"],
            "reason": reason,
        })

    grounding_rate = float(supported_count / len(claims))
    ungrounded_rate = float(1.0 - grounding_rate)

    return grounding_rate, ungrounded_rate, audit
