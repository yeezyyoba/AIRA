"""Tool 4: Model Results Analyzer (Category: Code & Compute).

Extracts, normalizes, and compares experimental machine learning and NLP results across papers.
Uses Python/Pandas/NumPy for deterministic numerical operations:
- Preserves units and metric names.
- Distinguishes percentages (0-100), fractions (0-1), and absolute scores.
- Enforces metric directionality: HIGHER_IS_BETTER (Accuracy, F1, BLEU) vs LOWER_IS_BETTER (WER, CER, Loss).
- Detects incompatible metrics (e.g. Macro-F1 vs Micro-F1, different dataset splits) and flags them rather than generating misleading rankings.
- Flags missing statistical significance (p-values) or standard deviations.
"""
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from logging_config.logger import get_logger
from memory.vector_store import ChromaVectorStore, vector_store as default_vector_store
from services.paper_repository import PaperRepository, paper_repository as default_paper_repo
from validation.schemas import (
    ExtractedMetricValueSchema,
    MetricCompatibility,
    MetricGoal,
    MetricUnitType,
    MetricComparisonResultSchema,
    PaperExtractionResultSchema,
)

logger = get_logger("result_analyzer")

# Metric optimization directionality definition
METRIC_GOALS: Dict[str, MetricGoal] = {
    "accuracy": MetricGoal.HIGHER_IS_BETTER,
    "precision": MetricGoal.HIGHER_IS_BETTER,
    "recall": MetricGoal.HIGHER_IS_BETTER,
    "f1": MetricGoal.HIGHER_IS_BETTER,
    "macro_f1": MetricGoal.HIGHER_IS_BETTER,
    "micro_f1": MetricGoal.HIGHER_IS_BETTER,
    "weighted_f1": MetricGoal.HIGHER_IS_BETTER,
    "bleu": MetricGoal.HIGHER_IS_BETTER,
    "bleu_4": MetricGoal.HIGHER_IS_BETTER,
    "sacrebleu": MetricGoal.HIGHER_IS_BETTER,
    "rouge": MetricGoal.HIGHER_IS_BETTER,
    "rouge_1": MetricGoal.HIGHER_IS_BETTER,
    "rouge_2": MetricGoal.HIGHER_IS_BETTER,
    "rouge_l": MetricGoal.HIGHER_IS_BETTER,
    "auc": MetricGoal.HIGHER_IS_BETTER,
    "roc_auc": MetricGoal.HIGHER_IS_BETTER,
    "pr_auc": MetricGoal.HIGHER_IS_BETTER,
    "map": MetricGoal.HIGHER_IS_BETTER,
    "cer": MetricGoal.LOWER_IS_BETTER,
    "character_error_rate": MetricGoal.LOWER_IS_BETTER,
    "wer": MetricGoal.LOWER_IS_BETTER,
    "word_error_rate": MetricGoal.LOWER_IS_BETTER,
    "loss": MetricGoal.LOWER_IS_BETTER,
    "perplexity": MetricGoal.LOWER_IS_BETTER,
}

# Regex patterns for extracting metric scores from academic text
METRIC_EXTRACTION_PATTERNS = [
    # E.g.: "BLEU score of 31.2", "Accuracy reached 97.9%", "Macro-F1 of 0.841", "ROUGE-1 of 42.5"
    r"(?P<variant>macro[- ]?|micro[- ]?|weighted[- ]?|sacre[- ]?|bleu[- ]?4|roc[- ]?|pr[- ]?)?(?P<metric>accuracy|precision|recall|f1(?:-score)?|bleu(?:-[1234])?|sacrebleu|rouge(?:-[12l])?|cer|wer|roc_auc|pr_auc|auc|loss|perplexity)\s*(?:of|is|was|reached|reaches?|achieved|achieves?|scored|scores?|=|:)?\s*(?P<val>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>%|percent)?",
    # E.g.: "Conformer-Large achieved 2.1% WER", "Model A scored 0.88 Macro-F1", "42.5 ROUGE-1"
    r"(?P<val>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>%|percent)?\s*(?P<variant>macro[- ]?|micro[- ]?|weighted[- ]?|sacre[- ]?|bleu[- ]?4|roc[- ]?|pr[- ]?)?(?P<metric>accuracy|precision|recall|f1(?:-score)?|bleu(?:-[1234])?|sacrebleu|rouge(?:-[12l])?|cer|wer|roc_auc|pr_auc|auc|loss|perplexity)",
    # E.g.: "BLEU: 31.2", "Accuracy = 88.5%", "ROUGE-1: 42.5"
    r"(?P<metric>accuracy|precision|recall|f1|bleu(?:-[1234])?|sacrebleu|rouge(?:-[12l])?|cer|wer|roc_auc|pr_auc|auc|loss|perplexity)\s*[:=]\s*(?P<val>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>%)?",
]


def normalize_metric_name(name: str) -> str:
    """Normalize metric name to canonical lowercase string."""
    clean = name.strip().lower().replace(" ", "_").replace("-", "_")
    if clean.endswith("_score"):
        clean = clean[:-6]
    return clean


def parse_numerical_metric(
    raw_val: Union[float, int, str],
    metric_name: str,
) -> Tuple[float, MetricUnitType]:
    """Parse numerical value and determine whether it is a percentage, fraction, or raw score.
    
    Normalizes percentage/fraction values into a standard 0-100 scale ONLY for unambiguous bounded metrics
    (accuracy, precision, recall, f1, macro_f1, micro_f1, weighted_f1, auc).
    For BLEU, ROUGE, and others where 0-100 vs 0-1 scaling may be ambiguous, preserves original numeric scale
    to avoid false 100x magnification.
    """
    if isinstance(raw_val, (int, float)):
        val = float(raw_val)
        unit = MetricUnitType.PERCENTAGE if val > 1.0 else MetricUnitType.FRACTION
    else:
        str_val = str(raw_val).strip()
        has_pct = "%" in str_val or "percent" in str_val.lower()
        cleaned = re.sub(r"[^0-9\.]", "", str_val)
        val = float(cleaned) if cleaned else 0.0
        unit = MetricUnitType.PERCENTAGE if has_pct else (MetricUnitType.FRACTION if val <= 1.0 else MetricUnitType.SCORE)

    norm_name = normalize_metric_name(metric_name)
    # Only normalize bounded [0, 1] classification metrics unambiguously to 0-100% scale
    if unit == MetricUnitType.FRACTION and norm_name in ("accuracy", "precision", "recall", "f1", "macro_f1", "micro_f1", "weighted_f1", "auc"):
        normalized = round(val * 100.0, 4)
    else:
        normalized = round(val, 4)

    return normalized, unit


def extract_metrics_from_text(
    paper_id: str,
    text: str,
    page_number: Optional[int] = None,
    default_model: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset_split: Optional[str] = None,
    dataset_version: Optional[str] = None,
    task: Optional[str] = None,
    language: Optional[str] = None,
    evaluation_protocol: Optional[str] = None,
    num_classes: Optional[int] = None,
) -> List[ExtractedMetricValueSchema]:
    """Extract numerical experimental findings from raw text or section content."""
    if not text:
        return []

    metrics: List[ExtractedMetricValueSchema] = []
    seen_keys = set()

    for pattern in METRIC_EXTRACTION_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            groups = match.groupdict()
            metric_raw = groups.get("metric", "").lower()
            variant_raw = (groups.get("variant") or "").lower().strip().replace("-", "").replace(" ", "")
            val_str = groups.get("val", "")
            unit_str = groups.get("unit", "")

            if not val_str:
                continue

            try:
                raw_num = float(val_str)
            except ValueError:
                continue

            # Full canonical metric name
            if variant_raw and variant_raw in ("macro", "micro", "weighted"):
                canonical_name = f"{variant_raw}_{metric_raw}"
            elif variant_raw and "sacre" in variant_raw:
                canonical_name = "sacrebleu"
            elif variant_raw and "bleu" in variant_raw and "4" in variant_raw:
                canonical_name = "bleu_4"
            else:
                canonical_name = metric_raw

            canonical_name = normalize_metric_name(canonical_name)

            norm_val, unit_type = parse_numerical_metric(
                f"{raw_num}{unit_str}",
                canonical_name,
            )

            # Check context for statistical significance / confidence intervals / split / dataset / dimensions
            start_span = max(0, match.start() - 60)
            end_span = min(len(text), match.end() + 60)
            context_window = text[start_span:end_span]

            ci_match = re.search(r"(?:±|\+/-)\s*([0-9]+(?:\.[0-9]+)?)", context_window)
            ci_val = ci_match.group(1) if ci_match else None

            p_match = re.search(r"p\s*[<:=]\s*0?\.[0-9]+", context_window, re.IGNORECASE)
            stat_sig = p_match.group(0) if p_match else None

            # Split extraction if not provided
            split = dataset_split
            if not split:
                split_match = re.search(r"\b(test(?:-clean|-other)?|val(?:idation)?|dev(?:elopment)?|train(?:ing)?)(?:\s+(?:set|split|data))?\b", context_window, re.IGNORECASE)
                if split_match:
                    sm = split_match.group(1).lower()
                    if "val" in sm or "dev" in sm:
                        split = "validation"
                    elif "train" in sm:
                        split = "train"
                    else:
                        split = sm

            # Dataset name extraction from context if not provided
            ds_name = dataset_name
            if not ds_name:
                ds_match = re.search(r"(?:on|using|evaluated on)\s+(?:the\s+)?([A-Z][A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+)?)", context_window)
                if ds_match:
                    cand = ds_match.group(1).strip()
                    if cand.lower() not in ("our", "the", "this", "table", "figure", "section", "all", "both", "benchmark", "test", "val", "dev"):
                        ds_name = cand

            # Dataset version extraction
            ds_ver = dataset_version
            if not ds_ver:
                ver_match = re.search(r"\b(v[0-9]+(?:\.[0-9]+)?)\b", context_window, re.IGNORECASE)
                if ver_match:
                    ds_ver = ver_match.group(1).lower()

            # Task extraction
            extracted_task = task
            if not extracted_task:
                task_match = re.search(r"\b(summarization|translation|question answering|qa|asr|speech recognition|text classification|sentiment analysis)\b", context_window, re.IGNORECASE)
                if task_match:
                    extracted_task = task_match.group(1).lower()

            # Language extraction
            extracted_lang = language
            if not extracted_lang:
                lang_match = re.search(r"\b(english|swahili|amharic|french|german|spanish|chinese|japanese)\b", context_window, re.IGNORECASE)
                if lang_match:
                    extracted_lang = lang_match.group(1).capitalize()

            # Evaluation protocol extraction
            extracted_proto = evaluation_protocol
            if not extracted_proto:
                proto_match = re.search(r"\b(zero[- ]shot|few[- ]shot|fine[- ]tuned|supervised|unsupervised)\b", context_window, re.IGNORECASE)
                if proto_match:
                    extracted_proto = proto_match.group(1).lower().replace(" ", "-")

            # Number of classes extraction
            extracted_classes = num_classes
            if extracted_classes is None:
                class_match = re.search(r"\b([0-9]+)[- ]class\b", context_window, re.IGNORECASE)
                if class_match:
                    extracted_classes = int(class_match.group(1))

            key = (paper_id, canonical_name, norm_val)
            if key not in seen_keys:
                seen_keys.add(key)
                metrics.append(
                    ExtractedMetricValueSchema(
                        paper_id=paper_id,
                        model_name=default_model or f"{paper_id}_model",
                        metric_name=canonical_name,
                        raw_value=val_str + (unit_str or ""),
                        normalized_value=norm_val,
                        unit_type=unit_type,
                        dataset_name=ds_name,
                        dataset_version=ds_ver,
                        task=extracted_task,
                        language=extracted_lang,
                        evaluation_protocol=extracted_proto,
                        num_classes=extracted_classes,
                        dataset_split=split,
                        variant=variant_raw or None,
                        confidence_interval=ci_val,
                        statistical_significance=stat_sig,
                        page_number=page_number,
                        evidence_passage=context_window.strip(),
                    )
                )

    return metrics


def check_precision_recall_tradeoff(
    all_extracted: List[ExtractedMetricValueSchema],
) -> Optional[str]:
    """Detect if any pair of models/papers exhibits a precision-recall trade-off."""
    prec_by_model: Dict[str, float] = {}
    rec_by_model: Dict[str, float] = {}

    for r in all_extracted:
        metric = normalize_metric_name(r.metric_name)
        if metric in ("precision",) and r.normalized_value is not None:
            prec_by_model[r.model_name] = r.normalized_value
        elif metric in ("recall",) and r.normalized_value is not None:
            rec_by_model[r.model_name] = r.normalized_value

    common_models = [m for m in prec_by_model if m in rec_by_model]
    if len(common_models) >= 2:
        for i in range(len(common_models)):
            for j in range(i + 1, len(common_models)):
                m1, m2 = common_models[i], common_models[j]
                p1, p2 = prec_by_model[m1], prec_by_model[m2]
                r1, r2 = rec_by_model[m1], rec_by_model[m2]
                if (p1 > p2 and r1 < r2) or (p1 < p2 and r1 > r2):
                    high_p, low_p = (m1, m2) if p1 > p2 else (m2, m1)
                    return (
                        f"Precision-Recall trade-off detected: '{high_p}' achieves higher precision "
                        f"({prec_by_model[high_p]:.2f}% vs {prec_by_model[low_p]:.2f}%) but lower recall "
                        f"({rec_by_model[high_p]:.2f}% vs {rec_by_model[low_p]:.2f}%) compared to '{low_p}'. "
                        "Neither model strictly dominates; model selection must be determined by the operational "
                        "cost of false positives versus false negatives."
                    )
    return None


def analyze_experimental_results(
    paper_ids: List[str],
    metrics: Optional[List[str]] = None,
    repository: Optional[PaperRepository] = None,
    vector_db: Optional[ChromaVectorStore] = None,
) -> List[MetricComparisonResultSchema]:
    """Analyze and compare experimental results across multiple academic research papers.
    
    Deterministic computation:
    - Normalizes percentages and fractions into comparable scales only when unambiguous.
    - Strictly separates ROUGE-1, ROUGE-2, and ROUGE-L.
    - Prevents cross-dataset ranking (flags INCOMPATIBLE or UNKNOWN, suppresses ranking).
    - Prevents cross-split ranking (test vs validation vs train).
    - Enforces F1 variant matching (macro vs micro vs weighted vs unspecified).
    - Evaluates Precision-Recall trade-offs without declaring isolated winners.
    - Detects missing standard deviations, sample sizes, and p-values.
    """
    if not paper_ids:
        raise ValueError("paper_ids must be a non-empty list of paper identifiers")

    repo = repository or default_paper_repo
    store = vector_db or default_vector_store

    # 1. Gather all extracted metric records for the requested papers
    all_extracted: List[ExtractedMetricValueSchema] = []

    for pid in paper_ids:
        clean_pid = pid.strip()
        paper_data: Optional[PaperExtractionResultSchema] = repo.get_paper(clean_pid)
        if not paper_data:
            logger.warning(f"Paper '{clean_pid}' not found in repository during results analysis.")
            continue

        model_label = clean_pid
        if paper_data.extracted_data and paper_data.extracted_data.models_and_algorithms:
            first_model = paper_data.extracted_data.models_and_algorithms.split(",")[0].split("\n")[0][:40]
            if first_model.strip():
                model_label = f"{clean_pid} ({first_model.strip()})"

        dataset_label = paper_data.extracted_data.dataset if paper_data.extracted_data else None

        # Scan structured results section if available
        if paper_data.extracted_data and paper_data.extracted_data.results:
            results_text = paper_data.extracted_data.results
            # Find evidence page for results if established
            res_page = None
            for ev in paper_data.extracted_data.evidence:
                if ev.section_heading and "result" in ev.section_heading.lower():
                    res_page = ev.page_number
                    break
            extracted = extract_metrics_from_text(
                paper_id=clean_pid,
                text=results_text,
                page_number=res_page,
                default_model=model_label,
                dataset_name=dataset_label,
            )
            all_extracted.extend(extracted)

        # Scan raw pages for numerical tables or metrics if structured section was sparse
        if len(all_extracted) == 0 and paper_data.raw_text_by_page:
            for p_num, p_text in paper_data.raw_text_by_page.items():
                if "result" in p_text.lower() or "eval" in p_text.lower() or "bleu" in p_text.lower() or "f1" in p_text.lower():
                    extracted = extract_metrics_from_text(
                        paper_id=clean_pid,
                        text=p_text,
                        page_number=p_num,
                        default_model=model_label,
                        dataset_name=dataset_label,
                    )
                    all_extracted.extend(extracted)

    if not all_extracted:
        logger.info(f"No numerical metrics extracted from papers: {paper_ids}")
        return []

    # Check overall precision-recall tradeoff across corpus
    corpus_pr_tradeoff = check_precision_recall_tradeoff(all_extracted)

    # 2. Group by distinct canonical metric family
    metric_groups: Dict[str, List[ExtractedMetricValueSchema]] = {}
    for item in all_extracted:
        canon_name = normalize_metric_name(item.metric_name)
        if metrics:
            normalized_filter = [normalize_metric_name(m) for m in metrics]
            if canon_name not in normalized_filter and not any(canon_name.endswith(m) or m in canon_name for m in normalized_filter):
                continue

        # Strict metric family separation: ROUGE-1, ROUGE-2, and ROUGE-L must NOT be merged!
        if canon_name in ("rouge_1", "rouge_2", "rouge_l"):
            family = canon_name
        elif "rouge_1" in canon_name or (canon_name.endswith("1") and "rouge" in canon_name):
            family = "rouge_1"
        elif "rouge_2" in canon_name or (canon_name.endswith("2") and "rouge" in canon_name):
            family = "rouge_2"
        elif "rouge_l" in canon_name or (canon_name.endswith("l") and "rouge" in canon_name):
            family = "rouge_l"
        elif "rouge" in canon_name:
            family = "rouge"
        elif "f1" in canon_name:
            family = "f1"
        elif "sacrebleu" in canon_name:
            family = "sacrebleu"
        elif "bleu" in canon_name:
            family = "bleu"
        elif "roc_auc" in canon_name:
            family = "roc_auc"
        elif "pr_auc" in canon_name:
            family = "pr_auc"
        elif "wer" in canon_name:
            family = "wer"
        elif "cer" in canon_name:
            family = "cer"
        elif "perplexity" in canon_name:
            family = "perplexity"
        else:
            family = canon_name

        metric_groups.setdefault(family, []).append(item)

    # 3. Analyze each metric group deterministically
    comparison_results: List[MetricComparisonResultSchema] = []

    for family, records in metric_groups.items():
        goal = METRIC_GOALS.get(family, MetricGoal.HIGHER_IS_BETTER)
        warnings: List[str] = []
        compatibility = MetricCompatibility.COMPATIBLE
        incomp_reason: Optional[str] = None
        trade_off_notes: Optional[str] = None

        if corpus_pr_tradeoff and family in ("precision", "recall", "f1"):
            trade_off_notes = corpus_pr_tradeoff
        elif family in ("precision", "recall"):
            trade_off_notes = "Isolated metric note: Optimizing precision or recall in isolation often degrades the other. Consider F1-score or PR-AUC for aggregate decision making."

        # Perplexity note
        if family == "perplexity":
            perp_note = (
                "Perplexity values are strongly dependent on tokenizer vocabulary size and tokenization protocol "
                "(e.g. byte-level vs subword BPE vs word-level). Comparisons across papers with different tokenizers "
                "or vocab sizes are invalid."
            )
            warnings.append(perp_note)
            if not trade_off_notes:
                trade_off_notes = perp_note

        distinct_papers = list(set(r.paper_id for r in records))

        # A. F1 Variant Compatibility Check
        if family == "f1":
            variants = set(r.variant for r in records)
            if len(variants) > 1:
                compatibility = MetricCompatibility.INCOMPATIBLE
                var_labels = sorted(str(v or "unspecified") for v in variants)
                incomp_reason = (
                    f"Incompatible evaluation criteria: papers report different F1 variants ({', '.join(var_labels)}). "
                    "Macro-F1 (unweighted class average), Micro-F1 (aggregate token/instance metric), Weighted-F1, "
                    "and Unspecified F1 cannot be directly compared or ranked."
                )
                warnings.append(incomp_reason)

        # B. BLEU Variant Check
        if family == "bleu":
            bleu_variants = set(r.variant for r in records if r.variant) | set(r.metric_name for r in records if "sacre" in r.metric_name or "bleu_4" in r.metric_name)
            if len(bleu_variants) > 1 and any("sacre" in str(v).lower() for v in bleu_variants):
                compatibility = MetricCompatibility.INCOMPATIBLE
                incomp_reason = (
                    "Incompatible BLEU variants: SacreBLEU uses standardized tokenization and reference detokenization, "
                    "which cannot be directly ranked against custom-tokenized BLEU-4 without identical reference strings."
                )
                warnings.append(incomp_reason)

        # C. Scale Ambiguity Check (Fraction vs Score for BLEU / ROUGE)
        if family in ("bleu", "rouge", "rouge_1", "rouge_2", "rouge_l"):
            units = set(r.unit_type for r in records)
            vals = [r.normalized_value for r in records if r.normalized_value is not None]
            has_fraction = any(v <= 1.0 for v in vals)
            has_score = any(v > 1.0 for v in vals)
            if has_fraction and has_score:
                compatibility = MetricCompatibility.INCOMPATIBLE
                incomp_reason = (
                    "Scale ambiguity detected: metric values span both fractional (<= 1.0) and score/percentage (> 1.0) "
                    "ranges without explicit percentage indication. Values cannot be safely compared without risking a 100x scaling error."
                )
                warnings.append(incomp_reason)

        # D. Dataset Identity Compatibility Check across distinct papers
        if len(distinct_papers) > 1 and compatibility == MetricCompatibility.COMPATIBLE:
            dataset_names = [r.dataset_name for r in records]
            if any(d is None or not str(d).strip() for d in dataset_names):
                compatibility = MetricCompatibility.UNKNOWN
                incomp_reason = "Dataset identity is unverified or unknown for one or more papers. Cross-dataset ranking is suppressed to prevent false equivalence."
                warnings.append(incomp_reason)
            else:
                norm_ds = set(str(d).strip().lower() for d in dataset_names if d)
                if len(norm_ds) > 1:
                    compatibility = MetricCompatibility.INCOMPATIBLE
                    incomp_reason = f"Incompatible datasets: papers evaluated on different datasets ({', '.join(sorted(norm_ds))}). Cross-dataset metric ranking is invalid."
                    warnings.append(incomp_reason)

        # E. Evaluation Split Compatibility Check across distinct papers
        if len(distinct_papers) > 1 and compatibility == MetricCompatibility.COMPATIBLE:
            splits = [r.dataset_split for r in records]
            if any(s is None or not str(s).strip() for s in splits):
                compatibility = MetricCompatibility.UNKNOWN
                incomp_reason = "Evaluation split (test vs val vs train) is unverified or unknown for one or more papers. Ranking suppressed to prevent cross-split false equivalence."
                warnings.append(incomp_reason)
            else:
                norm_splits = set(str(s).strip().lower() for s in splits if s)
                if len(norm_splits) > 1:
                    compatibility = MetricCompatibility.INCOMPATIBLE
                    incomp_reason = f"Incompatible evaluation splits: papers evaluated on different splits ({', '.join(sorted(norm_splits))}). Test set results cannot be ranked against validation/train results."
                    warnings.append(incomp_reason)

        # F. Dataset Version Compatibility Check across distinct papers
        if len(distinct_papers) > 1 and compatibility == MetricCompatibility.COMPATIBLE:
            all_versions = [r.dataset_version for r in records]
            has_specified_v = any(v is not None and str(v).strip() for v in all_versions)
            has_unspecified_v = any(v is None or not str(v).strip() for v in all_versions)
            versions = [str(v).lower().strip() for v in all_versions if v and str(v).strip()]
            if len(set(versions)) > 1:
                compatibility = MetricCompatibility.INCOMPATIBLE
                incomp_reason = f"Incompatible dataset versions: papers evaluated on different versions ({', '.join(sorted(set(versions)))}). Direct ranking is invalid."
                warnings.append(incomp_reason)
            elif has_specified_v and has_unspecified_v:
                compatibility = MetricCompatibility.UNKNOWN
                incomp_reason = "Dataset version is unverified or unspecified for one or more papers. Cross-version ranking suppressed."
                warnings.append(incomp_reason)

        # G. Task Compatibility Check across distinct papers
        if len(distinct_papers) > 1 and compatibility == MetricCompatibility.COMPATIBLE:
            all_tasks = [r.task for r in records]
            has_specified_t = any(t is not None and str(t).strip() for t in all_tasks)
            has_unspecified_t = any(t is None or not str(t).strip() for t in all_tasks)
            tasks = [str(t).lower().strip() for t in all_tasks if t and str(t).strip()]
            if len(set(tasks)) > 1:
                compatibility = MetricCompatibility.INCOMPATIBLE
                incomp_reason = f"Incompatible evaluation tasks: papers evaluated on different tasks ({', '.join(sorted(set(tasks)))})."
                warnings.append(incomp_reason)
            elif has_specified_t and has_unspecified_t:
                compatibility = MetricCompatibility.UNKNOWN
                incomp_reason = "Evaluation task is unspecified for one or more papers. Direct ranking suppressed."
                warnings.append(incomp_reason)

        # H. Language Compatibility Check across distinct papers
        if len(distinct_papers) > 1 and compatibility == MetricCompatibility.COMPATIBLE:
            all_langs = [r.language for r in records]
            has_specified_l = any(l is not None and str(l).strip() for l in all_langs)
            has_unspecified_l = any(l is None or not str(l).strip() for l in all_langs)
            langs = [str(l).lower().strip() for l in all_langs if l and str(l).strip()]
            if len(set(langs)) > 1:
                compatibility = MetricCompatibility.INCOMPATIBLE
                incomp_reason = f"Incompatible evaluation languages: papers evaluated on different languages ({', '.join(sorted(set(langs)))})."
                warnings.append(incomp_reason)
            elif has_specified_l and has_unspecified_l:
                compatibility = MetricCompatibility.UNKNOWN
                incomp_reason = "Evaluation language is unspecified for one or more papers. Direct ranking suppressed."
                warnings.append(incomp_reason)

        # I. Evaluation Protocol Compatibility Check across distinct papers
        if len(distinct_papers) > 1 and compatibility == MetricCompatibility.COMPATIBLE:
            all_protos = [r.evaluation_protocol for r in records]
            has_specified_p = any(p is not None and str(p).strip() for p in all_protos)
            has_unspecified_p = any(p is None or not str(p).strip() for p in all_protos)
            protocols = [str(p).lower().strip() for p in all_protos if p and str(p).strip()]
            if len(set(protocols)) > 1:
                compatibility = MetricCompatibility.INCOMPATIBLE
                incomp_reason = f"Incompatible evaluation protocols: papers evaluated under different protocols ({', '.join(sorted(set(protocols)))})."
                warnings.append(incomp_reason)
            elif has_specified_p and has_unspecified_p:
                compatibility = MetricCompatibility.UNKNOWN
                incomp_reason = "Evaluation protocol is unspecified for one or more papers. Direct ranking suppressed."
                warnings.append(incomp_reason)

        # J. Number of Classes Check across distinct papers
        if len(distinct_papers) > 1 and compatibility == MetricCompatibility.COMPATIBLE:
            all_classes = [r.num_classes for r in records]
            has_specified_c = any(c is not None for c in all_classes)
            has_unspecified_c = any(c is None for c in all_classes)
            classes = [c for c in all_classes if c is not None]
            if len(set(classes)) > 1:
                compatibility = MetricCompatibility.INCOMPATIBLE
                incomp_reason = f"Incompatible class counts: {min(classes)}-class task cannot be directly ranked against {max(classes)}-class task."
                warnings.append(incomp_reason)
            elif has_specified_c and has_unspecified_c:
                compatibility = MetricCompatibility.UNKNOWN
                incomp_reason = "Class count is unspecified for one or more papers. Direct ranking suppressed."
                warnings.append(incomp_reason)

        # Statistical Variance & Significance Check
        missing_ci = [r.paper_id for r in records if not r.confidence_interval]
        if missing_ci:
            warnings.append(
                f"Statistical variance (confidence intervals or std-dev) missing for: {', '.join(sorted(set(missing_ci)))}."
            )

        missing_p = [r.paper_id for r in records if not r.statistical_significance]
        if missing_p:
            warnings.append(
                f"Hypothesis testing p-values not reported for: {', '.join(sorted(set(missing_p)))}."
            )

        # Build comparison summary table
        rows: List[Dict[str, Any]] = []
        for r in records:
            rows.append({
                "paper_id": r.paper_id,
                "model_name": r.model_name,
                "metric_name": r.metric_name,
                "dataset_name": r.dataset_name or "Unknown",
                "dataset_version": r.dataset_version or "Standard/Unspecified",
                "dataset_split": r.dataset_split or "Unknown",
                "task": r.task or "Unspecified",
                "language": r.language or "Unspecified",
                "evaluation_protocol": r.evaluation_protocol or "Standard",
                "num_classes": r.num_classes if r.num_classes is not None else "N/A",
                "raw_value": str(r.raw_value),
                "normalized_value": r.normalized_value,
                "variant": r.variant or "standard",
                "confidence_interval": r.confidence_interval or "None reported",
                "statistical_significance": r.statistical_significance or "None reported",
                "page_number": r.page_number,
                "evidence_passage": r.evidence_passage,
            })

        # Rank models ONLY IF COMPATIBLE
        ranked: List[Dict[str, Any]] = []
        if compatibility == MetricCompatibility.COMPATIBLE:
            df = pd.DataFrame(rows)
            ascending = (goal == MetricGoal.LOWER_IS_BETTER)
            df_sorted = df.sort_values(by="normalized_value", ascending=ascending)

            best_val = df_sorted["normalized_value"].iloc[0] if len(df_sorted) > 0 else 0.0

            for rank_idx, (_, row) in enumerate(df_sorted.iterrows(), start=1):
                cur_val = float(row["normalized_value"]) if row["normalized_value"] is not None else 0.0
                delta = float(round(cur_val - float(best_val), 4))
                p_num = int(row["page_number"]) if pd.notna(row["page_number"]) and row["page_number"] is not None else None
                ranked.append({
                    "rank": int(rank_idx),
                    "paper_id": str(row["paper_id"]),
                    "model_name": str(row["model_name"]),
                    "metric_name": str(row["metric_name"]),
                    "score": cur_val,
                    "raw_value": str(row["raw_value"]),
                    "delta_from_best": delta,
                    "page_number": p_num,
                })
        else:
            warnings.append("Numerical ranking suppressed due to incompatible metric definitions or unverified evaluation conditions.")

        comparison_results.append(
            MetricComparisonResultSchema(
                metric_name=family,
                goal=goal,
                compatibility=compatibility,
                incompatibility_reason=incomp_reason,
                ranked_models=ranked,
                summary_table=rows,
                missing_data_warnings=warnings,
                trade_off_notes=trade_off_notes,
                notes=(
                    f"Goal: {goal.value}. "
                    f"Analyzed {len(records)} measurements across {len(distinct_papers)} papers."
                ),
            )
        )

    return comparison_results
