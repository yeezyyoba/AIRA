"""Evaluator for Research Paper Extraction and Dual-Engine PDF Parsing.

Measures:
- Field extraction accuracy (Title, Authors, Publication Year, Abstract, Methodology, Results)
- Page count integrity and section segmentation
- Metric and numerical extraction accuracy
- Scanned / image-only OCR detection accuracy
Evaluates across 4 archetypes:
1. Clean digital single-column paper
2. Two-column conference paper
3. Complex numerical tables paper
4. Scanned raster document (OCR required)
Zero live API quota consumed.
"""
import asyncio
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymupdf  # fitz

from evaluation.metrics import field_extraction_accuracy, text_token_overlap
from tools.paper_reader import read_research_paper
from validation.schemas import ExtractionStatus, PaperExtractionResultSchema


class PDFExtractionEvaluator:
    """Evaluates paper reader and PDF extractor against ground-truth benchmarks."""

    def __init__(self, benchmark_path: Optional[Path] = None):
        if benchmark_path is None:
            benchmark_path = Path(__file__).resolve().parent.parent / "benchmarks" / "pdf_extraction_benchmark.json"
        self.benchmark_path = benchmark_path
        with open(self.benchmark_path, "r", encoding="utf-8") as f:
            self.benchmark_data = json.load(f)

    def _generate_synthetic_pdf(self, item: Dict[str, Any]) -> bytes:
        """Programmatically synthesize a valid PDF matching the benchmark archetype using PyMuPDF."""
        gt = item["ground_truth"]
        doc = pymupdf.open()

        if item.get("paper_id") == "eval_pdf_digital_01":
            # 4 pages
            p1 = doc.new_page(width=595, height=842)
            p1.insert_text((50, 70), f"Title: {gt['title']}\n\nAuthors: {', '.join(gt['authors'])}\nPublished: {gt['publication_year']}\n\nAbstract\nThis paper investigates cross-lingual transfer and morphological complexity for low-resource Amharic language modeling.")
            p1.insert_text((50, 250), "1. Introduction\nRecent advancements in cross-lingual representation models have opened new avenues for low-resource NLP.")

            p2 = doc.new_page(width=595, height=842)
            p2.insert_text((50, 70), "2. Methodology\nWe deploy a 12-layer Transformer architecture combined with Byte-Pair Encoding (BPE) and iterative back-translation.")

            p3 = doc.new_page(width=595, height=842)
            p3.insert_text((50, 70), "3. Results and Empirical Evaluation\nIn experiments evaluated on the Amharic-English Parallel Corpus test split, the model achieved a 31.2 BLEU score over 50,000 sentence pairs.")

            p4 = doc.new_page(width=595, height=842)
            p4.insert_text((50, 70), "4. Conclusion and Future Work\nIn conclusion, our neural translation framework demonstrates effective transfer.")

        elif item.get("paper_id") == "eval_pdf_twocolumn_02":
            # 6 pages two-column
            p1 = doc.new_page(width=595, height=842)
            p1.insert_text((50, 60), f"{gt['title']}\n{', '.join(gt['authors'])} (2023)\n\nAbstract\nProcessing informal social media texts introduces significant orthographic variation and nested entities.")
            p1.insert_text((50, 200), "1. Introduction\nNamed entity recognition in Ethiopian social media contexts.")
            p1.insert_text((320, 200), "Social media text presents high orthographic variation and slang.")

            for i in range(2, 5):
                p = doc.new_page(width=595, height=842)
                p.insert_text((50, 70), f"Section {i}: Background & Data\nProcessing AmhSocialNER-v2 dataset.")
                p.insert_text((320, 70), "2. Methodology\nWe utilize AfroXLMR backbone with a CRF sequence layer and morphological feature injection.")

            p5 = doc.new_page(width=595, height=842)
            p5.insert_text((50, 70), "5. Results and Discussion\nOur model achieves a Macro-F1 of 84.6% on the AmhSocialNER-v2 test set with an annotated corpus of 12,500 posts.")

            p6 = doc.new_page(width=595, height=842)
            p6.insert_text((50, 70), "6. Conclusion\nFuture work will explore dialectal variants.")

        elif item.get("paper_id") == "eval_pdf_complextable_03":
            # 5 pages with complex table
            p1 = doc.new_page(width=595, height=842)
            p1.insert_text((50, 70), f"{gt['title']}\nBy {', '.join(gt['authors'])}\nPublished: {gt['publication_year']}\n\nAbstract\nWe evaluate unsupervised segmentation and root-and-pattern morphology for Ethiopic text classification.")

            for i in range(2, 4):
                p = doc.new_page(width=595, height=842)
                p.insert_text((50, 70), f"Section {i}: Methodology\nComparing Morfessor baseline with BPE 32k tokens and a convolutional neural network classifier.")

            p4 = doc.new_page(width=595, height=842)
            p4.insert_text((50, 70), "4. Results: Multi-Model Metric Comparison Table\n"
                                     "--------------------------------------------------\n"
                                     "Model               Dataset         Split  Accuracy  Macro-F1  Weighted-F1\n"
                                     "CNN + Morfessor     AmhNewsCorp-2022 test   89.4%     87.2%     88.9%\n"
                                     "Baseline LSTM       AmhNewsCorp-2022 test   82.1%     80.5%     81.3%\n"
                                     "--------------------------------------------------\n")

            p5 = doc.new_page(width=595, height=842)
            p5.insert_text((50, 70), "5. Conclusion\nMorphological segmenters significantly boost classification performance.")

        elif item.get("paper_id") == "eval_pdf_scanned_04":
            # 8 pages scanned / raster image only (trigger is_scanned)
            # Create pages with 0 or minimal text and an image pixmap
            pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 100), 0)
            pix.clear_with(255)
            img_bytes = pix.tobytes("png")

            for i in range(8):
                p = doc.new_page(width=595, height=842)
                # Insert image stream
                p.insert_image(pymupdf.Rect(50, 50, 500, 750), stream=img_bytes)
                # Scanned documents have near-zero text (e.g. empty or few noise chars)
                if i == 0:
                    p.insert_text((50, 30), "Manuscript scan 142a")

        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes

    async def evaluate_archetype(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate extraction for a single PDF benchmark archetype."""
        paper_id = item["paper_id"]
        category = item["category"]
        gt = item["ground_truth"]

        pdf_bytes = self._generate_synthetic_pdf(item)

        # Run extraction through paper_reader
        extraction_result: PaperExtractionResultSchema = await read_research_paper(
            paper_url_or_file=pdf_bytes,
            paper_id=paper_id,
            extraction_method="auto",
        )

        # Evaluate extraction dimensions
        page_count_correct = extraction_result.page_count == gt["page_count"]
        ocr_flag_correct = extraction_result.ocr_required == gt["ocr_required"]

        field_scores = {}
        if not gt["ocr_required"]:
            # Evaluate text fields
            ext_data = extraction_result.extracted_data
            if ext_data:
                # Title accuracy
                title_acc = field_extraction_accuracy(ext_data.title, gt["title"], threshold=0.5)
                # Authors accuracy
                authors_acc = field_extraction_accuracy(ext_data.authors, gt["authors"], threshold=0.5)
                # Year presence in document text/metadata
                p1_text = extraction_result.raw_text_by_page.get(1, "")
                year_acc = str(gt["publication_year"]) in p1_text or str(gt["publication_year"]) in str(extraction_result.document_metadata)
                # Abstract section presence
                abstract_text = ext_data.abstract or ""
                abstract_acc = any(kw.lower() in abstract_text.lower() for kw in gt["abstract_contains"])
                # Methodology presence
                meth_text = ext_data.methodology or ""
                meth_acc = any(kw.lower() in meth_text.lower() for kw in gt["methodology_contains"])
                # Results presence
                res_text = ext_data.results or ""
                res_acc = any(kw.lower() in res_text.lower() for kw in gt["results_contains"])

                field_scores = {
                    "title": 1.0 if title_acc else 0.0,
                    "authors": 1.0 if authors_acc else 0.0,
                    "publication_year": 1.0 if year_acc else 0.0,
                    "abstract": 1.0 if abstract_acc else 0.0,
                    "methodology": 1.0 if meth_acc else 0.0,
                    "results": 1.0 if res_acc else 0.0,
                }
            else:
                field_scores = {f: 0.0 for f in ["title", "authors", "publication_year", "abstract", "methodology", "results"]}
        else:
            # Scanned archetype: success is correctly identifying OCR required and not crashing
            field_scores = {
                "ocr_detection": 1.0 if ocr_flag_correct else 0.0,
                "graceful_handling": 1.0 if extraction_result.status in (ExtractionStatus.SUCCESS, ExtractionStatus.PARTIAL) else 0.0,
            }

        overall_score = sum(field_scores.values()) / max(len(field_scores), 1)

        return {
            "paper_id": paper_id,
            "category": category,
            "status": extraction_result.status.value,
            "extraction_method": extraction_result.extraction_method,
            "page_count_actual": extraction_result.page_count,
            "page_count_expected": gt["page_count"],
            "page_count_match": page_count_correct,
            "ocr_required_actual": extraction_result.ocr_required,
            "ocr_required_expected": gt["ocr_required"],
            "ocr_flag_match": ocr_flag_correct,
            "field_scores": field_scores,
            "archetype_score": round(overall_score, 4),
        }

    async def run(self) -> Dict[str, Any]:
        """Run complete PDF extraction evaluation across all benchmark archetypes."""
        archetype_results = []
        for item in self.benchmark_data:
            res = await self.evaluate_archetype(item)
            archetype_results.append(res)

        total_archetypes = len(archetype_results)
        if total_archetypes == 0:
            return {"error": "Empty PDF extraction benchmark dataset"}

        mean_score = sum(r["archetype_score"] for r in archetype_results) / total_archetypes
        ocr_accuracy = sum(1 for r in archetype_results if r["ocr_flag_match"]) / total_archetypes
        page_count_accuracy = sum(1 for r in archetype_results if r["page_count_match"]) / total_archetypes

        return {
            "evaluator": "PDFExtractionEvaluator",
            "benchmark_dataset": str(self.benchmark_path.name),
            "total_archetypes": total_archetypes,
            "aggregate_metrics": {
                "mean_extraction_score": round(mean_score, 4),
                "ocr_detection_accuracy": round(ocr_accuracy, 4),
                "page_count_accuracy": round(page_count_accuracy, 4),
            },
            "per_archetype_results": archetype_results,
        }
