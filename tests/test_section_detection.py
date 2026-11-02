"""Section detection: layout headings (primary) with the regex parser as fallback,
and the gap identifier's rule that author-acknowledged limitations only come from
Limitations / Discussion / Conclusion / Future Work sections.

All PDFs are generated in memory; nothing touches the network (see conftest.py).
"""
import threading
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pymupdf
import pytest

from services.paper_repository import PaperRepository
from tools import paper_reader
from tools.gap_identifier import identify_research_gaps, is_acknowledgement_heading
from tools.paper_reader import (
    _LayoutBox,
    _parse_heading,
    _parse_sections_with_layout,
    _reading_order,
    _split_body_headings,
    read_research_paper,
)
from validation.schemas import (
    EvidenceItemSchema,
    EvidenceType,
    ExtractedPaperDataSchema,
    ExtractionStatus,
    GapType,
    PaperExtractionResultSchema,
)

FILLER = (
    "We fine-tune multilingual transformer encoders on tweets in twelve African "
    "languages and report weighted F1 on the official test split."
)

Block = Tuple[str, str]  # ("h" heading | "p" paragraph, text)


def two_column_pdf(pages: List[Tuple[List[Block], List[Block]]]) -> bytes:
    """A two-column A4 paper: bold 12pt headings, 10pt body text."""
    doc = pymupdf.open()
    for left, right in pages:
        page = doc.new_page(width=595, height=842)
        for x0, blocks in ((50, left), (310, right)):
            y = 70
            for kind, text in blocks:
                if kind == "h":
                    rect = pymupdf.Rect(x0, y, x0 + 235, y + 40)
                    assert page.insert_textbox(rect, text, fontsize=12, fontname="hebo") >= 0
                    y += 24
                else:
                    rect = pymupdf.Rect(x0, y, x0 + 235, y + 200)
                    spare = page.insert_textbox(rect, text, fontsize=10, fontname="tiro")
                    assert spare >= 0
                    y += 200 - spare + 12
    data = doc.tobytes()
    doc.close()
    return data


def sample_paper() -> bytes:
    """Page 1: Abstract + 1 Introduction (left), 2 Method + 2.1 Dataset (right).
    Page 2: 3 Results + 4 Limitations (left), 5 Conclusion + References (right).
    Includes a bold footnote-style URL line, which must not become a heading, and an
    Introduction line starting with "Methods" that the regex parser would misread."""
    return two_column_pdf([
        (
            [("h", "Abstract"), ("p", "This paper studies sentiment analysis. " + FILLER),
             ("h", "1 Introduction"),
             ("p", "Introduction text starts here. Earlier work compared lexicon\nmethods with "
                   "neural models. " + FILLER)],
            [("h", "2 Method"), ("p", "Method text starts here. " + FILLER),
             ("h", "2.1 Dataset"), ("p", "Dataset text starts here. " + FILLER),
             ("h", "1https://github.com/example/repo")],
        ),
        (
            [("h", "3 Results"), ("p", "Results text starts here. " + FILLER),
             ("h", "4 Limitations"),
             ("p", "Limitations text starts here. Our models degrade on code-mixed tweets.")],
            [("h", "5 Conclusion"), ("p", "Conclusion text starts here. " + FILLER),
             ("h", "References"), ("p", "A. Author. 2023. A paper. In Proceedings.")],
        ),
    ])


def box(page: int, cls: str, x0: float, y0: float, x1: float, y1: float, text: str) -> _LayoutBox:
    """A layout box as pymupdf4llm would report it (A4 page, 595pt wide)."""
    markdown = f"### {text}" if cls == "section-header" else text
    return _LayoutBox(page=page, cls=cls, bbox=(x0, y0, x1, y1), page_width=595.0, markdown=markdown, raw=text)


# ==============================================================================
# Layout headings on a generated PDF (real pymupdf4llm run, no OCR, no network)
# ==============================================================================

@pytest.mark.anyio
async def test_layout_headings_give_clean_section_boundaries():
    result = await read_research_paper(sample_paper(), paper_id="two_col")

    assert result.status == ExtractionStatus.SUCCESS
    assert result.extraction_method == "pymupdf"
    assert result.page_count == 2
    assert not any("section detection instead" in w for w in result.warnings)
    data = result.extracted_data
    assert data is not None

    # Each section starts at its own first sentence and stops before the next heading.
    assert data.abstract.startswith("This paper studies sentiment analysis.")
    assert data.methodology.startswith("Method text starts here.")
    assert "Dataset text starts here." in data.methodology  # parent includes subsection 2.1
    assert "Results text" not in data.methodology
    assert data.dataset.startswith("Dataset text starts here.")
    assert data.results.startswith("Results text starts here.")
    assert "Limitations text" not in data.results
    assert data.limitations.startswith("Limitations text starts here.")
    assert data.conclusion.startswith("Conclusion text starts here.")
    assert "A. Author" not in data.conclusion  # References end the conclusion

    # The "Methods" line in the Introduction is not a section.
    assert "Introduction text" not in (data.methodology or "")
    # The footnote URL is not a heading.
    assert all("github" not in (ev.section_heading or "") for ev in data.evidence)

    pages = {ev.section_heading: ev.page_number for ev in data.evidence}
    assert pages["Abstract"] == 1
    assert pages["2 Method"] == 1
    assert pages["2.1 Dataset"] == 1
    assert pages["3 Results"] == 2
    assert pages["4 Limitations"] == 2
    assert pages["5 Conclusion"] == 2

    for ev in data.evidence:
        assert ev.evidence_type == EvidenceType.DIRECT_EXTRACT
        assert ev.passage in result.raw_text_by_page[ev.page_number]

    assert "future_work" in data.missing_sections
    assert data.future_work is None


@pytest.mark.anyio
async def test_layout_pass_runs_without_ocr():
    import pymupdf4llm

    calls = []
    real = pymupdf4llm.to_markdown

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    with patch.object(pymupdf4llm, "to_markdown", spy):
        result = await read_research_paper(sample_paper(), paper_id="ocr_check")
    assert result.status == ExtractionStatus.SUCCESS
    assert calls and all(kw.get("use_ocr") is False for kw in calls)


# ==============================================================================
# Reading order and numbering (pure functions on layout boxes)
# ==============================================================================

def test_reading_order_left_column_then_right_with_full_width_bands():
    shuffled = [
        box(1, "text", 310, 300, 545, 400, "right-bottom"),
        box(1, "text", 50, 500, 545, 560, "wide-table"),
        box(1, "text", 50, 300, 285, 400, "left-bottom"),
        box(1, "text", 310, 100, 545, 200, "right-top"),
        box(1, "text", 50, 40, 545, 60, "title"),
        box(1, "text", 50, 100, 285, 200, "left-top"),
        box(1, "text", 50, 600, 285, 700, "left-below-table"),
        box(1, "text", 310, 600, 545, 700, "right-below-table"),
    ]
    order = [b.raw for b in _reading_order(shuffled)]
    assert order == [
        "title", "left-top", "left-bottom", "right-top", "right-bottom",
        "wide-table", "left-below-table", "right-below-table",
    ]


def test_layout_order_error_is_corrected_so_subsections_keep_their_text():
    """pymupdf4llm emitted '4.2' (right column) before '4' and '4.1' (left column) on
    one real paper. Geometric ordering must restore 4 < 4.1 < 4.2 and keep 4.1's text."""
    emitted = [
        box(1, "section-header", 306, 236, 416, 245, "4.2 Evaluation Metric"),
        box(1, "text", 305, 255, 527, 347, "Metric text: weighted F1 over three classes."),
        box(1, "section-header", 70, 525, 199, 537, "4 Experiments"),
        box(1, "text", 70, 547, 291, 612, "Experiments text."),
        box(1, "section-header", 70, 624, 135, 632, "4.1 Datasets"),
        box(1, "text", 70, 641, 292, 733, "Dataset text: tweets in twelve languages."),
        box(1, "section-header", 306, 470, 435, 481, "4.3 Experimental Settings"),
        box(1, "text", 306, 486, 527, 524, "Settings text."),
    ]
    data, spans, problem = _parse_sections_with_layout(emitted, {1: ""}, "p")
    assert problem is None
    assert data.dataset == "Dataset text: tweets in twelve languages."
    assert data.evaluation_metrics == "Metric text: weighted F1 over three classes."
    assert data.experimental_setup.startswith("Experiments text.")
    assert spans["dataset"].start_page == spans["dataset"].end_page == 1


def test_out_of_order_numbering_is_reported_not_guessed():
    single_column = [
        box(1, "section-header", 50, 100, 200, 112, "5 Conclusion"),
        box(1, "text", 50, 120, 545, 200, "Conclusion text."),
        box(1, "section-header", 50, 300, 200, 312, "4.1 Datasets"),
        box(1, "text", 50, 320, 545, 400, "Dataset text."),
    ]
    data, spans, problem = _parse_sections_with_layout(single_column, {1: ""}, "p")
    assert data is None and spans == {}
    assert "out of order" in problem and "5 Conclusion" in problem and "4.1 Datasets" in problem


@pytest.mark.anyio
async def test_out_of_order_numbering_falls_back_to_regex_with_warning(monkeypatch):
    pdf = two_column_pdf([([("p", "Abstract\nShort abstract.\nResults\nAccuracy is 91.2%.")], [])])
    monkeypatch.setattr(paper_reader, "_extract_layout_boxes", lambda _: [
        box(1, "section-header", 50, 100, 200, 112, "5 Conclusion"),
        box(1, "section-header", 50, 300, 200, 312, "4.1 Datasets"),
    ])
    result = await read_research_paper(pdf, paper_id="bad_order")
    assert result.status == ExtractionStatus.SUCCESS
    assert any("out of order" in w and "section detection instead" in w for w in result.warnings)
    assert result.extracted_data.results is not None and "91.2%" in result.extracted_data.results


# ==============================================================================
# Heading acceptance
# ==============================================================================

@pytest.mark.parametrize("text", [
    "4.1 Datasets", "3 System Description", "5.3 mDeBERTaV3 base", "Abstract", "Limitations",
    "Results and Discussion", "Conclusion and Future Work", "References",
])
def test_numbered_or_known_headings_are_accepted(text):
    assert _parse_heading(text, page=1, back_matter=False) is not None


@pytest.mark.parametrize("text", [
    "The datasets are available via GitHub",  # unnumbered, not a section name
    "Task A: Multilingual Sentiment Classification",
    "12https://github.com/afrisenti-semeval/afrisent-semeval-2023",  # footnote URL
    "https://example.org/data",
    "1 We fine-tune every model for three epochs with a learning rate of 2e-5.",  # sentence
    "",
])
def test_false_headings_are_rejected(text):
    assert _parse_heading(text, page=1, back_matter=False) is None


def test_appendix_letters_only_count_after_references():
    assert _parse_heading("A Best Dataset", page=9, back_matter=True) is not None
    assert _parse_heading("A Best Dataset", page=2, back_matter=False) is None


def test_body_font_headings_are_split_out_of_text_boxes():
    segments = _split_body_headings("FastAPI endpoint test.\nMethodology\nTesting the HTTP route.")
    assert segments == [
        ("text", "FastAPI endpoint test."),
        ("heading", "Methodology"),
        ("text", "Testing the HTTP route."),
    ]
    # A wrapped line that happens to be a section word is body text.
    assert all(kind == "text" for kind, _ in _split_body_headings("we compare the\nResults\nof both."))
    # Generic single words (table cells) are not headings unless numbered.
    assert all(kind == "text" for kind, _ in _split_body_headings("Scores.\nModel\nF1"))
    assert ("heading", "4 Model") in _split_body_headings("Scores.\n4\nModel\nWe use XLM-R.")
    # Number and title on separate lines, as PyMuPDF often reports them.
    assert ("heading", "4.1 Datasets") in _split_body_headings("4.1\nDatasets\nTweets.")
    # Run-in abstract.
    assert _split_body_headings("Abstract—We study X.")[:2] == [("heading", "Abstract"), ("text", "We study X.")]


# ==============================================================================
# Fallback to the regex parser
# ==============================================================================

REGEX_PAGE = "Abstract\nShort abstract.\n\nMethodology\nWe use a CNN.\n\nResults\nAccuracy is 91.2%.\n"


def plain_pdf(text: str) -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((50, 72), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.mark.anyio
async def test_no_layout_headings_uses_regex_parser_silently(monkeypatch):
    monkeypatch.setattr(paper_reader, "_extract_layout_boxes", lambda _: [])
    result = await read_research_paper(plain_pdf(REGEX_PAGE), paper_id="no_headings")
    assert result.status == ExtractionStatus.SUCCESS
    assert result.extracted_data.methodology == "We use a CNN."
    assert not any("section detection" in w for w in result.warnings)


@pytest.mark.anyio
async def test_layout_failure_falls_back_to_regex_parser(monkeypatch):
    def boom(_):
        raise RuntimeError("layout model unavailable")

    monkeypatch.setattr(paper_reader, "_extract_layout_boxes", boom)
    result = await read_research_paper(plain_pdf(REGEX_PAGE), paper_id="layout_fails")
    assert result.status == ExtractionStatus.SUCCESS
    assert result.extracted_data.methodology == "We use a CNN."
    assert any("failed" in w and "layout model unavailable" in w for w in result.warnings)


@pytest.mark.anyio
async def test_layout_timeout_falls_back_within_pdf_timeout(monkeypatch):
    release = threading.Event()

    def hang(_):
        release.wait(10)
        return []

    monkeypatch.setattr(paper_reader, "_extract_layout_boxes", hang)
    try:
        with patch("app.core.config.settings.PDF_PROCESSING_TIMEOUT", 2.0):
            result = await read_research_paper(plain_pdf(REGEX_PAGE), paper_id="layout_slow")
    finally:
        release.set()  # free the layout worker for later tests
    assert result.status == ExtractionStatus.SUCCESS
    assert result.extracted_data.methodology == "We use a CNN."
    assert any("exceeded" in w for w in result.warnings)


# ==============================================================================
# Gap identifier: author-acknowledged limitations only from the right sections
# ==============================================================================

@pytest.mark.parametrize("heading", [
    "Limitations", "6 Limitations", "Discussion", "5 Results and Discussion",
    "7 Conclusion", "Conclusions and Future Work", "Future Work", "Discussion and Limitations",
    "Limitations and Ethical Considerations", "Threats to Validity",
])
def test_acknowledgement_headings_accepted(heading):
    assert is_acknowledgement_heading(heading)


@pytest.mark.parametrize("heading", [
    "3 Methodology", "Methods", "4 Experimental Setup", "limitation of computation resource languages, we",
    "Limitations of prior lexicon methods were addressed by", None, "",
])
def test_non_acknowledgement_headings_rejected(heading):
    assert not is_acknowledgement_heading(heading)


def _paper(pid: str, limitations: str, heading: Optional[str], passage: Optional[str] = None):
    evidence = []
    if heading is not None:
        evidence.append(EvidenceItemSchema(
            paper_id=pid, page_number=4, section_heading=heading,
            passage=passage or f"{heading}\n{limitations}", evidence_type=EvidenceType.DIRECT_EXTRACT,
        ))
    return PaperExtractionResultSchema(
        status=ExtractionStatus.SUCCESS, extraction_method="pymupdf", paper_id=pid, page_count=6,
        extracted_data=ExtractedPaperDataSchema(paper_id=pid, limitations=limitations, evidence=evidence),
    )


def _router():
    router = MagicMock()
    router.is_degraded_mode.return_value = False
    return router


@pytest.mark.anyio
async def test_limitations_from_a_methods_line_are_not_reported():
    repo = PaperRepository()  # conftest points it at a temporary directory
    # What the regex parser produced for a real paper: a Methods sentence starting
    # with "limitation" became the limitations "section".
    repo.save_paper(_paper(
        "methods_line", "divide the training into 5 folds and train the LaBSE model on noisy domain data",
        "limitation of computation resource languages, we",
    ))
    # Linked by passage, recorded under a Methods heading.
    text = "we found noise in compute budgets for low-resource scale experiments"
    repo.save_paper(_paper("methods_heading", text, "3 Methodology", passage=f"3 Methodology\n{text}"))
    repo.save_paper(_paper("real_section", "Our models degrade on noisy code-mixed domain text.", "6 Limitations"))

    result = await identify_research_gaps(
        ["methods_line", "methods_heading", "real_section"], "Sentiment", repository=repo, router=_router(),
    )
    acknowledged = [g for g in result.gaps if g.gap_type == GapType.AUTHOR_ACKNOWLEDGED_GAP]
    assert [g.supporting_papers for g in acknowledged] == [["real_section"]]
    assert acknowledged[0].evidence[0].section_heading == "6 Limitations"
    # Rejected text must not feed the shared-limitation direction either (needs >= 2 papers).
    assert not any(g.gap_type == GapType.POTENTIAL_RESEARCH_DIRECTION for g in result.gaps)


@pytest.mark.anyio
async def test_limitations_from_discussion_or_conclusion_sections_are_reported():
    repo = PaperRepository()
    repo.save_paper(_paper("disc", "The approach does not generalize to noisy speech domains.", "5 Discussion and Limitations"))
    repo.save_paper(_paper("conc", "Results are limited by noise in low-resource annotation.", "7 Conclusion"))
    result = await identify_research_gaps(["disc", "conc"], "ASR", repository=repo, router=_router())
    acknowledged = {g.supporting_papers[0] for g in result.gaps if g.gap_type == GapType.AUTHOR_ACKNOWLEDGED_GAP}
    assert acknowledged == {"disc", "conc"}
    assert any(g.gap_type == GapType.POTENTIAL_RESEARCH_DIRECTION for g in result.gaps)


@pytest.mark.anyio
async def test_regex_fallback_methods_limitation_is_not_an_author_gap(monkeypatch):
    """End to end: force the regex parser on a PDF whose Methods section has a line that
    starts with 'Limitation'; the reader may mislabel it, the gap identifier must not use it."""
    monkeypatch.setattr(paper_reader, "_extract_layout_boxes", lambda _: [])
    text = (
        "Abstract\nWe study sentiment analysis.\n\n"
        "Methodology\nBecause of the\nlimitation of computation resources, we\ntrain five folds.\n\n"
        "Conclusion\nEnsembles help.\n"
    )
    result = await read_research_paper(plain_pdf(text), paper_id="fallback_methods")
    # Precondition: the regex parser does mislabel the Methods line as "limitations".
    assert result.extracted_data.limitations is not None
    repo = PaperRepository()
    repo.save_paper(result)
    gaps = await identify_research_gaps(["fallback_methods"], "Sentiment", repository=repo, router=_router())
    assert not any(g.gap_type == GapType.AUTHOR_ACKNOWLEDGED_GAP for g in gaps.gaps)
