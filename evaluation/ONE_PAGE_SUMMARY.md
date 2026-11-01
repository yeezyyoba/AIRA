# AIRA — Evaluation Summary

## 1. Evaluation Overview

AIRA (Amharic Intelligent Research Assistant) was evaluated across six areas: academic paper search, PDF extraction, retrieval-augmented generation (RAG), research analysis, agentic behavior, and reliability.

The evaluation framework uses deterministic benchmark fixtures and mocked external services where appropriate. Its purpose is to verify core functionality, measure component-level performance, and identify limitations.

## 2. Evaluation Results

| Evaluation Area   |                    Result | Interpretation                          |
| ----------------- | ------------------------: | --------------------------------------- |
| Academic Search   |         Precision@5: 0.62 | Controlled benchmark ranking            |
| Academic Search   |            Recall@5: 1.00 | Relevant benchmark candidates retrieved |
| Academic Search   |               MRR: 0.8667 | Ranking performance                     |
| Academic Search   |              nDCG: 0.8546 | Ranking quality                         |
| PDF Extraction    |    Field accuracy: 0.8889 | Digital PDF extraction                  |
| PDF Extraction    | Page-count accuracy: 100% | Correct page counts                     |
| PDF Extraction    |   OCR flag accuracy: 100% | Scanned-document detection              |
| RAG               |  Correct-paper rate: 100% | Six evaluated questions                 |
| RAG               |    Correct-page rate: 50% | Six evaluated questions                 |
| Research Analysis |                       5/5 | Component challenge cases passed        |
| Agentic Behavior  |                       6/6 | Node and routing checks passed          |
| Reliability       |                       4/4 | Narrow fault-injection checks passed    |

## 3. Key Findings

**Academic Search:** The search pipeline achieved an nDCG of 0.8546 and MRR of 0.8667 against a controlled benchmark candidate pool. These results assess search, merging, deduplication, and ranking behavior rather than live academic discovery quality.

**PDF Extraction:** Digital PDF field extraction accuracy was 88.89%. Page-count and scanned-document detection checks achieved 100% accuracy. Scanned-document handling does not represent OCR transcription accuracy.

**RAG:** Six of eight benchmark questions were evaluated. The correct-paper rate was 100%, while the correct-page rate was 50%. The benchmark used synthetic documents and mock embeddings, so it does not establish production semantic retrieval quality.

**Research Analysis:** All five component challenge cases passed. These tests cover selected analysis and validation behaviors, not every possible scientific comparison.

**Agentic Behavior:** All six node and routing scenarios passed. These tests verify selected workflow components and conditional paths rather than complete end-to-end research tasks.

**Reliability:** All four targeted fault-injection checks passed, covering degraded-mode behavior, partial provider failure, corrupted PDF handling, and invalid input.

## 4. Failures and Limitations

The evaluation identified several important limitations:

- OCR transcription for scanned PDFs is not implemented.
- RAG evaluation uses mock embeddings rather than production semantic embeddings.
- Correct-page retrieval remains limited, with a 50% rate in the evaluated benchmark.
- Evidence-grounding metrics are proxies and do not guarantee factual correctness.
- Some agentic and analysis evaluations are component-level checks rather than complete workflow evaluations.
- Live academic search performance has not been established by the controlled search benchmark.
- LangGraph currently uses in-process `MemorySaver` checkpointing, which is not durable across process restarts.

## 5. Reliability and Regression Testing

The last recorded full backend regression suite completed with:

**197 tests passed, 7 warnings.**

The frontend production build also completed successfully during Phase 8 verification.

The offline evaluation runner completed all six evaluation areas. The evaluation fixtures do not require live LLM quota, although not every application workflow or project test is guaranteed to run offline.

## 6. Future Improvements

1. Integrate OCR for scanned academic documents.
2. Evaluate RAG using real production embeddings.
3. Expand English and Amharic retrieval benchmarks.
4. Improve page-level and claim-level evidence verification.
5. Add durable LangGraph checkpoint persistence.
6. Expand end-to-end agent testing and real-world research scenarios.
7. Increase the number and diversity of academic papers used in evaluation.

## 7. Conclusion

AIRA demonstrates a functioning agentic research architecture with academic search, document processing, retrieval, analysis, citation generation, provider fallback, and conditional workflow orchestration.

The evaluation confirms that the tested components operate under the defined benchmark conditions. However, the results should not be interpreted as proof of broad real-world research accuracy. Production embedding evaluation, OCR, durable checkpointing, and more comprehensive end-to-end testing remain important next steps.

**Evaluation scope:** Offline, deterministic, and component-focused.
**Detailed results:** `evaluation/evaluation_report.md`
**Evaluation data:** `evaluation/evaluation_results.json`
