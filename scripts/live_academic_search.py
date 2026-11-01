"""AIRA — Live Academic Search & Validation Demonstration.

Executes a live network search across OpenAlex, Semantic Scholar, and Crossref.
Does NOT invoke Gemini, Groq, or any LLM provider.
"""
import asyncio
import json
import sys

from tools.academic_search import search_academic_papers
from tools.metadata_validator import validate_paper_metadata


async def main():
    print("=" * 60)
    print("AIRA: Live Academic Paper Search (Zero LLM Invocation)")
    print("=" * 60)

    query = "Amharic Natural Language Processing"
    print(f"Query: '{query}'")
    print("Providers: OpenAlex, Semantic Scholar, Crossref\n")

    result = await search_academic_papers(
        query=query,
        sources=["openalex", "semantic_scholar", "crossref"],
        max_results=3,
    )

    print(f"Execution time:      {result['execution_time_seconds']}s")
    print(f"Total deduplicated:  {result['total_results']}")
    print(f"Raw results count:   {result['raw_results_count']}")
    print(f"Duplicates removed:  {result['duplicates_removed']}")
    print(f"Provider counts:     {result['provider_counts']}")
    print(f"Provider errors:     {result['provider_errors']}\n")

    print("-" * 60)
    print("Retrieved Papers (Live Network Results):")
    print("-" * 60)
    for i, paper in enumerate(result["papers"], 1):
        print(f"[{i}] {paper['title']}")
        print(f"    DOI:     {paper.get('doi')}")
        print(f"    Authors: {', '.join(paper.get('authors', []))}")
        print(f"    Year:    {paper.get('year')}")
        print(f"    Sources: {paper.get('sources')}")
        print(f"    Venue:   {paper.get('venue')}")
        print(f"    Citations: {paper.get('citation_count')}")
        if paper.get("conflicts"):
            print(f"    Discrepancies noted: {paper.get('conflicts')}")
        print()

    # If we found at least one paper with a DOI, test live metadata validation
    test_paper = next((p for p in result["papers"] if p.get("doi")), None)
    if test_paper:
        print("=" * 60)
        print("AIRA: Live Metadata Cross-Validation on Retrieved Paper")
        print("=" * 60)
        print(f"Validating DOI: {test_paper['doi']}")
        val_res = await validate_paper_metadata(test_paper)
        print(f"Validation Status: {val_res.status.value}")
        print(f"Explanation:       {val_res.explanation}")
        print(f"Sources Checked:   {val_res.sources_checked}")
        print("Field Comparisons:")
        for fname, comp in val_res.field_comparisons.items():
            print(f"  * {fname:8s} | match={comp.is_match} | {comp.details}")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
