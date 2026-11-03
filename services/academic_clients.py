"""Asynchronous Academic API Clients for OpenAlex, Semantic Scholar, and Crossref.

Provides resilient HTTP clients with bounded retries, rate-limit backoff,
DOI normalization, and strict metadata parsing without hallucination.
"""
import asyncio
import re
import string
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import httpx

from app.core.config import settings
from llm.usage_tracker import current_session_usage
from logging_config.logger import get_logger
from validation.schemas import PaperMetadataSchema

logger = get_logger("academic_clients")


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    """Normalize DOI to canonical lowercase format (e.g., '10.1145/3411764.3445518')."""
    if not doi or not isinstance(doi, str):
        return None
    cleaned = doi.strip().lower()
    # Strip common URL and prefix schemes
    prefixes = [
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi.org/",
        "dx.doi.org/",
        "doi:",
    ]
    for p in prefixes:
        if cleaned.startswith(p):
            cleaned = cleaned[len(p):].strip()
    # Basic DOI format sanity check (should start with 10.)
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", cleaned)
    if match:
        return match.group(0).rstrip(".,;")
    if cleaned.startswith("10."):
        return cleaned.rstrip(".,;")
    return None


def normalize_title(title: Optional[str]) -> str:
    """Normalize paper title for fuzzy/exact deduplication."""
    if not title:
        return ""
    # Lowercase, replace punctuation with spaces, collapse whitespace
    text = title.lower()
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    text = text.translate(translator)
    return " ".join(text.split())


def reconstruct_openalex_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
    """Reconstruct standard text abstract from OpenAlex inverted index."""
    if not inverted_index or not isinstance(inverted_index, dict):
        return None
    position_word_map: Dict[int, str] = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            position_word_map[pos] = word
    if not position_word_map:
        return None
    sorted_positions = sorted(position_word_map.keys())
    # Reassemble in order
    words = [position_word_map[pos] for pos in sorted_positions]
    return " ".join(words)


class AcademicClientError(Exception):
    """Base exception for academic API errors."""
    def __init__(self, message: str, provider: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class AcademicAuthError(AcademicClientError):
    """Authentication or authorization failure (permanent)."""
    pass


class AcademicRateLimitError(AcademicClientError):
    """Rate limit exceeded."""
    pass


class AcademicNotFoundError(AcademicClientError):
    """Resource not found (404)."""
    pass


class RequestPacer:
    """Process-wide minimum spacing between requests to one API.

    Loop-agnostic (a thread lock reserves the next slot; the wait itself is an
    ``asyncio.sleep``), so concurrent validations across sessions share one
    budget. A 429's ``Retry-After`` pushes the next slot out for everyone.
    """

    def __init__(self, name: str, interval_getter):
        self.name = name
        self._interval_getter = interval_getter
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def _interval(self) -> float:
        try:
            return max(0.0, float(self._interval_getter()))
        except (TypeError, ValueError):
            return 0.0

    async def wait_turn(self) -> float:
        """Wait for this caller's slot; returns the seconds waited."""
        interval = self._interval()
        if interval <= 0 and self._next_slot <= time.monotonic():
            return 0.0
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + interval
        delay = slot - now
        if delay > 0:
            await asyncio.sleep(delay)
        return delay

    def back_off(self, seconds: float) -> None:
        """Nobody calls this API again for ``seconds`` (a 429 told us so)."""
        with self._lock:
            self._next_slot = max(self._next_slot, time.monotonic() + max(0.0, seconds))

    def reset(self) -> None:
        with self._lock:
            self._next_slot = 0.0


# Semantic Scholar allows about one request per second (with or without a key).
semantic_scholar_pacer = RequestPacer(
    "Semantic Scholar", lambda: settings.SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS
)


def _retry_after_seconds(response: httpx.Response) -> Optional[float]:
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


class BaseAcademicClient:
    """Base client with resilient request execution and backoff."""

    # Optional process-wide pacer for APIs with a strict request rate.
    pacer: Optional[RequestPacer] = None

    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        self.timeout = float(settings.REQUEST_TIMEOUT_SECONDS)
        self.max_retries = int(settings.ACADEMIC_MAX_RETRIES)
        self.initial_backoff = float(settings.ACADEMIC_INITIAL_BACKOFF_SECONDS)

    async def _execute_request(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Execute HTTP request with bounded retry on transient errors."""
        last_exception: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                if self.pacer is not None:
                    await self.pacer.wait_turn()
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    timeout=self.timeout,
                )
                self._record_usage(response.status_code)

                # 1. Success
                if response.status_code == 200:
                    return response

                # 2. Permanent Client Errors (Do NOT retry)
                if response.status_code in (401, 403):
                    raise AcademicAuthError(
                        f"{self.provider_name} authentication failed (HTTP {response.status_code})",
                        provider=self.provider_name,
                        status_code=response.status_code,
                    )
                if response.status_code == 404:
                    raise AcademicNotFoundError(
                        f"{self.provider_name} resource not found (HTTP 404)",
                        provider=self.provider_name,
                        status_code=404,
                    )
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    raise AcademicClientError(
                        f"{self.provider_name} client error (HTTP {response.status_code}): {response.text[:200]}",
                        provider=self.provider_name,
                        status_code=response.status_code,
                    )

                # 3. Rate Limit (HTTP 429) -> Check Retry-After and retry
                if response.status_code == 429:
                    retry_after = _retry_after_seconds(response)
                    sleep_time = self.initial_backoff * (2 ** attempt)
                    if retry_after is not None:
                        sleep_time = max(retry_after, sleep_time)
                    sleep_time = min(sleep_time, 5.0)
                    if self.pacer is not None:
                        # Everyone waits, not just this request: the limit is per API key / IP.
                        self.pacer.back_off(max(sleep_time, retry_after or 0.0))
                        sleep_time = 0.0  # wait_turn() at the top of the loop does the waiting

                    if attempt < self.max_retries:
                        logger.warning(
                            f"{self.provider_name} rate limit (429). Retrying ({attempt + 1}/{self.max_retries}) in {sleep_time:.2f}s..."
                        )
                        await asyncio.sleep(sleep_time)
                        continue
                    else:
                        raise AcademicRateLimitError(
                            f"{self.provider_name} rate limit exceeded after {self.max_retries} retries",
                            provider=self.provider_name,
                            status_code=429,
                        )

                # 4. Server Errors (HTTP 5xx) -> Bounded retry
                if response.status_code >= 500:
                    sleep_time = min(self.initial_backoff * (2 ** attempt), 4.0)
                    if attempt < self.max_retries:
                        logger.warning(
                            f"{self.provider_name} server error ({response.status_code}). Retrying ({attempt + 1}/{self.max_retries}) in {sleep_time:.2f}s..."
                        )
                        await asyncio.sleep(sleep_time)
                        continue
                    else:
                        raise AcademicClientError(
                            f"{self.provider_name} server error {response.status_code} after retries: {response.text[:200]}",
                            provider=self.provider_name,
                            status_code=response.status_code,
                        )

            except (httpx.TimeoutException, httpx.NetworkError) as net_err:
                last_exception = net_err
                self._record_usage(None)
                sleep_time = min(self.initial_backoff * (2 ** attempt), 4.0)
                if attempt < self.max_retries:
                    logger.warning(
                        f"{self.provider_name} network/timeout error [{type(net_err).__name__}]. "
                        f"Retrying ({attempt + 1}/{self.max_retries}) in {sleep_time:.2f}s..."
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    logger.error(f"{self.provider_name} failed after {self.max_retries} retries: {net_err}")
                    raise AcademicClientError(
                        f"{self.provider_name} network/timeout error: {net_err}",
                        provider=self.provider_name,
                    ) from net_err

        if last_exception:
            raise AcademicClientError(f"{self.provider_name} error: {last_exception}", provider=self.provider_name)
        raise AcademicClientError(f"{self.provider_name} unexpected failure", provider=self.provider_name)

    def _record_usage(self, status_code: Optional[int]) -> None:
        session = current_session_usage()
        if session is not None:
            session.add_academic_request(
                # A 404 is an answer ("not indexed here"), not a failed request.
                self.provider_name, success=status_code in (200, 404), rate_limited=status_code == 429
            )

    async def _send_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> httpx.Response:
        """Send request using provided client or temporary client."""
        if client is not None:
            return await self._execute_request(client, method, url, headers=headers, params=params)
        async with httpx.AsyncClient() as http:
            return await self._execute_request(http, method, url, headers=headers, params=params)


class OpenAlexClient(BaseAcademicClient):
    """Client for OpenAlex API (https://api.openalex.org)."""

    def __init__(self, api_key: Optional[str] = None, mailto: Optional[str] = None):
        super().__init__("openalex")
        self.api_key = api_key or settings.OPENALEX_API_KEY
        self.mailto = mailto or settings.CROSSREF_MAILTO
        self.base_url = "https://api.openalex.org"

    def _get_headers_and_params(self) -> Tuple[Dict[str, str], Dict[str, Any]]:
        headers: Dict[str, str] = {
            "User-Agent": f"AIRA/0.1.0 (mailto:{self.mailto or 'researcher@example.com'})"
        }
        params: Dict[str, Any] = {}
        if self.mailto:
            params["mailto"] = self.mailto
        if self.api_key:
            params["api_key"] = self.api_key
        return headers, params

    async def search(
        self,
        query: str,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        max_results: int = 20,
        client: Optional[httpx.AsyncClient] = None,
    ) -> List[PaperMetadataSchema]:
        """Search OpenAlex works."""
        headers, params = self._get_headers_and_params()
        params["search"] = query
        params["per_page"] = min(max_results, 50)

        # Filters
        filters: List[str] = []
        if year_from and year_to:
            filters.append(f"publication_year:{year_from}-{year_to}")
        elif year_from:
            filters.append(f"from_publication_date:{year_from}-01-01")
        elif year_to:
            filters.append(f"to_publication_date:{year_to}-12-31")

        if filters:
            params["filter"] = ",".join(filters)

        url = f"{self.base_url}/works"
        resp = await self._send_request("GET", url, headers=headers, params=params, client=client)
        data = resp.json()

        results: List[PaperMetadataSchema] = []
        for item in data.get("results", []):
            parsed = self._parse_work(item)
            if parsed:
                results.append(parsed)
        return results

    async def get_by_doi(
        self,
        doi: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Optional[PaperMetadataSchema]:
        """Fetch work by DOI from OpenAlex."""
        clean_doi = normalize_doi(doi)
        if not clean_doi:
            return None

        headers, params = self._get_headers_and_params()
        url = f"{self.base_url}/works/https://doi.org/{clean_doi}"

        try:
            resp = await self._send_request("GET", url, headers=headers, params=params, client=client)
            return self._parse_work(resp.json())
        except AcademicNotFoundError:
            return None

    def _parse_work(self, item: Dict[str, Any]) -> Optional[PaperMetadataSchema]:
        title = item.get("title")
        if not title:
            return None

        authors: List[str] = []
        for authorship in item.get("authorships", []):
            author_obj = authorship.get("author", {})
            name = author_obj.get("display_name")
            if name:
                authors.append(name)

        clean_doi = normalize_doi(item.get("doi"))
        year = item.get("publication_year")

        # Abstract
        abstract = reconstruct_openalex_abstract(item.get("abstract_inverted_index"))

        # Venue
        venue = None
        primary_loc = item.get("primary_location") or {}
        source_obj = primary_loc.get("source") or {}
        if source_obj.get("display_name"):
            venue = source_obj.get("display_name")

        # URL & OA
        url = primary_loc.get("landing_page_url") or item.get("id")
        open_access = item.get("open_access") or {}
        is_oa = open_access.get("is_oa")
        oa_url = open_access.get("oa_url")
        best_oa_location = item.get("best_oa_location") or {}
        pdf_url = best_oa_location.get("pdf_url")
        citations = item.get("cited_by_count")

        paper_id = clean_doi or item.get("id") or f"openalex_{item.get('id', '').split('/')[-1]}"
        prov_dict = {
            "title": title.strip(),
            "authors": authors,
            "year": year,
            "doi": clean_doi,
            "venue": venue,
            "url": url,
            "citation_count": citations,
            "abstract": abstract,
            "is_open_access": is_oa,
            "open_access_url": oa_url,
            "pdf_url": pdf_url,
        }

        return PaperMetadataSchema(
            id=paper_id,
            title=title.strip(),
            authors=authors,
            year=year,
            doi=clean_doi,
            abstract=abstract,
            venue=venue,
            url=url,
            citation_count=citations,
            is_open_access=is_oa,
            open_access_url=oa_url,
            source="openalex",
            sources=["openalex"],
            provider_metadata={"openalex": prov_dict},
        )


class SemanticScholarClient(BaseAcademicClient):
    """Client for Semantic Scholar Graph API (https://api.semanticscholar.org)."""

    pacer = semantic_scholar_pacer

    def __init__(self, api_key: Optional[str] = None):
        super().__init__("semantic_scholar")
        self.api_key = api_key or settings.SEMANTIC_SCHOLAR_API_KEY
        self.base_url = "https://api.semanticscholar.org/graph/v1"

    def _get_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "User-Agent": "AIRA/0.1.0 (Amharic Intelligent Research Assistant)"
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    async def search(
        self,
        query: str,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        max_results: int = 20,
        client: Optional[httpx.AsyncClient] = None,
    ) -> List[PaperMetadataSchema]:
        """Search Semantic Scholar papers."""
        headers = self._get_headers()
        params: Dict[str, Any] = {
            "query": query,
            "limit": min(max_results, 50),
            "fields": "title,authors,year,externalIds,abstract,venue,url,citationCount,isOpenAccess,openAccessPdf",
        }

        if year_from and year_to:
            params["year"] = f"{year_from}-{year_to}"
        elif year_from:
            params["year"] = f"{year_from}-"
        elif year_to:
            params["year"] = f"-{year_to}"

        url = f"{self.base_url}/paper/search"
        resp = await self._send_request("GET", url, headers=headers, params=params, client=client)
        data = resp.json()

        results: List[PaperMetadataSchema] = []
        for item in data.get("data", []):
            parsed = self._parse_paper(item)
            if parsed:
                results.append(parsed)
        return results

    async def get_by_doi(
        self,
        doi: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Optional[PaperMetadataSchema]:
        """Fetch paper by DOI from Semantic Scholar."""
        clean_doi = normalize_doi(doi)
        if not clean_doi:
            return None

        headers = self._get_headers()
        params = {
            "fields": "title,authors,year,externalIds,abstract,venue,url,citationCount,isOpenAccess,openAccessPdf"
        }
        url = f"{self.base_url}/paper/DOI:{clean_doi}"

        try:
            resp = await self._send_request("GET", url, headers=headers, params=params, client=client)
            return self._parse_paper(resp.json())
        except AcademicNotFoundError:
            return None

    def _parse_paper(self, item: Dict[str, Any]) -> Optional[PaperMetadataSchema]:
        title = item.get("title")
        if not title:
            return None

        authors: List[str] = []
        for a in item.get("authors", []):
            name = a.get("name")
            if name:
                authors.append(name)

        ext_ids = item.get("externalIds") or {}
        raw_doi = ext_ids.get("DOI")
        clean_doi = normalize_doi(raw_doi)

        year = item.get("year")
        abstract = item.get("abstract") or None
        venue = item.get("venue") or None
        url = item.get("url")
        citations = item.get("citationCount")
        is_oa = item.get("isOpenAccess")

        oa_pdf = item.get("openAccessPdf") or {}
        oa_url = oa_pdf.get("url")

        paper_id = clean_doi or item.get("paperId") or f"s2_{item.get('paperId')}"
        prov_dict = {
            "title": title.strip(),
            "authors": authors,
            "year": year,
            "doi": clean_doi,
            "venue": venue,
            "url": url,
            "citation_count": citations,
            "abstract": abstract,
            "is_open_access": is_oa,
            "open_access_url": oa_url,
        }

        return PaperMetadataSchema(
            id=paper_id,
            title=title.strip(),
            authors=authors,
            year=year,
            doi=clean_doi,
            abstract=abstract,
            venue=venue,
            url=url,
            citation_count=citations,
            is_open_access=is_oa,
            open_access_url=oa_url,
            source="semantic_scholar",
            sources=["semantic_scholar"],
            provider_metadata={"semantic_scholar": prov_dict},
        )


class CrossrefClient(BaseAcademicClient):
    """Client for Crossref Works API (https://api.crossref.org)."""

    def __init__(self, mailto: Optional[str] = None):
        super().__init__("crossref")
        self.mailto = mailto or settings.CROSSREF_MAILTO
        self.base_url = "https://api.crossref.org"

    def _get_headers_and_params(self) -> Tuple[Dict[str, str], Dict[str, Any]]:
        headers = {
            "User-Agent": f"AIRA/0.1.0 (mailto:{self.mailto or 'researcher@example.com'})"
        }
        params: Dict[str, Any] = {}
        if self.mailto:
            params["mailto"] = self.mailto
        return headers, params

    async def search(
        self,
        query: str,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        max_results: int = 20,
        client: Optional[httpx.AsyncClient] = None,
    ) -> List[PaperMetadataSchema]:
        """Search Crossref works."""
        headers, params = self._get_headers_and_params()
        params["query"] = query
        params["rows"] = min(max_results, 50)

        # Filters
        filters: List[str] = []
        if year_from:
            filters.append(f"from-pub-date:{year_from}-01-01")
        if year_to:
            filters.append(f"until-pub-date:{year_to}-12-31")

        if filters:
            params["filter"] = ",".join(filters)

        url = f"{self.base_url}/works"
        resp = await self._send_request("GET", url, headers=headers, params=params, client=client)
        data = resp.json()

        results: List[PaperMetadataSchema] = []
        message = data.get("message", {})
        for item in message.get("items", []):
            parsed = self._parse_work(item)
            if parsed:
                results.append(parsed)
        return results

    async def get_by_doi(
        self,
        doi: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Optional[PaperMetadataSchema]:
        """Fetch authoritative record by DOI from Crossref."""
        clean_doi = normalize_doi(doi)
        if not clean_doi:
            return None

        headers, params = self._get_headers_and_params()
        url = f"{self.base_url}/works/{clean_doi}"

        try:
            resp = await self._send_request("GET", url, headers=headers, params=params, client=client)
            data = resp.json()
            return self._parse_work(data.get("message", {}))
        except AcademicNotFoundError:
            return None

    def _parse_work(self, item: Dict[str, Any]) -> Optional[PaperMetadataSchema]:
        titles = item.get("title", [])
        if not titles or not titles[0]:
            return None
        title = titles[0].strip()

        authors: List[str] = []
        for author in item.get("author", []):
            given = author.get("given", "").strip()
            family = author.get("family", "").strip()
            if given and family:
                authors.append(f"{given} {family}")
            elif family:
                authors.append(family)
            elif author.get("name"):
                authors.append(author.get("name").strip())

        clean_doi = normalize_doi(item.get("DOI"))

        # Year from published-print, published-online, or created
        year = None
        for date_key in ("published-print", "published-online", "published", "created"):
            date_parts = item.get(date_key, {}).get("date-parts", [[]])
            if date_parts and date_parts[0] and isinstance(date_parts[0][0], int):
                year = date_parts[0][0]
                break

        # Abstract (strip basic JATS XML tags if present)
        abstract = item.get("abstract")
        if abstract and isinstance(abstract, str):
            abstract = re.sub(r"<[^>]+>", "", abstract).strip() or None
        else:
            abstract = None

        venues = item.get("container-title", [])
        venue = venues[0].strip() if venues and venues[0] else None

        url = item.get("URL") or (f"https://doi.org/{clean_doi}" if clean_doi else None)
        pdf_url = next(
            (
                link.get("URL")
                for link in item.get("link", [])
                if isinstance(link, dict)
                and "pdf" in str(link.get("content-type", "")).lower()
                and isinstance(link.get("URL"), str)
                and link["URL"].startswith(("https://", "http://"))
            ),
            None,
        )
        citations = item.get("is-referenced-by-count")

        paper_id = clean_doi or url or f"crossref_{title[:30]}"
        prov_dict = {
            "title": title,
            "authors": authors,
            "year": year,
            "doi": clean_doi,
            "venue": venue,
            "url": url,
            "citation_count": citations,
            "abstract": abstract,
            "is_open_access": None,
            "open_access_url": None,
            "pdf_url": pdf_url,
        }

        return PaperMetadataSchema(
            id=paper_id,
            title=title,
            authors=authors,
            year=year,
            doi=clean_doi,
            abstract=abstract,
            venue=venue,
            url=url,
            citation_count=citations,
            is_open_access=None,
            open_access_url=None,
            source="crossref",
            sources=["crossref"],
            provider_metadata={"crossref": prov_dict},
        )
