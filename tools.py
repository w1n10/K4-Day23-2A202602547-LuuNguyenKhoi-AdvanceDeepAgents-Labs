"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
SUMMARY_CHARS = 600
FETCH_CHARS = 12_000
ARXIV_MIN_INTERVAL = 3.0       # arXiv API etiquette: at least 3 s between two calls
ARXIV_COOLDOWN = 600.0         # after arXiv kept answering 429, fail fast for this long instead of waiting again
QUOTA_EXHAUSTED_AFTER = 600    # a Retry-After longer than this is a spent daily quota, not a burst: do not wait
ATOM = {"a": "http://www.w3.org/2005/Atom"}
HTTP = httpx.Client(timeout=httpx.Timeout(60.0, connect=15.0), follow_redirects=True,
                    headers={"User-Agent": "deep-research-lab/1.0"})


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0, sleep=None):
    """Call fn(); when it raises RetryableError, wait and call it again.

    Waits `retry_after` seconds when the server said so, else exponential backoff base * 2**attempt plus random
    jitter; every wait is capped at `cap`. The last failure is re-raised without sleeping. Any other exception is
    not retried. `sleep` is injectable for tests (default: time.sleep, looked up at call time).
    """
    sleep = sleep or time.sleep
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            if exc.retry_after is not None:
                delay = min(float(exc.retry_after), cap)
            else:
                backoff = base * 2 ** attempt
                delay = min(backoff + random.uniform(0, backoff), cap)
            sleep(delay)


def _retry_after(response):
    """Seconds from a numeric Retry-After header, else None."""
    try:
        return max(0.0, float(response.headers.get("Retry-After", "")))
    except ValueError:
        return None


def _get(url, params):
    """One GET: 429/5xx and network errors -> RetryableError; any other HTTP error raises; returns the response."""
    try:
        response = HTTP.get(url, params=params)
    except httpx.TransportError as exc:
        raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
    if response.status_code in RETRYABLE_STATUS:
        raise RetryableError(f"HTTP {response.status_code} from {url}", _retry_after(response))
    response.raise_for_status()
    return response


def _clean(text, limit=None):
    """Collapse whitespace/newlines; cut to `limit` characters."""
    text = " ".join(str(text or "").split())
    return text[:limit].rstrip() + "..." if limit and len(text) > limit else text


def _error(exc):
    return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 2: arXiv ----
_arxiv_lock = threading.Lock()   # researchers run in parallel threads: serialise arXiv calls to keep the 3 s gap
_arxiv_last_call = 0.0
_arxiv_blocked_until = 0.0     # circuit breaker: arXiv rate-limits per IP, sometimes for a long time


def _arxiv_get(params):
    global _arxiv_last_call
    with _arxiv_lock:
        wait = ARXIV_MIN_INTERVAL - (time.monotonic() - _arxiv_last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            return _get(ARXIV_URL, params)
        finally:
            _arxiv_last_call = time.monotonic()


def _arxiv_records(xml_text):
    records = []
    for entry in ET.fromstring(xml_text).findall("a:entry", ATOM):
        raw_id = entry.findtext("a:id", "", ATOM).strip()
        if "/abs/" not in raw_id:
            continue
        paper_id = re.sub(r"v\d+$", "", raw_id.split("/abs/")[-1])
        records.append({
            "id": paper_id,
            "url": f"https://arxiv.org/abs/{paper_id}",
            "published": entry.findtext("a:published", "", ATOM)[:10],
            "title": _clean(entry.findtext("a:title", "", ATOM)),
            "summary": _clean(entry.findtext("a:summary", "", ATOM), SUMMARY_CHARS),
        })
    return records


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by a few keywords (e.g. "world model video"), newest first. Use it for recent research
    papers. Returns a JSON list of {id, url, published, title, summary}, "NO RESULTS", or "ERROR: ..." (then try
    another source)."""
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", query or "")
    terms = [t for t in terms if t.upper() not in {"AND", "OR", "ANDNOT", "ALL", "TI", "ABS"}][:8]
    if not terms:
        return "NO RESULTS"
    global _arxiv_blocked_until
    if time.monotonic() < _arxiv_blocked_until:
        return ("ERROR: arXiv is rate-limiting this IP (HTTP 429); do not call arxiv_search again now, "
                "use hf_search_papers, hf_daily_papers or web_search instead")
    params = {"search_query": " AND ".join(f"all:{t}" for t in terms), "sortBy": "submittedDate",
              "sortOrder": "descending", "start": 0, "max_results": max(1, min(int(max_results), 30))}
    try:
        response = with_retry(lambda: _arxiv_get(params), attempts=6, base=3.0, cap=60.0)
        records = _arxiv_records(response.text)
    except RetryableError as exc:  # still rate limited / down after every retry: open the circuit breaker
        _arxiv_blocked_until = time.monotonic() + ARXIV_COOLDOWN
        return _error(exc)
    except Exception as exc:  # a tool never raises
        return _error(exc)
    return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"


# ---- TODO 3: Hugging Face ----
def _hf_records(items):
    records = []
    for item in items if isinstance(items, list) else []:
        paper = (item or {}).get("paper") or {}
        paper_id = paper.get("id")
        if not paper_id:
            continue
        records.append({
            "id": paper_id,
            "url": f"https://huggingface.co/papers/{paper_id}",
            "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
            "title": _clean(paper.get("title") or item.get("title")),
            "summary": _clean(paper.get("ai_summary") or paper.get("summary") or item.get("summary"), SUMMARY_CHARS),
            "upvotes": paper.get("upvotes") or 0,
            "github": paper.get("githubRepo") or "",
            "stars": paper.get("githubStars") or 0,
        })
    return records


@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    params = {"limit": max(1, min(int(limit), 100))}
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", (date or "").strip()):
        params["date"] = date.strip()
    try:
        records = _hf_records(with_retry(lambda: _get(HF_DAILY_URL, params)).json())
    except Exception as exc:
        return _error(exc)
    needle = (keyword or "").strip().lower()
    if needle:
        records = [r for r in records if needle in f"{r['title']} {r['summary']}".lower()]
    records.sort(key=lambda r: r["upvotes"], reverse=True)
    return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic (a short phrase, e.g. "world models for robotics"). Good for popular,
    community-upvoted papers, often with code. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}, "NO RESULTS", or "ERROR: ..."."""
    if not (query or "").strip():
        return "NO RESULTS"
    params = {"q": query.strip(), "limit": max(1, min(int(limit), 50))}
    try:
        records = _hf_records(with_retry(lambda: _get(HF_SEARCH_URL, params)).json())[:params["limit"]]
    except Exception as exc:
        return _error(exc)
    return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _exa_key():
    return (os.getenv("EXA_API_KEY") or "").strip()


def _redact(text):
    """Remove the Exa key from any text that goes back to the agent (httpx errors quote the full URL)."""
    text, key = str(text), _exa_key()
    if key:
        text = text.replace(key, "***")
    return re.sub(r"(exaApiKey=)[^&\s\"']+", r"\1***", text)


def _mentions_rate_limit(value):
    return "rate limit" in str(value).lower()


def _mcp_message(response):
    """The JSON-RPC message of an MCP answer: plain JSON, or the `data:` line of a server-sent event."""
    if "text/event-stream" not in response.headers.get("content-type", ""):
        return response.json()
    for line in response.text.splitlines():
        if line.startswith("data:"):
            return json.loads(line[5:].strip())
    raise ValueError("no data line in the Exa event stream")


def _exa_call_once(name, arguments):
    url = f"{EXA_URL}?exaApiKey={_exa_key()}" if _exa_key() else EXA_URL
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
    try:
        response = HTTP.post(url, json=body, headers={"Accept": "application/json, text/event-stream"})
    except httpx.TransportError as exc:
        raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
    if response.status_code in RETRYABLE_STATUS:
        retry_after = _retry_after(response)
        if retry_after is not None and retry_after > QUOTA_EXHAUSTED_AFTER:
            raise RuntimeError(f"Exa quota exhausted (HTTP {response.status_code}, retry after {retry_after:.0f}s); "
                               "set EXA_API_KEY")
        raise RetryableError(f"HTTP {response.status_code} from Exa", retry_after)
    response.raise_for_status()
    message = _mcp_message(response)
    if "error" in message:
        error = message["error"]
        detail = error.get("message", error) if isinstance(error, dict) else error
        if _mentions_rate_limit(detail):
            raise RetryableError(f"Exa rate limited: {detail}")
        raise RuntimeError(f"Exa error: {detail}")
    result = message.get("result") or {}
    meta = result.get("_meta") or {}
    # Free tier: HTTP 200 whose text is a rate-limit notice, flagged in result._meta. It is NOT page content.
    if any("rate" in str(k).lower() and v for k, v in meta.items()) or (
            result.get("isError") and _mentions_rate_limit(result.get("content"))):
        raise RetryableError("Exa rate limited (flag in result._meta)")
    text = "\n\n".join(c.get("text", "") for c in result.get("content") or [] if c.get("type") == "text").strip()
    if result.get("isError"):
        raise RuntimeError(f"Exa tool error: {text[:300]}")
    return text


def _exa_call(name, arguments, limit=None):
    try:
        text = with_retry(lambda: _exa_call_once(name, arguments), attempts=6, base=5.0, cap=60.0)
    except Exception as exc:
        return _redact(_error(exc))
    if not text:
        return "NO RESULTS"
    return text[:limit] + "\n[... truncated]" if limit and len(text) > limit else text


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language (e.g. "blog post explaining JEPA world
    models"); `objective` says what you need from it. Use it for surveys, blogs, project pages and news. Returns clean
    text of the top results with their URLs, "NO RESULTS", or "ERROR: ..."."""
    if not (query or "").strip():
        return "NO RESULTS"
    arguments = {"query": query.strip(),
                 "objective": (objective or "").strip() or f"Find authoritative pages about: {query.strip()}",
                 "numResults": max(1, min(int(num_results), 10))}
    return _exa_call("web_search_exa", arguments)


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated
    to about 12000 characters. Returns the page text, "NO RESULTS", or "ERROR: ..."."""
    if not str(url or "").startswith(("http://", "https://")):
        return "ERROR: ValueError: url must start with http:// or https://"
    return _exa_call("web_fetch_exa", {"urls": [url]}, limit=FETCH_CHARS)


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
