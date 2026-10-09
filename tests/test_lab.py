"""Offline tests (no network, no LLM):   pip install pytest && python -m pytest -q"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tools  # noqa: E402
from check_citations import check  # noqa: E402
from finalize_citations import finalize  # noqa: E402
from research import save_outputs, slugify  # noqa: E402
from tools import RetryableError, with_retry  # noqa: E402

SOURCES = [
    {"n": 1, "id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001", "title": "A", "date": "2025-01-02",
     "source": "arxiv"},
    {"n": 2, "id": "2502.00002", "url": "https://huggingface.co/papers/2502.00002", "title": "B", "date": "2025-02-02",
     "source": "hf-search"},
]
GOOD = ("# T\n\nClaim [1]. Other [2].\n\n## References\n"
        "[1] A. arxiv. https://arxiv.org/abs/2501.00001 (2025-01-02)\n"
        "[2] B. hf-search. https://huggingface.co/papers/2502.00002 (2025-02-02)\n")


# ---- with_retry ----
def failing(times, retry_after=None):
    state = {"calls": 0}

    def fn():
        state["calls"] += 1
        if state["calls"] <= times:
            raise RetryableError("busy", retry_after)
        return "ok"
    return fn, state


def test_retry_succeeds_after_failures():
    sleeps = []
    fn, state = failing(2)
    assert with_retry(fn, attempts=5, base=1, cap=30, sleep=sleeps.append) == "ok"
    assert state["calls"] == 3 and len(sleeps) == 2
    assert 1 <= sleeps[0] <= 2 and 2 <= sleeps[1] <= 4   # exponential backoff + jitter


def test_retry_gives_up_without_sleeping_after_last_attempt():
    sleeps = []
    fn, state = failing(10)
    with pytest.raises(RetryableError):
        with_retry(fn, attempts=3, sleep=sleeps.append)
    assert state["calls"] == 3 and len(sleeps) == 2


def test_retry_respects_retry_after_and_cap():
    sleeps = []
    fn, _ = failing(2, retry_after=7)
    with_retry(fn, cap=30, sleep=sleeps.append)
    assert sleeps == [7, 7]
    sleeps.clear()
    fn, _ = failing(1, retry_after=500)
    with_retry(fn, cap=60, sleep=sleeps.append)
    assert sleeps == [60]


def test_retry_does_not_retry_other_errors():
    sleeps = []

    def boom():
        raise ValueError("bug")
    with pytest.raises(ValueError):
        with_retry(boom, sleep=sleeps.append)
    assert sleeps == []


# ---- tools (network-free paths) ----
def test_arxiv_empty_query_does_not_call_network(monkeypatch):
    monkeypatch.setattr(tools, "_arxiv_get", lambda params: pytest.fail("network called"))
    assert tools.arxiv_search.invoke({"query": "'\":: AND"}) == "NO RESULTS"


def test_arxiv_parses_atom(monkeypatch):
    xml = """<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2501.00001v2</id>
      <published>2025-01-02T00:00:00Z</published><title>World
      Models</title><summary>  A   summary.</summary></entry></feed>"""

    class Response:
        text = xml
    monkeypatch.setattr(tools, "_arxiv_get", lambda params: Response())
    records = json.loads(tools.arxiv_search.invoke({"query": "world model"}))
    assert records == [{"id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001", "published": "2025-01-02",
                        "title": "World Models", "summary": "A summary."}]


def test_tool_returns_error_string_instead_of_raising(monkeypatch):
    def down(url, params):
        raise RetryableError("HTTP 503")
    monkeypatch.setattr(tools, "_get", down)
    monkeypatch.setattr(tools.time, "sleep", lambda s: None)
    assert tools.hf_search_papers.invoke({"query": "x"}).startswith("ERROR:")


def test_exa_key_is_redacted(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "secret-key-123")
    assert "secret-key-123" not in tools._redact("GET https://mcp.exa.ai/mcp?exaApiKey=secret-key-123 failed")


# ---- slugify ----
@pytest.mark.parametrize("topic,slug", [
    ("survey about world model", "survey-about-world-model"),
    ("../../x", "x"),
    ("", "topic"),
    ("!!!", "topic"),
])
def test_slugify(topic, slug):
    assert slugify(topic) == slug


def test_slugify_is_capped():
    assert len(slugify("a" * 200)) <= 60


# ---- check_citations ----
def test_check_ok():
    assert check(GOOD, SOURCES) == []


def test_check_finds_problems():
    assert check(GOOD, []) == ["no sources in sources.json"]
    assert any("References" in p for p in check("Claim [1] [2].", SOURCES))
    assert any("[3] cited but missing" in p for p in check(GOOD.replace("Other [2]", "Other [2][3]"), SOURCES))
    assert any("never cited" in p for p in check(GOOD.replace("Other [2]", "Other"), SOURCES))


def test_check_rejects_bundled_and_wrong_reference_lines():
    bundled = GOOD.replace("(2025-01-02)", "; also https://example.org (2025-01-02)")
    assert any("exactly one URL" in p for p in check(bundled, SOURCES))
    wrong = GOOD.replace("https://arxiv.org/abs/2501.00001 (", "https://arxiv.org/abs/9999.99999 (")
    assert any("!=" in p for p in check(wrong, SOURCES))
    missing = GOOD.rsplit("[2] B", 1)[0]
    assert any("no reference line" in p for p in check(missing, SOURCES))


def test_check_understands_groups_and_ignores_code_and_links():
    grouped = GOOD.replace("Claim [1]. Other [2].", "Claim [1-2].")
    assert check(grouped, SOURCES) == []
    in_code = GOOD.replace("Other [2].", "Other `[2]` and [2](https://x.org).")
    assert any("[2] never cited" in p for p in check(in_code, SOURCES))


def test_check_duplicate_urls():
    dup = SOURCES + [dict(SOURCES[0], n=3)]
    assert any("duplicates" in p for p in check(GOOD, dup))


def test_finalizer_output_passes_validator():
    body = "# T\n\nClaim [2]. Other [1, 2]."
    report, sources, problems = finalize(body, SOURCES)
    assert problems == [] and check(report, sources) == []


# ---- save_outputs ----
class FakeBackend:
    def __init__(self, files):
        self.files = files

    def download_files(self, paths):
        from types import SimpleNamespace
        return [SimpleNamespace(path=p, content=self.files.get(p)) for p in paths]


def test_save_outputs_writes_nothing_on_failure(tmp_path):
    from agents import REPORT_PATH, SOURCES_PATH
    with pytest.raises(RuntimeError):
        save_outputs(FakeBackend({REPORT_PATH: b"", SOURCES_PATH: b"[]"}), "t", [], 1.0, "m", tmp_path)
    with pytest.raises(RuntimeError):
        save_outputs(FakeBackend({REPORT_PATH: b"# r", SOURCES_PATH: b"not json"}), "t", [], 1.0, "m", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_save_outputs_writes_three_files(tmp_path):
    from agents import REPORT_PATH, SOURCES_PATH
    backend = FakeBackend({REPORT_PATH: GOOD.encode(), SOURCES_PATH: json.dumps(SOURCES).encode()})
    path = save_outputs(backend, "My Topic", [], 12.34, "m", tmp_path)
    meta = json.loads((tmp_path / "my-topic.meta.json").read_text())
    assert path.name == "my-topic.md" and meta["n_sources"] == 2 and meta["elapsed_s"] == 12.3
    assert meta["source_families"] == ["arxiv", "hf-search"] and meta["topic"] == "My Topic"


# ---- model retry predicate ----
def test_transient_model_errors():
    import httpx
    from google.genai.errors import ClientError, ServerError

    from agents import _transient_model_error
    assert _transient_model_error(httpx.ReadError("connection reset"))
    assert _transient_model_error(ServerError(503, {"error": {"message": "overloaded"}}))
    assert _transient_model_error(ClientError(429, {"error": {"message": "quota"}}))
    assert not _transient_model_error(ClientError(400, {"error": {"message": "bad request"}}))
    assert not _transient_model_error(ValueError("bug"))
    wrapped = RuntimeError("wrapped")
    wrapped.__cause__ = httpx.ConnectTimeout("timeout")
    assert _transient_model_error(wrapped)


def test_arxiv_circuit_breaker_fails_fast_after_rate_limit(monkeypatch):
    calls = []

    def limited(params):
        calls.append(params)
        raise RetryableError("HTTP 429")
    monkeypatch.setattr(tools, "_arxiv_get", limited)
    monkeypatch.setattr(tools.time, "sleep", lambda s: None)
    monkeypatch.setattr(tools, "_arxiv_blocked_until", 0.0)
    assert tools.arxiv_search.invoke({"query": "world model"}).startswith("ERROR")
    tried = len(calls)
    second = tools.arxiv_search.invoke({"query": "world model"})
    assert second.startswith("ERROR") and "hf_daily_papers" in second and len(calls) == tried


# ---- review (deterministic requirement check before saving) ----
def test_review_flags_missing_families_and_themes():
    from research import review
    two_families = [{"source": "web"}, {"source": "hf-search"}]
    short = "# T\n## TL;DR\n## Background\n## Theme A\n## Trends and open problems\n## References\n"
    issues = review(short, two_families)
    assert len(issues) == 2 and "source families" in issues[0] and "thematic" in issues[1]
    full = short.replace("## Theme A\n", "## Theme A\n## Theme B\n## Theme C\n")
    assert review(full, two_families + [{"source": "hf-daily"}]) == []
