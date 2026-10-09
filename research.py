"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator
RECURSION_LIMIT = 1000   # LangGraph step cap of the lead graph (~2 steps per model -> tool turn)


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(topic or "").lower())[:60].strip("-")
    return slug or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (f"Research topic: {topic}\n\n"
            "Produce the survey report following your instructions: plan with write_todos, delegate the sub-questions "
            "to researcher subagents in parallel, build sources.json, write the report body, run the finalizer and "
            "the validator until it prints OK, and spot-check a few claims with citation-checker.")


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.
    Lead messages only: subagent tokens are not included, so this undercounts the real cost."""
    calls = Counter()
    tokens = {"input": 0, "output": 0}
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            calls[call["name"]] += 1
        usage = getattr(message, "usage_metadata", None) or {}
        tokens["input"] += usage.get("input_tokens", 0) or 0
        tokens["output"] += usage.get("output_tokens", 0) or 0
    return {"model": model_name, "elapsed_s": round(elapsed, 1), "subagent_calls": calls.get("task", 0),
            "tool_calls": dict(calls), "tokens": tokens}


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.
    Raises RuntimeError and writes nothing when the report or sources.json is missing or broken."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report = (files.get(REPORT_PATH) or b"").decode("utf-8", errors="replace")
    if not report.strip():
        raise RuntimeError(f"the agent produced no report ({REPORT_PATH} is missing or empty)")
    try:
        sources = json.loads(files.get(SOURCES_PATH) or b"")
    except ValueError as exc:
        raise RuntimeError(f"{SOURCES_PATH} is missing or not valid JSON: {exc}") from exc
    if not isinstance(sources, list) or not sources:
        raise RuntimeError(f"{SOURCES_PATH} is not a non-empty JSON list")

    meta = {"topic": topic, **summarize(messages, elapsed, model_name), "n_sources": len(sources),
            "source_families": sorted({s.get("source") for s in sources if isinstance(s, dict) and s.get("source")})}
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(topic)
    (reports_dir / f"{slug}.sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n",
                                                      encoding="utf-8")
    (reports_dir / f"{slug}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
    report_path = reports_dir / f"{slug}.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def _model_name(model):
    return getattr(model, "model_name", None) or getattr(model, "model", None) or type(model).__name__


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    topic = (topic or "").strip()
    if not topic:
        print('usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    model = make_model()
    start = time.monotonic()
    with open_sandbox() as backend:   # the sandbox is always stopped and removed, even on errors
        backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
        upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
        agent = build_lead_agent(backend, model)
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                  config={"recursion_limit": RECURSION_LIMIT})
            report_path = save_outputs(backend, topic, result["messages"], time.monotonic() - start,
                                       _model_name(model))
        except Exception as exc:  # GraphRecursionError, model/API errors, missing report: a failed run
            print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    print(f"Report saved to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
