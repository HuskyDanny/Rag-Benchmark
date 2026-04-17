"""TimeQA dataset adapter: converts TimeQA docs into TestCase instances.

TimeQA docs are Wikipedia-style passages with time-bounded QA pairs. Each doc
yields one TestCase with:
- episodes: the answer paragraph + one prev paragraph for context
- queries: one Query per TimeQA question, query_time = midpoint of parsed range
- expected_not: sibling answers from other questions in the same doc
  (these are the stale facts that should NOT appear at the current query_time)
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from src.models import Episode, Query, TestCase

_MONTH = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_RANGE_WITH_MONTHS = re.compile(
    r"(?:from|in)\s+"
    r"(?P<sm>[A-Z][a-z]{2,8})?\s*(?P<sy>\d{4})\s+"
    r"to\s+"
    r"(?P<em>[A-Z][a-z]{2,8})?\s*(?P<ey>\d{4})",
    re.IGNORECASE,
)


def parse_date_range(question: str) -> tuple[datetime, datetime] | None:
    """Extract (start, end) datetimes from a TimeQA question.

    Supports: 'from 2004 to 2005', 'in 2005 to 2006', 'From Feb 1981 to Jul 2019'.
    Returns None if no parseable range found.
    """
    m = _RANGE_WITH_MONTHS.search(question)
    if not m:
        return None
    try:
        sy = int(m.group("sy"))
        ey = int(m.group("ey"))
        sm_str = (m.group("sm") or "").lower()[:3]
        em_str = (m.group("em") or "").lower()[:3]
        sm = _MONTH.get(sm_str, 1)
        em = _MONTH.get(em_str, 12)
        start = datetime(sy, sm, 1, tzinfo=timezone.utc)
        end = datetime(ey, em, 28, tzinfo=timezone.utc)
        if end < start:
            return None
        return start, end
    except (ValueError, KeyError):
        return None


def midpoint(start: datetime, end: datetime) -> datetime:
    """Return the midpoint datetime between start and end."""
    return start + (end - start) / 2


def _build_episode_text(paras: list[str], answer_para_idx: int) -> str:
    """Answer paragraph + one previous paragraph (if available) for context."""
    if answer_para_idx == 0:
        return paras[0]
    prev = paras[answer_para_idx - 1]
    ans = paras[answer_para_idx]
    return f"{prev} {ans}"


def _slugify(link: str) -> str:
    """Convert '/wiki/Sabine_Hossenfelder' → 'sabine_hossenfelder'."""
    return link.replace("/wiki/", "").lower().replace(" ", "_")


def doc_to_testcase(doc: dict[str, Any]) -> TestCase:
    """Convert one TimeQA doc into a TestCase.

    Skips questions with empty answers or unparseable date ranges.
    Sets expected_not to sibling answers (facts that should NOT appear at
    the current query_time because they apply to a different period).
    """
    paras = doc["paras"]
    raw_questions = doc["questions"]
    doc_id = _slugify(doc["link"])

    # First pass: parse valid questions
    valid = []
    for q in raw_questions:
        question_text = q[0]
        answers = q[1]
        if not answers:
            continue
        answer = answers[0]["answer"].strip()
        if not answer:
            continue
        date_range = parse_date_range(question_text)
        if date_range is None:
            continue
        para_idx = answers[0]["para"]
        valid.append(
            {
                "question": question_text,
                "answer": answer,
                "query_time": midpoint(*date_range),
                "para": para_idx,
            }
        )

    # Second pass: build queries with expected_not = sibling answers
    queries = []
    all_answers = [v["answer"] for v in valid]
    for v in valid:
        siblings = [a for a in all_answers if a != v["answer"]]
        queries.append(
            Query(
                query=v["question"],
                expected_facts=[v["answer"]],
                expected_not=siblings,
                query_time=v["query_time"],
            )
        )

    # Single episode = answer paragraph(s) + context
    # If multiple questions reference different paras, concat all unique ones
    para_indices = sorted({v["para"] for v in valid})
    if para_indices:
        episode_text = _build_episode_text(paras, para_indices[0])
        # Append additional answer paras if questions span multiple
        for idx in para_indices[1:]:
            if idx != para_indices[0]:
                episode_text += " " + paras[idx]
    else:
        episode_text = paras[0] if paras else ""

    # Use earliest query_time as episode reference_time (ingestion "now")
    ref_time = min((v["query_time"] for v in valid), default=datetime.now(timezone.utc))

    episodes = [Episode(text=episode_text, reference_time=ref_time, order=0)]

    return TestCase(
        id=f"timeqa_{doc_id}",
        category="timeqa_evolving",
        tags=["timeqa", "evolving", "real-world"],
        episodes=episodes,
        queries=queries,
        triplets=[],
    )


def load_timeqa_testcases(path: str, max_docs: int | None = None) -> list[TestCase]:
    """Load TimeQA human_annotated_test.json and convert to TestCase list."""
    import json

    with open(path) as f:
        docs = json.load(f)
    if max_docs:
        docs = docs[:max_docs]
    test_cases = []
    for doc in docs:
        try:
            tc = doc_to_testcase(doc)
            if tc.queries:  # skip docs with no parseable questions
                test_cases.append(tc)
        except (KeyError, IndexError) as e:
            print(f"  Skipping doc {doc.get('link', '?')}: {e}")
    return test_cases
