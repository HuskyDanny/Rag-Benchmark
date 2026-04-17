"""Tests for TimeQA adapter: date parsing + document → TestCase conversion."""

from datetime import datetime

import pytest


# ── Date range parser ──


def test_parse_year_range_simple():
    from src.timeqa_adapter import parse_date_range

    result = parse_date_range(
        "Which American higher learning institution did Sabine Hossenfelder work from 2004 to 2005?"
    )
    assert result is not None
    start, end = result
    assert start.year == 2004
    assert end.year == 2005


def test_parse_year_range_in_prefix():
    from src.timeqa_adapter import parse_date_range

    result = parse_date_range(
        "At which California university did Hossenfelder work in 2005 to 2006?"
    )
    assert result is not None
    start, end = result
    assert start.year == 2005
    assert end.year == 2006


def test_parse_year_range_from_capitalized():
    from src.timeqa_adapter import parse_date_range

    result = parse_date_range(
        "From 2006 to 2009, which Canadian institution employed Hossenfelder?"
    )
    assert result is not None
    start, end = result
    assert start.year == 2006
    assert end.year == 2009


def test_parse_month_year_range():
    from src.timeqa_adapter import parse_date_range

    result = parse_date_range(
        "The park had what designation from Feb 1981 to Jul 2019?"
    )
    assert result is not None
    start, end = result
    assert start.year == 1981 and start.month == 2
    assert end.year == 2019 and end.month == 7


def test_parse_no_date_returns_none():
    from src.timeqa_adapter import parse_date_range

    result = parse_date_range("What is the most common skin cancer?")
    assert result is None


def test_midpoint_of_range():
    from src.timeqa_adapter import midpoint

    start = datetime(2004, 1, 1)
    end = datetime(2006, 1, 1)
    result = midpoint(start, end)
    # Midpoint is exactly halfway — within a day of 2005-01-01
    assert start < result < end
    delta_days = abs((result - datetime(2005, 1, 1)).total_seconds()) / 86400
    assert delta_days < 1


# ── Document adapter ──


def _sample_doc():
    return {
        "index": "/wiki/Sabine_Hossenfelder#P937",
        "type": "P937",
        "link": "/wiki/Sabine_Hossenfelder",
        "paras": [
            "Sabine Hossenfelder is a German physicist.",
            "Education info.",
            "More education.",
            "She did postdoc.",
            "Research career details.",
            "She moved to North America and worked at the University of Arizona, Tucson, University of California, Santa Barbara, and Perimeter Institute, Canada. She joined Nordita in 2009.",
        ],
        "questions": [
            [
                "Which American institution did Hossenfelder work from 2004 to 2005?",
                [
                    {
                        "para": 5,
                        "from": 0,
                        "end": 30,
                        "answer": "University of Arizona, Tucson",
                    }
                ],
            ],
            [
                "At which California university did she work in 2005 to 2006?",
                [
                    {
                        "para": 5,
                        "from": 0,
                        "end": 30,
                        "answer": "University of California, Santa Barbara",
                    }
                ],
            ],
        ],
    }


def test_adapter_creates_testcase_per_doc():
    from src.timeqa_adapter import doc_to_testcase

    tc = doc_to_testcase(_sample_doc())
    assert tc.id.startswith("timeqa_")
    assert tc.category == "timeqa_evolving"
    assert len(tc.queries) == 2


def test_adapter_episode_is_answer_para_plus_context():
    from src.timeqa_adapter import doc_to_testcase

    tc = doc_to_testcase(_sample_doc())
    assert len(tc.episodes) == 1
    # Episode should contain the answer paragraph (para 5)
    assert "Arizona" in tc.episodes[0].text
    # And the previous paragraph for context (para 4)
    assert "Research career details" in tc.episodes[0].text


def test_adapter_query_time_parsed_from_question():
    from src.timeqa_adapter import doc_to_testcase

    tc = doc_to_testcase(_sample_doc())
    q1 = tc.queries[0]
    # "from 2004 to 2005" → midpoint ~2004.5
    assert q1.query_time.year == 2004
    q2 = tc.queries[1]
    # "in 2005 to 2006" → midpoint ~2005.5
    assert q2.query_time.year == 2005


def test_adapter_expected_not_is_sibling_answers():
    """Each question's expected_not is the answers to OTHER questions in same doc."""
    from src.timeqa_adapter import doc_to_testcase

    tc = doc_to_testcase(_sample_doc())
    q1 = tc.queries[0]
    q2 = tc.queries[1]
    # Q1's expected answer is Arizona; expected_not should contain Santa Barbara (Q2's answer)
    assert q1.expected_facts == ["University of Arizona, Tucson"]
    assert "University of California, Santa Barbara" in q1.expected_not
    # And vice versa
    assert q2.expected_facts == ["University of California, Santa Barbara"]
    assert "University of Arizona, Tucson" in q2.expected_not


def test_adapter_skips_questions_with_empty_answer():
    from src.timeqa_adapter import doc_to_testcase

    doc = _sample_doc()
    doc["questions"].append(
        ["What was wrong in 2100?", [{"para": 5, "from": 0, "end": 0, "answer": ""}]]
    )
    tc = doc_to_testcase(doc)
    # Only 2 questions kept (empty answers filtered)
    assert len(tc.queries) == 2


def test_adapter_skips_questions_without_parseable_date():
    from src.timeqa_adapter import doc_to_testcase

    doc = _sample_doc()
    doc["questions"].append(
        [
            "What institution did she attend?",
            [{"para": 5, "from": 0, "end": 10, "answer": "MIT"}],
        ]
    )
    tc = doc_to_testcase(doc)
    # 2 original questions kept; the 3rd has no date range → skipped
    assert len(tc.queries) == 2
