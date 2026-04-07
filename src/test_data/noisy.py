"""Noisy fact test cases: typos, abbreviations, informal language."""

from __future__ import annotations

from src.test_data.common import (
    Episode,
    Query,
    TestCase,
    Triplet,
    TripletEdge,
    TripletNode,
    _dt,
)

_NOISY_ITEMS = [
    (
        "Amy",
        "Google",
        "Facebook",
        "amy workz at gogle as enginere.",
        "amy now workz at facebok.",
        "Amy works at Google as an engineer.",
        "Amy now works at Facebook.",
        "typo",
    ),
    (
        "Bob",
        "New York City",
        "San Francisco",
        "bob livs in new yrok city.",
        "bob movd to san fran.",
        "Bob lives in New York City.",
        "Bob moved to San Francisco.",
        "typo",
    ),
    (
        "Carol",
        "Team Alpha",
        "Team Beta",
        "carol mngz team alpha.",
        "carol now mngz team beta.",
        "Carol manages Team Alpha.",
        "Carol now manages Team Beta.",
        "typo",
    ),
    (
        "Dave",
        "Microsoft",
        "Amazon",
        "dave → msft as sr. dev",
        "dave → amzn as principal eng.",
        "Dave works at Microsoft as a senior developer.",
        "Dave works at Amazon as a principal engineer.",
        "abbreviation",
    ),
    (
        "Eve",
        "London office",
        "Tokyo office",
        "eve @ london office, analyst role",
        "eve transferred → tokyo office",
        "Eve works at the London office as an analyst.",
        "Eve transferred to the Tokyo office.",
        "abbreviation",
    ),
    (
        "Frank",
        "Startup X",
        "Startup Y",
        "frank is cto @ startup x",
        "frank quit & founded startup y",
        "Frank is the CTO at Startup X.",
        "Frank quit and founded Startup Y.",
        "abbreviation",
    ),
    (
        "Grace",
        "MIT",
        "Stanford",
        "grace doing research at mit lol",
        "grace moved to stanford ngl pretty cool",
        "Grace does research at MIT.",
        "Grace moved to Stanford.",
        "informal",
    ),
    (
        "Hank",
        "Department A",
        "Department B",
        "hank's in dept a as analyst rn",
        "hank got promoted to director in dept b",
        "Hank works in Department A as an analyst.",
        "Hank was promoted to director in Department B.",
        "informal",
    ),
    (
        "Ivy",
        "Berlin",
        "Paris",
        "ivy based in berlin office mgmt",
        "ivy relocated 2 paris as office mgr",
        "Ivy is based in Berlin in office management.",
        "Ivy relocated to Paris as office manager.",
        "mixed",
    ),
    (
        "Jack",
        "Company P",
        "Company Q",
        "jack interning @ co. P prev semester",
        "jack got ft offer frm co. Q & accepted",
        "Jack is interning at Company P.",
        "Jack got a full-time offer from Company Q and accepted.",
        "mixed",
    ),
]


def make_noisy_facts() -> list[TestCase]:
    """Generate 10 noisy_fact cases."""
    cases: list[TestCase] = []
    for i, (
        name,
        old_place,
        new_place,
        noisy_old,
        noisy_new,
        clean_old,
        clean_new,
        noise_type,
    ) in enumerate(_NOISY_ITEMS, 1):
        cases.append(
            TestCase(
                id=f"noisy_{i:03d}",
                category="noisy_fact",
                tags=["noisy", noise_type, "evolving"],
                episodes=[
                    Episode(text=noisy_old, reference_time=_dt(2024, 1, 1), order=0),
                    Episode(text=noisy_new, reference_time=_dt(2024, 6, 1), order=1),
                ],
                queries=[
                    Query(
                        query=f"Where does {name} work now?",
                        expected_facts=[clean_new],
                        expected_not=[clean_old],
                        query_time=_dt(2024, 7, 1),
                    )
                ],
                triplets=[
                    Triplet(
                        source=TripletNode(name=name, labels=["Person"]),
                        target=TripletNode(name=old_place, labels=["Organization"]),
                        edge=TripletEdge(
                            name="works_at",
                            fact=clean_old,
                            valid_at=_dt(2024, 1, 1),
                            invalid_at=_dt(2024, 6, 1),
                        ),
                    ),
                    Triplet(
                        source=TripletNode(name=name, labels=["Person"]),
                        target=TripletNode(name=new_place, labels=["Organization"]),
                        edge=TripletEdge(
                            name="works_at",
                            fact=clean_new,
                            valid_at=_dt(2024, 6, 1),
                        ),
                    ),
                ],
            )
        )
    return cases
