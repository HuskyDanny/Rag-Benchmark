"""Evolving fact test cases: temporal updates where entity state changes."""

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


def make_evolving_facts() -> list[TestCase]:
    """Generate 15 evolving_fact test cases."""
    evolutions = [
        (
            "Amy",
            "works_at",
            "Google",
            "Facebook",
            "Amy works at Google.",
            "Amy now works at Facebook.",
        ),
        (
            "Bob",
            "lives_in",
            "NYC",
            "SF",
            "Bob lives in New York City.",
            "Bob moved to San Francisco.",
        ),
        (
            "Carol",
            "manages",
            "Team Alpha",
            "Team Beta",
            "Carol manages Team Alpha.",
            "Carol now manages Team Beta.",
        ),
        (
            "Dave",
            "uses",
            "Python 3.9",
            "Python 3.12",
            "Dave uses Python 3.9.",
            "Dave upgraded to Python 3.12.",
        ),
        (
            "Eve",
            "studies_at",
            "MIT",
            "Stanford",
            "Eve studies at MIT.",
            "Eve transferred to Stanford.",
        ),
        (
            "Frank",
            "drives",
            "Toyota",
            "Tesla",
            "Frank drives a Toyota.",
            "Frank now drives a Tesla.",
        ),
        (
            "Grace",
            "works_at",
            "Microsoft",
            "Apple",
            "Grace works at Microsoft.",
            "Grace joined Apple.",
        ),
        (
            "Henry",
            "lives_in",
            "London",
            "Berlin",
            "Henry lives in London.",
            "Henry relocated to Berlin.",
        ),
        (
            "Ivy",
            "reports_to",
            "Manager A",
            "Manager B",
            "Ivy reports to Manager A.",
            "Ivy now reports to Manager B.",
        ),
        (
            "Jack",
            "owns",
            "Startup X",
            "Startup Y",
            "Jack owns Startup X.",
            "Jack sold X and founded Startup Y.",
        ),
        (
            "Kate",
            "works_at",
            "Amazon",
            "Netflix",
            "Kate works at Amazon.",
            "Kate moved to Netflix.",
        ),
        (
            "Leo",
            "lives_in",
            "Paris",
            "Tokyo",
            "Leo lives in Paris.",
            "Leo relocated to Tokyo.",
        ),
        (
            "Mia",
            "studies_at",
            "Harvard",
            "Oxford",
            "Mia studies at Harvard.",
            "Mia transferred to Oxford.",
        ),
        (
            "Nick",
            "manages",
            "Project A",
            "Project B",
            "Nick manages Project A.",
            "Nick now leads Project B.",
        ),
        (
            "Olivia",
            "uses",
            "Java 11",
            "Java 21",
            "Olivia uses Java 11.",
            "Olivia upgraded to Java 21.",
        ),
    ]
    cases: list[TestCase] = []
    for i, (entity, rel, old_val, new_val, old_text, new_text) in enumerate(
        evolutions, 1
    ):
        cases.append(
            TestCase(
                id=f"evolving_{i:03d}",
                category="evolving_fact",
                episodes=[
                    Episode(text=old_text, reference_time=_dt(2023, 1, 1), order=0),
                    Episode(text=new_text, reference_time=_dt(2024, 6, 1), order=1),
                ],
                queries=[
                    Query(
                        query=f"Where does {entity} currently {rel.replace('_', ' ')}?",
                        expected_facts=[new_text],
                        expected_not=[old_text],
                        query_time=_dt(2024, 7, 1),
                    ),
                ],
                triplets=[
                    Triplet(
                        source=TripletNode(name=entity, labels=["Person"]),
                        target=TripletNode(name=old_val, labels=["Entity"]),
                        edge=TripletEdge(
                            name=rel,
                            fact=old_text,
                            valid_at=_dt(2023, 1, 1),
                            invalid_at=_dt(2024, 6, 1),
                        ),
                    ),
                    Triplet(
                        source=TripletNode(name=entity, labels=["Person"]),
                        target=TripletNode(name=new_val, labels=["Entity"]),
                        edge=TripletEdge(
                            name=rel,
                            fact=new_text,
                            valid_at=_dt(2024, 6, 1),
                        ),
                    ),
                ],
            )
        )
    return cases
