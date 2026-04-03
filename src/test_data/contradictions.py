"""Contradiction resolution test cases: conflicting info where recency wins."""

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


def make_contradictions() -> list[TestCase]:
    """Generate 10 contradiction_resolution test cases."""
    contradictions = [
        (
            "Omega Corp",
            "revenue",
            "$10M",
            "$12M",
            "Omega Corp revenue is $10M.",
            "Omega Corp revenue updated to $12M.",
        ),
        (
            "Alpha Inc",
            "employees",
            "500",
            "750",
            "Alpha Inc has 500 employees.",
            "Alpha Inc grew to 750 employees.",
        ),
        (
            "Beta Labs",
            "valuation",
            "$50M",
            "$80M",
            "Beta Labs valued at $50M.",
            "Beta Labs valuation rose to $80M.",
        ),
        (
            "Gamma Tech",
            "office_count",
            "3",
            "5",
            "Gamma Tech has 3 offices.",
            "Gamma Tech expanded to 5 offices.",
        ),
        (
            "Delta AI",
            "funding",
            "$5M",
            "$20M",
            "Delta AI raised $5M.",
            "Delta AI closed $20M Series B.",
        ),
        (
            "Epsilon Bio",
            "patents",
            "10",
            "25",
            "Epsilon Bio holds 10 patents.",
            "Epsilon Bio now holds 25 patents.",
        ),
        (
            "Zeta Fin",
            "customers",
            "1000",
            "2500",
            "Zeta Fin has 1000 customers.",
            "Zeta Fin reached 2500 customers.",
        ),
        (
            "Eta Media",
            "subscribers",
            "50K",
            "120K",
            "Eta Media has 50K subscribers.",
            "Eta Media grew to 120K subscribers.",
        ),
        (
            "Theta Health",
            "clinics",
            "8",
            "15",
            "Theta Health operates 8 clinics.",
            "Theta Health expanded to 15 clinics.",
        ),
        (
            "Iota Energy",
            "capacity",
            "100MW",
            "250MW",
            "Iota Energy capacity is 100MW.",
            "Iota Energy expanded to 250MW.",
        ),
    ]
    cases: list[TestCase] = []
    for i, (entity, metric, old_val, new_val, old_text, new_text) in enumerate(
        contradictions, 1
    ):
        cases.append(
            TestCase(
                id=f"contradiction_{i:03d}",
                category="contradiction_resolution",
                episodes=[
                    Episode(text=old_text, reference_time=_dt(2023, 6, 1), order=0),
                    Episode(text=new_text, reference_time=_dt(2024, 6, 1), order=1),
                ],
                queries=[
                    Query(
                        query=f"What is the current {metric} of {entity}?",
                        expected_facts=[new_text],
                        expected_not=[old_text],
                        query_time=_dt(2024, 7, 1),
                    ),
                ],
                triplets=[
                    Triplet(
                        source=TripletNode(name=entity, labels=["Organization"]),
                        target=TripletNode(name=old_val, labels=["Value"]),
                        edge=TripletEdge(
                            name=metric,
                            fact=old_text,
                            valid_at=_dt(2023, 6, 1),
                            invalid_at=_dt(2024, 6, 1),
                        ),
                    ),
                    Triplet(
                        source=TripletNode(name=entity, labels=["Organization"]),
                        target=TripletNode(name=new_val, labels=["Value"]),
                        edge=TripletEdge(
                            name=metric,
                            fact=new_text,
                            valid_at=_dt(2024, 6, 1),
                        ),
                    ),
                ],
            )
        )
    return cases
