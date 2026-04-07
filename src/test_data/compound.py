"""Evolving compound test cases: compound episodes merging departure + arrival."""

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


def make_evolving_compound() -> list[TestCase]:
    """Generate 10 evolving_compound cases."""
    items = [
        ("Amy", "Google", "Facebook", "engineer", "product manager"),
        ("Bob", "New York", "San Francisco", "NYC resident", "SF resident"),
        ("Carol", "Team Alpha", "Team Beta", "Alpha lead", "Beta lead"),
        ("Dave", "Microsoft", "Amazon", "developer", "senior developer"),
        ("Eve", "London", "Tokyo", "London-based analyst", "Tokyo-based analyst"),
        ("Frank", "Startup X", "Startup Y", "CTO", "founder"),
        ("Grace", "MIT", "Stanford", "MIT researcher", "Stanford professor"),
        ("Hank", "Department A", "Department B", "analyst", "director"),
        ("Ivy", "Berlin", "Paris", "Berlin office manager", "Paris office manager"),
        ("Jack", "Company P", "Company Q", "intern", "full-time engineer"),
    ]
    cases: list[TestCase] = []
    for i, (name, old_place, new_place, old_role, new_role) in enumerate(items, 1):
        old_fact = f"{name} works at {old_place} as {old_role}."
        compound_ep = f"{name} left {old_place} and joined {new_place} as {new_role}."
        current_fact = f"{name} works at {new_place} as {new_role}."
        cases.append(
            TestCase(
                id=f"compound_{i:03d}",
                category="evolving_compound",
                tags=["compound", "evolving"],
                episodes=[
                    Episode(text=old_fact, reference_time=_dt(2024, 1, 1), order=0),
                    Episode(text=compound_ep, reference_time=_dt(2024, 6, 1), order=1),
                ],
                queries=[
                    Query(
                        query=f"Where does {name} work now?",
                        expected_facts=[current_fact],
                        expected_not=[old_fact],
                        query_time=_dt(2024, 7, 1),
                    )
                ],
                triplets=[
                    Triplet(
                        source=TripletNode(name=name, labels=["Person"]),
                        target=TripletNode(name=old_place, labels=["Organization"]),
                        edge=TripletEdge(
                            name="works_at",
                            fact=old_fact,
                            valid_at=_dt(2024, 1, 1),
                            invalid_at=_dt(2024, 6, 1),
                        ),
                    ),
                    Triplet(
                        source=TripletNode(name=name, labels=["Person"]),
                        target=TripletNode(name=new_place, labels=["Organization"]),
                        edge=TripletEdge(
                            name="works_at",
                            fact=current_fact,
                            valid_at=_dt(2024, 6, 1),
                        ),
                    ),
                ],
            )
        )
    return cases
