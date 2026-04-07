"""Static fact test cases: simple entity-relationship pairs that never change."""

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


def make_static_facts() -> list[TestCase]:
    """Generate 10 static_fact test cases."""
    facts = [
        ("Paris", "capital_of", "France", "Paris is the capital of France."),
        (
            "Python",
            "created_by",
            "Guido van Rossum",
            "Python was created by Guido van Rossum.",
        ),
        ("Earth", "orbits", "Sun", "Earth orbits the Sun."),
        ("Water", "chemical_formula", "H2O", "Water has the chemical formula H2O."),
        (
            "Amazon",
            "headquartered_in",
            "Seattle",
            "Amazon is headquartered in Seattle.",
        ),
        (
            "Linux",
            "created_by",
            "Linus Torvalds",
            "Linux was created by Linus Torvalds.",
        ),
        ("Tokyo", "capital_of", "Japan", "Tokyo is the capital of Japan."),
        ("Tesla", "founded_by", "Elon Musk", "Tesla was founded by Elon Musk."),
        (
            "JavaScript",
            "created_by",
            "Brendan Eich",
            "JavaScript was created by Brendan Eich.",
        ),
        ("Berlin", "capital_of", "Germany", "Berlin is the capital of Germany."),
    ]
    cases: list[TestCase] = []
    for i, (src, rel, tgt, text) in enumerate(facts, 1):
        cases.append(
            TestCase(
                id=f"static_{i:03d}",
                category="static_fact",
                episodes=[
                    Episode(text=text, reference_time=_dt(2024, 1, 1), order=0),
                ],
                queries=[
                    Query(
                        query=f"What is the relationship between {src} and {tgt}?",
                        expected_facts=[text],
                        expected_not=[],
                        query_time=_dt(2024, 6, 1),
                    ),
                ],
                triplets=[
                    Triplet(
                        source=TripletNode(name=src, labels=["Entity"]),
                        target=TripletNode(name=tgt, labels=["Entity"]),
                        edge=TripletEdge(
                            name=rel,
                            fact=text,
                            valid_at=_dt(2024, 1, 1),
                        ),
                    ),
                ],
            )
        )
    return cases
