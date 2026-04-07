"""Entity disambiguation test cases: same name, different entities."""

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


def make_disambiguation() -> list[TestCase]:
    """Generate 10 entity_disambiguation test cases."""
    disambiguations = [
        (
            "John Smith",
            "engineer",
            "doctor",
            "John Smith is a software engineer at Google.",
            "John Smith is a cardiologist at Mayo Clinic.",
        ),
        (
            "Sarah Johnson",
            "teacher",
            "lawyer",
            "Sarah Johnson teaches math at Lincoln High.",
            "Sarah Johnson is a patent lawyer in NYC.",
        ),
        (
            "Michael Brown",
            "chef",
            "pilot",
            "Michael Brown is a chef in Paris.",
            "Michael Brown is a commercial pilot based in Dallas.",
        ),
        (
            "Emily Davis",
            "artist",
            "scientist",
            "Emily Davis is a painter in Brooklyn.",
            "Emily Davis is a biochemist at Caltech.",
        ),
        (
            "James Wilson",
            "coach",
            "journalist",
            "James Wilson coaches basketball at Duke.",
            "James Wilson is a war correspondent for Reuters.",
        ),
        (
            "Lisa Chen",
            "architect",
            "musician",
            "Lisa Chen is an architect in Shanghai.",
            "Lisa Chen is a concert pianist in Vienna.",
        ),
        (
            "Robert Taylor",
            "banker",
            "farmer",
            "Robert Taylor is an investment banker in London.",
            "Robert Taylor runs a dairy farm in Vermont.",
        ),
        (
            "Maria Garcia",
            "nurse",
            "developer",
            "Maria Garcia is a nurse at Johns Hopkins.",
            "Maria Garcia is a frontend developer at Stripe.",
        ),
        (
            "David Lee",
            "professor",
            "athlete",
            "David Lee is a physics professor at MIT.",
            "David Lee is an Olympic swimmer.",
        ),
        (
            "Jennifer Martinez",
            "dentist",
            "author",
            "Jennifer Martinez is a dentist in Chicago.",
            "Jennifer Martinez is a bestselling novelist.",
        ),
    ]
    cases: list[TestCase] = []
    for i, (name, role1, role2, text1, text2) in enumerate(disambiguations, 1):
        cases.append(
            TestCase(
                id=f"disambig_{i:03d}",
                category="entity_disambiguation",
                episodes=[
                    Episode(text=text1, reference_time=_dt(2024, 1, 1), order=0),
                    Episode(text=text2, reference_time=_dt(2024, 1, 1), order=1),
                ],
                queries=[
                    Query(
                        query=f"What does {name} the {role1} do?",
                        expected_facts=[text1],
                        expected_not=[],
                        query_time=_dt(2024, 6, 1),
                    ),
                    Query(
                        query=f"What does {name} the {role2} do?",
                        expected_facts=[text2],
                        expected_not=[],
                        query_time=_dt(2024, 6, 1),
                    ),
                ],
                triplets=[
                    Triplet(
                        source=TripletNode(name=f"{name} ({role1})", labels=["Person"]),
                        target=TripletNode(name=role1, labels=["Profession"]),
                        edge=TripletEdge(
                            name="profession",
                            fact=text1,
                            valid_at=_dt(2024, 1, 1),
                        ),
                    ),
                    Triplet(
                        source=TripletNode(name=f"{name} ({role2})", labels=["Person"]),
                        target=TripletNode(name=role2, labels=["Profession"]),
                        edge=TripletEdge(
                            name="profession",
                            fact=text2,
                            valid_at=_dt(2024, 1, 1),
                        ),
                    ),
                ],
            )
        )
    return cases
