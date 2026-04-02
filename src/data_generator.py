"""Synthetic test case generator for the Graphiti temporal benchmark."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.models import Episode, Query, TestCase, Triplet, TripletEdge, TripletNode


def _dt(year: int, month: int, day: int) -> datetime:
    """Create a UTC datetime."""
    return datetime(year, month, day, tzinfo=timezone.utc)


def _make_static_facts() -> list[TestCase]:
    """Generate 10 static_fact test cases: simple entity-relationship pairs."""
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
        case_id = f"static_{i:03d}"
        cases.append(
            TestCase(
                id=case_id,
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


def _make_evolving_facts() -> list[TestCase]:
    """Generate 15 evolving_fact test cases: temporal updates where entity state changes."""
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
        case_id = f"evolving_{i:03d}"
        cases.append(
            TestCase(
                id=case_id,
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


def _make_multi_entity_evolution() -> list[TestCase]:
    """Generate 10 multi_entity_evolution cases: 2 entities evolving per case."""
    scenarios = [
        (
            "Acme Corp",
            "CEO",
            "Alice",
            "Bob",
            "Acme Corp CEO is Alice.",
            "Acme Corp CEO is now Bob.",
            "Acme Corp",
            "HQ",
            "Chicago",
            "Austin",
            "Acme Corp HQ is in Chicago.",
            "Acme Corp moved HQ to Austin.",
        ),
        (
            "Beta Inc",
            "CTO",
            "Charlie",
            "Diana",
            "Beta Inc CTO is Charlie.",
            "Beta Inc CTO changed to Diana.",
            "Beta Inc",
            "office",
            "Boston",
            "Denver",
            "Beta Inc office is in Boston.",
            "Beta Inc opened office in Denver.",
        ),
        (
            "Gamma LLC",
            "CEO",
            "Ed",
            "Fiona",
            "Gamma LLC CEO is Ed.",
            "Gamma LLC appointed Fiona as CEO.",
            "Gamma LLC",
            "HQ",
            "Miami",
            "Portland",
            "Gamma LLC HQ is in Miami.",
            "Gamma LLC relocated HQ to Portland.",
        ),
        (
            "Delta Co",
            "VP",
            "Greg",
            "Hannah",
            "Delta Co VP is Greg.",
            "Delta Co VP is now Hannah.",
            "Delta Co",
            "office",
            "Seattle",
            "Phoenix",
            "Delta Co office is in Seattle.",
            "Delta Co moved to Phoenix.",
        ),
        (
            "Epsilon Ltd",
            "CEO",
            "Ivan",
            "Julia",
            "Epsilon Ltd CEO is Ivan.",
            "Epsilon Ltd CEO changed to Julia.",
            "Epsilon Ltd",
            "HQ",
            "Dallas",
            "Atlanta",
            "Epsilon Ltd HQ is in Dallas.",
            "Epsilon Ltd moved HQ to Atlanta.",
        ),
        (
            "Zeta Corp",
            "CTO",
            "Kyle",
            "Laura",
            "Zeta Corp CTO is Kyle.",
            "Zeta Corp CTO is now Laura.",
            "Zeta Corp",
            "office",
            "Detroit",
            "Nashville",
            "Zeta Corp office in Detroit.",
            "Zeta Corp opened Nashville office.",
        ),
        (
            "Eta Inc",
            "CEO",
            "Mark",
            "Nancy",
            "Eta Inc CEO is Mark.",
            "Eta Inc appointed Nancy as CEO.",
            "Eta Inc",
            "HQ",
            "Philadelphia",
            "San Diego",
            "Eta Inc HQ in Philadelphia.",
            "Eta Inc relocated to San Diego.",
        ),
        (
            "Theta LLC",
            "VP",
            "Oscar",
            "Patty",
            "Theta LLC VP is Oscar.",
            "Theta LLC VP changed to Patty.",
            "Theta LLC",
            "office",
            "Minneapolis",
            "Tampa",
            "Theta LLC office in Minneapolis.",
            "Theta LLC moved to Tampa.",
        ),
        (
            "Iota Co",
            "CEO",
            "Quinn",
            "Rachel",
            "Iota Co CEO is Quinn.",
            "Iota Co CEO is now Rachel.",
            "Iota Co",
            "HQ",
            "Charlotte",
            "Orlando",
            "Iota Co HQ in Charlotte.",
            "Iota Co relocated HQ to Orlando.",
        ),
        (
            "Kappa Ltd",
            "CTO",
            "Sam",
            "Tina",
            "Kappa Ltd CTO is Sam.",
            "Kappa Ltd CTO changed to Tina.",
            "Kappa Ltd",
            "office",
            "Pittsburgh",
            "Las Vegas",
            "Kappa Ltd office in Pittsburgh.",
            "Kappa Ltd moved to Las Vegas.",
        ),
    ]
    cases: list[TestCase] = []
    for i, (
        org,
        role1,
        old1,
        new1,
        old_text1,
        new_text1,
        _org2,
        role2,
        old2,
        new2,
        old_text2,
        new_text2,
    ) in enumerate(scenarios, 1):
        case_id = f"multi_entity_{i:03d}"
        cases.append(
            TestCase(
                id=case_id,
                category="multi_entity_evolution",
                episodes=[
                    Episode(text=old_text1, reference_time=_dt(2023, 1, 1), order=0),
                    Episode(text=old_text2, reference_time=_dt(2023, 1, 1), order=1),
                    Episode(text=new_text1, reference_time=_dt(2024, 6, 1), order=2),
                    Episode(text=new_text2, reference_time=_dt(2024, 6, 1), order=3),
                ],
                queries=[
                    Query(
                        query=f"Who is the {role1} of {org} and where is the {role2}?",
                        expected_facts=[new_text1, new_text2],
                        expected_not=[old_text1, old_text2],
                        query_time=_dt(2024, 7, 1),
                    ),
                ],
                triplets=[
                    Triplet(
                        source=TripletNode(name=org, labels=["Organization"]),
                        target=TripletNode(name=old1, labels=["Person"]),
                        edge=TripletEdge(
                            name=role1,
                            fact=old_text1,
                            valid_at=_dt(2023, 1, 1),
                            invalid_at=_dt(2024, 6, 1),
                        ),
                    ),
                    Triplet(
                        source=TripletNode(name=org, labels=["Organization"]),
                        target=TripletNode(name=new1, labels=["Person"]),
                        edge=TripletEdge(
                            name=role1,
                            fact=new_text1,
                            valid_at=_dt(2024, 6, 1),
                        ),
                    ),
                    Triplet(
                        source=TripletNode(name=org, labels=["Organization"]),
                        target=TripletNode(name=old2, labels=["Location"]),
                        edge=TripletEdge(
                            name=role2,
                            fact=old_text2,
                            valid_at=_dt(2023, 1, 1),
                            invalid_at=_dt(2024, 6, 1),
                        ),
                    ),
                    Triplet(
                        source=TripletNode(name=org, labels=["Organization"]),
                        target=TripletNode(name=new2, labels=["Location"]),
                        edge=TripletEdge(
                            name=role2,
                            fact=new_text2,
                            valid_at=_dt(2024, 6, 1),
                        ),
                    ),
                ],
            )
        )
    return cases


def _make_contradiction_resolution() -> list[TestCase]:
    """Generate 10 contradiction_resolution cases: conflicting info, recency wins."""
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
        case_id = f"contradiction_{i:03d}"
        cases.append(
            TestCase(
                id=case_id,
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


def _make_entity_disambiguation() -> list[TestCase]:
    """Generate 10 entity_disambiguation cases: same name, different entities."""
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
        case_id = f"disambig_{i:03d}"
        cases.append(
            TestCase(
                id=case_id,
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


def generate_test_cases() -> list[TestCase]:
    """Generate all synthetic test cases for the benchmark."""
    cases: list[TestCase] = []
    cases.extend(_make_static_facts())
    cases.extend(_make_evolving_facts())
    cases.extend(_make_multi_entity_evolution())
    cases.extend(_make_contradiction_resolution())
    cases.extend(_make_entity_disambiguation())
    return cases


def save_test_cases(path: str) -> None:
    """Serialize test cases to JSON and write to file."""
    cases = generate_test_cases()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = [c.model_dump(mode="json") for c in cases]
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Saved {len(cases)} test cases to {out_path}")


if __name__ == "__main__":
    save_test_cases("data/test_cases.json")
