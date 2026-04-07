"""Phase 3: Pre-split text into atomic facts, then insert via add_episode()."""

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

from src.models import TestCase
from src.sentence_splitter import split_into_atomic_facts

GROUP_ID = "benchmark_presplit"


async def insert_test_case(graphiti: Graphiti, test_case: TestCase) -> None:
    """Pre-split episodes into atomic facts, then insert each via add_episode."""
    sorted_episodes = sorted(test_case.episodes, key=lambda e: e.order)
    for episode in sorted_episodes:
        atomic_facts = await split_into_atomic_facts(episode.text)
        for i, fact in enumerate(atomic_facts):
            await graphiti.add_episode(
                name=f"{test_case.id}_ep{episode.order}_part{i}",
                episode_body=fact,
                source=EpisodeType.text,
                source_description="benchmark test data (pre-split)",
                reference_time=episode.reference_time,
                group_id=GROUP_ID,
            )


async def insert_all(graphiti: Graphiti, test_cases: list[TestCase]) -> None:
    """Insert all test cases with pre-splitting."""
    for i, tc in enumerate(test_cases):
        print(f"  Inserting test case {i + 1}/{len(test_cases)}: {tc.id}")
        await insert_test_case(graphiti, tc)
