"""Phase 2: Insert raw text episodes via add_episode()."""

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

from src.models import TestCase

GROUP_ID = "benchmark_pipeline"


async def insert_test_case(graphiti: Graphiti, test_case: TestCase) -> None:
    """Insert all episodes from a test case via add_episode()."""
    sorted_episodes = sorted(test_case.episodes, key=lambda e: e.order)
    for episode in sorted_episodes:
        await graphiti.add_episode(
            name=f"{test_case.id}_ep{episode.order}",
            episode_body=episode.text,
            source=EpisodeType.text,
            source_description="benchmark test data",
            reference_time=episode.reference_time,
            group_id=GROUP_ID,
        )


async def insert_all(graphiti: Graphiti, test_cases: list[TestCase]) -> None:
    """Insert all test cases via the full LLM extraction pipeline."""
    for i, tc in enumerate(test_cases):
        print(f"  Inserting test case {i + 1}/{len(test_cases)}: {tc.id}")
        await insert_test_case(graphiti, tc)
