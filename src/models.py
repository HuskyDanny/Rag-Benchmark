"""Pydantic models for test cases and benchmark results."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class Episode(BaseModel):
    text: str
    reference_time: datetime
    order: int


class Query(BaseModel):
    query: str
    expected_facts: list[str]
    expected_not: list[str]
    query_time: datetime


class TripletNode(BaseModel):
    name: str
    labels: list[str]


class TripletEdge(BaseModel):
    name: str
    fact: str
    valid_at: datetime
    invalid_at: datetime | None = None


class Triplet(BaseModel):
    source: TripletNode
    target: TripletNode
    edge: TripletEdge


class TestCase(BaseModel):
    id: str
    category: str
    tags: list[str] = []
    episodes: list[Episode]
    queries: list[Query]
    triplets: list[Triplet]


class QueryResult(BaseModel):
    test_case_id: str
    query: str
    strategy: str
    returned_facts: list[str]
    expected_facts: list[str]
    expected_not: list[str]
    precision_at_5: float
    recall_at_5: float
    mrr: float
    temporal_accuracy: bool


class CategoryReport(BaseModel):
    phase: str
    category: str
    strategy: str
    avg_precision_at_5: float
    avg_recall_at_5: float
    avg_mrr: float
    temporal_accuracy_pct: float
    num_queries: int


class RunMetadata(BaseModel):
    run_id: str
    experiment_type: str
    phase: str
    params: dict[str, Any] = {}
    started_at: datetime
    completed_at: datetime | None = None


class IngestionResult(BaseModel):
    test_case_id: str
    category: str
    entity_recall: float
    entity_precision: float
    edge_recall: float
    edge_precision: float
    temporal_invalidation_accuracy: float
    dedup_score: float


class IngestionReport(BaseModel):
    phase: str
    category: str
    avg_entity_recall: float
    avg_entity_precision: float
    avg_edge_recall: float
    avg_edge_precision: float
    avg_temporal_invalidation_accuracy: float
    avg_dedup_score: float
    composite_score: float
    num_cases: int
