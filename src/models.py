"""Pydantic models for test cases and benchmark results."""

from __future__ import annotations

from datetime import datetime
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
