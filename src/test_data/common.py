"""Shared helpers and re-exports for test data generators."""

from __future__ import annotations

from datetime import datetime, timezone

from src.models import (
    EdgeExpectation,
    Episode,
    IngestionExpectation,
    NodeExpectation,
    Query,
    StepExpectation,
    TestCase,
    Triplet,
    TripletEdge,
    TripletNode,
)


def _dt(year: int, month: int, day: int) -> datetime:
    """Create a UTC datetime."""
    return datetime(year, month, day, tzinfo=timezone.utc)


__all__ = [
    "_dt",
    "EdgeExpectation",
    "Episode",
    "IngestionExpectation",
    "NodeExpectation",
    "Query",
    "StepExpectation",
    "TestCase",
    "Triplet",
    "TripletEdge",
    "TripletNode",
]
