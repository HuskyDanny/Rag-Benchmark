# Graph Agent Tools Design Spec

**Date:** 2026-04-07
**Status:** Draft
**Scope:** Tool definitions + agent system prompt for a single agent with graph-aware retrieval tools

## Context

A generic agentic loop needs to retrieve context from a knowledge graph (Neo4j/Graphiti). Rather than splitting into separate "Graph Agent" and "RAG Agent," the graph capabilities are exposed as **tools within a single agent's toolbox**. The agent decides which tools to use based on the question.

## Design Decisions

1. **Single agent, not multi-agent** — graph tools are skills in the agent's toolbox, not a separate agent. Avoids orchestration overhead and routing heuristics.
2. **Tiered tools (B)** — a high-level `graph_search` handles 80% of queries in one call. Power tools handle the remaining 20% (traversal, paths, raw Cypher).
3. **Graph signal in every response** — even Tier 1 returns entity metadata (degree, relationship types, 1-hop neighbors). The agent always sees structural context, not just flat text.

## Tool Definitions

### Tier 1: Default Search

#### `graph_search`

Semantic search that returns facts WITH their graph context. This is what differentiates it from flat vector RAG — every result carries structural metadata.

```
Parameters:
  query: str              — natural language question
  top_k: int = 5          — max results
  time_context: str?      — ISO timestamp; if set, filters to facts valid at that time

Returns:
  results: list[Fact]
    - fact: str                    — "Alice works at Google"
    - source: Entity
        - name: str                — "Alice"
        - type: str                — "Person"
        - degree: int              — total connections (12)
        - top_relationships:       — ["WORKS_AT(2)", "FRIENDS_WITH(5)", "LIVES_IN(1)"]
          list[str]
    - target: Entity
        - name: str                — "Google"
        - type: str                — "Company"
        - degree: int              — 45
        - top_relationships:       — ["EMPLOYS(30)", "LOCATED_IN(3)"]
          list[str]
    - relationship: str            — "WORKS_AT"
    - valid_from: str?             — when this fact became true
    - invalid_at: str?             — when superseded (null = still current)
    - score: float                 — relevance (0-1, BM25 + cosine + graph signal)
    - related_facts: list[str]     — 1-hop neighbors of the edge
                                     ["Alice lives in SF", "Google HQ in Mountain View"]

Behavior:
  - Hybrid search: BM25 keyword + cosine embedding + RRF fusion
  - time_context set: returns facts where valid_from <= time_context
    AND (invalid_at IS NULL OR invalid_at > time_context)
  - time_context omitted: returns all facts including historical
```

**What this gives the agent over flat RAG:**
- **Entity metadata** — degree + relationship types tell the agent "Alice is well-connected via FRIENDS_WITH" vs "Google is a hub with 30 employees"
- **`related_facts`** — 1-hop neighborhood surfaced automatically. Context the agent didn't ask for but might need.
- **Temporal bounds** — the agent sees the lifecycle of every fact, not just "current or not"

### Tier 2: Power Tools

#### `explore_entity`

Full neighborhood scan — "tell me everything about X."

```
Parameters:
  entity: str                      — entity name or partial match
  hops: int = 1                    — traversal depth (1-3)
  relationship_types: list[str]?   — filter to specific edge types
  time_context: str?               — snapshot at a point in time

Returns:
  entity: Entity
    - name, type, degree, created_at
  edges: list[Edge]
    - fact: str
    - relationship: str
    - direction: "outgoing" | "incoming"
    - neighbor: Entity
    - valid_from, invalid_at
  timeline: list[Edge]             — same edges sorted chronologically
```

**Use when:** "What do we know about Dave?" "How has Alice's career evolved?" The `relationship_types` filter is the key graph primitive — the agent reads types from Tier 1 results, then drills into a specific dimension.

#### `find_paths`

Discover how two entities are connected — what vector search literally cannot do.

```
Parameters:
  from_entity: str
  to_entity: str
  max_hops: int = 3
  relationship_types: list[str]?   — constrain traversal edges

Returns:
  paths: list[Path]
    - length: int
    - edges: list[Edge]            — ordered facts forming the path
    - summary: str                 — "Alice -> WORKS_AT -> Meta -> RUNS -> Project X"
  shortest_path: Path
```

**Use when:** "How does Alice know about Project X?" "What's the connection between Company A and Person B?"

#### `get_relationships`

List all relationship TYPES for an entity — the schema of its world.

```
Parameters:
  entity: str

Returns:
  relationships: list[RelType]
    - type: str                    — "WORKS_AT"
    - direction: "outgoing" | "incoming"
    - count: int
    - examples: list[str]          — up to 3 sample facts
```

**Use when:** Before `explore_entity` with a filter — check what types exist, then decide which to drill into. "Look at the map before walking."

#### `raw_cypher`

Escape hatch for queries the structured tools cannot express.

```
Parameters:
  query: str                       — Cypher query string
  params: dict?                    — parameterized values

Returns:
  rows: list[dict]
  columns: list[str]

Guards:
  - READ-ONLY (no CREATE, MERGE, DELETE, SET)
  - Timeout: 5 seconds
  - Result limit: 100 rows
```

**Use when:** Complex aggregations, pattern matching, structural queries. Last resort.

## Tool Selection Matrix

| Question Type | Tool |
|---|---|
| Factual lookup ("What is X?") | `graph_search` |
| Entity profile ("Tell me about X") | `explore_entity` |
| Connection ("How are X and Y related?") | `find_paths` |
| Relationship schema ("What kinds of...?") | `get_relationships` |
| Temporal comparison ("What changed?") | `explore_entity` (timeline) |
| Complex structural / aggregation | `raw_cypher` |

## Agent System Prompt

```
# Knowledge Graph Tools

You have access to a knowledge graph that stores entities, relationships,
and temporal facts. Use these tools to retrieve context for answering questions.

## Tool Overview

**Tier 1 — Default (use first):**
- `graph_search` — semantic search with graph context. Handles most questions
  in a single call. Returns facts + entity metadata + 1-hop neighbors.

**Tier 2 — Power tools (use when Tier 1 isn't enough):**
- `explore_entity` — full neighborhood scan with optional relationship filter
- `find_paths` — discover how two entities are connected
- `get_relationships` — list relationship types for an entity (use before explore)
- `raw_cypher` — arbitrary read-only Cypher (last resort)

## Decision Rules

1. **Start with `graph_search`.** For 80% of questions, one call is enough.

2. **Escalate to Tier 2 when:**
   - The question asks HOW two things are connected -> `find_paths`
   - The question asks for a full profile / "everything about X" -> `explore_entity`
   - You got results from `graph_search` but need to follow a specific
     relationship type deeper -> `get_relationships` then `explore_entity`
   - The question involves comparing an entity across time periods
     -> `explore_entity` with `time_context` at each point, or without
     `time_context` and read the `timeline`

3. **Never call `raw_cypher` before trying the structured tools.**
   It exists for queries the other tools structurally cannot express
   (aggregations, pattern matching, complex filters).

## Chaining Patterns

### Pattern: Drill-Down
When `graph_search` returns a high-degree entity and you need specifics:
1. `graph_search(query)` -> notice entity has degree=45, many relationship types
2. `get_relationships(entity)` -> see which types are relevant
3. `explore_entity(entity, relationship_types=[relevant_type])` -> focused context

### Pattern: Connection Discovery
When the question is about relationships between entities:
1. `graph_search(query)` -> identify the two entities
2. `find_paths(entity_a, entity_b)` -> discover structural connections
3. Optionally: `explore_entity` on intermediate nodes for more context

### Pattern: Temporal Comparison
When the question asks "what changed" or "before vs after":
1. `explore_entity(entity)` -> read the `timeline` field
   OR
2. `graph_search(query, time_context=T1)` -> state at time 1
3. `graph_search(query, time_context=T2)` -> state at time 2
4. Compare the two result sets

### Pattern: Multi-Entity Reasoning
When the question involves several entities and their interactions:
1. `graph_search(query)` -> identify key entities from results
2. For each key entity: `explore_entity` with relevant relationship filter
3. If entities might be connected: `find_paths` between them

## Reading Results

- **`degree`** tells you how connected an entity is. High degree = hub,
  likely important. Low degree = peripheral.
- **`top_relationships`** tells you the SHAPE of an entity's connections.
  A person with mostly WORKS_AT edges has career data. One with mostly
  FRIENDS_WITH has social data. Use this to judge what the graph knows.
- **`related_facts`** from `graph_search` gives you free context -- facts
  you didn't ask for but are 1-hop away. Read these before making
  another tool call; the answer might already be there.
- **`timeline`** from `explore_entity` gives you the narrative arc.
  Read it chronologically to understand how an entity evolved.
- **`invalid_at`** on any fact means it was superseded. If non-null,
  a newer fact replaced it. The replacement is usually in the same results
  or 1-hop away.

## What You Cannot Do

- Write to the graph (all tools are read-only)
- Search across disconnected subgraphs in one call (use multiple searches)
- Get guaranteed completeness -- the graph contains what was ingested,
  not all world knowledge

## When the Graph Doesn't Have the Answer

If `graph_search` returns no results or low-score results (< 0.3),
say so. Do not hallucinate facts. You can say: "The knowledge graph
doesn't have information about X" and suggest what might need to be
ingested.
```

## Why This Design Over Alternatives

### vs. Multi-Agent (Graph Agent + RAG Agent)
A separate RAG agent would just be `graph_search` without the graph metadata — a strictly weaker version. Since Graphiti already combines BM25 + cosine + graph structure, there's no retrieval capability a "RAG agent" has that the graph tools don't cover.

### vs. Flat Tool Registry (all tools equal)
Maximum flexibility but the agent over-explores on simple queries. The tiered model ensures simple questions stay fast (1 call) while complex questions get the full graph toolkit.

### vs. Macro Tools (predefined strategies)
Too rigid. The whole point of giving an LLM graph tools is letting it reason about traversal. Macros prevent the agent from inventing novel chains for unforeseen question types.

## Open Questions

1. **Embedding model alignment** — `graph_search` quality depends on the embedding model matching the query style. May need query rewriting for informal questions.
2. **Hop depth limits** — `explore_entity` with hops=3 on a hub entity could return thousands of edges. May need result capping or relevance filtering within the expansion.
3. **Community detection** — a potential Tier 2 tool (`get_community`) that returns the cluster an entity belongs to. Deferred until there's a concrete use case.
