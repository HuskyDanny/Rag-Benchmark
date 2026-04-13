# Graphiti Temporal Retrieval Benchmark

Benchmarks whether Graphiti's temporal knowledge graph improves retrieval over flat BM25/cosine search, especially for evolving facts.

## Quick Start

```bash
# 1. Install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pip install httpx[socks]  # if using SOCKS proxy

# 2. Configure
cp .env.example .env
# Edit .env with your credentials:
#   NEO4J_PASSWORD=your_password
#   OPENAI_API_KEY=your_key
#   OPENAI_BASE_URL=https://api.siliconflow.cn/v1  (or OpenAI)
#   LLM_MODEL=Qwen/Qwen3.5-397B-A17B
#   EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B

# 3. Start Neo4j
docker run -d --name neo4j-benchmark -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/your_password neo4j:5

# 4. Run a benchmark phase
python -m src.benchmark_runner controlled --clean
```

## Benchmark Phases

| Phase | What It Tests | Speed |
|-------|--------------|-------|
| `controlled` | Search quality with perfect data (direct Neo4j save) | ~15 min |
| `pipeline` | Full Graphiti pipeline (LLM extraction via `add_episode`) | ~30 min |
| `pipeline_presplit` | Pre-split text into atomic facts, then pipeline | ~40 min |

## Running the Benchmark

### Single Phase

```bash
# Run one phase (insert + evaluate + report)
python -m src.benchmark_runner controlled --clean

# Run with specific Neo4j port
python -m src.benchmark_runner pipeline --port 7688 --clean
```

### Individual Stages

Each phase has 3 stages that can run independently. Useful for debugging or resuming after crashes:

```bash
# Insert only (populate Neo4j)
python -m src.benchmark_runner controlled --stage insert --clean

# Evaluate only (run queries + LLM judge — requires data in Neo4j)
python -m src.benchmark_runner controlled --stage evaluate

# Report only (aggregate + print table — requires evaluate results)
python -m src.benchmark_runner controlled --stage report
```

### Parallel Execution (3x faster)

Runs all 3 phases simultaneously using separate Neo4j containers:

```bash
./scripts/run_parallel.sh         # resume from checkpoints
./scripts/run_parallel.sh --clean # fresh run
```

Requires Docker. Starts 3 Neo4j containers on ports 7687/7688/7689.

### Checkpoint / Resume

Every case and query is checkpointed to `results/checkpoints/`. If a run crashes or times out:

```bash
# Just re-run the same command — it skips completed work
python -m src.benchmark_runner pipeline

# Check progress
cat results/checkpoints/pipeline_evaluate.json | python3 -c \
  "import json,sys; d=json.load(sys.stdin); print(f'{len(d[\"completed\"])} queries done')"

# Force fresh start
python -m src.benchmark_runner pipeline --clean
```

## Test Data

75 test cases across 7 categories:

| Category | Count | What It Tests |
|----------|-------|---------------|
| `static_fact` | 10 | Baseline — facts that never change |
| `evolving_fact` | 15 | Core test — facts change over time |
| `evolving_compound` | 10 | Compound sentences ("left X and joined Y") |
| `multi_entity_evolution` | 10 | Multiple entities evolving simultaneously |
| `contradiction_resolution` | 10 | Conflicting info from different sources |
| `entity_disambiguation` | 10 | Same name, different entities |
| `noisy_fact` | 10 | Typos, abbreviations, informal text |

Each case has `tags` for filtering (e.g., `["compound", "evolving"]`, `["noisy", "typo"]`).

Regenerate test data:
```bash
python -m src.data_generator
```

## Experiment Types

The benchmark supports pluggable experiment types:

```bash
# List available experiments
python -m src.benchmark_runner --list-experiments

# Run specific experiment (default: retrieval)
python -m src.benchmark_runner controlled -e retrieval
python -m src.benchmark_runner controlled -e ingestion
python -m src.benchmark_runner controlled -e search_tuning --param mmr_lambda=0.3

# Run multiple experiments (shared insertion)
python -m src.benchmark_runner controlled -e retrieval -e ingestion

# Compare tuning runs
python -m src.benchmark_runner controlled -e search_tuning --param mmr_lambda=0.3 --run-id tune1
python -m src.benchmark_runner controlled -e search_tuning --param mmr_lambda=0.7 --run-id tune2
python -m src.benchmark_runner --compare tune1 tune2
```

| Experiment | What It Measures |
|-----------|-----------------|
| `retrieval` | Search quality: P@5, R@5, MRR, TempAcc per strategy |
| `ingestion` | Graph quality: entity/edge recall, temporal invalidation accuracy, dedup |
| `search_tuning` | Parameterized search config comparison (mmr_lambda, bfs_depth, reranker, etc.) |

## Search Strategies

| Strategy | Method | Temporal Filter |
|----------|--------|----------------|
| `hybrid` | BM25 + cosine + RRF | Yes (only returns current facts) |
| `bm25_only` | Keyword fulltext | No (returns all facts including stale) |
| `cosine_only` | Embedding similarity | No (returns all facts including stale) |

The gap between hybrid (filtered) and baselines (unfiltered) measures the temporal graph's value.

## Results

Output saved to `results/`:
- `results/runs/{run_id}/` — per-experiment results, reports, metadata
- `results/{phase}_{timestamp}.json` — legacy aggregate metrics
- `results/checkpoints/` — per-case/per-query checkpoint state (crash recovery)
- `results/judge_cache/` — persistent LLM fact-matching cache
- `results/llm_cache/` — LLM response cache (per-file, for crash recovery)

### Metrics

| Metric | What It Measures |
|--------|-----------------|
| **P@5** | Fraction of top-5 results matching expected facts |
| **R@5** | Fraction of expected facts found in top-5 results |
| **MRR** | 1/rank of first correct result |
| **TempAcc** | Did the results avoid returning outdated facts? |

## Tests

```bash
# Run all unit tests
python -m pytest tests/ -v

# Skip slow LLM tests
python -m pytest tests/ -v -k "not split_compound and not split_noisy and not split_simple_pass"
```

## Project Structure

```
src/
  benchmark_runner.py       # CLI + orchestrator (insert + experiment dispatch)
  experiments/              # Pluggable experiment modules
    __init__.py             # ExperimentBase ABC, RunConfig, registry
    retrieval.py            # Search quality experiment (P@5, R@5, MRR, TempAcc)
    ingestion.py            # Graph quality experiment (entity/edge recall, temporal inv.)
    search_tuning.py        # Parameterized search config comparison
  checkpoint.py             # Checkpoint persistence for crash recovery
  models.py                 # Pydantic models (TestCase, QueryResult, IngestionResult, etc.)
  search_strategies.py      # Hybrid/BM25/cosine SearchConfig + build_search_config()
  search_utils.py           # Shared search + temporal filter utilities
  graph_inspector.py        # Neo4j Cypher queries for graph state inspection
  evaluator.py              # P@5, R@5, MRR, temporal accuracy
  judge.py                  # LLM-as-judge semantic fact matching
  judge_cache.py            # Persistent LLM judge result cache
  caching_llm_client.py     # Prompt-level LLM response cache (LRU + disk)
  contradiction_resolver.py # Post-ingestion contradiction detection
  reporter.py               # Aggregate + print result tables
  sentence_splitter.py      # Decompose compound/noisy text into atomic facts
  controlled_inserter.py    # Phase 1: direct Neo4j save
  pipeline_inserter.py      # Phase 2: add_episode (LLM extraction)
  presplit_inserter.py      # Phase 3: split then add_episode
  data_generator.py         # Combiner — imports from test_data/
  test_data/                # One file per category (compound.py, noisy.py, etc.)
scripts/
  run_parallel.sh           # Parallel Docker orchestration
  run_pipeline_benchmark.py # Long-running pipeline benchmark with resume
  spike_test.py             # Neo4j + Graphiti connectivity test
  trace_scenarios.py        # Detailed trace of scenarios with graph state
  explore_graphiti.py       # Interactive Graphiti API exploration
tests/
  quality-gates.md          # Quality gate definitions (hard gates)
  functional-gates.md       # Functional scenario definitions (hard gates)
  test_*.py                 # 73 unit tests
```
